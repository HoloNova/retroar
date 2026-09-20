"""Serial experiment supervisor; only terminates its own child process group."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def available_mib() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('Cannot read available RAM; refusing unmonitored run')


def rss_mib(pid: int) -> float:
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        return 0.0
    return 0.0


def terminate(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def aggregate(root: Path) -> dict:
    grouped: dict[str, list] = {}
    for path in sorted(root.glob('*/result.json')):
        result = json.loads(path.read_text())
        grouped.setdefault(result['mode'], []).append(result)
    summary = {}
    for mode, records in grouped.items():
        values = [r['evaluation']['sequence_accuracy'] for r in records]
        summary[mode] = {
            'seeds': [r['seed'] for r in records], 'completed_runs': len(records),
            'evaluation_split': records[0]['evaluation_split'],
            'sequence_accuracy_mean': statistics.mean(values),
            'sequence_accuracy_stdev': statistics.stdev(values) if len(values) > 1 else None,
            'token_accuracy_mean': statistics.mean(r['evaluation']['token_accuracy'] for r in records),
            'training_seconds_mean': statistics.mean(r['training_seconds'] for r in records),
            'peak_rss_mib_max': max(r['peak_rss_mib'] for r in records),
            'completed_steps': [r['completed_steps'] for r in records],
            'best_steps': [r['best_step'] for r in records],
            'stop_reasons': [r['stop_reason'] for r in records],
            'forward_calls_per_example': records[0]['evaluation']['forward_calls_per_example'],
        }
    write_json(root / 'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', default=['ar', 'extra_compute', 'hard', 'soft', 'soft_feedback'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config': config, 'modes': args.modes, 'seeds': args.seeds,
        'dataset_sha256': hashlib.sha256((ROOT / config['dataset']).read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(b''.join(
            p.name.encode() + p.read_bytes()
            for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest(),
    }
    manifest_path = root / 'suite.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing suite requires --resume and exactly matching configuration')
    write_json(manifest_path, manifest)
    protection = config['protection']
    child = None
    active = None
    try:
        for seed in args.seeds:
            for mode in args.modes:
                active = f'{mode}_seed{seed}'
                output = root / active
                if args.resume and (output / 'result.json').exists():
                    continue
                if available_mib() < protection['startup_available_mib']:
                    raise RuntimeError('Insufficient available RAM to start next run')
                output.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-m', 'retroar_min.train', '--config', str(config_path),
                       '--mode', mode, '--seed', str(seed), '--output', str(output)]
                if args.resume:
                    cmd.append('--resume')
                env = {**os.environ, 'OMP_NUM_THREADS': str(config['threads']),
                       'MKL_NUM_THREADS': str(config['threads']), 'OPENBLAS_NUM_THREADS': '1'}
                print(f'START {active}, available={available_mib():.0f}MiB', flush=True)
                write_json(root / 'status.json', {'state': 'running', 'active': active})
                begin = time.monotonic()
                last_notice = begin
                with (output / 'console.log').open('a', buffering=1) as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    while child.poll() is None:
                        rss = rss_mib(child.pid)
                        available = available_mib()
                        elapsed = time.monotonic() - begin
                        if rss > protection['max_rss_mib']:
                            raise RuntimeError(f'{active}: RSS {rss:.0f}MiB exceeded resource limit')
                        if available < protection['min_available_mib']:
                            raise RuntimeError(f'{active}: host available RAM fell to {available:.0f}MiB')
                        if elapsed > 60 and not (output / 'metadata.json').exists():
                            raise RuntimeError(f'{active}: initialization timeout')
                        if elapsed > config['max_seconds'] + 180:
                            raise RuntimeError(f'{active}: outer wall-clock timeout')
                        if time.monotonic() - last_notice >= 60:
                            detail = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
                            print(f'PROGRESS {active}: {elapsed:.0f}s, RSS={rss:.0f}MiB, {detail}', flush=True)
                            last_notice = time.monotonic()
                        time.sleep(1)
                if child.returncode != 0:
                    raise RuntimeError(f'{active}: child exited {child.returncode}; see {output / "console.log"}')
                if not (output / 'result.json').exists():
                    raise RuntimeError(f'{active}: child exited without a result.json')
                result = json.loads((output / 'result.json').read_text())
                if (result['source_sha256'] != manifest['source_sha256'] or
                        result['dataset_sha256'] != manifest['dataset_sha256']):
                    raise RuntimeError(f'{active}: source or dataset changed during suite')
                print(f'DONE {active} in {time.monotonic() - begin:.1f}s', flush=True)
                aggregate(root)
                child = None
        summary = aggregate(root)
        write_json(root / 'status.json', {'state': 'completed', 'runs': len(args.modes) * len(args.seeds)})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        if child is not None:
            terminate(child)
        write_json(root / 'status.json', {'state': 'blocked', 'active': active,
                                         'error': f'{type(exc).__name__}: {exc}'})
        aggregate(root)
        raise


if __name__ == '__main__':
    main()
