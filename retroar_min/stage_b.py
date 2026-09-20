"""At most two independent training workers; serial final evaluation and hard suite deadline."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .run_local import ROOT, available_mib, rss_mib, terminate, write_json


def source_hash() -> str:
    return hashlib.sha256(b''.join(p.name.encode() + p.read_bytes()
                                  for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()


def checked_result(path: Path, config: dict, mode: str, seed: int, source: str, dataset: str) -> dict:
    row = json.loads(path.read_text())
    if not (row['config'] == config and row['mode'] == mode and row['seed'] == seed
            and row['source_sha256'] == source and row['dataset_sha256'] == dataset
            and row['completed_steps'] == config['steps'] and row['stop_reason'] == 'max_steps'):
        raise RuntimeError(f'Incomplete or mismatched run: {path}')
    return row


def run(config_path: Path, root: Path, resume: bool = False) -> None:
    config = json.loads(config_path.read_text())
    workers = config['workers']
    if workers not in (1, 2):
        raise ValueError('workers must be 1 or 2')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    names = [v['name'] for v in config['variants']]
    if not names or len(set(names)) != len(names) or any(not n.replace('_', '').isalnum() for n in names):
        raise ValueError('Variant names must be unique safe identifiers')
    for variant in config['variants']:
        temperature = variant.get('overrides', {}).get('soft_temperature', 1.0)
        if temperature <= 0:
            raise ValueError('soft_temperature must be positive')
    if config['training'].get('evaluate_test', True):
        raise ValueError('Training must not evaluate test data')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    done = []
    begin = time.monotonic()
    previous = 0.0
    phase = 'setup'
    total = 0
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    def status(state, error=None):
        detail = {}
        for key, value in active.items():
            p = value['directory'] / 'status.json'
            detail[key] = json.loads(p.read_text()) if p.exists() else {'state': 'initializing'}
        write_json(root / 'status.json', {'state': state, 'phase': phase,
                   'active': list(active), 'details': detail, 'completed_jobs': done,
                   'total_jobs': total, 'workers': workers,
                   'elapsed_seconds': previous + time.monotonic() - begin, 'error': error})

    def check_limits():
        if previous + time.monotonic() - begin > config['suite_seconds']:
            raise RuntimeError('Suite wall-clock budget exhausted')
        if available_mib() < config['training']['protection']['min_available_mib']:
            raise RuntimeError('Host available memory below limit')
        for key, v in active.items():
            if rss_mib(v['process'].pid) > config['training']['protection']['max_rss_mib']:
                raise RuntimeError(f'{key}: RSS limit exceeded')
            if time.monotonic() - v['begin'] > config['training']['max_seconds'] + 180:
                raise RuntimeError(f'{key}: worker timeout')

    def launch(key, command, directory):
        waiting = time.monotonic()
        while available_mib() < config['training']['protection']['startup_available_mib']:
            if active or time.monotonic() - waiting >= config.get('startup_wait_seconds', 30):
                return False
            check_limits()
            status('waiting_for_memory')
            time.sleep(1)
        log = (directory / f'{phase}.log').open('a')
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                                          'OPENBLAS_NUM_THREADS': '1'}, start_new_session=True)
        finally:
            log.close()
        active[key] = {'process': child, 'begin': time.monotonic(), 'directory': directory}
        return True

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupted)
        source = source_hash()
        dataset = hashlib.sha256((ROOT / config['training']['dataset']).read_bytes()).hexdigest()
        manifest = {'config': config, 'source_sha256': source, 'dataset_sha256': dataset}
        suite = root / 'suite.json'
        if suite.exists():
            if not resume or json.loads(suite.read_text()) != manifest:
                raise RuntimeError('Existing suite needs exact matching config/source and --resume')
            if (root / 'status.json').exists():
                previous = json.loads((root / 'status.json').read_text())['elapsed_seconds']
        elif resume:
            raise RuntimeError('No suite to resume')
        write_json(suite, manifest)
        jobs = []
        for seed in config['seeds']:
            for variant in config['variants']:
                key = f'{variant["name"]}_seed{seed}'
                directory = root / key
                directory.mkdir(exist_ok=True)
                training = {**config['training'], **variant.get('overrides', {}),
                            'steps': variant['steps'],
                            'eval_every': variant.get('eval_every', config['training']['eval_every'])}
                path = directory / 'config.json'
                write_json(path, training)
                jobs.append((key, directory, path, training, variant['mode'], seed))
        total = len(jobs)
        phase = 'training'
        queue = []
        for job in jobs:
            key, directory, path, training, mode, seed = job
            result = directory / 'result.json'
            if result.exists():
                checked_result(result, training, mode, seed, source, dataset)
                done.append(key)
            else:
                queue.append(job)
        while queue or active:
            check_limits()
            while queue and len(active) < workers:
                key, directory, path, training, mode, seed = queue[0]
                command = [sys.executable, '-m', 'retroar_min.train', '--config', str(path),
                           '--mode', mode, '--seed', str(seed), '--output', str(directory)]
                if resume and (directory / 'last.pt').exists():
                    command.append('--resume')
                if not launch(key, command, directory):
                    if not active:
                        raise RuntimeError('Insufficient RAM to start worker')
                    break
                queue.pop(0)
            for key, v in list(active.items()):
                code = v['process'].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f'{key} exited {code}; inspect training.log')
                job = next(j for j in jobs if j[0] == key)
                checked_result(job[1] / 'result.json', job[3], job[4], job[5], source, dataset)
                done.append(key)
                del active[key]
            status('running')
            if queue or active:
                time.sleep(1)
        # Training finishes before any held-out evaluation; no competing suite workers.
        if source_hash() != source:
            raise RuntimeError('Source changed before serial evaluation')
        phase = 'serial_evaluation'
        for key, directory, path, training, mode, seed in jobs:
            output = directory / f'serial_{config["final_split"]}.json'
            if output.exists():
                row = json.loads(output.read_text())
                if row['dataset_sha256'] != dataset or row['seed'] != seed or row['mode'] != mode:
                    raise RuntimeError('Mismatched evaluation')
                continue
            check_limits()
            if not launch(key, [sys.executable, '-m', 'retroar_min.stage_b_eval',
                               '--directory', str(directory), '--split', config['final_split']], directory):
                raise RuntimeError('Insufficient RAM for final evaluation')
            while active[key]['process'].poll() is None:
                check_limits()
                status('running')
                time.sleep(1)
            if active[key]['process'].returncode or not output.exists():
                raise RuntimeError(f'{key}: serial evaluation failed')
            del active[key]
        report = ['# Stage B results', '', 'Timing is serial batch=1 end-to-end, not pure model latency.', '',
                  '| Run | Candidates (0=revision) | Accuracy | Calls/example | Seconds |',
                  '|---|---:|---:|---:|---:|']
        for key, directory, *_ in jobs:
            row = json.loads((directory / f'serial_{config["final_split"]}.json').read_text())
            for k, metrics in row['policies'].items():
                report.append(f'| {key} | {k} | {metrics["sequence_accuracy"]:.4f} | '
                              f'{metrics["forward_calls_per_example"]} | {metrics["end_to_end_seconds"]:.3f} |')
        (root / 'report.md').write_text('\n'.join(report) + '\n')
        phase = 'finished'
        status('completed')
    except BaseException as exc:
        for v in active.values():
            terminate(v['process'])
        status('blocked', f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(Path(args.config).resolve(), Path(args.output).resolve(), args.resume)


if __name__ == '__main__':
    main()
