"""Bounded serial/dual-worker throughput comparison on the SAME two training jobs."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from .run_local import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise RuntimeError('Benchmark output exists')
    root.mkdir(parents=True)
    base = json.loads((ROOT / 'configs/cpu_stage_b.json').read_text())
    base.update(seeds=[71, 72], suite_seconds=600, final_split='val',
                variants=[{'name': 'soft', 'mode': 'soft', 'steps': 300}])
    base['training'].update(dataset='data/smoke.json', eval_every=300, max_seconds=240)
    elapsed, hashes = {}, {}
    for workers in (1, 2):
        config = {**base, 'workers': workers}
        path = root / f'workers{workers}.json'
        write_json(path, config)
        output = root / f'workers{workers}'
        begin = time.perf_counter()
        subprocess.run([sys.executable, '-m', 'retroar_min.stage_b', '--config', str(path),
                        '--output', str(output)], cwd=ROOT, check=True, timeout=600)
        elapsed[str(workers)] = time.perf_counter() - begin
        # Tensor equality checked in separate Python below; archive checkpoint hashes too.
        hashes[str(workers)] = {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in output.glob('*/best.pt')}
    import torch
    for seed in (71, 72):
        a = torch.load(root / 'workers1' / f'soft_seed{seed}' / 'last.pt', weights_only=True)['model']
        b = torch.load(root / 'workers2' / f'soft_seed{seed}' / 'last.pt', weights_only=True)['model']
        if not all(torch.equal(a[k], b[k]) for k in a):
            raise RuntimeError('Serial and parallel weights differ')
    speedup = elapsed['1'] / elapsed['2']
    write_json(root / 'benchmark.json', {'elapsed_seconds': elapsed, 'speedup': speedup,
               'recommended_workers': 2 if speedup >= 1.1 else 1,
               'equal_final_weights': True, 'checkpoint_hashes': hashes,
               'note': 'Two 300-step jobs; includes process startup and serial val evaluation. Not a long-run guarantee.'})
    print((root / 'benchmark.json').read_text())


if __name__ == '__main__':
    main()
