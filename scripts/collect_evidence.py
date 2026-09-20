"""Collect every finished run under results/ into one machine-readable evidence table.

Three result formats exist in this repository, produced by three experiment
suites:

* ``serial_test.json`` with ``policies``  -- stage B / belief suites
* ``serial_test.json`` with ``metrics``   -- delayed-evidence suite (plus controls)
* ``result.json`` with ``evaluation``     -- single-mode training runs

Two rules matter for correctness:

1. ``serial_test.json`` always describes the official held-out evaluation, while
   ``result.json`` describes whatever split that run was configured to evaluate
   (stage B and belief ran with ``evaluate_test: false``, so their ``result.json``
   holds *validation* numbers). The collector therefore prefers ``serial_test.json``
   and records the split for every row, so validation and test numbers are never
   silently mixed.
2. The collector only reads. It never rewrites a run.

Variant names are the run directory name with the ``_seedN`` suffix removed, so
seeds of the same configuration aggregate into one row with mean, standard
deviation and n.

Usage:
    .venv/bin/python -m scripts.collect_evidence [--root results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

SEED = re.compile(r'_seed(\d+)$')
METRICS = ('sequence_accuracy', 'constraint_success', 'token_accuracy')


def rows_for(run: Path, experiment: str) -> list[dict]:
    """One row per measured policy in a single run directory."""
    variant = SEED.sub('', run.name)
    match = SEED.search(run.name)
    seed = int(match.group(1)) if match else None
    source = f'{experiment}/{run.name}'

    def row(scope: str, metrics: dict, split, calls) -> dict:
        return {'variant': variant, 'seed': seed, 'scope': scope, 'split': split,
                **{key: metrics.get(key) for key in METRICS},
                'forward_calls_per_example': calls,
                'source': f'{source}/{metrics["__file__"]}'}

    serial = run / 'serial_test.json'
    if serial.exists():
        data = json.loads(serial.read_text())
        out = []
        for index, policy in (data.get('policies') or {}).items():
            out.append(row(f'policy{index}', {**policy, '__file__': 'serial_test.json'},
                           data.get('split'), policy.get('forward_calls_per_example')))
        if 'metrics' in data:
            metrics = data['metrics']
            out.append(row('main', {**metrics, '__file__': 'serial_test.json'},
                           data.get('split'), metrics.get('effective_forward_calls')))
            for name, control in (metrics.get('controls') or {}).items():
                out.append(row(name, {**control, '__file__': 'serial_test.json'},
                               data.get('split'), None))
        return out
    result = run / 'result.json'
    if result.exists():
        data = json.loads(result.read_text())
        metrics = data.get('evaluation') or {}
        if 'sequence_accuracy' in metrics:
            return [row('main', {**metrics, '__file__': 'result.json'},
                        data.get('evaluation_split'), metrics.get('forward_calls_per_example'))]
    return []


def discover(root: Path) -> dict[str, list[Path]]:
    """Map each experiment directory to its run directories."""
    found = {}
    for experiment in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = [p for p in sorted(experiment.iterdir())
                if p.is_dir() and ((p / 'serial_test.json').exists() or (p / 'result.json').exists())]
        if runs:
            found[experiment.name] = runs
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results')
    args = parser.parse_args()
    root = Path(args.root)
    table: dict[str, dict[str, list]] = {}
    for experiment, runs in discover(root).items():
        for run in runs:
            for row in rows_for(run, experiment):
                if row['sequence_accuracy'] is None:
                    continue
                key = f"{row['variant']} :: {row['scope']} :: {row['split']}"
                table.setdefault(experiment, {}).setdefault(key, []).append(row)
    summary = {}
    for experiment, groups in table.items():
        for key, rows in sorted(groups.items()):
            values = [r['sequence_accuracy'] for r in rows]
            summary.setdefault(experiment, {})[key] = {
                'runs': len(values),
                'seeds': sorted(r['seed'] for r in rows if r['seed'] is not None),
                'sequence_accuracy_mean': statistics.mean(values),
                'sequence_accuracy_stdev': statistics.stdev(values) if len(values) > 1 else None,
                'constraint_success_mean': statistics.mean(
                    r['constraint_success'] for r in rows if r['constraint_success'] is not None),
                'forward_calls_per_example': next(
                    (r['forward_calls_per_example'] for r in rows
                     if r['forward_calls_per_example'] is not None), None),
                'example_source': rows[0]['source']}
    (root / 'evidence_table.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['| experiment | variant :: scope :: split | sequence accuracy | sd (pp) | runs | forward calls |',
             '|---|---|---:|---:|---:|---:|']
    for experiment, groups in summary.items():
        for key, value in sorted(groups.items(), key=lambda kv: -kv[1]['sequence_accuracy_mean']):
            sd = value['sequence_accuracy_stdev']
            lines.append(f"| {experiment} | {key} | {100*value['sequence_accuracy_mean']:.2f}% | "
                         f"{'—' if sd is None else f'{100*sd:.2f}'} | {value['runs']} | "
                         f"{value['forward_calls_per_example']} |")
    (root / 'evidence_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f"\nexperiments={len(summary)} rows={sum(len(v) for v in summary.values())}")


if __name__ == '__main__':
    main()
