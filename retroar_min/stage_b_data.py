"""Create a new split with exact isomorphism exclusion against all supplied datasets."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import canonical_key, make_dataset
from .run_local import write_json


def prepare(output: Path, excludes: list[Path], train=1024, val=128, test=256,
            seed=73021, seconds=180):
    if output.exists():
        raise RuntimeError('Refusing to overwrite dataset')
    excluded = set()
    fingerprints = {}
    for path in excludes:
        raw = path.read_bytes()
        old = json.loads(raw)
        if old['n'] != 6:
            raise ValueError('Exclusion datasets must have n=6')
        fingerprints[str(path)] = hashlib.sha256(raw).hexdigest()
        for split in ('train', 'val', 'test'):
            for row in old[split]:
                excluded.add(canonical_key(row['adj'], 6))
    data = make_dataset(train=train, val=val, test=test, seed=seed,
                        seconds=seconds, excluded_keys=excluded)
    keys = [canonical_key(r['adj'], 6) for split in ('train', 'val', 'test') for r in data[split]]
    if len(keys) != len(set(keys)) or set(keys) & excluded:
        raise RuntimeError('Isomorphism exclusion failed')
    data['excluded_datasets'] = fingerprints
    write_json(output, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude', nargs='+', required=True)
    args = parser.parse_args()
    prepare(Path(args.output), [Path(p) for p in args.exclude])


if __name__ == '__main__':
    main()
