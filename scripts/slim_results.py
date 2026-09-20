"""Build the publishable results tree from the full local archive.

The full run directories total ~300 MiB, almost all of it per-example prediction
dumps (``policies[*].details`` inside ``serial_test.json`` / ``serial_val.json``)
plus model checkpoints and console logs. Aggregated metrics are the evidence used
by every claim in ``docs/``, so the published tree keeps them verbatim and drops
only the bulky per-example payloads.

Nothing is silently altered: every stripped file records how many records were
removed plus the SHA-256 of the removed payload, and ``results/MANIFEST.json``
records source hashes so a reader can verify what was dropped and regenerate the
full tree by re-running the documented commands with the same seeds.

Usage:
    .venv/bin/python -m scripts.slim_results [--source results_full] [--dest results]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STRIPPED_FILES = ('serial_test.json', 'serial_val.json')
STRIPPED_KEYS = ('details',)
SKIP_SUFFIXES = ('.pt', '.log')


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def slim_file(path: Path) -> dict:
    """Remove per-example payloads, returning a record of exactly what was removed."""
    original = json.loads(path.read_text())
    removed = {}
    for name, policy in (original.get('policies') or {}).items():
        for key in STRIPPED_KEYS:
            if key in policy:
                payload = policy.pop(key)
                removed[f'policies.{name}.{key}'] = {
                    'records': len(payload) if isinstance(payload, list) else None,
                    'sha256': digest(payload)}
    original['stripped_for_publication'] = {
        'removed': removed,
        'reason': 'per-example prediction dumps omitted to keep the repository small; '
                  'aggregated metrics are complete and unmodified',
        'regenerate': 're-run the suite with the same seeds and configuration'}
    return original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='results_full')
    parser.add_argument('--dest', default='results')
    args = parser.parse_args()
    source, dest = Path(args.source), Path(args.dest)
    if not source.is_dir():
        raise SystemExit(f'Missing archive: {source}')
    if dest.exists():
        raise SystemExit(f'Refusing to overwrite {dest}; remove it first')
    manifest = {'source': str(source), 'skipped_suffixes': list(SKIP_SUFFIXES),
                'stripped_keys': list(STRIPPED_KEYS), 'files': {}, 'skipped': [],
                'source_total_bytes': 0, 'published_total_bytes': 0}
    for path in sorted(source.rglob('*')):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        manifest['source_total_bytes'] += path.stat().st_size
        if path.suffix in SKIP_SUFFIXES:
            manifest['skipped'].append({'path': str(relative), 'reason': path.suffix,
                                        'bytes': path.stat().st_size})
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'source_bytes': path.stat().st_size}
        if path.name in STRIPPED_FILES:
            value = slim_file(path)
            record['stripped'] = sorted(value['stripped_for_publication']['removed'])
            target.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
        else:
            target.write_bytes(path.read_bytes())
        record['published_bytes'] = target.stat().st_size
        manifest['files'][str(relative)] = record
        manifest['published_total_bytes'] += record['published_bytes']
    (dest / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({
        'source': f"{manifest['source_total_bytes']/1048576:.1f} MiB",
        'published': f"{manifest['published_total_bytes']/1048576:.1f} MiB",
        'published_files': len(manifest['files']),
        'skipped_files': len(manifest['skipped'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
