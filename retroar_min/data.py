"""Generate satisfiable graph-coloring tasks; never expose target colors as input."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path


COLORS = 3


def solve(adj: list[list[int]], n: int, limit: int = 2) -> list[list[int]]:
    """Count up to limit solutions, with the final three vertices precolored."""
    assigned = [-1] * n + [0, 1, 2]
    solutions: list[list[int]] = []

    def visit() -> None:
        if len(solutions) >= limit:
            return
        choices = []
        for node in range(n):
            if assigned[node] != -1:
                continue
            forbidden = {assigned[j] for j, edge in enumerate(adj[node])
                         if edge and assigned[j] != -1}
            allowed = [c for c in range(COLORS) if c not in forbidden]
            if not allowed:
                return
            choices.append((len(allowed), node, allowed))
        if not choices:
            solutions.append(assigned[:n])
            return
        _, node, allowed = min(choices)
        for color in allowed:
            assigned[node] = color
            visit()
            assigned[node] = -1
            if len(solutions) >= limit:
                return

    visit()
    return solutions


def canonical_key(adj: list[list[int]], n: int) -> str:
    """Exact canonical key, invariant to free-node and global color permutations.

    Refine vertex partitions first, then enumerate only unresolved permutations.
    The three anchor vertices are distinguished from free vertices.
    """
    best = None
    for anchors in itertools.permutations(range(n, n + 3)):
        labels = [3] * n + [0] * 3
        for label, node in enumerate(anchors):
            labels[node] = label
        while True:
            count = max(labels) + 1
            signatures = []
            for i in range(n + 3):
                neighbors = [0] * count
                for j, edge in enumerate(adj[i]):
                    if edge:
                        neighbors[labels[j]] += 1
                signatures.append((labels[i], tuple(neighbors)))
            mapping = {sig: k for k, sig in enumerate(sorted(set(signatures)))}
            refined = [mapping[sig] for sig in signatures]
            if refined == labels:
                break
            labels = refined
        groups = [[i for i in range(n) if labels[i] == label]
                  for label in sorted(set(labels[:n]))]
        for parts in itertools.product(*(itertools.permutations(g) for g in groups)):
            order = list(anchors) + [i for part in parts for i in part]
            bits = ''.join(str(adj[order[i]][order[j]])
                           for i in range(n + 3) for j in range(i + 1, n + 3))
            if best is None or bits < best:
                best = bits
    return f'{n}:{best}'


def valid_assignment(adj: list[list[int]], answer: list[int]) -> bool:
    if any(c not in range(COLORS) for c in answer):
        return False
    full = answer + [0, 1, 2]
    if len(adj) != len(full):
        return False
    return all(not adj[i][j] or full[i] != full[j]
               for i in range(len(full)) for j in range(i + 1, len(full)))


def make_dataset(n: int = 6, train: int = 1024, val: int = 128,
                 test: int = 256, seed: int = 2026, seconds: float = 180,
                 attempts: int = 100000, excluded_keys: set[str] | None = None) -> dict:
    if n < 2 or min(train, val, test) < 1:
        raise ValueError('Need at least two free vertices and nonempty splits')
    rng = random.Random(seed)
    start = time.monotonic()
    records, seen = [], set(excluded_keys or ())
    total = train + val + test
    tried = 0
    for tried in range(1, attempts + 1):
        if time.monotonic() - start > seconds:
            break
        planted = [rng.randrange(COLORS) for _ in range(n)] + [0, 1, 2]
        p = rng.uniform(0.35, 0.75)
        adj = [[0] * (n + 3) for _ in range(n + 3)]
        for i in range(n + 3):
            for j in range(i + 1, n + 3):
                if planted[i] != planted[j] and (i >= n or rng.random() < p):
                    adj[i][j] = adj[j][i] = 1
        ambiguous = sum(sum(adj[i][n:]) < 2 for i in range(n))
        # Do not accept tasks where nearly every answer is directly forced by anchors.
        if ambiguous < max(2, n // 2):
            continue
        solutions = solve(adj, n)
        if len(solutions) != 1:
            continue
        key = canonical_key(adj, n)
        if key in seen:
            continue
        seen.add(key)
        records.append({'adj': adj, 'target': solutions[0], 'key': key,
                        'initially_ambiguous': ambiguous})
        if len(records) == total:
            break
    if len(records) != total:
        raise RuntimeError(f'Dataset incomplete: {len(records)}/{total}, '
                           f'{tried} attempts, {time.monotonic() - start:.1f}s; '
                           'no repeated examples were added')
    rng.shuffle(records)
    return {
        'schema': 1, 'n': n, 'seed': seed,
        'generation': {'attempts': tried, 'seconds': time.monotonic() - start,
                       'canonicalization': 'free-node and global color permutations',
                       'excluded_keys': len(excluded_keys or ())},
        'train': records[:train], 'val': records[train:train + val],
        'test': records[train + val:],
    }


def load_dataset(path: str | Path) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/color6.json')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--train', type=int, default=1024)
    parser.add_argument('--val', type=int, default=128)
    parser.add_argument('--test', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--seconds', type=float, default=180)
    args = parser.parse_args()
    data = make_dataset(args.n, args.train, args.val, args.test, args.seed, args.seconds)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'path': str(path), **data['generation'],
                      'sizes': {s: len(data[s]) for s in ('train', 'val', 'test')}}), flush=True)


if __name__ == '__main__':
    main()
