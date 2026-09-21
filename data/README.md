# Generated datasets

The committed datasets are small, generated graph-colouring instances for mechanism diagnostics. Each record describes a graph, a target colouring, a canonical key, and which free positions were initially ambiguous. They are not language data and are not intended as a practical benchmark.

| File | Generation seed | Used by | Purpose |
|---|---:|---|---|
| [`color6.json`](color6.json) | 2026 | Long run, Stage A, pilot | Main six-vertex task family |
| [`color6_stage_b.json`](color6_stage_b.json) | 73021 | Stage B, Belief, Delayed | Isomorphism-deduplicated Stage B task family |
| [`color6_delayed_smoke.json`](color6_delayed_smoke.json) | 99811 | Delayed smoke suite | Short delayed-evidence pipeline check |
| [`smoke.json`](smoke.json) | 999 | Local and suite smoke checks | Tiny fast dataset for wiring and guard tests |

## Schema and generation

The JSON files record a schema version, `n = 6`, a generation seed, and split arrays such as `train`, `val`, and `test`. A graph record contains:

- `adj`: the adjacency matrix;
- `target`: the three-colour target for the free vertices;
- `key`: a canonical graph/colouring identity;
- `initially_ambiguous`: the number of free positions whose colour is not determined by the early input.

Generation applies free-node and global-colour canonicalization. The task family is constructed to avoid trivial duplicate instances and to support unique-solution or controlled-ambiguity diagnostics.

## Important caveat

`color6_stage_b.json` is reused by the delayed-evidence round. Full graphs are isomorphism-disjoint across splits, but partial observations can coincide: the final synthesis records 5/128 validation and 20/256 test partial observations that also occur in training. New seeds therefore do not turn the delayed round into an independent dataset replication.

Dataset fingerprints are written into suite and run metadata. The training code checks those fingerprints before resuming; do not edit a committed dataset in place and expect an old result directory to remain comparable.
