# Repository map

This map describes the committed tree without changing its paths. The experiment IDs and filenames are part of the audit trail, so the organization is intentionally explanatory rather than a physical source-tree rewrite.

## Top-level directories

| Path | Role | Start with |
|---|---|---|
| [`retroar_min/`](../../retroar_min) | Model, rollout policies, training, suite supervisors | [`train.py`](../../retroar_min/train.py), [`run_local.py`](../../retroar_min/run_local.py), [`stage_b.py`](../../retroar_min/stage_b.py), [`delayed.py`](../../retroar_min/delayed.py) |
| [`configs/`](../../configs) | CPU experiment and smoke-suite JSON | [`configs/README.md`](../../configs/README.md) |
| [`data/`](../../data) | Generated graph-colouring datasets | [`data/README.md`](../../data/README.md) |
| [`scripts/`](../../scripts) | Evidence aggregation, probes, reports, publication slimming | [`collect_evidence.py`](../../scripts/collect_evidence.py) |
| [`tests/`](../../tests) | Regression and invariant tests | [`test_core.py`](../../tests/test_core.py) |
| [`results/`](../../results) | Published run summaries and provenance | [`evidence_table.json`](../../results/evidence_table.json), [`MANIFEST.json`](../../results/MANIFEST.json) |
| [`docs/`](..) | English navigation plus the Chinese chronological log | [`English index`](README.md), [`Chinese index`](../00_索引.md) |

## Source entry points

- `retroar_min.train`: one mode, one seed, one output directory.
- `retroar_min.run_local`: guarded sequential local runs for the initial and Stage A suites.
- `retroar_min.stage_b`: multi-worker Stage B and temperature-sweep suites.
- `retroar_min.delayed`: delayed-evidence policies, post-hoc controls, and the suite supervisor.
- `scripts.collect_evidence`: discovers `serial_test.json` and `result.json`, prefers official serial evaluation, and writes the aggregate table.
- `scripts.analyze_delayed_probe`: intervenes in frozen soft models at late evidence arrival and checks the unchanged branch.
- `scripts.report_results`: creates per-suite human-readable reports from completed runs.
- `scripts.slim_results`: converts a local `results_full/` archive into the committed publication tree and writes its manifest.

## Configuration families

- `cpu_smoke.json`: one short single-mode smoke run.
- `cpu_pilot.json`: 500-step validation-oriented local run.
- `cpu_long.json`: three-seed, 5,000-step initial long run.
- `cpu_stage_a.json`: five-seed local comparison with online, after-generation, and extra-compute modes.
- `cpu_stage_b.json`: five-seed matched-supervision and budget suite.
- `cpu_belief.json`: five-seed temperature sweep over hard, soft, and post-hoc policies.
- `cpu_delayed.json`: five-seed delayed-evidence suite.
- `*_smoke.json`: short suite versions using small data and a single seed.

See [`configs/README.md`](../../configs/README.md) for the exact matrix and dataset references.

## Data and results

The primary datasets are generated six-vertex, three-colour graph-colouring instances. Generation records include a seed, canonicalization rule, graph splits, targets, and initially ambiguous positions. `color6_stage_b.json` is the principal Stage B/Belief/Delayed task family; the delayed suite intentionally reuses that family, which limits independence. See [`data/README.md`](../../data/README.md).

`results/` is a committed publication tree. It contains aggregate metrics, traces, statuses, reports, and metadata, while omitting checkpoints (`*.pt`), console logs, and per-example details. [`MANIFEST.json`](../../results/MANIFEST.json) records source hashes and publication-time removals. Do not interpret a smoke or pilot result as evidence for the research claim.

## Stable paths versus local artefacts

Stable and committed:

- source code, tests, configs, datasets;
- published result summaries and manifest;
- English documentation and Chinese research log.

Local and intentionally ignored:

- `.venv/`, Python caches;
- `results_full/`;
- model checkpoints and console logs.

The repository is therefore readable and auditable from Git alone, but exact checkpoint-level reruns require the local full archive or a fresh execution with the same configuration and seeds.
