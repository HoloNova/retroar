# Reproducibility

**Chinese source:** [00 — 中文索引](../00_索引.md) · [13 — Final synthesis](../13_收尾实验与可行性总结.md)

## Environment

The committed lock file targets a CPU-only PyTorch environment:

```bash
uv venv .venv
.venv/bin/pip install -r requirements-cpu.lock
```

The original machine had two CPU cores and about 3.8 GiB RAM. The experiment supervisors are intentionally conservative:

- one computational thread per worker;
- at most two workers for suite configurations;
- 1 GiB RSS ceiling per worker;
- refuse startup below 1536 MiB available RAM;
- stop if host-available RAM falls below 768 MiB;
- write `blocked`, rather than `completed`, on a resource or timeout stop.

## Checks before running experiments

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The committed test suite covers leakage and masking, rollout invariants, supervision parity, scheduler guards, checkpoint behaviour, and resume determinism. The archived run record reports 41 tests.

## Reproduce the principal suites

The following commands use the committed configuration paths and output locations. Do not delete a finished output and reuse its name unless you intend to create a new result record.

```bash
# Initial 3-seed long run; run_local supplies the recorded default modes/seeds.
.venv/bin/python -m retroar_min.run_local \
  --config configs/cpu_long.json --output results/long_v1

# Stage A uses the same local supervisor with its recorded mode list and five seeds.
.venv/bin/python -m retroar_min.run_local \
  --config configs/cpu_stage_a.json --output results/stage_a_v1 \
  --modes ar ar_matched extra_compute hard soft soft_feedback posthoc_hard posthoc_soft \
  --seeds 0 1 2 3 4

# Stage B and the belief sweep use the suite supervisor.
.venv/bin/python -m retroar_min.stage_b \
  --config configs/cpu_stage_b.json --output results/stage_b_v1
.venv/bin/python -m retroar_min.stage_b \
  --config configs/cpu_belief.json --output results/belief_v1

# Delayed evidence uses a separate multi-policy supervisor.
.venv/bin/python -m retroar_min.delayed \
  --config configs/cpu_delayed.json --output results/delayed_v1
```

Smoke configurations are deliberately short and use the smaller `data/smoke.json` or delayed smoke data. They validate plumbing, not conclusions. The exact recorded suite parameters, variants, seeds, source hash, and dataset hash are stored in each result directory's `suite.json`.

## Resume and provenance rules

- `run_local.py`, `stage_b.py`, and `delayed.py` refuse to resume with a changed configuration.
- Runs record a source fingerprint and dataset fingerprint; mismatches block continuation.
- Checkpoints are selected from validation performance where configured; official test evaluation is performed after training.
- Stage B and Belief deliberately set `evaluate_test: false` during training. Their official numbers come from the later `serial_test.json`, not the exploratory validation values in `result.json`.
- `scripts/collect_evidence.py` reads runs and aggregates them; it does not rewrite a run.
- `scripts/analyze_delayed_probe.py` regenerates the frozen-model state intervention and verifies the unchanged treatment against the original result.

## Rebuild published summaries

```bash
.venv/bin/python -m scripts.collect_evidence
.venv/bin/python -m scripts.analyze_delayed_probe
```

The first command rewrites `results/evidence_table.json` and `results/evidence_table.md` from the available result tree. The second command writes the delayed probe output. The repository's committed `results/` tree is already the slim publication view; the ignored `results_full/` archive contains checkpoints and per-example details when available.

## Reproducibility is not independence

The delayed-evidence round uses the Stage B dataset split. Full graphs are isomorphism-disjoint across splits, but 5/128 validation and 20/256 test partial observations coincide with training observations. New seeds are new training runs on the same task family, not independent datasets. Preserve this caveat whenever quoting the result.
