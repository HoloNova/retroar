# RetroAR

**English-first research archive for a falsified hypothesis.**

RetroAR asked whether an autoregressive generator improves when its newest outputs remain editable drafts—possibly as probability distributions—and are revised as generation continues. On a small CPU-only graph-colouring task, the committed experiments do **not** support that online mechanism as a distinctive advantage.

> **Status: archived. No further training is planned.** The repository preserves the code, preregistered protocols, published run data, analyses, limitations, and negative result.

- [English documentation](docs/en/README.md)
- [中文完整研究日志](docs/00_索引.md)
- [MIT license](LICENSE)

## Bottom line

1. **After-generation revision beat online revision.** Across the two matched 5-seed rounds, it was better in 9/10 paired seed comparisons.
2. **Soft categorical drafts did not establish the proposed source of gain.** Temperature sweeps produced genuinely non-collapsed states, but did not meet their preregistered continuation bar.
3. **The strongest delayed-evidence soft result did not require the early draft details.** Replacing the early state with uniform probabilities, another question's state, or zeros produced essentially the same final accuracy.
4. These are **mechanism diagnostics**, not evidence of practical usefulness: the task has six free vertices, three colours, a roughly 30k-parameter model, and can be solved exactly by classical backtracking.

## Evidence snapshot

All values below are whole-question accuracy on the held-out test split unless stated otherwise. A question counts as correct only when all six free positions are correct; values are means over the listed seeds.

| Round | Primary comparison | Result | Interpretation |
|---|---|---:|---|
| `long_v1` · 3 seeds | soft online vs. plain AR | 87.37% vs. 77.47% | Initial positive signal, not budget-matched |
| `stage_a_v1` · 5 seeds | after-generation soft vs. online soft | 92.50% vs. 88.05% | Timing advantage moved away from online revision |
| `stage_b_v1` · 5 new seeds | after-generation soft vs. online soft | 93.36% vs. 85.23% | Matched-budget replication favoured after-generation revision |
| `belief_v1` · 5 new seeds | after-generation soft, T=2 | 94.53% | Flattening the state did not yield the proposed online benefit |
| `delayed_v1` · 5 new seeds | soft revision vs. two-candidate waiting | 90.23% vs. 79.53% | Real signal, but post-hoc state replacement reached 90.39% |

The machine-readable aggregation is [`results/evidence_table.json`](results/evidence_table.json); the generated human-readable table is [`results/evidence_table.md`](results/evidence_table.md). Read the interpretation and caveats in [the English results note](docs/en/results.md), not from a single headline number.

## What this repository does and does not claim

**It does claim:** the tested implementation repeatedly demonstrates useful iterative solving after the full relevant evidence is available, and it records why that result is insufficient to support the original online-revision story.

**It does not claim:** that iterative refinement is useless; that online revision is impossible; that language models behave the same way; that the mechanism is practically useful; or that the descriptive five-seed comparisons establish conventional statistical significance. The smallest two-sided sign-test p-value for five positive paired seeds is 0.0625.

The delayed-evidence round reuses the Stage B task split. Full graphs are isomorphism-disjoint across splits, but some partial observations overlap training observations. This is a limitation, not an independent replication.

## Repository map

```text
retroar_min/     model, rollout policies, training, and experiment supervisors
configs/         committed CPU experiment configurations
scripts/         evidence collection, report generation, probes, and result slimming
tests/           regression tests for leakage, masking, rollout, budgets, and resume logic
data/            generated graph-colouring datasets with split metadata
results/         published metrics, reports, statuses, traces, evidence table, and manifest
docs/            English navigation plus the complete Chinese chronological research log
```

The detailed map, configuration catalogue, data policy, and terminology are in the [English documentation index](docs/en/README.md). Existing Chinese documents retain their original numbered paths so historical links and audit trails remain stable.

## Reproduce the published workflow

The project is CPU-only and deliberately guarded for a small machine: one thread per worker, at most two workers, a 1 GiB RSS limit per worker, and a stop condition when host-available memory falls below 768 MiB.

```bash
uv venv .venv
.venv/bin/pip install -r requirements-cpu.lock
.venv/bin/python -m unittest discover -s tests -v

# Single-process suite used for the first long run
.venv/bin/python -m retroar_min.run_local \
  --config configs/cpu_long.json --output results/long_v1

# Multi-policy delayed-evidence suite
.venv/bin/python -m retroar_min.delayed \
  --config configs/cpu_delayed.json --output results/delayed_v1

# Rebuild the committed aggregate table from published run files
.venv/bin/python -m scripts.collect_evidence
```

The [reproducibility guide](docs/en/reproducibility.md) lists the other suite entry points, resume rules, fingerprints, and publication policy. Finished runs refuse accidental overwrite; resume requires matching source and dataset fingerprints. A resource stop is recorded as `blocked`, not as a successful result.

## Published-data policy

`results/` is the deliberately slimmed, committed tree. It contains aggregate metrics, reports, traces, statuses, and provenance. Model checkpoints (`*.pt`), console logs, and per-example prediction details are excluded from the published tree; [`results/MANIFEST.json`](results/MANIFEST.json) records the source hash and removals. The full local archive is `results_full/`, which is ignored by Git and is not required for reading the committed record.

The committed data files are generated graph-colouring instances. They are provided as-is, with no warranty of correctness or fitness for any purpose. See [the data note](docs/en/repository-map.md#data-and-results) for the split and provenance caveats.
