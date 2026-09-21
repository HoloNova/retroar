# CPU experiment configurations

These JSON files are committed run specifications. They keep experiment IDs, dataset paths, seeds, model size, budgets, and resource guards explicit. The English experiment matrix explains the scientific purpose; this file is the operational catalogue.

| File | Entrypoint | Dataset | Role | Seed / split |
|---|---|---|---|---|
| [`cpu_smoke.json`](cpu_smoke.json) | `retroar_min.run_local` | [`data/smoke.json`](../data/smoke.json) | 20-step single-mode plumbing check | one local run |
| [`cpu_pilot.json`](cpu_pilot.json) | `retroar_min.run_local` | [`data/color6.json`](../data/color6.json) | 500-step validation-oriented pilot | validation by default |
| [`cpu_long.json`](cpu_long.json) | `retroar_min.run_local` | [`data/color6.json`](../data/color6.json) | Initial long comparison | seeds 0–2, test |
| [`cpu_stage_a.json`](cpu_stage_a.json) | `retroar_min.run_local` | [`data/color6.json`](../data/color6.json) | Online vs. after-generation revision | seeds 0–4, test |
| [`cpu_stage_b.json`](cpu_stage_b.json) | `retroar_min.stage_b` | [`data/color6_stage_b.json`](../data/color6_stage_b.json) | Fair supervision, budget, sampling, and training controls | seeds 10–14, test after training |
| [`cpu_belief.json`](cpu_belief.json) | `retroar_min.stage_b` | [`data/color6_stage_b.json`](../data/color6_stage_b.json) | Soft-state temperature sweep | seeds 20–24, test after training |
| [`cpu_delayed.json`](cpu_delayed.json) | `retroar_min.delayed` | [`data/color6_stage_b.json`](../data/color6_stage_b.json) | Delayed-evidence policy comparison | seeds 30–34, test |
| [`cpu_stage_b_smoke.json`](cpu_stage_b_smoke.json) | `retroar_min.stage_b` | [`data/smoke.json`](../data/smoke.json) | Short Stage B pipeline check | seed 81, validation |
| [`cpu_belief_smoke.json`](cpu_belief_smoke.json) | `retroar_min.stage_b` | [`data/smoke.json`](../data/smoke.json) | Short temperature-sweep check | seed 31, validation |
| [`cpu_delayed_smoke.json`](cpu_delayed_smoke.json) | `retroar_min.delayed` | [`data/color6_delayed_smoke.json`](../data/color6_delayed_smoke.json) | Short delayed-evidence check | seed 99, validation |

## Reading the fields

- `dataset` is resolved relative to the repository root and is fingerprinted before a run starts.
- `model` is the small CPU model used throughout the committed experiments: width 32, two heads, feed-forward width 64, one layer.
- `steps`, `eval_every`, and `max_seconds` control one training run; suite files add worker, seed, variant, and suite limits.
- `evaluate_test: false` is deliberate in Stage B and Belief. Their official test values come from the later serial evaluation, not training-time `result.json`.
- `protection` is part of the reproducibility contract, not an optional performance hint.

## Commands

The verified suite commands are collected in [`docs/en/reproducibility.md`](../docs/en/reproducibility.md). Do not change a config in place and treat an existing result directory as a different experiment: supervisors require matching configuration and source/data fingerprints when resuming.
