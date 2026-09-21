# Experiment matrix

**Chinese source:** [03 — 首轮长实验协议](../03_协议_首轮长实验.md) · [05 — 阶段 A 协议](../05_协议_阶段A.md) · [07 — 阶段 B 协议](../07_协议_阶段B.md) · [09 — 草稿信念协议](../09_协议_草稿信念.md) · [11 — 延迟证据协议](../11_协议_延迟证据.md)

The repository contains **five main evidence rounds**, plus pipeline checks and one historical design draft. Protocols were written before the corresponding runs; later analyses may narrow or overturn an earlier interpretation but do not rewrite the protocol.

| Round | Result ID | Question isolated | Main comparison | Seeds / split | Analysis record |
|---|---|---|---|---|---|
| First long run | [`long_v1`](../../results/long_v1) | Does editable revision help at all? | soft/hard online revision vs. plain AR and extra-compute controls | 3 / test | [Chinese design and first result](../02_最小实验设计与本机运行评估.md) |
| Stage A | [`stage_a_v1`](../../results/stage_a_v1) | Is the gain about online timing? | online revision vs. after-generation revision | 5 / test | [06 — Stage A analysis](../06_阶段A复核分析.md) |
| Stage B | [`stage_b_v1`](../../results/stage_b_v1) | Does the result survive fairer budget and supervision controls? | after-generation, online, multi-candidate sampling, and more training | 5 new / test | [08 — Stage B analysis](../08_阶段B分析.md) |
| Belief | [`belief_v1`](../../results/belief_v1) | Was a soft draft fairly tested? | temperature sweep from near-hard to genuinely non-collapsed states | 5 new / test | [10 — Belief analysis](../10_草稿信念实验分析.md) |
| Delayed evidence | [`delayed_v1`](../../results/delayed_v1) | Is an early draft useful before late evidence arrives? | wait, restart, hard revision, and soft revision | 5 new / test | [13 — Final synthesis](../13_收尾实验与可行性总结.md) |

## Interpretation of the rounds

- **Long run:** a useful first signal, but its extra-compute control was not a fair matched-budget baseline.
- **Stage A:** after-generation revision beat online revision on all five seeds, moving the explanatory burden from timing to general iterative solving.
- **Stage B:** the result remained after matched supervision and additional sampling controls; more training did not recover the online variant.
- **Belief:** temperatures changed state entropy from nearly one-hot to genuinely uncertain, so collapse alone does not explain the missing advantage.
- **Delayed evidence:** soft revision scored highest, but the post-hoc state probe showed that the specific early draft was not necessary for the final score.

## Evidence tiers

| Tier | Examples | How to use them |
|---|---|---|
| Main evidence | `long_v1`, `stage_a_v1`, `stage_b_v1`, `belief_v1`, `delayed_v1` | Supports the research conclusion, with the stated caveats |
| Pipeline checks | `smoke`, `pilot`, `*_smoke_v*` | Confirms wiring and safeguards; not evidence for model quality |
| Diagnostic / benchmark | `stage_b_benchmark_v1`, post-hoc controls | Answers a narrow implementation question; do not promote to a new claim |

For every row, check the split, seed count, forward-call budget, and source path in [`results/evidence_table.json`](../../results/evidence_table.json). The collector prefers `serial_test.json` for official evaluation; some `result.json` files in the Stage B and Belief suites contain validation scores by design.
