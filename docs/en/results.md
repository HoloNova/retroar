# Results and evidence

**Chinese source:** [06 — Stage A analysis](../06_阶段A复核分析.md) · [08 — Stage B analysis](../08_阶段B分析.md) · [10 — Belief analysis](../10_草稿信念实验分析.md) · [13 — Final synthesis](../13_收尾实验与可行性总结.md) · [14 — Evidence table](../14_证据总表.md)

## Headline values

The metric is **whole-question accuracy**: all six free positions must be correct. Percentages are means over training seeds; standard deviations are seed-to-seed variation, not confidence intervals.

| Round | Main result | Paired reading |
|---|---:|---|
| `long_v1` | soft 87.37%; plain AR 77.47% | Positive initial signal, but the control was not budget matched |
| `stage_a_v1` | after-generation soft 92.50%; online soft 88.05% | 5/5 seeds favoured after-generation revision |
| `stage_b_v1` | after-generation soft 93.36%; online soft 85.23% | New 5-seed round with stronger budget controls |
| `belief_v1` | after-generation soft T=2 94.53%; online soft T=2 83.67% | Higher temperature did not restore online performance |
| `delayed_v1` | soft revision 90.23%; wait-two-candidates 79.53% | Strong signal, but the early state was not causally established |

The full generated table—including smoke, pilot, control, validation, and diagnostic rows—is [`results/evidence_table.md`](../../results/evidence_table.md). The structured source with seed lists, split labels, forward calls, and example paths is [`results/evidence_table.json`](../../results/evidence_table.json).

## Why the delayed soft result is not a proof of the original idea

The five trained `revise_soft` models were evaluated again after replacing the state seen at the first late-evidence revision call:

| State supplied at late evidence | Final accuracy |
|---|---:|
| Original question-specific soft state | 90.23% |
| Uniform `1/3, 1/3, 1/3` state | 90.39% |
| Another question's state | 89.92% |
| Zero colour vectors, `present` retained | 90.63% |
| Argmax hardening of the original state | 86.80% |

This is a **post-hoc intervention on frozen trained models**, not a separately trained `restart_soft` condition and not an independent replication. It shows that the final score is insensitive to the details of the early state in this trained configuration; it does not prove that every soft revision mechanism is equivalent.

The probe is recorded in [`results/delayed_v1/posthoc_state_probe.json`](../../results/delayed_v1/posthoc_state_probe.json) and can be regenerated with [`scripts/analyze_delayed_probe.py`](../../scripts/analyze_delayed_probe.py).

## Audit path

For an individual number, follow this chain:

1. Start at [`evidence_table.json`](../../results/evidence_table.json).
2. Read the row's `example_source` path.
3. Inspect the corresponding `serial_test.json` or `result.json`.
4. Check the run's `config.json`, `metadata.json`, `status.json`, and `suite.json` for split, seed, fingerprint, best step, and stopping state.
5. Use [`results/MANIFEST.json`](../../results/MANIFEST.json) to understand which large source artefacts were removed from the committed tree.

The committed result tree is a publication view, not the full local archive. Checkpoint files, console logs, and per-example prediction details were intentionally omitted. The raw run summaries and provenance needed to interpret the reported aggregates remain committed.

## Limitations that change the strength of the conclusion

- The task is tiny and exactly solvable; accuracy is not a practical benchmark.
- Five seeds provide directional evidence, not a conventional significance claim.
- The delayed round reuses the Stage B task split; some partial observations overlap training observations.
- The late-evidence task is a controlled single reveal followed by a final answer, not a full streaming system with irreversible output deadlines.
- The post-hoc probe is distribution-shifting and does not replace an independently trained `restart_soft` baseline.

The defensible conclusion is therefore narrow: **this implementation did not demonstrate a distinctive advantage for generating and revising an early draft online.**
