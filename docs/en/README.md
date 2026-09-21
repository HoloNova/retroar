# RetroAR English documentation

This is the compact, English-first reading path for the repository. The numbered Chinese files remain the complete chronological research log and are linked as primary records where protocol wording or historical context matters.

- **[Research question](research-question.md)** — the mechanism, task, and claim boundaries.
- **[Experiment matrix](experiment-matrix.md)** — what each committed round was designed to isolate.
- **[Results and evidence](results.md)** — headline numbers, evidence tiers, and failure interpretation.
- **[Reproducibility](reproducibility.md)** — environment, commands, safeguards, fingerprints, and resume rules.
- **[Repository map](repository-map.md)** — source modules, configurations, datasets, scripts, tests, and published artefacts.
- **[Literature and limitations](literature-and-limitations.md)** — overlap with prior work and why the result is narrow.
- **[Glossary](glossary.md)** — stable meanings for the recurring experimental terms.

## Recommended routes

| Reader | Start here | Then read |
|---|---|---|
| Wants only the conclusion | [Root README](../../README.md) | [Results](results.md) |
| Wants to understand the idea | [Research question](research-question.md) | [Experiment matrix](experiment-matrix.md) |
| Wants to audit the evidence | [Results](results.md) | [`results/evidence_table.json`](../../results/evidence_table.json), then the linked run files |
| Wants to reproduce it | [Reproducibility](reproducibility.md) | [`configs/`](../../configs), [`scripts/`](../../scripts) |
| Wants the full historical record | [中文索引](../00_索引.md) | Chinese protocols and analyses in numeric order |

## Canonical records

The English pages are an index and interpretation layer, not a replacement for raw evidence:

- **Machine-readable aggregation:** [`results/evidence_table.json`](../../results/evidence_table.json)
- **Generated table:** [`results/evidence_table.md`](../../results/evidence_table.md)
- **Published-tree provenance:** [`results/MANIFEST.json`](../../results/MANIFEST.json)
- **Full Chinese chronology:** [`docs/00_索引.md`](../00_索引.md)
- **Final Chinese synthesis:** [`docs/13_收尾实验与可行性总结.md`](../13_收尾实验与可行性总结.md)
- **Chinese evidence table:** [`docs/14_证据总表.md`](../14_证据总表.md)

The experiment IDs, configuration paths, and result paths are intentionally unchanged. This keeps old links, fingerprints, and audit references stable.
