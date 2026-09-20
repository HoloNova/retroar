# RetroAR — research log of a hypothesis that did not survive its own experiments

This repository is the complete record of a small, self-contained research idea and its
falsification. It contains the code, the pre-registered protocols, all run data, and the
analysis — including the parts where the idea failed.

**Short version:** we asked whether an autoregressive generator becomes better if its most
recent outputs stay *uncommitted* — held as modifiable drafts, optionally as probability
distributions — and are revised as generation continues. Four rounds of controlled
experiments on a small CPU-only graph-colouring task say **no, not in the way the idea
needed**. Revising a finished sequence beats revising while generating, and a diagnostic
showed that the strongest configuration scored just as well when its question-specific
early draft was replaced by a constant.

**Status: archived research log. No further training is planned.** All documents are written
in Chinese; this README gives the English summary and links into them.

---

## 1. The idea in one paragraph

Human speech is drafted and corrected before it is uttered. A transformer does not work that
way: once a token is written it is frozen and becomes the context for everything after it.
RetroAR asked whether the model should instead keep a short tail of positions as *soft draft*
that continues to be corrected as new positions appear, with the tail eventually committed.
The mechanism is intentionally simple: a generation module, a revision module, and a small
editable draft buffer. No pretrained model is used anywhere in this repository.

## 2. What was tested, and what the data says

All numbers are whole-question accuracy (every one of the six free positions correct), on the
held-out test split, averaged over seeds. `results/evidence_table.json` holds the machine-readable
version of every row, including seed lists and forward-call counts.

| Round | Question it isolated | Seeds | Headline result |
|---|---|---|---|
| First long run (`long_v1`) | Does editing uncommitted positions at all help? | 3 | revision 85.0–87.4% vs plain generation 77.5% |
| Stage A (`stage_a_v1`) | Online revision vs. revising after generation | 5 | after-generation 88.3–92.5% > online 84.5–88.1% |
| Stage B (`stage_b_v1`) | Matched budgets: sampling, double training, supervision | 5 new | after-generation 90.9–93.4% > online 84.7–85.2%; two-candidate sampling 82.0%; double training did not help |
| Belief (`belief_v1`) | Was "soft draft" ever fairly tested? | 5 new | flattening the draft to a real distribution did not help at any temperature (T=2…8); high T was worse |
| Delayed evidence (`delayed_v1`) | Does revising a draft help when evidence arrives late? | 5 new | revising 80.2% ≈ two-candidate waiting 79.5% (failed the pre-registered bar); soft config 90.2% but see below |

Three findings are worth stating plainly:

1. **Revising the whole unfinished sequence helps; revising it *while generating* does not.**
   The ordering held across two independent 5-seed rounds (9/10 and 5/5 seeds). The online
   variant also repaired only ~46% of wrong answers while breaking 28–36 previously correct
   ones; the after-generation variant repaired ~76% and broke 6.
2. **Keeping probabilities instead of hard choices is not the source of the gain.** A belief-temperature
   sweep produced genuinely non-collapsed drafts (state entropy 0.03 → 0.88 nats, max probability
   0.99 → 0.64) and the gain did not appear at any temperature. Pushing uncertainty up made results
   worse and higher-variance. This constrains *this* representation intervention; it does not prove
   every categorical-belief mechanism is useless.
3. **The best configuration did not need the draft the idea was built on.** The 90.2% soft result in
   the delayed-evidence round survives replacing the entire early draft with a uniform 1/3 per colour
   (90.4%), with another question's draft (89.9%), or with zeros (90.6%). The early draft's entropy was
   ~1.09 of a maximum 1.099 nats, i.e. nearly uninformative — consistent with the guess being formed
   later, from the complete evidence, during the parallel revision passes.

## 3. What is *not* claimed

- Not a claim that iterative revision is useless — it clearly helps here, and that is **already
  covered by prior work** (Mask-Predict, COrAL, Projected Autoregression, Corrector Sampling/RPT,
  Stream of Revision; see [docs/04](docs/04_文献核验.md)).
- Not a claim about language models, natural language, or production systems. This is a six-vertex
  graph-colouring task with a 30k-parameter model on a 2-core CPU.
- Not a claim that "online revision is impossible" — only that the implementations tested here do not
  beat the stronger alternatives, and that the pre-registered bar was not met.
- Not a claim of statistical significance in the usual sense. Five seeds give a minimum two-sided
  sign-test p of 0.0625; the reported p-values are descriptive. Round 1's 3 seeds are worse still.
- The task is solved to 100% by classical backtracking search. All accuracy numbers are therefore
  *mechanism diagnostics*, never evidence of practical usefulness.

## 4. Repository layout

```
retroar_min/     model, rollout policies, training, suites (delayed/), supervisors
configs/         JSON experiment configs (smoke, pilot, long, stage A/B, belief, delayed)
scripts/         result aggregation (collect_evidence.py), data-driven probes,
                 results slimming for publication (slim_results.py)
tests/           41 regression tests: leakage, masking, rollout invariants,
                 supervision parity, scheduler guards, resume determinism
data/            generated datasets (unique-solution graph colouring, isomorphism-deduped)
results/         all run data: aggregated metrics per run, per-experiment summaries,
                 reports, status files, machine-readable evidence table
docs/            Chinese research documents in chronological order — see docs/00_索引.md
```

Document index (Chinese, in reading order): **[docs/00_索引.md](docs/00_索引.md)**. The two
documents a reader should open first are the closing summary
([docs/13](docs/13_收尾实验与可行性总结.md)) and the literature overlap check
([docs/04](docs/04_文献核验.md)).

## 5. Reproducing

CPU-only, single-threaded, memory-guarded. The original machine had 2 cores and 3.8 GiB RAM;
every suite was designed around that ceiling (≤2 worker processes, RSS cap 1 GiB per worker,
stop if host-available RAM drops below 768 MiB).

```bash
uv venv .venv && .venv/bin/pip install -r requirements-cpu.lock     # torch 2.10.0+cpu
.venv/bin/python -m unittest discover -s tests -v                    # 41 tests
.venv/bin/python -m retroar_min.run_local --config configs/cpu_long.json --output results/long_v1
.venv/bin/python -m retroar_min.delayed --config configs/cpu_delayed.json --output results/delayed_v1
.venv/bin/python -m scripts.collect_evidence                         # rebuild results/evidence_table.json
```

Every suite is resumable, refuses to overwrite finished runs, compares source/dataset
fingerprints before resuming, and writes `blocked` (not success) when it stops on a resource
guard. Training is seeded and deterministic (`use_deterministic_algorithms`), so a re-run with
the same config reproduces the same weights.

## 6. Data policy

`results/` contains the published tree: per-run aggregated metrics (`serial_test.json`),
training curves, summaries, reports and status files — about 8 MiB, 1309 files. Two things were
deliberately left out to keep the repository small, and `results/MANIFEST.json` records the
source hash of every published file plus exactly what was removed:

- model checkpoints (`*.pt`, ~37 MiB) and console logs;
- the per-example prediction dumps inside `serial_test.json` (`policies[*].details`, ~242 MiB).
  Each affected file lists the number of removed records and the SHA-256 of the removed payload.

The full local tree (~300 MiB) is kept out of git as `results_full/`; `scripts/slim_results.py`
rebuilds the published tree from it, and re-running the suites with the same seeds regenerates
the dumps.

One caveat worth repeating: the delayed-evidence round reuses the stage B dataset split. Full
graphs are isomorphism-disjoint across splits, but 5/128 validation and 20/256 test
*partial observations* coincide with training observations, and the rounds with new seeds are
new training runs on a shared task family — not an independent replication on new data.

## 7. License

MIT — see [LICENSE](LICENSE). The datasets and result files are provided as-is, with no warranty
of correctness or fitness for any purpose; use the numbers together with the limitations above.
