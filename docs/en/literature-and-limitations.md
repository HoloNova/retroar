# Literature boundary and limitations

**Chinese source:** [04 — 文献核验](../04_文献核验.md) · [13 — 收尾实验与方向可行性总结](../13_收尾实验与可行性总结.md)

## Relation to nearby work

The repository is not a claim that revision is a new idea. The Chinese literature check compares the proposal with nearby directions including Mask-Predict, COrAL, Projected Autoregression, Corrector Sampling / RPT, and Stream of Revision. Those lines of work already cover important combinations of iterative refinement, editable suffixes, continuous or soft states, and correction after generation.

The narrow question here was whether this particular **online editable-tail schedule**, with a question-specific draft carried forward during generation, provides a distinctive benefit on the controlled task. The experiments did not establish that distinction.

## Main methodological limitations

1. **Small exact task.** Six free vertices and three colours are useful for mechanism diagnostics, but too small to support claims about language generation or practical systems.
2. **The task has an exact solver.** Backtracking reaches 100%, so model accuracy measures relative mechanism behaviour rather than usefulness.
3. **Descriptive seed counts.** Five seeds are directional evidence; they are not a conventional significance study. The minimum two-sided sign-test p-value for 5/5 is 0.0625.
4. **Shared task family.** The delayed round reuses the Stage B split. Full graphs are isomorphism-disjoint, but partial observations can overlap training observations.
5. **Not full streaming.** The delayed task reveals evidence once and then produces a final answer. It does not impose an irreversible output deadline while evidence continues to arrive.
6. **Post-hoc soft probe.** Replacing the early state is an intervention on frozen models. It is not an independently trained `restart_soft` baseline and may induce distribution shift.
7. **Historical controls evolved.** The first long run's extra-compute comparison was later identified as insufficiently matched; later protocols explicitly strengthened the controls.

## Correct conclusion strength

The strongest defensible statement is:

> On this implementation and task family, the repository repeatedly finds effective iterative solving after complete evidence is available, but does not show that generating and revising an early draft online has a distinctive advantage over simpler alternatives.

That conclusion supports stopping the current implementation line. It does not declare the entire research area invalid. A future project would need a genuinely new question, an independently trained soft restart baseline, stronger streaming constraints, and cross-task validation before reusing the original story.
