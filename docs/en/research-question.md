# Research question and claim boundary

**Chinese source:** [01 — 研究构想与交接说明](../01_研究构想与交接说明.md) · [02 — 最小实验设计与本机运行评估](../02_最小实验设计与本机运行评估.md)

## The proposed mechanism

A standard autoregressive generator commits each new token or position and feeds it back as fixed context. RetroAR explored a different schedule:

1. generate a short tail;
2. keep that tail as an editable draft;
3. optionally represent each draft position as a categorical distribution rather than a hard colour;
4. revise the draft as more positions or external evidence become available;
5. eventually commit the result.

The implementation contains a generation module, a revision module, and a small editable state. It uses no pretrained model.

## What was tested

The committed code tests variants of this schedule on a six-vertex, three-colour graph-colouring task. The model is small—roughly 30k parameters—and the task can be solved to 100% by classical backtracking. Therefore the score is a controlled mechanism diagnostic, not a product benchmark.

The main comparisons ask:

- whether editing unfinished positions helps at all;
- whether online revision is better than revising after generation;
- whether a matched forward-call budget changes that answer;
- whether a non-collapsed probability state is the relevant source of gain;
- whether a draft formed before late evidence is useful once that evidence arrives.

## What the evidence supports

- Iterative revision after the relevant evidence is available is effective on this task.
- The tested online schedule did not beat the stronger after-generation alternative.
- The temperature sweep produced real soft states, but did not establish the preregistered advantage for online revision.
- The delayed-evidence soft result is a real observed signal, but its early state was nearly uniform and could be replaced without lowering final accuracy.

## What it does not support

This repository does **not** establish any of the following:

- that iterative refinement is useless;
- that online revision is impossible in general;
- that the same result holds for language models or natural language;
- that the implementation is practically useful;
- that a categorical belief mechanism is universally ineffective;
- that five-seed descriptive comparisons are conventionally statistically significant.

The smallest two-sided sign-test p-value for five positive paired seeds is 0.0625. The published language therefore stays descriptive and mechanism-specific.

## Boundary with prior work

Revision, editable suffixes, soft or continuous states, and iterative refinement all overlap existing research directions. The novelty question was not settled by this repository; see [Literature and limitations](literature-and-limitations.md) and the full [Chinese literature check](../04_文献核验.md).
