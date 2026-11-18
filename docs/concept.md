# DataFlow-RSI: Bringing Recursive Self-Improvement to Data Production

DataFlow-RSI turns task feedback into reusable data-production programs.
The active framework evolves executable operators and the code that composes
them over a fixed source corpus. Users supply output requirements, quality
criteria, and the feedback sources appropriate to the task.

## The pipeline is the recursive state

1. A pipeline agent authors a candidate or revises a retained program.
2. The framework executes it and evaluates the resulting dataset.
3. Task feedback, execution diagnostics, and history guide the next revision.
4. The selected program can produce further data using the retained strategy.

Revisions can change generation, verification, selection, and pipeline structure.
The framework provides a common execution and evolution mechanism; individual
tasks supply their own requirements and evaluation. Text review is built in,
and visual tasks connect suitable operators and evaluation signals. Downstream
training can provide additional feedback; it is optional.

See the [framework](../rsi/framework/README.md),
[feedback sources](feedback-sources.md), and
[multimodal interfaces](dataflow-multimodal-extension.md) for implementation details.

## DataLite-RSI: Data efficiency

**DataLite-RSI is a research direction within DataFlow-RSI.** It asks when
feedback can improve the choice and construction of training data enough to
make a compact final dataset more useful: **Less is More for RSI**.

The reference experiments study distinct mechanisms: selecting examples,
evolving synthesis policies, refining programs, and revising a corpus. They
inform the data-efficiency research direction; their scores are not a single
cross-domain leaderboard or evidence that every method uses the active loop.

### What “less” means here

| Evidence type | Current case | What it supports |
| --- | --- | --- |
| Smaller selected training sets | OPSD-Data-Lite | Final 32-example round versus a 9,600-example full-pool control at 100 reported optimizer steps |
| Better data at a fixed final budget | Policy-Evolving Edit Synthesis | Evolving policies versus stratified synthesis, both with 1,000 accepted pairs |
| Transfer from a compact dataset | VideoRSI | 4,541 SFT examples tested against base models across five families |
| Program and corpus refinement | Evolver / Self-Improver | Different recursive data-processing mechanisms, including mixed comparisons |

A compact final set does not imply that only those examples were inspected.
OPSD scores a larger candidate pool, image synthesis rejects candidates, and
program search adds evaluation and generation work. Report candidate access,
accepted data, training exposures, and total compute separately.

## What the evidence does not establish

The current collection does not demonstrate that less data always wins, that
recursion always beats one-shot selection, or that total compute is lower.
Evolver improves over the base model but loses to DataFlow's expert-authored
Math-3K reference under its
reported protocol. Video results lack a matched-data non-recursive control.
Some test subsets contribute feedback, and several methods have incomplete
public artifacts. These boundaries are part of the research record.

Start with the [homepage](https://haolpku.github.io/DataFlow-RSI/), inspect the
[protocols](protocol.md), then follow [reproduction](reproduction.md) and
[open evidence gaps](release-gaps.md).
