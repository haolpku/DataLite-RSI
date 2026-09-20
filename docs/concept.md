# Less is More for RSI

DataLite-RSI studies a practical question: can recursive feedback improve the
**choice and construction of training data**, allowing a compact final dataset
to produce useful model improvements?

The recursive state can be a selection rule, instruction policy, processing
program, or data corpus. Feedback from an earlier attempt changes a later data
decision. Depending on the method, model training occurs inside the loop or after
pipeline search. This is a collection of distinct mechanisms, not one shared
algorithm or a single cross-domain leaderboard.

## What “less” means here

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
Evolver improves over the base model but loses to original Math-3K under its
reported protocol. Video results lack a matched-data non-recursive control.
Some test subsets contribute feedback, and several methods have incomplete
public artifacts. These boundaries are part of the research record.

Start with the [homepage](https://haolpku.github.io/DataLite-RSI/), inspect the
[protocols](protocol.md), then follow [reproduction](reproduction.md) and
[open evidence gaps](release-gaps.md).
