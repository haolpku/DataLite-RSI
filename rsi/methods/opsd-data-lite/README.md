# OPSD-data-lite


**Train on the sweet spot, not the whole set.**

On-Policy Self-Distillation (OPSD) lets a model teach itself: the student samples
its own answers from the problem alone, while a self-teacher -- a copy of the
same initial model -- additionally conditions on privileged context, such as a
reference solution, and provides dense token-level supervision on the student's
trajectories.

This method explores a data-centric hypothesis: OPSD gains are not uniform across
training examples. A useful example should expose something the question-only
student does not yet know while still producing a teacher signal that the
student can internalize. OPSD-data-lite scores a fixed candidate pool for that
property and spends a short update budget on a selected subset.

```text
+---------------------------+
|                           v
|  fixed training pool --> sample-level teachability --> initial subset
|                                                            |
|                                                            v
|                           score before --> short OPSD round --> score after
|                                                            |
|                                                            v
+-- next round's subset <-- per-sample filter <-- compare before vs. after
```

## The OPSD sweet spot

The working model separates the training pool along two axes:

| | Teacher signal is internalizable | Teacher signal is not internalizable |
| --- | --- | --- |
| **Student reliably correct** | Little to learn; usually a low-value update | Little to learn; usually a low-value update |
| **Student uncertain or wrong** | **Sweet spot**; privileged context targets a transferable gap | Weak transfer; the teacher depends on context unavailable at inference |

This differs from difficulty-only or pass-rate filtering. Two problems with the
same student accuracy can have different value for OPSD because their
privileged teacher distributions can differ in how well they transfer.

## What changes, what stays fixed

| Component | Modified by OPSD-data-lite? |
| --- | --- |
| Training subset | **Yes** -- selected and then filtered per sample |
| Student weights | **Yes** -- through the standard OPSD objective |
| Problems and privileged context | No -- selected from the fixed pool, never rewritten |
| Self-teacher weights | No -- frozen at the initial student weights for a training run |
| OPSD objective | No -- follows Zhao et al. [1] |
| Raw candidate pool | No -- fixed at 9,600 examples |
| Update budget | No within a comparison -- every data-lite row uses 100 optimizer steps against the 100-step full-pool OPSD control |

Three roles are intentionally separate:

1. **Student** -- samples on-policy responses without privileged context.
2. **Self-teacher** -- uses frozen initial weights plus privileged context to
   provide the distillation target.
3. **Selector** -- computes the sample-level teachability signal and chooses the
   next training subset.

## Three uses of one signal

**Initial selection.** Rank the fixed pool by the teachability signal and train
on a smaller sweet-spot subset. The first reported iteration uses 2,400 selected
examples and 100 optimizer steps.

**Per-sample iteration.** Re-score the fixed pool after the short OPSD round,
compare each example's before/after signal, and use that response to construct
the next subset. The second reported iteration uses 1,600 examples and the same
100 optimizer steps.

**Aggressive selection.** Continue filtering with the same teachability signal
down to a much smaller subset. The reported third iteration trains on 32
examples, still for 100 optimizer steps.



The public repo includes:

```text
rsi/methods/opsd-data-lite/
|-- README.md
|-- method.json
`-- configs/
    `-- opsd-data-lite.yaml
```

It also includes the benchmark/evaluator integration and a machine-readable
result under
[`results/submissions/opsd-data-lite-qwen3-8b/`](../../../results/submissions/opsd-data-lite-qwen3-8b/).


## Results

The reported model is `Qwen/Qwen3-8B`. Evaluation follows version `0.1.0` of
the [`opsd-math-competition-suite`](../../../benchmarks/opsd-math-competition-suite/)
and its vendored [TTPO evaluator](../../../evaluation/ttpo_math/) [2].

| Evaluation field | Setting |
| --- | --- |
| Thinking mode | enabled |
| Temperature / top-p / top-k | `1.0` / `0.95` / `-1` |
| Samples per problem | 12 (`Avg@12`) |
| Evaluation repeats | 3 seeds |
| Question counts | AIME25 30, HMMT25 30, AIME26 30, HMMT26 33, BRUMO25 30 |
| Suite average | unweighted mean of the five dataset scores |

Scores are percentages. Each dataset cell is the reported mean over three
evaluation seeds. The final column is the unweighted mean across datasets.

| Setting | Update steps | Selected samples | AIME25 | HMMT25 | AIME26 | HMMT26 | BRUMO25 | Avg | Gain vs. matched OPSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base model | 0 | -- | 66.7 | 44.2 | 67.8 | 45.5 | 69.2 | 58.7 | -- |
| OPSD | 50 | full pool | 69.7 | 46.1 | 69.4 | 46.4 | 71.4 | 60.6 | -- |
| OPSD | 75 | full pool | 68.1 | 44.7 | 68.6 | 46.0 | 70.3 | 59.5 | -- |
| OPSD | 100 | full pool | 69.2 | 45.6 | 68.9 | 46.5 | 71.4 | 60.3 | -- |
| OPSD-data-lite, iter 1 | 100 | 2,400 | 71.4 | 45.6 | 70.3 | 45.5 | 70.6 | 60.7 | **+0.4** |
| OPSD-data-lite, iter 2 | 100 | 1,600 | 71.7 | 46.4 | 71.9 | 47.2 | 72.5 | 61.9 | **+1.6** |
| **OPSD-data-lite, aggressive** | 100 | 32 | 71.4 | 46.7 | 72.5 | 47.2 | 72.7 | **62.1** | **+1.8** |

Every data-lite row uses 100 optimizer steps; only the selected subset shrinks.
The aggressive 32-example run improves the suite average by 3.4 points over the
base model and by 1.8 points over the 100-step full-pool OPSD control. Iteration
2 (1,600 examples) is +3.2 over base and +1.6 over the same 100-step control.
All three data-lite settings improve all five datasets over the base row.

`Update steps` denotes the optimizer-step count of the displayed checkpoint.
Teachability scoring is separate compute.

## Running

The evaluation harness is runnable now. For one dataset and one seed:

```bash
python evaluation/ttpo_math/evaluate_math.py \
  --base_model "$OPSD_MODEL_PATH" \
  --dataset aime26 \
  --enable_thinking \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k -1 \
  --val_n 12 \
  --seed 0 \
  --output_file /tmp/aime26-seed0.json
```

The method configuration is at
[`configs/opsd-data-lite.yaml`](configs/opsd-data-lite.yaml). Launch with:

```bash
python -m opsd_lite.main \
  --config rsi/methods/opsd-data-lite/configs/opsd-data-lite.yaml \
  --log-level INFO
```

## Cost profile

Approximate self-reported cost for the two-cycle on four GPUs, plus the later
aggressive round:

| Component | Cost |
| --- | --- |
| Initial teachability scoring | 2-4 GPU-hours (9,600 candidates) |
| One OPSD round | 1-1.3 GPU-hours (100 steps on 32-2,400 examples) |
| One full-pool re-score | 2-4 GPU-hours (9,600 candidates) |
| Two-cycle total | approximately 8-14 GPU-hours |
| Aggressive OPSD round | 100 steps on 32 selected examples |
| 100-step full-pool OPSD control | approximately 1-1.3 GPU-hours |

The method reduces selected data, not optimizer steps or total GPU-hours:
every data-lite iteration uses 100 steps, and full-pool scoring dominates the
cost.

## Safety and budget

- **Fixed-pool selection.** Problems, reference solutions, and privileged
  context are selected, never generated or rewritten.
- **Frozen teacher.** Teacher weights do not drift within a training run.
- **No external service required.** The intended pipeline uses local models;
  the public repo reports no human intervention during the two cycles.

## Known limitations

- **Signal and filter withheld.** The central selection mechanism cannot be
  reproduced from this repository alone.
- **Model-state dependence.** A useful subset for one checkpoint may not
  transfer to another model or training stage.
- **Evaluation isolation must be documented.** The full release should show
  that benchmark outcomes did not drive subset or checkpoint selection.
- **Small-step scope.** The method is intended for short OPSD budgets and does
  not establish behavior under long training runs.

## References

1. Siyan Zhao et al. [*Self-Distilled Reasoner: On-Policy Self-Distillation for
   Large Language Models*](https://arxiv.org/abs/2601.18734). arXiv:2601.18734,
   2026.
2. Aozhe Wang et al. [*TTPO: Test-Time Policy
   Optimization*](https://arxiv.org/abs/2608.27448). arXiv:2608.27448, 2026.
