# Policy-Evolving Edit Synthesis

A generative-track RSI method. It synthesises image-editing training pairs — a
source image, an edit instruction, and the edited result — and recursively
evolves the **instruction-writing policy** from the judge's decisions, rather
than evolving the dataset or the model.

## The loop

A run covers exactly one edit type and walks its scene list in a fixed order.
Samples are processed in batches of ten.

```text
      policy library --> instruction planner --> edit model --> 3-axis judge
             ^                 (one scene per sample)                |
             |                                                       v
      promote if supported in a 2nd batch <-- analyst reads the decided batch
             |
             +-- unsupported: stays an open claim; undecided for 2 batches: retired
```

Per batch, in order:

1. **Plan.** The next ten unattempted scene/edit pairs are taken from the
   schedule. The active policies are appended to the instruction planner's
   prompt.
2. **Write.** A vision-language planner sees the source image and the edit-type
   definition, then writes the instruction, the preserve list, and the intended
   end state.
3. **Edit.** An image model applies the instruction.
4. **Judge.** A frozen three-axis rubric accepts or rejects the pair. A pair is
   accepted only if the source image passes an environment-validity precheck and
   every axis clears both its own minimum and the aggregate:
   `instruction_following >= 4.0`, `visual_consistency >= 3.0`,
   `visual_quality >= 3.0`, and the equal-weight mean `>= 3.5`. One axis below
   its minimum fails the sample regardless of the mean.
5. **Reflect.** An analyst reads the decided batch — judgments, per-axis scores,
   failure reasons, instruction text — and proposes, revises, or withdraws
   claims about what makes an instruction succeed for this edit type.
6. **Gate.** A claim becomes an active policy only after being supported in two
   distinct batches; it is refused while contradicted at least as often as
   supported, and retired after two consecutive batches in which no sample could
   decide it.

A standing constraint on the analyst forbids any policy that raises the pass
rate by degrading the edit — asking for a tiny, barely visible change being the
typical case. It takes no verdicts and no policy may override it.

The analyst never sees the images and the judge never sees the policies, so
neither can argue with the other. Everything except the policies — the base
prompt, the rubric, the scene schedule — is held fixed across arms.

## Repository layout

```text
method.json                    Method manifest
README.md                      This file
README.zh-CN.md                Chinese translation
configs/
  rubrics/if_vc_vq.json        The frozen acceptance rubric
  environment/edits.jsonl      The 23 edit-type definitions
src/deltasynth/
  harness/evolve/              The RSI core: analyst, gate, policy state,
                               batch statistics, committer
  harness/                     Batch planners, the AgenticLoop orchestrator,
                               failure diagnosis, run ledger, coverage
  operators/recipe/            Instruction planner and compatibility gate
  operators/generate/          Source-image generation and edit application
  operators/verify/            The three-axis judge
  operators/filter/            Accept/reject tagging at the threshold
  core/, environment/          Sample schema, storage, parallelism, task universe
  serving/                     Abstract model interfaces
tests/                         Gate, retirement, and rubric unit tests
```

Arms are selected by `setting_id`: `random` and `stratified` are the baselines,
`evolving` adds policy evolution, and `policy-evolving` adds the standing
constraint on top — that last one is this method.

Two things are not included. The concrete provider adapters are excluded
because they are endpoint and credential plumbing; implement `LLMServingABC`,
`VLMServingABC`, and `ImageGenServingABC` from `src/deltasynth/serving/base.py`
against your own models. The scene pool is a dataset rather than part of the
method, so no environment snapshot ships here; `src/deltasynth/environment/schema.py`
specifies what one must contain.

The evaluation and training environment is pinned in
[`docker/generative-image-edit/`](../../../docker/generative-image-edit/).

## External services

An image-generation or image-edit endpoint; a vision-language planner; a
vision-language judge; and a text-only analyst LLM. Concrete provider adapters
are not in this tree: implement `LLMServingABC`, `VLMServingABC`, and
`ImageGenServingABC` against your own models. Credentials must come from the
runtime environment, never from configs.

## Safety, rollback, and budget

- **Frozen judge.** The three-axis rubric never sees the evolved policies, so
  the analyst cannot argue the acceptance rule into a higher pass rate.
- **Standing constraint.** Policies that raise yield by degrading the edit
  (typically a request for a barely visible change) are forbidden and cannot
  be overridden.
- **Promotion gate.** A claim becomes an active policy only after support in
  two distinct batches, is refused while contradicted at least as often as
  supported, and is retired after two consecutive undecided batches.
- **Rollback of a rejected claim is inherent:** it stays an open hypothesis or
  is retired; it is not written into the planner prompt.
- **Budget** is a fixed batch schedule (`max_batches × batch_size`) plus a
  hard cap on total API calls. There is no convergence criterion. The final
  batch skips reflection because no later batch could use it.

## Known limitations

- **Provider adapters and the scene pool are not in this contribution.** A
  rerun needs your own serving implementations and an environment snapshot
  that matches `src/deltasynth/environment/schema.py`.
- **Per-run seeds were not recorded** in the reference submissions.
- **Three edit types regress on acceptance rate** versus the stratified arm
  (H1, H2, G2). Mean yield still rises.
- **The destination-entropy mean excludes H3 and O9**, as documented in the
  table note; do not compare that mean against a 23-type average.
- **Downstream gains are small** relative to the in-loop yield gap. Transfer
  is measured by a hosted VLM judge, not a vendored deterministic grader.

## Results

Machine-readable submissions (status `unverified`):

- [`policy-evolving-edit-synthesis-qwen-image-edit-2511`](../../../results/submissions/policy-evolving-edit-synthesis-qwen-image-edit-2511/)
- [`policy-evolving-edit-synthesis-flux2-klein-9b`](../../../results/submissions/policy-evolving-edit-synthesis-flux2-klein-9b/)

The transfer suite is [`image-edit-transfer-suite`](../../../benchmarks/image-edit-transfer-suite/).
`primary_score` is `0.5 * (gedit_bench / 10 + imgedit_bench / 5)`.

Both arms produced 1,000 accepted pairs across the same 23 edit types, under the
same environment, rubric, and scene schedule.

### Acceptance rate (%)

| Edit type | Stratified | Policy-evolving | Δ |
| --- | ---: | ---: | ---: |
| O1 background_change | 81.2 | 82.0 | +0.8 |
| O2 material_change | 73.4 | 90.0 | +16.6 |
| O3 subject_replace | 90.8 | 92.0 | +1.2 |
| O4 object_remove | 75.3 | 86.0 | +10.7 |
| O5 object_add | 92.0 | 94.0 | +2.0 |
| O6 style_transfer_object | 77.1 | 84.0 | +6.9 |
| O7 color_change | 80.9 | 92.0 | +11.1 |
| O8 texture_change | 68.0 | 80.0 | +12.0 |
| O9 object_extract | 79.4 | 80.0 | +0.6 |
| T1 text_modify | 85.4 | 94.0 | +8.6 |
| T2 text_remove | 69.6 | 90.0 | +20.4 |
| T3 text_add | 75.0 | 84.0 | +9.0 |
| T4 text_translate | 45.8 | 80.0 | +34.2 |
| T5 text_replace | 87.0 | 96.0 | +9.0 |
| T6 text_layout | 53.5 | 76.0 | +22.5 |
| T7 text_style_change | 79.0 | 96.0 | +17.0 |
| H1 action_change | 95.1 | 90.0 | -5.1 |
| H2 appearance_change | 95.2 | 94.0 | -1.2 |
| H3 portrait_beautify | 65.2 | 80.0 | +14.8 |
| G1 style_transfer | 96.7 | 98.0 | +1.3 |
| G2 layout_change | 80.2 | 76.0 | -4.2 |
| G3 scene_change | 83.4 | 92.0 | +8.6 |
| G4 lighting_change | 92.2 | 94.0 | +1.8 |
| **Mean** | **79.2** | **87.8** | **+8.6** |

### Edit-destination diversity

Normalised Shannon entropy over edit destinations, 0 to 1, higher is more even.
Destinations from both arms were pooled, shuffled under a fixed seed, and
categorised blind — the categoriser built its own categories with no list
supplied — then split back by arm.

| Edit type | Stratified | Policy-evolving | Δ |
| --- | ---: | ---: | ---: |
| O1 background_change | 0.974 | 0.958 | -0.016 |
| O2 material_change | 0.912 | 0.950 | +0.038 |
| O3 subject_replace | 0.947 | 0.958 | +0.011 |
| O4 object_remove | 0.898 | 0.821 | -0.077 |
| O5 object_add | 0.982 | 0.962 | -0.020 |
| O6 style_transfer_object | 0.629 | 0.868 | +0.239 |
| O7 color_change | 0.897 | 0.941 | +0.044 |
| O8 texture_change | 0.678 | 0.860 | +0.182 |
| O9 object_extract | 0.998 | 0.996 | -0.001 |
| T1 text_modify | 0.969 | 0.967 | -0.003 |
| T2 text_remove | 0.810 | 0.810 | +0.000 |
| T3 text_add | 0.984 | 0.980 | -0.004 |
| T4 text_translate | 0.624 | 0.713 | +0.089 |
| T5 text_replace | 0.881 | 0.843 | -0.038 |
| T6 text_layout | 0.895 | 0.876 | -0.019 |
| T7 text_style_change | 0.634 | 0.812 | +0.177 |
| H1 action_change | 0.922 | 0.911 | -0.011 |
| H2 appearance_change | 0.818 | 0.921 | +0.104 |
| H3 portrait_beautify | 0.509 | 0.840 | +0.331 |
| G1 style_transfer | 0.734 | 0.876 | +0.142 |
| G2 layout_change | 0.962 | 0.993 | +0.031 |
| G3 scene_change | 0.908 | 0.886 | -0.022 |
| G4 lighting_change | 0.849 | 0.858 | +0.009 |
| **Mean (excluding H3 and O9)** | **0.853** | **0.894** | **+0.041** |

### Downstream training

Identical LoRA recipes on two base models, trained on each arm's 1,000 pairs.

| Base model | Training data | GEdit-Bench | ImgEdit-Bench |
| --- | --- | ---: | ---: |
| qwen-image-edit-2511 | none (baseline) | 8.104 | 4.449 |
| qwen-image-edit-2511 | stratified pipeline | 8.138 | 4.578 |
| qwen-image-edit-2511 | **policy-evolving** | **8.171** | **4.598** |
| flux2-klein-9b-base | none (baseline) | 7.653 | 4.111 |
| flux2-klein-9b-base | stratified pipeline | 7.844 | 4.188 |
| flux2-klein-9b-base | **policy-evolving** | **7.920** | **4.280** |
