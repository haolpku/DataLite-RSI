# Video RSI skill: what it does and how to run it

## What the skill does

The Video RSI skill is provider-specific guidance for the coding agent that
builds or repairs a video-data pipeline. It asks the agent to route records by
available evidence (captions/entities, event graphs, video-only, or invalid
provenance), choose task-specific operators, and preserve evidence and rejection
provenance. It also gives quality-design priors for visual grounding, caption
shortcut risk, distractor contrast, frame visibility, answer-position balance,
and diversity across questions from the same video.

The skill does not run a model, start a GPU service, train model weights, or
evaluate Video-MME. The framework executes the generated pipeline and manages
iteration, artifact validation, cache reuse, and candidate promotion. Provider
skill copies live under `providers/codex/`, `providers/claude/`, and
`providers/opencode/`; the task selects them with
`metadata.pipeline_skill_profile: video-rsi-router`.

## Choose the evaluation-model role

- `target`: use only when the evaluation endpoint serves the same model family
  and checkpoint whose downstream training benefit the campaign is intended
  to optimize. Keep it fixed for the campaign. A GPU by itself does not make a
  model a target; the endpoint must serve that exact target.
- `reference`: use for a general-purpose API or another model that supplies
  quality feedback but is not the training target. Report its judgments as
  reference feedback, never as target-model gains.
- `none`: omit model-based auditing. Deterministic schema/provenance checks can
  still run, but a pipeline's own quality labels do not count as independent
  verification or promotion evidence.

The runner machine does not need a GPU. Pipeline generation and evaluation
models are accessed through configured services. If your setup has no GPU
serving the actual downstream target model, select `reference` for a general
VLM API; do not call it `target` merely because it can judge video examples.
For a true target-model campaign, the endpoint must serve the exact model and
checkpoint being optimized (for example, your own GPU-hosted service); otherwise
its judgments are reference feedback. The endpoint must accept sampled video
frames or resolvable media references.

## Run the skill

1. Install the framework dependencies and the selected coding-agent CLI (Codex,
   Claude, or OpenCode). Configure the provider credentials in the environment.
2. Set the fixed input corpus and a writable run directory in
   `rsi/framework/configs/video-rsi.example.yaml`. Configure
   `VIDEO_RSI_PIPELINE_API_URL`, `VIDEO_RSI_PIPELINE_MODEL`, and
   `VIDEO_RSI_PIPELINE_API_KEY` for the model used by generated operators. This
   is separate from the target/reference evaluator.
3. Inject a video evaluator when calling the framework. The video task requires
   `candidate_evaluator`; the CLI currently does not construct a model auditor
   automatically:

   ```python
   from rsi.framework import run
   from rsi.framework.evolution.evaluation import VideoRSICandidateEvaluator
   from my_project.feedback import make_signal_provider

   evaluator = VideoRSICandidateEvaluator(
       signal_provider=make_signal_provider(),  # local, CPU, or remote API adapter
       model_role="reference",                  # use "target" only for the real training target
       model_name="provider/vision-model-id",
   )
   manifest = run(
       "rsi/framework/tasks/video-rsi-router.json",
       resources={"candidate_evaluator": evaluator},
   )
   print(manifest["run_id"], manifest["status"])
   ```

   `make_signal_provider()` is an application-supplied callable that receives a
   candidate record and returns independent audit fields such as
   `visual_grounding_outcome`, `caption_shortcut_risk`, `distractor_quality`,
   and `training_value_label`. The framework intentionally does not assume a
   specific model family or ship a universal video-API adapter.

The task template is `rsi/framework/tasks/video-rsi-router.json`; run artifacts
are written beneath the configured workspace's `runs/<run_id>/`. Inspect the
run manifest, candidate feedback, and generated dataset before using it for
training. This skill covers data-pipeline evolution, not the subsequent model
training/evaluation grid.
