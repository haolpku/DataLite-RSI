# Evolution loop and multimodal extension boundary

`rsi.framework.run` owns the active evolution loop: fixed input,
PipelineAgent challenger, compile and execution repair, candidate evaluation,
and best-so-far incumbent selection. Text, image, video, and mixed tasks all
use `method_id: "dataflow-evolver"`. The `rsi/methods/` trees are source
references; the active runtime imports no code, config, or skill from them.

The implementation is rooted at [`rsi/framework/`](../rsi/framework/README.md):
`core/` supplies the operator and pipeline contracts, checkpoint and provenance;
`io/` supplies step storage, `RecordStore`, `ArtifactStore`, `BlobStore` and
serving; `runtime/` loads tasks and skills; `evolution/` contains the candidate
loop, agents, execution wrapper, evaluation, provider sessions and telemetry.
Configurations and skills live under `rsi/framework/configs/` and
`rsi/framework/skills/`.

The [mixed task template](../rsi/framework/tasks/dataflow-mixed.json) and its
[private settings](../rsi/framework/configs/multimodal.example.yaml) describe
a fixed JSONL entry. A row can carry several modalities:

```json
{"instruction":"What changes?","image":{"kind":"image","uri":"media/frame.png","media_type":"image/png"},"video":{"kind":"video_reference","uri":"media/clip.mp4","media_type":"video/mp4"}}
```

`InputContract` validates declared fields and artifact references. It does not
decode media or verify remote URI availability. Operators can use
`rsi.framework.resolve_local_input_artifact` for local source files, then write
images through `rsi.framework.artifact_store_for(ITERATION_DIR).put_image` and
keep video references through its `put_video_reference`. Timestamped evidence can
remain structured records. Generated image blobs are moved into the final run's
`ArtifactStore`; a run-local blob reference invalidates cross-run step reuse.
Checkpoints bind the fixed JSONL content hash, so replacing content at the same
path does not silently resume an old candidate.

Visual or mixed tasks must provide a task-owned evaluator:

```python
from rsi.framework import CandidateFeedback, run

class TaskEvaluator:
    def review(self, dataset_path, task, **kwargs):
        # Inspect the actual image/video evidence and apply this task's gates.
        return CandidateFeedback(
            score=0.7,
            passed=True,
            domain_feedback={"example_metric": 0.7},
        )

manifest = run(
    "rsi/framework/tasks/dataflow-mixed.json",
    resources={"candidate_evaluator": TaskEvaluator()},
)
```

The example score illustrates the interface, not an implemented visual metric.
The evaluator's score must be comparable across candidates in one task, finite,
and within `[0, 1]`; `domain_feedback` must be JSON-serializable. The loop
compares `(passed, score)` and retains the incumbent on a tie. Task evidence
feeds the next proposal in bounded form. PipelineDiagnosticAgent is read-only
and its artifact never enters this comparison. Text-only tasks without an
injected evaluator use ReviewAgent, hard metrics, DAS, Vendi, and contamination
checks as configured.

`PipelineLLMServing` is a text transport. Visual model calls, image editing,
temporal evidence, and task-specific scoring belong to future operators and
evaluators attached to the same loop. The current tests run two candidates
with tiny image bytes and a video reference; they do not establish a live
visual model run. Policy IF/VC/VQ and VideoRSI frozen-target/frontier feedback
have not been connected to this controller.

Generated code imports the step-based contracts from `rsi.framework`
(`OperatorABC`, `PipelineABC`, `FileStorage`, `PipelineLLMServing`); provider
authoring skills teach the same imports. Media artifacts are reached through
`rsi.framework.artifact_store_for(ITERATION_DIR)`, a DataLite extension with no
counterpart in the reference runtime. The diagnostic role uses a separate
built-in prompt and isolated runtime. The generated pipeline has no
`open-dataflow` dependency.
