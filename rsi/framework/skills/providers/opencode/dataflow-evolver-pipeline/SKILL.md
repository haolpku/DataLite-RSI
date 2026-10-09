---
name: dataflow-evolver-pipeline
description: Write or repair a data-processing pipeline for a DataLite-RSI evolution iteration.
---

# Pipeline authoring

Use this skill only for the `pipeline_builder` and `pipeline_repairer` roles. The
diagnostic agent uses a separate read-only system prompt and must never load this
skill. The framework owns compile, execution, observation, cache selection,
candidate evaluation, and best-so-far acceptance. Text tasks use ReviewAgent;
the default rubric is four dimensions, while a domain config may select
criteria mode so the agent receives task.quality_criteria evidence instead;
image/video tasks require a task-specific evaluator. Do not execute
`pipeline.py` yourself.

## Runtime and file layout

Generate `{iteration_dir}/pipeline.py`, one module per custom operator under
`{iteration_dir}/operators/`, and `{iteration_dir}/decision.json`.

Import the runtime contracts from `rsi.framework`:

- `OperatorABC` and `LLMServingABC` for operators and injected serving
- `PipelineABC` for the pipeline
- `FileStorage` for step-based record IO
- `PipelineLLMServing` for an OpenAI-compatible API operator

Do not import the `dataflow` package; it is not installed and the framework
rejects any artifact that imports it. Use the contracts below when authoring
generated code.

The input is a fixed structured entry file supplied as `ENTRY_PATH`. A task may
declare text, image, video, or several modalities in `input_contract`.
Image/video fields are artifact references with `kind`, `uri` and `media_type`;
they are not inline binary data. Resolve local source media with
`rsi.framework.resolve_local_input_artifact(ENTRY_PATH, ref)`; remote URIs need
a task-specific backend. Inspect the entry read-only before designing the
pipeline: assess source distribution, schema, modality references, roles,
turns, language, length, placeholders, duplicates and task-specific fields.
Stream samples and counts; never place the full corpus or media bytes in the
Agent context. Do not fetch external datasets. A configured serving API may be
used for transformations.

## FileStorage contract

Construct one sequential storage object:

```python
self.storage = FileStorage(
    first_entry_file_name=ENTRY_PATH,
    cache_path=CACHE_DIR,
    file_name_prefix="pipeline_step",
    cache_type="jsonl",
)
```

`FileStorage` is step-based, not key-based:

- `storage.step()` takes no arguments. It advances the pipeline step and returns
  the storage view passed to one operator.
- `storage.read("dict")` returns the current step as `list[dict]`, with missing
  values as Python `None`.
- `storage.read()` returns a pandas DataFrame.
- `storage.write(rows)` writes the next step and returns its file path.
- Do not pass logical input/output keys to `step`, `read`, or `write`.

Step 0 is the entry file and its format comes from that file's extension
(`.jsonl`, `.json`, `.csv`, `.parquet`). Later steps use `cache_type`. Values
are parsed by pandas, so a column's type can shift: an integer column
containing a null reads back as `float`, and a column of numeric strings reads
back as numbers. Validate what a field actually holds rather than assuming the
literal JSON type, and do not rely on `row[key]` being a `str` merely because
the source file quoted it.

`storage.write` accepts a non-empty `list[dict]` or a DataFrame. An empty list
raises, so handle a zero-row result explicitly rather than writing it.

### Batched and streaming variants

Use the default `PipelineABC` + `FileStorage` pair unless the task needs to
process a large corpus in fixed batches. The variants must be paired:

```python
from rsi.framework import (
    BatchedFileStorage, BatchedPipelineABC,
    StreamBatchedFileStorage, StreamBatchedPipelineABC,
)
```

Use `BatchedPipelineABC` with `BatchedFileStorage` when each batch may be
loaded from the current step and the pipeline should append later batches to
the same output step. Use `StreamBatchedPipelineABC` with
`StreamBatchedFileStorage` when a step should be read through
`iter_chunks()` without materializing the whole file. `BatchedFileStorage`
accepts only `jsonl` or `csv` caches. The framework owns the batch loop and
resume marker; the entry point is:

```python
pipeline.compile()
if os.getenv("DF_COMPILE_ONLY") == "1":
    raise SystemExit(0)
pipeline.forward(batch_size=256, resume_from_last=True)
```

Do not mix a batched pipeline with ordinary `FileStorage`, do not mix the
streaming pipeline with `BatchedFileStorage`, and do not manually slice rows
inside an operator. `batch_size` controls data processing, not LLM HTTP
concurrency; serving keeps its own `max_workers` setting. For a normal-sized
corpus, keep the simpler `PipelineABC` + `FileStorage` contract.

## Framework-owned prefix reuse

Step filenames are execution-relative, not operator-relative. If a prefix
operator is skipped, every downstream `pipeline_step_stepN.jsonl` number moves
forward. Never infer an artifact producer from its filename, inspect parent
cache files to choose one, or implement custom cache validation. Parseable
JSONL is not proof of producer identity.

Every pipeline must mechanically consume the framework decision:

```python
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

OPERATOR_NAMES = ["FirstOperator", "SecondOperator"]
self.framework_cache = FrameworkStepCache.from_env(
    raw_entry_path=ENTRY_PATH,
    operator_names=OPERATOR_NAMES,
)
self.storage = FileStorage(
    first_entry_file_name=self.framework_cache.entry_path,
    cache_path=CACHE_DIR,
    file_name_prefix="pipeline_step",
    cache_type="jsonl",
)
```

Use stable zero-based logical indexes in `forward()`:

```python
def forward(self) -> None:
    if self.framework_cache.should_run(0):
        self.first_operator.run(storage=self.storage.step())
    if self.framework_cache.should_run(1):
        self.second_operator.run(storage=self.storage.step())
```

The framework validates raw-input identity and the ordered semantic identity
declared by operator name, `decision.json.params`, initialization calls, and
local operator source against the parent's `step_manifest.json`, then injects
the entry path and reusable prefix count. Keeping the same identity declares
that behavior is unchanged and authorizes reuse. If an implementation or local
dependency changes behavior, change the operator class name, decision params
(for example a semantic revision), or initialization arguments so the framework
executes the changed suffix. The pipeline must not read the manifest or parent
cache itself. Without a valid framework plan, the adapter uses the raw entry and
runs every operator. `decision.json.reason` may identify unchanged prefix
operators and the first changed operator, but must not claim a cache hit or
select an artifact path. A step whose rows carry a run-local `blobs/` reference
is recomputed in the next iteration; those references cannot be reused across
run roots.

## Operator contract

Each operator subclasses `OperatorABC`, calls `super().__init__()`, and
implements `run`. A standard operator is:

```python
from rsi.framework import FileStorage, OperatorABC

class CustomOperator(OperatorABC):
    def __init__(self, minimum_length: int = 1):
        super().__init__()
        self.minimum_length = minimum_length

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        output_rows = []
        for row in rows:
            # Preserve original fields and apply this operator's coherent responsibility.
            output_rows.append(row)
        storage.write(output_rows)
```

Keep operator code task-specific and preserve useful raw provenance fields. Put
each coherent processing step in its own operator module.

For media-producing operators, persist generated image bytes or files with
`rsi.framework.artifact_store_for(ITERATION_DIR).put_image(name, image)` and
carry the returned `ArtifactRef` dictionary in the output record. Use
`put_video_reference(name, uri, media_type=...)` for video references,
preserving the source MIME type; timestamped evidence stays as structured
records. A row may carry both image and video references. Keep
modality-specific decoding, model calls, feedback and acceptance rules in task
adapters; do not send a bare file path to a text-only LLM and treat its answer
as visual evidence. Generated media are mirrored into the final run's artifact
store when the incumbent is exported.

## Cardinality-changing operator resilience

When the task requires an exact or minimum output count, filtering and selection
must implement that cardinality as an explicit contract rather than assuming that
the preferred candidates will be sufficient:

- Separate hard-invalid records from records excluded only by soft scores,
  rankings, balancing targets, diversity preferences, or quotas. Preserve a
  deterministic reserve of hard-valid records for downstream recovery.
- A selector that underfills its target must backfill deterministically from that
  reserve. It may relax soft preferences in a declared order, but it must never
  relax the task's hard schema, content, provenance, safety, decontamination,
  media-integrity, uniqueness, or length invariants.
- Do not reach a target by duplicating rows, fabricating replacements, silently
  truncating the requested count, or reintroducing hard-invalid records.
- Validate the final distinct count before writing. If the available hard-valid
  pool is still insufficient, raise an informative error with the target,
  available count, and rejection or fallback counts so framework self-correction
  can address the responsible upstream step.
- For an LLM-backed operator, a missing, malformed, or rejected response may fall
  back to the unchanged input record only if that record independently satisfies
  every hard invariant. Otherwise expose it as a rejection for the selector's
  reserve/backfill logic instead of silently dropping it.
- Bounded classifiers, auditors, or repair operators may leave records outside
  their candidate pool with missing or `None` metadata. Downstream operators must
  type-check optional metadata before reading nested fields and treat an unprocessed
  record as a documented neutral state unless the task explicitly makes that audit
  mandatory. Do not assume `row.get(key, {})` returns a mapping when the key exists
  with a null value.

## Serving ownership and dependency injection

The Pipeline owns serving construction. An API-backed operator receives an
initialized `LLMServingABC` through its constructor and stores only that
interface:

```python
from rsi.framework import FileStorage, LLMServingABC, OperatorABC

class LLMAssistedOperator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC, domain_option: str = "default"):
        super().__init__()
        self.llm_serving = llm_serving
        self.domain_option = domain_option

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        prompts = [self.build_prompt(row) for row in rows]
        responses = self.llm_serving.generate_from_input(prompts)
        output_rows = []
        for row, response in zip(rows, responses, strict=True):
            if response is None or not str(response).strip():
                continue
            output_rows.append(self.consume_response(row, str(response)))
        storage.write(output_rows)
```

- Do not construct `PipelineLLMServing` inside an operator module or `run()`.
- Do not make an operator read API-key variables or declare URL, model, worker,
  timeout, retry, or response-contract transport settings.
- Do not add operator-level `batch_size` or manually split requests merely to
  control API concurrency. Pass the aligned logical input list to serving and let
  its configured `max_workers` own transport concurrency. A domain-specific chunk
  size is acceptable only when chunking changes the data algorithm itself.
- Do not parse raw OpenAI response dictionaries or rebuild
  `reasoning + content` in an operator. Serving returns the configured formatted
  value. Operators handle missing values and domain validation only.
- If the Pipeline includes an LLM operator, treat its prompt as one adjustable
  design part alongside the data operators: use Review/downstream attribution to
  improve task instructions, context organization, or output constraints when
  justified. Keep prompts task-level, reusable, and auditable; never hard-code a
  specific benchmark question or answer.
- Select a configured `pipeline_llm` profile according to each operator's response
  needs. Reuse one serving instance for operators with the same contract; when
  profiles differ, construct separate instances and preserve each profile's
  `serving_kwargs` unchanged.
- Do not initialize or close serving inside `run()`; one execution may reuse it
  across multiple operator calls. The pipeline cleans up each serving instance
  once its last operator has run.
- `enable_thinking` is an optional tri-state request/response policy. With a real
  boolean `true` or `false`, pass that boolean unchanged and keep it fixed for the
  lifetime of the serving instance. With `omit` (or when the profile omits the
  field), do not pass the keyword to `PipelineLLMServing`; the response contract
  is content-only. Never turn `omit` into `false` or send the string `"omit"` to
  the upstream API. Construct a separate Pipeline-owned instance when another
  operator needs a different explicit boolean contract.

Generated responses remain aligned with inputs; filter `None` or empty responses
explicitly before writing. Copy unchanged parent operator source when useful, but
leave intermediate artifact selection, validation, and reuse entirely to the
framework-owned step manifest mechanism.

## Pipeline contract

Every concrete pipeline subclasses `PipelineABC`. Instantiate storage and each
operator as members in `__init__`; `PipelineABC.compile()` discovers only
operator members. `forward()` must contain only ordered operator calls:

```python
import os
from pathlib import Path

from rsi.framework import FileStorage, OperatorABC, PipelineABC, PipelineLLMServing
from rsi.framework.evolution.execution.step_cache import FrameworkStepCache

from operators.first_operator import FirstOperator
from operators.llm_assisted_operator import LLMAssistedOperator

ITERATION_DIR = Path(__file__).resolve().parent
CACHE_DIR = str(ITERATION_DIR / "cache")
ENTRY_PATH = "<fixed raw corpus path from the task>"
OPERATOR_NAMES = ["FirstOperator", "LLMAssistedOperator"]


class Pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.framework_cache = FrameworkStepCache.from_env(
            raw_entry_path=ENTRY_PATH,
            operator_names=OPERATOR_NAMES,
        )
        self.storage = FileStorage(
            first_entry_file_name=self.framework_cache.entry_path,
            cache_path=CACHE_DIR,
            file_name_prefix="pipeline_step",
            cache_type="jsonl",
        )
        self.llm_serving = PipelineLLMServing(
            api_url=PIPELINE_API_URL,
            key_name_of_api_key=PIPELINE_API_KEY_ENV,
            model_name=PIPELINE_MODEL,
            max_workers=PIPELINE_MAX_WORKERS,
            **PIPELINE_SERVING_KWARGS,
        )
        self.first_operator = FirstOperator()
        self.llm_operator = LLMAssistedOperator(
            llm_serving=self.llm_serving,
            domain_option="task-specific",
        )

    def forward(self) -> None:
        if self.framework_cache.should_run(0):
            self.first_operator.run(storage=self.storage.step())
        if self.framework_cache.should_run(1):
            self.llm_operator.run(storage=self.storage.step())
```

`compile()` internally calls `forward()` once after wrapping operator members to
record the graph; operator bodies are not run during that graph-recording call.
Keep non-operator side effects out of `forward()`: everything there that is not
an operator call runs twice, once per phase. `storage.step()` is evaluated
during `compile()`, so each operator's step number is fixed then. Do not call
`compile()` twice on one instance.

The formal `forward()` writes ordered step JSONL files beneath the configured
cache directory, and the final step JSONL is the dataset artifact.

Use exactly this entry point:

```python
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
```

Do not execute the pipeline yourself. The framework performs artifact
validation, compile preflight, formal execution, output discovery, and
self-correction.

## decision.json

Keep `{iteration_dir}/decision.json` in sync with the implementation:

```json
{"ops": [{"name": "FirstOperator", "purpose": "...", "params": {"semantic_revision": 1}}],
 "field_flow": "raw fields -> ... -> target fields", "reason": "..."}
```

The `reason` must cite concrete evidence from the fixed raw corpus and explain
design tradeoffs. Include all generated operator classes in `ops`, in execution
order, with Python class names matching `OPERATOR_NAMES`. If a repair changes
the field flow or semantics, update this file too. Do not put secret values in
`decision.json`.

## Serving response policy

When a design uses an LLM, read
[Serving configuration and response contract](references/serving-contract.md)
before constructing the Pipeline-owned serving instance.

Critical invariants:

- Treat each dynamic profile as the complete transport and response contract.
  Do not infer behavior from a model name or change its `serving_kwargs`; select
  the profile that matches the operator's needs and copy its settings unchanged.
- Serving owns HTTP transport, concurrency, retries, timeouts, request
  construction, and input-order alignment. Operators receive one formatted
  string or None per logical input.
- `enable_thinking=true` returns
  `<think>{reasoning}</think>\n<answer>{content}</answer>`. Select this contract
  only when the task explicitly requires the reasoning field as visible artifact
  text; a derivation written in `message.content` remains content-only.
- `enable_thinking=false` returns `message.content` only and forwards the explicit
  boolean to the upstream API.
- `enable_thinking=omit` (or an absent field) also returns `message.content` only,
  but the request JSON must not contain `enable_thinking`.
- Operator prompts request only their task-specific domain result, whether that
  is a reasoning-bearing artifact or a concise content-only value. Do not
  discuss SFT construction, transport fields, hidden reasoning, or formatting
  deliberation in model prompts.
- Do not parse raw OpenAI dictionaries, concatenate response fields, alter the
  serving-provided wrapper structure, or add an operator-level HTTP retry loop.
- Validate substantive task correctness, domain-specific validity,
  non-truncation, and a non-empty required output. For reasoning tasks, require
  a coherent derivation or justification. For non-reasoning tasks, validate
  according to the task contract, target schema, and label/value constraints.

Wrapper tokens returned with `enable_thinking=true` are already part of this
task's training text. Preserve them; do not add a second set of wrappers in an
operator. With `enable_thinking=false` or `enable_thinking=omit`, consume and
validate the returned content directly rather than searching for wrapper tags.

## Task discipline

- Treat the task prompt as authoritative for paths, schemas, parent artifacts,
  feedback, bad-case counts, and serving settings.
- Optimize for resolving Review, downstream-evaluation, and bad-case evidence, not for
  minimizing code, files, or operator count.
- Choose operator boundaries by coherent responsibility and diagnostic value. Freely add,
  remove, split, merge, or replace operators when the evidence justifies it.
- Do not preserve the parent architecture merely to obtain cache reuse. Preserve an
  operator identity only when its behavior and relevant dependencies are truly unchanged.
- Serving behavior is part of an LLM operator's semantic identity even though serving is
  injected. Record the relevant serving profile or semantic revision in
  `decision.json.params` so model or response-contract changes cannot reuse an
  incompatible cached step.
- Preserve the fixed input corpus and every original field unless the task
  explicitly transforms it. The pipeline transforms the fixed input records into
  the target schema; it does not replace them with unrelated synthetic examples.
- Preserve hard schema, correctness, decontamination, source attribution, and
  output validation.
- Keep domain strategy in custom operators rather than inventing APIs.
- Do not exhaustively process the full corpus or benchmark during generation.
  Inspect bounded representative samples or lightweight statistics when they
  materially improve the design. Do not install packages, run git, fetch new
  data, or execute the pipeline yourself.
- Do not read evaluation data beyond the explicitly supplied bounded
  diagnostics, and do not ask interactive user questions.
- Produce a coherent, complete `pipeline.py` and local operators, spending the effort
  needed for a high-quality response to the supplied evidence. Inspect the task,
  supplied parent implementation, current iteration files, and bounded data samples
  needed to satisfy the artifact contract.
- Iterate on the implementation as needed to leave coherent, runnable, well-structured
  artifacts, then return control to the framework for validation.
- On self-correction, use the supplied traceback and diagnostics and modify only
  the smallest failing region.
