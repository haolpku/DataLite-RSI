---
name: dataflow-evolver-pipeline
description: Design and implement a high-quality custom DataFlow-Evolver pipeline using open-dataflow 1.0.10.
---

# DataFlow-Evolver Pipeline Contract

Use this skill when you generate or self-correct a DataFlow-Evolver pipeline. It is the
complete API contract for this task. Use the dynamic task context, Review feedback,
downstream results, and bad cases to make the strongest justified pipeline design.

## Imports and runtime

- Runtime distribution: `open-dataflow==1.0.10`; Python imports use `dataflow.*`.
- Import `OperatorABC` and `LLMServingABC` from `dataflow.core`,
  `PipelineABC` from `dataflow.pipeline`, and `FileStorage` from
  `dataflow.utils.storage`.
- Pipeline modules import `APILLMServing_request` from
  `dataflow.serving.api_llm_serving_request`; operator modules depend only on
  an injected `LLMServingABC`.
- Generated operators and pipelines are local custom code. Available serving
  profiles are runtime capabilities, not prescribed operator roles. You decide whether, where, and how to use them from the task and feedback.
  When used, construct serving in the Pipeline from the dynamic configuration.

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
- `storage.read("dict")` returns the current step as `list[dict]`.
- `storage.read()` returns a pandas DataFrame.
- `storage.write(rows)` writes the next step and returns its file path.
- Do not pass logical input/output keys to `step`, `read`, or `write`.

## Framework-owned prefix reuse

Step filenames are execution-relative, not operator-relative. If a prefix operator
is skipped, every downstream `pipeline_step_stepN.jsonl` number moves forward.
Never infer an artifact producer from its filename, inspect parent cache files to
choose one, or implement custom cache validation. Parseable JSONL is not proof of
producer identity.

Every pipeline must mechanically consume the framework decision:

```python
from dataflow_evolver.engine.step_cache import FrameworkStepCache

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

The framework validates raw-input identity and the ordered semantic identity declared
by operator name, `decision.json.params`, and initialization calls against the parent's
`step_manifest.json`, then injects the entry path and reusable prefix count. Source bytes
and source-file hashes are intentionally not compared. Keeping the same identity declares
that behavior is unchanged and authorizes reuse. If an implementation or local dependency
changes behavior, change the operator class name, decision params (for example a semantic
revision), or initialization arguments so the framework executes the changed suffix. The pipeline must not read the manifest or parent cache itself. Without a
valid framework plan, the adapter uses the raw entry and runs every operator.
`decision.json.reason` may identify unchanged prefix operators and the first
changed operator, but must not claim a cache hit or select an artifact path.

## Operator contract

Each operator subclasses `OperatorABC`, calls `super().__init__()`, and implements
`run`. A standard operator is:

```python
class CustomOperator(OperatorABC):
    def __init__(self, ...):
        super().__init__()

    def run(self, storage: FileStorage) -> None:
        rows = storage.read("dict")
        output_rows = []
        for row in rows:
            # Preserve original fields and apply this operator's coherent responsibility.
            output_rows.append(row)
        storage.write(output_rows)
```

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

The Pipeline owns serving construction. An API-backed operator receives an initialized
`LLMServingABC` through its constructor and stores only that interface:

```python
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

- Do not construct `APILLMServing_request` inside an operator module or `run()`.
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
  across multiple operator calls.
- `enable_thinking` is an optional tri-state request/response policy. With a real
  boolean `true` or `false`, pass that boolean unchanged and keep it fixed for the
  lifetime of the serving instance. With `omit` (or when the profile omits the
  field), do not pass the keyword to `APILLMServing_request`; the response contract
  is content-only. Never turn `omit` into `false` or send the string `"omit"` to
  the upstream API. Construct a separate Pipeline-owned instance when another
  operator needs a different explicit boolean contract.

Generated responses remain aligned with inputs; filter `None` or empty responses
explicitly before writing. Copy unchanged parent operator source when useful, but
leave intermediate artifact selection, validation, and reuse entirely to the
framework-owned step manifest mechanism.

## Pipeline contract

Every concrete pipeline subclasses `PipelineABC`. Instantiate storage and each
operator as members in `__init__`; `PipelineABC.compile()` discovers only operator
members. `forward()` must contain only ordered operator calls:

```python
class Pipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(...)
        self.llm_serving = APILLMServing_request(
            api_url=PIPELINE_API_URL,
            key_name_of_api_key=PIPELINE_API_KEY_ENV,
            model_name=PIPELINE_MODEL,
            max_workers=PIPELINE_MAX_WORKERS,
            **PIPELINE_SERVING_KWARGS,
        )
        self.first_operator = FirstOperator(...)
        self.llm_operator = LLMAssistedOperator(
            llm_serving=self.llm_serving,
            domain_option="task-specific",
        )

    def forward(self) -> None:
        self.first_operator.run(storage=self.storage.step())
        self.llm_operator.run(storage=self.storage.step())
```

`compile()` internally calls `forward()` once after wrapping operator members to
record the graph; operator bodies are not run during that graph-recording call.
Keep non-operator side effects out of `forward()`. The formal `forward()` writes
ordered step JSONL files beneath the configured cache directory, and the final
step JSONL is the dataset artifact.

Use exactly this entry point:

```python
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.compile()
    if os.getenv("DF_COMPILE_ONLY") == "1":
        raise SystemExit(0)
    pipeline.forward()
```

Do not execute the pipeline yourself. Evolver performs artifact validation,
compile, formal execution, output discovery, and self-correction.

## Serving response policy

When a design uses an LLM, read
[Serving configuration and response contract](references/serving-contract.md)
before constructing the Pipeline-owned serving instance.

Critical invariants:

- Treat each dynamic profile as the complete transport and response contract.
  Do not infer behavior from a model name or change its `serving_kwargs`; select
  the profile that matches the operator's needs and copy its settings unchanged.
- DataFlow owns HTTP transport, concurrency, retries, timeouts, request construction, and input-order alignment. Operators receive one formatted string or None per logical input.
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
  DataFlow-provided wrapper structure, or add an operator-level HTTP retry loop.
- Validate substantive task correctness, domain-specific validity, non-truncation, and a non-empty required output.
  For reasoning tasks, require a coherent derivation or justification. For non-reasoning tasks, validate
  according to the task contract, target schema, and label/value constraints.
Wrapper tokens returned with `enable_thinking=true` are already part of this
task's training text. Preserve them; do not add a second set of wrappers in an
operator. With `enable_thinking=false` or `enable_thinking=omit`, consume and validate the
returned content directly rather than searching for wrapper tags.

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
  incompatible cached step. Do not put secret values in `decision.json`.
- Preserve the fixed input corpus and every original field unless the task
  explicitly transforms it.
- Keep domain strategy in custom operators rather than inventing APIs.
- Do not exhaustively process the full corpus or benchmark during generation.
  Inspect bounded representative samples or lightweight statistics when they
  materially improve the design. Do not install packages, run git, or execute the
  pipeline yourself.
- Produce a coherent, complete `pipeline.py` and local operators, spending the effort
  needed for a high-quality response to the supplied evidence. Inspect the task,
  supplied parent implementation, current iteration files, and bounded data samples
  needed to satisfy the artifact contract.
- Iterate on the implementation as needed to leave coherent, runnable, well-structured
  artifacts, then return control to Evolver for framework validation.
- On self-correction, use the supplied traceback and diagnostics and modify only
  the smallest failing region.
