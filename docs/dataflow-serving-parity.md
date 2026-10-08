# open-dataflow 1.0.10 parity audit

The active runtime in `rsi/framework/` reimplements the operator, pipeline,
storage and serving contracts of `open-dataflow==1.0.10` so that generated
pipelines need no `open-dataflow` installation and no `dataflow` import. The
goal is **behavioral alignment, not a redefined execution model**.

The comparison target is the *effective* baseline: `open-dataflow==1.0.10` plus
the DataFlow-Evolver `compat/` fixes at commit
`ba6e489524bbf5835a06cccad555c5182e4ce4f4`, which correct SSE aggregation, the
`reasoning` field alias, the tri-state `enable_thinking` policy, and pandas
missing values leaking out of `FileStorage.read("dict")`. Those fixes are built
into this implementation rather than installed as runtime patches.

## How the claim is checked

`tests/parity/cases.py` defines 183 cases across four groups — storage
read/write and type coercion, compile/forward tracing and graph construction,
batched and streaming pipelines, and serving request/response behavior. Each
case is written once and evaluated twice, against the reference and against
`rsi.framework`, through the adapters in `tests/parity/adapters.py`.

| Suite | Environment | What it proves |
| --- | --- | --- |
| `tests/test_dataflow_parity_offline.py` | no `open-dataflow` | the active runtime matches the recorded baseline where it actually runs |
| `tests/parity/test_open_dataflow_parity.py` (marker `parity`) | `open-dataflow==1.0.10` installed | the recording still matches the real package, and both implementations agree in one process |

The second suite is what keeps the recording honest: without it, the stored
baseline could drift into an echo of this implementation. Run it in an isolated
environment, never as part of the default suite:

```bash
conda run -n dfe-open-dataflow-parity python -m pytest tests/parity -m parity
conda run -n dfe-open-dataflow-parity python tests/parity/record_baseline.py  # re-record
```

Recorded environment: open-dataflow 1.0.10, pandas 3.0.6, Python 3.11.
Both suites currently pass with all 183 cases agreeing.

## Verified equivalent

**Storage.** `operator_step` starting at `-1`; `step()` returning a shallow copy
so each operator keeps its own step number; `{file_name_prefix}_step{N}.{cache_type}`
naming; step 0 resolving to the entry file with its format taken from the file
extension; `ValueError` on read/write before `step()`; `FileNotFoundError` with
the reference's message text; `[]` and an empty frame for an empty entry name;
`IndexError` on `write([])`; `ValueError` on a dict or scalar list; ragged rows
unioned with `null` fill and row key order normalized to column order;
`NaN`/`inf` written as `null`; surrogate cleaning; `read("dict")` returning
Python `None` for missing values while `read("dataframe")` keeps pandas
semantics; `ValueError` on an unsupported output type.

**pandas type inference**, which is the reason pandas is a dependency rather
than a reimplemented rule set: an integer column containing a null reads back
as `float`; `"12"`, `" 12"`, `"1e3"`, `"+5"`, `"1_000"` and full-width digits
read back as numbers while `"0x10"` and `"ab"` stay strings; a `date` or
`timestamp` column of date-like strings becomes `Timestamp`; a bool column with
a null becomes `1.0`/`0.0`/`None`; mixed columns keep per-row types; a 30-digit
integer raises `ValueError: Value is too big!`.

**compile/forward.** `compile()` calls `forward()` exactly once, running
non-operator side effects while operator bodies stay unexecuted; `storage.step()`
is evaluated during that call, fixing step numbers before anything runs; no
cache file is written by compile; `forward` is rebound to `_compiled_forward`;
one operator called twice records and runs twice; a missing `storage` argument
raises `TypeError` at compile time; non-key keyword arguments are preserved and
forwarded; `input_*`/`output_*` arguments drive key validation with the
reference's `KeyError` text; `accumulated_keys`, `final_keys`,
`DATASET-INPUT`/`DATASET-OUTPUT` sentinel nodes and `last_modified_index_of_keys`
match; zero operators compile; `forward()` without `compile()` runs the authored
body; `resume_step` applies the same `idx - 1` offset; serving reference counting
calls `cleanup()` at the same points.

**Batched and streaming pipelines.** Batch slicing with a reset index;
append-on-later-batch writes; the `{prefix}_last_success_step.txt` resume marker
format and write points; `RUN_TIMES` derivation; `ValueError` when `resume_step`
and `resume_from_last` are both set; `BatchedFileStorage` rejecting cache types
other than `jsonl`/`csv`; `get_record_count` caching; `iter_chunks` streaming.

**Serving.** `temperature: 0.0` entering `configs`; `max_workers=10`,
`max_retries=5`, `(10.0, 120.0)` timeouts; the `Authorization`/`Content-Type`/
`User-Agent` headers; `ValueError` at construction when the key environment
variable is unset; a reused `requests.Session` with the same bounded adapter
pool; the default `system` message followed by the user message; the strict
`json_schema` `response_format`; `generate_from_conversations` and
`generate_embedding_from_input` bodies; the deprecated `timeout=` keyword;
non-200, read-timeout and malformed-body responses retried up to `max_retries`
with exponential pauses and `None` as the final value; connect timeouts and
refused connections raised as `RuntimeError` and absorbed by the thread pool;
results refilled by input id so order always matches the prompt list; the
tri-state `enable_thinking` policy across absent/`None`/`omit`/`true`/`false`
including pre-wrapped content, null content, empty choices and non-mapping
messages; the `reasoning` alias; `TypeError` for an invalid policy value at
format time; SSE aggregation with multi-choice index ordering, field
accumulation, metadata and usage retention, and forced UTF-8 decoding.

## Not equivalent

These are the known gaps. "Interface defined" is not "behavior aligned", and the
DataLite extensions below are not evidence of equivalence.

| Behavior | Status | Impact |
| --- | --- | --- |
| `hf:` / `ms:` remote dataset entries | not implemented | The reference loads a HuggingFace or ModelScope dataset when the entry starts with these prefixes. The evolution loop forbids fetching data over the network and operates on a fixed local corpus, so the entry is always a local file and such a value is simply a missing path. A task needing remote data must materialize it locally first. |
| `write()` before `step()` | raises instead of overwriting the entry file | The reference resolves step `-1 + 1` to `first_entry_file_name` and overwrites the fixed corpus in place. That breaks the immutable-input premise; it is also not a sensible call. |
| SSE adaptation scope | decoded inside this implementation's own request path | The reference's compat patches `requests.sessions.Session.send` process-wide. Generated pipelines see identical text; unrelated `requests` callers are left alone. |
| `draw_graph` | not implemented | Reference starts an HTTP server and opens a browser, which conflicts with headless runs. The graph structures it renders are all built and readable. |
| `OPERATOR_REGISTRY`, `get_operator`, `ALLOWED_PROMPTS` | not implemented | Generated pipelines instantiate local operator classes directly and never resolve one by registry name or use the built-in prompt library. |
| `DummyStorage`, `LazyFileStorage`, `MyScaleDBStorage` | not implemented | Not used by generated pipelines; the last needs a ClickHouse pool. |
| Non-API serving backends (vllm, sglang, VLM, local embedding, LiteLLM, Vertex AI) | not implemented | Only the OpenAI-compatible HTTP path is part of the generated-pipeline contract. |
| `tqdm` progress text and log formatting | differ | Presentation only; the parity suite compares behavior, not log output. |
| pandas-version sensitivity | verified on pandas 3.0.6 only | Type-inference cases could legitimately move on a different pandas major version. The offline suite skips those groups and says so when the local major version differs from the recording. |
| Real network behavior | not covered | All serving cases use injected transport. Upstream API compatibility, real retry timing, throughput and cost are unverified. |

## DataLite extensions

These exist alongside the reference contract and are **not** part of the
equivalence claim: framework-owned step caching and prefix reuse
(`step_manifest.json`), run checkpoints, provenance events, execution
observation, the artifact store reached through `artifact_store_for`, and
image/video artifact references. The reference runtime has no equivalent, so
nothing about them is measured against it.

One extension is deliberately **stricter than the DataFlow-Evolver prototype**
and that difference is observable, so it belongs on the record rather than in
the list above.

The prototype's step cache fingerprints an operator from three things:
`operator_name`, its `decision.json` `params`, and its constructor-call AST.
This implementation adds a fourth, `source`, covering the operator's own class
body, the storage/serving constructions handed to it, its own `run(...)`
arguments, and a `shared` digest spanning every module under `operators/`.

The `shared` component couples operators that are otherwise independent:
renaming, adding or removing *any* operator module changes it for all of them.
Two real validation runs showed the cost:

- **codex**: kept `PrepareGSM8KRecords` byte-identical across three iterations
  with identical params and constructor call, but renamed the two downstream
  operator files each round. `prefix_count` stayed 0 where the prototype would
  have reused 1 step.
- **claude**: kept `GsmGoldNormalizeOperator` byte-identical and merely *added*
  a fourth operator module in iteration 2. That addition alone changed `shared`,
  so step 1 was recomputed. (Step 2 was correctly invalidated on its own merit —
  that module really did change.)

The trade is intentional: the prototype can reuse a cached artifact after a
local helper changed behavior, because it never looks at source. This
implementation recomputes instead, which is never wrong, only slower. But
coding agents reorganize files routinely, so in practice reuse is lost more
often than the prototype would lose it. If cache hit rate matters more than
that safety margin, narrow `shared` to the operator's own module.
`tests/test_framework_step_cache_identity.py` pins both directions.
