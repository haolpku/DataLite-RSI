# Pipeline LLM serving contract

When an operator needs model access, the generated Pipeline constructs
`rsi.framework.PipelineLLMServing` and injects it into the operator. The shared
`LLMServingABC` interface defines `generate_from_input(prompts, ...)`: it returns
one response per prompt, in input order, with `str` or `None` for each result.
The concrete serving class also supports conversations and embeddings, described
below. Use this document and `rsi.framework` imports for generated pipelines.

## Request fields

| Field | Meaning |
|---|---|
| api_url | Full OpenAI-compatible chat-completions endpoint. |
| key_name_of_api_key | Environment-variable name containing the API key, not the key value. |
| model_name | Request-body model; it does not define response parsing. |
| temperature | Sampling temperature. Sent as `0.0` when the caller omits it. |
| enable_thinking | Optional tri-state policy: `true`/`false` are forwarded booleans; `omit` (or absence) omits the request field and uses content-only parsing. |
| extra keyword arguments | The remaining dynamic serving profile, for example `top_p`, `max_tokens`, `stream`, and `stream_options`. |
| max_workers | HTTP concurrency and connection-pool size. |
| max_retries | Total attempts per logical input, including the first. Defaults to 5. |
| connect_timeout / read_timeout | Connection and response-read timeouts. Default to 10s and 120s. |

Copy the dynamic serving profile to `PipelineLLMServing` and keep all transport
settings unchanged. For `enable_thinking=true` or `false`, pass the real boolean
unchanged. For `enable_thinking=omit` (or an absent field), omit that keyword
from the constructor so it is absent from the upstream JSON; do not send the
string `"omit"`. Never infer a value from the model name.

The constructor reads the API key from the named environment variable and raises
`ValueError` when it is unset, so a missing credential fails at construction
rather than as a batch of null responses. `generate_from_input` sends a default
`system` message (`"You are a helpful assistant"`) followed by the user prompt;
pass `system_prompt` to replace it.

Besides the base interface method `generate_from_input`, the concrete
`PipelineLLMServing` class offers `generate_from_conversations` for pre-built
message lists and `generate_embedding_from_input` for embeddings. Its
`generate_from_input` also accepts a per-call `json_schema` argument that sets
a strict `response_format`.

## Response contracts

With `enable_thinking=true`, serving returns:

```text
<think>{reasoning}</think>
<answer>{content}</answer>
```

Use this contract only when the task explicitly requires the reasoning field as
visible artifact text. A complete derivation in `message.content` remains
content-only. Preserve the returned string and validate it against the task.

With `enable_thinking=false`, serving returns `message.content` only and the
explicit boolean is forwarded upstream. With `enable_thinking=omit` (or when the
field is absent), serving also returns `message.content` only, but the upstream
request must not contain `enable_thinking`. Judges, scorers, classifiers,
routers, extractors, and concise normalizers are examples, not an exhaustive
operator list. Parse and validate the content according to the operator's
declared domain output.

`message.reasoning` is accepted as an alias for `message.reasoning_content`.
Buffered streaming SSE is aggregated into an ordinary chat-completion response
before formatting, so `stream: true` in a profile needs no operator-side
handling. A malformed or empty response becomes `None` rather than an exception.

Operators receive one aligned `str` or `None` per input. They must not inspect
raw OpenAI dictionaries, parse streaming events, concatenate reasoning and
content fields, add another wrapper layer, add transport retries, or construct a
second HTTP client. An operator may parse its returned domain content, such as a
score, but must not parse raw transport fields or streaming events.

## Failure behavior

A non-200 response, a read timeout, or a malformed body yields no result for
that input and is retried up to `max_retries` times with an exponential pause
between attempts; the final value is `None`. A connect timeout or a refused
connection is treated as "the server is unreachable" and raised, so a
misconfigured endpoint fails loudly instead of silently producing a dataset of
nulls. Results are refilled by input id, so the returned list always aligns with
the prompt list regardless of completion order.

## Prompt and validation ownership

Prompts request only the task-specific domain result appropriate to the chosen
response category. Reference material may be supplied as a correctness anchor,
but the visible response must not discuss reference matching. Do not mention
response field names, transport serialization, SFT construction, or formatting
deliberation unless the task itself genuinely requires that visible text.

For a reasoning-retaining call, validate the returned formatted text against the
task's substantive requirements and do not strip the `<think>/<answer>`
structure. For a content-only call, validate the returned content against that
operator's domain contract and reject malformed values rather than guessing.

An operator validates the resulting domain content. A missing, malformed, or
rejected response may leave the original record unchanged only when the original
independently satisfies all hard task invariants. Prompt content must be
task-level, reusable, and auditable; no benchmark-specific hardcoding.
