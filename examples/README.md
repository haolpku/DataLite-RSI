# Examples

Runnable examples for the evolution framework. Start with
[docs/getting-started.md](../docs/getting-started.md) for the walkthrough; these
are the files it refers to.

| File | What it shows |
| --- | --- |
| `validation.example.yaml` | A complete config with every section filled in |
| `run_validation.sh` | Driving one full run, then checking it |
| `check_run.py` | Asserting the loop's invariants from a real run's artifacts |
| `fake_downstream.py` | Implementing the `downstream_eval` contract |

Everything here is a **framework check**: a tiny corpus, and fabricated
downstream numbers where no GPU is available. Nothing produced by these files
supports a quality claim.

## A complete configuration

`rsi/framework/configs/default.yaml` is the minimal starting point.
`validation.example.yaml` is the opposite: every section filled in, including
the `downstream_eval` and `pipeline_diagnostic` blocks that `default.yaml`
omits. Copy it when you want to see what a fully-specified run looks like.

Every path and endpoint is an environment placeholder with no default, so the
file is portable and holds no credentials.

## Drive one run

```bash
export DATALITE_API_KEY=sk-...
export DATALITE_GATEWAY=https://your-gateway/v1
export DATALITE_CORPUS=./data/corpus.jsonl

./examples/run_validation.sh codex      # or claude | opencode
```

Output lands in `validation-runs/<backend>/` (override with `DATALITE_OUT`).
The script points each backend's home directory at the run, so session
transcripts stay with their run instead of accumulating in a shared `~/.codex`.

Run backends one at a time: three concurrent agents against one gateway invites
rate limits that look like framework failures.

## Check a run

```bash
python examples/check_run.py ./validation-runs/codex/workspace validate_codex
```

It reads the run's own artifacts and reports manifest status, accepted
iterations, per-iteration prefix reuse, the final dataset schema, and whether
the diagnostic artifact stayed out of the acceptance score — then exits
non-zero on any violation.

The test suite covers these invariants for stubbed runs; this covers them for a
real one.

## Implementing downstream feedback

`downstream_eval.command` names an external program: the framework writes a
request JSON, runs your command with `--request <path> --result <path>`, and
validates what you write back. It never imports your training code.

`fake_downstream.py` is a working implementation of that contract. It stands in
for real SFT plus benchmark evaluation when no GPU is free, and satisfies the
**real** result schema — so request serialization, subprocess invocation,
result validation, bad-case consistency rules and attribution input are all
genuinely exercised. Only the numbers are invented, and bad cases are drawn
from the actual candidate dataset so attribution has real text to reason about.

Its `model` field is literally `FABRICATED-no-real-training`. Never present
output derived from it as a benchmark result. For a real experiment, replace it
with your own runner implementing the same contract — reading it is the fastest
way to see what that contract requires.

## Environment notes

- **A coding-agent backend must be installed** (`codex`, `claude`, or
  `opencode`); see the getting-started guide.
- **Embedding dataset-quality review is off in this config.** That path speaks
  the vLLM chat-embedding form (`messages`), not standard OpenAI `input`, so it
  needs a vLLM-compatible embedding server. Enable it only when
  `DF_DAS_EMBEDDING_URL` points at one.
- **The ReviewAgent model must accept an explicit `temperature`.** Its serving
  always sends one, so a model that rejects it will fail every review call.
