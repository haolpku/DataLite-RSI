# Getting started with the evolution framework

This guide installs the runtime and drives one complete evolution run. It is
about `rsi/framework/` — the active pipeline-evolution loop. For validating
contributions or reproducing published results, see the
[reproduction guide](reproduction.md) instead.

What the loop does: a coding agent writes a data-processing pipeline over a
**fixed** corpus, the framework compiles and executes it, an evaluator scores
the output, and a better passing candidate replaces the incumbent. Repeat.

---

## 1. Install

Python 3.10 or newer. No GPU is needed to install or to run the offline checks.

```bash
git clone https://github.com/haolpku/DataLite-RSI.git
cd DataLite-RSI
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Pick the extras you need:

```bash
pip install -e .                   # import and run generated pipelines
pip install -e ".[review]"         # + ReviewAgent and embedding review
pip install -e ".[review,claude]"  # + the Claude coding-agent backend
pip install -e ".[dev]"            # + the offline test suite
pip install -e ".[all,dev]"        # everything
```

| Extra | Adds | Needed for |
| --- | --- | --- |
| *(base)* | pandas, numpy, requests, tqdm, colorlog, PyYAML | importing the runtime, running a generated pipeline |
| `review` | openai | ReviewAgent scoring, embedding quality review |
| `claude` | claude-agent-sdk | the `claude` coding-agent backend |
| `parquet` | pyarrow | `.parquet` entry files or step caches |
| `dev` | pytest, pydantic, Pillow | the repository test suite |

Verify the install:

```bash
python -c "import rsi.framework; print('ok')"
datalite-rsi --help
pytest -q                          # needs [dev]; expect ~145 passed
```

> `pytest -q` excludes the `parity` group by default. That group compares the
> runtime against a real `open-dataflow==1.0.10` install and needs its own
> environment; see [the parity audit](dataflow-serving-parity.md).

## 2. Install a coding-agent backend

The loop needs exactly one agent to author pipelines. Pick whichever you have
credentials for — all three are supported equally, and `agent.backend` in the
config selects between them.

All three ship as **Node CLIs**, installed with `npm`, not `pip`. Only Claude
additionally needs a Python package, because the runtime talks to it through an
SDK; Codex and OpenCode are invoked as subprocesses.

| Backend | Install | Extra Python package |
| --- | --- | --- |
| `codex` | `npm install -g @openai/codex` | none |
| `opencode` | `npm install -g opencode-ai` | none |
| `claude` | `npm install -g @anthropic-ai/claude-code` | `pip install -e ".[claude]"` |

> There are PyPI packages named `codex`, `openai-codex` and `opencode-ai`, but
> none of them is what this runtime uses — it shells out to the CLI binaries
> above. Installing them will not make a backend work.

Check the binary you chose is on `PATH`:

```bash
codex --version      # or: opencode --version / claude --version
```

## 3. Prepare a fixed corpus

The input is **one** structured file that the loop never regenerates: `.jsonl`,
`.json`, `.csv` or `.parquet`. Every row is a record; the pipeline transforms
them into the target schema.

```bash
mkdir -p data
cat > data/corpus.jsonl <<'EOF'
{"question": "What is 2+3?", "answer": "2+3 = 5. #### 5"}
{"question": "What is 7*6?", "answer": "7*6 = 42. #### 42"}
EOF
```

Keep it small while you are checking the plumbing — a few dozen rows is enough
to see the loop work, and large corpora make each iteration expensive.

> Values are parsed with pandas, so a column's type can shift: an integer column
> containing a null reads back as `float`, and a column of numeric strings reads
> back as numbers. This matches the reference runtime; the authoring skill warns
> generated operators about it.

## 4. Write a configuration

Copy a template and edit it:

```bash
cp rsi/framework/configs/default.yaml my-config.yaml
```

The settings that actually matter on a first run:

```yaml
project:
  workspace_dir: ./workspace          # every artifact lands here
  input_path: ./data/corpus.jsonl     # the fixed corpus
  run_name: ${DF_RUN_NAME:run}        # a fresh name per run

task:
  target_schema:                      # the loop never infers this
    instruction: str
    output: str
  quality_criteria:
    - Require a non-empty instruction that states the source problem.
    - Require a self-contained step-by-step output.

review:
  # Optional domain rubric mode. It replaces the default four-dimension
  # ReviewAgent prompt with one criterion-by-criterion assessment.
  mode: criteria
  criteria:
    pass_rate: 0.6
  embedding_quality:
    review_score_weight: 0.25       # configurable embedding share of review_score
    metrics:
      das: {enabled: false, weight: 0.40}
      vendi: {enabled: false, weight: 0.30}
      nearest_neighbor: {enabled: false, weight: 0.30}

llm:
  api:                                # ReviewAgent's model
    api_url: ${DF_API_URL:}           # .../v1/chat/completions
    model_name: ${DF_REVIEW_MODEL:gpt-4o}
    api_key: ${DF_API_KEY:}
  pipeline_llm:
    generation:                       # the model generated operators may call
      api_url: ${DF_PIPELINE_API_URL:}
      model_name: ${DF_PIPELINE_MODEL:gpt-4o}
      api_key: ${DF_PIPELINE_API_KEY:}

agent:
  backend: ${AGENT_BACKEND:codex}     # codex | claude | opencode

loop:
  max_iterations: ${DF_MAX_ITERATIONS:3}
  execution_repairs: ${DF_EXECUTION_REPAIRS:1}
  python_exe: ${DF_AGENT_PYTHON:}     # interpreter that runs generated pipelines
```

`${VAR:default}` is expanded from the environment at load time. **Put
credentials in the environment, never in the file**, and never commit a config
holding a real key.

In `review.mode: criteria`, the reviewer returns `met`, `evidence`, and
`suggestion` for every `task.quality_criteria` entry. The framework computes
the sample-level score as the proportion of criteria marked `met`; it does not
use the correctness/relevance/difficulty/schema dimensions in this mode. When one or more
`review.embedding_quality.metrics.*.enabled` switches are true, the sample-level criteria score is
blended with the enabled embedding metrics using `review.embedding_quality.review_score_weight`
(a value in `[0, 1]` supplied by the configuration), just as in the default review mode. When all three
switches are false, the criteria proportion is the review score before deterministic safety handling.
Deterministic checks for malformed JSON, required fields, duplicates, and
benchmark contamination remain separate safety gates.

For the complete review configuration reference, including deterministic validation,
decontamination, embedding quality, and MMD options, see
[review-configuration.md](review-configuration.md).

## 5. Write a task

```bash
cat > my-task.json <<'EOF'
{
  "schema_version": "0.1",
  "task_id": "my-first-run",
  "method_id": "dataflow-evolver",
  "objective": "Turn each record into an instruction/output pair with a step-by-step solution.",
  "data": {"input_path": "./data/corpus.jsonl"},
  "output": {"required_keys": ["evolution_result"], "artifact_types": ["structured_records"]},
  "metadata": {"method_config_ref": "my-config.yaml"},
  "workspace": "./workspace",
  "provider": "codex",
  "role": "pipeline_builder"
}
EOF
```

`provider` must match `agent.backend`. `method_config_ref` names a file in
`rsi/framework/configs/`, or an absolute path.

## 6. Run

```bash
export DF_API_KEY=sk-...             # ReviewAgent
export DF_AGENT_API_KEY=sk-...       # coding agent
export DF_PIPELINE_API_KEY=sk-...    # generated-pipeline serving
export DF_AGENT_PYTHON="$(which python)"

datalite-rsi --task my-task.json --run-id my-first-run
```

Equivalent: `python -m rsi.framework --task my-task.json`. The command prints a
one-line manifest and exits non-zero if the run failed.

From Python, with your own evaluator:

```python
from rsi.framework import CandidateFeedback, run

class MyEvaluator:
    def review(self, dataset_path, task, **kwargs):
        rows = sum(1 for _ in open(dataset_path, encoding="utf-8"))
        return CandidateFeedback(score=min(1.0, rows / 100), passed=rows > 0)

manifest = run("my-task.json", resources={"candidate_evaluator": MyEvaluator()})
print(manifest["status"], manifest["acceptance"])
```

An injected `candidate_evaluator` replaces ReviewAgent entirely — which is how
image, video and mixed-modality tasks are scored, since ReviewAgent is
text-only.

## 7. Read the results

See [the framework README](../rsi/framework/README.md#where-a-run-puts-its-files)
for the full directory tree. The short version, for workspace `W` and run `R`:

| Path | What |
| --- | --- |
| `W/runs/R/run_manifest.json` | status, acceptance, fingerprints |
| `W/runs/R/records/dataflow/final_dataset.jsonl` | the winning dataset |
| `W/runs/R/native/runs/evolution/iteration_NNN/pipeline.py` | what the agent wrote |
| `…/iteration_NNN/review.jsonl` | the evaluator's evidence |
| `…/iteration_NNN/sessions/` | agent session and tool logs |
| `…/iteration_NNN/attempt_NN_*.log` | compile and execution output |

There is a checker for the invariants:

```bash
python examples/check_run.py ./workspace my-first-run
```

It reports the manifest status, accepted iterations, per-iteration prefix reuse,
the final dataset schema, and whether the diagnostic artifact stayed out of the
acceptance score. It exits non-zero on any violation.

## 8. When something fails

The loop records a `failure_stage` per candidate; start there.

| `failure_stage` | Meaning | Look at |
| --- | --- | --- |
| `artifact_validation` | `pipeline.py` broke the contract before running | the agent's `pipeline.py`; it probably imported `dataflow` or missed the `compile()` → `DF_COMPILE_ONLY` → `forward()` entry |
| `compile` | the preflight import/construction failed | `attempt_NN_compile_stderr.log` |
| `execution` | an operator raised | `attempt_NN_execution_stderr.log`, then `execution_observation.json` for the first step whose row count hits 0 |
| `output_validation` | it ran but produced no usable dataset | the step cache files under `iteration_NNN/cache/` |

A failing candidate gets up to `loop.execution_repairs` repair attempts with the
traceback fed back to the agent. Exhausting them fails that iteration without
ending the run.

Common setup problems:

| Symptom | Cause |
| --- | --- |
| `missing an optional runtime dependency: No module named 'openai'` | install the `review` extra |
| `requires the optional claude-agent-sdk package` | install the `claude` extra |
| `requires codex, claude, or opencode provider` | `provider` in the task does not match `agent.backend` |
| `temperature does not support 0` | the ReviewAgent model rejects an explicit temperature; its serving always sends one, so choose a model that accepts it |
| `运行目录已存在且非空` | reuse of a `run_name`; pick a fresh one |
| embedding errors in embedding quality review | that path speaks the vLLM chat-embedding form (`messages`), not standard OpenAI `input`; it needs a vLLM-compatible embedding server, or disable all `review.embedding_quality.metrics.*.enabled` switches |

## Next steps

- [Framework README](../rsi/framework/README.md) — architecture and the run layout
- [Parity audit](dataflow-serving-parity.md) — what matches `open-dataflow` 1.0.10 and what does not
- [Multimodal boundary](dataflow-multimodal-extension.md) — image/video tasks and the evaluator contract
- [Contribution guide](../CONTRIBUTING.md) — submitting methods, datasets or results
