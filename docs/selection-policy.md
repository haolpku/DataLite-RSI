# Configurable iteration selection

Selection chooses the pipeline and dataset inherited by the next iteration,
used by periodic downstream evaluation, and exported at the end of a run.
Every candidate still receives criteria review. Its score, issues and domain
feedback still reach PipelineAgent even if selection uses another objective.

The default `criteria` rule preserves existing behavior: passed candidates
rank first, then higher `review_score` wins. Ties retain the incumbent unless
`loop.accept_equal_score` is enabled. Execution failures cannot be selected.

## Configure a policy

In the task's private YAML under `rsi/framework/configs/`, add:

```yaml
selection_policy:
  kind: python
  policy_file: /absolute/path/to/selection_policy.py
  description: Prefer the largest passing dataset
  timeout_sec: 30
  config:
    require_passed: true
```

A relative `policy_file` is resolved from that YAML file's directory. The user
can supply this file, or discuss goals, constraints, tie-breaking and missing
metrics with an Agent that writes it. The framework does not call an LLM to
invent a rule at runtime. Policy files are executable Python with the run's
privileges: use only trusted code. Do not put secrets in `config`; it is saved
in the audit. To restore the built-in rule, omit `selection_policy` or set
`kind: criteria`.

The main policy file is syntax-checked and hashed when the run is built. If it
changes during a run, selection fails rather than mixing versions. Start a new
run with a new `project.run_name` after editing it. Imported helper files are
not included in this fingerprint.

## Python interface

Provide one synchronous function:

```python
def select(context):
    candidates = context["candidates"]
    best = max(candidates, key=lambda c: (c["output_rows"] or 0, c["iteration"]),
               default=None)
    return {"iteration": best["iteration"] if best else None,
            "reason": "Prefer more output rows, then the newer iteration."}
```

The result must be JSON-serializable. `iteration` is a one-based integer in
`context["candidates"]`, or `None`; `reason` is a non-empty string. A `None`
decision leaves no incumbent, so the loop cannot generate another challenger
or export a dataset unless the final policy call selects one. For missing
metrics, return a valid previous `incumbent_iteration` or an explicit fallback.

The JSON `context` contains `schema_version` (currently 1), `event` (`review`,
`downstream` or `final`), `workspace_dir`, `run_dir`, `task_description`,
`quality_criteria`, `incumbent_iteration`, `policy`, `config`, `candidates` and
`feedback`. Candidates have iteration, pipeline/dataset paths, review score,
passed status, issues, score components, failure signals, domain feedback,
output row count when available, observation path and elapsed time. Only
successfully executed and reviewed candidates with an existing dataset enter
this list; a failed criteria review is still eligible.

Each downstream feedback item has stage, status, checkpoint and evaluated
iteration, benchmark scores and deltas. It contains no raw bad cases. Baseline
feedback has a null iteration. Compare only compatible benchmark results.
The policy receives new feedback after an already configured checkpoint; it
does not schedule extra training or evaluation. See the working
[example policy](../examples/selection_policy.py).

## Audit and errors

Each custom decision writes a directory under
`native/runs/<run_name>/selection_history/` with `request.json`,
`result.json`, `policy.py`, `policy.log` and `selection.json`. The native
`manifest.json` records policy metadata, reasons and final selection; the
outer run writes `records/dataflow/selection_history.jsonl`. The outer
`selected_iteration` is one-based; legacy `best_iteration` and candidate
records remain zero-based.

An invalid, changed, timed-out or crashing policy fails the run with the
audit location. The framework does not silently fall back to criteria.
