# Agents' Last Exam

This benchmark registers the official
[Agents' Last Exam](https://github.com/rdi-berkeley/agents-last-exam) framework
as a DataLite-RSI LLM-track benchmark. The upstream code provisions a sandbox,
runs an agent on a task, and grades the artifacts left in that sandbox. The
source checkout used for this integration is pinned separately at upstream
commit `0b6465b13c85b5a0a017d4c88bcf979a519a5e1f`; this repository does not copy
the framework or its task data.

## Inputs and outputs

Run the upstream framework first. Its `RunWriter` output root contains one
timestamped directory per attempt with `run.json` and `eval_result.json` (and,
when configured, `trajectory.json`, `origin_log/`, and `output/`). Pass that
directory or an experiment output root to the adapter:

```bash
python benchmarks/agents-last-exam/evaluator.py .logs/ale/my_experiment \
  --output metrics.json
```

The JSON result contains one record per discovered run and aggregate
`metrics.primary_score`, `metrics.pass_rate`, `metrics.mean_cost_usd`, and
`metrics.mean_duration_s`. `primary_score` is the arithmetic mean of the
official per-task scores in `[0, 1]`; `pass_rate` is the fraction scoring `1.0`.
Cost and duration are descriptive, lower-is-better metrics and average only
records that report the corresponding value.

Malformed run records are reported in `errors` and excluded from aggregates.
A run that has a valid ALE record but failed or timed out without a score is
assigned score `0.0`, so infrastructure-visible failures cannot improve a
result by disappearing from the denominator. Attempts are retained as separate
records; select one output root per experimental condition to avoid mixing
retries or unrelated agents.

To compare an RSI intervention, run the adapter once for the baseline output
root and once for the post-RSI output root, then copy the two metric objects
into a DataLite-RSI result manifest's `metrics.baseline` and `metrics.final`.
Record the model, seeds, iteration/compute budget, code revision, the pinned
archive revision, container digest, and exact upstream command there.

## Data and limitations

The task archive is gated and pinned under
[`datasets/agents-last-exam-data-archive`](../../datasets/agents-last-exam-data-archive/).
It contains hidden references and must not be present in an agent sandbox
before the agent finishes. The upstream task set spans desktop GUI and CLI
workflows, multiple operating systems, external software, and optional judge
services. This adapter only aggregates saved scores; it does not provision
sandboxes, execute agents, verify hidden references, or make judge calls.

The upstream software is Apache-2.0 and the task data is CC-BY-4.0. Review the
upstream licenses and each task's safety, privacy, and contamination notes
before running a task outside its documented environment.
