#!/bin/bash
# Drive one full evolution run against a real coding agent.
#
# This is a framework check: it proves the loop runs clean end to end. It is not
# a research setup -- use a small corpus and expect nothing from the scores.
#
#   export DATALITE_API_KEY=sk-...
#   export DATALITE_GATEWAY=https://your-gateway/v1
#   export DATALITE_CORPUS=./data/corpus.jsonl
#   ./examples/run_validation.sh codex          # or claude | opencode
#
# Everything else has a sensible default relative to the repository root.
set -uo pipefail

BACKEND="${1:-codex}"
case "$BACKEND" in
  codex|claude|opencode) ;;
  *) echo "usage: $0 <codex|claude|opencode>" >&2; exit 2 ;;
esac

for required in DATALITE_API_KEY DATALITE_GATEWAY DATALITE_CORPUS; do
  if [ -z "${!required:-}" ]; then
    echo "error: export $required before running" >&2
    exit 2
  fi
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${DATALITE_OUT:-$REPO/validation-runs}/$BACKEND"
PY="${DATALITE_PYTHON:-python}"

# One key reaches three consumers: the reviewer, the coding agent, and the
# serving that generated operators call.
export DF_API_KEY="$DATALITE_API_KEY"
export DF_AGENT_API_KEY="$DATALITE_API_KEY"
export DF_PIPELINE_API_KEY="$DATALITE_API_KEY"

export DF_API_URL="${DF_API_URL:-$DATALITE_GATEWAY/chat/completions}"
export DF_PIPELINE_API_URL="${DF_PIPELINE_API_URL:-$DATALITE_GATEWAY}"
export DF_AGENT_BASE_URL="${DF_AGENT_BASE_URL:-$DATALITE_GATEWAY}"
export DF_WORKSPACE="$OUT/workspace"
export RAW_DATA_PATH="$DATALITE_CORPUS"
export DF_AGENT_PYTHON="${DF_AGENT_PYTHON:-$(command -v "$PY")}"
export DF_DOWNSTREAM_PYTHON="${DF_DOWNSTREAM_PYTHON:-$DF_AGENT_PYTHON}"
export DF_DOWNSTREAM_RUNNER="${DF_DOWNSTREAM_RUNNER:-$REPO/examples/fake_downstream.py}"
export AGENT_BACKEND="$BACKEND"

# Per-backend agent home, so session history is auditable per run instead of
# accumulating in a shared ~/.codex.
export CODEX_HOME="$OUT/agent-home/codex"
export CLAUDE_CONFIG_DIR="$OUT/agent-home/claude"
export XDG_DATA_HOME="$OUT/agent-home/xdg-data"
export XDG_CONFIG_HOME="$OUT/agent-home/xdg-config"
mkdir -p "$CODEX_HOME" "$CLAUDE_CONFIG_DIR" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME"

rm -rf "$OUT/workspace"
mkdir -p "$OUT"

cat > "$OUT/task.json" <<EOF
{
  "schema_version": "0.1",
  "task_id": "framework-validation-$BACKEND",
  "method_id": "dataflow-evolver",
  "objective": "Transform the fixed corpus into instruction/output supervised records whose output is a step-by-step solution ending with the final answer in \\\\boxed{}.",
  "data": {"input_path": "$DATALITE_CORPUS"},
  "output": {"required_keys": ["evolution_result"], "artifact_types": ["structured_records"]},
  "metadata": {"method_config_ref": "$REPO/examples/validation.example.yaml"},
  "workspace": "$OUT/workspace",
  "provider": "$BACKEND",
  "role": "pipeline_builder"
}
EOF

echo "=== backend=$BACKEND start=$(date -Is) ==="
cd "$REPO"
timeout "${DATALITE_TIMEOUT:-7200}" "$PY" -m rsi.framework \
  --task "$OUT/task.json" --run-id "validate_${BACKEND}" < /dev/null
status=$?
echo "EXITCODE=$status"
echo "=== end=$(date -Is) ==="

if [ $status -eq 0 ]; then
  "$PY" "$REPO/examples/check_run.py" "$OUT/workspace" "validate_${BACKEND}"
fi
exit $status
