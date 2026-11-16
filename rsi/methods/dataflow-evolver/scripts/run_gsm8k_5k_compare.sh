#!/usr/bin/env bash
set -euo pipefail

: "${DF_API_KEY:?Export DF_API_KEY before launch.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${DFE_PYTHON:-python}"

export DF_AGENT_API_KEY="${DF_AGENT_API_KEY:-${DF_API_KEY}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${DF_API_KEY}}"
export DF_API_URL="${DF_API_URL:-http://123.129.219.111:3000/v1/chat/completions}"
export DF_AGENT_BASE_URL="${DF_AGENT_BASE_URL:-http://123.129.219.111:3000/v1}"
export CODEX_BIN="${CODEX_BIN:-codex}"
export PYTHONUNBUFFERED=1

if [[ -n "${NODE_BIN:-}" ]]; then
  NODE_DIR="$(cd "$(dirname "${NODE_BIN}")" && pwd)"
  export PATH="${NODE_DIR}:${PATH}"
elif [[ "${CODEX_BIN}" == */* ]]; then
  CODEX_DIR="$(cd "$(dirname "${CODEX_BIN}")" && pwd)"
  export PATH="${CODEX_DIR}:${PATH}"
fi

CODEX_VERSION="$("${CODEX_BIN}" --version 2>/dev/null || true)"
if [[ "${CODEX_VERSION}" != "codex-cli 0.118.0" ]]; then
  echo "Expected Codex CLI 0.118.0, got: ${CODEX_VERSION:-not found}" >&2
  exit 2
fi

TASK=$(cat <<'EOF'
Transform only the supplied GSM8K 5k raw records into exactly 5,000 SFT records.
Do not download external data, read GSM8K test, or synthesize replacement problems.
Every output record must contain the required non-empty string fields instruction, input, and output.
Set instruction to the same task instruction: Solve the math problem. Return an ordered list of steps and the final answer.
Set input to the concrete source GSM8K question.
Rewrite or normalize output into correct ordered mathematical reasoning grounded in the source answer.
Preserve the GSM8K answer convention: output must end with #### <final answer>, and that value must be mathematically equivalent to the source answer.
Preserve one output record per supplied raw record and do not silently drop or duplicate records.
EOF
)

cd "${REPO_ROOT}"
exec "${PYTHON_EXE}" -m dataflow_evolver.main \
  --config configs/gsm8k_5k_compare.yaml \
  --task "${TASK}"
