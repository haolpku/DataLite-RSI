#!/usr/bin/env bash
# Restart a resumable VideoRSI production run if its stage cache stops changing.
# The driver persists every stage, so a restart only retries in-flight calls.
set -euo pipefail

OUT_ROOT="${1:?production output root required}"
shift
STALE_SECONDS="${RSI_STALE_SECONDS:-1200}"
POLL_SECONDS="${RSI_WATCH_POLL_SECONDS:-60}"
LOG="$OUT_ROOT/watchdog.log"
PID_FILE="$OUT_ROOT/production.pid"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" >> "$LOG"; }

latest_stage_mtime() {
  find "$OUT_ROOT" -path '*/videos/*/stages/*.json' -type f -printf '%T@\n' 2>/dev/null \
    | sort -nr | head -n 1
}

while true; do
  log "launching resumable production driver"
  setsid "$@" >> "$OUT_ROOT/production.log" 2>&1 &
  child=$!
  printf '%s\n' "$child" > "$PID_FILE"
  previous="$(latest_stage_mtime || true)"
  stale_since="$(date +%s)"

  while kill -0 "$child" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    current="$(latest_stage_mtime || true)"
    if [[ -n "$current" && "$current" != "$previous" ]]; then
      previous="$current"
      stale_since="$(date +%s)"
      continue
    fi
    now="$(date +%s)"
    if (( now - stale_since >= STALE_SECONDS )); then
      log "no stage progress for ${STALE_SECONDS}s; terminating process group $child"
      kill -TERM -- "-$child" 2>/dev/null || true
      sleep 20
      kill -KILL -- "-$child" 2>/dev/null || true
      break
    fi
  done
  wait "$child" 2>/dev/null || true

  summary=$(find "$OUT_ROOT" -path '*/outputs/*/run_summary.json' -type f | head -n 1 || true)
  if [[ -n "$summary" ]] && grep -q '"status": "complete"' "$summary"; then
    log "production complete"
    exit 0
  fi
  log "driver exited before completion; resuming in 15 seconds"
  sleep 15
done
