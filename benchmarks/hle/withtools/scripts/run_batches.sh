#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s ENV_FILE SAMPLE_IDS CONFIG DATA_DIR OUTPUT_ROOT\n' "$0" >&2
  printf 'env: IMAGE, RUN_ID, BATCH_SIZES (default: 1,4,5,10)\n' >&2
}

if [[ $# -ne 5 ]]; then
  usage
  exit 2
fi

ENV_FILE=$1
SAMPLE_FILE=$2
CONFIG=$3
DATA_DIR=$4
OUTPUT_ROOT=$5
IMAGE=${IMAGE:-ghcr.io/haolpku/datalite-rsi-hle-with-tools:0.1.0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
BATCH_SIZES=${BATCH_SIZES:-1,4,5,10}

for path in "$ENV_FILE" "$SAMPLE_FILE" "$CONFIG"; do
  [[ -f "$path" ]] || { printf 'missing file: %s\n' "$path" >&2; exit 1; }
done
[[ -d "$DATA_DIR" ]] || { printf 'missing data directory: %s\n' "$DATA_DIR" >&2; exit 1; }
command -v docker >/dev/null || { printf 'docker is required\n' >&2; exit 1; }

mapfile -t SAMPLE_IDS < <(sed '/^[[:space:]]*$/d' "$SAMPLE_FILE")
[[ ${#SAMPLE_IDS[@]} -gt 0 ]] || { printf 'sample file is empty\n' >&2; exit 1; }

IFS=',' read -r -a SIZES <<< "$BATCH_SIZES"
total=0
for size in "${SIZES[@]}"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { printf 'invalid batch size: %s\n' "$size" >&2; exit 1; }
  total=$((total + size))
done
[[ $total -eq ${#SAMPLE_IDS[@]} ]] || {
  printf 'batch sizes sum to %d, but sample file has %d IDs\n' "$total" "${#SAMPLE_IDS[@]}" >&2
  exit 1
}

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
printf '{"run_id":"%s","image":"%s","started_at":"%s","batch_sizes":[%s]}\n' \
  "$RUN_ID" "$IMAGE" "$(date -u +%FT%TZ)" "$BATCH_SIZES" > "$RUN_DIR/run.json"

active_names=()
cleanup() {
  for name in "${active_names[@]}"; do
    docker rm -f "$name" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM EXIT

offset=0
for batch_size in "${SIZES[@]}"; do
  batch_dir="$RUN_DIR/batch_$batch_size"
  mkdir -p "$batch_dir"
  names=()
  started=$(date +%s)
  : > "$batch_dir/host_stats.jsonl"
  : > "$batch_dir/container_stats.jsonl"

  for ((slot = 0; slot < batch_size; slot++)); do
    index=$((offset + slot))
    sample_id=${SAMPLE_IDS[$index]}
    sample_dir="$batch_dir/sample_$(printf '%02d' $((index + 1)))"
    name="hle-${RUN_ID,,}-$((index + 1))"
    if docker container inspect "$name" >/dev/null 2>&1; then
      printf 'container already exists: %s\n' "$name" >&2
      exit 1
    fi
    mkdir -p "$sample_dir"
    chmod 0777 "$sample_dir"
    printf '%s\n' "$sample_id" > "$sample_dir/sample_id.txt"
    docker run -d --name "$name" \
      --env-file "$ENV_FILE" \
      -v "$CONFIG:/config/eval.json:ro" \
      -v "$DATA_DIR:/data:ro" \
      -v "$sample_dir:/output" \
      "$IMAGE" --config /config/eval.json --sample-id "$sample_id" \
      > "$sample_dir/container_id.txt"
    names+=("$name")
    active_names+=("$name")
  done

  while :; do
    running=0
    for name in "${names[@]}"; do
      [[ "$(docker inspect -f '{{.State.Running}}' "$name")" == true ]] && running=$((running + 1))
    done
    now=$(date -u +%FT%TZ)
    printf '{"timestamp":"%s","running":%d,"loadavg":"%s"}\n' \
      "$now" "$running" "$(tr -d '\n' < /proc/loadavg)" >> "$batch_dir/host_stats.jsonl"
    docker stats --no-stream --format \
      '{"timestamp":"'"$now"'","name":"{{.Name}}","cpu":"{{.CPUPerc}}","memory":"{{.MemUsage}}","memory_percent":"{{.MemPerc}}","net_io":"{{.NetIO}}","block_io":"{{.BlockIO}}","pids":"{{.PIDs}}"}' \
      "${names[@]}" >> "$batch_dir/container_stats.jsonl" 2>/dev/null || true
    [[ $running -eq 0 ]] && break
    sleep 2
  done

  failed=0
  for ((slot = 0; slot < batch_size; slot++)); do
    index=$((offset + slot))
    sample_dir="$batch_dir/sample_$(printf '%02d' $((index + 1)))"
    name=${names[$slot]}
    exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$name")
    printf '%s\n' "$exit_code" > "$sample_dir/exit_code.txt"
    [[ "$exit_code" -eq 0 ]] || failed=1
    docker logs "$name" > "$sample_dir/container.log" 2>&1 || true
    docker rm "$name" >/dev/null
  done
  active_names=()

  finished=$(date +%s)
  printf '{"concurrency":%d,"samples":%d,"elapsed_seconds":%d,"finished_at":"%s"}\n' \
    "$batch_size" "$batch_size" "$((finished - started))" "$(date -u +%FT%TZ)" > "$batch_dir/summary.json"
  [[ $failed -eq 0 ]] || { printf 'batch concurrency=%d had failed containers\n' "$batch_size" >&2; exit 1; }
  offset=$((offset + batch_size))
done

printf '%s\n' "$(date -u +%FT%TZ)" > "$RUN_DIR/finished_at.txt"
trap - INT TERM EXIT
printf '%s\n' "$RUN_DIR"
