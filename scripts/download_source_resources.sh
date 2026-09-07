#!/usr/bin/env bash
# Resumably download the gated Agents' Last Exam archive and multipart QEMU disks.
# The HF resolve URL is refreshed after every stalled curl invocation.
set -uo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/data/xuebinrui/code/ale/source}
HF_TOKEN=${HF_TOKEN:?set HF_TOKEN in the environment}
QEMU_ROOT="$SOURCE_ROOT/qemu"
PARTS_ROOT="$QEMU_ROOT/parts"
MANIFEST_ROOT="$QEMU_ROOT/manifests"
ARCHIVE_ROOT="$SOURCE_ROOT/task-data-archive"
mkdir -p "$PARTS_ROOT" "$MANIFEST_ROOT" "$ARCHIVE_ROOT"

download_object() {
  local repo=$1 revision=$2 remote=$3 dest=$4 expected_size=$5 expected_sha=$6
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" && "$(stat -c %s "$dest")" -eq "$expected_size" ]]; then
    [[ "$(sha256sum "$dest" | awk '{print $1}')" == "$expected_sha" ]] && return
    rm -f "$dest"
  fi
  rm -rf "$dest.ranges"
  mkdir -p "$dest.ranges"
  local chunk_size=250000000 start end range_file header_file
  local -a workers=()
  local max_workers=${RANGE_WORKERS:-16}
  for start in $(seq 0 "$chunk_size" $((expected_size - 1))); do
    end=$((start + chunk_size - 1))
    (( end >= expected_size )) && end=$((expected_size - 1))
    range_file="$dest.ranges/$start"
    header_file="$range_file.headers"
    (
      while :; do
        if [[ -f "$range_file" && "$(stat -c %s "$range_file")" -eq $((end - start + 1)) ]] \
          && grep -Eiq "^content-range: bytes[[:space:]]+$start-$end/" "$header_file"; then
          break
        fi
        rm -f "$range_file" "$header_file"
        env -u all_proxy -u ALL_PROXY timeout 240 curl -fL --range "$start-$end" \
          --connect-timeout 20 --max-time 210 --speed-time 45 --speed-limit 65536 \
          --retry 1 --retry-delay 2 -D "$header_file" \
          -H "Authorization: Bearer $HF_TOKEN" -o "$range_file" \
          "https://huggingface.co/datasets/$repo/resolve/$revision/$remote" || true
      done
    ) &
    workers+=("$!")
    # Keep at most four independent ranges in flight without relying on job control.
    if (( ${#workers[@]} >= max_workers )); then
      wait "${workers[0]}" || true
      workers=("${workers[@]:1}")
    fi
  done
  for worker in "${workers[@]}"; do wait "$worker" || true; done
  : > "$dest.tmp"
  for start in $(seq 0 "$chunk_size" $((expected_size - 1))); do
    cat "$dest.ranges/$start" >> "$dest.tmp"
  done
  [[ "$(stat -c %s "$dest.tmp")" -eq "$expected_size" ]]
  [[ "$(sha256sum "$dest.tmp" | awk '{print $1}')" == "$expected_sha" ]]
  mv "$dest.tmp" "$dest"
  rm -rf "$dest.ranges"
}

download_archive() {
  local manifest="$SOURCE_ROOT/.metadata/agents-last-exam-data-archive.json"
  local size sha
  size=$(jq -r '.[] | select(.path == "ale-tasks-data.tar.gz") | .size' "$manifest")
  sha=$(jq -r '.[] | select(.path == "ale-tasks-data.tar.gz") | .lfs.sha256' "$manifest")
  download_object agents-last-exam/agents-last-exam-data-archive \
    95e419efaae5fd3874b709f7ef63c357aa29379c ale-tasks-data.tar.gz \
    "$ARCHIVE_ROOT/ale-tasks-data.tar.gz" "$size" "$sha"
  gzip -t "$ARCHIVE_ROOT/ale-tasks-data.tar.gz"
  rm -f "$ARCHIVE_ROOT/ale-tasks-data.tar.gz.aria2"
}

download_vm() {
  local image=$1
  local manifest="$MANIFEST_ROOT/$image.qcow2.manifest.json"
  local partial="$QEMU_ROOT/$image.qcow2.partial"
  local final="$QEMU_ROOT/$image.qcow2"
  local image_parts="$PARTS_ROOT/$image"
  local repo=agents-last-exam/ale-images-qcow2
  local revision=31374caa105f15c9cf3c20fe6abcf9e40ec1a636
  local total expected
  total=$(jq -r '.size' "$manifest")
  expected=$(jq -r '.sha256' "$manifest")
  mkdir -p "$image_parts"
  [[ -f "$partial" ]] || : > "$partial"
  local offset=0
  for index in $(jq -r '.parts | to_entries[] | .key' "$manifest"); do
    local remote name size sha file current
    remote=$(jq -r ".parts[$index].filename" "$manifest")
    name=${remote##*/}
    size=$(jq -r ".parts[$index].size" "$manifest")
    sha=$(jq -r ".parts[$index].sha256" "$manifest")
    file="$image_parts/$name"
    current=$(stat -c %s "$partial")
    if [[ "$current" -ge "$((offset + size))" ]]; then
      offset=$((offset + size))
      continue
    fi
    [[ "$current" -eq "$offset" ]] || {
      echo "unexpected partial size for $image: $current (expected $offset)" >&2
      return 1
    }
    download_object "$repo" "$revision" "$remote" "$file" "$size" "$sha"
    cat "$file" >> "$partial"
    rm -f "$file" "$file.aria2"
    offset=$((offset + size))
    echo "appended $image $name: $(stat -c %s "$partial")/$total" >&2
  done
  [[ "$(stat -c %s "$partial")" -eq "$total" ]]
  [[ "$(sha256sum "$partial" | awk '{print $1}')" == "$expected" ]]
  mv "$partial" "$final"
}

case "${1:-archive}" in
  archive) download_archive ;;
  ubuntu) download_vm ale-ubuntu22 ;;
  windows) download_vm ale-win10 ;;
  *) echo "usage: $0 archive|ubuntu|windows" >&2; exit 2 ;;
esac
