#!/usr/bin/env bash
# Run from the learner's WSL account. Never replaces an existing learner file.
set -Eeuo pipefail
lab_mode=${1:-run}
if [[ $# -gt 1 || ( "$lab_mode" != run && "$lab_mode" != --prepare ) ]]; then
    printf 'Usage: bash 11-triton-store.sh [--prepare]\n'
    exit 2
fi
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
lab_activation="$lab_project/experiments/activate.sh"
[[ -f "$lab_activation" ]] || { printf 'Missing activation file: %s\n' "$lab_activation"; exit 1; }
source "$lab_activation"
cd -- "$lab_project"

mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/first-triton-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "TRITON_LAB_STOPPED: preserve the first error above. Log: %s\n" "$lab_console"' ERR
lab_revision=$(git rev-parse HEAD)
[[ "$lab_revision" == '9a91cfafe754aa85daee49998176275667eb58f2' ]] || { printf 'COMMIT_MISMATCH: %s; no checkout changed.\n' "$lab_revision"; exit 1; }
if ! git diff --quiet HEAD -- python/minisgl/kernel; then
    printf 'ORIGINAL_KERNEL_CHANGED: review changes before this pinned-baseline lesson.\n'
    exit 1
fi

lab_file="$lab_project/experiments/triton_store_first.py"
if [[ -e "$lab_file" ]]; then
    printf 'EXISTING_FILE_KEPT=%s\n' "$lab_file"
else
    (set -o noclobber; cat "$lab_source/11-triton-store-template.py" > "$lab_file")
    printf 'CREATED=%s\n' "$lab_file"
fi
printf 'PROJECT_COMMIT=%s\n' "$lab_revision"
printf 'LEARNER_SOURCE_BEGIN\n'
nl -ba "$lab_file"
printf 'LEARNER_SOURCE_END\n'
if [[ "$lab_mode" == --prepare ]]; then
    printf 'PREPARE_COMPLETE: replace pass with your four read/write lines; no GPU kernel was run.\n'
else
    python "$lab_source/11-triton-store-run.py" --student "$lab_file" --commit "$lab_revision"
fi
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
