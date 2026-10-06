#!/usr/bin/env bash
# Run inside the learner's existing WSL account; preserves an existing lab file.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
lab_activation="$lab_project/experiments/activate.sh"
[[ -f "$lab_activation" ]] || { printf 'Missing activation file: %s\n' "$lab_activation"; exit 1; }
source "$lab_activation"
cd -- "$lab_project"

mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/complete-kv-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "LAB_STOPPED: preserve the first error above. Log: %s\n" "$lab_console"' ERR

lab_revision=$(git rev-parse HEAD)
[[ "$lab_revision" == '9a91cfafe754aa85daee49998176275667eb58f2' ]] || { printf 'COMMIT_MISMATCH: %s; no checkout changed.\n' "$lab_revision"; exit 1; }
if ! git diff --quiet HEAD -- python/minisgl/kernel; then
    printf 'KERNEL_SOURCE_CHANGED: review edits before using this original-kernel lesson.\n'
    exit 1
fi

lab_file="$lab_project/experiments/kv_write_lab.py"
if [[ -e "$lab_file" ]]; then
    printf 'EXISTING_FILE_KEPT=%s\n' "$lab_file"
else
    (set -o noclobber; cat "$lab_source/10-kv-write-lab.py" > "$lab_file")
    printf 'CREATED=%s\n' "$lab_file"
fi
printf 'PROJECT_COMMIT=%s\n' "$lab_revision"
printf 'SCOPE=complete small learning experiment; not a benchmark or a full correctness suite\n'
printf 'SAVED_SOURCE_BEGIN\n'
nl -ba "$lab_file"
printf 'SAVED_SOURCE_END\n'
python "$lab_file"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
