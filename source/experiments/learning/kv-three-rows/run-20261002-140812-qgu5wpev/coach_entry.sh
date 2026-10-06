#!/usr/bin/env bash
# Run in the learner's WSL. Keep the learner's existing kernel/checker.
# --width changes test fixtures, not the learner's source.
set -Eeuo pipefail
if [[ $# -ne 0 && $# -ne 2 ]]; then
    printf 'Usage: bash 13-check-kv-three-rows.sh [--width 64|128]\n'
    exit 2
fi
if [[ $# -eq 2 && ( "$1" != --width || ( "$2" != 64 && "$2" != 128 ) ) ]]; then
    printf 'Usage: bash 13-check-kv-three-rows.sh [--width 64|128]\n'
    exit 2
fi
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/kv-three-rows-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "THREE_ROWS_CHECK_STOPPED: preserve the first error. Log: %s\n" "$lab_console"' ERR
python "$lab_source/13-check-kv-three-rows.py" --project "$lab_project" --shared-results "$lab_source/logs/kv-three-rows-results" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
