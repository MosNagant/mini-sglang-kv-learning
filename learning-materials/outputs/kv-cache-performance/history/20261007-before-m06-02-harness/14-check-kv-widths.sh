#!/usr/bin/env bash
# Run in the learner's WSL. Keep the learner's existing kernel/checker.
set -Eeuo pipefail
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --prepare ) ]]; then
    printf 'Usage: bash 14-check-kv-widths.sh [--prepare]\n'
    exit 2
fi
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/kv-width-check-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "WIDTH_CHECK_STOPPED: preserve the first error. Log: %s\n" "$lab_console"' ERR
python "$lab_source/14-check-kv-widths.py" --project "$lab_project" --shared-results "$lab_source/logs/kv-width-check-results" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
