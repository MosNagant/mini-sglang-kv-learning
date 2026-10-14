#!/usr/bin/env bash
# Three fresh real-model diagnostic processes; exact comparisons, no timing.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m08-model-check-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "MODEL_CHECK_STOPPED: first error and evidence preserved. Log: %s\n" "$lab_console"' ERR
python -B -u "$lab_source/41-m08-model-check.py" --project "$lab_project" --shared-results "$lab_source/logs/m08-model-check-results" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
