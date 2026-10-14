#!/usr/bin/env bash
# M07 grouped efficiency: two cases, three arms, three fresh process rounds.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/kv-grouped-timing-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "GROUPED_TIMING_STOPPED: preserve the first error. Log: %s\n" "$lab_console"' ERR
python -B "$lab_source/30-compare-kv-grouped.py" --project "$lab_project" --shared-results "$lab_source/logs/m07-grouped-timing-results" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
