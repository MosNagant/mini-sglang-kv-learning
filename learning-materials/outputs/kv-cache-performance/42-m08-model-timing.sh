#!/usr/bin/env bash
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m08-model-timing-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "MODEL_TIMING_STOPPED: first error preserved. Log: %s\n" "$lab_console"' ERR
python -B -u "$lab_source/42-m08-model-timing.py" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
