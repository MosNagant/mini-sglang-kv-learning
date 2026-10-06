#!/usr/bin/env bash
# Coach-owned M05 evidence collection. Preserve all earlier runs.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m05-closeout-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "M05_CLOSEOUT_STOPPED: preserve the first error. Log: %s\n" "$lab_console"' ERR
python "$lab_source/20-m05-closeout.py" --shared-results "$lab_source/logs/m05-closeout-results"
python "$lab_source/14-check-kv-widths.py" --project "$lab_project" --shared-results "$lab_source/logs/kv-width-check-results" --prepare
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
