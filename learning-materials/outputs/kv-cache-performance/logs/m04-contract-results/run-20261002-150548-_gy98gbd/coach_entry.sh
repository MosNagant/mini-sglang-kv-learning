#!/usr/bin/env bash
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m04-contract-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "M04_STOPPED: preserve first error. Log: %s\n" "$lab_console"' ERR
python "$lab_source/15-check-kv-contract.py" --project "$lab_project" --shared-results "$lab_source/logs/m04-contract-results"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
