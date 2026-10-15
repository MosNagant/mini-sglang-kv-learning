#!/usr/bin/env bash
# Explicit selected-candidate research integration, gated by saved numerical checks.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m08-gate-up-model-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "GATE_UP_MODEL_STOPPED: error and evidence preserved. Log: %s\n" "$lab_console"' ERR
python -B -u "$lab_source/51-m08-gate-up-model.py" --project "$lab_project" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
