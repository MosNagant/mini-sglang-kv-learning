#!/usr/bin/env bash
# Capture real decode fixtures using original model math; no timing or installs.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/m08-gate-up-fixture-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "GATE_UP_CAPTURE_STOPPED: error and evidence preserved. Log: %s\n" "$lab_console"' ERR
python -B -u "$lab_source/48-m08-capture-gate-up.py" --project "$lab_project" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
