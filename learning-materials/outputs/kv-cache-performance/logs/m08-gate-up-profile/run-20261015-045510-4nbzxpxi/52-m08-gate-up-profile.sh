#!/usr/bin/env bash
# One bounded NCU campaign; never installs tools or changes driver permissions.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_log=$(mktemp "$lab_source/logs/m08-gate-up-profile-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_log") 2>&1
python -B -u "$lab_source/52-m08-gate-up-profile.py" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_log"
