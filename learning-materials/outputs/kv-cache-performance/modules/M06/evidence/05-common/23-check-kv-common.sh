#!/usr/bin/env bash
# M06-05: common original/Triton correctness; existing learner files are read only.
set -Eeuo pipefail
# Arguments are validated by argparse before any GPU imports or launch.
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_project="$HOME/projects/mini-sglang-kv"
[[ -f "$lab_project/experiments/activate.sh" ]] || { printf 'PROJECT_ACTIVATION_MISSING\n'; exit 1; }
source "$lab_project/experiments/activate.sh"
cd -- "$lab_project"
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/kv-common-check-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "COMMON_CHECK_STOPPED: preserve the first error. Log: %s\n" "$lab_console"' ERR
python "$lab_source/23-check-kv-common.py" --project "$lab_project" --shared-results "$lab_source/logs/m06-common-results" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
