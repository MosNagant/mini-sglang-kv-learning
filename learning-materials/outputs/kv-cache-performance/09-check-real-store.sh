#!/usr/bin/env bash
# Run in the learner's existing WSL environment. No installs or model downloads.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_activation="$HOME/projects/mini-sglang-kv/experiments/activate.sh"
[[ -f "$lab_activation" ]] || { printf 'Missing activation file: %s\n' "$lab_activation"; exit 1; }
mkdir -p "$lab_source/logs"
lab_console=$(mktemp "$lab_source/logs/first-store-check-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$lab_console") 2>&1
trap 'printf "CHECK_STOPPED: preserve the first error above. Log: %s\n" "$lab_console"' ERR
source "$lab_activation"
python "$lab_source/09-check-real-store.py" "$@"
printf 'SHARED_CONSOLE_LOG=%s\n' "$lab_console"
