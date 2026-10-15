#!/usr/bin/env bash
set -euo pipefail
PROJECT=/home/weltlaw/projects/mini-sglang-kv
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
source experiments/activate.sh
exec python -B -u "$HERE/55-verify-graph-greedy.py" "$@"
