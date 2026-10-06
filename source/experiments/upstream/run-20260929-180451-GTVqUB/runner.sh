#!/usr/bin/env bash
# Runs the unmodified upstream test. Does not implement a kernel or timing loop.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_repo="$HOME/projects/mini-sglang-kv"
lab_sha=9a91cfafe754aa85daee49998176275667eb58f2
lab_stamp=$(date +%Y%m%d-%H%M%S)
[[ -f "$lab_repo/experiments/activate.sh" ]] || {
    printf 'Missing project activation script. Report this message.\n' >&2
    exit 1
}
mkdir -p "$lab_repo/experiments/upstream" "$lab_source/logs"
lab_run=$(mktemp -d "$lab_repo/experiments/upstream/run-$lab_stamp-XXXXXX")
lab_shared="$lab_source/logs/upstream-$(basename "$lab_run").log"
exec > >(tee -a "$lab_shared" "$lab_run/run.log") 2>&1
lab_finish() {
    lab_rc=$?
    printf '%s\n' "$lab_rc" > "$lab_run/exit-code.txt"
    if [[ $lab_rc -ne 0 ]]; then
        printf '\nUPSTREAM_TEST_FAILED: exit=%s. Keep the FIRST error above.\n' "$lab_rc"
    fi
    printf 'Run directory: %s\nShared log: %s\n' "$lab_run" "$lab_shared"
    exit "$lab_rc"
}
trap lab_finish EXIT
source "$lab_repo/experiments/activate.sh"
[[ $(git rev-parse HEAD) == "$lab_sha" ]] || {
    printf 'Unexpected commit. Existing checkout was preserved.\n' >&2
    exit 1
}
git diff --exit-code
git diff --cached --exit-code
printf 'Commit: %s\nRandom seed: 0\n' "$(git rev-parse HEAD)"
printf 'Scope: upstream correctness and exploratory timing; NOT the project performance baseline.\n'
printf 'Existing compiler/JIT caches are retained; their initial contents are not controlled.\n'
date -Is > "$lab_run/start-time.txt"
git rev-parse HEAD > "$lab_run/commit.txt"
git status --short > "$lab_run/git-status.txt"
git archive --format=tar HEAD python/minisgl/kernel python/minisgl/benchmark/perf.py \
    tests/kernel/test_store.py pyproject.toml > "$lab_run/upstream-sources.tar"
cp "$lab_repo/experiments/environment/requirements.lock" "$lab_run/requirements.lock"
sha256sum "$lab_run/requirements.lock" > "$lab_run/lock-sha256.txt"
"$lab_repo/.lab-tools/uv" pip freeze --python "$lab_repo/.venv/bin/python" > "$lab_run/installed-packages.txt"
cp "$lab_source/03-run-upstream-test.sh" "$lab_run/runner.sh"
cp "$lab_source/PROGRESS.md" "$lab_run/coach-progress.md"
if [[ -f "$lab_repo/experiments/PROGRESS.md" ]]; then
    cp "$lab_repo/experiments/PROGRESS.md" "$lab_run/previous-progress.md"
fi
cp "$lab_source/PROGRESS.md" "$lab_repo/experiments/PROGRESS.md"
nvcc --version > "$lab_run/nvcc.txt"
uname -a > "$lab_run/kernel.txt"
if ! /usr/lib/wsl/lib/nvidia-smi > "$lab_run/gpu-before.txt" 2>&1; then
    printf 'GPU status snapshot unavailable; the test will check GPU access.\n'
fi
cat "$lab_run/gpu-before.txt"

# Keep upstream test code and its timing implementation unchanged.
# Disable unrelated third-party pytest plugins and repository coverage defaults.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONHASHSEED=0
export LOG_LEVEL=INFO
export LAB_RUN_DIR="$lab_run"
python -u - <<'PY'
import json, os, pathlib, random, sys
import numpy as np
import pytest
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
report = {
    'python': sys.version, 'python_executable': sys.executable,
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'seed': seed,
    'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING'),
    'scope': 'unmodified upstream test; exploratory timing only',
    'jit_cache_state': 'existing caches retained; initial contents uncontrolled',
}
print(json.dumps(report, indent=2), flush=True)
(pathlib.Path(os.environ['LAB_RUN_DIR']) / 'run-environment.json').write_text(
    json.dumps(report, indent=2) + '\n'
)
sys.exit(pytest.main([
    '-o', 'addopts=', '-q', '-s',
    'tests/kernel/test_store.py::test_store_cache',
]))
PY
if ! /usr/lib/wsl/lib/nvidia-smi > "$lab_run/gpu-after.txt" 2>&1; then
    printf 'Post-test GPU status snapshot unavailable.\n'
fi
printf '\nUPSTREAM_TEST_PASS: the original store test completed successfully.\n'
printf 'Raw timings are exploratory; correctness coverage is limited to the upstream assertions.\n'
date -Is > "$lab_run/end-time.txt"
