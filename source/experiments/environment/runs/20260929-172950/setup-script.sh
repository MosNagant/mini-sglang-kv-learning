#!/usr/bin/env bash
# Environment scaffolding only. Does not implement or time a CUDA kernel.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_stamp=$(date +%Y%m%d-%H%M%S)
lab_sha=9a91cfafe754aa85daee49998176275667eb58f2
lab_url=https://github.com/sgl-project/mini-sglang.git
lab_repo="$HOME/projects/mini-sglang-kv"
mkdir -p "$lab_source/logs"
lab_log="$lab_source/logs/setup-kernel-env-$lab_stamp.log"
exec > >(tee -a "$lab_log") 2>&1
trap 'printf "\nSETUP STOPPED at line %s. Log: %s\n" "$LINENO" "$lab_log" >&2' ERR
fail() { printf '\nERROR: %s\n' "$*" >&2; return 1; }

[[ $EUID -ne 0 ]] || fail 'Run as your normal Linux user, without sudo.'
[[ ${WSL_DISTRO_NAME:-} == CUDA-Ubuntu-24.04 ]] || fail 'Use the CUDA-Ubuntu-24.04 WSL distribution.'
[[ -x /usr/local/cuda-12.8/bin/nvcc ]] || fail 'Expected the previously verified CUDA 12.8 compiler.'
for lab_tool in git curl python3 g++ ninja; do command -v "$lab_tool"; done
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
printf 'Installing the kernel-study environment. No model download or benchmark in this step.\n'

printf '\n[1/5] Check out the verified upstream commit\n'
mkdir -p "$HOME/projects"
if [[ -e $lab_repo || -L $lab_repo ]]; then
    [[ -d "$lab_repo/.git" ]] || fail 'The target directory already exists but is not the expected Git checkout.'
    [[ $(git -C "$lab_repo" remote get-url origin) == "$lab_url" ]] || fail 'Existing checkout has a different origin; it was preserved.'
    [[ $(git -C "$lab_repo" rev-parse HEAD) == "$lab_sha" ]] || fail 'Existing checkout has a different HEAD; it was preserved.'
    git -C "$lab_repo" diff --quiet || fail 'Existing checkout has source edits; it was preserved.'
    git -C "$lab_repo" diff --cached --quiet || fail 'Existing checkout has staged edits; it was preserved.'
else
    git clone --no-checkout "$lab_url" "$lab_repo"
    git -C "$lab_repo" switch -c kv-cache-study "$lab_sha"
fi
cd "$lab_repo"
[[ $(git rev-parse HEAD) == "$lab_sha" ]] || fail 'Checkout SHA did not match the selected baseline.'
printf 'PINNED_COMMIT=%s\n' "$(git rev-parse HEAD)"
lab_env="$lab_repo/experiments/environment"
lab_run="$lab_env/runs/$lab_stamp"
mkdir -p "$lab_run"
git rev-parse HEAD > "$lab_run/upstream-commit.txt"
git status --short > "$lab_run/git-status-before.txt"
printf '%s\n' "$lab_sha" > "$lab_env/upstream-commit.txt"
if [[ ! -e "$lab_repo/experiments/PROGRESS.md" ]]; then
    cp "$lab_source/PROGRESS.md" "$lab_repo/experiments/PROGRESS.md"
fi
if [[ ! -e "$lab_env/initial-observation.md" ]]; then
    cp "$lab_source/ENVIRONMENT-20260929.md" "$lab_env/initial-observation.md"
fi
cp -- "$lab_source/02-setup-kernel-env.sh" "$lab_run/setup-script.sh"
python3 - "$HOME" "$lab_run" <<'PY'
import pathlib, shutil, sys
logs = sorted(pathlib.Path(sys.argv[1]).glob('cuda-env-check-*.txt'), key=lambda p: p.stat().st_mtime)
if logs:
    shutil.copy2(logs[-1], pathlib.Path(sys.argv[2]) / 'preflight-original.txt')
PY

printf '\n[2/5] Install uv in a project-local tools directory\n'
lab_uv_dir="$lab_repo/.lab-tools"
lab_uv="$lab_uv_dir/uv"
if [[ ! -x $lab_uv ]]; then
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 180 \
        https://astral.sh/uv/0.12.20/install.sh -o "$lab_run/uv-installer.sh"
    UV_INSTALL_DIR="$lab_uv_dir" UV_NO_MODIFY_PATH=1 sh "$lab_run/uv-installer.sh"
fi
"$lab_uv" --version | tee "$lab_run/uv-version.txt"
[[ $("$lab_uv" --version) == 'uv 0.12.20'* ]] || fail 'Unexpected uv version in the project-local tools directory.'
for lab_ignore in '/.lab-tools/' '/.venv/'; do
    grep -Fqx "$lab_ignore" .git/info/exclude || printf '%s\n' "$lab_ignore" >> .git/info/exclude
done

printf '\n[3/5] Create an isolated Python 3.12 environment and lock dependencies\n'
if [[ -e "$lab_repo/.venv" ]]; then
    [[ -f "$lab_env/managed-kernel-environment" ]] || fail 'An existing unmarked .venv was preserved. Report this message.'
else
    "$lab_uv" venv --python /usr/bin/python3 "$lab_repo/.venv"
    touch "$lab_env/managed-kernel-environment"
fi
if [[ -f "$lab_env/requirements.in" ]]; then
    cmp -s "$lab_source/kernel-requirements.in" "$lab_env/requirements.in" || fail 'Requirements changed; existing lock was preserved for review.'
else
    cp "$lab_source/kernel-requirements.in" "$lab_env/requirements.in"
fi
if [[ ! -f "$lab_env/requirements.lock" ]]; then
    "$lab_uv" pip compile "$lab_env/requirements.in" \
        --python "$lab_repo/.venv/bin/python" --torch-backend cu128 \
        --default-index https://pypi.org/simple --generate-hashes \
        --output-file "$lab_env/requirements.lock.partial" > "$lab_run/dependency-resolution.txt"
    mv "$lab_env/requirements.lock.partial" "$lab_env/requirements.lock"
fi
sha256sum "$lab_env/requirements.lock" | tee "$lab_run/lock-sha256.txt"
printf 'Dependency lock saved. Installing packages; this may download several GB.\n'
"$lab_uv" pip install --python "$lab_repo/.venv/bin/python" \
    --torch-backend cu128 --default-index https://pypi.org/simple \
    --require-hashes -r "$lab_env/requirements.lock"
"$lab_uv" pip check --python "$lab_repo/.venv/bin/python"

printf '\n[4/5] Configure this shell to use the original source checkout\n'
lab_activate="$lab_repo/experiments/activate.sh"
if [[ ! -f $lab_activate ]]; then
    cat > "$lab_activate" <<'SH'
# Source from Bash: source ~/projects/mini-sglang-kv/experiments/activate.sh
lab_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$lab_root/.venv/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$lab_root/.lab-tools:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$lab_root/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$lab_root"
unset lab_root
SH
fi
source "$lab_activate"
export LAB_RUN_DIR="$lab_run"

printf '\n[5/5] Validate imports, GPU access, and upstream test collection\n'
python - <<'PY'
import importlib.metadata as md
import json, os, pathlib, platform, subprocess, sys
import torch
import tvm_ffi
from minisgl.kernel import store_cache
from minisgl.benchmark.perf import compare_memory_kernel_perf

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available(), 'PyTorch cannot access the GPU'
# This is a small environment check, not a KV kernel or a benchmark.
probe = torch.ones(1, device='cuda')
assert probe.item() == 1.0
torch.cuda.synchronize()
props = torch.cuda.get_device_properties(0)
report = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'python': sys.version, 'python_executable': sys.executable,
    'kernel': platform.release(), 'os_release': platform.freedesktop_os_release(),
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'gpu': props.name, 'compute_capability': [props.major, props.minor],
    'total_memory_bytes': props.total_memory,
    'store_source': __import__('inspect').getfile(store_cache),
    'packages': {p: md.version(p) for p in ['apache-tvm-ffi', 'transformers', 'triton', 'pytest', 'pyzmq', 'msgpack']},
    'scope': 'kernel-study dependencies only; full inference stack and counters not verified',
    'kv_kernel_execution': 'not yet tested',
}
path = pathlib.Path(os.environ['LAB_RUN_DIR']) / 'environment.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print('ENV_IMPORT_PASS')
PY
python -m pytest -o addopts= --collect-only -q tests/kernel/test_store.py | tee "$lab_run/test-collection.txt"
"$lab_uv" pip freeze --python "$lab_repo/.venv/bin/python" > "$lab_run/installed-packages.txt"
git diff --exit-code -- python tests pyproject.toml
printf 'SETUP_PASS: original source imports, GPU access works, upstream store test is collected.\n'
printf 'Not yet a KV-kernel correctness result or a performance baseline.\n'
printf 'Project: %s\nLog: %s\n' "$lab_repo" "$lab_log"
printf 'Next session: source ~/projects/mini-sglang-kv/experiments/activate.sh\n'
cp "$lab_run/environment.json" "$lab_source/logs/environment-$lab_stamp.json"
