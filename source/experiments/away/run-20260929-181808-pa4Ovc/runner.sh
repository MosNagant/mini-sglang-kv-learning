#!/usr/bin/env bash
# Repair the known missing dependency, run upstream once, then probe profilers.
# This is scaffolding only: no new CUDA kernel or benchmark timing implementation.
set -Eeuo pipefail
lab_profiler_only=false
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --profilers-only ) ]]; then
    printf 'Usage: bash 04-away-run.sh [--profilers-only]\n' >&2
    exit 2
fi
if [[ ${1:-} == --profilers-only ]]; then lab_profiler_only=true; fi
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_repo="$HOME/projects/mini-sglang-kv"
[[ $EUID -ne 0 ]] || { printf 'Run this script as weltlaw, without sudo.\n' >&2; exit 1; }
[[ -f "$lab_repo/experiments/activate.sh" ]] || { printf 'Project environment missing.\n' >&2; exit 1; }
mkdir -p "$lab_repo/experiments/away" "$lab_source/logs"
lab_run=$(mktemp -d "$lab_repo/experiments/away/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
lab_shared="$lab_source/logs/away-$(basename "$lab_run")"
mkdir -p "$lab_shared"
exec > >(tee -a "$lab_run/console.log" "$lab_shared/console.log") 2>&1
lab_stage=START
lab_headers=NOT_RUN
lab_test=NOT_RUN
lab_nsys=NOT_RUN
lab_ncu=NOT_RUN
lab_finish() {
    lab_exit=$?
    set +e
    {
        printf 'EXIT_CODE=%s\nLAST_STAGE=%s\n' "$lab_exit" "$lab_stage"
        printf 'PYTHON_HEADERS=%s\nUPSTREAM_TEST=%s\n' "$lab_headers" "$lab_test"
        printf 'NSYS_PROBE=%s\nNCU_PROBE=%s\n' "$lab_nsys" "$lab_ncu"
        printf 'PROFILER_SCOPE=existing cuda-starter vector_add; not KV performance evidence\n'
        printf 'COUNTER_VALUES=report content must be reviewed before claiming availability\n'
        printf 'RUN_DIRECTORY=%s\nWINDOWS_ARTIFACTS=%s\n' "$lab_run" "$lab_shared"
        date -Is
    } | tee "$lab_run/summary.txt" "$lab_shared/summary.txt"
    for lab_artifact in "$lab_run"/*.txt "$lab_run"/*.csv "$lab_run"/*.nsys-rep \
        "$lab_run"/*.ncu-rep "$lab_run"/*.sqlite "$lab_run"/nsys-*.log "$lab_run"/ncu-*.log; do
        if [[ -f $lab_artifact ]]; then cp "$lab_artifact" "$lab_shared/"; fi
    done
    if [[ $lab_exit -eq 0 ]]; then
        printf '\nAWAY_RUN_COMPLETE: read the individual profiler statuses above.\n'
    else
        printf '\nAWAY_RUN_STOPPED: stage=%s exit=%s; earlier logs were preserved.\n' "$lab_stage" "$lab_exit"
    fi
    printf 'Return with the summary above. Keep this terminal open while running.\n'
    exit "$lab_exit"
}
trap lab_finish EXIT
cp "$lab_source/04-away-run.sh" "$lab_run/runner.sh"
cp "$lab_source/PROGRESS.md" "$lab_run/coach-progress.md"
if [[ $lab_profiler_only == true ]]; then
    printf 'Profiler-only follow-up: no package installation or upstream test rerun.\n'
    lab_headers=NOT_CHECKED_PROFILER_ONLY_MODE
    lab_test=NOT_RERUN_PROFILER_ONLY_MODE
else
printf 'This run installs Python development headers, runs the original KV test once, then probes Nsight.\n'
printf 'Only the dependency installation needs sudo. Enter your Linux password at the prompt.\n'

lab_stage=INSTALL_PYTHON_HEADERS
lab_headers=INSTALLING
sudo -v
# Authenticate before the long-running root command; no later password prompts.
# Do not forcibly time out dpkg while it is changing the package database.
sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l bash -c '
    set -e
    apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
    apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
        install -y --no-install-recommends --no-remove python3.12-dev
'
source "$lab_repo/experiments/activate.sh"
python -c 'import pathlib, sysconfig; p=pathlib.Path(sysconfig.get_path("include"))/"Python.h"; assert p.is_file(), str(p); print("PYTHON_HEADERS_PASS:", p)'
lab_headers=PASS
dpkg-query -W python3.12 python3.12-dev libpython3.12-dev > "$lab_run/python-system-packages.txt"

lab_stage=UPSTREAM_TEST
lab_test=RUNNING
printf '\nRunning the original KV test, with a 30-minute limit.\n'
if timeout --kill-after=30s 1800s bash "$lab_source/03-run-upstream-test.sh"; then
    lab_test=PASS
else
    lab_code=$?
    lab_test="FAILED_EXIT_$lab_code"
    exit "$lab_code"
fi
fi
source "$lab_repo/experiments/activate.sh"

lab_stage=PROFILER_CAPABILITY_PROBES
lab_probe="$HOME/projects/cuda-starter/build/vector_add"
if [[ -x $lab_probe ]]; then
    sha256sum "$lab_probe" > "$lab_run/probe-executable-sha256.txt"
    printf '\nProfiler probes use the previously validated vector_add executable.\n'
    printf 'Each collection is limited to 3 minutes; profiler failures are recorded independently.\n'
    if command -v nsys >/dev/null 2>&1; then
        nsys --version > "$lab_run/nsys-version.txt"
        if timeout --kill-after=10s 180s nsys profile --trace=cuda,nvtx \
            --sample=none --cpuctxsw=none --output="$lab_run/nsys-vector-add" \
            "$lab_probe" > "$lab_run/nsys-profile.log" 2>&1; then
            lab_nsys=NO_REPORT
            if [[ -s "$lab_run/nsys-vector-add.nsys-rep" ]]; then
                lab_nsys=REPORT_GENERATED_PENDING_REVIEW
                if timeout --kill-after=10s 120s nsys stats --report cuda_gpu_kern_sum \
                    --format csv "$lab_run/nsys-vector-add.nsys-rep" \
                    > "$lab_run/nsys-kernels.csv" 2> "$lab_run/nsys-stats.log"; then
                    if grep -q 'vector_add' "$lab_run/nsys-kernels.csv"; then
                        lab_nsys=VECTOR_ADD_KERNEL_TRACE_PRESENT
                    elif grep -q 'does not contain CUDA kernel data' "$lab_run/nsys-kernels.csv"; then
                        lab_nsys=REPORT_HAS_NO_CUDA_KERNEL_DATA
                    fi
                fi
            fi
        else
            lab_nsys="FAILED_EXIT_$?"
        fi
    else
        lab_nsys=NOT_ON_PATH
    fi
    if command -v ncu >/dev/null 2>&1; then
        ncu --version > "$lab_run/ncu-version.txt"
        if timeout --kill-after=10s 180s ncu --set basic --launch-count 1 \
            --export "$lab_run/ncu-vector-add" "$lab_probe" \
            > "$lab_run/ncu-profile.log" 2>&1; then
            lab_ncu=NO_REPORT
            if [[ -s "$lab_run/ncu-vector-add.ncu-rep" ]]; then
                lab_ncu=REPORT_GENERATED_PENDING_REVIEW
                if timeout --kill-after=10s 120s ncu --import "$lab_run/ncu-vector-add.ncu-rep" \
                    --page raw --csv > "$lab_run/ncu-metrics.csv" 2> "$lab_run/ncu-import.log"; then
                    if grep -q 'vector_add' "$lab_run/ncu-metrics.csv"; then
                        lab_ncu=VECTOR_ADD_REPORT_EXPORTED_PENDING_METRIC_REVIEW
                    fi
                fi
            fi
        else
            lab_ncu="FAILED_EXIT_$?"
        fi
        if grep -q 'ERR_NVGPUCTRPERM' "$lab_run/ncu-profile.log"; then
            lab_ncu=GPU_COUNTER_PERMISSION_DENIED
        fi
    else
        lab_ncu=NOT_ON_PATH
    fi
else
    lab_nsys=SKIPPED_EXISTING_PROBE_NOT_FOUND
    lab_ncu=SKIPPED_EXISTING_PROBE_NOT_FOUND
fi
lab_stage=COMPLETE
