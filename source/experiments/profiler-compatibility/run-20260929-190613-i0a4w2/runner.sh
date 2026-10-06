#!/usr/bin/env bash
# Install one compatible Systems version; independently probe minimal Compute metrics.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_repo="$HOME/projects/mini-sglang-kv"
lab_package=nsight-systems-2025.1.3
lab_expected_version=2025.1.3.140
[[ $EUID -ne 0 ]] || { printf 'Run as your normal Linux user, without sudo.\n' >&2; exit 1; }
source "$lab_repo/experiments/activate.sh"
lab_probe="$HOME/projects/cuda-starter/build/vector_add"
[[ -x $lab_probe ]] || { printf 'Existing vector_add probe not found.\n' >&2; exit 1; }
mkdir -p "$lab_repo/experiments/profiler-compatibility" "$lab_source/logs"
lab_run=$(mktemp -d "$lab_repo/experiments/profiler-compatibility/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
lab_shared="$lab_source/logs/profiler-compatibility-$(basename "$lab_run")"
mkdir -p "$lab_shared"
exec > >(tee -a "$lab_run/console.log" "$lab_shared/console.log") 2>&1
lab_stage=START
lab_ncu=NOT_RUN
lab_nsys=NOT_RUN
lab_finish() {
    lab_exit=$?
    set +e
    {
        printf 'EXIT_CODE=%s\nLAST_STAGE=%s\n' "$lab_exit" "$lab_stage"
        printf 'NCU_MINIMAL=%s\nNSYS_PROBE=%s\n' "$lab_ncu" "$lab_nsys"
        printf 'SCOPE=vector_add profiler capability checks; not KV benchmark data\n'
        printf 'WINDOWS_ARTIFACTS=%s\nRUN_DIRECTORY=%s\n' "$lab_shared" "$lab_run"
        date -Is
    } | tee "$lab_run/summary.txt" "$lab_shared/summary.txt"
    for lab_artifact in "$lab_run"/*.txt "$lab_run"/*.json "$lab_run"/*.csv \
        "$lab_run"/*.sqlite "$lab_run"/*.nsys-rep "$lab_run"/*.ncu-rep \
        "$lab_run"/nsys-*.log "$lab_run"/ncu-*.log; do
        if [[ -f $lab_artifact ]]; then cp "$lab_artifact" "$lab_shared/"; fi
    done
    exit "$lab_exit"
}
trap lab_finish EXIT
cp "$lab_source/05-profiler-compatibility.sh" "$lab_run/runner.sh"
cp "$lab_source/PROGRESS.md" "$lab_run/coach-progress.md"
sha256sum "$lab_probe" > "$lab_run/probe-executable-sha256.txt"
nvcc --version > "$lab_run/nvcc-before.txt"

lab_stage=INSTALL_SYSTEMS_2025_1_3
printf 'Preparing Nsight Systems %s from your configured APT source. Enter your Linux password if prompted.\n' "$lab_expected_version"
sudo -v
sudo -n apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
LC_ALL=C apt-cache policy "$lab_package" > "$lab_run/package-policy.txt"
# A repository .deb filename suffix need not equal the package's Version field.
# Select the real APT candidate, require the intended upstream build, then pin it.
lab_package_version=$(awk '$1 == "Candidate:" {print $2}' "$lab_run/package-policy.txt")
if [[ $lab_package_version != "$lab_expected_version" && $lab_package_version != "$lab_expected_version"-* ]]; then
    printf 'Expected Nsight Systems build %s, but APT candidate is %s. See package-policy.txt.\n' \
        "$lab_expected_version" "$lab_package_version" >&2
    exit 1
fi
printf 'NSYS_APT_VERSION=%s\n' "$lab_package_version"
printf '%s=%s\n' "$lab_package" "$lab_package_version" > "$lab_run/package-selection.txt"
LC_ALL=C apt-get -s --no-install-recommends install "$lab_package=$lab_package_version" > "$lab_run/install-plan.txt"
if grep -Eq '^Inst (cuda-|nvidia-|libnvidia-|linux-)|^Remv ' "$lab_run/install-plan.txt"; then
    printf 'Unexpected driver, toolkit, kernel, or removal in package plan. Plan saved; installation stopped.\n' >&2
    exit 1
fi
sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get \
    -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 \
    install -y --no-install-recommends --no-remove "$lab_package=$lab_package_version"
lab_nsys_bin=''
while IFS= read -r lab_path; do
    if [[ $lab_path == */bin/nsys && -x $lab_path ]]; then lab_nsys_bin=$lab_path; fi
done < <(dpkg-query -L "$lab_package")
[[ -n $lab_nsys_bin ]] || { printf 'Cannot locate the installed package-specific nsys executable.\n' >&2; exit 1; }
printf 'NSYS_EXECUTABLE=%s\n' "$lab_nsys_bin"
"$lab_nsys_bin" --version | tee "$lab_run/nsys-version.txt"
dpkg-query -W "$lab_package" > "$lab_run/nsys-package.txt"
[[ $(dpkg-query -W -f='${Version}' "$lab_package") == "$lab_package_version" ]] || {
    printf 'Installed package version does not match the recorded selection.\n' >&2
    exit 1
}
nvcc --version > "$lab_run/nvcc-after.txt"
cmp "$lab_run/nvcc-before.txt" "$lab_run/nvcc-after.txt"

lab_stage=COMPUTE_MINIMAL_PROBE
ncu --version > "$lab_run/ncu-version.txt"
printf '\nRunning Compute FIRST, with only gpu__time_duration.sum requested.\n'
if timeout --kill-after=10s 180s ncu --metrics gpu__time_duration.sum --launch-count 1 \
    --export "$lab_run/ncu-minimal" "$lab_probe" > "$lab_run/ncu-profile.log" 2>&1; then
    lab_ncu=NO_REPORT
    if [[ -s "$lab_run/ncu-minimal.ncu-rep" ]]; then
        lab_ncu=REPORT_GENERATED_PENDING_REVIEW
        if timeout --kill-after=10s 60s ncu --import "$lab_run/ncu-minimal.ncu-rep" \
            --page raw --csv > "$lab_run/ncu-metrics.csv" 2> "$lab_run/ncu-import.log"; then
            lab_ncu=MINIMAL_REPORT_EXPORTED_PENDING_METRIC_REVIEW
        fi
    fi
else
    lab_ncu="FAILED_EXIT_$?"
fi
if grep -Eq '^==ERROR==[[:space:]]+ERR_NVGPUCTRPERM([[:space:]]|$)' "$lab_run/ncu-profile.log"; then
    lab_ncu=GPU_COUNTER_PERMISSION_DENIED
elif grep -Fq 'Profiling failed because a driver resource was unavailable' "$lab_run/ncu-profile.log"; then
    lab_ncu=DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE
fi
cat "$lab_run/ncu-profile.log"

lab_stage=SYSTEMS_COMPATIBILITY_PROBE
if timeout --kill-after=10s 180s "$lab_nsys_bin" profile --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none --output="$lab_run/nsys-vector-add" \
    "$lab_probe" > "$lab_run/nsys-profile.log" 2>&1; then
    lab_nsys=NO_REPORT
    if [[ -s "$lab_run/nsys-vector-add.nsys-rep" ]]; then
        lab_nsys=REPORT_GENERATED_PENDING_REVIEW
        if timeout --kill-after=10s 120s "$lab_nsys_bin" stats --report cuda_gpu_kern_sum \
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
cat "$lab_run/nsys-profile.log"
export LAB_RUN_DIR="$lab_run"
python - <<'PY'
import json, os, pathlib, sqlite3
folder = pathlib.Path(os.environ['LAB_RUN_DIR'])
p = folder / 'nsys-vector-add.sqlite'
if p.exists():
    c = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)
    names = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    diagnostics = []
    if 'DIAGNOSTIC_EVENT' in names:
        diagnostics = [dict(zip(('severity', 'text'), row)) for row in c.execute('select severity, text from DIAGNOSTIC_EVENT')]
    counts = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL'):
        counts[table] = c.execute('select count(*) from '+table).fetchone()[0] if table in names else None
    result = {'event_counts': counts, 'diagnostics': diagnostics}
    (folder / 'nsys-diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
else:
    print('No SQLite report available; retain profiler logs for review.')
PY
lab_stage=COMPLETE
