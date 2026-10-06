#!/usr/bin/env bash
# Test NVIDIA's WSL timestamp-conversion workaround on the existing vector_add.
set -Eeuo pipefail
lab_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lab_repo="$HOME/projects/mini-sglang-kv"
[[ $EUID -ne 0 ]] || { printf 'Run as your normal Linux user, without sudo.\n' >&2; exit 1; }
source "$lab_repo/experiments/activate.sh"
lab_nsys=''
while IFS= read -r lab_path; do
    if [[ $lab_path == */bin/nsys && -x $lab_path ]]; then lab_nsys=$lab_path; fi
done < <(dpkg-query -L nsight-systems-2025.1.3)
[[ -n $lab_nsys ]] || { printf 'The verified Nsight Systems 2025.1.3 package is missing.\n' >&2; exit 1; }
lab_probe="$HOME/projects/cuda-starter/build/vector_add"
[[ -x $lab_probe ]] || { printf 'Existing vector_add probe is missing.\n' >&2; exit 1; }
mkdir -p "$lab_repo/experiments/nsys-wsl-timestamps" "$lab_source/logs"
lab_run=$(mktemp -d "$lab_repo/experiments/nsys-wsl-timestamps/run-$(date +%Y%m%d-%H%M%S)-XXXXXX")
lab_shared="$lab_source/logs/nsys-wsl-timestamps-$(basename "$lab_run")"
mkdir -p "$lab_shared"
exec > >(tee -a "$lab_run/console.log" "$lab_shared/console.log") 2>&1
lab_status=NOT_RUN
lab_config_state=NOT_APPLIED_THIS_RUN
lab_finish() {
    lab_exit=$?
    set +e
    {
        printf 'EXIT_CODE=%s\nNSYS_WSL_TRACE=%s\n' "$lab_exit" "$lab_status"
        printf 'TIMESTAMP_SETTING=%s\n' "$lab_config_state"
        printf 'TIMESTAMP_PRECISION=the false setting uses a less precise conversion mechanism\n'
        printf 'NCU_STATUS=unresolved; not retested in this step\n'
        printf 'SCOPE=vector_add capability probe; not KV performance evidence\n'
        printf 'WINDOWS_ARTIFACTS=%s\nRUN_DIRECTORY=%s\n' "$lab_shared" "$lab_run"
        date -Is
    } | tee "$lab_run/summary.txt" "$lab_shared/summary.txt"
    for lab_artifact in "$lab_run"/*.txt "$lab_run"/*.json "$lab_run"/*.ini \
        "$lab_run"/*.csv "$lab_run"/*.sqlite "$lab_run"/*.nsys-rep "$lab_run"/nsys-*.log; do
        if [[ -f $lab_artifact ]]; then cp "$lab_artifact" "$lab_shared/"; fi
    done
    exit "$lab_exit"
}
trap lab_finish EXIT
cp "$lab_source/06-nsys-wsl-timestamps.sh" "$lab_run/runner.sh"
cp "$lab_source/PROGRESS.md" "$lab_run/coach-progress.md"
if [[ -f "$lab_repo/experiments/PROGRESS.md" ]]; then
    cp "$lab_repo/experiments/PROGRESS.md" "$lab_run/previous-progress.md"
fi
cp "$lab_source/PROGRESS.md" "$lab_repo/experiments/PROGRESS.md"
"$lab_nsys" --version | tee "$lab_run/nsys-version.txt"
sha256sum "$lab_probe" > "$lab_run/probe-executable-sha256.txt"
lab_config=$("$lab_nsys" -z)
printf 'NSYS_CONFIG_PATH=%s\n' "$lab_config"
export LAB_RUN_DIR="$lab_run" LAB_NSYS_CONFIG="$lab_config" LAB_CONFIG_HOME="$HOME"
python - <<'PY'
import json, os, pathlib, re, shutil
folder = pathlib.Path(os.environ['LAB_RUN_DIR'])
home = pathlib.Path(os.environ['LAB_CONFIG_HOME']).resolve()
config = pathlib.Path(os.environ['LAB_NSYS_CONFIG'])
if not config.is_absolute() or not config.resolve().is_relative_to(home):
    raise SystemExit(f'Unexpected nsys config path; no configuration changed: {config}')
existed = config.exists()
before = config.read_bytes() if existed else b''
if existed:
    shutil.copy2(config, folder / 'nsys-config.before.ini')
text = before.decode('utf-8-sig')
pattern = r'(?m)^[ \t]*CuptiUseRawGpuTimestamps[ \t]*=([^\r\n]*)'
values = re.findall(pattern, text)
already_false = bool(values) and all(v.strip().lower() == 'false' for v in values)
if already_false:
    (folder / 'config-already-false.txt').write_text('Setting was already false; no change or redundant profiling run.\n')
    after = text
else:
    after = re.sub(pattern, 'CuptiUseRawGpuTimestamps=false', text)
    if not values:
        after += ('' if not after or after.endswith('\n') else '\n') + 'CuptiUseRawGpuTimestamps=false\n'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(after, encoding='utf-8')
    print('CONFIG_UPDATED: CuptiUseRawGpuTimestamps=false')
shutil.copy2(config, folder / 'nsys-config.after.ini')
record = {
    'config_path': str(config), 'existed_before': existed,
    'changed': not already_false, 'previous_values': values,
    'setting': 'CuptiUseRawGpuTimestamps=false',
    'backup': str(folder / 'nsys-config.before.ini') if existed else None,
    'timing_limitation': 'CUPTI timestamp conversion workaround; lower precision than the normal mechanism',
    'reference': 'https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536',
}
(folder / 'configuration-change.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
lab_config_state=CuptiUseRawGpuTimestamps_false
if [[ -f "$lab_run/config-already-false.txt" ]]; then
    lab_status=SETTING_ALREADY_FALSE_PROBE_NOT_REPEATED
    exit 0
fi
if timeout --kill-after=10s 180s "$lab_nsys" profile --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none --output="$lab_run/nsys-vector-add" \
    "$lab_probe" > "$lab_run/nsys-profile.log" 2>&1; then
    lab_status=REPORT_PENDING_VALIDATION
else
    lab_status="PROFILE_FAILED_EXIT_$?"
    cat "$lab_run/nsys-profile.log"
    exit 1
fi
cat "$lab_run/nsys-profile.log"
timeout --kill-after=10s 120s "$lab_nsys" stats --report cuda_gpu_kern_sum \
    --format csv "$lab_run/nsys-vector-add.nsys-rep" \
    > "$lab_run/nsys-kernels.csv" 2> "$lab_run/nsys-stats.log"
cat "$lab_run/nsys-kernels.csv"
python - <<'PY'
import json, os, pathlib, sqlite3
folder = pathlib.Path(os.environ['LAB_RUN_DIR'])
database = folder / 'nsys-vector-add.sqlite'
if not database.exists():
    raise SystemExit('No SQLite export; retain reports for review.')
c = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
tables = {row[0] for row in c.execute("select name from sqlite_master where type='table'")}
counts, valid_counts = {}, {}
for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL'):
    counts[table] = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] if table in tables else None
    if 'KERNEL' in table and table in tables:
        valid_counts[table] = c.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "end" > "start" AND "start" >= 0').fetchone()[0]
diagnostics = [dict(zip(('severity', 'text'), row)) for row in c.execute('select severity, text from DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
status = 'GPU_KERNEL_EVENTS_PRESENT' if sum(valid_counts.values()) > 0 else 'NO_VALID_GPU_KERNEL_EVENTS'
record = {'status': status, 'event_counts': counts, 'positive_duration_gpu_events': valid_counts,
          'timestamp_mode': 'CuptiUseRawGpuTimestamps=false',
          'timing_limitation': 'Workaround has lower timestamp precision; not a microbenchmark baseline.',
          'diagnostics': diagnostics}
(folder / 'nsys-diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
(folder / 'nsys-status.txt').write_text(status + '\n')
print(json.dumps(record, indent=2))
PY
lab_status=$(cat "$lab_run/nsys-status.txt")
