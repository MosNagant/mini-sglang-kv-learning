"""M05 closeout evidence: one fresh repeat and one real KV trace attempt.
Coach-owned orchestration. Never changes the original kernel or Nsight config.
Run from the activated mini-sglang repository. No M05 acceptance is auto-claimed.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import traceback

COMMIT = '9a91cfafe754aa85daee49998176275667eb58f2'
PILOT = 'experiments/learning/m05-original-pilot/run-d9gjhywx'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def checked_repo(repo):
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, timeout=15).strip()
    status = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True, timeout=15)
    if commit != COMMIT or status.strip():
        raise RuntimeError('Upstream commit or tracked files changed: review before measuring.')
    if not __debug__:
        raise RuntimeError('Python assertions must be enabled.')
    return commit


def run_command(command, log, timeout=180):
    # Linux only. A timeout stops this command's own process group, not WSL.
    print('STEP:', log.stem, flush=True)
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONOPTIMIZE': '0'})
        try:
            code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    record = {'command': [str(x) for x in command], 'exit_code': code, 'timed_out': timed_out, 'log': log.name}
    print(log.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
    return record


def trace_child(output):
    import torch
    from minisgl.kernel import store_cache
    checked_repo(Path.cwd())
    batch, width, capacity = 256, 128, 1024
    generator = torch.Generator().manual_seed(20261004)
    k = torch.randint(0, 1024, (batch, width), generator=generator, dtype=torch.int32).to(device='cuda', dtype=torch.float16)
    v = k + 1024
    indices = torch.arange(batch, device='cuda', dtype=torch.int32)
    pool = torch.full((capacity, 2, width), -1, device='cuda', dtype=torch.float16)
    kc, vc = pool[:, 0, :], pool[:, 1, :]
    kb, vb, ib = k.cpu().clone(), v.cpu().clone(), indices.cpu().clone()
    expected = pool.cpu().clone()
    expected[:batch, 0, :] = kb
    expected[:batch, 1, :] = vb
    def call():
        store_cache(kc, vc, indices, k, v)
    def check():
        if not (torch.equal(pool.cpu(), expected) and torch.equal(k.cpu(), kb)
                and torch.equal(v.cpu(), vb) and torch.equal(indices.cpu(), ib)):
            raise AssertionError('KV trace workload changed targets, untouched storage or inputs.')
    call()
    torch.cuda.synchronize()
    check()
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    pool.fill_(-1)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push('M05_ORIGINAL_KV_STEADY_20_CALLS')
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    check()
    save(output, {'status': 'KV_TRACE_WORKLOAD_OK', 'commit': COMMIT,
                  'batch': batch, 'width': width, 'capacity': capacity,
                  'calls_in_nvtx_range': 20, 'expected_total_store_calls': 41,
                  'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'input_stride': list(k.stride()),
                  'cache_stride': list(kc.stride()), 'dtype': str(k.dtype),
                  'scope': 'profiling workload; no benchmark time used'})
    print('KV_TRACE_WORKLOAD_OK: full cache and inputs checked', flush=True)


def inspect_trace(database):
    if not database.exists():
        return {'status': 'NO_SQLITE_EXPORT', 'located_store_events': 0}
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'StringIds' not in tables:
            return {'status': 'UNKNOWN_SQLITE_SCHEMA', 'located_store_events': 0}
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        found = []
        total = 0
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL'):
            if table not in tables:
                continue
            columns = {x[1] for x in db.execute('PRAGMA table_info("' + table + '")')}
            name_column = next((c for c in ('demangledName', 'shortName', 'mangledName') if c in columns), None)
            if name_column is None or not {'start', 'end'} <= columns:
                continue
            for start, end, name_id in db.execute('SELECT start,end,"' + name_column + '" FROM "' + table + '"'):
                total += 1
                name = strings.get(name_id, str(name_id))
                if 'store_kv_cache' in name and end > start >= 0:
                    found.append({'table': table, 'name': name, 'start_ns': start,
                                  'end_ns': end, 'duration_ns': end-start})
        diagnostics = []
        if 'DIAGNOSTIC_EVENT' in tables:
            columns = {x[1] for x in db.execute('PRAGMA table_info("DIAGNOSTIC_EVENT")')}
            if {'severity', 'text'} <= columns:
                diagnostics = [{'severity': s, 'text': t} for s, t in db.execute('SELECT severity,text FROM DIAGNOSTIC_EVENT')]
    return {'status': 'KV_KERNEL_LOCATED' if found else 'NO_VALID_KV_KERNEL_EVENTS',
            'located_store_events': len(found), 'all_kernel_events': total,
            'store_events': found, 'diagnostics': diagnostics,
            'limit': 'Kernel location evidence only. No profiled duration substitutes for unprofiled Event timings.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-child', type=Path)
    parser.add_argument('--shared-results', type=Path)
    args = parser.parse_args()
    if args.trace_child:
        trace_child(args.trace_child)
        return
    if sys.platform != 'linux':
        parser.error('Run in the activated WSL environment.')
    repo = Path.cwd()
    checked_repo(repo)
    source = Path(__file__).resolve().parent
    pilot = repo / PILOT
    if not (pilot / 'summary.json').exists():
        raise FileNotFoundError('Existing M05 pilot is missing; do not silently replace it.')
    parent = repo / 'experiments/learning/m05-closeout'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    shutil.copyfile(__file__, run / 'closeout.py')
    shutil.copyfile(source / '17-compare-event-graph.py', run / 'reproduction_runner.py')
    result = {'status': 'STARTED', 'commit': COMMIT, 'run': str(run),
              'old_pilot': str(pilot), 'steps': {}, 'full_m05_acceptance': False}
    print('M05_CLOSEOUT_RUN:', run, flush=True)
    try:
        # Independent of Nsight: repeat the frozen B256/D128 method once.
        step = run_command([sys.executable, str(run / 'reproduction_runner.py'),
                            '--batch', '256', '--width', '128', '--random-seed', '20261004',
                            '--result-parent', str(run / 'reproduction')], run / 'reproduction.log')
        result['steps']['reproduction'] = step
        if step['exit_code'] != 0:
            raise RuntimeError('Representative reproduction failed; inspect reproduction.log.')
        files = list((run / 'reproduction').glob('run-*/result.json'))
        if len(files) != 1:
            raise RuntimeError('Expected exactly one reproduction result.')
        new = json.loads(files[0].read_text())
        if new.get('status') != 'DIAGNOSTIC_COMPLETE' or new.get('tracked_git_status', '').strip():
            raise RuntimeError('Reproduction status/source rejected.')
        old = json.loads((pilot / 'summary.json').read_text())
        old_case = next(x for x in old['summary'] if x['batch'] == 256 and x['width'] == 128)
        comparison = {}
        for method in ('graph', 'eager'):
            values = old_case[method]['round_medians_us']
            value = new['summary_us_per_call'][method]['median']
            reference = statistics.median(values)
            comparison[method] = {'new_median_us': value, 'old_round_medians_us': values,
                                  'old_representative_us': reference,
                                  'difference_pct': (value / reference - 1) * 100,
                                  'within_observed_old_range': min(values) <= value <= max(values),
                                  'interpretation': 'A descriptive comparison, not a tolerance test or confidence interval.'}
        result['reproduction_comparison'] = comparison
        print('REPRODUCTION_COMPARISON:', json.dumps(comparison), flush=True)

        nsys = Path('/opt/nvidia/nsight-systems/2025.1.3/bin/nsys')
        if not nsys.is_file():
            found = shutil.which('nsys')
            nsys = Path(found) if found else None
        if nsys is None:
            result['profile'] = {'status': 'NSYS_UNAVAILABLE', 'reason': 'Existing nsys executable not found; no installation attempted.'}
        else:
            result['steps']['nsys_version'] = run_command([str(nsys), '--version'], run / 'nsys-version.log', 20)
            config_step = run_command([str(nsys), '-z'], run / 'nsys-config-path.log', 20)
            result['steps']['nsys_config'] = config_step
            config_name = (run / 'nsys-config-path.log').read_text().strip()
            config = Path(config_name)
            if config.is_absolute() and config.is_file():
                shutil.copyfile(config, run / 'nsys-existing-config.ini')
            result['timestamp_note'] = 'Configuration is recorded, never changed. CuptiUseRawGpuTimestamps=false has lower precision; trace is not baseline timing.'
            prefix = run / 'original-kv'
            step = run_command([str(nsys), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-event-trace=false',
                                '--output='+str(prefix), sys.executable, str(run / 'closeout.py'),
                                '--trace-child', str(run / 'trace-workload.json')], run / 'nsys-profile.log')
            result['steps']['nsys_profile'] = step
            report = prefix.with_suffix('.nsys-rep')
            if step['exit_code'] == 0 and report.exists():
                result['steps']['nsys_stats'] = run_command([str(nsys), 'stats', '--report', 'cuda_gpu_kern_sum',
                                                             '--format', 'csv', str(report)], run / 'nsys-kernels.csv', 120)
                result['profile'] = inspect_trace(prefix.with_suffix('.sqlite'))
                workload = run / 'trace-workload.json'
                result['profile']['workload_correctness'] = json.loads(workload.read_text()).get('status') if workload.exists() else 'MISSING'
                if result['profile']['workload_correctness'] != 'KV_TRACE_WORKLOAD_OK':
                    result['profile']['status'] = 'PROFILE_WORKLOAD_NOT_VERIFIED'
            else:
                result['profile'] = {'status': 'PROFILE_FAILED', 'exit_code': step['exit_code'],
                                     'timed_out': step['timed_out'], 'log': 'nsys-profile.log'}
        result['status'] = 'EVIDENCE_COLLECTION_COMPLETE_REVIEW_REQUIRED'
        print('M05_PROFILE_STATUS:', result['profile']['status'], flush=True)
    except BaseException:
        result.update(status='ERROR', error=traceback.format_exc())
        raise
    finally:
        result['script_sha256'] = hashlib.sha256((run / 'closeout.py').read_bytes()).hexdigest()
        save(run / 'closeout.json', result)
        print('SAVED:', run / 'closeout.json', flush=True)
        if args.shared_results:
            args.shared_results.mkdir(parents=True, exist_ok=True)
            shared = args.shared_results / run.name
            shutil.copytree(run, shared)
            print('SHARED_CLOSEOUT:', shared, flush=True)
    print('M05_EVIDENCE_READY: review reproduction and trace/limitations; no automatic learning or baseline PASS.')


if __name__ == '__main__':
    main()
