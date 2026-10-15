"""Saved real-model experiment for the cached-metadata host-path hypothesis.

Coach implementation. This does not modify mini-sglang or the accepted 40/41/42
files. Check mode reuses the exact 41 worker with the explicit 44 adapter; timing
reuses 42 with one documented, identical extra measured-path warmup for all arms.
No installation, model download, default-backend change or guaranteed speedup.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
HELPER = '39-m08-model-smoke.py'
BASE = '40-m08-kv-backends.py'
CHECK = '41-m08-model-check.py'
TIMING = '42-m08-model-timing.py'
CANDIDATE = '44-m08-kv-cached-metadata.py'
PACKAGES = ('torch', 'triton', 'flashinfer-python', 'transformers', 'safetensors', 'apache-tvm-ffi')
LABELS = ('original', 'checked_triton', 'cached_triton')
EXPECTED_SOURCE_COMMIT = '9a91cfafe754aa85daee49998176275667eb58f2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def verify_frozen(run):
    plan = read(run / 'experiment.json')
    for name, sha in plan['frozen_files'].items():
        require(digest(run / name) == sha, 'Frozen helper/plan changed: ' + name)
    for name, sha in plan['source_manifest'].items():
        require(digest(run / 'source' / name) == sha, 'Frozen upstream source changed: ' + name)
    require(package_versions() == plan['packages'], 'Package versions differ from accepted environment')
    return plan


def verify_device(plan):
    import torch
    actual = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'torch_cuda': torch.version.cuda, 'capability': list(torch.cuda.get_device_capability(0))}
    require(actual == plan['accepted_device'], 'Device/Torch/CUDA differs from accepted 41 environment')


def verified_references(args):
    old_check = args.accepted_check.resolve(strict=True)
    old_timing = args.accepted_timing.resolve(strict=True)
    accepted = read(old_check / 'result.json')
    request = read(old_check / 'request.json')
    timing = read(old_timing / 'plan.json')
    summary = read(old_timing / 'summary.json')
    require(accepted['status'] == 'MODEL_AB_CORRECTNESS_PASS', 'Accepted 41 correctness is not PASS')
    require(summary['status'] == 'MODEL_TIMING_COLLECTION_COMPLETE', 'Accepted 42 timing is incomplete')
    require(len(summary['workers']) == 12, 'Accepted 42 must contain all 12 workers')
    require(digest(HERE / HELPER) == request['helper_sha256'], '39 helper differs from accepted 41')
    require(digest(HERE / BASE) == request['adapter_sha256'] == timing['adapter_sha256'], '40 adapter differs from accepted 41/42')
    require(digest(HERE / CHECK) == request['runner_sha256'], '41 worker differs from accepted evidence')
    require(digest(HERE / TIMING) == digest(old_timing / 'runner.py'), '42 worker differs from accepted evidence')
    require(digest(HERE / CANDIDATE) == args.candidate_sha256, '44 candidate differs from requested frozen digest')
    commit_record = request['git_commit']
    require(commit_record['returncode'] == 0 and commit_record['stdout'].strip() == EXPECTED_SOURCE_COMMIT,
            'Accepted original commit changed')
    require(request['input_token_ids'] == timing['input_token_ids'] and len(request['input_token_ids']) == 128,
            'Input token workload differs between accepted 41/42')
    require(request['max_new_tokens'] == timing['max_new_tokens'] == 32, 'Expected 32 generated steps')
    require(request['engine_options'] == timing['engine_options'], 'Accepted engine options differ')
    require(accepted['raw_expected_token_ids'] == timing['raw_expected_token_ids'], 'Expected original outputs differ')
    require(request['packages'] == timing['timing_packages'] == package_versions(), 'Accepted package environment changed')
    sources = read(old_check / 'source-manifest.json')
    require(sources == timing['source_manifest'], 'Accepted original source snapshots differ')
    for name, sha in sources.items():
        require(digest(old_check / 'source' / name) == sha, 'Accepted 41 source changed: ' + name)
        require(digest(old_timing / 'source' / name) == sha, 'Accepted 42 source changed: ' + name)
    return old_check, old_timing, accepted, request, timing, sources


def make_timing_worker(source):
    """Only add an untimed measured-entry warmup; preserve all 42 measurements."""
    anchor = '        kernel_api.store_cache=measured\n'
    require(source.count(anchor) == 1, '42 measured-entry swap anchor changed; review before deriving')
    addition = '''        # 46 controlled change: warm the measured callable's own metadata cache.
        # Match this extra request for every arm; it is outside every timer.
        generate()
        torch.cuda.synchronize()
        if raw!=expected:
            raise RuntimeError('Measured-path warmup differs from accepted original')
        result['extra_measured_path_warmups']=1
        result['measured_metadata_cache_before_samples']=(measured.metadata_cache_info()
            if hasattr(measured,'metadata_cache_info') else None)
'''
    return source.replace(anchor, anchor + addition)


def freeze(args, run, refs):
    old_check, old_timing, accepted, request, timing, sources = refs
    helper = load(HERE / HELPER, 'host_path_snapshot_helper')
    current_sources = helper.snapshot_source(args.project, run)
    require(current_sources == sources, 'Current upstream source differs from accepted 41/42')
    # The helper's runner.py is retained under its actual canonical name below.
    (run / 'runner.py').rename(run / 'snapshot-helper.py')
    for name in (HELPER, BASE, CHECK, TIMING, CANDIDATE):
        shutil.copyfile(HERE / name, run / name)
    shutil.copyfile(__file__, run / 'runner.py')
    model = helper.inspect_model(Path(request['model']['path']), request['model']['revision'])
    expected_weights = [(x['name'], x['sha256']) for x in request['model']['weights']]
    require([(x['name'], x['sha256']) for x in model['weights']] == expected_weights, 'Real model weight digest changed')
    require(model['metadata_sha256'] == request['model']['metadata_sha256'] == timing['model']['metadata_sha256'],
            'Real model/tokenizer metadata changed')
    save(run / 'accepted-request.json', request)
    save(run / 'accepted-result.json', accepted)
    save(run / 'accepted-timing-plan.json', timing)
    plan = {
        'mode': args.mode, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_sha256': args.candidate_sha256, 'base_sha256': digest(run / BASE),
        'source_manifest': sources, 'model': model, 'packages': package_versions(),
        'accepted_device': {key: read(old_check / 'original/result.json')['environment'][key]
                            for key in ('gpu', 'torch', 'torch_cuda', 'capability')},
        'accepted_check': str(old_check), 'accepted_timing': str(old_timing),
        'reference_hashes': {str(p): digest(p) for p in (old_check / 'request.json', old_check / 'result.json',
                                                      old_timing / 'plan.json', old_timing / 'summary.json')},
        'authorship': 'coach implementation and experiment; independent learner mastery not inferred',
        'hypothesis': 'Repeated Python metadata validation contributes to observed Eager regression; cache only validated full metadata signatures while retaining the exact 40 device kernel.',
        'scope': 'One Qwen3-0.6B FP16 TP1 request, 128 input tokens, 32 greedy generation steps; no model quality, multi-request, AMD or service-throughput claim.',
        'frozen_files': {},
    }
    return plan


def seal(run, plan, extra_files=()):
    names = [HELPER, BASE, CHECK, TIMING, CANDIDATE, 'runner.py', 'snapshot-helper.py',
             'source-manifest.json', 'accepted-request.json', 'accepted-result.json', 'accepted-timing-plan.json']
    names.extend(extra_files)
    plan['frozen_files'] = {name: digest(run / name) for name in names}
    save(run / 'experiment.json', plan)


def run_process(command, cwd, log, timeout=1800):
    with Path(log).open('xb') as handle:
        proc = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    require(proc.returncode == 0, f'Worker failed ({proc.returncode}); retained log: {log}')


def check_worker(run, arm):
    plan = verify_frozen(run)
    verify_device(plan)
    module = load(run / CHECK, 'host_path_frozen_check')
    module.ADAPTER = CANDIDATE
    module.ADAPTER_SHA256 = plan['candidate_sha256']
    # All original 41 checks, counts, exact byte comparisons and saved artifacts
    # remain in its worker. Only the explicitly selected adapter pin changes.
    module.worker(run, arm)


def run_check(args, run, plan, summary):
    request = read(run / 'accepted-request.json')
    request.update(model=plan['model'], adapter_sha256=plan['candidate_sha256'],
                   runner_sha256=digest(run / CHECK),
                   authorship='coach 46 orchestration reuses frozen 41 checks with explicit 44 host adapter')
    save(run / 'request.json', request)
    seal(run, plan, ['request.json'])
    for arm in ('original', 'triton'):
        print('RUN strict-check ' + arm, flush=True)
        run_process([sys.executable, '-B', '-u', str(run / 'runner.py'), '--worker-check', str(run), '--arm', arm],
                    args.project, run / (arm + '.log'))
        result = read(run / arm / 'result.json')
        require(result['status'] == 'MODEL_ARM_CHECKED' and result['full_cache_checks'] == 896,
                'Strict 41 worker missing expected 896 checks')
        summary['workers'].append(result)
        save(run / 'result.json', summary)
    module = load(run / CHECK, 'host_path_compare_check')
    comparisons = [module.compare_arms(run / 'original', run / 'triton')]
    comparisons.append(module.compare_arms(Path(plan['accepted_check']) / 'original', run / 'original'))
    expected = read(run / 'accepted-result.json')['raw_expected_token_ids']
    require(all(x['raw_generated_token_ids'] == expected for x in summary['workers']), 'Outputs differ from frozen original')
    summary.update(status='HOST_PATH_MODEL_CORRECTNESS_PASS', comparisons=comparisons,
                   raw_expected_token_ids=expected, total_store_checks=1792, fresh_workers=2,
                   candidate_sha256=plan['candidate_sha256'], experiment_sha256=digest(run / 'experiment.json'),
                   scope='Eager diagnostic: exact complete final cache, selected logits, saved K/V samples and all 896 calls per worker; no Graph full-cache correctness claim.')


def executable_ptx(text):
    """Drop only debug metadata/path lines; never rewrite executable instructions."""
    kept = []
    for line in text.splitlines():
        if re.match(r'\s*\.section\s+\.debug_', line):
            break
        if re.match(r'\s*\.(?:file|loc)\b', line):
            continue
        kept.append(line)
    return '\n'.join(kept).strip() + '\n'


def timing_worker(run, label, policy, round_id):
    plan = verify_frozen(run)
    verify_device(plan)
    backend = run / 'backends' / label
    actual_arm = 'original' if label == 'original' else 'triton'
    require(digest(backend / 'adapter.py') == plan['backend_adapter_sha256'][label], 'Selected adapter changed')
    require(digest(backend / BASE) == plan['base_sha256'], 'Backend 40 kernel changed')
    module = load(run / '42-derived-timing-worker.py', 'host_path_frozen_timing')
    module.worker(backend, actual_arm, policy, round_id)


def run_timing(args, run, plan, summary):
    require(args.accepted_new_check is not None, '--accepted-new-check is required before timing')
    checked = args.accepted_new_check.resolve(strict=True)
    result = read(checked / 'result.json')
    checked_plan = verify_frozen(checked)
    require(result['status'] == 'HOST_PATH_MODEL_CORRECTNESS_PASS', 'New adapter strict correctness not accepted')
    require(result['experiment_sha256'] == digest(checked / 'experiment.json'), 'Accepted new experiment plan changed')
    require(result['candidate_sha256'] == checked_plan['candidate_sha256'] == plan['candidate_sha256'], '44 changed after new correctness')
    require(checked_plan['source_manifest'] == plan['source_manifest'], 'Source differs from new correctness')
    require(checked_plan['reference_hashes'] == plan['reference_hashes'], 'Accepted original references differ')
    require(checked_plan['model']['metadata_sha256'] == plan['model']['metadata_sha256'], 'Model metadata differs from new correctness')
    request = read(run / 'accepted-request.json')
    expected = read(run / 'accepted-result.json')['raw_expected_token_ids']
    require(result['raw_expected_token_ids'] == expected, 'New correctness output differs from original')
    save(run / 'accepted-new-result.json', result)
    save(run / 'accepted-new-experiment.json', checked_plan)
    original_timing = (run / TIMING).read_text(encoding='utf-8')
    derived = make_timing_worker(original_timing)
    (run / '42-derived-timing-worker.py').write_text(derived, encoding='utf-8')
    (run / '42-derived-timing-worker.diff').write_text(''.join(difflib.unified_diff(
        original_timing.splitlines(keepends=True), derived.splitlines(keepends=True),
        fromfile=TIMING, tofile='42-derived-timing-worker.py')), encoding='utf-8')
    plan.update(accepted_new_check=str(checked), accepted_new_result_sha256=digest(checked / 'result.json'),
                backend_adapter_sha256={}, fresh_processes=18, samples_per_worker=5,
                warmups_per_worker=3, warmup_policy='2 observed 42 warmups plus 1 measured-entry warmup for every arm',
                derivation='42 unchanged except one untimed generate+token-check after replacing observer with measured callable; full derived source retained',
                measurement='Same 42 perf_counter(generate + GPU completion): scheduler/sampling/D2H/raw-token-record/detokenize included; prepared IDs, no model load/JIT/warmup/network.',
                signal_rule='For a given policy and comparator, all 3 paired process medians >=3% faster is a scoped engineering signal only, not statistical significance. Keep every outcome; no resampling until win.')
    extra = ['42-derived-timing-worker.py', '42-derived-timing-worker.diff',
             'accepted-new-result.json', 'accepted-new-experiment.json']
    for label in LABELS:
        backend = run / 'backends' / label
        backend.mkdir(parents=True)
        shutil.copytree(run / 'source', backend / 'source')
        shutil.copyfile(run / (CANDIDATE if label == 'cached_triton' else BASE), backend / 'adapter.py')
        shutil.copyfile(run / BASE, backend / BASE)
        plan['backend_adapter_sha256'][label] = digest(backend / 'adapter.py')
        worker_plan = {'model': plan['model'], 'input_token_ids': request['input_token_ids'],
                       'max_new_tokens': request['max_new_tokens'], 'raw_expected_token_ids': expected,
                       'engine_options': request['engine_options'], 'source_manifest': plan['source_manifest'],
                       'adapter_sha256': digest(backend / 'adapter.py'), 'timing_packages': plan['packages'],
                       'host_path_arm': label, 'parent_experiment': '../../experiment.json'}
        save(backend / 'plan.json', worker_plan)
        extra += [f'backends/{label}/{name}' for name in ('adapter.py', BASE, 'plan.json')]
    order = []
    for rnd in (1, 2, 3):
        labels = LABELS[rnd - 1:] + LABELS[:rnd - 1]
        policies = ('decode_graph', 'eager') if rnd % 2 else ('eager', 'decode_graph')
        order += [dict(round=rnd, policy=policy, label=label) for policy in policies for label in labels]
    plan['order'] = order
    seal(run, plan, extra)
    for item in order:
        label, policy, rnd = item['label'], item['policy'], item['round']
        name = f'{policy}-round{rnd}-{label}'
        print('RUN ' + name, flush=True)
        run_process([sys.executable, '-B', '-u', str(run / 'runner.py'), '--worker-timing', str(run),
                     '--label', label, '--policy', policy, '--round', str(rnd)], args.project, run / (name + '.log'))
        actual_arm = 'original' if label == 'original' else 'triton'
        path = run / 'backends' / label / f'{policy}-round{rnd}-{actual_arm}' / 'result.json'
        entry = read(path)
        require(entry['status'] == 'MODEL_TIMING_WORKER_COMPLETE' and len(entry['samples']) == 5, 'Incomplete timing worker')
        require(entry.get('extra_measured_path_warmups') == 1, 'Measured-entry warmup missing')
        require(all(s['raw_generated_token_ids'] == expected for s in entry['samples']), 'Timed outputs differ')
        entry.update(host_path_arm=label, saved_result=path.relative_to(run).as_posix(), result_sha256=digest(path))
        summary['workers'].append(entry)
        save(run / 'result.json', summary)
        print(f"COLLECTED {name} median_ms={entry['median_seconds']*1000:.6f}", flush=True)
    ptx_pairs = []
    for policy in ('eager', 'decode_graph'):
        for rnd in (1, 2, 3):
            paths = [run / 'backends' / label / f'{policy}-round{rnd}-triton' / 'warmup-kernel.ptx'
                     for label in ('checked_triton', 'cached_triton')]
            normalized = [executable_ptx(path.read_text(encoding='utf-8')) for path in paths]
            equal = normalized[0] == normalized[1]
            record = {'policy': policy, 'round': rnd, 'raw_sha256': [digest(p) for p in paths],
                      'normalized_sha256': [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in normalized],
                      'executable_ptx_equal': equal, 'scope': 'last observed warmup specialization only; removed .file/.loc and trailing .debug sections, no instruction rewriting'}
            ptx_pairs.append(record)
    summary['warmup_ptx_pairs'] = ptx_pairs
    save(run / 'ptx-comparison.json', ptx_pairs)
    require(all(x['executable_ptx_equal'] for x in ptx_pairs), 'Retained old/new warmup executable PTX differs; no same-kernel causal claim')
    summary['comparisons'] = {}
    for policy in ('eager', 'decode_graph'):
        comparisons = {}
        for comparator in ('original', 'checked_triton'):
            pairs = []
            for rnd in (1, 2, 3):
                workers = {x['host_path_arm']: x for x in summary['workers'] if x['policy'] == policy and x['round'] == rnd}
                a, b = (workers[label]['median_seconds'] for label in (comparator, 'cached_triton'))
                pairs.append({'round': rnd, 'comparator_ms': a*1000, 'cached_triton_ms': b*1000,
                              'comparator_over_cached': a/b, 'cached_time_change_pct': (b/a-1)*100})
            comparisons[comparator] = {'pairs': pairs, 'all_rounds_at_least_3pct_faster': all(x['cached_time_change_pct'] <= -3 for x in pairs)}
        summary['comparisons'][policy] = comparisons
    summary.update(status='HOST_PATH_MODEL_TIMING_COLLECTION_COMPLETE', fresh_workers=18, timed_generations=90,
                   candidate_sha256=plan['candidate_sha256'], experiment_sha256=digest(run / 'experiment.json'),
                   limit=plan['scope'] + '; output-token checks in both policies; strict full cache/logits checks are the separate Eager check run; no unconditional speedup claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'timing'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--accepted-check', type=Path)
    parser.add_argument('--accepted-timing', type=Path)
    parser.add_argument('--accepted-new-check', type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--worker-check', type=Path)
    parser.add_argument('--worker-timing', type=Path)
    parser.add_argument('--arm', choices=('original', 'triton'))
    parser.add_argument('--label', choices=LABELS)
    parser.add_argument('--policy', choices=('eager', 'decode_graph'))
    parser.add_argument('--round', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.worker_check:
        require(args.arm is not None, 'Check worker needs --arm')
        check_worker(args.worker_check.resolve(), args.arm)
        return
    if args.worker_timing:
        require(args.label is not None and args.policy is not None and args.round is not None, 'Incomplete timing worker arguments')
        timing_worker(args.worker_timing.resolve(), args.label, args.policy, args.round)
        return
    if not args.mode or not args.accepted_check or not args.accepted_timing or not args.candidate_sha256:
        parser.error('--mode, --accepted-check, --accepted-timing and --candidate-sha256 are required')
    args.project = args.project.expanduser().resolve(strict=True)
    parent = args.project / 'experiments' / ('m08-host-path-' + args.mode)
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('RUN_DIRECTORY=' + str(run), flush=True)
    summary = {'status': 'FAILED', 'mode': args.mode, 'workers': []}
    primary = None
    try:
        refs = verified_references(args)
        plan = freeze(args, run, refs)
        (run_check if args.mode == 'check' else run_timing)(args, run, plan, summary)
    except BaseException:
        primary = traceback.format_exc()
        summary.update(status='FAILED', error=primary)
        (run / 'controller-error.txt').write_text(primary, encoding='utf-8')
        raise
    finally:
        save(run / 'result.json', summary)
        shared = HERE / 'logs' / ('m08-host-path-' + args.mode + '-results') / run.name
        try:
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run, shared)
            print('SHARED_RUN_DIRECTORY=' + str(shared), flush=True)
        except Exception:
            print('SHARED_COPY_FAILED; local run retained: ' + str(run), flush=True)
            if primary is None:
                raise
    print('STATUS=' + summary['status'], flush=True)
    if 'comparisons' in summary:
        print(json.dumps(summary['comparisons'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # CLI/preflight failures occurring before run creation are also retained.
        try:
            parent = HERE / 'logs/m08-host-path-errors'
            parent.mkdir(parents=True, exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix='error-', dir=parent))
            save(out / 'error.json', {'argv': sys.argv, 'error': traceback.format_exc(), 'runner_sha256': digest(__file__)})
        except Exception:
            pass
        raise
