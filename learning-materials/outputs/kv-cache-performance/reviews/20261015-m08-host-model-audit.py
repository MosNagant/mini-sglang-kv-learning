"""Independent CPU audit of saved 46 host-path model checks/timings; no GPU import."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def local(saved):
    """Resolve saved WSL references to their existing shared Windows artifacts."""
    p = Path(saved)
    if p.exists():
        return p.resolve()
    text = str(saved).replace('\\', '/')
    match = re.match(r'^/mnt/([a-zA-Z])/(.*)$', text)
    if match:
        mapped = Path(match[1].upper() + ':/' + match[2])
        if mapped.exists():
            return mapped.resolve()
    prefix = '/home/weltlaw/projects/mini-sglang-kv/experiments/'
    if text.startswith(prefix):
        rest = text[len(prefix):]
        folder, suffix = rest.split('/', 1)
        mapped = HERE / 'logs' / (folder if folder.endswith('-results') else folder + '-results') / suffix
        if mapped.exists():
            return mapped.resolve()
    raise FileNotFoundError('Saved evidence reference is not locally available: ' + text)


def safe_child(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Manifest path escapes run: ' + name)
    return path


def frozen(run):
    plan = read(run / 'experiment.json')
    files = {}
    for name, digest in plan['frozen_files'].items():
        path = safe_child(run, name)
        require(sha(path) == digest, 'Frozen file digest mismatch: ' + name)
        files[name] = digest
    for name, digest in plan['source_manifest'].items():
        require(sha(safe_child(run / 'source', name)) == digest, 'Frozen original source mismatch: ' + name)
    require(read(run / 'source-manifest.json') == plan['source_manifest'], 'Source manifest representations differ')
    require(len(plan['source_manifest']) == 102, 'Expected 102 original source files')
    require(sha(run / '44-m08-kv-cached-metadata.py') == plan['candidate_sha256'], '44 digest changed')
    require(sha(run / '40-m08-kv-backends.py') == plan['base_sha256'], '40 digest changed')
    refs = {}
    for path, digest in plan['reference_hashes'].items():
        require(sha(local(path)) == digest, 'Accepted reference digest mismatch: ' + path)
        refs[path] = digest
    accepted = read(run / 'accepted-request.json')
    require(sha(run / '39-m08-model-smoke.py') == accepted['helper_sha256'], '39 no longer bound to accepted 41')
    require(sha(run / '41-m08-model-check.py') == accepted['runner_sha256'], '41 no longer bound to accepted check')
    require(sha(run / '40-m08-kv-backends.py') == accepted['adapter_sha256'], '40 no longer bound to accepted check')
    old_timing = local(plan['accepted_timing'])
    require(sha(run / '42-m08-model-timing.py') == sha(old_timing / 'runner.py'), '42 no longer bound to accepted timing')
    require(accepted['packages'] == plan['packages'], 'Package declarations differ')
    require(accepted['model']['metadata_sha256'] == plan['model']['metadata_sha256'], 'Model metadata hashes differ')
    require([(w['name'], w['sha256']) for w in accepted['model']['weights']] ==
            [(w['name'], w['sha256']) for w in plan['model']['weights']], 'Model weight hash declarations differ')
    return plan, {'frozen_files': files, 'source_entries': 102, 'accepted_reference_hashes': refs}


def array_file(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


def compare(left, right):
    files = ('layer0-prefill.npz', 'layer0-decode.npz', 'sampled-logits.npz', 'final-cache.npz')
    arrays, bytes_per_side = [], 0
    for filename in files:
        a, b = array_file(left / filename), array_file(right / filename)
        require(a.keys() == b.keys(), 'Cross-arm array names differ')
        for key in sorted(a):
            require(same(a[key], b[key]), f'Cross-arm bytes differ: {filename}:{key}')
            arrays.append({'file': filename, 'name': key, 'shape': list(a[key].shape),
                           'dtype': str(a[key].dtype), 'bytes_per_side': a[key].nbytes})
            bytes_per_side += a[key].nbytes
    require(read(left / 'outputs.json') == read(right / 'outputs.json'), 'Returned outputs differ')
    require(read(left / 'result.json')['raw_generated_token_ids'] == read(right / 'result.json')['raw_generated_token_ids'],
            'Cross-arm tokens differ')
    return {'left': str(left), 'right': str(right), 'arrays': arrays,
            'array_count': len(arrays), 'bytes_per_side': bytes_per_side, 'tokens_and_returned_outputs_equal': True}


def audit_check_arm(run, arm, request, expected, device):
    root = run / arm
    result = read(root / 'result.json')
    require(result['status'] == 'MODEL_ARM_CHECKED', 'Strict worker incomplete')
    require(result['raw_generated_token_ids'] == expected and result['input_token_ids'] == request['input_token_ids'], 'Strict token workload differs')
    require(all(result['environment'][k] == v for k, v in device.items()), 'Strict GPU environment differs')
    artifacts = result['artifacts']
    require(set(artifacts) == {p.name for p in root.iterdir() if p.suffix in ('.npz', '.jsonl')}, 'Artifact set differs')
    for name, item in artifacts.items():
        require((root / name).stat().st_size == item['bytes'] and sha(root / name) == item['sha256'], 'Strict artifact digest mismatch: ' + name)
    raw = read(root / 'raw-output-tokens.json')
    require([x['token_id'] for x in raw] == expected and len(raw) == 32, 'Saved raw output differs')
    require(all(x['uid'] == 0 and x['finished'] == (i == 31) for i, x in enumerate(raw)), 'Raw generation termination differs')
    require(read(root / 'outputs.json')[0]['token_ids'] == expected, 'Returned generated IDs differ')
    trace = [json.loads(line) for line in (root / 'store-checks.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(trace) == 1792, 'Need 896 before/success trace pairs')
    phases, compiled = Counter(), Counter()
    for call in range(896):
        before, after = trace[call*2:call*2+2]
        require(before['status'] == 'BEFORE_BACKEND' and after['status'] == 'FULL_BACKING_AND_READONLY_CHECKED', 'Runtime check missing success')
        require(before['call_id'] == after['call_id'] == call, 'Call numbering differs')
        require(all(after[k] == v for k, v in before.items() if k != 'status'), 'Before/after trace identity differs')
        step, layer = divmod(call, 28)
        phase, batch = ('prefill', 128) if step == 0 else ('decode', 1)
        ids = list(range(128)) if step == 0 else [127 + step]
        inputs = request['input_token_ids'] if step == 0 else [expected[step-1]]
        require(after['phase'] == phase and after['layer'] == layer and after['tokens_B'] == batch, 'Phase/layer/workload mismatch')
        require(after['indices'] == after['positions'] == ids and after['input_token_ids'] == inputs, 'Actual index/input sequence mismatch')
        require(after['D'] == 1024 and after['readonly_allocations'] == 2, 'KV width/readonly allocation count mismatch')
        t = after['tensors']
        for name in ('k_cache', 'v_cache', 'k', 'v', 'indices'):
            m = t[name]
            require(m['device'] == 'cuda:0' and m['dtype'] == ('torch.int32' if name == 'indices' else 'torch.float16'), 'Actual device/dtype mismatch')
            require(m['data_ptr'] == m['storage_ptr'] + m['element_size'] * m['storage_offset'], 'Pointer/storage metadata mismatch')
        require(t['k']['shape'] == t['v']['shape'] == [batch, 1024] and t['k']['stride'] == t['v']['stride'] == [4096, 1], 'Source layout mismatch')
        require(t['k_cache']['shape'] == t['v_cache']['shape'] == [513, 8, 128] and t['k_cache']['stride'] == t['v_cache']['stride'] == [1024, 128, 1], 'Cache layout mismatch')
        require(t['indices']['shape'] == [batch] and t['indices']['stride'] == [1], 'Index layout mismatch')
        require(t['k_cache']['storage_ptr'] == t['v_cache']['storage_ptr'], 'Expected shared cache backing')
        require(t['k']['storage_ptr'] == t['v']['storage_ptr'] != t['k_cache']['storage_ptr'], 'Expected disjoint source/cache backing')
        require(t['indices']['storage_ptr'] not in (t['k']['storage_ptr'], t['k_cache']['storage_ptr']), 'Indices backing aliases inputs/cache')
        require(t['k_cache']['storage_offset'] == layer*513*1024 and t['v_cache']['storage_offset'] == (28+layer)*513*1024, 'Layer cache offset mismatch')
        expected_launch = {'backend': 'triton' if arm == 'triton' else 'original', 'B': batch, 'C': 513, 'D': 1024,
            'input_k_stride': 4096, 'input_v_stride': 4096, 'cache_k_stride': 1024, 'cache_v_stride': 1024,
            'grid': [batch] if arm == 'triton' else None, 'BLOCK_D': 1024 if arm == 'triton' else None,
            'num_warps': 4 if arm == 'triton' else None}
        require(all(after['actual_launch'][k] == v for k, v in expected_launch.items()), 'Actual backend launch mismatch')
        if arm == 'triton':
            require(after['compiled_kernel_type'] == 'CompiledKernel' and re.fullmatch('[0-9a-f]{64}', after['compiled_kernel_hash']), 'Compiled kernel evidence missing')
            compiled[after['compiled_kernel_hash']] += 1
        phases[phase] += 1
    require(phases == {'prefill': 28, 'decode': 868}, 'Phase counts differ')
    require(result['store_calls'] == result['full_cache_checks'] == result['selected_launch_count'] == 896, 'Result launch/check counts differ')
    require(result['readonly_allocation_checks'] == 1792 and result['original_calls'] == (0 if arm == 'triton' else 896), 'Readonly/routing counts differ')
    data = {name: array_file(root / name) for name in artifacts if name.endswith('.npz')}
    for phase in ('prefill', 'decode'):
        sample = data[f'layer0-{phase}.npz']
        indices = sample['indices']
        require(indices.dtype == np.int32 and indices.tolist() == (list(range(128)) if phase == 'prefill' else [128]), 'Saved sample indices differ')
        for kind in ('k', 'v'):
            reconstructed = sample[f'before_{kind}'].copy().reshape(513, 1024)
            require(sample[kind].dtype == np.float16 and sample[kind].shape == (len(indices), 1024), 'Saved K/V dimensions differ')
            for src, dst in enumerate(indices):
                reconstructed[int(dst)] = sample[kind][src]
            reconstructed = reconstructed.reshape(sample[f'after_{kind}'].shape)
            require(same(reconstructed, sample[f'expected_{kind}']) and same(reconstructed, sample[f'after_{kind}']), 'Independent saved sample reconstruction failed')
            if phase == 'prefill':
                require(np.all(sample[f'before_{kind}'] == np.float16(-7)), 'Initial sentinel differs')
            else:
                require(same(sample[f'before_{kind}'], data['layer0-prefill.npz'][f'after_{kind}']), 'Saved sample chain differs')
    final = data['final-cache.npz']['cache']
    require(final.shape == (2, 28, 513, 1, 8, 128) and final.dtype == np.float16 and np.isfinite(final).all(), 'Final cache contract differs')
    require(np.all(final[:, :, 159:] == np.float16(-7)), 'Untouched final cache/dummy slots changed')
    for index, kind in enumerate(('k', 'v')):
        require(same(final[index, 0, :129], data['layer0-decode.npz'][f'after_{kind}'][:129]), 'Saved sample/final cache mismatch')
    for key, step in (('first_prefill', 0), ('first_decode', 1), ('last_decode', 31)):
        logits = data['sampled-logits.npz'][key]
        require(logits.dtype == np.float16 and logits.shape == (1, 151936) and np.isfinite(logits).all(), 'Saved logits contract differs')
        require(int(np.argmax(logits[0])) == expected[step], 'Saved logits argmax/token mismatch')
    return {'result_sha256': sha(root / 'result.json'), 'artifact_hashes': artifacts, 'trace_pairs': 896,
            'phase_counts': dict(phases), 'compiled_hash_counts': dict(compiled), 'saved_samples_reconstructed': 2,
            'saved_logits_checked': 3, 'raw_output_steps': 32}


def audit_check(run, plan, result):
    require(result['status'] == 'HOST_PATH_MODEL_CORRECTNESS_PASS', 'New correctness is not PASS')
    require(result['fresh_workers'] == 2 and result['total_store_checks'] == 1792, 'Expected two strict workers')
    request = read(run / 'request.json')
    require(request['adapter_sha256'] == plan['candidate_sha256'], 'Strict check not bound to44')
    require(request['runner_sha256'] == sha(run / '41-m08-model-check.py'), 'Strict check not bound to41')
    require(request['engine_options']['cuda_graph_bs'] == [] and len(request['input_token_ids']) == 128 and request['max_new_tokens'] == 32, 'Strict workload changed')
    expected = read(run / 'accepted-result.json')['raw_expected_token_ids']
    require(result['raw_expected_token_ids'] == expected, 'Accepted/new output tokens differ')
    arms = {arm: audit_check_arm(run, arm, request, expected, plan['accepted_device']) for arm in ('original', 'triton')}
    require(result['workers'] == [read(run / arm / 'result.json') for arm in ('original', 'triton')], 'Aggregate strict worker payloads differ')
    pairs = [compare(run / 'original', run / 'triton'), compare(local(plan['accepted_check']) / 'original', run / 'original')]
    return {'arms': arms, 'pairs': pairs, 'successful_runtime_records': 1792,
            'paired_arrays_compared': sum(x['array_count'] for x in pairs),
            'paired_bytes_per_side': sum(x['bytes_per_side'] for x in pairs),
            'limit': '1792 runtime success records/source audited; only four saved layer0 samples independently reconstructed. Exact final cache and three logits/arm verified. Eager only; no Graph full-cache or all-intermediate-array replay.'}


def executable_ptx(text):
    lines = []
    for line in text.splitlines():
        if re.match(r'\s*\.section\s+\.debug_', line):
            break
        if not re.match(r'\s*\.(?:file|loc)\b', line):
            lines.append(line)
    return '\n'.join(lines).strip() + '\n'


def audit_timing(run, plan, result):
    require(result['status'] == 'HOST_PATH_MODEL_TIMING_COLLECTION_COMPLETE', 'Timing run incomplete')
    require(len(result['workers']) == plan['fresh_processes'] == result['fresh_workers'] == 18, 'Expected 18 timing workers')
    require(plan['warmups_per_worker'] == 3 and plan['samples_per_worker'] == 5, 'Timing policy differs')
    checked = local(plan['accepted_new_check'])
    require(sha(checked / 'result.json') == plan['accepted_new_result_sha256'], 'New correctness reference changed')
    require(read(checked / 'result.json')['status'] == 'HOST_PATH_MODEL_CORRECTNESS_PASS', 'New correctness not accepted')
    require(read(run / 'accepted-new-result.json') == read(checked / 'result.json'), 'New acceptance snapshot differs')
    require(read(run / 'accepted-new-experiment.json') == read(checked / 'experiment.json'), 'New correctness experiment snapshot differs')
    original = (run / '42-m08-model-timing.py').read_text(encoding='utf-8')
    derived = (run / '42-derived-timing-worker.py').read_text(encoding='utf-8')
    start = derived.index('        # 46 controlled change:')
    end = derived.index("        result['environment']=", start)
    require(derived[:start] + derived[end:] == original, 'Derived42 changes extend beyond declared warmup insertion')
    insertion = derived[start:end]
    require('        generate()\n        torch.cuda.synchronize()\n' in insertion and 'if raw!=expected:' in insertion and "result['extra_measured_path_warmups']=1" in insertion, 'Measured-path warmup/token-check missing')
    expected_diff = ''.join(difflib.unified_diff(original.splitlines(keepends=True), derived.splitlines(keepends=True),
        fromfile='42-m08-model-timing.py', tofile='42-derived-timing-worker.py'))
    require((run / '42-derived-timing-worker.diff').read_text(encoding='utf-8') == expected_diff, 'Saved derivation diff differs')
    expected = read(run / 'accepted-result.json')['raw_expected_token_ids']
    workers, caches, sample_count = {}, [], 0
    for item in result['workers']:
        label, policy, rnd = item['host_path_arm'], item['policy'], item['round']
        key = (label, policy, rnd)
        require(key not in workers and label in ('original', 'checked_triton', 'cached_triton') and policy in ('eager', 'decode_graph') and rnd in (1, 2, 3), 'Timing identity duplicate/invalid')
        path = safe_child(run, item['saved_result'])
        saved = read(path)
        require(sha(path) == item['result_sha256'], 'Timing worker saved result hash differs')
        require(all(item[k] == v for k, v in saved.items()), 'Aggregate worker payload differs')
        require(saved['status'] == 'MODEL_TIMING_WORKER_COMPLETE' and len(saved['samples']) == 5, 'Timing worker incomplete')
        require(saved['extra_measured_path_warmups'] == 1, 'Unequal measured path warmup')
        env = saved['environment']
        require(env['packages'] == plan['packages'] and env['gpu'] == plan['accepted_device']['gpu'] and env['torch'] == plan['accepted_device']['torch'] and env['cuda'] == plan['accepted_device']['torch_cuda'], 'Timing environment differs')
        options = dict(read(run / 'accepted-request.json')['engine_options'])
        options['cuda_graph_bs'] = [1] if policy == 'decode_graph' else []
        require(env['engine_options'] == options, 'Actual engine options differ')
        root = run / 'backends' / label
        backend_plan = read(root / 'plan.json')
        require(backend_plan['input_token_ids'] == read(run / 'accepted-request.json')['input_token_ids'] and backend_plan['raw_expected_token_ids'] == expected, 'Timing workload differs')
        require(sha(root / 'adapter.py') == backend_plan['adapter_sha256'] == plan['backend_adapter_sha256'][label], 'Backend adapter binding differs')
        require(sha(root / '40-m08-kv-backends.py') == plan['base_sha256'], 'Device kernel base source differs')
        for name, digest in plan['source_manifest'].items():
            require(sha(safe_child(root / 'source', name)) == digest, 'Per-backend source copy differs')
        for sample_index, sample in enumerate(saved['samples']):
            seconds = sample['seconds']
            require(sample['sample'] == sample_index and math.isfinite(seconds) and seconds > 0, 'Invalid timer/sample sequence')
            require(sample['raw_generated_token_ids'] == expected and sample['raw_generated_steps'] == 32, 'Timed generation output differs')
            require(sample['output'][0]['token_ids'] == expected and sample['returned_tokens'] == 32, 'Returned output differs')
            require(close(sample['raw_output_tokens_per_second'], 32 / seconds), 'Raw token rate differs')
            sample_count += 1
        median = statistics.median(x['seconds'] for x in saved['samples'])
        require(close(median, saved['median_seconds']) and close(32/median, saved['raw_steps_per_second_at_median']), 'Saved median/rate differs')
        cache = saved['measured_metadata_cache_before_samples']
        if label == 'cached_triton':
            require(isinstance(cache, dict) and cache['entries'] == (2 if policy == 'eager' else 1), 'Expected warmed metadata signatures missing')
            require(cache['max_entries'] == 64 and cache['custom_hit_miss_counters'] is False and cache['eviction'] == 'FIFO',
                    'Cached metadata policy differs')
            caches.append({'policy': policy, 'round': rnd, 'metadata_cache': cache})
        else:
            require(cache is None, 'Uncached arm unexpectedly has metadata cache')
        require(saved['initial_backend_observation']['launch_count'] > 0, 'No observed backend launch')
        workers[key] = saved
    require(sample_count == result['timed_generations'] == 90, 'Expected 90 timed generations')
    require([(x['label'], x['policy'], x['round']) for x in plan['order']] == list(workers), 'Collection order differs from preregistered order')
    comparisons = {}
    for policy in ('eager', 'decode_graph'):
        comparisons[policy] = {}
        for comparator in ('original', 'checked_triton'):
            pairs = []
            for rnd in (1, 2, 3):
                a, b = [workers[label, policy, rnd]['median_seconds'] for label in (comparator, 'cached_triton')]
                pairs.append({'round': rnd, 'comparator_ms': a*1000, 'cached_triton_ms': b*1000,
                              'comparator_over_cached': a/b, 'cached_time_change_pct': (b/a-1)*100})
            expected_comp = {'pairs': pairs, 'all_rounds_at_least_3pct_faster': all(p['cached_time_change_pct'] <= -3 for p in pairs)}
            require(expected_comp == result['comparisons'][policy][comparator], 'Pair statistics/signal flag differs')
            comparisons[policy][comparator] = expected_comp
    recorded_ptx = read(run / 'ptx-comparison.json')
    require(recorded_ptx == result['warmup_ptx_pairs'] and len(recorded_ptx) == 6, 'PTX summary differs')
    ptx = []
    for entry in recorded_ptx:
        paths = [run / 'backends' / label / f"{entry['policy']}-round{entry['round']}-triton" / 'warmup-kernel.ptx'
                 for label in ('checked_triton', 'cached_triton')]
        normalized = [executable_ptx(path.read_text(encoding='utf-8')) for path in paths]
        require([sha(p) for p in paths] == entry['raw_sha256'], 'Raw PTX digest differs')
        require([hashlib.sha256(t.encode()).hexdigest() for t in normalized] == entry['normalized_sha256'], 'Normalized PTX digest differs')
        require(normalized[0] == normalized[1] and entry['executable_ptx_equal'], 'Executable PTX differs')
        ptx.append(entry)
    return {'workers_verified': 18, 'samples_verified': sample_count, 'output_tokens_verified': sample_count*32,
            'comparisons': comparisons, 'cache_observations': caches, 'ptx_pairs_verified': ptx,
            'derived42_exact_single_insertion_verified': True,
            'limit': 'All model sample outputs checked; timings include the existing generate boundary, not pure GPU kernels. PTX equality covers last observed warmup specialization only. The all-round >=3% rule is engineering evidence, not statistical significance or general speedup.'}


def audit(run):
    run = run.resolve(strict=True)
    plan, hashes = frozen(run)
    result = read(run / 'result.json')
    require(result['experiment_sha256'] == sha(run / 'experiment.json'), 'Final result/experiment digest differs')
    require(result['candidate_sha256'] == plan['candidate_sha256'], 'Final candidate digest differs')
    details = (audit_check if plan['mode'] == 'check' else audit_timing)(run, plan, result)
    return {'status': 'INDEPENDENT_CPU_HOST_MODEL_AUDIT_PASS', 'mode': plan['mode'], 'gpu_run': False,
            'run': str(run), 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
            'audit_script_sha256': sha(Path(__file__)), 'python': sys.version,
            'result_sha256': sha(run / 'result.json'), 'experiment_sha256': sha(run / 'experiment.json'),
            'hash_checks': hashes, 'details': details,
            'global_limit': 'No GPU/model rerun, no checkpoint reread from WSL, no learner-mastery credit. Hashes of model weights are the recorded experiment evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.run / 'host-path-evidence-audit.json'
    if destination.exists():
        raise FileExistsError('Preserve existing audit; specify a new --output for a deliberate repeat')
    try:
        report = audit(args.run)
    except BaseException:
        report = {'status': 'INDEPENDENT_CPU_HOST_MODEL_AUDIT_FAILED', 'gpu_run': False, 'error': traceback.format_exc(),
                  'run': str(args.run), 'audit_script_sha256': sha(Path(__file__))}
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'mode': report['mode'], 'saved': str(destination)}, ensure_ascii=False))
