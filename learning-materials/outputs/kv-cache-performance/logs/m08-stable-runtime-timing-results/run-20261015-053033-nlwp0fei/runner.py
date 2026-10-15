"""Verify/timestamp53 original Eager versus original whole-decode CUDA Graph.

Two fresh strict workers retain all32 logits; only a numerical/token PASS enables
three fresh timing pairs, each with two warmups and five complete requests.
This tests an upstream execution feature, not a newly faster Triton kernel.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
RUNTIME = '53-m08-stable-runtime.py'
ATOL = RTOL = 0.002


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_frozen(run):
    cfg = read(run/'plan.json')
    for name, expected in cfg['frozen_files'].items():
        require(sha(run/name) == expected, 'Frozen experiment file changed: '+name)
    for name, expected in cfg['source_manifest'].items():
        require(sha(run/'source'/name) == expected, 'Frozen original source changed: '+name)
    runtime = load(run/'stable_runtime.py', 'frozen_stable_runtime')
    require(runtime.CONTRACT == cfg['contract'], 'Embedded stable contract changed')
    return cfg, runtime


def worker(run, mode, policy, round_id):
    import numpy as np
    out = run/(policy if mode == 'check' else f'{policy}-round{round_id}')
    out.mkdir()
    result = {'status': 'FAILED', 'mode': mode, 'policy': policy, 'round': round_id, 'samples': []}
    runtime = None
    try:
        cfg, runtime_module = validate_frozen(run)
        runtime = runtime_module.StableRuntime(execution=policy, project=run/'source', model_path=cfg['model_path'])
        result['environment'] = runtime.environment
        result['source_validation'] = runtime.validation
        expected = cfg['contract']['raw_expected_token_ids']
        if mode == 'check':
            logits, dtypes = [], []
            sampler = runtime.model.engine.sampler
            original_sample = sampler.sample
            graph_runner = runtime.model.engine.graph_runner
            original_replay = graph_runner.replay
            replay_count = 0

            def sample(value, args):
                require(tuple(value.shape) == (1,151936), 'Unexpected sampled logits shape')
                require(str(value.dtype) in ('torch.float16','torch.float32'), 'Unexpected sampled logits dtype')
                logits.append(value.detach().clone())
                dtypes.append(str(value.dtype))
                return original_sample(value, args)

            def replay(batch):
                nonlocal replay_count
                replay_count += 1
                return original_replay(batch)

            sampler.sample = sample
            graph_runner.replay = replay
            try:
                output = runtime.generate()
                runtime.synchronize()
            finally:
                sampler.sample = original_sample
                graph_runner.replay = original_replay
            require(len(logits) == 32, 'Missing sampled logits')
            require(replay_count == (31 if policy == 'decode_graph' else 0), 'Actual Graph replay count differs')
            # Eager returns FP16; Graph copies into an upstream FP32 output buffer.
            # Convert CPU snapshots to common FP32, preserving all source values.
            arrays = {f'step{index:02d}': value.cpu().numpy().astype(np.float32)
                      for index, value in enumerate(logits)}
            np.savez(out/'all-logits.npz', **arrays)
            result.update(status='STABLE_RUNTIME_LOGITS_COLLECTED', output=output,
                raw_generated_token_ids=list(runtime.raw_token_ids), logits_count=32,
                logits_source_dtypes=dtypes, saved_logit_dtype='float32', logits_sha256=sha(out/'all-logits.npz'),
                graph_replays_observed=replay_count,
                limit='All32 sampled logits from one fixed request; observer only in strict diagnostic, never timing.')
        else:
            for _ in range(2):
                runtime.generate()
                runtime.synchronize()
                require(runtime.raw_token_ids == expected, 'Warmup output differs from accepted tokens')
            runtime._torch.cuda.reset_peak_memory_stats()
            for index in range(5):
                runtime.synchronize()
                started = time.perf_counter()
                output = runtime.generate()
                runtime.synchronize()
                seconds = time.perf_counter()-started
                record = {'sample': index, 'seconds': seconds, 'raw_generated_token_ids': list(runtime.raw_token_ids),
                          'raw_generated_steps': len(runtime.raw_token_ids), 'output': output,
                          'raw_output_tokens_per_second': len(runtime.raw_token_ids)/seconds if seconds > 0 else None}
                result['samples'].append(record)
                save(out/'result.json', result)
                require(math.isfinite(seconds) and seconds > 0 and runtime.raw_token_ids == expected,
                        'Invalid timed duration/output')
            result.update(status='STABLE_RUNTIME_TIMING_WORKER_COMPLETE',
                median_seconds=statistics.median(row['seconds'] for row in result['samples']),
                peak_allocated_bytes=runtime._torch.cuda.max_memory_allocated(),
                measurement='perf_counter around persistent StableRuntime.generate and GPU completion; API guards/raw token recording/scheduler/sampling/detokenize included; model loading/hash/JIT/capture/warmup excluded')
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                result['shutdown_error'] = traceback.format_exc()
                result['status'] = 'FAILED'
        save(out/'result.json', result)
    require(result['status'] in ('STABLE_RUNTIME_LOGITS_COLLECTED','STABLE_RUNTIME_TIMING_WORKER_COMPLETE'), 'Worker cleanup failed')


def compare_logits(run, expected):
    import numpy as np
    records = {policy: read(run/policy/'result.json') for policy in ('eager','decode_graph')}
    require(all(record['status'] == 'STABLE_RUNTIME_LOGITS_COLLECTED' for record in records.values()), 'Strict workers incomplete')
    require(records['eager']['raw_generated_token_ids'] == expected, 'Fresh original Eager differs from historical accepted tokens')
    steps = []
    with np.load(run/'eager/all-logits.npz', allow_pickle=False) as eager, \
         np.load(run/'decode_graph/all-logits.npz', allow_pickle=False) as graph:
        require(set(eager.files) == set(graph.files) == {f'step{i:02d}' for i in range(32)}, 'Incomplete logit step set')
        for step in range(32):
            a, b = eager[f'step{step:02d}'], graph[f'step{step:02d}']
            require(a.shape == b.shape == (1,151936) and a.dtype == b.dtype == np.float32, 'Saved logits metadata differs')
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            error = np.abs(b-a)
            violations = int(np.count_nonzero(~(np.isfinite(a)&np.isfinite(b)&(error <= ATOL+RTOL*np.abs(a)))))
            steps.append({'step': step, 'elements': a.size, 'finite': finite, 'violations': violations,
                          'max_abs_error': float(error.max()) if finite else None,
                          'equal_fp32_bit_patterns': bool(np.array_equal(a.view(np.uint32),b.view(np.uint32)))})
    exact_tokens = records['decode_graph']['raw_generated_token_ids'] == expected
    numerical = all(row['finite'] and row['violations'] == 0 for row in steps)
    return {'atol': ATOL, 'rtol': RTOL, 'steps': steps,
            'elements_compared': sum(row['elements'] for row in steps),
            'violations': sum(row['violations'] for row in steps),
            'exact_32_tokens_equal': exact_tokens, 'all_logits_within_tolerance': numerical,
            'pass': numerical and exact_tokens,
            'scope': 'Original Eager versus original whole-decode Graph, unchanged model math/KV. One fixed workload, no broad quality claim.'}


def prepare_run(args, run):
    require(sha(HERE/'stable/stable_infer.py') == sha(HERE/RUNTIME),
            'Release runtime differs from verified canonical53 source')
    module = load(HERE/RUNTIME, 'stable_runtime_source')
    verified = module.verify_environment(args.project, args.model_path)
    shutil.copyfile(HERE/RUNTIME, run/'stable_runtime.py')
    shutil.copyfile(__file__, run/'runner.py')
    shutil.copyfile(HERE/'54-verify-stable-runtime.sh', run/'54-verify-stable-runtime.sh')
    for name, expected in module.CONTRACT['source_manifest'].items():
        target = run/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.project/name, target)
        require(sha(target) == expected, 'Original source changed during snapshot: '+name)
    cfg = {'mode': args.mode, 'contract': module.CONTRACT, 'model_path': verified['model_path'],
           'source_manifest': module.CONTRACT['source_manifest'], 'validated_environment': verified,
           'frozen_files': {name: sha(run/name) for name in ('stable_runtime.py','runner.py','54-verify-stable-runtime.sh')},
           'numerical_contract': {'atol': ATOL, 'rtol': RTOL, 'all32logits': True, 'exact32tokens': True},
           'warmups_per_worker': 2, 'samples_per_worker': 5, 'fresh_rounds': 3,
           'comparison_baseline': 'original Eager using the identical persistent stable API',
           'method': 'enable upstream whole-decode Graph[1]; prefill remains Eager; no new kernel/math/KV replacement',
           'signal_rule': 'Graph lower latency by at least3% in all3 paired process medians is a scoped engineering signal, not statistical significance'}
    if args.mode == 'timing':
        require(args.accepted_check is not None, 'Timing requires --accepted-check')
        accepted = args.accepted_check.resolve(strict=True)
        result, old = read(accepted/'result.json'), read(accepted/'plan.json')
        require(result['status'] == 'STABLE_RUNTIME_CORRECTNESS_PASS' and result['comparison']['pass'], 'Strict stable-runtime check failed')
        require(result['plan_sha256'] == sha(accepted/'plan.json'), 'Accepted check plan changed')
        for key in ('contract','source_manifest','numerical_contract'):
            require(old[key] == cfg[key], 'Accepted check differs: '+key)
        require(old['frozen_files'] == cfg['frozen_files'], 'Runtime/verification source changed after correctness')
        for policy in ('eager','decode_graph'):
            record = read(accepted/policy/'result.json')
            require(sha(accepted/policy/'all-logits.npz') == record['logits_sha256'], 'Accepted logits changed')
            require(record in result['workers'], 'Saved strict worker differs from controller record')
        require(compare_logits(accepted, cfg['contract']['raw_expected_token_ids']) == result['comparison'], 'Recomputed strict comparison differs')
        shutil.copyfile(accepted/'result.json', run/'accepted-check.json')
        cfg['frozen_files']['accepted-check.json'] = sha(run/'accepted-check.json')
        cfg['accepted_check'] = str(accepted)
    save(run/'plan.json', cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check','timing'), required=True)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--accepted-check', type=Path)
    parser.add_argument('--worker-run', type=Path)
    parser.add_argument('--policy', choices=('eager','decode_graph'))
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()
    if args.worker_run:
        require(args.policy is not None, 'Worker needs policy')
        worker(args.worker_run.resolve(), args.mode, args.policy, args.round)
        return
    args.project = args.project.resolve(strict=True)
    parent = args.project/'experiments'/('m08-stable-runtime-'+args.mode)
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('RUN_DIRECTORY='+str(run), flush=True)
    result, primary = {'status': 'FAILED', 'mode': args.mode, 'workers': []}, None
    try:
        cfg = prepare_run(args, run)
        result['plan_sha256'] = sha(run/'plan.json')
        order = ([{'policy': p, 'round': 0} for p in ('eager','decode_graph')] if args.mode == 'check' else
                 [{'policy': p, 'round': rnd} for rnd in (1,2,3)
                  for p in (('decode_graph','eager') if rnd%2 else ('eager','decode_graph'))])
        save(run/'order.json', order)
        result['order_sha256'] = sha(run/'order.json')
        for item in order:
            label = item['policy'] if args.mode == 'check' else f"{item['policy']}-round{item['round']}"
            print('RUN '+label, flush=True)
            with (run/(label+'.log')).open('xb') as log:
                process = subprocess.run([sys.executable,'-B','-u',str(run/'runner.py'), '--mode',args.mode,
                    '--worker-run',str(run),'--policy',item['policy'],'--round',str(item['round'])],
                    cwd=args.project, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            require(process.returncode == 0, 'Worker failed; preserve '+label+'.log')
            record = read(run/label/'result.json')
            result['workers'].append(record)
            save(run/'result.json', result)
            if args.mode == 'timing':
                print(f"COLLECTED {label} median_ms={record['median_seconds']*1000:.6f}", flush=True)
        if args.mode == 'check':
            comparison = compare_logits(run, cfg['contract']['raw_expected_token_ids'])
            result.update(status='STABLE_RUNTIME_CORRECTNESS_PASS' if comparison['pass'] else 'STABLE_RUNTIME_CORRECTNESS_REJECTED',
                          comparison=comparison, fresh_workers=2)
        else:
            pairs = []
            for rnd in (1,2,3):
                records = {row['policy']: row for row in result['workers'] if row['round'] == rnd}
                eager, graph = [records[p]['median_seconds'] for p in ('eager','decode_graph')]
                pairs.append({'round': rnd, 'eager_ms': eager*1000, 'graph_ms': graph*1000,
                              'eager_over_graph': eager/graph, 'graph_latency_change_pct': (graph/eager-1)*100})
            result.update(status='STABLE_RUNTIME_TIMING_COLLECTION_COMPLETE', fresh_workers=6, sample_count=30,
                          pairs=pairs, all_rounds_at_least3pct_faster=all(p['graph_latency_change_pct'] <= -3 for p in pairs))
        result['method'] = cfg['method']
        result['limit'] = 'Performance comparison is against original Eager, not against original Graph or CUDA generally. Fixed128/32 workload, no service/AMD/broad quality claim.'
    except BaseException:
        primary = traceback.format_exc()
        result['error'] = primary
        (run/'controller-error.txt').write_text(primary, encoding='utf-8')
        raise
    finally:
        save(run/'result.json', result)
        shared = HERE/'logs'/('m08-stable-runtime-'+args.mode+'-results')/run.name
        try:
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run, shared)
            print('SHARED_RUN_DIRECTORY='+str(shared), flush=True)
        except BaseException:
            print('SHARED_COPY_FAILED; local evidence retained: '+str(run), flush=True)
            if primary is None:
                raise
    print('STATUS='+result['status'], flush=True)


if __name__ == '__main__':
    main()
