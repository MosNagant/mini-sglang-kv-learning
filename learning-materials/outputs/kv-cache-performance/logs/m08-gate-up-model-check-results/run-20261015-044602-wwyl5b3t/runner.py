"""Gated real-model validation/timing for one frozen Gate/Up candidate.

Coach implementation. Original KV and prefill MLP remain unchanged. A selected
single-token decode Gate/Up path uses candidate.prepare/call_with_input, then
the original down projection. GEMV-only candidates retain the original SiLU
activation. Fused candidates return the activation output. This is an explicit
research patch, never a permanent default change or a model-quality claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
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
HELPER = '39-m08-model-smoke.py'
ATOL = 0.002
RTOL = 0.002


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen(run):
    cfg = read(run / 'plan.json')
    for name, digest in cfg['frozen_files'].items():
        require(sha(run / name) == digest, 'Frozen helper changed: ' + name)
    for name, digest in cfg['source_manifest'].items():
        require(sha(run / 'source' / name) == digest, 'Frozen original source changed: ' + name)
    require({name: importlib.metadata.version(name) for name in cfg['timing_packages']} == cfg['timing_packages'], 'Package environment changed')
    for name, digest in cfg['model']['metadata_sha256'].items():
        require(sha(Path(cfg['model']['path']) / name) == digest, 'Model/tokenizer metadata changed: ' + name)
    for item in cfg['model']['weights']:
        stat = (Path(cfg['model']['path']) / item['name']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Model weights changed after full hash')
    return cfg


def install_decode_candidate(candidate, name):
    """Patch before Engine constructs/captures; return state and restoration."""
    import torch
    from minisgl.core import get_global_ctx
    from minisgl.models.utils import GatedMLP
    original = GatedMLP.forward
    specs = {spec['name']: spec for spec in candidate.candidates()}
    require(name in specs, 'Selected candidate absent from frozen candidate source')
    state = {'observe': True, 'plans': {}, 'instances': {}, 'decode_python_calls': 0,
             'capture_python_calls': 0, 'prefill_original_python_calls': 0,
             'original_decode_fallback_calls': 0, 'current_input_calls': 0}

    def forward(self, x):
        ctx = get_global_ctx()
        if ctx.batch.is_prefill:
            if state['observe']:
                state['prefill_original_python_calls'] += 1
            return original(self, x)
        require(ctx.batch.is_decode and x.shape == (1, 1024), 'Selected model candidate supports one-token decode only')
        key = id(self)
        prepared = state['plans'].get(key)
        if prepared is None:
            require(not torch.cuda.is_current_stream_capturing(), 'First candidate prepare occurred inside Graph capture')
            require(self.gate_up_proj.bias is None, 'Selected Gate/Up path requires no bias')
            prepared = candidate.prepare(name, x, self.gate_up_proj.weight)
            state['plans'][key] = prepared
            state['instances'][key] = self
            prepared['original_weight_object'] = self.gate_up_proj.weight
            prepared['original_weight_pointer'] = self.gate_up_proj.weight.data_ptr()
        require(prepared['original_weight_object'] is self.gate_up_proj.weight and
                prepared['original_weight_pointer'] == self.gate_up_proj.weight.data_ptr(),
                'Layer weight identity changed; rebuild immutable-weight candidate plan')
        # Critical: pass CURRENT x, never use prepare()'s retained initial x.
        value = prepared['call_with_input'](x)
        if state['observe']:
            state['decode_python_calls'] += 1
            state['current_input_calls'] += 1
            if torch.cuda.is_current_stream_capturing():
                state['capture_python_calls'] += 1
        activated = self.act_fn(value) if prepared['kind'] == 'gemv' else value
        return self.down_proj.forward(activated)

    GatedMLP.forward = forward

    def restore():
        GatedMLP.forward = original

    return state, restore


def candidate_evidence(state, model, out, policy):
    """Save real compiled objects and Python routing counts outside timing."""
    layers = model.engine.model.model.layers.op_list
    require(len(layers) == 28 and len(state['plans']) == 28, 'Candidate did not prepare all 28 actual model layers')
    require(state['decode_python_calls'] == state['current_input_calls'] > 0, 'Current-input candidate calls missing')
    require(state['original_decode_fallback_calls'] == 0, 'Candidate used an original decode fallback')
    if policy == 'decode_graph':
        require(state['capture_python_calls'] == 28, 'Expected one captured candidate call per layer for Graph B1')
    else:
        require(state['capture_python_calls'] == 0, 'Unexpected graph capture in Eager')
    compiled_dir = out / 'compiled'
    compiled_dir.mkdir(exist_ok=True)
    artifacts, records = {}, []
    for layer_id, layer in enumerate(layers):
        prepared = state['plans'].get(id(layer.mlp))
        require(prepared is not None, 'Prepared candidate does not map to actual layer')
        kernels = []
        for compiled in prepared['compiled_kernels']:
            require(compiled is not None, 'Candidate device kernel was not compiled/executed')
            key = str(getattr(compiled, 'hash', compiled.name))
            if key not in artifacts:
                entry = {'name': compiled.name, 'hash': key, 'files': {}}
                for kind in ('ptx', 'ttgir'):
                    path = compiled_dir / f'{key}.{kind}'
                    path.write_text(compiled.asm[kind], encoding='utf-8')
                    entry['files'][kind] = {'path': path.relative_to(out).as_posix(), 'sha256': sha(path)}
                artifacts[key] = entry
            kernels.append(key)
        records.append({'layer': layer_id, 'candidate': prepared['name'], 'kind': prepared['kind'],
            'workspace_bytes': prepared['workspace_bytes'], 'prepack_bytes': prepared['prepack_bytes'],
            'setup_seconds': prepared['setup_seconds'], 'kernel_launches_per_call': prepared['kernel_launches_per_call'],
            'kernel_hashes': kernels, 'input_policy': 'call_with_input(current x)',
            'weight_shape': list(layer.mlp.gate_up_proj.weight.shape)})
    counters = {key: state[key] for key in ('decode_python_calls', 'capture_python_calls',
                'prefill_original_python_calls', 'original_decode_fallback_calls', 'current_input_calls')}
    return {'counters': counters, 'layers': records, 'compiled_artifacts': artifacts,
            'scope': 'Python observations including capture/warmup; not Graph replay node counts or hardware counters. Counter observation is disabled before timing samples.'}


def worker(run, arm, mode, policy, round_id):
    import numpy as np
    cfg = verify_frozen(run)
    out = run / (arm if mode == 'check' else f'{policy}-round{round_id}-{arm}')
    out.mkdir()
    result = {'status': 'FAILED', 'arm': arm, 'mode': mode, 'policy': policy, 'round': round_id, 'samples': []}
    sys.path.insert(0, str(run / 'source/python'))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MINISGL_DISABLE_OVERLAP_SCHEDULING='1')
    import torch
    from minisgl.llm import LLM
    from minisgl.core import SamplingParams
    import minisgl.kernel as kernel_api
    from minisgl.kernel.store import store_cache
    require(kernel_api.store_cache is store_cache, 'Original KV backend was replaced')
    raw, logits = [], []
    state, restore = None, None
    model = None

    class CountedLLM(LLM):
        def offline_send_result(self, reply):
            raw.extend(int(msg.next_token) for msg in reply)
            return super().offline_send_result(reply)

    try:
        if arm == 'candidate':
            candidate = load(run / 'candidate.py', 'selected_gate_up_candidate')
            state, restore = install_decode_candidate(candidate, cfg['candidate_name'])
        options = dict(cfg['engine_options'], cuda_graph_bs=[1] if policy == 'decode_graph' else [])
        # Engine owns first CUDA initialization and calls one eager model warmup
        # before entering its Graph capture. Lazy plans must be ready by capture.
        model = CountedLLM(cfg['model']['path'], dtype=torch.float16, **options)
        require(bool(model.engine.graph_runner.max_graph_bs) == (policy == 'decode_graph'), 'Actual Graph policy differs')
        result['environment'] = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'packages': {name: importlib.metadata.version(name) for name in cfg['timing_packages']},
            'engine_options': options}
        require(all(result['environment'][key] == value for key, value in cfg['accepted_device'].items()), 'Accepted GPU environment changed')
        params = SamplingParams(temperature=0., ignore_eos=True, max_tokens=cfg['max_new_tokens'])

        def generate():
            raw.clear()
            return model.generate([cfg['input_token_ids']], params)

        if mode == 'check':
            require(policy == 'eager', 'Strict all-logits diagnostic is Eager only')
            original_sample = model.engine.sampler.sample

            def sample(observed_logits, args):
                require(observed_logits.shape == (1, 151936) and observed_logits.dtype == torch.float16,
                        'Unexpected real-model logits shape/dtype')
                logits.append(observed_logits.detach().clone())
                return original_sample(observed_logits, args)

            model.engine.sampler.sample = sample
            output = generate()
            torch.cuda.synchronize()
            require(len(logits) == len(raw) == 32, 'Incomplete strict output/logits sequence')
            arrays = {f'step{step:02d}': value.cpu().numpy() for step, value in enumerate(logits)}
            np.savez(out / 'all-logits.npz', **arrays)
            if state is not None:
                require(state['decode_python_calls'] == 868 and state['prefill_original_python_calls'] == 28,
                        'Strict candidate did not replace exactly all 868 decode Gate/Up calls')
                result['candidate_evidence'] = candidate_evidence(state, model, out, policy)
            result.update(status='MODEL_ALL_LOGITS_COLLECTED', raw_generated_token_ids=list(raw), output=output,
                logits_sha256=sha(out / 'all-logits.npz'), logits_count=32,
                all_logits_finite=all(bool(np.isfinite(value).all()) for value in arrays.values()),
                input_token_ids=cfg['input_token_ids'], numerical_contract={'atol': ATOL, 'rtol': RTOL, 'every_element': True},
                limit='All 32 logits saved in an Eager diagnostic; candidate acceptance is a later cross-process comparison, not assumed here.')
        else:
            for _ in range(2):
                generate()
                torch.cuda.synchronize()
                require(raw == cfg['raw_expected_token_ids'], 'Warmup tokens differ from strict accepted sequence')
            if state is not None:
                result['candidate_evidence'] = candidate_evidence(state, model, out, policy)
                state['observe'] = False
            torch.cuda.reset_peak_memory_stats()
            for index in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = generate()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                result['samples'].append({'sample': index, 'seconds': seconds,
                    'raw_generated_token_ids': list(raw), 'raw_generated_steps': len(raw), 'output': output,
                    'raw_output_tokens_per_second': len(raw)/seconds if seconds > 0 else None})
                save(out / 'result.json', result)
                require(math.isfinite(seconds) and seconds > 0 and raw == cfg['raw_expected_token_ids'], 'Timed duration/output invalid')
            result.update(status='GATE_UP_MODEL_TIMING_WORKER_COMPLETE',
                median_seconds=statistics.median(x['seconds'] for x in result['samples']),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                measurement='perf_counter around generate+GPU completion; scheduler/sampling/D2H/raw-token-record/detokenize included; no model loading/JIT/setup/warmup/network')
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        if restore is not None:
            restore()
        if model is not None:
            try:
                model.shutdown()
            except BaseException:
                result['shutdown_error'] = traceback.format_exc()
                result['status'] = 'FAILED'
        save(out / 'result.json', result)
    require(result['status'] in ('MODEL_ALL_LOGITS_COLLECTED', 'GATE_UP_MODEL_TIMING_WORKER_COMPLETE'), 'Worker cleanup failed')


def compare_all_logits(run, expected):
    import numpy as np
    baseline, candidate = [read(run / arm / 'result.json') for arm in ('original', 'candidate')]
    require(baseline['status'] == candidate['status'] == 'MODEL_ALL_LOGITS_COLLECTED', 'Strict workers incomplete')
    require(baseline['raw_generated_token_ids'] == expected, 'Fresh original differs from accepted original tokens')
    steps = []
    with np.load(run / 'original/all-logits.npz', allow_pickle=False) as left, np.load(run / 'candidate/all-logits.npz', allow_pickle=False) as right:
        require(set(left.files) == set(right.files) == {f'step{i:02d}' for i in range(32)}, 'All-step logit set differs')
        for step in range(32):
            a, b = left[f'step{step:02d}'], right[f'step{step:02d}']
            require(a.shape == b.shape == (1, 151936) and a.dtype == b.dtype == np.float16, 'Saved logit metadata differs')
            av, bv = a.astype(np.float32), b.astype(np.float32)
            finite = bool(np.isfinite(av).all() and np.isfinite(bv).all())
            delta = np.abs(av-bv)
            allowed = ATOL + RTOL*np.abs(av)
            failed = ~(np.isfinite(av) & np.isfinite(bv) & (delta <= allowed))
            steps.append({'step': step, 'finite': finite, 'elements': a.size,
                'violations': int(np.count_nonzero(failed)),
                'max_abs_error': float(delta.max()) if finite else None,
                'max_allowed_error_ratio': float((delta/allowed).max()) if finite else None})
    tokens_equal = candidate['raw_generated_token_ids'] == expected
    numerical_pass = all(x['finite'] and x['violations'] == 0 for x in steps)
    return {'atol': ATOL, 'rtol': RTOL, 'comparison': 'abs(candidate-reference) <= atol + rtol*abs(reference), every saved FP16 logit element',
            'steps': steps, 'all_logits_finite_and_within_tolerance': numerical_pass, 'exact_32_tokens_equal': tokens_equal,
            'elements_checked': sum(x['elements'] for x in steps), 'violations': sum(x['violations'] for x in steps),
            'pass': numerical_pass and tokens_equal,
            'limit': 'One fixed autoregressive workload; exact tokens plus elementwise logits tolerance do not establish model quality or broad numerical equivalence.'}


def prepare_run(args, run):
    fixture_run = args.fixture_run.resolve(strict=True)
    fixture, fixture_result, fixture_plan = [read(fixture_run / name) for name in ('fixture.json', 'result.json', 'plan.json')]
    require(fixture_result['status'] == fixture['status'] == 'GATE_UP_FIXTURE_CAPTURE_PASS', 'Real fixture capture not accepted')
    require(sha(fixture_run / 'fixture.json') == fixture_result['fixture_sha256'], 'Accepted fixture manifest changed')
    require(sha(fixture_run / 'plan.json') == fixture_result['plan_sha256'] == fixture['plan_sha256'],
            'Accepted fixture plan changed')
    micro_run = args.accepted_micro_run.resolve(strict=True)
    micro = read(micro_run / 'result.json')
    require(micro['status'] == 'GATE_UP_CAMPAIGN_COMPLETE', '49 microcampaign not complete')
    require(micro['fixture_sha256'] == sha(fixture_run / 'fixture.json'), '49 was run on different fixture evidence')
    require(micro['candidate_source_sha256'] == sha(args.candidate_source), 'Candidate source differs from49')
    require(args.candidate_name in micro['numerical_eligible'] and args.candidate_name == micro['selected_candidate'],
            'Only the frozen numerically eligible selected49 candidate may reach model checks')
    require(micro['holdout']['status'] == 'PASS', 'Selected candidate holdout checks did not pass')
    require(micro['selection']['strong_local_gain'] or args.investigator_candidate,
            'Local gain not strong; explicit --investigator-candidate is required for a numerical-pass-only investigation')
    require(sha(HERE / HELPER) == fixture_plan['frozen_files'][HELPER], '39 helper changed since fixture capture')
    require({name: importlib.metadata.version(name) for name in fixture_plan['timing_packages']} == fixture_plan['timing_packages'], 'Package environment changed')
    helper = load(HERE / HELPER, 'gate_up_model_snapshot_helper')
    sources = helper.snapshot_source(args.project, run)
    require(sources == fixture_plan['source_manifest'], 'Original source changed since fixture capture')
    (run / 'runner.py').rename(run / HELPER)
    shutil.copyfile(__file__, run / 'runner.py')
    shutil.copyfile(HERE / '51-m08-gate-up-model.sh', run / '51-m08-gate-up-model.sh')
    shutil.copyfile(args.candidate_source, run / 'candidate.py')
    require(sha(run / 'candidate.py') == micro['candidate_source_sha256'], 'Candidate changed during snapshot')
    model = helper.inspect_model(Path(fixture_plan['model']['path']), fixture_plan['model']['revision'])
    require([(x['name'], x['sha256']) for x in model['weights']] == [(x['name'], x['sha256']) for x in fixture_plan['model']['weights']], 'Model weights changed')
    require(model['metadata_sha256'] == fixture_plan['model']['metadata_sha256'], 'Model/tokenizer metadata changed')
    cfg = dict(fixture_plan)
    cfg.update(model=model, source_manifest=sources, mode=args.mode, candidate_name=args.candidate_name,
        candidate_source_sha256=sha(run / 'candidate.py'), fixture_run=str(fixture_run), fixture_sha256=sha(fixture_run / 'fixture.json'),
        accepted_micro_run=str(micro_run), accepted_micro_result_sha256=sha(micro_run / 'result.json'),
        investigator_candidate=args.investigator_candidate, numerical_contract={'atol': ATOL, 'rtol': RTOL, 'exact_tokens': True},
        frozen_files={name: sha(run / name) for name in (HELPER, 'runner.py', 'candidate.py', '51-m08-gate-up-model.sh', 'source-manifest.json')},
        authorship='Coach integration/experiment; independent learner mastery not inferred',
        changed_scope='Decode Gate/Up + optional activation only; original prefill, down projection, KV backend and other model math remain intact.',
        signal_rule='All three matched process median latencies at least3% lower is a scoped engineering signal, not statistical significance; keep every outcome.')
    shutil.copyfile(micro_run / 'result.json', run / 'accepted-micro-result.json')
    shutil.copyfile(fixture_run / 'fixture.json', run / 'accepted-fixture.json')
    shutil.copyfile(fixture_run / 'plan.json', run / 'accepted-fixture-plan.json')
    require(read(run / 'accepted-micro-result.json') == micro, 'Microcampaign changed during snapshot')
    require(sha(run / 'accepted-fixture.json') == cfg['fixture_sha256'], 'Fixture changed during snapshot')
    require(sha(run / 'accepted-fixture-plan.json') == fixture['plan_sha256'], 'Fixture plan changed during snapshot')
    cfg['frozen_files'].update({name: sha(run / name) for name in
        ('accepted-micro-result.json', 'accepted-fixture.json', 'accepted-fixture-plan.json')})
    if args.mode == 'timing':
        require(args.accepted_model_check is not None, 'Timing requires --accepted-model-check')
        accepted = args.accepted_model_check.resolve(strict=True)
        checked, checked_cfg = read(accepted / 'result.json'), read(accepted / 'plan.json')
        require(checked['status'] == 'GATE_UP_MODEL_CORRECTNESS_PASS' and checked['comparison']['pass'], 'Strict model check did not pass')
        require(checked['plan_sha256'] == sha(accepted / 'plan.json'), 'Accepted model-check plan changed')
        for key in ('candidate_name', 'candidate_source_sha256', 'source_manifest', 'fixture_sha256', 'accepted_micro_result_sha256', 'numerical_contract'):
            require(checked_cfg[key] == cfg[key], 'Accepted model check differs: ' + key)
        require(checked_cfg['frozen_files']['runner.py'] == sha(run / 'runner.py'), 'Integration runner changed after model correctness')
        for arm in ('original', 'candidate'):
            saved = read(accepted / arm / 'result.json')
            require(sha(accepted / arm / 'all-logits.npz') == saved['logits_sha256'], 'Accepted logit evidence changed')
            require(saved in checked['workers'], 'Accepted worker record differs from saved controller record')
        require(compare_all_logits(accepted, cfg['raw_expected_token_ids']) == checked['comparison'],
                'Recomputed accepted model comparison differs')
        cfg.update(accepted_model_check=str(accepted), accepted_model_check_result_sha256=sha(accepted / 'result.json'))
        shutil.copyfile(accepted / 'result.json', run / 'accepted-model-check.json')
        cfg['frozen_files']['accepted-model-check.json'] = sha(run / 'accepted-model-check.json')
    save(run / 'plan.json', cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'timing'))
    parser.add_argument('--fixture-run', type=Path)
    parser.add_argument('--accepted-micro-run', type=Path)
    parser.add_argument('--candidate-name')
    parser.add_argument('--candidate-source', type=Path)
    parser.add_argument('--investigator-candidate', action='store_true')
    parser.add_argument('--accepted-model-check', type=Path)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--worker-run', type=Path)
    parser.add_argument('--arm', choices=('original', 'candidate'))
    parser.add_argument('--policy', choices=('eager', 'decode_graph'), default='eager')
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()
    if args.worker_run:
        require(args.mode is not None and args.arm is not None, 'Worker needs mode/arm')
        worker(args.worker_run.resolve(), args.arm, args.mode, args.policy, args.round)
        return
    if not all((args.mode, args.fixture_run, args.accepted_micro_run, args.candidate_name, args.candidate_source)):
        parser.error('--mode, --fixture-run, --accepted-micro-run, --candidate-name and --candidate-source are required')
    args.project = args.project.resolve(strict=True)
    parent = args.project / 'experiments' / ('m08-gate-up-model-' + args.mode)
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('RUN_DIRECTORY=' + str(run), flush=True)
    result, primary = {'status': 'FAILED', 'mode': args.mode, 'workers': []}, None
    try:
        cfg = prepare_run(args, run)
        result['plan_sha256'] = sha(run / 'plan.json')
        if args.mode == 'check':
            order = [{'arm': arm, 'policy': 'eager', 'round': 0} for arm in ('original', 'candidate')]
        else:
            order = []
            for rnd in (1, 2, 3):
                policies = ('decode_graph', 'eager') if rnd % 2 else ('eager', 'decode_graph')
                arms = ('original', 'candidate') if rnd % 2 else ('candidate', 'original')
                order += [{'arm': arm, 'policy': policy, 'round': rnd} for policy in policies for arm in arms]
        save(run / 'order.json', order)
        result['order_sha256'] = sha(run / 'order.json')
        for item in order:
            label = item['arm'] if args.mode == 'check' else f"{item['policy']}-round{item['round']}-{item['arm']}"
            print('RUN ' + label, flush=True)
            with (run / (label + '.log')).open('xb') as log:
                process = subprocess.run([sys.executable, '-B', '-u', str(run / 'runner.py'), '--worker-run', str(run),
                    '--mode', args.mode, '--arm', item['arm'], '--policy', item['policy'], '--round', str(item['round'])],
                    cwd=args.project, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            require(process.returncode == 0, 'Model worker failed; preserve ' + label + '.log')
            entry = read(run / label / 'result.json')
            result['workers'].append(entry)
            save(run / 'result.json', result)
        if args.mode == 'check':
            comparison = compare_all_logits(run, cfg['raw_expected_token_ids'])
            result.update(status='GATE_UP_MODEL_CORRECTNESS_PASS' if comparison['pass'] else 'GATE_UP_MODEL_CORRECTNESS_REJECTED',
                comparison=comparison, fresh_workers=2, raw_expected_token_ids=cfg['raw_expected_token_ids'],
                candidate_name=cfg['candidate_name'], candidate_source_sha256=cfg['candidate_source_sha256'])
        else:
            comparisons = {}
            for policy in ('eager', 'decode_graph'):
                pairs = []
                for rnd in (1, 2, 3):
                    entries = {x['arm']: x for x in result['workers'] if x['policy'] == policy and x['round'] == rnd}
                    a, b = [entries[arm]['median_seconds'] for arm in ('original', 'candidate')]
                    pairs.append({'round': rnd, 'original_ms': a*1000, 'candidate_ms': b*1000,
                                  'original_over_candidate': a/b, 'candidate_time_change_pct': (b/a-1)*100})
                comparisons[policy] = {'pairs': pairs, 'all_rounds_at_least3pct_faster': all(p['candidate_time_change_pct'] <= -3 for p in pairs)}
            result.update(status='GATE_UP_MODEL_TIMING_COLLECTION_COMPLETE', comparisons=comparisons,
                          fresh_workers=12, samples=60, candidate_name=cfg['candidate_name'])
        result['limit'] = 'Original KV unchanged. Fixed real-model workload only. Full logits checks are Eager; timed Graph replays check exact generated tokens. No general model-quality, service or AMD claim.'
    except BaseException:
        primary = traceback.format_exc()
        result['error'] = primary
        (run / 'controller-error.txt').write_text(primary, encoding='utf-8')
        raise
    finally:
        save(run / 'result.json', result)
        shared = HERE / 'logs' / ('m08-gate-up-model-' + args.mode + '-results') / run.name
        try:
            shared.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run, shared)
            print('SHARED_RUN_DIRECTORY=' + str(shared), flush=True)
        except BaseException:
            print('SHARED_COPY_FAILED; local evidence retained: ' + str(run), flush=True)
            if primary is None:
                raise
    print('STATUS=' + result['status'], flush=True)


if __name__ == '__main__':
    main()
