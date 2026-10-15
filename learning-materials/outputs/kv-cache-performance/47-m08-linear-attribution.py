"""One original Eager model trace mapping F.linear call sites to CUDA kernels.

Coach diagnostic only: two untimed warmups and one NVTX-instrumented generate.
The wrapper delegates unchanged F.linear math. Instrumentation is not a benchmark
sample and Eager duration shares must not be transferred to CUDA Graph execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = Path('/home/weltlaw/projects/mini-sglang-kv')
NSYS = Path('/opt/nvidia/nsight-systems/2025.1.3/bin/nsys')
HELPER = '39-m08-model-smoke.py'
ANALYZER = '47-analyze-linear-attribution.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metadata(tensor):
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'storage_offset': tensor.storage_offset(),
            'data_ptr': tensor.data_ptr()}


def linear_registry(model):
    registry = {}

    def add(module_path, component, layer, module, expected_shape, effective_weight=None):
        weight = module.weight if effective_weight is None else effective_weight
        require(id(weight) not in registry, 'Linear weight object unexpectedly shared by multiple call sites')
        require(tuple(weight.shape) == expected_shape and str(weight.dtype) == 'torch.float16' and weight.device.type == 'cuda',
                'Loaded projection shape/dtype/device differs: ' + module_path)
        require(module.bias is None, 'This Qwen3 projection was expected to have no bias')
        registry[id(weight)] = {'module_path': module_path, 'component': component, 'layer': layer,
            'weight': metadata(weight), 'weight_object_id': id(weight), 'bias': None,
            'effective_weight_path': 'model.embed_tokens.weight' if component == 'lm_head' else module_path + '.weight'}

    layers = model.model.layers.op_list
    require(len(layers) == 28, 'Expected 28 Qwen3 layers')
    for layer_id, layer in enumerate(layers):
        prefix = f'model.layers.{layer_id}'
        add(prefix + '.self_attn.qkv_proj', 'qkv_proj', layer_id, layer.self_attn.qkv_proj, (4096, 1024))
        add(prefix + '.self_attn.o_proj', 'o_proj', layer_id, layer.self_attn.o_proj, (1024, 2048))
        add(prefix + '.mlp.gate_up_proj', 'gate_up_proj', layer_id, layer.mlp.gate_up_proj, (6144, 1024))
        add(prefix + '.mlp.down_proj', 'down_proj', layer_id, layer.mlp.down_proj, (1024, 3072))
    require(model.lm_head.tied_embedding is model.model.embed_tokens, 'Expected tied word embedding/LM head')
    add('lm_head', 'lm_head', None, model.lm_head, (151936, 1024), model.lm_head.tied_embedding.weight)
    require(len(registry) == 113, 'Expected 112 decoder projections plus one LM head')
    return registry


def verify_call_sequence(records):
    require(len(records) == 3616, 'Expected 3616 actual F.linear calls')
    components = ('qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj')
    counts = Counter()
    for call_id, record in enumerate(records):
        step, slot = divmod(call_id, 113)
        phase = 'prefill' if step == 0 else 'decode'
        component = components[slot % 4] if slot < 112 else 'lm_head'
        layer = slot // 4 if slot < 112 else None
        rows = 128 if phase == 'prefill' and component != 'lm_head' else 1
        require(record['call_id'] == call_id and record['label'] == f'M08_LINEAR_{call_id:06d}', 'Call ID/label order differs')
        require(record['status'] == 'RETURNED' and record['step'] == step and record['phase'] == phase,
                'Actual raw-token step or phase differs from predicted single-request sequence')
        require(record['component'] == component and record['layer'] == layer, 'Actual projection ordering differs')
        require(record['input']['shape'][0] == record['output']['shape'][0] == rows, 'Actual linear row count differs')
        require(record['input']['shape'][1] == record['weight']['shape'][1] and
                record['output']['shape'][1] == record['weight']['shape'][0], 'Linear shape relation differs')
        require(record['input']['dtype'] == record['weight']['dtype'] == record['output']['dtype'] == 'torch.float16', 'Linear dtype differs')
        counts[(phase, component)] += 1
    require(sum(count for (phase, component), count in counts.items() if phase == 'prefill' and component != 'lm_head') == 112,
            'Prefill decoder linear count differs')
    require(sum(count for (phase, component), count in counts.items() if phase == 'decode' and component != 'lm_head') == 3472,
            'Decode decoder linear count differs')
    require(sum(count for (_, component), count in counts.items() if component == 'lm_head') == 32,
            'LM head count differs')
    return {f'{phase}/{component}': count for (phase, component), count in sorted(counts.items())}


def child(run):
    cfg = read(run / 'plan.json')
    for name, digest in cfg['frozen_files'].items():
        require(sha(run / name) == digest, 'Frozen diagnostic source changed: ' + name)
    for name, digest in cfg['source_manifest'].items():
        require(sha(run / 'source' / name) == digest, 'Frozen upstream source changed: ' + name)
    for name, digest in cfg['model']['metadata_sha256'].items():
        require(sha(Path(cfg['model']['path']) / name) == digest, 'Model metadata changed: ' + name)
    for item in cfg['model']['weights']:
        stat = (Path(cfg['model']['path']) / item['name']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Model checkpoint stat changed')
    sys.path.insert(0, str(run / 'source/python'))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MINISGL_DISABLE_OVERLAP_SCHEDULING='1')
    import torch
    import torch.nn.functional as F
    from minisgl.llm import LLM
    from minisgl.core import SamplingParams, get_global_ctx
    import minisgl.kernel as kernel_api
    from minisgl.kernel.store import store_cache
    require(kernel_api.store_cache is store_cache, 'Original KV backend has been replaced')
    original_linear = F.linear
    raw, records = [], []

    class CountedLLM(LLM):
        def offline_send_result(self, reply):
            raw.extend(int(msg.next_token) for msg in reply)
            return super().offline_send_result(reply)

    model, profiling = None, False
    result = {'status': 'FAILED', 'backend': 'original', 'policy': 'eager', 'not_timing_sample': True}
    try:
        options = dict(cfg['engine_options'], cuda_graph_bs=[])
        # Engine must initialize CUDA itself; do not probe a CUDA device earlier.
        model = CountedLLM(cfg['model']['path'], dtype=torch.float16, **options)
        require(model.engine.graph_runner.max_graph_bs == 0, 'Attribution needs Eager launches, not Graph replay')
        result['environment'] = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'packages': {name: importlib.metadata.version(name) for name in cfg['timing_packages']}}
        require(result['environment']['packages'] == cfg['timing_packages'], 'Package versions changed')
        require(all(result['environment'][key] == value for key, value in cfg['accepted_device'].items()),
                'GPU/Torch/CUDA differs from accepted 42')
        registry = linear_registry(model.engine.model)
        save(run / 'linear-registry.json', list(registry.values()))
        params = SamplingParams(temperature=0., ignore_eos=True, max_tokens=cfg['max_new_tokens'])
        for _ in range(2):
            raw.clear()
            model.generate([cfg['input_token_ids']], params)
            torch.cuda.synchronize()
            require(raw == cfg['raw_expected_token_ids'], 'Warmup token sequence differs from accepted original')

        def traced_linear(input, weight, bias=None):
            entry = registry.get(id(weight))
            require(entry is not None, 'F.linear used an unregistered weight object; preserve evidence')
            require(bias is None, 'Unexpected linear bias')
            ctx = get_global_ctx()
            phase = 'prefill' if ctx.batch.is_prefill else 'decode'
            caller = sys._getframe(1)
            call_id = len(records)
            record = {'call_id': call_id, 'label': f'M08_LINEAR_{call_id:06d}',
                'module_path': entry['module_path'], 'component': entry['component'], 'layer': entry['layer'],
                'phase': phase, 'step': len(raw), 'input': metadata(input), 'weight': metadata(weight),
                'bias': None, 'source_file': caller.f_code.co_filename, 'source_line': caller.f_lineno,
                'status': 'BEFORE_LINEAR'}
            del caller
            records.append(record)
            torch.cuda.nvtx.range_push(record['label'])
            try:
                output = original_linear(input, weight, bias)
            finally:
                torch.cuda.nvtx.range_pop()
            record['output'] = metadata(output)
            record['status'] = 'RETURNED'
            return output

        raw.clear()
        F.linear = traced_linear
        torch.cuda.profiler.start()
        profiling = True
        torch.cuda.nvtx.range_push('M08_ORIGINAL_EAGER_LINEAR_ATTRIBUTION')
        try:
            output = model.generate([cfg['input_token_ids']], params)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
            torch.cuda.profiler.stop()
            profiling = False
            F.linear = original_linear
        require(raw == cfg['raw_expected_token_ids'] and len(raw) == 32, 'Instrumented original output differs')
        counts = verify_call_sequence(records)
        result.update(status='LINEAR_ATTRIBUTION_WORKLOAD_CHECKED', raw_generated_token_ids=list(raw), output=output,
            engine_options=options, linear_calls=len(records), observed_phase_component_counts=counts,
            expected_counts={'decoder_prefill': 112, 'decoder_decode': 3472, 'lm_head_total': 32},
            observation='F.linear wrapper delegates original callable once; input/weight/output metadata only, no tensor-value copy or added synchronization in each range.',
            limitations='One instrumented Eager generate after two warmups; exact original token output, not full KV/logits diagnostic. No Graph duration attribution or unprofiled speed claim.')
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        F.linear = original_linear
        if profiling:
            try:
                torch.cuda.profiler.stop()
            except BaseException:
                result['profiler_stop_error'] = traceback.format_exc()
        save(run / 'linear-calls.json', records)
        if model is not None:
            try:
                model.shutdown()
            except BaseException:
                result['shutdown_error'] = traceback.format_exc()
                result['status'] = 'FAILED'
        save(run / 'workload.json', result)
    require(result['status'] == 'LINEAR_ATTRIBUTION_WORKLOAD_CHECKED', 'Workload cleanup failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-run', type=Path)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child.resolve())
        return
    if args.timing_run is None:
        parser.error('--timing-run is required (accepted original42 run)')
    project = args.project.resolve(strict=True)
    parent = project / 'experiments/m08-linear-attribution-results'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('RUN_DIRECTORY=' + str(run), flush=True)
    result = {'status': 'FAILED', 'policy': 'eager', 'backend': 'original', 'not_timing_sample': True}
    primary = None
    try:
        timing = args.timing_run.resolve(strict=True)
        accepted_summary, cfg = read(timing / 'summary.json'), read(timing / 'plan.json')
        require(accepted_summary['status'] == 'MODEL_TIMING_COLLECTION_COMPLETE' and len(accepted_summary['workers']) == 12,
                'Use completed accepted42 timing run')
        require(len(cfg['input_token_ids']) == 128 and cfg['max_new_tokens'] == 32 and len(cfg['raw_expected_token_ids']) == 32,
                'Unexpected accepted42 workload')
        require(sha(HERE / HELPER) == sha(timing / 'smoke-helper.py'), '39 helper differs from accepted42 snapshot')
        require(HERE.joinpath(ANALYZER).is_file(), 'Prepared independent SQL analyzer is required')
        require({name: importlib.metadata.version(name) for name in cfg['timing_packages']} == cfg['timing_packages'],
                'Current package versions differ from accepted42')
        helper = load(HERE / HELPER, 'linear_attribution_snapshot_helper')
        sources = helper.snapshot_source(project, run)
        require(sources == cfg['source_manifest'], 'Current original source differs from accepted42')
        (run / 'runner.py').rename(run / HELPER)
        shutil.copyfile(__file__, run / 'm08_linear_worker.py')
        shutil.copyfile(HERE / '47-m08-linear-attribution.sh', run / '47-m08-linear-attribution.sh')
        shutil.copyfile(HERE / ANALYZER, run / ANALYZER)
        model = helper.inspect_model(Path(cfg['model']['path']), cfg['model']['revision'])
        require([(x['name'], x['sha256']) for x in model['weights']] == [(x['name'], x['sha256']) for x in cfg['model']['weights']],
                'Model weights differ from accepted42')
        require(model['metadata_sha256'] == cfg['model']['metadata_sha256'], 'Model/tokenizer metadata differs')
        original_env = next(w['environment'] for w in accepted_summary['workers'] if w['arm'] == 'original')
        cfg.update(model=model, accepted_device={key: original_env[key] for key in ('gpu', 'torch', 'cuda')},
            accepted_timing=str(timing), accepted_timing_plan_sha256=sha(timing / 'plan.json'),
            accepted_timing_summary_sha256=sha(timing / 'summary.json'), attribution_policy='eager',
            source_manifest=sources, authored_by='coach diagnostic; no independent learner mastery inferred',
            frozen_files={name: sha(run / name) for name in (HELPER, 'm08_linear_worker.py',
                                                          '47-m08-linear-attribution.sh', ANALYZER, 'source-manifest.json')})
        save(run / 'plan.json', cfg)
        shutil.copyfile(timing / 'plan.json', run / 'accepted-timing-plan.json')
        shutil.copyfile(timing / 'summary.json', run / 'accepted-timing-summary.json')
        config = Path.home() / '.config/NVIDIA Corporation/nsys-config.ini'
        if config.exists():
            shutil.copyfile(config, run / 'nsys-config.ini')
        prefix = run / 'original-model'
        command = [str(NSYS), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--cuda-event-trace=false', '--capture-range=cudaProfilerApi', '--capture-range-end=stop',
                   '--output=' + str(prefix), sys.executable, '-B', '-u', str(run / 'm08_linear_worker.py'), '--child', str(run)]
        result.update(command=command, plan_sha256=sha(run / 'plan.json'))
        with (run / 'nsys-profile.log').open('xb') as log:
            process = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        result['profile_returncode'] = process.returncode
        require(process.returncode == 0, 'Nsight workload failed; raw log retained')
        require(read(run / 'workload.json')['status'] == 'LINEAR_ATTRIBUTION_WORKLOAD_CHECKED', 'Profile workload did not pass')
        with (run / 'nsys-kernels.csv').open('xb') as log:
            process = subprocess.run([str(NSYS), 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
                                      str(prefix.with_suffix('.nsys-rep'))], stdout=log, stderr=subprocess.STDOUT, timeout=180)
        require(process.returncode == 0 and prefix.with_suffix('.sqlite').is_file(), 'Nsight SQLite export failed')
        require(sha(run / ANALYZER) == cfg['frozen_files'][ANALYZER], 'Frozen analyzer changed')
        analysis = load(run / ANALYZER, 'saved_linear_attribution_analyzer').analyze(run)
        result.update(status='LINEAR_ATTRIBUTION_COMPLETE', attribution=analysis,
            artifact_sha256={name: sha(run / name) for name in ('workload.json', 'linear-calls.json', 'linear-registry.json', 'original-model.sqlite')},
            limit='One instrumented Eager original workload; direct launch-to-callsite attribution only. No CUDA Graph share transfer, wall-clock critical path, bandwidth, occupancy or performance improvement claim.')
    except BaseException:
        primary = traceback.format_exc()
        result['error'] = primary
        (run / 'controller-error.txt').write_text(primary, encoding='utf-8')
        raise
    finally:
        save(run / 'result.json', result)
        shared = HERE / 'logs/m08-linear-attribution-results' / run.name
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
    try:
        main()
    except BaseException:
        try:
            parent = HERE / 'logs/m08-linear-attribution-errors'
            parent.mkdir(parents=True, exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix='error-', dir=parent))
            save(out / 'error.json', {'argv': sys.argv, 'runner_sha256': sha(__file__), 'error': traceback.format_exc()})
        except BaseException:
            pass
        raise
