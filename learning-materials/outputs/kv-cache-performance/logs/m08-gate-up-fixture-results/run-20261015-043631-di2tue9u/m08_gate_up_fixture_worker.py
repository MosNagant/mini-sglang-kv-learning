"""Capture original Qwen3 decode Gate/Up + SiLU fixtures for controlled experiments.

Coach diagnostic; no optimized math, timing or default backend change. The three
selected zero-based generation steps are 1 (first decode), 16 and 31. All 28
layers are retained. GPU clones are taken in the actual engine stream; NumPy
files are written only after the complete generation and token checks finish.
"""
from __future__ import annotations

import argparse
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
HELPER = '39-m08-model-smoke.py'
CAPTURE_STEPS = (1, 16, 31)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


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
            'device': str(tensor.device), 'storage_offset': tensor.storage_offset(), 'data_ptr': tensor.data_ptr()}


def child(run):
    import numpy as np
    cfg = read(run / 'plan.json')
    for name, digest in cfg['frozen_files'].items():
        require(sha(run / name) == digest, 'Frozen diagnostic source differs: ' + name)
    for name, digest in cfg['source_manifest'].items():
        require(sha(run / 'source' / name) == digest, 'Frozen original source differs: ' + name)
    for name, digest in cfg['model']['metadata_sha256'].items():
        require(sha(Path(cfg['model']['path']) / name) == digest, 'Model metadata differs: ' + name)
    for item in cfg['model']['weights']:
        stat = (Path(cfg['model']['path']) / item['name']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['bytes'], item['mtime_ns']), 'Model weight stat differs')
    sys.path.insert(0, str(run / 'source/python'))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MINISGL_DISABLE_OVERLAP_SCHEDULING='1')
    import torch
    from minisgl.llm import LLM
    from minisgl.core import SamplingParams, get_global_ctx
    import minisgl.kernel as kernel_api
    from minisgl.kernel.store import store_cache
    require(kernel_api.store_cache is store_cache, 'Original KV backend was replaced')
    raw, pending, captures, saved_methods = [], {}, {}, []
    weight_clones, weight_objects = {}, {}
    gate_calls = {layer: 0 for layer in range(28)}
    activation_calls = {layer: 0 for layer in range(28)}

    class CountedLLM(LLM):
        def offline_send_result(self, reply):
            raw.extend(int(msg.next_token) for msg in reply)
            return super().offline_send_result(reply)

    model = None
    fixture = {'status': 'FAILED', 'backend': 'original', 'policy': 'eager', 'not_timing_sample': True,
               'capture_steps_zero_based': list(CAPTURE_STEPS), 'weights': [], 'samples': []}
    primary = None
    try:
        options = dict(cfg['engine_options'], cuda_graph_bs=[])
        # Engine must own CUDA initialization, as in accepted 41/42/47 workers.
        model = CountedLLM(cfg['model']['path'], dtype=torch.float16, **options)
        require(model.engine.graph_runner.max_graph_bs == 0, 'Fixture capture must use Eager')
        environment = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'packages': {name: importlib.metadata.version(name) for name in cfg['timing_packages']}}
        require(environment['packages'] == cfg['timing_packages'], 'Package versions differ')
        require(all(environment[k] == v for k, v in cfg['accepted_device'].items()), 'Accepted GPU/Torch/CUDA environment differs')
        fixture.update(environment=environment, engine_options=options, model=cfg['model'],
                       input_token_ids=cfg['input_token_ids'], expected_raw_token_ids=cfg['raw_expected_token_ids'],
                       plan_sha256=sha(run / 'plan.json'))
        layers = model.engine.model.model.layers.op_list
        require(len(layers) == 28, 'Expected 28 real Qwen3 layers')
        for layer_id, layer in enumerate(layers):
            weight = layer.mlp.gate_up_proj.weight
            require(tuple(weight.shape) == (6144, 1024) and weight.dtype == torch.float16 and weight.device.type == 'cuda',
                    'Gate/Up weight layout differs')
            require(layer.mlp.gate_up_proj.bias is None, 'Unexpected Gate/Up bias')
            weight_objects[layer_id] = weight
            # Real weight clones retained for read-only equality and export.
            weight_clones[layer_id] = weight.detach().clone()
        torch.cuda.synchronize()
        params = SamplingParams(temperature=0., ignore_eos=True, max_tokens=cfg['max_new_tokens'])
        for _ in range(2):
            raw.clear()
            model.generate([cfg['input_token_ids']], params)
            torch.cuda.synchronize()
            require(raw == cfg['raw_expected_token_ids'], 'Warmup output differs from accepted original')

        def bind(layer_id, mlp):
            gate = mlp.gate_up_proj
            original_forward, original_act = gate.forward, mlp.act_fn
            saved_methods.append((gate, 'forward', 'forward' in gate.__dict__, original_forward))
            saved_methods.append((mlp, 'act_fn', 'act_fn' in mlp.__dict__, original_act))

            def forward(input):
                gate_calls[layer_id] += 1
                ctx = get_global_ctx()
                step = len(raw)
                selected = step in CAPTURE_STEPS
                snapshot = None
                if selected:
                    require(ctx.batch.is_decode and tuple(input.shape) == (1, 1024), 'Selected call is not one-token decode')
                    require((step, layer_id) not in captures and layer_id not in pending, 'Duplicate/nested capture')
                    snapshot = {'step': step, 'layer': layer_id, 'phase': 'decode',
                        'module_path': f'model.layers.{layer_id}.mlp.gate_up_proj',
                        'input_metadata': metadata(input), 'weight_metadata': metadata(gate.weight),
                        'x': input.detach().clone(), 'token_ids': ctx.batch.input_ids.detach().clone()}
                # Preserve the original F.linear call, including its FP16 output.
                z = original_forward(input)
                if selected:
                    require(tuple(z.shape) == (1, 6144) and z.dtype == torch.float16, 'Original Gate/Up result differs')
                    snapshot.update(z=z.detach().clone(), z_metadata=metadata(z), live_z_pointer=z.data_ptr())
                    pending[layer_id] = snapshot
                return z

            def activate(z):
                activation_calls[layer_id] += 1
                snapshot = pending.get(layer_id)
                if snapshot is not None:
                    require(snapshot['step'] == len(raw) and snapshot['live_z_pointer'] == z.data_ptr(),
                            'Captured Gate/Up output does not feed original activation')
                # Preserve the existing FlashInfer-backed activation exactly.
                y = original_act(z)
                if snapshot is not None:
                    require(tuple(y.shape) == (1, 3072) and y.dtype == torch.float16, 'Original activation output differs')
                    snapshot.update(y=y.detach().clone(), y_metadata=metadata(y))
                    captures[snapshot['step'], layer_id] = snapshot
                    del pending[layer_id]
                return y

            gate.forward, mlp.act_fn = forward, activate

        for layer_id, layer in enumerate(layers):
            bind(layer_id, layer.mlp)
        raw.clear()
        outputs = model.generate([cfg['input_token_ids']], params)
        torch.cuda.synchronize()
        require(raw == cfg['raw_expected_token_ids'] and len(raw) == 32, 'Captured generation differs from original tokens')
        require(not pending and set(captures) == {(step, layer) for step in CAPTURE_STEPS for layer in range(28)},
                'Missing selected Gate/Up/activation captures')
        require(all(n == 32 for n in gate_calls.values()) and all(n == 32 for n in activation_calls.values()),
                'Original Gate/Up/activation call counts differ')
        require(all(torch.equal(weight_objects[layer], weight_clones[layer]) for layer in range(28)),
                'Original Gate/Up weights changed during warmup/capture')
        # All disk I/O and tensor-to-CPU conversion occur after generation.
        weights_dir, samples_dir = run / 'weights', run / 'samples'
        weights_dir.mkdir()
        samples_dir.mkdir()
        for layer_id in range(28):
            array = weight_clones[layer_id].cpu().numpy()
            require(array.dtype == np.float16 and array.shape == (6144, 1024) and np.isfinite(array).all(), 'Invalid exported weight')
            path = weights_dir / f'layer{layer_id:02d}.npy'
            np.save(path, array, allow_pickle=False)
            fixture['weights'].append({'layer': layer_id, 'path': path.relative_to(run).as_posix(),
                'sha256': sha(path), 'file_bytes': path.stat().st_size, 'array_bytes': array.nbytes,
                'shape': list(array.shape), 'dtype': str(array.dtype), 'metadata': metadata(weight_objects[layer_id]),
                'read_only_equal_after_generation': True})
        for (step, layer_id), captured in sorted(captures.items()):
            arrays = {name: captured[name].cpu().numpy() for name in ('x', 'z', 'y')}
            require(all(value.dtype == np.float16 and np.isfinite(value).all() for value in arrays.values()), 'Non-finite/non-FP16 captured values')
            actual_input_tokens = captured['token_ids'].cpu().tolist()
            require(actual_input_tokens == [cfg['raw_expected_token_ids'][step-1]], 'Actual decode token does not match previous accepted output')
            path = samples_dir / f'step{step:02d}-layer{layer_id:02d}.npz'
            # Uncompressed fixed arrays avoid object/pickle payloads.
            np.savez(path, **arrays)
            fixture['samples'].append({'step': step, 'layer': layer_id, 'phase': 'decode',
                'module_path': captured['module_path'], 'path': path.relative_to(run).as_posix(),
                'sha256': sha(path), 'file_bytes': path.stat().st_size,
                'actual_input_token_ids': actual_input_tokens,
                'arrays': {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.nbytes}
                           for name, value in arrays.items()},
                'source_metadata': {'x': captured['input_metadata'], 'z': captured['z_metadata'], 'y': captured['y_metadata']}})
        fixture.update(status='GATE_UP_FIXTURE_CAPTURE_PASS', raw_generated_token_ids=list(raw), outputs=outputs,
            layer_count=28, sample_count=84, weight_count=28, weight_array_bytes=sum(x['array_bytes'] for x in fixture['weights']),
            observed_gate_up_calls=sum(gate_calls.values()), observed_activation_calls=sum(activation_calls.values()),
            numerical_contract='Saved reference z is the original FP16 F.linear result; y is the original FlashInfer-backed silu_and_mul result on that z. New fused math may round differently and requires a separately declared numerical contract.',
            intended_split='step1 fixture for initial engineering; steps16 and31 held out from initial selection; all28layers retained.',
            auth='Coach capture and evidence; independent learner mastery not inferred.',
            limitations='One fixed real-model workload. Capturing GPU clones changes runtime and is not timing evidence. No candidate, speedup, whole-model quality, Graph or AMD claim.')
    except BaseException:
        primary = traceback.format_exc()
        fixture['error'] = primary
        raise
    finally:
        for obj, name, had_instance_value, value in reversed(saved_methods):
            if had_instance_value:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        if model is not None:
            try:
                model.shutdown()
            except BaseException:
                fixture['shutdown_error'] = traceback.format_exc()
                fixture['status'] = 'FAILED'
        save(run / 'fixture.json', fixture)
    require(fixture['status'] == 'GATE_UP_FIXTURE_CAPTURE_PASS', 'Fixture cleanup failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribution-run', type=Path)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child.resolve())
        return
    if args.attribution_run is None:
        parser.error('--attribution-run is required (completed47 original attribution)')
    project = args.project.resolve(strict=True)
    parent = project / 'experiments/m08-gate-up-fixture-results'
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-'), dir=parent))
    print('RUN_DIRECTORY=' + str(run), flush=True)
    result, primary = {'status': 'FAILED'}, None
    try:
        accepted = args.attribution_run.resolve(strict=True)
        old_result, cfg = read(accepted / 'result.json'), read(accepted / 'plan.json')
        require(old_result['status'] == 'LINEAR_ATTRIBUTION_COMPLETE', 'Accepted47 attribution incomplete')
        require(read(accepted / 'workload.json')['status'] == 'LINEAR_ATTRIBUTION_WORKLOAD_CHECKED', 'Accepted47 workload incomplete')
        require(sha(accepted / 'plan.json') == old_result['plan_sha256'], 'Accepted47 plan changed')
        for name, digest in cfg['frozen_files'].items():
            require(sha(accepted / name) == digest, 'Accepted47 frozen helper changed')
        require(sha(HERE / HELPER) == cfg['frozen_files'][HELPER], '39 helper changed since47')
        require({name: importlib.metadata.version(name) for name in cfg['timing_packages']} == cfg['timing_packages'], 'Package versions changed')
        require(len(cfg['input_token_ids']) == 128 and cfg['max_new_tokens'] == len(cfg['raw_expected_token_ids']) == 32, 'Accepted workload changed')
        helper = load(HERE / HELPER, 'gate_up_snapshot_helper')
        sources = helper.snapshot_source(project, run)
        require(sources == cfg['source_manifest'], 'Current source differs from accepted47')
        (run / 'runner.py').rename(run / HELPER)
        shutil.copyfile(__file__, run / 'm08_gate_up_fixture_worker.py')
        shutil.copyfile(HERE / '48-m08-capture-gate-up.sh', run / '48-m08-capture-gate-up.sh')
        model = helper.inspect_model(Path(cfg['model']['path']), cfg['model']['revision'])
        require([(x['name'], x['sha256']) for x in model['weights']] == [(x['name'], x['sha256']) for x in cfg['model']['weights']], 'Model weight digest changed')
        require(model['metadata_sha256'] == cfg['model']['metadata_sha256'], 'Model/tokenizer metadata changed')
        cfg.update(model=model, source_manifest=sources, accepted_attribution=str(accepted),
            accepted_attribution_result_sha256=sha(accepted / 'result.json'), capture_steps=list(CAPTURE_STEPS),
            frozen_files={name: sha(run / name) for name in (HELPER, 'm08_gate_up_fixture_worker.py', '48-m08-capture-gate-up.sh', 'source-manifest.json')})
        save(run / 'plan.json', cfg)
        shutil.copyfile(accepted / 'plan.json', run / 'accepted-attribution-plan.json')
        shutil.copyfile(accepted / 'result.json', run / 'accepted-attribution-result.json')
        result.update(plan_sha256=sha(run / 'plan.json'), accepted_attribution=str(accepted))
        with (run / 'capture.log').open('xb') as log:
            process = subprocess.run([sys.executable, '-B', '-u', str(run / 'm08_gate_up_fixture_worker.py'), '--child', str(run)],
                                     cwd=project, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        result['worker_returncode'] = process.returncode
        require(process.returncode == 0, 'Fixture worker failed; capture.log retained')
        fixture = read(run / 'fixture.json')
        require(fixture['status'] == 'GATE_UP_FIXTURE_CAPTURE_PASS', 'Fixture capture did not pass')
        require(fixture['sample_count'] == 84 and fixture['weight_count'] == 28, 'Fixture count differs')
        for item in fixture['weights'] + fixture['samples']:
            require(sha(run / item['path']) == item['sha256'], 'Saved fixture file digest mismatch')
        result.update(status='GATE_UP_FIXTURE_CAPTURE_PASS', fixture_sha256=sha(run / 'fixture.json'),
            samples=84, weights=28, weight_array_bytes=fixture['weight_array_bytes'],
            limit='Original Eager data capture only; no optimized candidate, timing, improvement or numerical acceptance claim.')
    except BaseException:
        primary = traceback.format_exc()
        result['error'] = primary
        (run / 'controller-error.txt').write_text(primary, encoding='utf-8')
        raise
    finally:
        save(run / 'result.json', result)
        shared = HERE / 'logs/m08-gate-up-fixture-results' / run.name
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
            parent = HERE / 'logs/m08-gate-up-fixture-errors'
            parent.mkdir(parents=True, exist_ok=True)
            out = Path(tempfile.mkdtemp(prefix='error-', dir=parent))
            save(out / 'error.json', {'argv': sys.argv, 'runner_sha256': sha(__file__), 'error': traceback.format_exc()})
        except BaseException:
            pass
        raise
