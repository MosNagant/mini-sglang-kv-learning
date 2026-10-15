"""CPU-only 51 integration controls; no CUDA/Triton import or device execution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

base = Path(__file__).resolve().parents[1]
source = base / '51-m08-gate-up-model.py'
ast.parse(source.read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('model51_cpu', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []


def rejects(call, substring):
    try:
        call()
    except RuntimeError as error:
        assert substring in str(error), (substring, str(error))
        return
    raise AssertionError('Expected rejection: ' + substring)


# Synthetic full-vocabulary data exercises every step, not a miniature duplicate.
with tempfile.TemporaryDirectory(prefix='model51-cpu-') as temp:
    run = Path(temp)
    for arm in ('original', 'candidate'):
        (run / arm).mkdir()
        module.save(run / arm / 'result.json', {
            'status': 'MODEL_ALL_LOGITS_COLLECTED', 'raw_generated_token_ids': list(range(32))})
    reference = {f'step{i:02d}': np.ones((1, 151936), dtype=np.float16) for i in range(32)}
    np.savez(run / 'original/all-logits.npz', **reference)
    np.savez(run / 'candidate/all-logits.npz', **reference)
    result = module.compare_all_logits(run, list(range(32)))
    assert result['pass'] and result['elements_checked'] == 4861952
    checks.append('all_32_steps_4861952_elements_exact_pass')
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate['step17'][0, 81234] = np.float16(1.003)
    np.savez(run / 'candidate/all-logits.npz', **candidate)
    assert module.compare_all_logits(run, list(range(32)))['pass']
    checks.append('within_declared_tolerance_pass')
    candidate['step17'][0, 81234] = np.float16(1.01)
    np.savez(run / 'candidate/all-logits.npz', **candidate)
    result = module.compare_all_logits(run, list(range(32)))
    assert not result['pass'] and result['violations'] == 1
    checks.append('one_logit_violation_rejected')
    candidate['step17'][0, 81234] = np.nan
    np.savez(run / 'candidate/all-logits.npz', **candidate)
    assert not module.compare_all_logits(run, list(range(32)))['pass']
    checks.append('nonfinite_logit_rejected')
    np.savez(run / 'candidate/all-logits.npz', **reference)
    module.save(run / 'candidate/result.json', {
        'status': 'MODEL_ALL_LOGITS_COLLECTED', 'raw_generated_token_ids': [99] + list(range(1, 32))})
    result = module.compare_all_logits(run, list(range(32)))
    assert not result['pass'] and result['all_logits_finite_and_within_tolerance']
    checks.append('token_mismatch_rejected_despite_equal_logits')


# Fake model objects test routing and prepare/capture order without torch import.
capture = [False]
ctx = NS(batch=NS(is_prefill=True, is_decode=False))
class GatedMLP:
    def forward(self, x):
        return ('original', x)

class Tensor:
    shape = (1, 1024)

class Weight:
    def data_ptr(self):
        return id(self)

fake_torch = ModuleType('torch')
fake_torch.cuda = NS(is_current_stream_capturing=lambda: capture[0])
fake_core = ModuleType('minisgl.core')
fake_core.get_global_ctx = lambda: ctx
fake_utils = ModuleType('minisgl.models.utils')
fake_utils.GatedMLP = GatedMLP
with patch.dict(sys.modules, {'torch': fake_torch, 'minisgl.core': fake_core,
                             'minisgl.models.utils': fake_utils}):
    for kind in ('gemv', 'swiglu'):
        prepared_inputs, current_inputs, activations = [], [], []
        def prepare(name, x, weight):
            prepared_inputs.append(x)
            return {'kind': kind, 'call_with_input': lambda current: current_inputs.append(current) or 'new-value'}
        candidate_module = NS(candidates=lambda: [{'name': 'test'}], prepare=prepare)
        original = GatedMLP.forward
        state, restore = module.install_decode_candidate(candidate_module, 'test')
        layer = GatedMLP()
        layer.gate_up_proj = NS(weight=Weight(), bias=None)
        layer.act_fn = lambda value: activations.append(value) or 'activated'
        layer.down_proj = NS(forward=lambda value: ('down', value))
        x1, x2 = Tensor(), Tensor()
        ctx.batch = NS(is_prefill=True, is_decode=False)
        assert layer.forward(x1) == ('original', x1) and not prepared_inputs
        ctx.batch = NS(is_prefill=False, is_decode=True)
        capture[0] = True
        rejects(lambda: layer.forward(x1), 'First candidate prepare')
        capture[0] = False
        layer.forward(x1)
        capture[0] = True
        output = layer.forward(x2)
        assert prepared_inputs == [x1] and current_inputs == [x1, x2]
        assert state['capture_python_calls'] == 1 and state['current_input_calls'] == 2
        assert output == ('down', 'activated' if kind == 'gemv' else 'new-value')
        assert len(activations) == (2 if kind == 'gemv' else 0)
        layer.gate_up_proj.weight = Weight()
        rejects(lambda: layer.forward(x2), 'weight identity changed')
        restore()
        assert GatedMLP.forward is original
        capture[0] = False
        checks.append(kind + '_prefill_current_input_capture_order_weight_identity_restore_pass')

result = {'status': 'CPU_CONTROLS_PASS', 'runner_sha256': module.sha(source), 'checks': checks,
          'limit': 'Synthetic CPU arrays and fake model objects only; no GPU execution or model numerical/performance acceptance.'}
path = Path(__file__).with_suffix('.json')
module.save(path, result)
print(json.dumps(result, indent=2))
