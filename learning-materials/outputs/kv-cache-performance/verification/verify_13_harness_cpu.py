"""Exercise three-row fixture/launch/check wiring with NumPy; no GPU validation."""

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root = Path(__file__).resolve().parent.parent


class Array(np.ndarray):
    def clone(self):
        return self.copy()

    def cuda(self):
        return self.copy()

    def cpu(self):
        return self.copy()

    def stride(self):
        return tuple(s // self.itemsize for s in self.strides)


stand_in = SimpleNamespace(
    float16=np.float16, int32=np.int32,
    arange=lambda *a, **kw: np.arange(*a, **kw).view(Array),
    full=lambda *a, device=None, **kw: np.full(*a, **kw).view(Array),
    tensor=lambda *a, **kw: np.array(*a, **kw).view(Array),
    equal=np.array_equal, all=np.all,
    cuda=SimpleNamespace(synchronize=lambda: None),
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def must_raise(exception, operation):
    try:
        operation()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


class KernelStandIn:
    """Execute the requested row count on CPU to observe launch wiring only."""
    def __init__(self):
        self.grids = []

    def __getitem__(self, grid):
        self.grids.append(grid)

        def execute(kc, vc, idx, k, v, num_warps):
            assert num_warps == 4
            for pid in range(grid[0]):
                kc[idx[pid]] = k[pid]
                vc[idx[pid]] = v[pid]
        return execute


sys.modules['torch'] = stand_in
harness = load(root / '13-check-kv-three-rows.py', 'harness13')
checker = load(root / '12-check-kv-targets.py', 'checker13').check_kv_targets
for name in ('13-check-kv-three-rows.py', '13-check-kv-three-rows.sh'):
    assert b'\r' not in (root / name).read_bytes()
ast.parse((root / '13-check-kv-three-rows.py').read_text())
harness.validate_kernel('def store_kv_first(a,b,c,d,e):\n    return a\n')
must_raise(RuntimeError, lambda: harness.validate_kernel(
    'def store_kv_first(a,b,c,d,e,k,v,num_warps=4):\n    return a\n'))
controls = harness.checker_controls(stand_in, checker)
assert len(controls['negative_controls']) == 7
kernel = KernelStandIn()
launcher = harness.make_learner_launcher(SimpleNamespace(store_kv_first=kernel))
with tempfile.TemporaryDirectory() as directory:
    run = Path(directory)
    harness.run_case(stand_in, launcher, checker, (7, 0, 3), run / 'valid', 'learner_triton')
    saved_input = json.loads((run / 'valid/input.json').read_text())
    saved_result = json.loads((run / 'valid/result.json').read_text())
    assert saved_input['input_shape'] == [3, 128]
    assert saved_input['triton_grid'] == [3]
    assert kernel.grids == [(3,)]
    assert saved_result['k_cache'][3] == saved_input['expected_k'][2]
    assert saved_result['v_cache'][3] == saved_input['expected_v'][2]
    # A mistakenly retained two-program launch must leave evidence and fail.
    two_only = lambda kc, vc, idx, k, v: kernel[(2,)](kc, vc, idx, k, v, num_warps=4)
    must_raise(AssertionError, lambda: harness.run_case(
        stand_in, two_only, checker, (7, 0, 3), run / 'bad', 'learner_triton'))
    assert json.loads((run / 'bad/status.json').read_text())['status'] == 'FAILED'
    assert (run / 'bad/result.json').is_file()
    assert (run / 'bad/error.txt').is_file()
    must_raise(RuntimeError, lambda: launcher(
        None, None, np.zeros(3), np.zeros((2, 128)), np.zeros((2, 128))))

result = {
    'scope': 'NumPy stand-in only; not PyTorch, Triton compilation or GPU execution',
    'python_syntax_and_LF': 'PASS',
    'three_row_fixture_and_grid': 'PASS',
    'checker_deliberate_errors_rejected': 7,
    'incorrect_two_program_launch_rejected_and_results_preserved': True,
    'mismatched_input_rows_rejected_before_launch': True,
    'mixed_definition_and_launch_signature_rejected': True,
    'bash_execution': 'NOT_RUN', 'gpu_execution': 'NOT_RUN',
}
(root / 'verification/13-checks-local-result.json').write_text(
    json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
