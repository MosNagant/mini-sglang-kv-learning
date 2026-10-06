"""Local width-fixture/launch checks with NumPy stand-ins; NOT GPU validation."""

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
    def __init__(self, fixed_width=None):
        self.fixed_width = fixed_width
        self.calls = []

    def __getitem__(self, grid):
        def execute(kc, vc, idx, k, v, WIDTH, num_warps):
            assert num_warps == 4
            self.calls.append((grid, WIDTH))
            columns = WIDTH if self.fixed_width is None else self.fixed_width
            for pid in range(grid[0]):
                kc[idx[pid], :columns] = k[pid, :columns]
                vc[idx[pid], :columns] = v[pid, :columns]
        return execute


sys.modules['torch'] = stand_in
harness = load(root / '14-check-kv-widths.py', 'harness14')
checker = load(root / '12-check-kv-targets.py', 'checker14').check_kv_targets
for name in ('14-check-kv-widths.py', '14-triton-store-width-template.py'):
    ast.parse((root / name).read_text(encoding='utf-8'))
for name in ('14-check-kv-widths.py', '14-check-kv-widths.sh', '14-triton-store-width-template.py'):
    assert b'\r' not in (root / name).read_bytes()
template = (root / '14-triton-store-width-template.py').read_text(encoding='utf-8')
must_raise(RuntimeError, lambda: harness.validate_kernel(template))
harness.validate_kernel('def store_kv_width(a,b,c,d,e,WIDTH:tl.constexpr):\n    return a\n')
must_raise(RuntimeError, lambda: harness.validate_kernel(
    'def store_kv_width(a,b,c,d,e,WIDTH):\n    return a\n'))
controls = {width: harness.checker_controls(stand_in, checker, width) for width in harness.WIDTHS}
assert sum(len(c['negative_controls']) for c in controls.values()) == 14
kernel = KernelStandIn()
launcher = harness.make_learner_launcher(SimpleNamespace(store_kv_width=kernel))
with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    learner = harness.prepare_student(temp / 'project')
    assert learner.read_text(encoding='utf-8') == template
    learner.write_text('# Existing learner edit must be preserved.\n', encoding='utf-8')
    harness.prepare_student(temp / 'project')
    assert learner.read_text(encoding='utf-8').startswith('# Existing learner edit')
    for width in harness.WIDTHS:
        case = temp / f'w{width}'
        harness.run_case(stand_in, launcher, checker, (7, 0, 3), width, case, 'learner_triton')
        saved = json.loads((case / 'input.json').read_text())
        assert saved['input_shape'] == [3, width]
        assert saved['input_stride'] == [width, 1]
        assert saved['cache_view_stride'] == [2 * width, 1]
        assert json.loads((case / 'status.json').read_text())['status'] == 'PASS'
    assert kernel.calls == [((3,), 64), ((3,), 128)]
    fixed = harness.make_learner_launcher(SimpleNamespace(store_kv_width=KernelStandIn(64)))
    bad = temp / 'fixed64-on128'
    must_raise(AssertionError, lambda: harness.run_case(
        stand_in, fixed, checker, (7, 0, 3), 128, bad, 'learner_triton'))
    assert json.loads((bad / 'status.json').read_text())['status'] == 'FAILED'
    assert (bad / 'result.json').is_file() and (bad / 'error.txt').is_file()
    must_raise(RuntimeError, lambda: launcher(
        None, None, np.zeros(3), np.zeros((3, 96)), np.zeros((3, 96))))
    # Correct logical cache shape but separate contiguous buffers: not this lesson's layout.
    must_raise(RuntimeError, lambda: launcher(
        stand_in.full((8, 64), -1), stand_in.full((8, 64), -1), np.zeros(3),
        stand_in.full((3, 64), 1), stand_in.full((3, 64), 2)))

result = {
    'scope': 'NumPy stand-ins test harness organization only, not PyTorch/Triton or GPU execution',
    'syntax_and_LF': 'PASS',
    'width_fixture_and_launch_pairs': [[3, 64], [3, 128]],
    'checker_deliberate_errors_rejected': 14,
    'fixed64_on_width128_rejected_and_raw_result_preserved': True,
    'incomplete_template_and_missing_constexpr_rejected': True,
    'existing_learner_file_preserved': True,
    'unsupported_width_and_wrong_cache_layout_rejected': True,
    'bash_execution': 'NOT_RUN', 'gpu_execution': 'NOT_RUN',
}
(root / 'verification/14-checks-local-result.json').write_text(
    json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
