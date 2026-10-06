"""Verify selectable-width test wiring using NumPy, not PyTorch or a GPU."""

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class Array(np.ndarray):
    def clone(self):
        return self.copy()

    def cuda(self):
        return self.copy()

    def cpu(self):
        return self.copy()

    def stride(self):
        return tuple(s // self.itemsize for s in self.strides)


torch = SimpleNamespace(
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


def must_raise(exception, operation, contains=None):
    try:
        operation()
    except exception as error:
        if contains is not None:
            assert contains in str(error), str(error)
        return
    raise AssertionError(f"Expected {exception.__name__}")


class KernelStandIn:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def execute(kc, vc, indices, k, v, num_warps):
            assert num_warps == 4
            self.calls.append((grid, k.shape[1], k.stride(), kc.stride()))
            for pid in range(grid[0]):
                kc[indices[pid]] = k[pid]
                vc[indices[pid]] = v[pid]
        return execute


sys.modules['torch'] = torch
harness = load(ROOT / '13-check-kv-three-rows.py', 'harness13_width')
checker = load(ROOT / '12-check-kv-targets.py', 'checker13_width').check_kv_targets
for name in ('13-check-kv-three-rows.py', '13-check-kv-three-rows.sh'):
    assert b'\r' not in (ROOT / name).read_bytes(), f"Expected LF: {name}"
ast.parse((ROOT / '13-check-kv-three-rows.py').read_text(encoding='utf-8'))
sources = {
    64: (ROOT / 'logs/kv-three-rows-results/run-20261002-135746-9uoe5ife/triton_store_first.py').read_text(encoding='utf-8'),
    128: (ROOT / 'logs/kv-three-rows-results/run-20261002-135009-xhfsj8ry/triton_store_first.py').read_text(encoding='utf-8'),
}
for width in (64, 128):
    assert harness.validate_kernel(sources[width], width)['cache_row_stride'] == 2 * width
    must_raise(RuntimeError, lambda w=width: harness.validate_kernel(sources[w], 192 - w),
               'KERNEL_LAYOUT_MISMATCH')
must_raise(ValueError, lambda: harness.validate_kernel(sources[64], 96))

kernel = KernelStandIn()
controls = {}
with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)
    for width in (64, 128):
        controls[width] = harness.checker_controls(torch, checker, width)
        assert len(controls[width]['negative_controls']) == 7
        launcher = harness.make_learner_launcher(SimpleNamespace(store_kv_first=kernel), width)
        case = temp / f'width{width}'
        harness.run_case(torch, launcher, checker, (7, 0, 3), case, 'learner_triton', width)
        before = json.loads((case / 'input.json').read_text())
        assert before['input_shape'] == [3, width]
        assert before['cache_shape'] == [8, 2, width]
        assert before['input_stride'] == [width, 1]
        assert before['cache_view_stride'] == [2 * width, 1]
        assert json.loads((case / 'status.json').read_text())['status'] == 'PASS'
        # Logical shape alone is insufficient: reject separate cache layout.
        must_raise(RuntimeError, lambda w=width, f=launcher: f(
            torch.full((8, w), -1), torch.full((8, w), -1), torch.tensor([7, 0, 3]),
            torch.full((3, w), 1), torch.full((3, w), 2)), 'CACHE_LAYOUT')

    # A partial-row write must fail and leave the raw evidence on disk.
    def bad_copy(kc, vc, indices, k, v):
        for pid, pos in enumerate(indices):
            kc[pos] = k[pid]
            vc[pos, :-1] = v[pid, :-1]
    bad = temp / 'missing-last-v-value'
    must_raise(AssertionError, lambda: harness.run_case(
        torch, bad_copy, checker, (7, 0, 3), bad, 'learner_triton', 64))
    assert (bad / 'result.json').is_file()
    assert json.loads((bad / 'status.json').read_text())['status'] == 'FAILED'

    # Exercise main's mismatch path: archive evidence and preserve student file
    # before attempting any GPU import or launch. Git answers are test stubs.
    project = temp / 'project'
    (project / 'experiments').mkdir(parents=True)
    student = project / 'experiments/triton_store_first.py'
    student.write_text(sources[64], encoding='utf-8')
    (project / 'experiments/check_kv_targets.py').write_text(
        (ROOT / '12-check-kv-targets.py').read_text(encoding='utf-8'), encoding='utf-8')
    shared = temp / 'shared'
    previous_args = sys.argv
    previous_git = harness.subprocess.check_output
    def fake_git(command, **kwargs):
        return harness.COMMIT if command[3] == 'rev-parse' else ''
    try:
        harness.subprocess.check_output = fake_git
        sys.argv = ['runner', '--project', str(project), '--shared-results', str(shared), '--width', '128']
        must_raise(RuntimeError, harness.main, 'KERNEL_LAYOUT_MISMATCH')
    finally:
        sys.argv = previous_args
        harness.subprocess.check_output = previous_git
    assert student.read_text(encoding='utf-8') == sources[64]
    archived = list(shared.glob('run-*'))
    assert len(archived) == 1
    state = json.loads((archived[0] / 'status.json').read_text())
    assert state['status'] == 'FAILED' and state['passed_cases'] == []
    assert 'KERNEL_LAYOUT_MISMATCH' in state['error']
    assert not list(archived[0].glob('original_cuda*'))

assert kernel.calls == [((3,), 64, (64, 1), (128, 1)), ((3,), 128, (128, 1), (256, 1))]
result = {
    'scope': 'Local Python/NumPy harness checks only; not PyTorch/Triton execution',
    'syntax_and_lf': 'PASS',
    'fixtures_and_launches': [64, 128],
    'checker_deliberate_errors_rejected': 14,
    'both_width_mismatches_rejected_before_launch': True,
    'wrong_layout_rejected': True,
    'partial_write_failure_and_preflight_failure_archived': True,
    'student_source_preserved': True,
    'bash_execution': 'NOT_RUN',
    'gpu_execution': 'NOT_RUN',
}
(ROOT / 'verification/13-width-selection-local-result.json').write_text(
    json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
