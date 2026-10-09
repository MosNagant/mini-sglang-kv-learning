"""Coach local NumPy simulation: validates harness/launch parameters, never GPU code."""
import ast
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

base = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("width_harness", base / "14-check-kv-widths.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
ast.parse((base / "14-check-kv-widths.py").read_text(encoding="utf-8"))

class Tensor:
    def __init__(self, a): self.a = a
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    def stride(self, dim=None):
        values = tuple(n // self.a.itemsize for n in self.a.strides)
        return values if dim is None else values[dim]
    def __getitem__(self, index): return Tensor(self.a[index])
    def cuda(self): return Tensor(self.a.copy())
    def cpu(self): return self
    def numpy(self): return self.a
    def tolist(self): return self.a.tolist()

torch = SimpleNamespace(from_numpy=Tensor, cuda=SimpleNamespace(synchronize=lambda: None))
def reference(kc, vc, indices, k, v):
    for pid, pos in enumerate(indices.a):
        kc.a[pos] = k.a[pid]
        vc.a[pos] = v.a[pid]
def checker(kc, vc, k, v, rows):
    for pid, pos in enumerate(rows):
        np.testing.assert_array_equal(kc.a[pos], k.a[pid])
        np.testing.assert_array_equal(vc.a[pos], v.a[pid])

class MockKernel:
    def __init__(self): self.calls = []
    def __getitem__(self, grid):
        def launch(kc, vc, indices, k, v, **kwargs):
            assert grid == (3,)
            assert kwargs == dict(WIDTH=k.shape[1], SRC_ROW_STRIDE=k.stride(0),
                                  DST_ROW_STRIDE=kc.stride(0), num_warps=4)
            self.calls.append(dict(grid=grid, **kwargs))
            reference(kc, vc, indices, k, v)
        return launch

assert len(h.select_configurations()) == 2
assert len(h.select_configurations(layouts=True)) == 8
assert h.select_configurations(width=64, input_gap=64) == [dict(width=64, layout="interleaved", input_gap=64)]
try:
    h.select_configurations(width=64, layouts=True)
except ValueError:
    pass
else:
    raise AssertionError("mixed CLI modes accepted")

kernel = MockKernel()
launcher = h.make_learner_launcher(SimpleNamespace(store_kv_v1=kernel))
run = Path(tempfile.mkdtemp(prefix="m06-02-local-", dir=base / "reviews"))
for n, cfg in enumerate(h.select_configurations(layouts=True)):
    for name, call in (("reference", reference), ("mock_triton", launcher)):
        directory = run / f"{n}-{name}"
        h.run_case(torch, call, checker, h.TARGET_ROWS, cfg["width"], directory,
                   name, cfg["layout"], cfg["input_gap"])
        result = json.loads((directory / "result.json").read_text())
        assert "backing_after" in result
        assert json.loads((directory / "status.json").read_text())["status"] == "PASS"
assert len(kernel.calls) == 8

def bad(kc, vc, indices, k, v):
    reference(kc, vc, indices, k, v)
    kc.a[2, 0] = -99
failed = run / "deliberate-untouched-corruption"
try:
    h.run_case(torch, bad, checker, h.TARGET_ROWS, 64, failed, "mock_bad", "separate", 64)
except AssertionError:
    assert (failed / "result.json").is_file()
    assert (failed / "error.txt").is_file()
    assert json.loads((failed / "status.json").read_text())["status"] == "FAILED"
else:
    raise AssertionError("corrupted untouched row accepted")

for name, args in (
    ("source_KV_stride", (Tensor(np.zeros((8,64),np.float16)), Tensor(np.zeros((8,64),np.float16)), Tensor(np.array([7,0,3],np.int32)), Tensor(np.zeros((3,128),np.float16)[:,:64]), Tensor(np.zeros((3,64),np.float16)))),
    ("column_stride", (Tensor(np.zeros((8,64),np.float16)), Tensor(np.zeros((8,64),np.float16)), Tensor(np.array([7,0,3],np.int32)), Tensor(np.zeros((3,128),np.float16)[:,::2]), Tensor(np.zeros((3,128),np.float16)[:,::2]))),
):
    try:
        launcher(*args)
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid launch accepted: " + name)

summary = dict(status="LOCAL_HARNESS_AND_LAUNCH_MOCK_PASS", GPU="NOT_RUN", Triton_kernel="NOT_EXECUTED", simulated_cases=16, parameter_sets=kernel.calls, failed_case_preserved=True, rejected_launch_contracts=2, artifacts=str(run))
(base / "reviews/20261007-m06-02-local-check.json").write_text(json.dumps(summary, indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary))
