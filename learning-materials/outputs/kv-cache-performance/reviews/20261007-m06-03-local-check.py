"""Local harness simulation only: no CUDA runtime or Triton compilation."""
import ast
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

base = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("mask_harness_test", base / "21-check-kv-mask.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class Tensor:
    def __init__(self, a): self.a = a
    @property
    def shape(self): return self.a.shape
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

class LaunchProbe:
    def __init__(self): self.calls = []
    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **kwargs):
            assert grid == (3,)
            assert kwargs["WIDTH"] == k.shape[1]
            assert kwargs["SRC_ROW_STRIDE"] == k.stride(0)
            assert kwargs["DST_ROW_STRIDE"] == kc.stride(0)
            block = kwargs["BLOCK_SIZE"]
            assert block >= k.shape[1] and block & (block-1) == 0
            assert block == {64:64, 128:128, 192:256}[k.shape[1]]
            self.calls.append(dict(grid=grid, **kwargs))
            reference(kc, vc, indices, k, v)
        return call

root = Path(tempfile.mkdtemp(prefix="m06-03-local-", dir=base / "reviews"))
probe = LaunchProbe()
launch = m.make_launcher(SimpleNamespace(store_kv_v1=probe))
cfgs = m.configurations()
assert len(cfgs) == 14 and len({tuple(c.items()) for c in cfgs}) == 14
for i, cfg in enumerate(cfgs):
    for name, call in (("numpy_reference", reference), ("mock_launch", launch)):
        directory = root / f"{i}-{name}"
        m.run_case(torch, call, checker, cfg, directory, name)
        assert json.loads((directory / "status.json").read_text())["status"] == "PASS"
        before = json.loads((directory / "input.json").read_text())
        after = json.loads((directory / "result.json").read_text())
        assert before["input_shape"] == [3, cfg["width"]]
        assert before["cache_view_shape"] == [8, cfg["width"]]
        assert "backing_after" in after
assert len(probe.calls) == 14

# Corrupt an allocated masked cache tail, after correct logical writes.
cfg = dict(width=192, layout="separate", input_gap=64, cache_guard=64)
def bad_guard(kc, vc, indices, k, v):
    reference(kc, vc, indices, k, v)
    # NumPy view follows the shared root allocation; this is CPU-only.
    allocation = kc.a.base
    allocation[0, 7, 192] = 123
failure = root / "deliberate-cache-guard-corruption"
try:
    m.run_case(torch, bad_guard, checker, cfg, failure, "bad_cpu_only")
except AssertionError:
    assert (failure / "result.json").is_file()
    assert (failure / "error.txt").is_file()
    assert json.loads((failure / "status.json").read_text())["status"] == "FAILED"
else:
    raise AssertionError("guard corruption not caught")

# Installation must not overwrite current project source or later mask edits.
project = root / "mock_project"
(project / "experiments").mkdir(parents=True)
old = project / "experiments" / "triton_store_v1.py"
old.write_text("# previous source stays untouched\n")
student = m.prepare(project)
assert student.read_text() == (base / "21-triton-store-mask.py").read_text()
original = old.read_bytes()
student.write_text("# user edit; preserve it\n")
assert m.prepare(project).read_text() == "# user edit; preserve it\n"
assert old.read_bytes() == original

source = (base / "21-triton-store-mask.py").read_text()
m.validate_mask_source(source)
mutations = [
    source.replace("mask = cols < WIDTH", "mask = cols <= WIDTH"),
    source.replace("cols = tl.arange(0, BLOCK_SIZE)", "cols = tl.arange(0, WIDTH)"),
    source.replace("pid * SRC_ROW_STRIDE", "pid * BLOCK_SIZE"),
]
for line in source.splitlines():
    if "mask=mask" in line:
        mutations.append(source.replace(line, line.replace(", mask=mask", "")))
for changed in mutations:
    try: m.validate_mask_source(changed)
    except RuntimeError: pass
    else: raise AssertionError("unreviewed mask source accepted")

summary = dict(status="LOCAL_HARNESS_PASS", GPU="NOT_RUN", Triton_compilation="NOT_RUN",
               simulated_cases=28, mask_source_mutations_rejected=len(mutations),
               guard_failure_saved=True, install_preserves_old_and_existing_sources=True,
               launch_parameter_sets=probe.calls, artifacts=str(root))
(base / "reviews/20261007-m06-03-local-check.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary))
