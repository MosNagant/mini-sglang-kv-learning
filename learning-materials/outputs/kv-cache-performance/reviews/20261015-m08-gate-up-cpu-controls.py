"""CPU-only candidate interface controls; fake launches do NOT test GPU arithmetic.

Imports 50 only while torch/triton are replaced with explicit NumPy-backed fakes.
The archiver control uses a fake package whose __init__ raises if executed.
Numerical-gate controls call 49's actual CPU helper, without importing CUDA.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "50-m08-gate-up-candidates.py"
ARCHIVER = Path(__file__).with_name("archive-installed-flashinfer.py")
RUNNER = ROOT / "49-m08-gate-up-experiments.py"


class Device:
    def __init__(self, kind="cuda", index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)


class Tensor:
    def __init__(self, array, device=None, dtype=None, layout="strided", contiguous=None):
        self.array = array
        self.device = Device() if device is None else device
        self.dtype = str(array.dtype) if dtype is None else dtype
        self.layout = layout
        self.contiguous = contiguous

    @property
    def shape(self):
        return self.array.shape

    def is_contiguous(self):
        return self.array.flags.c_contiguous if self.contiguous is None else self.contiguous

    def view(self, *shape):
        return Tensor(self.array.reshape(shape), self.device, self.dtype, self.layout)

    def permute(self, *axes):
        return Tensor(self.array.transpose(axes), self.device, self.dtype, self.layout)

    def copy_(self, other):
        np.copyto(self.array, other.array)
        return self

    def numel(self):
        return self.array.size

    def element_size(self):
        return self.array.itemsize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejects(fn, exceptions=(ValueError, TypeError, RuntimeError)):
    try:
        fn()
    except exceptions:
        return
    raise AssertionError("Invalid control was unexpectedly accepted")


def candidate_controls():
    launches = []
    state = {"capture": False, "synchronizes": 0, "allocations": 0}
    torch = types.ModuleType("torch")
    torch.Tensor, torch.float16, torch.float32, torch.strided = Tensor, "float16", "float32", "strided"

    def empty(shape, dtype, device):
        state["allocations"] += 1
        return Tensor(np.empty(shape, dtype=dtype), device=device)

    def synchronize(device=None):
        state["synchronizes"] += 1

    torch.empty = empty
    torch.cuda = types.SimpleNamespace(device=lambda device: contextlib.nullcontext(),
                                      is_current_stream_capturing=lambda: state["capture"], synchronize=synchronize)
    triton, tl = types.ModuleType("triton"), types.ModuleType("triton.language")
    triton.cdiv = lambda a, b: (a + b - 1) // b

    class Kernel:
        def __init__(self, fn):
            self.name = fn.__name__

        def __getitem__(self, grid):
            def submit(*args, **kwargs):
                launches.append({"name": self.name, "grid": grid, "args": args, "kwargs": kwargs})
                return types.SimpleNamespace(name=self.name, fake_compilation=True)
            return submit

    triton.jit, triton.language = Kernel, tl
    names = {"torch": torch, "triton": triton, "triton.language": tl}
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update(names)
        spec = importlib.util.spec_from_file_location("candidate50_cpu_fake", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        descriptors = module.candidates()
        assert len(descriptors) == 7 and len({entry["name"] for entry in descriptors}) == 7
        assert sum(entry["kind"] == "gemv" for entry in descriptors) == 4
        descriptors[0]["row_tile"] = 128
        assert module.candidates()[0]["row_tile"] == 4
        # Each row has a different exactly representable FP16 value; gate and up
        # halves are distinct, so packing mistakes cannot pass by equal fixtures.
        rows = (np.arange(6144) % 1024).astype(np.float16)
        rows[3072:] *= -1
        weight = Tensor(np.broadcast_to(rows[:, None], (6144, 1024)).copy())
        before = weight.array.copy()
        x1 = Tensor(np.ones((1, 1024), dtype=np.float16))
        x2 = Tensor(np.full((1, 1024), 2, dtype=np.float16))
        results = []
        for entry in module.candidates():
            plan = module.prepare(entry["name"], x1, weight)
            assert all(compiled is None for compiled in plan["compiled_kernels"])
            assert plan["output"].shape == (1, 6144 if plan["kind"] == "gemv" else 3072)
            allocated, synchronized = state["allocations"], state["synchronizes"]
            first = len(launches)
            assert plan["call_with_input"](x2) is plan["output"]
            assert launches[first]["args"][0] is x2, "stale initial input"
            assert len(launches) - first == plan["kernel_launches_per_call"]
            assert all(compiled is not None for compiled in plan["compiled_kernels"])
            first = len(launches)
            assert plan["call"]() is plan["output"]
            assert launches[first]["args"][0] is x1
            assert (state["allocations"], state["synchronizes"]) == (allocated, synchronized)
            packed = plan["packed_weight"]
            if packed is not None:
                assert packed.shape == (3072, 2, 1024) and packed.is_contiguous()
                assert np.array_equal(packed.array[:, 0], weight.array[:3072])
                assert np.array_equal(packed.array[:, 1], weight.array[3072:])
                assert plan["prepack_bytes"] == 6144 * 1024 * 2
                assert launches[first]["args"][1] is packed
            if entry["family"] == "split_k":
                assert plan["workspace"].shape == (4, 6144)
                assert plan["workspace_bytes"] == 4 * 6144 * 4
                assert launches[first]["grid"] == (1536, 4)
                assert launches[first + 1]["grid"] == (24,)
            invalid_inputs = [object(), Tensor(np.ones((2, 1024), dtype=np.float16)),
                              Tensor(x1.array, dtype="float32"), Tensor(x1.array, device=Device("cpu")),
                              Tensor(x1.array, device=Device("cuda", 1)), Tensor(x1.array, layout="sparse"),
                              Tensor(x1.array, contiguous=False)]
            for bad in invalid_inputs:
                launches_before = len(launches)
                rejects(lambda bad=bad: plan["call_with_input"](bad))
                assert len(launches) == launches_before
            # Return metadata is not a mutable control channel for the callable.
            old_kwargs = dict(launches[first]["kwargs"])
            plan["spec"].update(family="unknown", row_tile=512, num_warps=32)
            start = len(launches)
            plan["call"]()
            assert launches[start]["kwargs"] == old_kwargs
            results.append({"name": entry["name"], "launches": plan["kernel_launches_per_call"],
                            "invalid_current_inputs_rejected": len(invalid_inputs),
                            "packed_roundtrip_checked": packed is not None})
        assert np.array_equal(before, weight.array)
        rejects(lambda: module.prepare("unknown", x1, weight))
        for bad_x, bad_w in ((object(), weight), (x1, object()),
                             (Tensor(np.ones((2, 1024), dtype=np.float16)), weight),
                             (x1, Tensor(weight.array, dtype="float32")),
                             (x1, Tensor(weight.array, device=Device("cuda", 1))),
                             (x1, Tensor(weight.array, contiguous=False)),
                             (x1, Tensor(weight.array, layout="sparse"))):
            rejects(lambda: module.prepare(module.candidates()[0]["name"], bad_x, bad_w))
        state["capture"] = True
        rejects(lambda: module.prepare(module.candidates()[0]["name"], x1, weight))
        return {"status": "PASS", "candidate_results": results, "prepare_rejection_controls": 9,
                "source_sha256": sha(SOURCE), "gpu_math_tested": False,
                "limit": "NumPy fakes prove Python dispatch, lifetime and metadata conditions only, not Triton compilation or arithmetic."}
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def archive_control():
    with tempfile.TemporaryDirectory(prefix="flashinfer-source-cpu-") as temp:
        base = Path(temp)
        package = base / "flashinfer"
        package.mkdir()
        (package / "__init__.py").write_text("raise AssertionError('Package code must never execute')\n")
        for relative in ("activation.py", "jit/activation.py", "include/flashinfer/activation.cuh", "include/flashinfer/math.cuh"):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// fake source: silu(x) = x / (1 + __expf(-x));\n")
        env = dict(os.environ, PYTHONPATH=str(base), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", str(ARCHIVER), "--output", str(base / "archive")],
                                capture_output=True, text=True, check=True, env=env, cwd=base)
        manifest = json.loads((base / "archive/manifest.json").read_text())
        assert manifest["status"] == "SOURCES_ARCHIVED" and len(manifest["files"]) == 4
        assert manifest["gpu_execution"] is False
        for entry in manifest["files"]:
            assert sha(base / "archive" / entry["archived_path"]) == entry["sha256"]
            assert entry["activation_math_lines"]
        second = subprocess.run([sys.executable, "-B", str(ARCHIVER), "--output", str(base / "archive")],
                                capture_output=True, text=True, env=env, cwd=base)
        assert second.returncode != 0 and "preserving existing evidence" in second.stderr
        return {"status": "PASS", "source_sha256": sha(ARCHIVER), "package_initializer_not_executed": True,
                "source_hash_roundtrip": True, "existing_evidence_preserved": True, "stdout": result.stdout.strip()}


def numerical_controls():
    # 49 imports stdlib at module level; numerical_check imports only NumPy.
    # Explicit module-delta check prevents a future accidental GPU import here.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("runner49_cpu_controls", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    check = runner.numerical_check
    passed = []

    def expect(name, actual, reference, oracle, outcome, extra=None):
        result = check(actual, reference, oracle)
        assert result["pass"] is outcome, (name, result)
        if extra is not None:
            extra(result)
        # Every numerical result must remain machine-readable strict JSON.
        json.dumps(result, allow_nan=False)
        passed.append(name)

    zeros, ones = np.zeros((1, 16)), np.ones((1, 16))
    expect("zero_oracle_exact_zero", zeros, zeros, zeros, True)
    expect("zero_oracle_nonzero_error", zeros + 1e-5, zeros, zeros, False)
    expect("zero_oracle_equal_baseline", ones, ones, zeros, True)
    expect("exact_reference", ones, ones, ones, True)
    expect("nrmse_improvement", ones + .009, ones + .01, ones, True)
    expect("nrmse_within_10_percent", ones + .0109, ones + .01, ones, True)
    expect("nrmse_over_10_percent_despite_allclose", ones + .0112, ones + .01, ones, False)
    expect("nrmse_absolute_floor_allows_small_error", ones + 5e-7, ones, ones, True)
    expect("nrmse_absolute_floor_rejects_large_error", ones + 2e-6, ones, ones, False)
    expect("allclose_inclusive_absolute_boundary", np.array([.002]), np.array([0.]), np.array([1.]), True)
    expect("allclose_just_outside_absolute_boundary", np.array([np.nextafter(.002, np.inf)]),
           np.array([0.]), np.array([1.]), False)
    baseline = np.ones(1024) + .1
    candidate = baseline.copy()
    candidate[0] += .01
    result = check(candidate, baseline, np.ones(1024))
    assert result["candidate_nrmse"] <= result["nrmse_limit"] and result["violations"] == 1 and not result["pass"]
    passed.append("one_element_failure_despite_nrmse_acceptance")
    for index, field in enumerate(("actual", "reference", "oracle")):
        for bad_value, label in ((np.nan, "nan"), (np.inf, "positive_inf"), (-np.inf, "negative_inf")):
            arrays = [ones.copy() for _ in range(3)]
            arrays[index][0, 0] = bad_value
            expect(f"{field}_{label}_rejected", *arrays, False)
    expect("actual_shape_mismatch", ones[:, :8], ones, ones, False)
    expect("oracle_shape_mismatch", ones, ones, ones[:, :8], False)
    expect("rank_mismatch_not_broadcast", ones.ravel(), ones, ones, False)
    rejects(lambda: check(np.empty(0), np.empty(0), np.empty(0)), exceptions=(ValueError,))
    passed.append("empty_fixed_scope_rejected")
    new_modules = set(sys.modules) - before
    assert not any(name == "torch" or name.startswith(("torch.", "triton", "flashinfer")) for name in new_modules)
    return {"status": "PASS", "source_sha256": sha(RUNNER), "controls": passed,
            "control_count": len(passed), "zero_denominator_guard": "actual helper uses max(norm(oracle), 1e-12)",
            "dtype_scope": "helper converts to float64; actual runner must separately require FP16 outputs",
            "gpu_execution": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = {"status": "PASS", "gpu_execution": False, "candidate_controls": candidate_controls(),
              "archiver_controls": archive_control(), "numerical_controls": numerical_controls(),
              "test_sha256": sha(__file__)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "candidate_source_sha256": sha(SOURCE)}))


if __name__ == "__main__":
    main()
