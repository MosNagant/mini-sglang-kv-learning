"""CPU-only contract test of candidate 44 using fake metadata and launcher.

Uses the actual unchanged 40 metadata validator; never imports real Torch or
Triton, never invokes a GPU, and does not establish kernel/model correctness.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Device:
    def __init__(self, name):
        self.name, self.type = name, name.split(":")[0]

    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Tensor:
    def __init__(self, shape, strides, dtype="float16", device="cuda:0", layout="strided"):
        self.shape, self.strides = tuple(shape), tuple(strides)
        self.dtype, self.device, self.layout = dtype, Device(device), layout

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def data_ptr(self):
        raise AssertionError("The metadata cache must not read/cache pointers")


class Kernel:
    def __init__(self):
        self.calls = []
        self.compiled = object()

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.compiled
        return launch


def fixture(b=3, d=64, c=128, src=256, dst=None):
    dst = d if dst is None else dst
    return (Tensor((c, d), (dst, 1)), Tensor((c, d), (dst, 1)),
            Tensor((b,), (1,), "int32"), Tensor((b, d), (src, 1)),
            Tensor((b, d), (src, 1)))


def main():
    paths = [ROOT / "40-m08-kv-backends.py", ROOT / "44-m08-kv-cached-metadata.py"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    base, candidate = (load(p, f"cpu_test_m08_{i}") for i, p in enumerate(paths))
    fake_torch = SimpleNamespace(Tensor=Tensor, strided="strided", int32="int32", float16="float16")
    sentinel = object()
    prior_torch = sys.modules.get("torch", sentinel)
    sys.modules["torch"] = fake_torch
    kernel = Kernel()
    base._triton_kernel = lambda: (SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()), kernel)
    original_validator = base._validate_metadata
    validations = []

    def counted_validator(*args):
        validations.append(1)
        return original_validator(*args)

    base._validate_metadata = counted_validator
    checks = []
    try:
        call = candidate.make_store_cache("triton", observe=True, base_module=base)
        assert call.kernel is kernel
        tensors = fixture()
        call(*tensors)
        call(*tensors)
        replacement = fixture()
        call(*replacement)
        assert len(validations) == 1
        assert all(a is b for a, b in zip(kernel.calls[-1][1], replacement))
        assert call.launch_count == call.call_count == 3
        assert call.compiled_kernel is kernel.compiled
        assert call.last_launch["grid"] == [3] and call.last_launch["num_warps"] == 4
        checks.append("first miss, repeat hit, new-object hit, current launch arguments and observation")

        # The same object can have new metadata; it must not reuse old offsets.
        replacement[3].strides = (320, 1)
        call(*replacement)
        assert len(validations) == 2 and kernel.calls[-1][2]["INPUT_K_STRIDE"] == 320
        assert kernel.calls[-1][2]["INPUT_V_STRIDE"] == 256
        call(*fixture(b=1))
        assert len(validations) == 3 and kernel.calls[-1][0] == (1,)
        three_d = list(fixture(d=128, src=512))
        for t in three_d[:2]:
            t.shape, t.strides = (128, 8, 16), (128, 16, 1)
        call(*three_d)
        assert len(validations) == 4 and kernel.calls[-1][2]["D"] == 128
        assert kernel.calls[-1][2]["BLOCK_D"] == 128
        checks.append("same-object stride change, B/shape change, 3-D trailing layout and independent strides")

        bad_cases = []
        x = list(fixture()); x[3].dtype = "float32"; bad_cases.append(x)
        x = list(fixture()); x[3].device = Device("cpu"); bad_cases.append(x)
        x = list(fixture()); x[3].device = Device("cuda:1"); bad_cases.append(x)
        x = list(fixture()); x[3].layout = "sparse"; bad_cases.append(x)
        x = list(fixture()); x[3].strides = (256, 2); bad_cases.append(x)
        x = list(fixture()); x[2].shape = (4,); bad_cases.append(x)
        x = list(fixture()); x[4].shape = (2, 64); bad_cases.append(x)
        for invalid in bad_cases:
            previous = len(validations)
            launched = len(kernel.calls)
            entries = call.metadata_cache_info()["entries"]
            for _ in range(2):
                try:
                    call(*invalid)
                except (TypeError, ValueError):
                    pass
                else:
                    raise AssertionError("Invalid signature accepted")
            assert len(validations) == previous + 2, "Failures must not be cached"
            assert len(kernel.calls) == launched and call.metadata_cache_info()["entries"] == entries
        try:
            call(*fixture()[:4], object())
        except TypeError:
            pass
        else:
            raise AssertionError("Non-Tensor accepted")
        checks.append("dtype/device/device-index/layout/column-stride/index-count/V-shape/type rejection; failures revalidated")

        previous = len(kernel.calls)
        call(*fixture(b=0))
        call(*fixture(b=0))
        assert len(kernel.calls) == previous
        assert call.launch_count == call.call_count
        checks.append("empty validated input returns without a launch")

        silent = candidate.make_store_cache("triton", observe=False, base_module=base)
        first = fixture(b=1)
        silent(*first)
        for b in range(2, 67):
            silent(*fixture(b=b))
            assert silent.metadata_cache_info()["entries"] <= 64
        assert silent.metadata_cache_info()["entries"] == 64
        previous = len(validations)
        silent(*first)
        assert len(validations) == previous + 1, "Oldest signature must be evicted"
        assert silent.call_count == silent.launch_count == 0
        assert silent.last_launch is None and silent.compiled_kernel is None
        checks.append("64-entry bound, FIFO eviction, observe=False has no custom dispatch counters")

        base_call = base.make_store_cache("triton", observe=True)
        candidate_call = candidate.make_store_cache("triton", observe=True, base_module=base)
        assert base_call.kernel is candidate_call.kernel
        for d in (64, 128, 192, 1024):
            current = fixture(d=d, src=4*d)
            base_call(*current)
            before = kernel.calls[-1]
            candidate_call(*current)
            after = kernel.calls[-1]
            assert before == after, "Caching must not alter arguments, grid or launch parameters"
            assert base_call.last_launch == candidate_call.last_launch
        checks.append("same base kernel object, tensors, grid/parameters and observation for D64/128/192/1024")

        # Delegation is deliberate; do not duplicate or weaken original behavior.
        calls = []
        base.make_store_cache = lambda *args, **kwargs: (calls.append((args, kwargs)), sentinel)[1]
        assert candidate.make_store_cache("original", original_store=sentinel, observe=True, base_module=base) is sentinel
        assert calls == [(("original",), {"original_store": sentinel, "observe": True})]
        checks.append("original backend delegates unchanged")
    finally:
        if prior_torch is sentinel:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = prior_torch
    print(json.dumps({"status": "CPU_FAKE_METADATA_CHECK_PASS", "checks": checks,
                      "source_sha256": hashes,
                      "limit": "Fake tensor metadata/launcher only; no GPU, kernel, Graph, model or speedup validation"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
