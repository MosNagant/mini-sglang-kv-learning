"""CPU-only metadata/dispatch checks and NumPy interpretation, not GPU evidence."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "40-m08-kv-backends.py"
before_imports = set(sys.modules)
spec = importlib.util.spec_from_file_location("m08_backends_cpu", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not ({"torch", "triton"} & (set(sys.modules) - before_imports))


class Vector(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(Vector)


def vec(value):
    return np.asarray(value).view(Vector)


class Device:
    def __init__(self, name="cuda:0"):
        self.name = name
        self.type = name.split(":")[0]

    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name


class Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, value):
        return Pointer(self.data, self.offset + value)


class Tensor:
    def __init__(self, shape, strides, dtype="float16", device="cuda:0", layout="strided"):
        self.shape, self.strides, self.dtype = tuple(shape), tuple(strides), dtype
        self.ndim, self.device, self.layout = len(shape), Device(device), layout
        length = max(1, 1 + sum(max(0, n - 1) * s for n, s in zip(shape, strides)))
        self.data = np.full(length, -77, dtype=np.int32 if dtype == "int32" else np.float16)
        self.address = 0x1000

    def stride(self, axis):
        return self.strides[axis]

    def data_ptr(self):
        return self.address

    def __add__(self, value):
        return Pointer(self.data, value)

    def view(self, rows, width):
        result = object.__new__(Tensor)
        result.__dict__ = self.__dict__.copy()
        result.shape, result.strides, result.ndim = (rows, width), (self.stride(0), 1), 2
        return result


fake_torch = types.ModuleType("torch")
fake_torch.Tensor = Tensor
fake_torch.strided = "strided"
fake_torch.float16, fake_torch.int32 = "float16", "int32"
sys.modules["torch"] = fake_torch
tl = types.ModuleType("triton.language")
tl.constexpr, tl.int64 = object(), np.int64
tl.program_id = lambda axis: vec(current_row[0])
tl.arange = lambda start, end: vec(np.arange(start, end))


def load(pointer, mask=None, other=0):
    offsets = np.asarray(pointer.offset)
    mask = np.ones_like(offsets, dtype=bool) if mask is None else np.asarray(mask)
    selected = offsets[mask]
    assert np.all((selected >= 0) & (selected < len(pointer.data))), "unmasked OOB load"
    result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
    result[mask] = pointer.data[selected]
    return vec(result)


def store(pointer, value, mask):
    offsets, mask = np.asarray(pointer.offset), np.asarray(mask)
    selected = offsets[mask]
    assert np.all((selected >= 0) & (selected < len(pointer.data))), "unmasked OOB store"
    pointer.data[selected] = np.asarray(value)[mask]


tl.load, tl.store = load, store
current_row, launches = [0], []


class JIT:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            launches.append({"grid": list(grid), **kwargs})
            assert kwargs.pop("num_warps") == 4
            for row in range(grid[0]):
                current_row[0] = row
                self.fn(*args, **kwargs)
            return {"cpu_mock_compiled_kernel": True}
        return launch


triton = types.ModuleType("triton")
triton.jit = JIT
triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
triton.language = tl
sys.modules["triton"], sys.modules["triton.language"] = triton, tl


def fixture(width=1024, batch=3, capacity=7, independent=False, mixed=False):
    head = 128 if width == 1024 else 64
    cache_shape = (capacity, width // head, head) if width % head == 0 else (capacity, width)
    cache_tail = (head, 1) if len(cache_shape) == 3 else (1,)
    k = Tensor((batch, width), (width * 4, 1))
    v = Tensor((batch, width), (width * (5 if independent else 4), 1))
    if mixed and width % head == 0:
        v.shape, v.strides, v.ndim = (batch, width // head, head), (v.stride(0), head, 1), 3
    kc = Tensor(cache_shape, (width + (1 if independent else 0), *cache_tail))
    vc = Tensor(cache_shape, (width + (3 if independent else 0), *cache_tail))
    indices = Tensor((batch,), (1,), "int32")
    for row in range(batch):
        k.data[row * k.stride(0):row * k.stride(0) + width] = (np.arange(width) + row * 7) % 301
        v.data[row * v.stride(0):row * v.stride(0) + width] = -((np.arange(width) + row * 11) % 301)
    indices.data[:batch] = [capacity - 1, 0, 3][:batch]
    return [kc, vc, indices, k, v]


def forbidden_original(**kwargs):
    raise AssertionError("Triton fell back to original")


tested = []
for width in (64, 65, 192, 1024):
    args = fixture(width, independent=True, mixed=True)
    kc, vc, indices, k, v = args
    before = [t.data.copy() for t in args]
    expected_k, expected_v = before[0].copy(), before[1].copy()
    for row, dst in enumerate(indices.data):
        expected_k[dst * kc.stride(0):dst * kc.stride(0) + width] = k.data[row * k.stride(0):row * k.stride(0) + width]
        expected_v[dst * vc.stride(0):dst * vc.stride(0) + width] = v.data[row * v.stride(0):row * v.stride(0) + width]
    adapter = module.make_store_cache("triton", original_store=forbidden_original, observe=True)
    adapter(*args)
    assert np.array_equal(kc.data, expected_k) and np.array_equal(vc.data, expected_v)
    assert all(np.array_equal(t.data, b) for t, b in zip(args[2:], before[2:]))
    assert adapter.call_count == adapter.launch_count == 1
    assert adapter.last_launch["grid"] == [3]
    assert adapter.last_launch["BLOCK_D"] == triton.next_power_of_2(width)
    assert adapter.compiled_kernel == {"cpu_mock_compiled_kernel": True}
    tested.append(f"numpy_interpreted_kernel_D{width}_four_independent_strides")

adapter = module.make_store_cache("triton", observe=True)
initial_launches = len(launches)
adapter(*fixture(batch=0, capacity=0))
assert len(launches) == initial_launches and adapter.call_count == adapter.launch_count == 0
tested.append("B0_C0_no_launch_no_counter")
quiet = module.make_store_cache("triton")
quiet(*fixture())
assert quiet.call_count == quiet.launch_count == 0 and quiet.last_launch is None
tested.append("observe_false_no_recording")

bad_cases = {
    "cpu_tensor": lambda a: setattr(a[3], "device", Device("cpu")),
    "different_gpu": lambda a: setattr(a[4], "device", Device("cuda:1")),
    "bf16_input": lambda a: setattr(a[3], "dtype", "bfloat16"),
    "int64_indices": lambda a: setattr(a[2], "dtype", "int64"),
    "sparse_tensor": lambda a: setattr(a[3], "layout", "sparse"),
    "bad_rank": lambda a: setattr(a[3], "ndim", 1),
    "noncontiguous_columns": lambda a: setattr(a[3], "strides", (4096, 2)),
    "noncontiguous_heads": lambda a: setattr(a[0], "strides", (1024, 129, 1)),
    "overlapping_rows": lambda a: setattr(a[3], "strides", (1023, 1)),
    "batch_mismatch": lambda a: setattr(a[4], "shape", (2, 1024)),
    "capacity_mismatch": lambda a: setattr(a[1], "shape", (8, 8, 128)),
    "width_mismatch": lambda a: setattr(a[3], "shape", (3, 1023)),
    "wrong_indices_length": lambda a: setattr(a[2], "shape", (2,)),
    "strided_indices": lambda a: setattr(a[2], "strides", (2,)),
    "too_many_rows": lambda a: [setattr(t, "shape", (2, 8, 128)) for t in a[:2]],
    "not_a_tensor": lambda a: a.__setitem__(3, object()),
}
for name, mutate in bad_cases.items():
    args = fixture()
    mutate(args)
    try:
        adapter(*args)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError(name + " was accepted")
assert adapter.call_count == adapter.launch_count == 0
tested.extend("reject_" + name for name in bad_cases)
for width in (0, 4097):
    try:
        adapter(*fixture(width))
    except ValueError:
        tested.append(f"reject_width_{width}")
    else:
        raise AssertionError("unsupported width accepted")

original_calls = []
original = module.make_store_cache("original", original_store=lambda **kw: original_calls.append(kw), observe=True)
original(*fixture(mixed=True))
assert len(original_calls) == 1 and original.launch_count == 1
assert all(original_calls[0][name].ndim == 2 for name in ("k_cache", "v_cache", "k", "v"))
assert original_calls[0]["k"].stride(0) == 4096
tested.append("original_explicit_dispatch_2D_views_preserve_stride")
for name, args in (("independent_stride", fixture(independent=True)), ("unaligned_width", fixture(65)),
                   ("unaligned_pointer", fixture())):
    if name == "unaligned_pointer":
        args[3].address += 2
    try:
        original(*args)
    except ValueError:
        tested.append("original_reject_" + name)
    else:
        raise AssertionError(name + " accepted")
assert original.launch_count == 1
try:
    module.make_store_cache("auto")
except ValueError:
    tested.append("reject_implicit_backend")
else:
    raise AssertionError("implicit backend allowed")

tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in {"cpu", "item", "contiguous", "synchronize", "cuda", "numpy"}
tested.append("AST_no_value_read_copy_sync_or_device_init")
report = {"status": "CPU_ONLY_PASS", "gpu_run": False,
          "scope": "NumPy execution of kernel Python body with fake Triton; no Triton compilation or GPU evidence",
          "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "python": sys.version, "checks": tested, "check_count": len(tested)}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
