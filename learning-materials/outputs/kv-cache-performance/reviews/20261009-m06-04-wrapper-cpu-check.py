"""CPU-only execution of the coach reference public wrapper and real support.

Uses NumPy-backed fake tensors with fake CUDA tags; does not import torch,
Triton, or the real store kernel. The three reference expressions exist only
in memory, never in a learner file. A CPU scatter stub tests wrapper plumbing
and fixture checks, NOT the real GPU kernel. Run with Python -B; the JSON is
created exclusively so a later failure cannot overwrite prior evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    def __init__(self, kind="cuda", index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __ne__(self, other):
        return not self == other


class Tensor:
    def __init__(self, array, device=None, strides=None, counters=None):
        self.array = array
        self.device = Device() if device is None else device
        self.strides = strides
        self.counters = {"cpu": 0, "tolist": 0} if counters is None else counters

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype

    def stride(self, dimension=None):
        value = self.strides
        if value is None:
            value = tuple(n // self.array.itemsize for n in self.array.strides)
        return value if dimension is None else value[dimension]

    def data_ptr(self):
        return self.array.__array_interface__["data"][0]

    def element_size(self):
        return self.array.itemsize

    def __getitem__(self, key):
        return Tensor(self.array[key], self.device)

    def cpu(self):
        self.counters["cpu"] += 1
        return Tensor(self.array.copy(), Device("cpu"), counters=self.counters)

    def tolist(self):
        self.counters["tolist"] += 1
        return self.array.tolist()

    def float(self):
        return Tensor(self.array.astype(np.float32), self.device)

    def long(self):
        return Tensor(self.array.astype(np.int64), self.device)


class CpuScatterRecorder:
    """Records launch syntax and simulates its intended scatter on the CPU."""
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **parameters):
            self.calls.append({"grid": list(grid), "parameters": parameters})
            for pid, position in enumerate(indices.array):
                kc.array[position] = k.array[pid]
                vc.array[position] = v.array[pid]
        return call


def packet(runner, cfg):
    before = runner.fixture(cfg)
    arrays = {n: before[n].copy() for n in ("cache", "k", "v", "indices")}
    d = cfg["width"]
    kc, vc = runner.h.cache_views(arrays["cache"], cfg["layout"])
    # NumPy empty views have zero strides; ordinary torch.empty/sliced views
    # preserve the specified row stride and use index stride 1.
    source_strides = (d + cfg["gap"], 1) if cfg["batch"] == 0 else None
    index_strides = (1,) if cfg["batch"] == 0 else None
    args = [Tensor(kc[:, :d]), Tensor(vc[:, :d]),
            Tensor(arrays["indices"], strides=index_strides),
            Tensor(arrays["k"][:, :d], strides=source_strides),
            Tensor(arrays["v"][:, :d], strides=source_strides)]
    return before, arrays, args


def bad_packet(runner, kind):
    cfg = runner.configurations()[0].copy()
    if kind == "unsupported_width":
        cfg["width"] = 96
    if kind == "unsupported_source_stride":
        cfg["gap"] = 32
    if kind == "empty_with_one_index":
        cfg.update(batch=0, rows=[])
    before, arrays, a = packet(runner, cfg)
    if kind == "indices_too_many":
        a[2] = Tensor(np.array([7, 0, 3, 5, 1], dtype=np.int32))
    elif kind == "indices_too_few":
        a[2] = a[2][:2]
    elif kind == "v_shape":
        a[4] = a[4][:2]
    elif kind == "k_dtype":
        a[3] = a[3].float()
    elif kind == "indices_dtype":
        a[2] = a[2].long()
    elif kind == "cpu_k":
        a[3] = a[3].cpu()
    elif kind == "cpu_indices":
        a[2] = a[2].cpu()
    elif kind == "column_stride":
        a[3] = Tensor(np.zeros((3, 128), dtype=np.float16))[:, ::2]
        a[4] = Tensor(np.zeros((3, 128), dtype=np.float16))[:, ::2]
    elif kind == "index_stride":
        a[2] = Tensor(np.array([7, 0, 0, 0, 3, 0], dtype=np.int32))[::2]
    elif kind == "unequal_source_stride":
        a[4] = Tensor(np.zeros((3, 128), dtype=np.float16))[:, :64]
    elif kind == "cache_overlap":
        a[1] = a[0]
    elif kind == "source_cache_overlap":
        a[3], a[4] = a[0][:3], a[1][:3]
    elif kind in ("duplicate_indices", "negative_index", "out_of_range_index", "empty_with_one_index"):
        values = {"duplicate_indices": [7, 7, 3], "negative_index": [-1, 0, 3],
                  "out_of_range_index": [8, 0, 3], "empty_with_one_index": [7]}[kind]
        a[2] = Tensor(np.array(values, dtype=np.int32))
    for tensor in a:
        tensor.counters.update(cpu=0, tolist=0)
    return before, arrays, a


def run():
    runner = load(HERE / "22-check-kv-wrapper.py", "cpu_wrapper_runner")
    support = load(HERE / "22-kv-contract-support.py", "cpu_real_support")
    reference = runner.reference_source()
    runner.review_api(reference)
    stub = CpuScatterRecorder()
    probe = runner.LaunchProbe(stub)
    fake_torch = types.SimpleNamespace(Tensor=Tensor, float16=np.dtype("float16"), int32=np.dtype("int32"))
    mocked_modules = {"torch": fake_torch, "kv_contract_support": support,
                      "triton_store_mask": types.SimpleNamespace(store_kv_v1=probe)}
    records = []
    expected_rejection = {
        "indices_too_many": "INDEX_COUNT", "indices_too_few": "INDEX_COUNT",
        "v_shape": "INPUT_SHAPE", "k_dtype": "TENSOR_DTYPE", "indices_dtype": "TENSOR_DTYPE",
        "cpu_k": "CUDA_DEVICE", "cpu_indices": "CUDA_DEVICE", "unsupported_width": "WIDTH_SCOPE",
        "column_stride": "COLUMN_STRIDE", "index_stride": "INDEX_STRIDE",
        "unequal_source_stride": "SHARED_SOURCE_STRIDE", "unsupported_source_stride": "SOURCE_LAYOUT_SCOPE",
        "cache_overlap": "CACHE_WRITE_ALIAS", "source_cache_overlap": "SOURCE_CACHE_ALIAS",
        "duplicate_indices": "INDEX_DUPLICATE", "negative_index": "INDEX_RANGE",
        "out_of_range_index": "INDEX_RANGE", "empty_with_one_index": "INDEX_COUNT",
    }
    assert set(expected_rejection) == set(runner.INVALID_KINDS)
    with patch.dict(sys.modules, mocked_modules):
        api = {"__name__": "coach_reference_in_memory"}
        exec(compile(reference, "<coach-reference-only>", "exec"), api)
        for cfg in runner.configurations():
            before, arrays, a = packet(runner, cfg)
            probe.reset(cfg["batch"] > 0)
            stub.calls.clear()
            api["store_cache_triton"](*a)
            expected_count = int(cfg["batch"] > 0)
            assert len(probe.attempts) == expected_count
            assert len(stub.calls) == expected_count
            assert a[2].counters == {"cpu": expected_count, "tolist": expected_count}
            if expected_count:
                d = cfg["width"]
                parameters = dict(WIDTH=d, SRC_ROW_STRIDE=d + cfg["gap"],
                                  DST_ROW_STRIDE=(d + cfg["guard"]) * (2 if cfg["layout"] == "interleaved" else 1),
                                  BLOCK_SIZE=1 << (d - 1).bit_length(), num_warps=4)
                assert stub.calls == [dict(grid=[cfg["batch"]], parameters=parameters)]
            runner.h.verify_backing(before, arrays)
            records.append(dict(kind="positive", status="CPU_PASS", config=cfg,
                                launches=stub.calls.copy(), index_transfers=a[2].counters.copy(),
                                full_backing="CPU_STUB_MATCHES_FIXTURE"))
        for kind in runner.INVALID_KINDS:
            before, arrays, a = bad_packet(runner, kind)
            saved = [tensor.array.copy() for tensor in a]
            probe.reset(False)
            stub.calls.clear()
            rejection = None
            try:
                api["store_cache_triton"](*a)
            except (TypeError, ValueError) as error:
                rejection = {"type": type(error).__name__, "message": str(error)}
            assert rejection is not None, kind
            assert rejection["message"].startswith(expected_rejection[kind] + ":"), (kind, rejection)
            assert not probe.attempts and not stub.calls, kind
            for old, tensor in zip(saved, a):
                np.testing.assert_array_equal(old, tensor.array, err_msg=kind)
            index_value_case = kind in ("duplicate_indices", "negative_index", "out_of_range_index")
            expected_count = int(index_value_case)
            assert a[2].counters == {"cpu": expected_count, "tolist": expected_count}, (kind, a[2].counters)
            owner = "validate_index_values" if index_value_case else (
                "learner_outer_index_count" if expected_rejection[kind] == "INDEX_COUNT" else "validate_metadata")
            records.append(dict(kind=kind, status="CPU_REJECTED_BEFORE_KV_LAUNCH", rejection=rejection,
                                rejection_owner=owner, kv_launches=0, index_transfers=a[2].counters.copy()))

    variants = {
        "wrong_empty_condition": reference.replace("if batch == 0:", "if batch == 3:"),
        "wrong_grid": reference.replace("store_kv_v1[(batch,)]", "store_kv_v1[(width,)]"),
        "wrong_index_length": reference.replace("indices.shape != (batch,)", "indices.shape != (width,)"),
    }
    for name, source in variants.items():
        assert source != reference
        try:
            runner.review_api(source)
        except RuntimeError as error:
            assert str(error).startswith("WRAPPER_SOURCE_REVIEW_REQUIRED"), str(error)
            records.append(dict(kind=name, status="CPU_PREFLIGHT_REJECTED", message=str(error)))
        else:
            raise AssertionError("bad reference accepted: " + name)
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ("22-check-kv-wrapper.py", "22-triton-store-api-template.py",
                            "22-kv-contract-support.py", "21-check-kv-mask.py", "14-check-kv-widths.py")}
    return dict(status="CPU_REFERENCE_WRAPPER_CHECK_PASS", GPU="NOT_RUN", learner="NOT_RUN",
                scope="Coach reference executed only in memory; 6 complete public calls, 18 invalid calls, 3 bad-source controls; fake tensor CPU metadata plus CPU scatter stub, not GPU kernel validation",
                cases=len(records), source_hashes=sources, results=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = run()
    result["test_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(result["status"] + ": 6 positive, 18 invalid, 3 preflight controls; GPU NOT_RUN; learner NOT_RUN")
