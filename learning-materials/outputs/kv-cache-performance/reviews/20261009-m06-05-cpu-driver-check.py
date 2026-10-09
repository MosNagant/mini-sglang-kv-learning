"""CPU simulation of the actual M06-05 run_case and both checked call paths.

Reuses the earlier fake tensor definitions and the accepted wrapper/checker
snapshots. CUDA tags, synchronization, and kernels are simulated; no real
PyTorch/Triton imports or GPU calls occur. Actual runner files are written in
a temporary directory, read back, and removed. Only this review and its JSON
are permanent. This checks the driver and serialization, not GPU correctness.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent.parent
ACCEPTED = HERE / "logs/m06-wrapper-results/run-20261009-134214-o635nwfm"
EVENTS = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load(HERE / "reviews/20261009-m06-04-wrapper-cpu-check.py", "previous_cpu_fake")


class Tensor(legacy.Tensor):
    """Preserve real NumPy alias addresses and model empty torch view strides."""
    def __getitem__(self, key):
        array = self.array[key]
        explicit = None
        if array.size == 0:
            keys = key if isinstance(key, tuple) else (key,)
            keys += (slice(None),) * (self.ndim - len(keys))
            explicit = tuple(stride * (part.step or 1)
                             for part, stride in zip(keys, self.stride())
                             if isinstance(part, slice))
        return Tensor(array, self.device, strides=explicit)

    def cuda(self):
        return Tensor(self.array.copy(), legacy.Device(), strides=self.strides)

    def cpu(self):
        EVENTS.append(dict(event="cpu", shape=list(self.shape), dtype=str(self.dtype)))
        self.counters["cpu"] += 1
        return Tensor(self.array.copy(), legacy.Device("cpu"), strides=self.strides,
                      counters=self.counters)

    def tolist(self):
        EVENTS.append(dict(event="tolist", shape=list(self.shape)))
        return super().tolist()

    def numpy(self):
        return self.array


def empty(shape, dtype, device):
    steps = []
    stride = 1
    for extent in reversed(shape):
        steps.append(stride)
        stride *= max(extent, 1)
    return Tensor(np.empty(shape, dtype=dtype), legacy.Device(device), strides=tuple(reversed(steps)))


def synchronize():
    EVENTS.append(dict(event="synchronize"))


FAKE_TORCH = types.SimpleNamespace(
    Tensor=Tensor, float16=np.dtype("float16"), int32=np.dtype("int32"),
    from_numpy=lambda a: Tensor(a, legacy.Device("cpu")), empty=empty,
    equal=lambda a, b: np.array_equal(a.array, b.array),
    cuda=types.SimpleNamespace(synchronize=synchronize),
)


class CpuScatter:
    def __init__(self):
        self.calls = []
        self.corrupt_once = False

    def scatter(self, kc, vc, indices, k, v):
        for pid, pos in enumerate(indices.array):
            kc.array[pos] = k.array[pid]
            vc.array[pos] = v.array[pid]
        if self.corrupt_once:
            self.corrupt_once = False
            kc.array[indices.array[-1], -1] = -99

    def original(self, kc, vc, indices, k, v):
        self.calls.append(dict(operator="cpu_original_scatter", input_rows=k.shape[0]))
        self.scatter(kc, vc, indices, k, v)

    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **parameters):
            self.calls.append(dict(grid=list(grid), parameters=parameters))
            self.scatter(kc, vc, indices, k, v)
        return call


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_files(directory, before, cfg, implementation, expect_failed=False):
    expected_files = {"metadata.json", "before.npz", "after.npz", "submissions.json", "status.json"}
    assert {p.name for p in directory.iterdir()} == expected_files
    metadata = read_json(directory / "metadata.json")
    status = read_json(directory / "status.json")
    submissions = read_json(directory / "submissions.json")
    assert metadata["config"] == cfg and metadata["implementation"] == implementation
    assert status["config"] == cfg and status["implementation"] == implementation
    assert status["status"] == ("FAILED" if expect_failed else ("PASS" if cfg["batch"] else "EMPTY_ADAPTER_PASS"))
    with np.load(directory / "before.npz") as saved_before:
        assert set(saved_before.files) == set(before)
        for name, value in before.items():
            np.testing.assert_array_equal(saved_before[name], value)
    with np.load(directory / "after.npz") as saved_after:
        assert set(saved_after.files) == {"cache", "k", "v", "indices"}
        for name in saved_after.files:
            if expect_failed and name == "cache":
                assert not np.array_equal(saved_after[name], before["expected_cache"])
            else:
                np.testing.assert_array_equal(saved_after[name], before["expected_cache" if name == "cache" else name])
    count = int(cfg["batch"] > 0)
    assert len(submissions) == count
    if count:
        d = cfg["width"]
        src = d + cfg["gap"]
        dst = (d + cfg["guard"]) * (2 if cfg["layout"] == "interleaved" else 1)
        if implementation == "learner_triton":
            expected = dict(WIDTH=d, SRC_ROW_STRIDE=src, DST_ROW_STRIDE=dst,
                            BLOCK_SIZE=1 << (d - 1).bit_length(), num_warps=4)
            assert submissions == [dict(grid=[cfg["batch"]], parameters=expected)]
        else:
            assert submissions[0]["input_rows"] == cfg["batch"]
            assert submissions[0]["width"] == d
            assert submissions[0]["src_stride"] == src and submissions[0]["dst_stride"] == dst
    if cfg["batch"] == 0:
        assert metadata["tensors"][2]["stride"] == [1]
        assert metadata["tensors"][3]["stride"] == [cfg["width"] + cfg["gap"], 1]
        assert metadata["tensors"][4]["stride"] == [cfg["width"] + cfg["gap"], 1]
    return status, submissions, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def run():
    runner = load(HERE / "23-check-kv-common.py", "common_cpu_driver")
    support = load(HERE / "22-kv-contract-support.py", "common_real_support")
    api_source = (ACCEPTED / "triton_store_api.py").read_text(encoding="utf-8-sig")
    runner.w.review_api(api_source)
    assert (ACCEPTED / "kv_contract_support.py").read_bytes() == (HERE / "22-kv-contract-support.py").read_bytes()
    stub = CpuScatter()
    probe = runner.w.LaunchProbe(stub)
    original_adapter = runner.OriginalAdapter(stub.original, support)
    mocks = {"torch": FAKE_TORCH, "kv_contract_support": support,
             "triton_store_mask": types.SimpleNamespace(store_kv_v1=probe)}
    records = []
    pair_count = 0
    failed_control = None
    with patch.dict(sys.modules, mocks), tempfile.TemporaryDirectory(prefix="m06-05-cpu-driver-") as temp:
        destination = Path(temp)
        api = types.ModuleType("accepted_wrapper_cpu_only")
        exec(compile(api_source, str(ACCEPTED / "triton_store_api.py"), "exec"), api.__dict__)
        checker = load(ACCEPTED / "check_kv_targets.py", "accepted_checker_cpu_only").check_kv_targets
        for index, cfg in enumerate(runner.configurations()):
            before = runner.fixture(cfg)
            immutable_before = {n: a.copy() for n, a in before.items()}
            outputs = []
            for implementation in runner.IMPLEMENTATIONS:
                EVENTS.clear()
                stub.calls.clear()
                directory = destination / f"case-{index:03d}-{implementation}"
                with contextlib.redirect_stdout(io.StringIO()):
                    record, after = runner.run_case(FAKE_TORCH, original_adapter, api, probe, checker,
                                                   cfg, implementation, directory, before)
                saved_status, submissions, hashes = verify_files(directory, before, cfg, implementation)
                assert saved_status == record
                assert len(stub.calls) == int(cfg["batch"] > 0)
                sync_at = next(i for i, e in enumerate(EVENTS) if e["event"] == "synchronize")
                assert sum(e["event"] == "synchronize" for e in EVENTS) == 1
                before_sync = EVENTS[:sync_at]
                if cfg["batch"]:
                    assert before_sync == [dict(event="cpu", shape=[cfg["batch"]], dtype="int32"),
                                           dict(event="tolist", shape=[cfg["batch"]])]
                else:
                    assert before_sync == []
                assert sum(e["event"] == "cpu" for e in EVENTS[sync_at+1:]) == 4
                for name, value in immutable_before.items():
                    np.testing.assert_array_equal(before[name], value, err_msg="CPU_ORACLE_MUTATED")
                outputs.append(after)
                records.append(dict(case=index, implementation=implementation, config=cfg,
                                    status="CPU_DRIVER_PASS", real_runner_status=record["status"],
                                    submissions=submissions, pre_readback_index_events=before_sync,
                                    serialized_files_verified=sorted(hashes), serialized_sha256=hashes))
            for name in ("cache", "k", "v", "indices"):
                np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
            pair_count += 1
        # Force a wrong stored K value AFTER a valid submission. The real driver
        # must save raw after.npz and submissions.json and mark this case FAILED.
        cfg = next(c for c in runner.configurations() if c["batch"] == 17 and c["width"] == 192 and c["guard"] == 64)
        before = runner.fixture(cfg)
        directory = destination / "deliberate-bad-result"
        EVENTS.clear()
        stub.calls.clear()
        stub.corrupt_once = True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_case(FAKE_TORCH, original_adapter, api, probe, checker, cfg,
                                "learner_triton", directory, before)
        except AssertionError:
            pass
        else:
            raise AssertionError("BAD_RESULT_CONTROL_WAS_NOT_REJECTED")
        status, submissions, hashes = verify_files(directory, before, cfg, "learner_triton", expect_failed=True)
        assert status["error_type"] == "AssertionError"
        assert len(probe.attempts) == 1 and len(stub.calls) == 1
        failed_control = dict(status="CPU_BAD_RESULT_REJECTED_AND_SAVED", config=cfg,
                              real_runner_status=status, submissions=submissions,
                              serialized_files_verified=sorted(hashes), serialized_sha256=hashes,
                              meaning="Intentional CPU scatter corruption; not a GPU failure")
    assert len(records) == 196 and pair_count == 98
    assert sum(bool(r["submissions"]) for r in records) == 168
    assert sum(not r["submissions"] for r in records) == 28
    source_paths = [HERE / n for n in ("23-check-kv-common.py", "22-check-kv-wrapper.py",
                    "22-kv-contract-support.py", "21-check-kv-mask.py", "14-check-kv-widths.py",
                    "15-check-kv-contract.py", "reviews/20261009-m06-04-wrapper-cpu-check.py")]
    source_paths += [ACCEPTED / n for n in ("triton_store_api.py", "check_kv_targets.py", "kv_contract_support.py")]
    return dict(status="CPU_COMMON_DRIVER_CHECK_PASS", GPU="NOT_RUN", learner_new_acceptance="NOT_INFERRED",
                method="Actual 23.run_case, OriginalAdapter, 22.LaunchProbe, current support and archived accepted wrapper/checker; NumPy CPU scatter replaces both GPU kernels",
                cases=196, pairs=98, nonempty_cpu_submissions=168, empty_returns=28,
                temporary_serialization="All five runner files per case were written, read back, and checked in system temp; temporary directories then removed",
                source_hashes={str(p.relative_to(HERE)).replace(chr(92), "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                results=records, deliberate_failure_control=failed_control)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    result = run()
    result["test_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print("CPU_COMMON_DRIVER_CHECK_PASS: 196 cases, 98 pairs, 168 CPU submissions, 28 empty returns; bad-result preservation PASS; GPU NOT_RUN")
