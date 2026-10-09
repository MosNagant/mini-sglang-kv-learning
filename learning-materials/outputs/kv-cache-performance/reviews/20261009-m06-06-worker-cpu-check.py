"""Execute the actual M06-06 worker with CPU NumPy and an in-memory filesystem.

Only this script and its review JSON are written. Worker artifact bytes are
really serialized/read back; hashes/shapes are retained in the JSON. Torch, Triton,
kernel loaders, metadata validation, telemetry and source identity are mocked.
CUDA scheduling/performance/correctness are not tested. GPU NOT_RUN.
"""
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import traceback
import types
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SOURCE = HERE / "25-compare-kv-v1.py"


class MemoryPath:
    def __init__(self, files, path):
        self.files, self.path = files, PurePosixPath(path)

    def __truediv__(self, part):
        return MemoryPath(self.files, self.path / part)

    def __str__(self):
        return str(self.path)

    def mkdir(self, parents=False):
        pass

    def write_text(self, text, encoding="utf-8"):
        self.files[str(self)] = text.encode(encoding)

    def read_text(self, encoding="utf-8"):
        return self.files[str(self)].decode(encoding)


class Tensor:
    def __init__(self, array, world):
        self.array, self.world = array, world

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, key):
        return Tensor(self.array[key], self.world)

    def __add__(self, value):
        return Tensor(self.array + value, self.world)

    def to(self, dtype):
        return Tensor(self.array.astype(dtype), self.world)

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def fill_(self, value):
        self.world.resets.append(dict(value=value, shape=list(self.shape)))
        self.array.fill(value)
        return self


class Generator:
    def manual_seed(self, seed):
        self.random = np.random.default_rng(seed)
        return self


class Stream:
    def wait_stream(self, other):
        pass


class Event:
    def __init__(self, world, enable_timing):
        assert enable_timing
        self.world = world

    def record(self):
        self.tick = self.world.tick

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        count = other.tick - self.tick
        self.world.interval_calls.append(count)
        return count * 0.25  # Synthetic number, never GPU timing evidence.


class Graph:
    def __init__(self, world):
        self.world, self.calls, self.replays = world, [], 0
        world.graphs.append(self)

    def replay(self):
        assert self.calls
        impl = self.calls[0][0]
        self.replays += 1
        if self.replays == 1:
            assert np.all(self.calls[0][1][0].array == -1)
            assert np.all(self.calls[0][1][1].array == -1)
            self.world.reset_replay_proofs.append(impl)
        if self.world.noop_impl == impl:
            return
        for name, args in self.calls:
            self.world.submit(name, args)


class Kernel:
    def __init__(self, world):
        self.world = world

    def __getitem__(self, grid):
        assert grid == (1,)

        def launch(*args, **kwargs):
            assert kwargs == dict(WIDTH=64, SRC_ROW_STRIDE=64,
                                 DST_ROW_STRIDE=128, BLOCK_SIZE=64, num_warps=4)
            self.world.submit("learner_triton", args)
        return launch


class World:
    def __init__(self, noop_impl=None):
        self.noop_impl = noop_impl
        self.files, self.json_history, self.loads = {}, [], []
        self.tick, self.capture = 0, None
        self.graphs, self.resets, self.interval_calls = [], [], []
        self.initial_proofs, self.reset_replay_proofs = [], []
        self.counts = dict(original_cuda=0, learner_triton=0)
        self.identities = {}
        self.snapshot_checks, self.metadata_checks, self.index_checks = 0, 0, 0

    def submit(self, name, args):
        if self.capture is not None:
            self.capture.calls.append((name, args))
            return
        kc, vc, indices, k, v = args
        identity = tuple(t.array.__array_interface__["data"][0] for t in args)
        self.identities.setdefault(name, identity)
        assert self.identities[name] == identity
        if not self.counts[name]:
            assert np.all(kc.array == -1) and np.all(vc.array == -1)
            self.initial_proofs.append(name)
        for row, pos in enumerate(indices.array):
            kc.array[pos] = k.array[row]
            vc.array[pos] = v.array[row]
        self.counts[name] += 1
        self.tick += 1

    @contextlib.contextmanager
    def graph(self, graph, stream):
        assert self.capture is None
        self.capture = graph
        try:
            yield
        finally:
            self.capture = None

    def write_json(self, path, value):
        serialized = json.dumps(value, indent=2).encode("utf-8")
        self.files[str(path)] = serialized
        if str(path).endswith("result.json"):
            self.json_history.append(json.loads(serialized))

    def save_npz(self, path, **arrays):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        self.files[str(path)] = stream.getvalue()

    def validate_metadata(self, kc, vc, indices, k, v):
        self.metadata_checks += 1
        assert kc.shape == vc.shape == (1024, 64)
        assert k.shape == v.shape == (1, 64) and indices.shape == (1,)
        assert kc.array.strides == vc.array.strides == (256, 2)
        assert k.array.strides == v.array.strides == (128, 2)
        return 1, 64, 64, 128, 64

    def validate_indices(self, indices, batch, capacity):
        self.index_checks += 1
        assert (batch, capacity) == (1, 1024)
        np.testing.assert_array_equal(indices.array, np.array([0], dtype=np.int32))

    def load_module(self, path, name):
        self.loads.append(dict(path=str(path), name=name))
        if name == "kv_contract_support":
            return types.SimpleNamespace(validate_metadata=self.validate_metadata,
                                         validate_index_values=self.validate_indices)
        assert name == "triton_store_mask"
        return types.SimpleNamespace(store_kv_v1=Kernel(self))

    def verify_snapshot(self, run, project):
        self.snapshot_checks += 1

    def torch(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda n: "SIMULATED_CPU_NO_GPU",
            get_device_properties=lambda n: "SIMULATED_CPU_NO_GPU",
            synchronize=lambda: None, Stream=Stream, current_stream=Stream,
            stream=lambda side: contextlib.nullcontext(), graph=self.graph,
            CUDAGraph=lambda: Graph(self), Event=lambda **kw: Event(self, **kw))
        return types.SimpleNamespace(
            __version__="2.9.1+cu128", version=types.SimpleNamespace(cuda="12.8"), cuda=cuda,
            Generator=Generator, int32=np.int32, float16=np.float16,
            randint=lambda lo, hi, shape, generator, dtype: Tensor(generator.random.integers(lo, hi, shape, dtype=dtype), self),
            from_numpy=lambda a: Tensor(a, self), equal=lambda a, b: np.array_equal(a.array, b.array))


def namespace(world, source):
    tree = ast.parse(source)
    wanted = {"load_reviewed_timer", "measurement_order", "measure_graph", "make_plan", "worker"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in selected} == wanted
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ("IMPLEMENTATIONS", "POLICIES", "CASES", "TIMER_SOURCE")
                         for t in node.targets)]
    import random
    project = Path("C:/CPU_ONLY_MOCK_PROJECT")
    n = dict(ast=ast, np=types.SimpleNamespace(full=np.full, float16=np.float16,
             arange=np.arange, int32=np.int32, savez_compressed=world.save_npz),
             math=math, random=random, Path=Path, traceback=traceback,
             inspect=types.SimpleNamespace(getsourcefile=lambda fn: project / "python/minisgl/kernel/store.py"),
             h=types.SimpleNamespace(write_json=world.write_json, load_module=world.load_module),
             w=types.SimpleNamespace(tensor_info=lambda t: dict(shape=list(t.shape))),
             COMMIT="CPU_MOCK_PINNED_COMMIT", verify_snapshot=world.verify_snapshot,
             telemetry=lambda: dict(status="MOCKED_CPU_ONLY_GPU_NOT_RUN"))
    exec(compile(ast.Module(body=constants + selected, type_ignores=[]), str(SOURCE), "exec"), n)
    # The real timer AST gate is executed against in-memory canonical source.
    world.files["run/learner-timing.py"] = n["TIMER_SOURCE"].encode()
    return n, project


def decode_npz(data):
    with np.load(io.BytesIO(data)) as archive:
        return {name: archive[name].copy() for name in archive.files}


def artifacts(world):
    record = {}
    for path, data in world.files.items():
        if not path.startswith("worker/"):
            continue
        entry = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if path.endswith(".npz"):
            entry["arrays"] = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                               for key, value in decode_npz(data).items()}
            entry["serialization_readback"] = "VERIFIED"
        elif path.endswith("error.txt"):
            entry["error"] = data.decode("utf-8").splitlines()[-1]
        record[path] = entry
    return record


def one_case(source, noop_impl=None):
    world = World(noop_impl)
    n, project = namespace(world, source)
    mocks = {"torch": world.torch(), "triton": types.SimpleNamespace(__version__="3.5.1"),
             "minisgl": types.ModuleType("minisgl"), "minisgl.kernel": types.ModuleType("minisgl.kernel"),
             "minisgl.kernel.store": types.SimpleNamespace(store_cache=lambda *a: world.submit("original_cuda", a))}
    output = io.StringIO()
    error = None
    with patch.dict(sys.modules, mocks), contextlib.redirect_stdout(output):
        try:
            n["worker"](project, MemoryPath(world.files, "run"), MemoryPath(world.files, "worker"), 1, 64, 1)
        except AssertionError as exc:
            error = str(exc)
    result = json.loads(world.files["worker/result.json"])
    assert world.loads == [dict(path="run/kv_contract_support.py", name="kv_contract_support"),
                           dict(path="run/triton_store_mask.py", name="triton_store_mask")]
    assert world.metadata_checks == world.index_checks == 1
    before = decode_npz(world.files["worker/before.npz"])
    assert before["cache"].shape == (1024, 2, 64) and np.all(before["cache"] == -1)
    np.testing.assert_array_equal(before["expected_cache"][0, 0], before["k"][0])
    np.testing.assert_array_equal(before["expected_cache"][0, 1], before["v"][0])
    assert np.all(before["expected_cache"][1:] == -1)
    for path, data in world.files.items():
        if not path.endswith(".npz") or path.endswith("before.npz") or "/failed-" in path:
            continue
        after = decode_npz(data)
        assert set(after) == {"cache", "k", "v", "indices"}
        for key, value in after.items():
            np.testing.assert_array_equal(value, before["expected_cache" if key == "cache" else key])
    if noop_impl is None:
        assert error is None and result["status"] == "COLLECTED"
        assert result["correctness_checks"] == 127
        assert world.initial_proofs == world.reset_replay_proofs == ["original_cuda", "learner_triton"]
        assert world.counts == dict(original_cuda=6621, learner_triton=6621)
        assert world.identities["original_cuda"] == world.identities["learner_triton"]
        assert len(world.resets) == 4 and all(item["value"] == -1 for item in world.resets)
        assert len(world.graphs) == 2
        assert all(len(graph.calls) == 100 and graph.replays == 36 for graph in world.graphs)
        assert world.interval_calls == [100] * 120
        assert len(result["measurements"]) == 120
        assert world.snapshot_checks == 2
        expected_names = {"result.json", "tensors.json", "before.npz", "final.npz"}
        for impl in ("original_cuda", "learner_triton"):
            expected_names.update((impl + "-initial.npz", impl + "-graph-replay.npz"))
            for policy in ("graph", "eager"):
                assert result["samples_ms_per_call"][impl][policy] == [0.25] * 30
        assert {p.removeprefix("worker/") for p in world.files if p.startswith("worker/")} == expected_names
        forward = [("original_cuda", "graph"), ("learner_triton", "graph"),
                   ("original_cuda", "eager"), ("learner_triton", "eager")]
        for sample in range(30):
            block = result["measurements"][sample * 4:(sample + 1) * 4]
            assert [r["sample"] for r in block] == [sample] * 4
            expected = list(reversed(forward)) if sample % 2 == 0 else forward
            assert [(r["implementation"], r["policy"]) for r in block] == expected
        assert len(world.json_history) == 32
        assert [len(r["measurements"]) for r in world.json_history[1:-1]] == list(range(4, 121, 4))
    else:
        assert error == noop_impl + "-graph-replay: cache mismatch"
        assert result["status"] == "FAILED" and result["error_type"] == "AssertionError"
        assert result["correctness_checks"] == (1 if noop_impl == "original_cuda" else 4)
        assert not result["measurements"]
        assert error.encode() in world.files["worker/error.txt"]
        failed = decode_npz(world.files["worker/failed-" + noop_impl + "-graph-replay.npz"])
        assert np.all(failed["cache"] == -1)
        assert not np.array_equal(failed["cache"], before["expected_cache"])
        for key in ("k", "v", "indices"):
            np.testing.assert_array_equal(failed[key], before[key])
    return dict(status="CPU_WORKER_PASS" if noop_impl is None else "NOOP_GRAPH_REJECTED_AND_SAVED",
                noop_implementation=noop_impl, real_worker_status=result["status"],
                correctness_checks=result["correctness_checks"], initial_sentinel_proofs=world.initial_proofs,
                replay_sentinel_proofs=world.reset_replay_proofs, captured_calls=[len(g.calls) for g in world.graphs],
                graph_replays=[g.replays for g in world.graphs], total_scatter_calls=world.counts,
                measured_interval_call_counts=world.interval_calls, measurement_blocks=len(result["measurements"]) // 4,
                original_and_triton_share_buffers=(len(world.identities) == 2 and len(set(world.identities.values())) == 1),
                stdout=output.getvalue(), artifacts=artifacts(world))


def run():
    source = SOURCE.read_text(encoding="utf-8-sig")
    cases = [one_case(source), one_case(source, "original_cuda"), one_case(source, "learner_triton")]
    return dict(status="CPU_PAIRED_WORKER_CHECK_PASS", GPU="NOT_RUN", performance_evidence="NONE",
                learner_acceptance="NOT_INFERRED", config=dict(batch=1, width=64, round=1),
                method="Unchanged worker/load_reviewed_timer/measurement_order/measure_graph/make_plan AST bodies; NumPy tensors and scatter; capture queues 100 calls and replay executes them",
                mocked=["Torch/CUDA/Triton", "both GPU kernels", "support metadata/index validators", "module loading", "source identity/hash verification", "telemetry", "filesystem"],
                limits="CPU orchestration and serialized evidence checks only; no CUDA capture legality, async streams, JIT, real GPU correctness, timing, environment checks or driver execution",
                artifact_storage="Worker JSON/text/NPZ bytes serialized in memory, read back and verified; review retains hashes/byte lengths/NPZ shapes and failure messages. Raw temporary bytes discarded; no extra filesystem artifacts created.",
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)


if __name__ == "__main__":
    result = run()
    destination = Path(__file__).with_suffix(".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("CPU_PAIRED_WORKER_CHECK_PASS: B1 D64; 127 checks; 120 intervals/100 calls; 30 order blocks; both no-op replay controls rejected and serialized; GPU NOT_RUN")
