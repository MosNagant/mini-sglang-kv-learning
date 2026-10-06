"""Coach-supplied M05 diagnostic: eager submission versus CUDA Graph replay.

Fixed B=1/D=64/C=1024, original CUDA only; no operator optimization claim.
Reuses the learner's reviewed 100-call event measurement from its archive.
Graph capture/replay is a separate execution policy, not interchangeable data.
"""

import ast
import importlib.util
import inspect
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import traceback
from pathlib import Path


def main():
    repo = Path.cwd()
    archived = (repo / "experiments/learning/m05-event-first"
                / "run-vtqqu_54/submitted_timing.py")
    # Freeze the already reviewed timing body; refuse to label a different
    # implementation as the same 100-call measurement.
    expected = ast.parse("""
def measure_once(call, start, end):
    start.record()
    for _ in range(100):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100
""").body[0]
    tree = ast.parse(archived.read_text(encoding="utf-8-sig"))
    actual = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "measure_once")
    if ast.dump(actual) != ast.dump(expected):
        raise RuntimeError("Archived timing differs from reviewed body; submit it for review.")
    parent = repo / "experiments/learning/m05-event-graph"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    shutil.copyfile(__file__, run / "diagnostic.py")
    shutil.copyfile(archived, run / "learner_source.py")
    result = {"status": "STARTED", "scope": "method diagnostic, not final baseline",
              "calls_per_sample": 100, "batch": 1, "width": 64,
              "capacity": 1024, "indices": [0], "layout": "interleaved",
              "samples_ms_per_call": {"eager": [], "graph": []}}
    print("RUN_DIRECTORY:", run, flush=True)
    try:
        import torch
        from minisgl.kernel import store_cache

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if commit != "9a91cfafe754aa85daee49998176275667eb58f2":
            raise RuntimeError("Upstream commit changed; review before comparing.")
        result.update(commit=commit, torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0),
                      tracked_git_status=subprocess.check_output(
                          ["git", "status", "--porcelain", "--untracked-files=no"], text=True))
        spec = importlib.util.spec_from_file_location("reviewed_timing", archived)
        learner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(learner)
        result["reviewed_measure_once"] = inspect.getsource(learner.measure_once)

        pool = torch.full((1024, 2, 64), -1.0, dtype=torch.float16, device="cuda")
        kc, vc = pool[:, 0, :], pool[:, 1, :]
        k = torch.arange(64, dtype=torch.float16, device="cuda").reshape(1, 64) + 10
        v = k + 1000
        indices = torch.tensor([0], dtype=torch.int32, device="cuda")
        k_before, v_before, index_before = k.clone(), v.clone(), indices.clone()
        expected_pool = pool.clone()
        expected_pool[0, 0] = k_before[0]
        expected_pool[0, 1] = v_before[0]
        result.update(dtype=str(k.dtype), input_stride=list(k.stride()),
                      cache_stride=list(kc.stride()))

        def call():
            store_cache(kc, vc, indices, k, v)

        def check():
            assert torch.equal(pool, expected_pool), "cache mismatch"
            assert torch.equal(k, k_before), "K input changed"
            assert torch.equal(v, v_before), "V input changed"
            assert torch.equal(indices, index_before), "indices changed"

        call()  # Compilation outside measurement and capture.
        torch.cuda.synchronize()
        check()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(20):
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(100):
                call()

        # Check that replay actually writes: do not accept the previously
        # correct eager output as evidence that capture worked.
        pool.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        check()
        print("REPLAY_CORRECTNESS_OK: targets, untouched storage and inputs", flush=True)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        end.synchronize()

        def measure_graph():
            start.record()
            graph.replay()  # One replay contains 100 captured calls.
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 100

        # Alternate order across 30 pairs. Same data/addresses for both modes.
        for pair in range(30):
            order = ("eager", "graph") if pair % 2 == 0 else ("graph", "eager")
            for method in order:
                ms = (learner.measure_once(call, start, end)
                      if method == "eager" else measure_graph())
                if not math.isfinite(ms) or ms < 0:
                    raise ValueError(f"Invalid timing: {method} {ms}")
                result["samples_ms_per_call"][method].append(ms)
            check()  # Outside both timed intervals.
        result["summary_us_per_call"] = {}
        for method, values in result["samples_ms_per_call"].items():
            summary = {"median": statistics.median(values) * 1000,
                       "min": min(values) * 1000, "max": max(values) * 1000}
            result["summary_us_per_call"][method] = summary
            print(f"{method.upper()}_US_PER_CALL:", summary, flush=True)
        result.update(status="DIAGNOSTIC_COMPLETE",
                      policy="30 alternating pairs; same reused tensors; 100 calls/interval",
                      limit="graph replay versus eager; not optimization speedup or pure kernel/DRAM measurement")
    except BaseException:
        result.update(status="ERROR", error=traceback.format_exc())
        raise
    finally:
        (run / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("SAVED:", run / "result.json", flush=True)
    print("DIAGNOSTIC_COMPLETE: compare execution policies; no final baseline PASS.")


if __name__ == "__main__":
    main()
