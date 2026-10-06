"""M05 learning exercise: one configuration, not a completed baseline.

Coach: data, finite-value checks, records, repetition and event construction.
Learner: warmup and measure_once bodies. Submit these bodies for review.
The prepared ownership exercise is still pending; this scaffold does not
establish independent implementation of data preparation or correctness tests.
"""

import json
import math
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import torch
from minisgl.kernel import store_cache


def warmup(call, repeats):
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()


def measure_once(call, start, end):
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main():
    repo = Path.cwd()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=repo
    ).strip()
    if commit != "9a91cfafe754aa85daee49998176275667eb58f2":
        raise RuntimeError(f"Review changed upstream commit before timing: {commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True, cwd=repo
    )
    capacity, batch, width = 8, 3, 64
    rows = [7, 0, 3]
    pool = torch.full((capacity, 2, width), -1.0,
                      dtype=torch.float16, device="cuda")
    k_cache, v_cache = pool[:, 0, :], pool[:, 1, :]
    k = torch.arange(batch * width, dtype=torch.float16,
                     device="cuda").reshape(batch, width) + 10
    v = k + 1000
    indices = torch.tensor(rows, dtype=torch.int32, device="cuda")
    expected_k, expected_v = k.clone(), v.clone()
    before_indices = indices.clone()
    expected_pool = pool.clone()
    for pid, pos in enumerate(rows):
        expected_pool[pos, 0] = expected_k[pid]
        expected_pool[pos, 1] = expected_v[pid]

    def call():
        # This wrapper submits the original CUDA operator; it is not a timer.
        store_cache(k_cache, v_cache, indices, k, v)

    def check():
        assert torch.equal(pool, expected_pool), "cache targets/untouched mismatch"
        assert torch.equal(k, expected_k), "K input changed"
        assert torch.equal(v, expected_v), "V input changed"
        assert torch.equal(indices, before_indices), "indices changed"

    call()  # Initial JIT/setup and finite-value check are outside measurement.
    torch.cuda.synchronize()
    check()
    print("FIXED_CASE_CHECK_OK B=3 D=64 capacity=8 indices=[7,0,3]")

    # These are supplied objects, not timestamps read on the CPU.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize the CUDA event resources outside the learner's measurement.
    start.record()
    end.record()
    end.synchronize()
    warmup(call, 20)

    # Teaching policy: 30 individually synchronized samples, reusing tensors.
    # This small, reused working set is not a DRAM bandwidth benchmark.
    samples_ms = []
    for _ in range(30):
        value = float(measure_once(call, start, end))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid reported elapsed_ms: {value}")
        samples_ms.append(value)
    torch.cuda.synchronize()
    check()

    parent = repo / "experiments" / "learning" / "m05-event-first"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    shutil.copyfile(__file__, run / "submitted_timing.py")
    result = {
        "scope": "single-case event exercise; measurement body needs review",
        "commit": commit, "tracked_git_status": status,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch": batch, "width": width, "capacity": capacity,
        "indices": rows, "dtype": str(k.dtype),
        "k_stride": list(k.stride()), "cache_stride": list(k_cache.stride()),
        "warmup_requested": 20, "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "policy": "one call per sample; same tensors; synchronized samples",
        "limit": "event interval may include host submission gaps; no pure-kernel or DRAM claim",
    }
    (run / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("SAMPLES_MS:", samples_ms)
    print("MEDIAN_MS:", result["median_ms"])
    print("MEDIAN_US:", result["median_ms"] * 1000)
    print("SAVED:", run)
    print("REVIEW_REQUIRED: submit timing bodies and output; no baseline PASS claimed.")

if __name__ == "__main__":
    main()
