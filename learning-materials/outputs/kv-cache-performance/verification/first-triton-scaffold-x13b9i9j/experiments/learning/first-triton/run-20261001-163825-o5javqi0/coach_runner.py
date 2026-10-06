"""Coach harness: snapshots, a fixed fixture, and the learner's earlier checks.

No original mini-sglang kernel is called here. No timings are collected.
The student module must supply store_kv_first and implement its GPU reads/writes.
"""

import argparse
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def check_draft(source):
    tree = ast.parse(source)
    kernel = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "store_kv_first"),
        None,
    )
    if kernel is None:
        raise RuntimeError("Keep the top-level store_kv_first function and its arguments.")
    if any(isinstance(node, ast.Pass) for node in ast.walk(kernel)):
        raise RuntimeError(
            "LEARNER_CODE_PENDING: replace pass with your four read/write lines. "
            "No GPU kernel was launched."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    student = args.student.resolve()
    source = student.read_text(encoding="utf-8-sig")
    parent = student.parent / "learning" / "first-triton"
    parent.mkdir(parents=True, exist_ok=True)
    record_dir = Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent
    ))
    snapshot = record_dir / "triton_store_first.py"
    snapshot.write_text(source, encoding="utf-8")
    shutil.copyfile(__file__, record_dir / "coach_runner.py")
    print("RUN_DIRECTORY=" + str(record_dir), flush=True)
    print("SCOPE=learner Triton fixed-case correctness; not a benchmark or full suite", flush=True)
    state = {"status": "SOURCE_SAVED", "scope": "one fixed case", "commit": args.commit}
    write_json(record_dir / "status.json", state)
    try:
        if sys.flags.optimize:
            raise RuntimeError("Run without -O: these checks use Python assertions.")
        check_draft(source)
        # Import GPU dependencies only after checking the saved draft.
        import torch
        import triton

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment.")
        spec = importlib.util.spec_from_file_location("learner_triton_first", snapshot)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        capacity, batch, width = 8, 2, 128
        kv_cache = torch.full((capacity, 2, width), -1, dtype=torch.float16, device="cuda")
        k_cache = kv_cache[:, 0, :]
        v_cache = kv_cache[:, 1, :]
        k = torch.arange(batch * width, dtype=torch.float16, device="cuda").reshape(batch, width) + 10
        v = k + 1000
        indices = torch.tensor([5, 1], dtype=torch.int32, device="cuda")
        # Snapshot expected inputs BEFORE running student code.
        expected_k, expected_v = k.cpu(), v.cpu()
        expected_indices = indices.cpu()
        metadata = {
            "scope": "one fixed Triton case; not a benchmark or full suite",
            "commit": args.commit,
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cache_shape": list(kv_cache.shape),
            "k_input_stride": list(k.stride()),
            "k_cache_stride": list(k_cache.stride()),
            "dtype": str(k.dtype),
            "indices": expected_indices.tolist(),
            "k_expected": expected_k.tolist(),
            "v_expected": expected_v.tolist(),
            "initial_cache_value": -1,
            "launch_grid": [2],
            "num_warps": 4,
            "attribution": "coach harness and guided address formulas; learner read/write body",
        }
        write_json(record_dir / "input.json", metadata)
        print("GPU=" + metadata["gpu"], flush=True)
        print(f"TORCH={torch.__version__} TRITON={triton.__version__}", flush=True)
        print("CALLING=store_kv_first, grid=(2,), width=128, no timing", flush=True)
        state["status"] = "LAUNCHING"
        write_json(record_dir / "status.json", state)
        module.store_kv_first[(batch,)](k_cache, v_cache, indices, k, v, num_warps=4)
        torch.cuda.synchronize()

        # Save all raw results BEFORE running the assertions.
        actual_k_cache, actual_v_cache = k_cache.cpu(), v_cache.cpu()
        actual_k, actual_v, actual_indices = k.cpu(), v.cpu(), indices.cpu()
        result = dict(metadata)
        result.update({
            "k_cache": actual_k_cache.tolist(), "v_cache": actual_v_cache.tolist(),
            "k_after": actual_k.tolist(), "v_after": actual_v.tolist(),
            "indices_after": actual_indices.tolist(),
            "verification_status": "PENDING; see status.json for final outcome",
        })
        write_json(record_dir / "result.json", result)
        print("RAW_RESULT_SAVED=" + str(record_dir / "result.json"), flush=True)
        for row in range(capacity):
            print(f"CACHE {row}: K[:4]={actual_k_cache[row, :4].tolist()} "
                  f"V[:4]={actual_v_cache[row, :4].tolist()}")

        # Reuse the learner's four assertions and guided untouched-row loop.
        # Expected values come from pre-launch copies, not possibly changed inputs.
        assert torch.equal(actual_k_cache[5], expected_k[0]), "K input 0 -> cache 5"
        assert torch.equal(actual_v_cache[5], expected_v[0]), "V input 0 -> cache 5"
        assert torch.equal(actual_k_cache[1], expected_k[1]), "K input 1 -> cache 1"
        assert torch.equal(actual_v_cache[1], expected_v[1]), "V input 1 -> cache 1"
        for row in [0, 2, 3, 4, 6, 7]:
            assert torch.all(actual_k_cache[row] == -1).item(), f"Untouched K row {row}"
            assert torch.all(actual_v_cache[row] == -1).item(), f"Untouched V row {row}"
        # Coach adds checks that the read-only inputs were not overwritten.
        assert torch.equal(actual_k, expected_k), "K input was modified"
        assert torch.equal(actual_v, expected_v), "V input was modified"
        assert torch.equal(actual_indices, expected_indices), "Indices were modified"
        state["status"] = "FIXED_CASE_PASS"
        write_json(record_dir / "status.json", state)
        print("FIRST_TRITON_CASE_PASS: K/V destinations, untouched rows and inputs checked.")
        print("COVERAGE_LIMIT: only batch=2, width=128, interleaved FP16 cache, indices=[5,1].")
    except Exception as error:
        state.update({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        write_json(record_dir / "status.json", state)
        (record_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
