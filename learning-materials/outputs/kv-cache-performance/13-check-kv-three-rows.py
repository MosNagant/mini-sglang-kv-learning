"""Coach harness for three rows at a selected width (64 or 128).

Fresh fixtures for each implementation/case; independent CPU expectations;
source snapshots and full raw results retained, including on check failures.
The learner's fixed-width kernel is kept unchanged. --width controls fixtures,
not kernel rewriting; a preflight guard checks the three address expressions.
"""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
CASES = ((7, 0, 3),)
WIDTHS = (64, 128)


def describe_scope(width):
    return (f"two implementations, indices=[7,0,3]; FP16 batch=3 width={width} "
            "capacity=8; interleaved cache; no timing")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_draft(source, function_name):
    tree = ast.parse(source)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == function_name), None)
    if function is None or any(isinstance(node, ast.Pass) for node in ast.walk(function)):
        raise RuntimeError(f"INCOMPLETE_SOURCE: {function_name}")


def checker_controls(torch, checker, width):
    """Check both valid data and independent K/V corruptions, without a GPU."""
    rows = list(CASES[0])
    batch = len(rows)
    expected_k = torch.arange(batch * width, dtype=torch.float16).reshape(batch, width) + 10
    expected_v = expected_k + 1000
    k_cache = torch.full((8, width), -1, dtype=torch.float16)
    v_cache = torch.full((8, width), -1, dtype=torch.float16)
    for pid, pos in enumerate(rows):
        k_cache[pos] = expected_k[pid]
        v_cache[pos] = expected_v[pid]
    checker(k_cache, v_cache, expected_k, expected_v, rows)
    detected = []
    for kind in ("K", "V"):
        for pid, pos in enumerate(rows):
            bad_k, bad_v = k_cache.clone(), v_cache.clone()
            (bad_k if kind == "K" else bad_v)[pos, width - 1] = -99
            try:
                checker(bad_k, bad_v, expected_k, expected_v, rows)
            except AssertionError:
                detected.append(f"{kind} input {pid}: last column corruption detected")
            else:
                raise RuntimeError(f"CHECKER_MISSED_CORRUPTION: {kind} input {pid}")
    try:
        checker(k_cache, v_cache, expected_k, expected_v, rows[::-1])
    except AssertionError:
        detected.append("swapped destination mapping detected")
    else:
        raise RuntimeError("CHECKER_MISSED_SWAPPED_MAPPING")
    return {"positive_control": "PASS", "negative_controls": detected}


def check_other_state(torch, k_cache, v_cache, k_after, v_after, indices_after,
                      expected_k, expected_v, expected_indices, rows):
    # The learner already checked untouched rows in the earlier lesson;
    # the coach adapts that loop to the current target list and checks inputs.
    for row in range(8):
        if row not in rows:
            assert torch.all(k_cache[row] == -1).item(), f"Untouched K row {row} changed"
            assert torch.all(v_cache[row] == -1).item(), f"Untouched V row {row} changed"
    assert torch.equal(k_after, expected_k), "Input K changed"
    assert torch.equal(v_after, expected_v), "Input V changed"
    assert torch.equal(indices_after, expected_indices), "Input indices changed"


def run_case(torch, launcher, checker, rows, directory, implementation, width):
    if width not in WIDTHS:
        raise ValueError(f"Unsupported width: {width}; choose 64 or 128")
    directory.mkdir()
    rows = list(rows)
    # CPU copies exist before launch and cannot alias these GPU inputs.
    batch = len(rows)
    expected_k = torch.arange(batch * width, dtype=torch.float16).reshape(batch, width) + 10
    expected_v = expected_k + 1000
    expected_indices = torch.tensor(rows, dtype=torch.int32)
    k, v, indices = expected_k.cuda(), expected_v.cuda(), expected_indices.cuda()
    cache = torch.full((8, 2, width), -1, dtype=torch.float16, device="cuda")
    k_cache, v_cache = cache[:, 0, :], cache[:, 1, :]
    if k.stride() != (width, 1) or v.stride() != (width, 1):
        raise RuntimeError("Unexpected input layout")
    if k_cache.stride() != (2 * width, 1) or v_cache.stride() != (2 * width, 1):
        raise RuntimeError("Unexpected cache layout")
    write_json(directory / "input.json", {
        "implementation": implementation, "indices": rows,
        "batch": batch, "width": width, "input_shape": list(k.shape),
        "triton_grid": [batch] if implementation == "learner_triton" else None,
        "cache_shape": [8, 2, width], "initial_value": -1,
        "input_stride": list(k.stride()), "cache_view_stride": list(k_cache.stride()),
        "dtype": "torch.float16", "expected_k": expected_k.tolist(),
        "expected_v": expected_v.tolist(),
    })
    print(f"CASE_SETUP implementation={implementation} input_shape={tuple(k.shape)} indices={rows}", flush=True)
    write_json(directory / "status.json", {"status": "LAUNCHING"})
    try:
        launcher(k_cache, v_cache, indices, k, v)
        torch.cuda.synchronize()
        actual_k_cache, actual_v_cache = k_cache.cpu(), v_cache.cpu()
        actual_k, actual_v, actual_indices = k.cpu(), v.cpu(), indices.cpu()
        # Raw data precedes verification; a failed check never destroys it.
        write_json(directory / "result.json", {
            "k_cache": actual_k_cache.tolist(), "v_cache": actual_v_cache.tolist(),
            "k_after": actual_k.tolist(), "v_after": actual_v.tolist(),
            "indices_after": actual_indices.tolist(),
        })
        checker(actual_k_cache, actual_v_cache, expected_k, expected_v, rows)
        check_other_state(torch, actual_k_cache, actual_v_cache, actual_k, actual_v,
                          actual_indices, expected_k, expected_v, expected_indices, rows)
        write_json(directory / "status.json", {"status": "PASS"})
    except Exception as error:
        write_json(directory / "status.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
        })
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(f"CASE_PASS implementation={implementation} width={width} indices={rows}", flush=True)


def validate_kernel(source, width):
    if width not in WIDTHS:
        raise ValueError(f"Unsupported width: {width}; choose 64 or 128")
    validate_draft(source, "store_kv_first")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "store_kv_first")
    args = function.args
    if (len(args.posonlyargs + args.args) != 5 or args.defaults or args.kwonlyargs
            or args.vararg or args.kwarg):
        raise RuntimeError("KERNEL_SIGNATURE: keep the original five pointer parameters; "
                           "grid and num_warps belong to the launch call, outside the kernel")
    # Only a guard for this learner's simple constant-width source form, not a
    # general Triton safety analyzer. GPU correctness checks remain necessary.
    required = {
        "cols": f"tl.arange(0, {width})",
        "src_offsets": f"pid * {width} + cols",
        "dst_offsets": f"pos * {2 * width} + cols",
    }
    for name, expression in required.items():
        matches = [node.value for node in ast.walk(function)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name
                           for target in node.targets)]
        expected = ast.parse(expression, mode="eval").body
        if len(matches) != 1 or ast.dump(matches[0]) != ast.dump(expected):
            found = "; ".join(ast.unparse(node) for node in matches) or "missing"
            raise RuntimeError(
                f"KERNEL_LAYOUT_MISMATCH: --width {width} expects {name} = {expression}; "
                f"saved source has {found}. No GPU kernel was launched. "
                "Match --width to the saved fixed-width kernel (64/64/128 or 128/128/256). "
                "Parameterized kernels need their corresponding adapter.")
    return {"width": width, "source_row_stride": width, "cache_row_stride": 2 * width}


def make_learner_launcher(module, width):
    # Coach wiring of the discussed one-program-per-input-row design.
    def learner_launch(k_cache, v_cache, indices, k, v):
        if k.ndim != 2:
            raise RuntimeError("INPUT_SHAPE: expected a matrix")
        batch = k.shape[0]
        if batch != 3 or k.shape != (batch, width) or v.shape != k.shape or indices.shape != (batch,):
            raise RuntimeError("INPUT_ROWS_MISMATCH: K/V rows and index count must agree")
        if k.stride() != (width, 1) or v.stride() != (width, 1):
            raise RuntimeError("INPUT_LAYOUT: contiguous rows required for this fixed kernel")
        if any(cache.shape != (8, width) or cache.stride() != (2 * width, 1)
               for cache in (k_cache, v_cache)):
            raise RuntimeError("CACHE_LAYOUT: interleaved cache required for this fixed kernel")
        print(f"TRITON_LAUNCH grid=({batch},) pid=0..{batch - 1} width={width}", flush=True)
        module.store_kv_first[(batch,)](k_cache, v_cache, indices, k, v, num_warps=4)
    return learner_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--shared-results", required=True, type=Path)
    parser.add_argument("--width", type=int, choices=WIDTHS, default=128,
                        help="Width of test data and checks; must match the saved fixed-width kernel")
    args = parser.parse_args()
    width = args.width
    scope = describe_scope(width)
    project = args.project.resolve()
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: the learner checks use assert")
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("COMMIT_MISMATCH: no checkout changed")
    if git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("ORIGINAL_KERNEL_CHANGED: inspect before using as the pinned original")
    student = project / "experiments" / "triton_store_first.py"
    if not student.is_file():
        raise FileNotFoundError(f"LEARNER_KERNEL_MISSING: {student}")
    checker_path = project / "experiments" / "check_kv_targets.py"
    if not checker_path.is_file():
        raise FileNotFoundError(f"LEARNER_CHECK_MISSING: {checker_path}; complete lesson 12 first")
    print(f"EXISTING_CHECK_KEPT={checker_path}", flush=True)
    parent = project / "experiments" / "learning" / "kv-three-rows"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = {"status": "PREPARING", "scope": scope, "width": width,
             "commit": COMMIT, "passed_cases": []}
    print(f"RUN_DIRECTORY={run}\nSCOPE={scope}", flush=True)
    try:
        shutil.copyfile(__file__, run / "coach_runner.py")
        shutil.copyfile(Path(__file__).with_suffix(".sh"), run / "coach_entry.sh")
        hashes = {}
        for source, relative in [(student, "triton_store_first.py"), (checker_path, "check_kv_targets.py")]:
            snapshot = run / relative
            snapshot.write_text(source.read_text(encoding="utf-8-sig"), encoding="utf-8")
            hashes[relative] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        for relative in git("ls-files", "--", "python/minisgl/kernel").splitlines():
            source, saved = project / relative, run / "original_source" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, saved)
            hashes[relative] = hashlib.sha256(saved.read_bytes()).hexdigest()
        write_json(run / "source_hashes.json", hashes)
        check_source = (run / "check_kv_targets.py").read_text(encoding="utf-8")
        print("SAVED_CHECK_BEGIN\n" + check_source + "SAVED_CHECK_END", flush=True)
        validate_draft(check_source, "check_kv_targets")
        kernel_source = (run / "triton_store_first.py").read_text(encoding="utf-8")
        print("SAVED_KERNEL_BEGIN\n" + kernel_source + "SAVED_KERNEL_END", flush=True)
        state["kernel_layout_preflight"] = validate_kernel(kernel_source, width)
        print(f"KERNEL_LAYOUT_MATCH: cols={width} input_stride={width} cache_stride={2 * width}", flush=True)
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project / "python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT: not the pinned project")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        checker = load_module(run / "check_kv_targets.py", "learner_index_checker").check_kv_targets
        controls = checker_controls(torch, checker, width)
        write_json(run / "checker_controls.json", controls)
        print("CHECKER_CONTROLS_PASS: correct three-row fixture accepted; seven deliberate errors rejected on CPU", flush=True)
        module = load_module(run / "triton_store_first.py", "learner_index_kernel")
        state.update({"torch": torch.__version__, "triton": triton.__version__,
                      "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                      "triton_grid_rule": "(k.shape[0],)", "triton_num_warps": 4})
        print(f"GPU={state['gpu']} TORCH={state['torch']} TRITON={state['triton']}", flush=True)
        learner_launch = make_learner_launcher(module, width)
        for implementation, launcher in (("original_cuda", store_cache), ("learner_triton", learner_launch)):
            for rows in CASES:
                name = implementation + "-" + "-".join(map(str, rows))
                state.update({"status": "RUNNING", "current_case": name})
                write_json(run / "status.json", state)
                run_case(torch, launcher, checker, rows, run / name, implementation, width)
                state["passed_cases"].append(name)
                write_json(run / "status.json", state)
        state.update({"status": "PASS", "current_case": None})
    except Exception as error:
        state.update({"status": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write_json(run / "status.json", state)
        args.shared_results.mkdir(parents=True, exist_ok=True)
        shared = args.shared_results / run.name
        try:
            shutil.copytree(run, shared)
            print(f"SHARED_ARTIFACTS={shared}", flush=True)
        except OSError as archive_error:
            print(f"SHARED_COPY_FAILED={archive_error}; WSL records retained at {run}", flush=True)
            if state["status"] == "PASS":
                raise
    print(f"KV_THREE_ROWS_PASS: 2/2 cases; width={width}; all target elements, untouched rows and inputs checked.", flush=True)
    print(f"COVERAGE_LIMIT: only FP16 batch=3 width={width} capacity=8, interleaved cache and indices=[7,0,3]; NOT a benchmark.")


if __name__ == "__main__":
    main()
