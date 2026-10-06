"""Coach harness for WIDTH=64/128 using one learner kernel source and checker.

Fresh fixtures for each implementation/case; independent CPU expectations;
source snapshots and full raw results retained, including on check failures.
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
TARGET_ROWS = (7, 0, 3)
WIDTHS = (64, 128)
SCOPE = "two implementations x widths 64/128; FP16 batch=3 capacity=8 indices=[7,0,3]; no timing"


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
    rows = list(TARGET_ROWS)
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


def run_case(torch, launcher, checker, rows, width, directory, implementation):
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


def validate_kernel(source):
    validate_draft(source, "store_kv_width")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "store_kv_width")
    args = function.args
    parameters = args.posonlyargs + args.args
    if (len(parameters) != 6 or args.defaults or args.kwonlyargs
            or args.vararg or args.kwarg):
        raise RuntimeError("KERNEL_SIGNATURE: five pointer parameters followed by WIDTH: tl.constexpr")
    width = parameters[-1]
    if width.arg != "WIDTH" or width.annotation is None or ast.unparse(width.annotation) != "tl.constexpr":
        raise RuntimeError("WIDTH_PARAMETER: keep WIDTH: tl.constexpr in the definition")


def prepare_student(project):
    student = project / "experiments" / "triton_store_width.py"
    template = Path(__file__).with_name("14-triton-store-width-template.py")
    student.parent.mkdir(parents=True, exist_ok=True)
    try:
        with student.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(template.read_text(encoding="utf-8"))
        print(f"CREATED_LEARNER_TEMPLATE={student}", flush=True)
    except FileExistsError:
        print(f"EXISTING_LEARNER_FILE_KEPT={student}", flush=True)
    return student


def make_learner_launcher(module):
    def learner_launch(k_cache, v_cache, indices, k, v):
        if k.ndim != 2:
            raise RuntimeError("INPUT_SHAPE: expected two-dimensional input")
        batch, width = k.shape
        if batch != 3 or width not in WIDTHS or v.shape != k.shape or indices.shape != (batch,):
            raise RuntimeError("INPUT_CONTRACT: batch3, width64/128, matching K/V and index count")
        if k.stride() != (width, 1) or v.stride() != (width, 1):
            raise RuntimeError("INPUT_LAYOUT: this lesson requires contiguous input rows")
        for cache in (k_cache, v_cache):
            if cache.shape != (8, width) or cache.stride() != (2 * width, 1):
                raise RuntimeError("CACHE_LAYOUT: this lesson requires interleaved K/V rows")
        print(f"TRITON_LAUNCH grid=({batch},) WIDTH={width} num_warps=4", flush=True)
        module.store_kv_width[(batch,)](
            k_cache, v_cache, indices, k, v, WIDTH=width, num_warps=4
        )
    return learner_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--shared-results", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true", help="Create missing learner template only; no GPU imports or run")
    args = parser.parse_args()
    project = args.project.resolve()
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: the learner checks use assert")
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("COMMIT_MISMATCH: no checkout changed")
    if git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("ORIGINAL_KERNEL_CHANGED: inspect before using as the pinned original")
    student = prepare_student(project)
    if args.prepare:
        print("LEARNER_SOURCE_BEGIN\n" + student.read_text(encoding="utf-8-sig") + "LEARNER_SOURCE_END", flush=True)
        print("PREPARE_COMPLETE: replace pass with three width-based address lines; no GPU was run.", flush=True)
        return
    checker_path = project / "experiments" / "check_kv_targets.py"
    if not checker_path.is_file():
        raise FileNotFoundError(f"LEARNER_CHECK_MISSING: {checker_path}; complete lesson 12 first")
    print(f"EXISTING_CHECK_KEPT={checker_path}", flush=True)
    parent = project / "experiments" / "learning" / "kv-width-check"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = {"status": "PREPARING", "scope": SCOPE, "commit": COMMIT, "passed_cases": []}
    print(f"RUN_DIRECTORY={run}\nSCOPE={SCOPE}", flush=True)
    try:
        shutil.copyfile(__file__, run / "coach_runner.py")
        shutil.copyfile(Path(__file__).with_suffix(".sh"), run / "coach_entry.sh")
        hashes = {}
        for source, relative in [(student, "triton_store_width.py"), (checker_path, "check_kv_targets.py")]:
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
        kernel_source = (run / "triton_store_width.py").read_text(encoding="utf-8")
        print("SAVED_KERNEL_BEGIN\n" + kernel_source + "SAVED_KERNEL_END", flush=True)
        validate_kernel(kernel_source)
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project / "python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT: not the pinned project")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        checker = load_module(run / "check_kv_targets.py", "learner_index_checker").check_kv_targets
        controls = {str(width): checker_controls(torch, checker, width) for width in WIDTHS}
        write_json(run / "checker_controls.json", controls)
        print("CHECKER_CONTROLS_PASS: 2 correct fixtures accepted; 14 deliberate errors rejected on CPU", flush=True)
        module = load_module(run / "triton_store_width.py", "learner_width_kernel")
        state.update({"torch": torch.__version__, "triton": triton.__version__,
                      "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                      "triton_grid_rule": "(k.shape[0],)", "triton_width_rule": "k.shape[1]", "triton_num_warps": 4})
        print(f"GPU={state['gpu']} TORCH={state['torch']} TRITON={state['triton']}", flush=True)
        learner_launch = make_learner_launcher(module)
        for implementation, launcher in (("original_cuda", store_cache), ("learner_triton", learner_launch)):
            for width in WIDTHS:
                rows = TARGET_ROWS
                name = f"{implementation}-w{width}-" + "-".join(map(str, rows))
                state.update({"status": "RUNNING", "current_case": name})
                write_json(run / "status.json", state)
                run_case(torch, launcher, checker, rows, width, run / name, implementation)
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
    print("KV_WIDTH_CASES_PASS: 4/4 cases; all target elements, untouched rows and inputs checked.", flush=True)
    print("COVERAGE_LIMIT: only FP16 batch=3 widths64/128 capacity=8, contiguous inputs/interleaved cache and indices=[7,0,3]; NOT a benchmark.")


if __name__ == "__main__":
    main()
