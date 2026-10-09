"""M06-01/02 coach harness: widths64/128 and selected layouts, one v1 source.

Fresh fixtures for each implementation/case; independent CPU expectations;
source snapshots and full raw results retained, including on check failures.
Default: original M06-01 four cases. M06-02 adds input gap and separate cache.
Only selected cases count as run; no arbitrary strides, masking or timing.
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


def cache_views(cache, layout):
    if layout == "interleaved":
        return cache[:, 0, :], cache[:, 1, :]
    return cache[0], cache[1]


def make_fixture(width, layout, gap, rows=TARGET_ROWS):
    import numpy as np
    if width not in WIDTHS or layout not in ("interleaved", "separate") or gap not in (0, 64):
        raise ValueError("LESSON_CONFIGURATION_OUT_OF_SCOPE")
    if tuple(rows) != TARGET_ROWS:
        raise ValueError("LESSON_INDICES: keep [7,0,3]")
    batch = len(rows)
    shape = (8, 2, width) if layout == "interleaved" else (2, 8, width)
    before = {"cache": np.full(shape, -1, dtype=np.float16),
              "k": np.full((batch, width + gap), -9, dtype=np.float16),
              "v": np.full((batch, width + gap), -9, dtype=np.float16),
              "indices": np.array(rows, dtype=np.int32)}
    values = np.arange(batch * width).reshape(batch, width).astype(np.float16) + 10
    before["k"][:, :width] = values
    before["v"][:, :width] = values + 1000
    before["expected_cache"] = before["cache"].copy()
    ek, ev = cache_views(before["expected_cache"], layout)
    for pid, pos in enumerate(rows):
        ek[pos] = before["k"][pid, :width]
        ev[pos] = before["v"][pid, :width]
    return before


def verify_backing(before, after):
    import numpy as np
    for name in ("cache", "k", "v", "indices"):
        expected = before["expected_cache" if name == "cache" else name]
        np.testing.assert_array_equal(after[name], expected, err_msg=name)


def backing_controls(configurations):
    """CPU checks of full-storage oracle; these do not execute CUDA/Triton."""
    records = []
    for spec in configurations:
        width, layout, gap = spec["width"], spec["layout"], spec["input_gap"]
        before = make_fixture(width, layout, gap)
        def good():
            return {name: before["expected_cache" if name == "cache" else name].copy()
                    for name in ("cache", "k", "v", "indices")}
        verify_backing(before, good())
        detected = []
        kinds = ["target_K_last", "target_V_last", "untouched_K", "untouched_V",
                 "input_K", "input_V", "indices"]
        if gap:
            kinds += ["input_K_gap", "input_V_gap"]
        for kind in kinds:
            after = good()
            kc, vc = cache_views(after["cache"], layout)
            if kind == "target_K_last":
                kc[7, width - 1] = -99
            elif kind == "target_V_last":
                vc[3, width - 1] = -99
            elif kind == "untouched_K":
                kc[2, 0] = -99
            elif kind == "untouched_V":
                vc[2, 0] = -99
            elif kind == "input_K":
                after["k"][1, 0] = -99
            elif kind == "input_V":
                after["v"][1, 0] = -99
            elif kind == "indices":
                after["indices"][0] = 2
            elif kind == "input_K_gap":
                after["k"][1, width] = -99
            else:
                after["v"][1, width] = -99
            try:
                verify_backing(before, after)
            except AssertionError:
                detected.append(kind)
            else:
                raise AssertionError("BACKING_CHECK_MISSED: " + kind)
        records.append({**spec, "positive_control": "PASS", "negative_controls": detected})
    return records


def select_configurations(width=None, layout=None, input_gap=None, layouts=False):
    if layouts and any(value is not None for value in (width, layout, input_gap)):
        raise ValueError("Use --layouts alone or select --width/--layout/--input-gap")
    widths = WIDTHS if width is None else (width,)
    cache_layouts = ("interleaved", "separate") if layouts else (layout or "interleaved",)
    gaps = (0, 64) if layouts else (0 if input_gap is None else input_gap,)
    return [{"width": d, "layout": kind, "input_gap": gap}
            for d in widths for kind in cache_layouts for gap in gaps]


def run_case(torch, launcher, checker, rows, width, directory, implementation,
             layout="interleaved", gap=0):
    directory.mkdir()
    rows = list(rows)
    batch = len(rows)
    before = make_fixture(width, layout, gap, rows)
    # Transfer full backing arrays FIRST; slicing on GPU preserves the gap.
    buffers = {name: torch.from_numpy(before[name].copy()).cuda()
               for name in ("cache", "k", "v", "indices")}
    cache, indices = buffers["cache"], buffers["indices"]
    k, v = buffers["k"][:, :width], buffers["v"][:, :width]
    k_cache, v_cache = cache_views(cache, layout)
    if k.stride() != (width + gap, 1) or v.stride() != (width + gap, 1):
        raise RuntimeError("Unexpected input layout")
    expected_dst = width * (2 if layout == "interleaved" else 1)
    if k_cache.stride() != (expected_dst, 1) or v_cache.stride() != (expected_dst, 1):
        raise RuntimeError("Unexpected cache layout")
    expected_k = torch.from_numpy(before["k"][:, :width].copy())
    expected_v = torch.from_numpy(before["v"][:, :width].copy())
    write_json(directory / "input.json", {
        "implementation": implementation, "indices": rows,
        "batch": batch, "width": width, "input_shape": list(k.shape),
        "layout": layout, "input_gap": gap, "gap_initial_value": -9,
        "triton_grid": [batch] if implementation == "learner_triton" else None,
        "cache_shape": list(cache.shape), "initial_value": -1,
        "input_stride": list(k.stride()), "cache_view_stride": list(k_cache.stride()),
        "v_input_stride": list(v.stride()), "v_cache_view_stride": list(v_cache.stride()),
        "stride_unit": "elements", "shared_kv_row_stride_contract": True,
        "dtype": "torch.float16", "expected_k": expected_k.tolist(),
        "expected_v": expected_v.tolist(),
        "backing_before": {name: value.tolist() for name, value in before.items()},
    })
    print(f"CASE_SETUP implementation={implementation} input_shape={tuple(k.shape)} "
          f"layout={layout} input_gap={gap} input_stride={k.stride()} "
          f"cache_stride={k_cache.stride()} indices={rows}", flush=True)
    write_json(directory / "status.json", {"status": "LAUNCHING"})
    try:
        launcher(k_cache, v_cache, indices, k, v)
        torch.cuda.synchronize()
        after = {name: value.cpu().numpy().copy() for name, value in buffers.items()}
        ak, av = cache_views(after["cache"], layout)
        # Save full storage before any assertion, including failed results.
        write_json(directory / "result.json", {
            "k_cache": ak.tolist(), "v_cache": av.tolist(),
            "k_after": after["k"][:, :width].tolist(),
            "v_after": after["v"][:, :width].tolist(),
            "indices_after": after["indices"].tolist(),
            "backing_after": {name: value.tolist() for name, value in after.items()},
        })
        checker(torch.from_numpy(ak.copy()), torch.from_numpy(av.copy()),
                expected_k, expected_v, rows)
        verify_backing(before, after)
        write_json(directory / "status.json", {"status": "PASS"})
    except Exception as error:
        write_json(directory / "status.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
        })
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(f"CASE_PASS implementation={implementation} width={width} "
          f"layout={layout} input_gap={gap} indices={rows}", flush=True)


def validate_kernel(source):
    validate_draft(source, "store_kv_v1")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "store_kv_v1")
    args = function.args
    parameters = args.posonlyargs + args.args
    expected_names = ["k_cache_ptr", "v_cache_ptr", "indices_ptr", "k_ptr", "v_ptr",
                      "WIDTH", "SRC_ROW_STRIDE", "DST_ROW_STRIDE"]
    if ([parameter.arg for parameter in parameters] != expected_names
            or args.defaults or args.kwonlyargs or args.vararg or args.kwarg):
        raise RuntimeError("KERNEL_SIGNATURE: five pointers, WIDTH, SRC_ROW_STRIDE, DST_ROW_STRIDE")
    for parameter in parameters[5:]:
        if parameter.annotation is None or ast.unparse(parameter.annotation) != "tl.constexpr":
            raise RuntimeError(f"KERNEL_PARAMETER: keep {parameter.arg}: tl.constexpr")
    # This preflight detects incomplete/signature-mismatched drafts only.
    # Actual address use still requires source review and result checks.


def prepare_student(project):
    student = project / "experiments" / "triton_store_v1.py"
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
    """Coach adapter for this lesson; broader validation belongs to M06-04."""
    def learner_launch(k_cache, v_cache, indices, k, v):
        if k.ndim != 2 or v.ndim != 2:
            raise RuntimeError("INPUT_SHAPE: expected two-dimensional K/V")
        batch, width = k.shape
        if batch != 3 or width not in WIDTHS or v.shape != k.shape or indices.shape != (batch,):
            raise RuntimeError("INPUT_CONTRACT: batch3, width64/128, matching K/V and index count")
        if any(cache.shape != (8, width) for cache in (k_cache, v_cache)):
            raise RuntimeError("CACHE_SHAPE: expected capacity8 and matching width")
        if any(tensor.stride(1) != 1 for tensor in (k, v, k_cache, v_cache)):
            raise RuntimeError("COLUMN_STRIDE: this kernel requires unit column strides")
        if indices.stride() != (1,):
            raise RuntimeError("INDEX_STRIDE: this kernel requires contiguous indices")
        # Shared offsets are valid only while the matching K/V strides agree.
        source_stride = k.stride(0)
        destination_stride = k_cache.stride(0)
        if v.stride(0) != source_stride:
            raise RuntimeError("SHARED_SOURCE_STRIDE: K/V input row strides must match")
        if v_cache.stride(0) != destination_stride:
            raise RuntimeError("SHARED_CACHE_STRIDE: K/V cache row strides must match")
        if source_stride not in (width, width + 64) or destination_stride not in (width, 2 * width):
            raise RuntimeError("LESSON_LAYOUT: only selected gap0/64 and interleaved/separate cases")
        print(f"TRITON_LAUNCH grid=({batch},) WIDTH={width} "
              f"SRC_ROW_STRIDE={source_stride} DST_ROW_STRIDE={destination_stride} num_warps=4",
              flush=True)
        module.store_kv_v1[(batch,)](
            k_cache, v_cache, indices, k, v, WIDTH=width,
            SRC_ROW_STRIDE=source_stride, DST_ROW_STRIDE=destination_stride,
            num_warps=4,
        )
    return learner_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--prepare", action="store_true", help="Create missing learner template only; no GPU imports or run")
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--layout", choices=("interleaved", "separate"))
    parser.add_argument("--input-gap", type=int, choices=(0, 64))
    parser.add_argument("--layouts", action="store_true", help="All 16 implementation/width/layout/gap cases")
    parser.add_argument("--cpu-self-test", type=Path, help="Coach full-storage oracle tests only; no GPU imports")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: checks use assert")
    try:
        configurations = select_configurations(args.width, args.layout, args.input_gap, args.layouts)
    except ValueError as error:
        parser.error(str(error))
    if args.cpu_self_test:
        controls = backing_controls(select_configurations(layouts=True))
        write_json(args.cpu_self_test, {"status": "CPU_HARNESS_PASS", "GPU": "NOT_RUN",
                   "learner": "NOT_RUN", "fixtures": len(controls), "controls": controls})
        print("CPU_HARNESS_PASS: 8 fixtures, 64 deliberate corruptions rejected; GPU NOT_RUN")
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and --shared-results required for the WSL run")
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
        print("PREPARE_COMPLETE: edit triton_store_v1.py; fill cols/src_offsets/dst_offsets using WIDTH and actual row strides; no GPU was run.", flush=True)
        return
    checker_path = project / "experiments" / "check_kv_targets.py"
    if not checker_path.is_file():
        raise FileNotFoundError(f"LEARNER_CHECK_MISSING: {checker_path}; complete lesson 12 first")
    print(f"EXISTING_CHECK_KEPT={checker_path}", flush=True)
    parent = project / "experiments" / "learning" / "kv-width-check"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    lesson = "M06-02" if any(c["layout"] != "interleaved" or c["input_gap"] != 0 for c in configurations) else "M06-01"
    scope = (f"{lesson}: two implementations; FP16 B3 C8 indices=[7,0,3]; "
             f"selected configurations={configurations}; full backing checked; no timing")
    state = {"status": "PREPARING", "scope": scope, "commit": COMMIT, "passed_cases": [],
             "configurations": configurations, "expected_cases": 2 * len(configurations)}
    print(f"RUN_DIRECTORY={run}\nSCOPE={scope}", flush=True)
    try:
        shutil.copyfile(__file__, run / "coach_runner.py")
        shutil.copyfile(Path(__file__).with_suffix(".sh"), run / "coach_entry.sh")
        hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                  for name in ("coach_runner.py", "coach_entry.sh")}
        for source, relative in [(student, "triton_store_v1.py"), (checker_path, "check_kv_targets.py")]:
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
        kernel_source = (run / "triton_store_v1.py").read_text(encoding="utf-8")
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
        checked_widths = sorted({c["width"] for c in configurations})
        controls = {str(width): checker_controls(torch, checker, width) for width in checked_widths}
        write_json(run / "checker_controls.json", controls)
        print(f"CHECKER_CONTROLS_PASS: {len(controls)} correct fixtures accepted; "
              f"{7 * len(controls)} deliberate errors rejected on CPU", flush=True)
        backing = backing_controls(configurations)
        write_json(run / "backing_controls.json", backing)
        print(f"BACKING_CONTROLS_PASS: {len(backing)} correct fixtures accepted; "
              f"{sum(len(c['negative_controls']) for c in backing)} corruptions rejected on CPU", flush=True)
        module = load_module(run / "triton_store_v1.py", "learner_width_kernel")
        state.update({"torch": torch.__version__, "triton": triton.__version__,
                      "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                      "triton_grid_rule": "(k.shape[0],)", "triton_width_rule": "k.shape[1]", "triton_num_warps": 4,
                      "triton_source_stride_rule": "k.stride(0), equal to v.stride(0)",
                      "triton_destination_stride_rule": "k_cache.stride(0), equal to v_cache.stride(0)",
                      "learner_file": "triton_store_v1.py",
                      "ownership": "coach harness; core source snapshot preserved; consult lesson review for actual authorship"})
        print(f"GPU={state['gpu']} TORCH={state['torch']} TRITON={state['triton']}", flush=True)
        learner_launch = make_learner_launcher(module)
        for implementation, launcher in (("original_cuda", store_cache), ("learner_triton", learner_launch)):
            for config in configurations:
                width, layout, gap = config["width"], config["layout"], config["input_gap"]
                rows = TARGET_ROWS
                name = f"{implementation}-w{width}-{layout}-gap{gap}-" + "-".join(map(str, rows))
                state.update({"status": "RUNNING", "current_case": name})
                write_json(run / "status.json", state)
                run_case(torch, launcher, checker, rows, width, run / name, implementation, layout, gap)
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
    label = "KV_LAYOUT_CASES_PASS" if lesson == "M06-02" else "KV_WIDTH_CASES_PASS"
    print(f"{label}: {len(state['passed_cases'])}/{state['expected_cases']} cases; "
          "all targets, untouched cache, complete input backing/gap and indices checked.", flush=True)
    print(f"COVERAGE_LIMIT: only selected configurations {configurations}; FP16 B3 C8 "
          "indices=[7,0,3], shared K/V row strides, unit column strides. "
          "NOT arbitrary-layout/mask/batch coverage or a benchmark.")


if __name__ == "__main__":
    main()
