"""M06-03: requested mask demonstration and28 scoped correctness cases; no timing.

Reuses lesson14 helpers without changing the accepted14 entry or old v1.
Creates a separate experiments/triton_store_mask.py if absent, otherwise keeps
the existing file. Records full storage, source hashes, controls and failures.
"""
import argparse
import ast
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m06_layout_helpers", HERE / "14-check-kv-widths.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.WIDTHS = (64, 128, 192)  # In-memory scope extension only;14 source is unchanged.
COMMIT = h.COMMIT
ROWS = h.TARGET_ROWS


def configurations():
    cases = [dict(width=d, layout=layout, input_gap=gap, cache_guard=0)
             for d in (64, 128, 192)
             for layout in ("interleaved", "separate") for gap in (0, 64)]
    cases += [dict(width=192, layout=layout, input_gap=64, cache_guard=64)
              for layout in ("interleaved", "separate")]
    return cases


def views(cache, layout, width):
    k, v = h.cache_views(cache, layout)
    return k[:, :width], v[:, :width]


def fixture(cfg):
    import numpy as np
    if cfg not in configurations():
        raise ValueError("LESSON_CONFIGURATION_OUT_OF_SCOPE")
    d, layout, gap, guard = (cfg[x] for x in ("width", "layout", "input_gap", "cache_guard"))
    if not guard:
        return h.make_fixture(d, layout, gap)
    # Guard cases have64 allocated elements after EACH valid K/V row.
    pitch = d + guard
    shape = (8, 2, pitch) if layout == "interleaved" else (2, 8, pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    ck, cv = h.cache_views(cache, layout)
    ck[:, d:] = -7
    cv[:, d:] = -7
    before = dict(cache=cache,
                  k=np.full((3, d + gap), -9, dtype=np.float16),
                  v=np.full((3, d + gap), -9, dtype=np.float16),
                  indices=np.array(ROWS, dtype=np.int32))
    values = np.arange(3 * d).reshape(3, d).astype(np.float16) + 10
    before["k"][:, :d] = values
    before["v"][:, :d] = values + 1000
    before["expected_cache"] = cache.copy()
    ek, ev = views(before["expected_cache"], layout, d)
    for pid, pos in enumerate(ROWS):
        ek[pos] = before["k"][pid, :d]
        ev[pos] = before["v"][pid, :d]
    return before


def storage_controls():
    records = []
    for cfg in configurations():
        before = fixture(cfg)
        def good():
            return {key: before["expected_cache" if key == "cache" else key].copy()
                    for key in ("cache", "k", "v", "indices")}
        h.verify_backing(before, good())
        d = cfg["width"]
        kinds = ["target_K_last", "target_V_last", "untouched_K", "untouched_V",
                 "input_K", "input_V", "indices"]
        if cfg["input_gap"]:
            kinds += ["input_K_gap", "input_V_gap"]
        if cfg["cache_guard"]:
            kinds += ["cache_K_guard_first", "cache_V_guard_first",
                      "cache_K_guard_last", "cache_V_guard_last"]
        detected = []
        for kind in kinds:
            after = good()
            kc, vc = h.cache_views(after["cache"], cfg["layout"])
            if kind == "target_K_last": kc[7, d - 1] = -99
            elif kind == "target_V_last": vc[3, d - 1] = -99
            elif kind == "untouched_K": kc[2, 0] = -99
            elif kind == "untouched_V": vc[2, 0] = -99
            elif kind == "input_K": after["k"][1, 0] = -99
            elif kind == "input_V": after["v"][1, 0] = -99
            elif kind == "indices": after["indices"][0] = 2
            elif kind == "input_K_gap": after["k"][1, d] = -99
            elif kind == "input_V_gap": after["v"][1, d] = -99
            else:
                cache = kc if kind.startswith("cache_K") else vc
                cache[7, d if kind.endswith("first") else d + 63] = -99
            try:
                h.verify_backing(before, after)
            except AssertionError:
                detected.append(kind)
            else:
                raise AssertionError("STORAGE_CONTROL_MISSED: " + kind)
        records.append(dict(**cfg, positive_control="PASS", negative_controls=detected))
    return records


def validate_mask_source(source):
    """Narrow reviewed lesson kernel preflight, not a general safety proof."""
    h.validate_draft(source, "store_kv_v1")
    tree = ast.parse(source)
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "store_kv_v1")
    names = ["k_cache_ptr", "v_cache_ptr", "indices_ptr", "k_ptr", "v_ptr",
             "WIDTH", "SRC_ROW_STRIDE", "DST_ROW_STRIDE", "BLOCK_SIZE"]
    args = function.args
    if ([a.arg for a in args.posonlyargs + args.args] != names or args.defaults
            or args.kwonlyargs or args.kw_defaults or args.vararg or args.kwarg):
        raise RuntimeError("MASK_SIGNATURE: keep the five pointers and four constexpr parameters")
    for arg in args.args[5:]:
        if arg.annotation is None or ast.unparse(arg.annotation) != "tl.constexpr":
            raise RuntimeError("MASK_CONSTEXPR: " + arg.arg)
    # Match the small demonstrated body before any non-guard GPU case.
    # This also catches missing load/store masks, <=, wrong row strides and
    # mask variables overwritten after their definition.
    reference = ast.parse((HERE / "21-triton-store-mask.py").read_text(encoding="utf-8"))
    canonical = next(n for n in reference.body
                     if isinstance(n, ast.FunctionDef) and n.name == "store_kv_v1")
    if ast.dump(ast.Module(body=function.body, type_ignores=[])) != ast.dump(
            ast.Module(body=canonical.body, type_ignores=[])):
        raise RuntimeError("MASK_SOURCE_REVIEW_REQUIRED: changed lesson body; no GPU submitted")
    return dict(status="REVIEWED_BODY_MATCH", vector_load_masks=2, vector_store_masks=2,
                meaning="Static source comparison only; not runtime invalid-read detection")


def prepare(project):
    student = project / "experiments" / "triton_store_mask.py"
    student.parent.mkdir(parents=True, exist_ok=True)
    demo = HERE / "21-triton-store-mask.py"
    try:
        with student.open("x", encoding="utf-8", newline="\n") as f:
            f.write(demo.read_text(encoding="utf-8"))
        print(f"CREATED_MASK_KERNEL={student}", flush=True)
    except FileExistsError:
        print(f"EXISTING_MASK_KERNEL_KEPT={student}", flush=True)
    print("OLD_V1_KEPT: experiments/triton_store_v1.py was not modified", flush=True)
    return student


def make_launcher(module):
    def launch(kc, vc, indices, k, v):
        batch, d = k.shape
        if batch != 3 or d not in (64, 128, 192) or v.shape != k.shape:
            raise RuntimeError("MASK_INPUT_CONTRACT")
        if kc.shape != (8, d) or vc.shape != kc.shape or indices.shape != (3,):
            raise RuntimeError("MASK_CACHE_INDEX_SHAPE")
        if any(t.stride(1) != 1 for t in (k, v, kc, vc)) or indices.stride() != (1,):
            raise RuntimeError("MASK_COLUMN_OR_INDEX_STRIDE")
        src, dst = k.stride(0), kc.stride(0)
        if v.stride(0) != src or vc.stride(0) != dst:
            raise RuntimeError("MASK_SHARED_KV_STRIDE")
        allowed = {(c["width"] + c["input_gap"],
                    (c["width"] + c["cache_guard"]) *
                    (2 if c["layout"] == "interleaved" else 1))
                   for c in configurations() if c["width"] == d}
        if (src, dst) not in allowed:
            raise RuntimeError("MASK_LAYOUT_OUT_OF_SCOPE")
        block = 1 << (d - 1).bit_length()
        print(f"TRITON_LAUNCH grid=({batch},) WIDTH={d} BLOCK_SIZE={block} "
              f"SRC_ROW_STRIDE={src} DST_ROW_STRIDE={dst} num_warps=4", flush=True)
        module.store_kv_v1[(batch,)](
            kc, vc, indices, k, v, WIDTH=d, SRC_ROW_STRIDE=src,
            DST_ROW_STRIDE=dst, BLOCK_SIZE=block, num_warps=4)
    return launch


def run_case(torch, call, checker, cfg, directory, implementation):
    directory.mkdir()
    before = fixture(cfg)
    d, layout, gap, guard = (cfg[x] for x in ("width", "layout", "input_gap", "cache_guard"))
    buffers = {key: torch.from_numpy(before[key].copy()).cuda()
               for key in ("cache", "k", "v", "indices")}
    k, v = buffers["k"][:, :d], buffers["v"][:, :d]
    kc, vc = views(buffers["cache"], layout, d)
    indices = buffers["indices"]
    src = d + gap
    dst = (d + guard) * (2 if layout == "interleaved" else 1)
    for t in (k, v):
        if t.stride() != (src, 1): raise RuntimeError("WRONG_SOURCE_STRIDE")
    for t in (kc, vc):
        if t.stride() != (dst, 1): raise RuntimeError("WRONG_CACHE_STRIDE")
    expected_k = torch.from_numpy(before["k"][:, :d].copy())
    expected_v = torch.from_numpy(before["v"][:, :d].copy())
    h.write_json(directory / "input.json", dict(
        implementation=implementation, **cfg, batch=3, capacity=8, indices=list(ROWS),
        dtype="torch.float16", index_dtype="torch.int32", input_stride=list(k.stride()),
        v_input_stride=list(v.stride()), cache_stride=list(kc.stride()),
        v_cache_stride=list(vc.stride()), stride_unit="elements",
        block_size=1 << (d - 1).bit_length(), input_shape=list(k.shape),
        cache_view_shape=list(kc.shape), cache_backing_shape=list(buffers["cache"].shape),
        sentinels=dict(cache=-1, input_gap=-9, cache_guard=-7),
        backing_before={key: a.tolist() for key, a in before.items()}))
    print(f"CASE_SETUP implementation={implementation} D={d} layout={layout} "
          f"input_gap={gap} cache_guard={guard} src_stride={src} dst_stride={dst}", flush=True)
    h.write_json(directory / "status.json", dict(status="LAUNCHING"))
    try:
        call(kc, vc, indices, k, v)
        torch.cuda.synchronize()
        after = {key: a.cpu().numpy().copy() for key, a in buffers.items()}
        ak, av = views(after["cache"], layout, d)
        h.write_json(directory / "result.json", dict(
            k_cache=ak.tolist(), v_cache=av.tolist(),
            k_after=after["k"][:, :d].tolist(), v_after=after["v"][:, :d].tolist(),
            indices_after=after["indices"].tolist(),
            backing_after={key: a.tolist() for key, a in after.items()}))
        checker(torch.from_numpy(ak.copy()), torch.from_numpy(av.copy()),
                expected_k, expected_v, list(ROWS))
        h.verify_backing(before, after)
        h.write_json(directory / "status.json", dict(status="PASS"))
    except Exception as error:
        h.write_json(directory / "status.json", dict(status="FAILED",
                     error_type=type(error).__name__, error=str(error)))
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print(f"CASE_PASS implementation={implementation} D={d} layout={layout} "
          f"input_gap={gap} cache_guard={guard}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cpu-self-test", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Run without-O; correctness assertions must execute")
    if args.cpu_self_test:
        controls = storage_controls()
        args.cpu_self_test.parent.mkdir(parents=True, exist_ok=True)
        h.write_json(args.cpu_self_test, dict(status="CPU_STORAGE_CONTROLS_PASS",
                     GPU="NOT_RUN", fixtures=len(controls), controls=controls))
        print(f"CPU_STORAGE_CONTROLS_PASS: {len(controls)} positive fixtures, "
              f"{sum(len(c['negative_controls']) for c in controls)} corruptions; GPU NOT_RUN")
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and--shared-results are required")
    project = args.project.resolve()
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    if git("rev-parse", "HEAD") != COMMIT:
        raise RuntimeError("COMMIT_MISMATCH")
    if git("status", "--porcelain", "--", "python/minisgl/kernel"):
        raise RuntimeError("ORIGINAL_KERNEL_CHANGED: inspect before testing")
    student = prepare(project)
    if args.prepare:
        print("MASK_SOURCE_BEGIN\n" + student.read_text(encoding="utf-8-sig") + "MASK_SOURCE_END")
        print("MASK_PREPARE_COMPLETE: separate source created or kept; GPU NOT_RUN")
        return
    checker_path = project / "experiments" / "check_kv_targets.py"
    if not checker_path.is_file():
        raise FileNotFoundError(checker_path)
    parent = project / "experiments" / "learning" / "m06-mask"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", lesson="M06-03", commit=COMMIT,
                 expected_cases=28, passed_cases=[], configurations=configurations(),
                 scope="FP16 B3 C8 D64/128/192; selected gaps/layouts/guards; no timing",
                 ownership="Core mask demonstration by coach at explicit learner request; independent mastery not inferred")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}", flush=True)
    try:
        files = [(Path(__file__), "coach_runner.py"),
                 (HERE / "21-check-kv-mask.sh", "coach_entry.sh"),
                 (HERE / "14-check-kv-widths.py", "layout_helpers.py"),
                 (HERE / "21-triton-store-mask.py", "coach_mask_demonstration.py"),
                 (student, "triton_store_mask.py"), (checker_path, "check_kv_targets.py")]
        old = project / "experiments" / "triton_store_v1.py"
        if old.is_file(): files.append((old, "previous_v1.py"))
        hashes = {}
        for source, rel in files:
            shutil.copyfile(source, run / rel)
            hashes[rel] = hashlib.sha256((run / rel).read_bytes()).hexdigest()
        for rel in git("ls-files", "--", "python/minisgl/kernel").splitlines():
            dest = run / "original_source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / rel, dest)
            hashes["original_source/" + rel] = hashlib.sha256(dest.read_bytes()).hexdigest()
        h.write_json(run / "source_hashes.json", hashes)
        source = (run / "triton_store_mask.py").read_text(encoding="utf-8-sig")
        print("MASK_SOURCE_BEGIN\n" + source + "MASK_SOURCE_END", flush=True)
        source_review = validate_mask_source(source)
        h.write_json(run / "mask_source_review.json", source_review)
        print("MASK_SOURCE_REVIEW_OK: two masked vector loads and two masked stores", flush=True)
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project / "python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        checker_source = (run / "check_kv_targets.py").read_text(encoding="utf-8-sig")
        h.validate_draft(checker_source, "check_kv_targets")
        checker = h.load_module(run / "check_kv_targets.py", "mask_learner_checker").check_kv_targets
        controls = {str(d): h.checker_controls(torch, checker, d) for d in (64, 128, 192)}
        h.write_json(run / "checker_controls.json", controls)
        print("CHECKER_CONTROLS_PASS:3 correct fixtures and21 deliberate errors on CPU", flush=True)
        backing = storage_controls()
        h.write_json(run / "backing_controls.json", backing)
        print(f"BACKING_CONTROLS_PASS: {len(backing)} correct fixtures; "
              f"{sum(len(c['negative_controls']) for c in backing)} corruptions rejected on CPU", flush=True)
        module = h.load_module(run / "triton_store_mask.py", "lesson_mask_kernel")
        state.update(torch=torch.__version__, triton=triton.__version__,
                     cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                     grid=[3], num_warps=4, source_review=source_review)
        print(f"GPU={state['gpu']} TORCH={state['torch']} TRITON={state['triton']}", flush=True)
        for implementation, call in (("original_cuda", store_cache),
                                     ("triton_mask", make_launcher(module))):
            for cfg in configurations():
                name = (f"{implementation}-w{cfg['width']}-{cfg['layout']}"
                        f"-gap{cfg['input_gap']}-guard{cfg['cache_guard']}")
                state.update(status="RUNNING", current_case=name)
                h.write_json(run / "status.json", state)
                run_case(torch, call, checker, cfg, run / name, implementation)
                state["passed_cases"].append(name)
                h.write_json(run / "status.json", state)
        state.update(status="PASS", current_case=None)
    except Exception as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json", state)
        args.shared_results.mkdir(parents=True, exist_ok=True)
        shared = args.shared_results / run.name
        try:
            shutil.copytree(run, shared)
            print(f"SHARED_ARTIFACTS={shared}", flush=True)
        except OSError as archive_error:
            print(f"SHARED_COPY_FAILED={archive_error}; WSL evidence retained at {run}", flush=True)
            if state["status"] == "PASS": raise
    print(f"MASK_CASES_PASS: {len(state['passed_cases'])}/28 cases;16 regression+8 width192+4 guard cases", flush=True)
    print("COVERAGE_LIMIT: FP16 B3 C8 indices[7,0,3], selected layouts and strides; "
          "full storage/guards checked. Load safety reviewed in source; no runtime invalid-read detector. "
          "No timing, arbitrary widths or batch coverage.")
    print("REVIEW_REQUIRED: submit output; no automatic independent-learning or performance PASS.")


if __name__ == "__main__":
    main()
