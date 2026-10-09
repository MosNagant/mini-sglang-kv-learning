"""M06-04: learner public wrapper, scoped contract checks, no benchmark.

4 nonempty GPU cases + 2 empty no-launch cases + deliberately invalid inputs.
The learner fills three expressions; all fixtures, guards and records are coach
scaffolding. An actual GPU PASS is required separately from local CPU checks.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mask_lesson_helpers", HERE / "21-check-kv-mask.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
h = m.h
COMMIT = m.COMMIT


def install_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as out:
            out.write(source.read_text(encoding="utf-8"))
        print(f"CREATED={destination}", flush=True)
    except FileExistsError:
        print(f"EXISTING_FILE_KEPT={destination}", flush=True)


def prepare(project):
    exp = project / "experiments"
    if not (exp / "triton_store_mask.py").is_file():
        raise FileNotFoundError("Accepted mask kernel missing: " + str(exp / "triton_store_mask.py"))
    install_once(HERE / "22-kv-contract-support.py", exp / "kv_contract_support.py")
    install_once(HERE / "22-triton-store-api-template.py", exp / "triton_store_api.py")
    return exp / "triton_store_api.py"


def reference_source():
    # CPU-only checking reference. Never installed over the learner's file.
    text = (HERE / "22-triton-store-api-template.py").read_text(encoding="utf-8")
    return (text.replace("indices.shape != (...,)", "indices.shape != (batch,)")
                .replace("batch == ...", "batch == 0")
                .replace("store_kv_v1[(...,)]", "store_kv_v1[(batch,)]"))


def review_api(source):
    tree = ast.parse(source)
    if any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        raise RuntimeError("LEARNER_TODO: fill the THREE ellipses in triton_store_api.py")
    # Narrow lesson preflight. Other correct implementations need a new review;
    # this is not a general-purpose proof of arbitrary Python safety.
    expected = ast.parse(reference_source())
    if ast.dump(tree) != ast.dump(expected):
        raise RuntimeError("WRAPPER_SOURCE_REVIEW_REQUIRED: three-slot lesson changed; no kernel submitted")
    return {"status": "THREE_EXPRESSIONS_REVIEWED", "independent_mastery": "not inferred"}


def configurations():
    return [
        dict(batch=3, width=64, layout="interleaved", gap=0, guard=0, rows=[7, 0, 3]),
        dict(batch=5, width=64, layout="interleaved", gap=0, guard=0, rows=[7, 0, 3, 5, 1]),
        dict(batch=5, width=192, layout="separate", gap=64, guard=64, rows=[7, 0, 3, 5, 1]),
        dict(batch=1, width=128, layout="separate", gap=0, guard=0, rows=[7]),
        dict(batch=0, width=64, layout="interleaved", gap=0, guard=0, rows=[]),
        dict(batch=0, width=192, layout="separate", gap=64, guard=64, rows=[]),
    ]


def fixture(cfg):
    import numpy as np
    b, d, gap, guard = (cfg[x] for x in ("batch", "width", "gap", "guard"))
    pitch = d + guard
    shape = (8, 2, pitch) if cfg["layout"] == "interleaved" else (2, 8, pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    kc, vc = h.cache_views(cache, cfg["layout"])
    kc[:, d:] = -7
    vc[:, d:] = -7
    before = dict(cache=cache, k=np.full((b, d + gap), -9, dtype=np.float16),
                  v=np.full((b, d + gap), -9, dtype=np.float16),
                  indices=np.array(cfg["rows"], dtype=np.int32))
    vals = np.arange(b*d).reshape(b, d).astype(np.float16) + 10
    before["k"][:, :d] = vals
    before["v"][:, :d] = vals + 1000
    before["expected_cache"] = cache.copy()
    ek, ev = h.cache_views(before["expected_cache"], cfg["layout"])
    for pid, pos in enumerate(cfg["rows"]):
        ek[pos, :d] = before["k"][pid, :d]
        ev[pos, :d] = before["v"][pid, :d]
    return before


def backing_controls():
    records = []
    for cfg in configurations():
        before = fixture(cfg)
        good = {name: before["expected_cache" if name == "cache" else name].copy()
                for name in ("cache", "k", "v", "indices")}
        h.verify_backing(before, good)
        count = 0
        # Test the independent full-storage checker, including empty cases.
        for name, array in good.items():
            if array.size == 0:
                continue
            bad = {n: a.copy() for n, a in good.items()}
            bad[name].flat[-1] = -99
            try:
                h.verify_backing(before, bad)
            except AssertionError:
                count += 1
            else:
                raise AssertionError("MISSED_BACKING_CORRUPTION: " + name)
        records.append(dict(config=cfg, positive="PASS", corruptions_rejected=count))
    return records


def gpu_packet(torch, cfg):
    before = fixture(cfg)
    buffers = {name: torch.from_numpy(before[name].copy()).cuda()
               for name in ("cache", "k", "v", "indices")}
    d = cfg["width"]
    if cfg["batch"] == 0:
        # NumPy empty arrays have zero strides. Construct genuine contiguous
        # PyTorch empty backings so the declared row/index strides are retained.
        for name in ("k", "v"):
            buffers[name] = torch.empty((0, d + cfg["gap"]), dtype=torch.float16, device="cuda")
        buffers["indices"] = torch.empty((0,), dtype=torch.int32, device="cuda")
    kc, vc = h.cache_views(buffers["cache"], cfg["layout"])
    args = [kc[:, :d], vc[:, :d], buffers["indices"],
            buffers["k"][:, :d], buffers["v"][:, :d]]
    return before, buffers, args


def tensor_info(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), data_ptr=t.data_ptr())


class LaunchProbe:
    """Checks each submission BEFORE forwarding to the unchanged GPU kernel."""
    def __init__(self, kernel):
        self.kernel = kernel
        self.reset(False)

    def reset(self, allow):
        self.allow = allow
        self.attempts = []

    def __getitem__(self, grid):
        def call(kc, vc, indices, k, v, **kwargs):
            self.attempts.append(dict(grid=list(grid), parameters=kwargs))
            if not self.allow:
                raise RuntimeError("UNEXPECTED_KV_LAUNCH_BLOCKED_BY_TEST")
            b, d = k.shape
            expected = dict(WIDTH=d, SRC_ROW_STRIDE=k.stride(0),
                            DST_ROW_STRIDE=kc.stride(0),
                            BLOCK_SIZE=1 << (d-1).bit_length(), num_warps=4)
            if grid != (b,) or kwargs != expected or b == 0 or indices.shape != (b,):
                raise RuntimeError("BAD_LAUNCH_BLOCKED_BY_TEST")
            return self.kernel[grid](kc, vc, indices, k, v, **kwargs)
        return call


def positive_case(torch, api, probe, checker, cfg, directory):
    directory.mkdir()
    before, buffers, args = gpu_packet(torch, cfg)
    h.write_json(directory / "input.json", dict(config=cfg,
                 args=[tensor_info(t) for t in args],
                 backing_before={name: a.tolist() for name, a in before.items()}))
    probe.reset(cfg["batch"] > 0)
    try:
        api.store_cache_triton(*args)
        torch.cuda.synchronize()
    finally:
        # Preserve attempted parameters even if launch/synchronization fails.
        # This does not require touching a possibly failed CUDA context.
        h.write_json(directory / "launches.json", probe.attempts)
    after = {name: t.cpu().numpy().copy() for name, t in buffers.items()}
    h.write_json(directory / "result.json", dict(
        backing_after={n: a.tolist() for n, a in after.items()}, launches=probe.attempts))
    h.verify_backing(before, after)
    kc, vc = h.cache_views(after["cache"], cfg["layout"])
    d = cfg["width"]
    checker(torch.from_numpy(kc[:, :d].copy()), torch.from_numpy(vc[:, :d].copy()),
            torch.from_numpy(before["k"][:, :d].copy()),
            torch.from_numpy(before["v"][:, :d].copy()), cfg["rows"])
    expected_count = int(cfg["batch"] > 0)
    if len(probe.attempts) != expected_count:
        raise AssertionError("LAUNCH_COUNT_MISMATCH")
    record = dict(status="PASS", config=cfg, kv_launches=len(probe.attempts))
    h.write_json(directory / "status.json", record)
    print(f"WRAPPER_CASE_PASS B={cfg['batch']} D={d} layout={cfg['layout']} "
          f"KV_launches={len(probe.attempts)}", flush=True)
    return record


INVALID_KINDS = ("indices_too_many", "indices_too_few", "v_shape", "k_dtype",
                 "indices_dtype", "cpu_k", "cpu_indices", "unsupported_width",
                 "column_stride", "index_stride", "unequal_source_stride",
                 "unsupported_source_stride", "cache_overlap", "source_cache_overlap",
                 "duplicate_indices", "negative_index", "out_of_range_index",
                 "empty_with_one_index")


def invalid_case(torch, api, probe, kind, directory):
    directory.mkdir()
    cfg = configurations()[0].copy()
    if kind == "unsupported_width": cfg["width"] = 96
    if kind == "unsupported_source_stride": cfg["gap"] = 32
    if kind == "empty_with_one_index": cfg.update(batch=0, rows=[])
    before, buffers, a = gpu_packet(torch, cfg)
    if kind == "indices_too_many": a[2] = torch.tensor([7, 0, 3, 5, 1], device="cuda", dtype=torch.int32)
    elif kind == "indices_too_few": a[2] = a[2][:2]
    elif kind == "v_shape": a[4] = a[4][:2]
    elif kind == "k_dtype": a[3] = a[3].float()
    elif kind == "indices_dtype": a[2] = a[2].long()
    elif kind == "cpu_k": a[3] = a[3].cpu()
    elif kind == "cpu_indices": a[2] = a[2].cpu()
    elif kind == "column_stride":
        a[3] = torch.zeros((3,128),device="cuda",dtype=torch.float16)[:,::2]
        a[4] = torch.zeros((3,128),device="cuda",dtype=torch.float16)[:,::2]
    elif kind == "index_stride": a[2] = torch.tensor([7,0,0,0,3,0],device="cuda",dtype=torch.int32)[::2]
    elif kind == "unequal_source_stride": a[4] = torch.zeros((3,128),device="cuda",dtype=torch.float16)[:,:64]
    elif kind == "cache_overlap": a[1] = a[0]
    elif kind == "source_cache_overlap": a[3], a[4] = a[0][:3], a[1][:3]
    elif kind in ("duplicate_indices", "negative_index", "out_of_range_index", "empty_with_one_index"):
        values = {"duplicate_indices":[7,7,3], "negative_index":[-1,0,3],
                  "out_of_range_index":[8,0,3], "empty_with_one_index":[7]}[kind]
        a[2] = torch.tensor(values,device="cuda",dtype=torch.int32)
    saved = [t.detach().cpu().clone() for t in a]
    h.write_json(directory / "input.json", dict(kind=kind, args=[tensor_info(t) for t in a],
                                               values=[t.tolist() for t in saved]))
    probe.reset(False)
    rejection = None
    try:
        api.store_cache_triton(*a)
    except (ValueError, TypeError) as error:
        rejection = dict(type=type(error).__name__, message=str(error))
    finally:
        after = [t.detach().cpu().clone() for t in a]
        h.write_json(directory / "result.json", dict(rejection=rejection,
                     values_after=[t.tolist() for t in after], launches=probe.attempts))
    if rejection is None or probe.attempts:
        raise AssertionError("INVALID_INPUT_NOT_REJECTED_BY_PUBLIC_WRAPPER: " + kind)
    for old, new in zip(saved, after):
        assert torch.equal(old, new), "INVALID_CALL_CHANGED_INPUT_OR_CACHE: " + kind
    record = dict(status="REJECTED_BEFORE_KV_LAUNCH", kind=kind, rejection=rejection)
    h.write_json(directory / "status.json", record)
    print("EXPECTED_REJECTION " + kind + ": " + rejection["message"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cpu-self-test", type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: assertions must execute")
    if args.cpu_self_test:
        controls = backing_controls()
        review_api(reference_source())
        try:
            review_api((HERE/"22-triton-store-api-template.py").read_text(encoding="utf-8"))
        except RuntimeError as e:
            if "LEARNER_TODO" not in str(e): raise
        else: raise AssertionError("Unfilled template was accepted")
        args.cpu_self_test.parent.mkdir(parents=True, exist_ok=True)
        h.write_json(args.cpu_self_test, dict(status="CPU_CONTROLS_PASS", controls=controls,
                     GPU="NOT_RUN", scope="fixtures, full-storage controls and three-slot source preflight"))
        print("CPU_CONTROLS_PASS: fixtures and source review only; GPU NOT_RUN")
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and --shared-results required")
    project = args.project.resolve()
    def git(*a): return subprocess.check_output(["git","-C",str(project),*a],text=True).strip()
    if git("rev-parse","HEAD") != COMMIT: raise RuntimeError("COMMIT_MISMATCH")
    if git("status","--porcelain","--","python/minisgl/kernel"):
        raise RuntimeError("ORIGINAL_KERNEL_CHANGED")
    student = prepare(project)
    if args.prepare:
        print("LEARNER_SOURCE_BEGIN\n" + student.read_text(encoding="utf-8-sig") + "LEARNER_SOURCE_END")
        print("WRAPPER_PREPARE_COMPLETE: fill three ellipses; no GPU run")
        return
    parent = project / "experiments/learning/m06-wrapper"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING", lesson="M06-04", commit=COMMIT, passed=[], rejected=[],
                 scope="4 nonempty KV calls, 2 empty no-op calls, 18 invalid public calls; no timing",
                 ownership="Learner fills three launch expressions; helper/fixtures/records by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}",flush=True)
    current_dir = None
    try:
        files = [(Path(__file__),"coach_runner.py"),(HERE/"22-check-kv-wrapper.sh","coach_entry.sh"),
                 (HERE/"22-triton-store-api-template.py","coach_template.py"),
                 (HERE/"22-kv-contract-support.py","coach_contract.py"),
                 (HERE/"21-check-kv-mask.py","mask_helpers.py"),
                 (HERE/"14-check-kv-widths.py","layout_helpers.py"),
                 (HERE/"21-triton-store-mask.py","coach_mask_reference.py"),
                 (student,"triton_store_api.py"),
                 (project/"experiments/kv_contract_support.py","kv_contract_support.py"),
                 (project/"experiments/triton_store_mask.py","triton_store_mask.py"),
                 (project/"experiments/check_kv_targets.py","check_kv_targets.py")]
        hashes={}
        for source, rel in files:
            shutil.copyfile(source,run/rel)
            hashes[rel]=hashlib.sha256((run/rel).read_bytes()).hexdigest()
        h.write_json(run/"source_hashes.json",hashes)
        if hashes["coach_contract.py"] != hashes["kv_contract_support.py"]:
            raise RuntimeError("EXISTING_SUPPORT_CHANGED: retained; review before run")
        api_source=(run/"triton_store_api.py").read_text(encoding="utf-8-sig")
        state["source_review"]=review_api(api_source)
        m.validate_mask_source((run/"triton_store_mask.py").read_text(encoding="utf-8-sig"))
        print("LEARNER_SOURCE_BEGIN\n"+api_source+"LEARNER_SOURCE_END",flush=True)
        import torch
        import triton
        if not torch.cuda.is_available(): raise RuntimeError("CUDA_NOT_AVAILABLE")
        support=h.load_module(run/"kv_contract_support.py","kv_contract_support")
        kernel=h.load_module(run/"triton_store_mask.py","triton_store_mask")
        api=h.load_module(run/"triton_store_api.py","learner_wrapper")
        checker=h.load_module(run/"check_kv_targets.py","wrapper_checker").check_kv_targets
        h.write_json(run/"checker_controls.json",{str(d):h.checker_controls(torch,checker,d) for d in (64,128,192)})
        h.write_json(run/"backing_controls.json",backing_controls())
        probe=LaunchProbe(kernel.store_kv_v1)
        api.store_kv_v1=probe
        state.update(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0),num_warps=4,
                     public_api_policy="checks index VALUES on CPU before nonempty launch; may synchronize; not timed")
        for i,cfg in enumerate(configurations()):
            current_dir=run/f"valid-{i}-B{cfg['batch']}-D{cfg['width']}"
            state.update(status="RUNNING",current_case=current_dir.name)
            h.write_json(run/"status.json",state)
            state["passed"].append(positive_case(torch,api,probe,checker,cfg,current_dir))
        for kind in INVALID_KINDS:
            current_dir=run/("invalid-"+kind)
            state.update(current_case=current_dir.name)
            h.write_json(run/"status.json",state)
            state["rejected"].append(invalid_case(torch,api,probe,kind,current_dir))
        state.update(status="PASS",current_case=None)
    except Exception as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        if current_dir and current_dir.is_dir():
            h.write_json(current_dir/"status.json",dict(status="FAILED",error=str(error)))
        raise
    finally:
        h.write_json(run/"status.json",state)
        args.shared_results.mkdir(parents=True,exist_ok=True)
        shared=args.shared_results/run.name
        try:
            shutil.copytree(run,shared)
            print(f"SHARED_ARTIFACTS={shared}",flush=True)
        except OSError as e:
            print(f"SHARED_COPY_FAILED={e}; WSL evidence retained at {run}",flush=True)
            if state["status"] == "PASS": raise
    print("WRAPPER_CASES_PASS: 4 nonempty GPU cases + 2 empty no-launch cases + 18 rejected invalid calls")
    print("COVERAGE_LIMIT: FP16/int32; selected B0/1/3/5 and D64/128/192 layouts/strides; not full M06-05 suite or benchmark")
    print("REVIEW_REQUIRED: review code and saved results; independent mastery is not inferred")


if __name__ == "__main__":
    main()
