"""Observe actual frozen-v1/grouped compiled specializations at B256 D64/128.

Four independent checked calls, C1024 FP16 interleaved, contiguous inputs,
unique int32 indices0..255. Both arms use four warps. No timing, cache removal,
fresh-compilation assertion, hardware-counter measurement or cause inference.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
def helper(name,filename):
    spec = importlib.util.spec_from_file_location(name,HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

t = helper("grouped_compile_accepted_timing","32-compare-kv-grouped-full.py")
e = helper("grouped_compile_export","28-compare-kv-warps-compile.py")
q,r,c,h,COMMIT = t.q,t.r,t.c,t.h,t.COMMIT
TIMING_RUN = "run-20261010-134931-4l7bljlh"
ARMS = ("frozen_v1","grouped")
PINNED = {
    "32-compare-kv-grouped-full.py":"75b55e6bcd966904ea000f854aa0e0822f6fae2a8adee6133c68794e45e50a97",
    "32-compare-kv-grouped-full.sh":"c60b498630fa45c96f0f57c39fbd5ead5089c8477225b98cf7c17c7ffabc5dda",
    "24-inspect-triton-compile.py":"cddbad7e745d628c6d0452702cd35568a32258521164021a97c32e200fc480a5",
    "28-compare-kv-warps-compile.py":"767c3c5adfe9ea353229c2df928af432fdfc48da28b731d386c05de549fbf055",
}
HELPERS = tuple(t.HELPERS) + tuple(t.NEW_FILES)
EXPORT_HELPERS = ("24-inspect-triton-compile.py","28-compare-kv-warps-compile.py")
NEW_FILES = ("33-inspect-kv-grouped-compile.py","33-inspect-kv-grouped-compile.sh")


def configurations():
    return [dict(batch=256,width=d,capacity=1024,layout="interleaved",gap=0,guard=0,
                 rows=list(range(256))) for d in (64,128)]


def make_plan():
    return dict(configurations=configurations(),arms=list(ARMS),checked_calls=4,
                num_warps=4,grids=dict(frozen_v1=[256],grouped=[64]),
                accepted_timing_run=TIMING_RUN,
                scope="four independent checked calls; actual retained compiler artifacts only; no timing")


def fixture(cfg):
    if cfg not in configurations():
        raise ValueError("UNPLANNED_COMPILER_OBSERVATION_CONFIG")
    b,d = cfg["batch"],cfg["width"]
    k = np.random.default_rng(20261010).integers(0,1024,(b,d),dtype=np.int32).astype(np.float16)
    k[:,0] = np.arange(b,dtype=np.float16)
    before,expected = t.fixture(k)
    return dict(before,expected_cache=expected)


class CompiledCapture:
    """Fixed-config preflight plus the accepted arm's probe; keep its real return."""
    def __init__(self,kernel,arm,cfg):
        if arm not in ARMS or cfg not in configurations():
            raise ValueError("UNPLANNED_COMPILER_OBSERVATION_ARM_OR_CONFIG")
        self.arm,self.width = arm,cfg["width"]
        self.probe = r.GroupedProbe(kernel) if arm == "grouped" else q.LaunchProbe(kernel,4)
        self.reset(False)

    @property
    def attempts(self):
        return self.probe.attempts

    def reset(self,allow):
        self.probe.reset(allow)
        self.compiled = None

    def __getitem__(self,grid):
        launch = self.probe[grid]
        def call(kc,vc,indices,k,v,**kwargs):
            d = self.width
            expected = t.launch_parameters(256,d,d,2*d,self.arm)
            shapes = ((1024,d),(1024,d),(256,),(256,d),(256,d))
            strides = (2*d,2*d,1,d,d)
            tensors = (kc,vc,indices,k,v)
            if (grid != ((64 if self.arm == "grouped" else 256),) or kwargs != expected
                or any(tuple(x.shape) != shape or x.stride(0) != stride
                       for x,shape,stride in zip(tensors,shapes,strides))):
                raise RuntimeError("FIXED_OBSERVATION_LAUNCH_BLOCKED")
            self.compiled = launch(kc,vc,indices,k,v,**kwargs)
            return self.compiled
        return call


def verify_local_sources(directory,accepted_hashes):
    for name in HELPERS:
        if q.sha(directory / name) != accepted_hashes.get(name):
            raise RuntimeError("ACCEPTED32_EXECUTED_HELPER_CHANGED: " + name)
    for name,expected in PINNED.items():
        if q.sha(directory / name) != expected:
            raise RuntimeError("PINNED_OBSERVATION_HELPER_CHANGED: " + name)


def verify_live_kernels(project):
    q.check_pinned(project / "experiments")
    if q.sha(project / "experiments" / r.GROUPED) != t.GROUPED_HASH:
        raise RuntimeError("LIVE_GROUPED_CHANGED")


def accepted_gate(accepted,project):
    status = json.loads((accepted / "status.json").read_text(encoding="utf-8"))
    if (status.get("status") != "COLLECTION_COMPLETE" or status.get("commit") != COMMIT
        or status.get("plan") != json.loads(json.dumps(t.make_plan()))):
        raise RuntimeError("ACCEPTED32_COMPLETE_COLLECTION_REQUIRED")
    if t.summarize(status["results"]) != status.get("summary"):
        raise RuntimeError("ACCEPTED32_SUMMARY_CHANGED")
    for result in status["results"]:
        worker = accepted / f"round-{result['round']}-B{result['batch']}-D{result['width']}"
        if json.loads((worker / "result.json").read_text(encoding="utf-8")) != result:
            raise RuntimeError("ACCEPTED32_WORKER_RECORD_CHANGED")
    t.verify_snapshot(accepted,project)
    old_hashes = json.loads((accepted / "source_hashes.json").read_text(encoding="utf-8"))
    for name in t.NEW_FILES:
        if old_hashes.get(name) != PINNED[name]:
            raise RuntimeError("ACCEPTED32_RUNNER_CHANGED")
    verify_local_sources(HERE,old_hashes)
    verify_live_kernels(project)
    return old_hashes,dict(status="ACCEPTED32_SOURCE_AND_RECORD_GATE_PASS",run=TIMING_RUN,
        status_sha256=q.sha(accepted / "status.json"),source_hashes_sha256=q.sha(accepted / "source_hashes.json"),
        grouped_sha256=t.GROUPED_HASH,
        scope="saved six-case timing source/plan/raw records/summary validated; no timing executed by33")


def verify_snapshot(run,project):
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    required = (*EXPORT_HELPERS,*NEW_FILES,"observation-lesson.md","accepted32-gate.json","accepted32-status.json")
    if any(name not in hashes for name in required):
        raise RuntimeError("INCOMPLETE_GROUPED_COMPILER_SNAPSHOT")
    t.verify_snapshot(run,project)
    for name,expected in PINNED.items():
        if q.sha(run / name) != expected:
            raise RuntimeError("FROZEN_OBSERVATION_HELPER_CHANGED: " + name)
    verify_live_kernels(project)


def snapshot(project,run):
    # Keep actual proposed inputs even if the acceptance/source gate then fails.
    for name in (*HELPERS,*EXPORT_HELPERS,*NEW_FILES):
        shutil.copyfile(HERE / name,run / name)
    for name in (*q.PINNED,r.GROUPED):
        shutil.copyfile(project / "experiments" / name,run / name)
    h.write_json(run / "input_source_hashes.json",{x.name:q.sha(x) for x in run.iterdir() if x.is_file()})
    accepted = project / "experiments/learning/m07-grouped-six-timing" / TIMING_RUN
    old_hashes,gate = accepted_gate(accepted,project)
    verify_local_sources(run,old_hashes)
    t.f.verify_bound_sources(run)
    for name,expected in old_hashes.items():
        destination = run / name
        if destination.exists():
            if q.sha(destination) != expected:
                raise RuntimeError("ACTUAL_SOURCE_DIFFERS_FROM_ACCEPTED32: " + name)
            continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(accepted / name,destination)
    shutil.copyfile(accepted / "status.json",run / "accepted32-status.json")
    shutil.copyfile(HERE / "lessons/m07-08-grouped-compile.md",run / "observation-lesson.md")
    h.write_json(run / "accepted32-gate.json",gate)
    h.write_json(run / "source_hashes.json",{x.relative_to(run).as_posix():q.sha(x) for x in run.rglob("*") if x.is_file()})
    verify_snapshot(run,project)


def checked_call(torch,api,probe,checker,cfg,arm,directory,before,state,case_runner=q.run_case):
    label = f"B256-D{cfg['width']}-{arm}"
    try:
        record,after = case_runner(torch,api,probe,checker,cfg,arm,directory,before)
    except BaseException:
        try:
            directory.mkdir(exist_ok=True)
            state["compiler_evidence"][label] = e.export_compiled(probe.compiled,directory)
        except Exception as capture_error:
            state.setdefault("failure_capture_errors",[]).append(dict(case=label,error=str(capture_error)))
        raise
    report = e.export_compiled(probe.compiled,directory)
    state["correctness"][label] = record
    state["compiler_evidence"][label] = report
    return after,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--shared-results",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    if args.dry_run:
        print(json.dumps(make_plan(),indent=2))
        return
    if args.project is None or args.shared_results is None:
        parser.error("--project and --shared-results required")
    project = args.project.resolve()
    parent = project / "experiments/learning/m07-grouped-compile"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 grouped compiler observation",plan=make_plan(),
        correctness={},compiler_evidence={},pairs=[],
        fixture="NumPy seed20261010 bounded integers with row ID in first column; same config as32, different input values",
        ownership="coach demonstrated grouped expressions; learner applied and ran accepted correctness/timing; coach supplies observation harness")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['plan']['scope']}",flush=True)
    try:
        commit = subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip()
        tracked = subprocess.check_output(["git","-C",str(project),"status","--porcelain","--untracked-files=no"],text=True).strip()
        if commit != COMMIT or tracked:
            raise RuntimeError("PINNED_TRACKED_PROJECT_CHANGED")
        snapshot(project,run)
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__,torch.version.cuda,triton.__version__) != ("2.9.1+cu128","12.8","3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        h.load_module(run / "kv_contract_support.py","kv_contract_support")
        kernels = dict(frozen_v1=h.load_module(run / "triton_store_mask.py","triton_store_mask").store_kv_v1)
        apis = dict(frozen_v1=h.load_module(run / "triton_store_api.py","observe_frozen_v1"),
                    grouped=h.load_module(run / r.GROUPED,"observe_grouped"))
        kernels["grouped"] = apis["grouped"].store_kv_grouped
        checker = h.load_module(run / "check_kv_targets.py","observe_grouped_checker").check_kv_targets
        state.update(status="RUNNING",commit=COMMIT,torch=torch.__version__,triton=triton.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device_properties=str(torch.cuda.get_device_properties(0)))
        for cfg in configurations():
            results = []
            for arm in ARMS:
                label = f"B256-D{cfg['width']}-{arm}"
                state["current_case"] = label
                h.write_json(run / "status.json",state)
                probe = CompiledCapture(kernels[arm],arm,cfg)
                setattr(apis[arm],"store_kv_grouped" if arm == "grouped" else "store_kv_v1",probe)
                directory = run / label
                after,report = checked_call(torch,apis[arm],probe,checker,cfg,arm,directory,fixture(cfg),state)
                results.append(after)
                state.setdefault("runtime_sources",{})[label] = e.runtime_sources(probe.compiled,kernels[arm],directory)
                print(f"CHECKED_CASE={label} ACTUAL_LAUNCH={probe.attempts} COMPILER_OUTPUT={report['status']}",flush=True)
                e.print_observation(label,report)
            for name in ("cache","k","v","indices"):
                np.testing.assert_array_equal(results[0][name],results[1][name],err_msg="PAIR_"+name)
            state["pairs"].append(dict(width=cfg["width"],status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"))
        verify_snapshot(run,project)
        state.update(status="OBSERVATION_READY",current_case=None)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        t.export_run(run,args.shared_results,state["status"] == "OBSERVATION_READY")
    print("GROUPED_COMPILE_OBSERVATION_READY: four checked calls; B256 D64/128; v1 grid256/grouped grid64; both4warps")
    print("LIMIT: no timing, fresh-compilation proof, occupancy/counters, vector-width assumption or performance-cause claim")


if __name__ == "__main__":
    main()
