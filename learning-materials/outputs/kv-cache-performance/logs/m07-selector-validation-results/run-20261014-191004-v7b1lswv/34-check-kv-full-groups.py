"""M07 full-group constexpr branch: one learner condition, scoped correctness.

--prepare creates an unfilled candidate once. Default checks 23 configurations
x two arms: 42 nonempty submissions and four public empty-wrapper returns.
Only four selected actual compiled returns are exported. No timing.
"""
import argparse
import ast
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

t = helper("full_groups_accepted_helpers","32-compare-kv-grouped-full.py")
e = helper("full_groups_compile_export","28-compare-kv-warps-compile.py")
r,q,c,h,COMMIT = t.r,t.q,t.c,t.h,t.COMMIT
CANDIDATE = "triton_store_full_groups.py"
ARMS = ("accepted_grouped","full_groups")
NEW_FILES = ("34-check-kv-full-groups.py","34-check-kv-full-groups.sh","34-full-groups-cpu-test.py")
EXTRA_PINS = {
    "32-compare-kv-grouped-full.py":"75b55e6bcd966904ea000f854aa0e0822f6fae2a8adee6133c68794e45e50a97",
    "32-compare-kv-grouped-full.sh":"c60b498630fa45c96f0f57c39fbd5ead5089c8477225b98cf7c17c7ffabc5dda",
    "24-inspect-triton-compile.py":"cddbad7e745d628c6d0452702cd35568a32258521164021a97c32e200fc480a5",
    "28-compare-kv-warps-compile.py":"767c3c5adfe9ea353229c2df928af432fdfc48da28b731d386c05de549fbf055",
}
HELPERS = tuple(t.HELPERS) + tuple(EXTRA_PINS)
ACCEPTED_TIMING_RUN = "run-20261010-134931-4l7bljlh"
CANDIDATE_DOCSTRING = '''"""M07 full groups: fill exactly ONE if condition marked TODO.

Coach supplies this branch structure and the accepted grouped implementation.
Learner supplies the constexpr condition; four rows/program, four warps,
addresses, column masks, tail branch and public contract stay fixed.
The accepted grouped row/grid expressions were demonstrated by the coach.
This checked public entry is not a timing/Graph entry.
"""'''
BRANCH_TEMPLATE = """    # TODO: fill only the condition of this compile-time branch.
    if ...:
        row_mask = tl.full((ROWS_PER_PROGRAM,), True, tl.int1)
    else:
        row_mask = rows < BATCH
"""


def configurations():
    selected = []
    for d,layout,gap,guard in ((64,"interleaved",0,0),(128,"separate",64,0),(192,"separate",64,64)):
        for b in (1,3,4,5,8,17,256):
            selected.append(dict(batch=b,width=d,capacity=512,layout=layout,gap=gap,guard=guard,
                                 rows=[(37*i+11)%512 for i in range(b)]))
    selected.extend(dict(batch=0,width=d,capacity=512,layout=layout,gap=gap,guard=guard,rows=[])
                    for d,layout,gap,guard in ((64,"interleaved",0,0),(192,"separate",64,64)))
    return selected


def make_plan():
    return dict(configurations=23,implementation_cases=46,matched_pairs=23,nonempty_submissions=42,
        empty_wrapper_returns=4,arms=list(ARMS),capacity=512,num_warps=4,rows_per_program=4,
        candidate_full_group_configurations=9,candidate_tail_group_configurations=12,
        retained_specializations=[dict(batch=256,width=128,branch="full"),dict(batch=5,width=192,branch="tail")],
        scope="FP16/int32; B1/3/4/5/8/17/256; three selected width/layout/gap/guard combinations and two B0 cases",
        limits="no timing or hardware out-of-bounds detector; B0 is a public-wrapper return; no arbitrary-layout/model/AMD claim")


def template_source(grouped_source):
    r.review_source(grouped_source)
    old_docstring = ast.get_source_segment(grouped_source,ast.parse(grouped_source).body[0])
    grouped_source = grouped_source.replace(old_docstring,CANDIDATE_DOCSTRING,1)
    old = "    row_mask = rows < BATCH\n"
    if grouped_source.count(old) != 1:
        raise RuntimeError("ACCEPTED_GROUPED_TEMPLATE_ANCHOR_CHANGED")
    return grouped_source.replace(old,BRANCH_TEMPLATE,1)


def review_source(source,grouped_source):
    tree = ast.parse(source)
    if any(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        raise RuntimeError("LEARNER_TODO: fill the single if condition in " + CANDIDATE)
    # Reference is only for source review; --prepare always writes the TODO.
    reference = template_source(grouped_source).replace("if ...:","if BATCH % ROWS_PER_PROGRAM == 0:",1)
    if ast.dump(tree) != ast.dump(ast.parse(reference)):
        raise RuntimeError("FULL_GROUPS_SOURCE_REVIEW_REQUIRED: only the reviewed constexpr condition is accepted; no GPU submitted")
    return dict(status="ONE_CONDITION_REVIEWED",num_warps=4,rows_per_program=4,
        full_branch="vector tl.int1 row mask; column mask unchanged",tail_branch="rows < BATCH",
        scope="narrow AST comparison; not a hardware memory-safety proof",
        ownership="learner fills one condition; coach supplies branch structure and checks; baseline grouped expressions previously demonstrated by coach")


def verify_baseline(directory):
    q.check_pinned(directory)
    if q.sha(directory / r.GROUPED) != t.GROUPED_HASH:
        raise RuntimeError("ACCEPTED_GROUPED_SOURCE_CHANGED")
    return (directory / r.GROUPED).read_text(encoding="utf-8-sig")


def prepare(project):
    exp = project / "experiments"
    source = template_source(verify_baseline(exp))
    destination = exp / CANDIDATE
    try:
        with destination.open("x",encoding="utf-8",newline="\n") as file:
            file.write(source)
    except FileExistsError:
        pass
    return destination


def fixture(cfg):
    if cfg not in configurations():
        raise ValueError("FULL_GROUPS_CONFIGURATION_OUT_OF_SCOPE")
    b,d,capacity = cfg["batch"],cfg["width"],cfg["capacity"]
    pitch = d+cfg["guard"]
    shape = (capacity,2,pitch) if cfg["layout"] == "interleaved" else (2,capacity,pitch)
    cache = np.full(shape,-1,dtype=np.float16)
    for view in h.cache_views(cache,cfg["layout"]):
        view[:,d:] = -7
    before = dict(cache=cache,k=np.full((b,d+cfg["gap"]),-9,dtype=np.float16),
                  v=np.full((b,d+cfg["gap"]),-9,dtype=np.float16),indices=np.array(cfg["rows"],dtype=np.int32))
    values = np.random.default_rng(20261011).integers(0,1024,(b,d),dtype=np.int32).astype(np.float16)
    values[:,0] = np.arange(b,dtype=np.float16)
    before["k"][:,:d] = values
    before["v"][:,:d] = values+np.float16(1024)
    before["expected_cache"] = cache.copy()
    ek,ev = h.cache_views(before["expected_cache"],cfg["layout"])
    for row,pos in enumerate(cfg["rows"]):
        ek[pos,:d],ev[pos,:d] = before["k"][row,:d],before["v"][row,:d]
    return before


class CompiledCapture(r.GroupedProbe):
    def reset(self,allow):
        super().reset(allow)
        self.compiled = None

    def __getitem__(self,grid):
        launch = super().__getitem__(grid)
        def call(*args,**kwargs):
            self.compiled = launch(*args,**kwargs)
            return self.compiled
        return call


def selected_compile(cfg):
    return (cfg["batch"],cfg["width"]) in ((256,128),(5,192))


def verify_snapshot(run):
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    for name,expected in hashes.items():
        if q.sha(run / name) != expected:
            raise RuntimeError("FULL_GROUPS_SNAPSHOT_CHANGED: " + name)
    for name,expected in EXTRA_PINS.items():
        if q.sha(run / name) != expected:
            raise RuntimeError("ACCEPTED_EXTRA_HELPER_CHANGED: " + name)
    review_source((run / CANDIDATE).read_text(encoding="utf-8-sig"),verify_baseline(run))


def snapshot(project,run):
    try:
        for name in (*HELPERS,*NEW_FILES):
            shutil.copyfile(HERE / name,run / name)
        for name in (*q.PINNED,r.GROUPED,CANDIDATE):
            shutil.copyfile(project / "experiments" / name,run / name)
        shutil.copyfile(HERE / "lessons/m07-09-full-groups.md",run / "lesson.md")
    finally:
        h.write_json(run / "source_hashes.json",{x.name:q.sha(x) for x in run.iterdir() if x.is_file()})
    grouped = verify_baseline(run)
    review = review_source((run / CANDIDATE).read_text(encoding="utf-8-sig"),grouped)
    accepted = project / "experiments/learning/m07-grouped-six-timing" / ACCEPTED_TIMING_RUN
    # Reuse the archived32 source/31-gate receipt, without rerunning or re-auditing old arrays.
    t.verify_snapshot(accepted,project)
    status = json.loads((accepted / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COLLECTION_COMPLETE" or status.get("commit") != COMMIT:
        raise RuntimeError("ACCEPTED32_COLLECTION_REQUIRED")
    gate = json.loads((accepted / "accepted_full_grouped_gate.json").read_text(encoding="utf-8"))
    if (gate.get("status") != "ACCEPTED_31_FULL_EVIDENCE_GATE_PASS" or gate.get("grouped_sha256") != t.GROUPED_HASH
        or gate.get("status_sha256") != q.sha(accepted / "prior_full_grouped_status.json")
        or gate.get("source_hashes_sha256") != q.sha(accepted / "prior_full_grouped_source_hashes.json")):
        raise RuntimeError("ACCEPTED31_ARCHIVED_GATE_RECEIPT_CHANGED")
    old_hashes = json.loads((accepted / "source_hashes.json").read_text(encoding="utf-8"))
    for name in t.HELPERS:
        if q.sha(run / name) != old_hashes.get(name):
            raise RuntimeError("ACCEPTED_EXECUTED_HELPER_CHANGED: " + name)
    h.write_json(run / "source_review.json",dict(candidate=review,accepted_grouped_sha256=t.GROUPED_HASH,
        candidate_sha256=q.sha(run/CANDIDATE),accepted31_gate=gate))
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    hashes["source_review.json"] = q.sha(run / "source_review.json")
    h.write_json(run / "source_hashes.json",hashes)
    verify_snapshot(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--shared-results",type=Path)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    if args.dry_run:
        print(json.dumps(make_plan(),indent=2))
        return
    if args.project is None or (not args.prepare and args.shared_results is None):
        parser.error("--project required; checking additionally requires --shared-results")
    project = args.project.resolve()
    if args.prepare:
        print(f"FULL_GROUPS_SOURCE={prepare(project)}\nPREPARE_COMPLETE: one if-condition TODO; existing candidate retained; GPU NOT_RUN")
        return
    parent = project / "experiments/learning/m07-full-groups"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 full-group constexpr condition",commit=COMMIT,plan=make_plan(),
        configurations=configurations(),cases=[],pairs=[],compiler_evidence={},
        ownership="learner supplies one constexpr condition; coach supplies scaffolding; baseline grouped expressions previously demonstrated by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['plan']['scope']}",flush=True)
    try:
        snapshot(project,run)
        if subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip() != COMMIT:
            raise RuntimeError("PINNED_PROJECT_COMMIT_CHANGED")
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__,torch.version.cuda,triton.__version__) != ("2.9.1+cu128","12.8","3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        h.load_module(run / "kv_contract_support.py","kv_contract_support")
        apis = {arm:h.load_module(run/name,"full_groups_"+arm) for arm,name in
                (("accepted_grouped",r.GROUPED),("full_groups",CANDIDATE))}
        kernels = {arm:api.store_kv_grouped for arm,api in apis.items()}
        probes = {arm:CompiledCapture(kernel) for arm,kernel in kernels.items()}
        for arm in ARMS:
            apis[arm].store_kv_grouped = probes[arm]
        checker = h.load_module(run / "check_kv_targets.py","full_groups_checker").check_kv_targets
        state.update(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        for index,cfg in enumerate(configurations()):
            before,results = fixture(cfg),[]
            for arm in ARMS:
                directory = run / f"case-{index:03d}-{arm}"
                state.update(status="RUNNING",current_case=directory.name)
                h.write_json(run / "status.json",state)
                try:
                    record,actual = q.run_case(torch,apis[arm],probes[arm],checker,cfg,arm,directory,before)
                except BaseException:
                    if selected_compile(cfg):
                        try:
                            state["compiler_evidence"][directory.name] = e.export_compiled(probes[arm].compiled,directory)
                        except Exception as capture_error:
                            state.setdefault("failure_capture_errors",[]).append(dict(case=directory.name,error=str(capture_error)))
                    raise
                state["cases"].append(record)
                results.append(actual)
                if selected_compile(cfg):
                    report = e.export_compiled(probes[arm].compiled,directory)
                    state["compiler_evidence"][directory.name] = report
                    state.setdefault("runtime_sources",{})[directory.name] = e.runtime_sources(probes[arm].compiled,kernels[arm],directory)
                    print(f"COMPILE_CASE={directory.name} B={cfg['batch']} D={cfg['width']} ACTUAL_LAUNCH={probes[arm].attempts} OUTPUT={report['status']}",flush=True)
                    e.print_observation(directory.name,report)
            for name in ("cache","k","v","indices"):
                np.testing.assert_array_equal(results[0][name],results[1][name],err_msg="PAIR_"+name)
            state["pairs"].append(dict(config=cfg,status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"))
        assert len(state["cases"]) == 46 and len(state["pairs"]) == 23
        assert sum(x["submissions"] for x in state["cases"]) == 42
        assert len(state["compiler_evidence"]) == 4
        verify_snapshot(run)
        state.update(status="PASS",current_case=None)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        t.export_run(run,args.shared_results,state["status"] == "PASS")
    print("FULL_GROUPS_CORRECTNESS_PASS: 46 results = 42 nonempty submissions + 4 empty wrapper returns; 23 pairs")
    print("LIMIT: no timing or hardware bounds detector; four actual compiled returns observed, cache reuse possible")


if __name__ == "__main__":
    main()
