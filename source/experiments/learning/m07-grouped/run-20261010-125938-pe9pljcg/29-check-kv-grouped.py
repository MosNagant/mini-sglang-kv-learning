"""M07 grouped rows, fixed four warps: three learner expressions, no timing.

--prepare installs an unfilled template once. Default refuses all remaining
ellipses before importing GPU libraries. 17 configurations x frozen v1/grouped:
30 nonempty submissions + four empty public-wrapper returns, not GPU B0 tests.
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
spec = importlib.util.spec_from_file_location("grouped_prior_helpers",HERE / "27-compare-kv-warps.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
q,c,h,w,COMMIT = p.q,p.c,p.h,p.w,p.COMMIT
GROUPED = "triton_store_grouped.py"
ARMS = ("frozen_v1","grouped")
NEW_FILES = ("29-check-kv-grouped.py","29-check-kv-grouped.sh","29-triton-store-grouped.py")
HELPERS = tuple(q.HELPERS) + ("25-compare-kv-v1.py","27-compare-kv-warps.py")
PRIOR_RUNNERS = {
    "25-compare-kv-v1.py":"02cd7fe12091edbbd1841c203829b9b115ae24fdb56639697e8dcfe7d78b9e53",
    "27-compare-kv-warps.py":"edb64af3a884ffbf7d727dae944fcd181bc9e975c886ce21f3057a27fab42aaa",
}


def configurations():
    common = c.configurations()
    selected = []
    for width,layout,gap,guard in ((64,"interleaved",0,0),(128,"separate",64,0),(192,"separate",64,64)):
        for batch in (1,3,4,5,17):
            selected.append(next(cfg.copy() for cfg in common if
                (cfg["batch"],cfg["width"],cfg["layout"],cfg["gap"],cfg["guard"]) == (batch,width,layout,gap,guard)))
    selected.extend(next(cfg.copy() for cfg in common if
        (cfg["batch"],cfg["width"],cfg["layout"],cfg["gap"],cfg["guard"]) == values)
        for values in ((0,64,"interleaved",0,0),(0,192,"separate",64,64)))
    return selected


def reference_source():
    """CPU source-review reference only; never installed over learner work."""
    source = (HERE / "29-triton-store-grouped.py").read_text(encoding="utf-8")
    return (source.replace("rows = ...","rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)",1)
                  .replace("row_mask = ...","row_mask = rows < BATCH",1)
                  .replace("grid = (...,)","grid = (triton.cdiv(batch, ROWS_PER_PROGRAM),)",1))


def review_source(source):
    tree = ast.parse(source)
    if any(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        raise RuntimeError("LEARNER_TODO: fill the THREE expressions in " + GROUPED)
    if ast.dump(tree) != ast.dump(ast.parse(reference_source())):
        raise RuntimeError("GROUPED_SOURCE_REVIEW_REQUIRED: only the three reviewed expressions may change; no GPU submitted")
    return dict(status="THREE_EXPRESSIONS_REVIEWED",rows_per_program=4,num_warps=4,
                row_index_load_mask="row_mask",kv_load_store_mask="row_mask[:,None] & col_mask[None,:]",
                scope="narrow AST preflight, not a general proof or GPU memory-safety result",
                ownership="learner fills row numbers, row predicate and grid; coach supplies 2-D address/mask scaffold")


def protect_prior_sources(directory):
    q.check_pinned(directory)
    if q.sha(directory / q.CANDIDATE) != p.CANDIDATE_HASH:
        raise RuntimeError("ACCEPTED_ONE_WARP_CANDIDATE_CHANGED: preserve for review; never overwritten")


def prepare(project):
    exp = project / "experiments"
    protect_prior_sources(exp)
    w.install_once(HERE / "29-triton-store-grouped.py",exp / GROUPED)
    return exp / GROUPED


class GroupedProbe:
    """Validate actual grouped launch parameters; one submission has ceil(B/4) programs."""
    def __init__(self,kernel):
        self.kernel = kernel
        self.reset(False)

    def reset(self,allow):
        self.allow,self.attempts = allow,[]

    def __getitem__(self,grid):
        def call(kc,vc,indices,k,v,**kwargs):
            self.attempts.append(dict(grid=list(grid),parameters=kwargs))
            b,d = k.shape
            expected = dict(BATCH=b,WIDTH=d,SRC_ROW_STRIDE=k.stride(0),DST_ROW_STRIDE=kc.stride(0),
                            BLOCK_SIZE=1 << (d-1).bit_length(),ROWS_PER_PROGRAM=4,num_warps=4)
            if not self.allow or b == 0 or grid != ((b+3)//4,) or kwargs != expected or indices.shape != (b,):
                raise RuntimeError("BAD_GROUPED_LAUNCH_BLOCKED_BY_TEST")
            return self.kernel[grid](kc,vc,indices,k,v,**kwargs)
        return call


def snapshot(project,run):
    # Snapshot actual installed learner sources first, even if source review fails.
    for name in HELPERS + NEW_FILES:
        shutil.copyfile(HERE / name,run / name)
    for name in (*q.PINNED,q.CANDIDATE,GROUPED):
        shutil.copyfile(project / "experiments" / name,run / name)
    shutil.copyfile(HERE / "lessons/m07-04-grouped-rows.md",run / "hypothesis-card.md")
    hashes = {f.name:q.sha(f) for f in run.iterdir() if f.is_file()}
    h.write_json(run / "source_hashes.json",hashes)
    protect_prior_sources(run)
    review = review_source((run / GROUPED).read_text(encoding="utf-8-sig"))
    accepted = project / "experiments/learning/m07-warps" / p.ACCEPTED_RUN
    gate = p.accepted_gate(accepted)
    old_hashes = json.loads((accepted / "source_hashes.json").read_text(encoding="utf-8"))
    for name in q.HELPERS:
        if hashes[name] != old_hashes[name]:
            raise RuntimeError("ACCEPTED_HELPER_CHANGED: " + name)
    for name,expected in PRIOR_RUNNERS.items():
        if hashes[name] != expected:
            raise RuntimeError("ACCEPTED_PRIOR_RUNNER_CHANGED: " + name)
    h.write_json(run / "source_review.json",dict(grouped=review,accepted_correctness_gate=gate,
                  frozen_sources=q.PINNED,protected_one_warp_candidate=p.CANDIDATE_HASH))


def cpu_self_test():
    template = (HERE / "29-triton-store-grouped.py").read_text(encoding="utf-8")
    reference = reference_source()
    review_source(reference)
    rejected = []
    mutations = {
        "unfilled_template":template,
        "wrong_row_numbers":reference.replace("pid * ROWS_PER_PROGRAM","pid"),
        "wrong_row_mask":reference.replace("rows < BATCH","rows <= BATCH"),
        "floor_grid":reference.replace("triton.cdiv(batch, ROWS_PER_PROGRAM)","batch // ROWS_PER_PROGRAM"),
        "num_warps_changed":reference.replace("NUM_WARPS = 4","NUM_WARPS = 1"),
        "rows_per_program_changed":reference.replace("ROWS_PER_PROGRAM = 4","ROWS_PER_PROGRAM = 2"),
        "index_row_mask_removed":reference.replace("mask=row_mask, other=0","other=0"),
        "kv_row_mask_removed":reference.replace("row_mask[:, None] & col_mask[None, :]","col_mask[None, :]"),
        "column_mask_changed":reference.replace("cols < WIDTH","cols <= WIDTH"),
        "source_stride_changed":reference.replace("rows[:, None] * SRC_ROW_STRIDE","rows[:, None] * WIDTH"),
        "store_mask_removed":reference.replace("k_values, mask=valid","k_values"),
    }
    for name,source in mutations.items():
        try:
            review_source(source)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError("SOURCE_MUTATION_NOT_REJECTED: " + name)
    controls = []
    for cfg in configurations():
        before = c.fixture(cfg)
        good = c.correct_after(before)
        h.verify_backing(before,good)
        caught = []
        for name,arr in good.items():
            if arr.size == 0:
                continue
            bad = {n:a.copy() for n,a in good.items()}
            bad[name].flat[-1] = -99
            try:
                h.verify_backing(before,bad)
            except AssertionError:
                caught.append(name)
            else:
                raise AssertionError("BACKING_CORRUPTION_NOT_REJECTED: "+name)
        controls.append(dict(config=cfg,positive="PASS",corruptions_rejected=caught))
    class Tensor:
        def __init__(self,shape,stride): self.shape,self.row_stride = shape,stride
        def stride(self,axis):
            assert axis == 0
            return self.row_stride
    class Kernel:
        def __init__(self): self.calls = []
        def __getitem__(self,grid):
            def launch(*args,**kwargs): self.calls.append(dict(grid=list(grid),parameters=kwargs))
            return launch
    forwards,blocked = 0,0
    for cfg in configurations():
        b,d = cfg["batch"],cfg["width"]
        src = d+cfg["gap"]
        dst = (d+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1)
        args = [Tensor((32,d),dst),Tensor((32,d),dst),Tensor((b,),1),Tensor((b,d),src),Tensor((b,d),src)]
        kwargs = dict(BATCH=b,WIDTH=d,SRC_ROW_STRIDE=src,DST_ROW_STRIDE=dst,
                      BLOCK_SIZE=1 << (d-1).bit_length(),ROWS_PER_PROGRAM=4,num_warps=4)
        kernel = Kernel()
        probe = GroupedProbe(kernel)
        probe.reset(b > 0)
        grid = ((b+3)//4,)
        if b:
            probe[grid](*args,**kwargs)
            assert kernel.calls == probe.attempts
            forwards += 1
        for bad_grid,bad_kwargs in ((grid,dict(kwargs,num_warps=1)),(grid,dict(kwargs,ROWS_PER_PROGRAM=2)),
                                   ((grid[0]+1,),kwargs),*(([(grid,kwargs)] if b == 0 else []))):
            count = len(kernel.calls)
            try:
                probe[bad_grid](*args,**bad_kwargs)
            except RuntimeError:
                blocked += 1
            else:
                raise AssertionError("BAD_GROUPED_PROBE_FORWARD")
            assert len(kernel.calls) == count
    with tempfile.TemporaryDirectory(prefix="m07-grouped-cpu-") as tmp:
        project = Path(tmp)
        exp = project / "experiments"
        exp.mkdir()
        for name in q.PINNED:
            shutil.copyfile(HERE / "modules/M06/src" / name,exp / name)
        shutil.copyfile(HERE / "logs/m07-warps-results" / p.ACCEPTED_RUN / q.CANDIDATE,exp / q.CANDIDATE)
        path = prepare(project)
        assert path.read_text(encoding="utf-8") == template
        path.write_text(reference,encoding="utf-8",newline="\n")
        preserved = path.read_bytes()
        assert prepare(project).read_bytes() == preserved
        protect_prior_sources(exp)
        (exp / "triton_store_mask.py").write_text("changed",encoding="utf-8")
        try:
            prepare(project)
        except RuntimeError:
            rejected.append("changed_prior_source")
        else:
            raise AssertionError("CHANGED_PRIOR_SOURCE_NOT_REJECTED")
        assert path.read_bytes() == preserved
    assert len(controls) == 17 and forwards == 15
    assert "torch" not in sys.modules and "triton" not in sys.modules
    return dict(status="CPU_GROUPED_HARNESS_CONTROLS_PASS",GPU="NOT_RUN",configurations=17,
                backing_controls=controls,source_changes_rejected=rejected,
                mock_nonempty_forwards=forwards,mock_bad_launches_blocked=blocked,
                installation="unfilled template created once; learner edit retained; frozen sources protected",
                limitation="AST, fixtures, file installation and launch-probe mocks only; no Triton compilation or GPU result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--shared-results",type=Path)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--cpu-self-test",type=Path)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    if args.cpu_self_test:
        if args.cpu_self_test.exists():
            raise FileExistsError("CPU_REPORT_EXISTS: choose a new report path")
        report = cpu_self_test()
        report["source_sha256"] = {name:q.sha(HERE/name) for name in NEW_FILES}
        args.cpu_self_test.parent.mkdir(parents=True,exist_ok=True)
        h.write_json(args.cpu_self_test,report)
        print("CPU_GROUPED_HARNESS_CONTROLS_PASS: 17 fixture configurations; GPU NOT_RUN")
        return
    if args.project is None or (not args.prepare and args.shared_results is None):
        parser.error("--project required; default checks also require --shared-results")
    project = args.project.resolve()
    if args.prepare:
        path = prepare(project)
        print(f"GROUPED_SOURCE={path}\nPREPARE_COMPLETE: fill THREE expressions; existing file retained; GPU NOT_RUN")
        return
    parent = project / "experiments/learning/m07-grouped"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 grouped-row correctness",commit=COMMIT,
                 arms=dict(frozen_v1=dict(num_warps=4,rows_per_program=1),grouped=dict(num_warps=4,rows_per_program=4)),
                 configurations=configurations(),cases=[],pairs=[],expected_cases=34,expected_nonempty_calls=30,
                 expected_empty_returns=4,scope="C32 FP16/int32; B1/3/4/5/17; selected layouts/gaps/guards; D64/128/192; two B0 configurations",
                 empty_policy="public wrappers return before launching; no GPU B0 or kernel-safety claim",
                 limitations="small positive/empty matrix only; invalid public-input cases not rerun; B256/C1024 not covered; no timing",
                 ownership="learner fills three expressions; 2-D addresses/masks, validation, fixtures and records supplied by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['scope']}",flush=True)
    try:
        snapshot(project,run)
        commit = subprocess.check_output(["git","-C",str(project),"rev-parse","HEAD"],text=True).strip()
        if commit != COMMIT:
            raise RuntimeError("PINNED_PROJECT_COMMIT_CHANGED")
        import torch
        import triton
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__,torch.version.cuda,triton.__version__) != ("2.9.1+cu128","12.8","3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        h.load_module(run / "kv_contract_support.py","kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py","triton_store_mask").store_kv_v1
        apis = dict(frozen_v1=h.load_module(run / "triton_store_api.py","grouped_frozen_v1"),
                    grouped=h.load_module(run / GROUPED,"grouped_learner"))
        probes = dict(frozen_v1=q.LaunchProbe(kernel,4),grouped=GroupedProbe(apis["grouped"].store_kv_grouped))
        apis["frozen_v1"].store_kv_v1 = probes["frozen_v1"]
        apis["grouped"].store_kv_grouped = probes["grouped"]
        checker = h.load_module(run / "check_kv_targets.py","grouped_checker").check_kv_targets
        state.update(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        for index,cfg in enumerate(configurations()):
            before,results = c.fixture(cfg),[]
            for arm in ARMS:
                directory = run / f"case-{index:03d}-{arm}"
                state.update(status="RUNNING",current_case=directory.name)
                h.write_json(run / "status.json",state)
                record,actual = q.run_case(torch,apis[arm],probes[arm],checker,cfg,arm,directory,before)
                state["cases"].append(record)
                results.append(actual)
            for name in ("cache","k","v","indices"):
                np.testing.assert_array_equal(results[0][name],results[1][name],err_msg="PAIR_"+name)
            state["pairs"].append(dict(config=cfg,status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"))
        assert len(state["cases"]) == 34 and len(state["pairs"]) == 17
        state.update(status="PASS",current_case=None)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        p.export_run(run,args.shared_results,state["status"] == "PASS")
    print("GROUPED_CORRECTNESS_PASS: 34/34 = 30 nonempty submissions + four empty wrapper returns; 17 pairs")
    print("LIMIT: small scoped matrix, not full regression or kernel B0 safety; no timing, warp-to-row mapping or independent mastery claim")


if __name__ == "__main__":
    main()
