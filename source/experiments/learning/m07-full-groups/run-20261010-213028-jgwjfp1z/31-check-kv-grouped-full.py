"""M07 grouped full common correctness, reusing accepted fixtures and checkers.

98 configurations x frozen_v1/grouped = 168 nonempty calls + 28 public empty
returns, plus 18 invalid-input rejections per arm. No timing or GPU B0 claim.
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
spec = importlib.util.spec_from_file_location("grouped_full_accepted_helpers",HERE / "30-compare-kv-grouped.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
r,q,c,h,w,COMMIT = m.r,m.q,m.c,m.h,m.w,m.COMMIT
ARMS = r.ARMS
PINNED_30 = {
    "30-compare-kv-grouped.py":"f575c7b08d321639386bc1e62544878ca7098e0eec1e29579f5e9cde7ca16e5b",
    "30-compare-kv-grouped.sh":"0e1b014ca1ed78686f3e611b5185ea554a8074a1be2491c3a662ab5278992f70",
}
NEW_FILES = ("31-check-kv-grouped-full.py","31-check-kv-grouped-full.sh")
HELPERS = tuple(m.HELPERS) + tuple(PINNED_30)


def make_plan():
    return dict(configurations=98,implementation_cases=196,nonempty_calls=168,empty_wrapper_returns=28,
                matched_pairs=98,invalid_rejections=36,invalid_kinds=list(w.INVALID_KINDS),
                arms=dict(frozen_v1=dict(num_warps=4,rows_per_program=1),grouped=dict(num_warps=4,rows_per_program=4)),
                scope="C32 FP16/int32 B0/1/2/3/4/5/17 D64/128/192; selected layouts/gaps/guards only",
                positive_checks="independent expectation; complete cache and input/indices backings, including gaps/guards",
                invalid_checks="contract rejection, zero KV submissions, five argument-view values unchanged; not all backing bytes",
                limitations="B0 is a wrapper return; no timing, arbitrary-layout, model, hardware-counter or AMD result")


def verify_bound_sources(directory):
    q.check_pinned(directory)
    if q.sha(directory / r.GROUPED) != m.GROUPED_HASH:
        raise RuntimeError("ACCEPTED_GROUPED_SOURCE_CHANGED")
    r.review_source((directory / r.GROUPED).read_text(encoding="utf-8-sig"))
    for name,expected in PINNED_30.items():
        if q.sha(directory / name) != expected:
            raise RuntimeError("ACCEPTED_30_HELPER_CHANGED: " + name)


def snapshot(project,run):
    for name in HELPERS + NEW_FILES:
        shutil.copyfile(HERE / name,run / name)
    for name in (*q.PINNED,r.GROUPED):
        shutil.copyfile(project / "experiments" / name,run / name)
    hashes = {f.name:q.sha(f) for f in run.iterdir() if f.is_file()}
    h.write_json(run / "source_hashes.json",hashes)
    verify_bound_sources(run)
    h.write_json(run / "source_review.json",dict(
        grouped=r.review_source((run / r.GROUPED).read_text(encoding="utf-8-sig")),
        actual_ownership="coach demonstrated the three grouped expressions; learner applied them and ran accepted correctness; independent authorship/mastery not inferred",
        frozen_sources=q.PINNED,grouped_sha256=m.GROUPED_HASH,pinned_30=PINNED_30))
    accepted = project / "experiments/learning/m07-grouped" / m.ACCEPTED_RUN
    gate = m.accepted_gate(accepted)  # Also validates all current imported historical helpers.
    h.write_json(run / "accepted_grouped_gate.json",gate)
    shutil.copyfile(HERE / "lessons/m07-06-grouped-full-regression.md",run / "lesson.md")
    hashes.update({name:q.sha(run/name) for name in ("source_review.json","accepted_grouped_gate.json","lesson.md")})
    h.write_json(run / "source_hashes.json",hashes)


def bind_probes(frozen_kernel,apis):
    probes = dict(frozen_v1=q.LaunchProbe(frozen_kernel,4),grouped=r.GroupedProbe(apis["grouped"].store_kv_grouped))
    apis["frozen_v1"].store_kv_v1 = probes["frozen_v1"]
    apis["grouped"].store_kv_grouped = probes["grouped"]
    return probes


def run_invalid_cases(torch,apis,probes,run,state,invalid_runner=w.invalid_case):
    for arm in ARMS:
        for kind in w.INVALID_KINDS:
            directory = run / f"reject-{arm}-{kind}"
            state["current_case"] = directory.name
            h.write_json(run / "status.json",state)
            record = invalid_runner(torch,apis[arm],probes[arm],kind,directory)
            if probes[arm].attempts or record["status"] != "REJECTED_BEFORE_KV_LAUNCH":
                raise AssertionError("INVALID_CASE_DID_NOT_REJECT_WITH_ZERO_SUBMISSIONS")
            state["rejected"].append(dict(record,implementation=arm,submissions=0))


def save_failure(run,state,error):
    state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
    (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")


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
    parent = project / "experiments/learning/m07-grouped-full"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 full grouped correctness",commit=COMMIT,plan=make_plan(),
                 configurations=c.configurations(),cases=[],pairs=[],rejected=[],
                 ownership="coach demonstrated grouped rows/mask/grid; learner applied and ran accepted correctness; coach supplies this regression harness")
    print(f"RUN_DIRECTORY={run}\nSCOPE={state['plan']['scope']}",flush=True)
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
        frozen_kernel = h.load_module(run / "triton_store_mask.py","triton_store_mask").store_kv_v1
        apis = dict(frozen_v1=h.load_module(run / "triton_store_api.py","full_frozen_api"),
                    grouped=h.load_module(run / r.GROUPED,"full_grouped_api"))
        probes = bind_probes(frozen_kernel,apis)
        checker = h.load_module(run / "check_kv_targets.py","full_target_checker").check_kv_targets
        h.write_json(run / "checker_controls.json",c.learner_checker_controls(torch,checker))
        state.update(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        for index,cfg in enumerate(c.configurations()):
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
        run_invalid_cases(torch,apis,probes,run,state)
        assert len(state["cases"]) == 196 and len(state["pairs"]) == 98 and len(state["rejected"]) == 36
        assert sum(x["submissions"] for x in state["cases"]) == 168
        state.update(status="PASS",current_case=None)
    except BaseException as error:
        save_failure(run,state,error)
        raise
    finally:
        h.write_json(run / "status.json",state)
        m.export_run(run,args.shared_results,state["status"] == "PASS")
    print("GROUPED_FULL_CORRECTNESS_PASS: 196/196 = 168 nonempty calls + 28 empty wrapper returns; 98 pairs; 36 invalid rejections with zero KV submissions")
    print("LIMIT: scoped FP16/int32 common matrix; invalid checks compare five argument views; no timing, model or AMD result")


if __name__ == "__main__":
    main()
