"""CPU-only tests for new M07 timing orchestration; no torch/WSL/GPU import."""
import ast
import copy
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grouped_timing_under_test", HERE / "30-compare-kv-grouped.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new report path")
    rejected = []
    def expect_failure(name, call, expected):
        try:
            call()
        except expected:
            rejected.append(name)
        else:
            raise AssertionError("NEGATIVE_CONTROL_NOT_REJECTED: " + name)
    plan = t.make_plan()
    assert len(plan["schedule"]) == 6 and plan["interval_averages"] == 1080
    assert {(x["round"],x["batch"],x["width"]) for x in plan["schedule"]} == {
        (r,b,d) for r in (1,2,3) for b,d in ((1,64),(256,128))}
    positions = {}
    for r in (1,2,3):
        counts = Counter()
        expected = {(i,p) for i in t.IMPLEMENTATIONS for p in t.POLICIES}
        for sample in range(30):
            order = t.measurement_order(sample,r)
            assert len(order) == 6 and set(order) == expected
            for position,item in enumerate(order):
                counts[(item,position)] += 1
        assert set(counts.values()) == {5} and len(counts) == 36
        positions[r] = "each of six arm/policy items appears at each position exactly five times"
    synthetic = []
    for item in plan["schedule"]:
        r = item["round"]
        bases = dict(original_cuda=(6+2*r)/1000, frozen_v1=.006, grouped=2*r/1000)
        values = {impl:{policy:[bases[impl]*(2 if policy == "eager" else 1)*(1+(sample-14.5)*.001)
                               for sample in range(30)] for policy in t.POLICIES} for impl in t.IMPLEMENTATIONS}
        records = [dict(sample=s,implementation=i,policy=p,ms_per_call=values[i][p][s])
                   for s in range(30) for i,p in t.measurement_order(s,r)]
        batch,width = item["batch"],item["width"]
        launches = dict(original_cuda=dict(entry="upstream store_cache",public_calls=1))
        for arm in ("frozen_v1","grouped"):
            launches[arm] = [dict(grid=[(batch+3)//4 if arm == "grouped" else batch],
                                 parameters=t.launch_parameters(batch,width,width,2*width,arm))]
        synthetic.append(dict(item,status="COLLECTED",correctness_checks=190,calls_per_interval=100,
                              capacity=1024,config=dict(layout="interleaved",input_gap=0),actual_launches=launches,
                              samples_ms_per_call=values,measurements=records))
    summary = t.summarize(synthetic)
    for row in summary:
        for policy in t.POLICIES:
            got = row[policy]
            assert math.isclose(got["speedup_original_over_grouped"],2.5)
            assert math.isclose(got["speedup_v1_over_grouped"],1.5)
            for observed, expected in zip(got["paired_round_speedups_v1_over_grouped"],(3,1.5,1)):
                assert math.isclose(observed,expected)
            for observed, expected in zip(got["paired_round_speedups_original_over_grouped"],(4,2.5,2)):
                assert math.isclose(observed,expected)
            assert math.isclose(got["grouped"]["representative_us"],4*(2 if policy == "eager" else 1))
    expect_failure("missing_worker",lambda:t.summarize(synthetic[:-1]),ValueError)
    expect_failure("duplicate_worker",lambda:t.summarize(synthetic+[synthetic[0]]),ValueError)
    changes = {
        "noncollected":lambda x:x[0].update(status="FAILED"),
        "unplanned_case":lambda x:x[0].update(batch=16),
        "missing_correctness":lambda x:x[0].update(correctness_checks=189),
        "wrong_interval_calls":lambda x:x[0].update(calls_per_interval=99),
        "wrong_grouped_grid":lambda x:x[0]["actual_launches"]["grouped"][0].update(grid=[999]),
        "wrong_grouped_warps":lambda x:x[0]["actual_launches"]["grouped"][0]["parameters"].update(num_warps=1),
        "short_samples":lambda x:x[0]["samples_ms_per_call"]["grouped"]["graph"].pop(),
        "swapped_order":lambda x:x[0]["measurements"].reverse(),
        "raw_sample_mismatch":lambda x:x[0]["measurements"][0].update(ms_per_call=999),
    }
    for value in (float("nan"),float("inf"),0,-1):
        changes["invalid_value_"+str(value)] = lambda x,v=value:x[0]["samples_ms_per_call"]["grouped"]["graph"].__setitem__(0,v)
    for name,change in changes.items():
        bad = copy.deepcopy(synthetic)
        change(bad)
        expect_failure(name,lambda:t.summarize(bad),ValueError)
    expect_failure("unplanned_B1_D128",lambda:t.validate_worker_case(1,128,1),ValueError)
    expect_failure("unplanned_round",lambda:t.validate_worker_case(1,64,4),ValueError)
    # Exercise the new raw grouped callable with no GPU/runtime implementation.
    tree = ast.parse((HERE/"30-compare-kv-grouped.py").read_text(encoding="utf-8"))
    worker_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "worker")
    grouped_node = next(n for n in ast.walk(worker_node) if isinstance(n,ast.FunctionDef) and n.name == "grouped_call")
    class TraceKernel:
        def __init__(self): self.calls = []
        def __getitem__(self,grid):
            def call(*args,**kwargs): self.calls.append(dict(grid=list(grid),parameters=kwargs))
            return call
    fixture_controls = []
    for batch,width in t.CASES:
        kernel = TraceKernel()
        namespace = dict(grouped_kernel=kernel,batch=batch,width=width,src=width,dst=2*width,block=width,
                         grouped_api=SimpleNamespace(NUM_WARPS=4,ROWS_PER_PROGRAM=4),
                         kc=None,vc=None,indices=None,k=None,v=None)
        exec(compile(ast.Module(body=[grouped_node],type_ignores=[]),"actual-grouped-launch","exec"),namespace)
        namespace["grouped_call"]()
        assert kernel.calls == [dict(grid=[(batch+3)//4],parameters=t.launch_parameters(batch,width,width,2*width,"grouped"))]
        values = np.random.default_rng(30).integers(0,1024,(batch,width),dtype=np.int32).astype(np.float16)
        values[:,0] = np.arange(batch,dtype=np.float16)
        before,expected_cache = t.fixture(values)
        oracle = dict(before,expected_cache=expected_cache)
        good = {n:(expected_cache if n == "cache" else a).copy() for n,a in before.items()}
        t.h.verify_backing(oracle,good)
        for label,array,index in (("target_K","cache",(batch-1,0,width-1)),("target_V","cache",(batch-1,1,width-1)),
                                  ("untouched_cache","cache",(1023,1,width-1)),("input_K","k",(batch-1,width-1)),
                                  ("input_V","v",(batch-1,width-1)),("indices","indices",batch-1)):
            bad = {n:a.copy() for n,a in good.items()}
            bad[array][index] = -99
            expect_failure(f"fixture_B{batch}_D{width}_{label}",lambda:t.h.verify_backing(oracle,bad),AssertionError)
        if batch == 256:
            bad = {n:a.copy() for n,a in good.items()}
            bad["cache"][[0,8]] = bad["cache"][[8,0]]
            expect_failure("fixture_wrong_mapping_rows0_8",lambda:t.h.verify_backing(oracle,bad),AssertionError)
        fixture_controls.append(dict(batch=batch,width=width,status="CPU_ORACLE_CONTROLS_PASS",raw_grouped_call=kernel.calls))
    accepted = HERE / "logs/m07-grouped-results" / t.ACCEPTED_RUN
    gate = t.accepted_gate(accepted)
    grouped = (accepted / t.r.GROUPED).read_text(encoding="utf-8-sig")
    assert t.r.review_source(grouped)["num_warps"] == 4
    with tempfile.TemporaryDirectory(prefix="m07-grouped-timing-cpu-") as tmp:
        root = Path(tmp)
        evidence = root / "accepted"
        evidence.mkdir()
        for name in json.loads((accepted/"source_hashes.json").read_text(encoding="utf-8")):
            shutil.copyfile(accepted/name,evidence/name)
        for name in ("status.json","source_hashes.json"):
            shutil.copyfile(accepted/name,evidence/name)
        for directory in accepted.glob("case-*"):
            target = evidence / directory.name
            target.mkdir()
            for name in ("status.json","submissions.json"):
                shutil.copyfile(directory/name,target/name)
        assert t.accepted_gate(evidence)["positive_and_empty_cases"] == 34
        def changed_file(name,path,payload,call,expected):
            original = path.read_bytes()
            try:
                path.write_bytes(payload)
                expect_failure(name,call,expected)
            finally:
                path.write_bytes(original)
        failed_status = json.loads((evidence/"status.json").read_text(encoding="utf-8"))
        failed_status["status"] = "FAILED"
        changed_file("gate_nonpass",evidence/"status.json",json.dumps(failed_status).encode(),lambda:t.accepted_gate(evidence),RuntimeError)
        changed_file("gate_grouped_hash",evidence/t.r.GROUPED,b"changed",lambda:t.accepted_gate(evidence),RuntimeError)
        launch = next(evidence.glob("case-*-grouped/submissions.json"))
        changed_file("gate_launch_record",launch,b'[{"grid": [999]}]',lambda:t.accepted_gate(evidence),RuntimeError)
        helpers = root / "helpers"
        helpers.mkdir()
        accepted_hashes = json.loads((accepted/"source_hashes.json").read_text(encoding="utf-8"))
        for name in t.HELPERS:
            shutil.copyfile(HERE/name,helpers/name)
        t.verify_helper_sources(helpers,accepted_hashes)
        changed_file("gate_executed_helper_changed",helpers/"29-check-kv-grouped.py",b"changed",
                     lambda:t.verify_helper_sources(helpers,accepted_hashes),RuntimeError)
        timer = root / "timer.py"
        timer.write_text(t.p.p.TIMER_SOURCE,encoding="utf-8")
        t.load_reviewed_timer(timer)
        timer.write_text(t.p.p.TIMER_SOURCE.replace("range(100)","range(99)"),encoding="utf-8")
        expect_failure("timer_99_calls",lambda:t.load_reviewed_timer(timer),RuntimeError)
        incomplete_snapshot = root / "incomplete-snapshot"
        incomplete_snapshot.mkdir()
        worker_directory = root / "failed-worker"
        expect_failure("worker_preflight_failure_saved",lambda:t.worker(root,incomplete_snapshot,worker_directory,256,128,1),FileNotFoundError)
        failed = json.loads((worker_directory/"result.json").read_text(encoding="utf-8"))
        assert failed["status"] == "FAILED" and failed["error_type"] == "FileNotFoundError"
        assert failed["correctness_checks"] == 0 and "source_hashes.json" in (worker_directory/"error.txt").read_text(encoding="utf-8")
    for name in ("30-compare-kv-grouped.py","30-grouped-timing-cpu-test.py"):
        ast.parse((HERE/name).read_text(encoding="utf-8"))
    for name in ("30-compare-kv-grouped.py","30-compare-kv-grouped.sh"):
        raw = (HERE/name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_ORCHESTRATION_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
                  source_sha256={name:t.q.sha(HERE/name) for name in
                                 ("30-compare-kv-grouped.py","30-compare-kv-grouped.sh","30-grouped-timing-cpu-test.py")},
                  plan=plan,order_positions=positions,synthetic_summary=summary,negative_controls_rejected=rejected,
                  accepted_gate=gate,fixture_controls=fixture_controls,
                  timer_control="reviewed old 100-call body accepted; changed99-call body rejected; unchanged timing helper reused",
                  failure_control="actual worker preserves FAILED result and traceback on pre-GPU snapshot failure",
                  limitation="CPU source/record/summary/timer mocks only; no new performance or GPU correctness result")
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_ORCHESTRATION_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 30-grouped-timing-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
