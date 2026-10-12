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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("warps_timing_under_test", HERE / "27-compare-kv-warps.py")
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
        bases = dict(original_cuda=(6+2*r)/1000, frozen_v1=.006, candidate=2*r/1000)
        values = {impl:{policy:[bases[impl]*(2 if policy == "eager" else 1)*(1+(sample-14.5)*.001)
                               for sample in range(30)] for policy in t.POLICIES} for impl in t.IMPLEMENTATIONS}
        records = [dict(sample=s,implementation=i,policy=p,ms_per_call=values[i][p][s])
                   for s in range(30) for i,p in t.measurement_order(s,r)]
        synthetic.append(dict(item,status="COLLECTED",correctness_checks=190,
                              samples_ms_per_call=values,measurements=records))
    summary = t.summarize(synthetic)
    for row in summary:
        for policy in t.POLICIES:
            got = row[policy]
            assert math.isclose(got["speedup_original_over_candidate"],2.5)
            assert math.isclose(got["speedup_v1_over_candidate"],1.5)
            for observed, expected in zip(got["paired_round_speedups_v1_over_candidate"],(3,1.5,1)):
                assert math.isclose(observed,expected)
            for observed, expected in zip(got["paired_round_speedups_original_over_candidate"],(4,2.5,2)):
                assert math.isclose(observed,expected)
            assert math.isclose(got["candidate"]["representative_us"],4*(2 if policy == "eager" else 1))
    expect_failure("missing_worker",lambda:t.summarize(synthetic[:-1]),ValueError)
    expect_failure("duplicate_worker",lambda:t.summarize(synthetic+[synthetic[0]]),ValueError)
    changes = {
        "noncollected":lambda x:x[0].update(status="FAILED"),
        "unplanned_case":lambda x:x[0].update(batch=16),
        "missing_correctness":lambda x:x[0].update(correctness_checks=189),
        "short_samples":lambda x:x[0]["samples_ms_per_call"]["candidate"]["graph"].pop(),
        "swapped_order":lambda x:x[0]["measurements"].reverse(),
        "raw_sample_mismatch":lambda x:x[0]["measurements"][0].update(ms_per_call=999),
    }
    for value in (float("nan"),float("inf"),0,-1):
        changes["invalid_value_"+str(value)] = lambda x,v=value:x[0]["samples_ms_per_call"]["candidate"]["graph"].__setitem__(0,v)
    for name,change in changes.items():
        bad = copy.deepcopy(synthetic)
        change(bad)
        expect_failure(name,lambda:t.summarize(bad),ValueError)
    expect_failure("unplanned_B1_D128",lambda:t.validate_worker_case(1,128,1),ValueError)
    expect_failure("unplanned_round",lambda:t.validate_worker_case(1,64,4),ValueError)
    # Exercise the actual nested failure handler without a fake CUDA framework.
    tree = ast.parse((HERE/"27-compare-kv-warps.py").read_text(encoding="utf-8"))
    worker = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "worker")
    check_node = next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name == "check")
    class FailedCuda:
        @staticmethod
        def equal(*args): raise ValueError("first CUDA failure")
    def failed_capture(stage): raise OSError("secondary capture failure")
    capture_result = {}
    namespace = dict(torch=FailedCuda,buffers={"cache":1},expected_gpu={"cache":1},
                     result=capture_result,save_arrays=failed_capture)
    exec(compile(ast.Module(body=[check_node],type_ignores=[]),"actual-check-function","exec"),namespace)
    try:
        namespace["check"]("controlled-failure")
    except ValueError as error:
        assert str(error) == "first CUDA failure"
    else:
        raise AssertionError("CHECK_PRIMARY_EXCEPTION_LOST")
    assert capture_result["failure_capture_errors"][0]["error"] == "secondary capture failure"
    accepted = HERE / "logs/m07-warps-results" / t.ACCEPTED_RUN
    gate = t.accepted_gate(accepted)
    candidate = (accepted / t.q.CANDIDATE).read_text(encoding="utf-8-sig")
    assert t.q.candidate_review(candidate)["num_warps"] == 1
    with tempfile.TemporaryDirectory(prefix="m07-timing-cpu-") as tmp:
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
        assert t.accepted_gate(evidence)["positive_cases"] == 196
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
        changed_file("gate_candidate_hash",evidence/t.q.CANDIDATE,b"changed",lambda:t.accepted_gate(evidence),RuntimeError)
        launch = next(evidence.glob("case-*-candidate/submissions.json"))
        changed_file("gate_launch_record",launch,b'[{"grid": [999]}]',lambda:t.accepted_gate(evidence),RuntimeError)
        timer = root / "timer.py"
        timer.write_text("raise RuntimeError('top-level must not execute')\n"+t.p.TIMER_SOURCE,encoding="utf-8")
        eager_timer = t.load_reviewed_timer(timer)
        class Event:
            def __init__(self): self.records = self.syncs = 0
            def record(self): self.records += 1
            def synchronize(self): self.syncs += 1
            def elapsed_time(self,end): return 100.0
        start,end = Event(),Event()
        calls = []
        assert eager_timer(lambda:calls.append(1),start,end) == 1 and len(calls) == 100
        assert start.records == end.records == end.syncs == 1
        timer.write_text(t.p.TIMER_SOURCE.replace("range(100)","range(99)"),encoding="utf-8")
        expect_failure("timer_99_calls",lambda:t.load_reviewed_timer(timer),RuntimeError)
        graph_calls = []
        class Graph:
            def replay(self): graph_calls.append(1)
        assert t.measure_graph(Graph(),Event(),Event()) == 1 and graph_calls == [1]
        run = root / "run"
        run.mkdir()
        (run/"status.json").write_text('{"status":"FAILED"}',encoding="utf-8")
        blocker = root / "export-is-file"
        blocker.write_text("keep",encoding="utf-8")
        try:
            try:
                raise ValueError("first failure")
            finally:
                t.export_run(run,blocker,False)
        except ValueError as error:
            assert str(error) == "first failure"
        else:
            raise AssertionError("PRIMARY_EXCEPTION_LOST")
        expect_failure("successful_run_export_failure_visible",lambda:t.export_run(run,blocker,True),OSError)
        export = root / "export"
        t.export_run(run,export,False)
        original = (export/run.name/"status.json").read_bytes()
        (run/"status.json").write_text("new",encoding="utf-8")
        t.export_run(run,export,False)
        assert (export/run.name/"status.json").read_bytes() == original
    for name in ("27-compare-kv-warps.py","27-warps-timing-cpu-test.py"):
        ast.parse((HERE/name).read_text(encoding="utf-8"))
    for name in ("27-compare-kv-warps.py","27-compare-kv-warps.sh"):
        raw = (HERE/name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_ORCHESTRATION_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
                  source_sha256={name:t.q.sha(HERE/name) for name in
                                 ("27-compare-kv-warps.py","27-compare-kv-warps.sh","27-warps-timing-cpu-test.py")},
                  plan=plan,order_positions=positions,synthetic_summary=summary,negative_controls_rejected=rejected,
                  accepted_gate=gate,event_timer="exactly 100 calls; single elapsed interval divided by 100",
                  graph_timer="one replay; elapsed interval divided by 100; graph construction requires GPU validation",
                  export_controls="primary failure preserved on mkdir failure; successful-run export failure visible; existing export never overwritten",
                  failed_array_capture="actual check function preserves first CUDA/check exception if diagnostic array capture also fails",
                  limitation="CPU source/record/summary/timer mocks only; no new performance or GPU correctness result")
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_ORCHESTRATION_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 27-warps-timing-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
