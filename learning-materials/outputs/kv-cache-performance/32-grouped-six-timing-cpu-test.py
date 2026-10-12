"""Bounded CPU controls for six-case expansion and the full31 acceptance gate."""
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
spec = importlib.util.spec_from_file_location("grouped_six_timing_under_test",HERE / "32-compare-kv-grouped-full.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new report path")
    rejected = []
    def expect_failure(name,call,expected):
        try:
            call()
        except expected:
            rejected.append(name)
        else:
            raise AssertionError("NEGATIVE_CONTROL_NOT_REJECTED: " + name)
    def worker_ast(name):
        tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "worker")
    worker = worker_ast("32-compare-kv-grouped-full.py")
    assert ast.dump(worker,include_attributes=False) == ast.dump(worker_ast("30-compare-kv-grouped.py"),include_attributes=False)
    cases = tuple((b,d) for b in (1,16,256) for d in (64,128))
    assert t.CASES == cases
    plan = t.make_plan()
    assert (plan["fresh_processes"],plan["interval_averages"],len(plan["schedule"])) == (18,3240,18)
    assert {(x["round"],x["batch"],x["width"]) for x in plan["schedule"]} == {
        (r,b,d) for r in (1,2,3) for b,d in cases}
    assert t.make_plan() == plan
    positions = {}
    for round_id in (1,2,3):
        counts = Counter()
        expected = {(i,p) for i in t.IMPLEMENTATIONS for p in t.POLICIES}
        for sample in range(30):
            order = t.measurement_order(sample,round_id)
            assert len(order) == 6 and set(order) == expected
            for position,item in enumerate(order):
                counts[(item,position)] += 1
        assert set(counts.values()) == {5} and len(counts) == 36
        positions[round_id] = "each of six arm/policy items occupies every position five times"
    synthetic = []
    for item in plan["schedule"]:
        round_id,batch,width = (item[k] for k in ("round","batch","width"))
        t.validate_worker_case(batch,width,round_id)
        bases = dict(original_cuda=(6+2*round_id)/1000,frozen_v1=.006,grouped=2*round_id/1000)
        values = {impl:{policy:[bases[impl]*(2 if policy == "eager" else 1)*(1+(sample-14.5)*.001)
            for sample in range(30)] for policy in t.POLICIES} for impl in t.IMPLEMENTATIONS}
        records = [dict(sample=s,implementation=i,policy=p,ms_per_call=values[i][p][s])
                   for s in range(30) for i,p in t.measurement_order(s,round_id)]
        launches = dict(original_cuda=dict(entry="upstream store_cache",public_calls=1))
        for arm in ("frozen_v1","grouped"):
            launches[arm] = [dict(grid=[(batch+3)//4 if arm == "grouped" else batch],
                parameters=t.launch_parameters(batch,width,width,2*width,arm))]
        synthetic.append(dict(item,status="COLLECTED",correctness_checks=190,calls_per_interval=100,
            capacity=1024,config=dict(layout="interleaved",input_gap=0),actual_launches=launches,
            samples_ms_per_call=values,measurements=records))
    summary = t.summarize(synthetic)
    assert [(x["batch"],x["width"]) for x in summary] == list(cases)
    for row in summary:
        for policy in t.POLICIES:
            got = row[policy]
            assert math.isclose(got["speedup_original_over_grouped"],2.5)
            assert math.isclose(got["speedup_v1_over_grouped"],1.5)
            assert np.allclose(got["paired_round_speedups_v1_over_grouped"],(3,1.5,1))
            assert np.allclose(got["paired_round_speedups_original_over_grouped"],(4,2.5,2))
            assert math.isclose(got["grouped"]["representative_us"],4*(2 if policy == "eager" else 1))
    expect_failure("missing_worker",lambda:t.summarize(synthetic[:-1]),ValueError)
    expect_failure("duplicate_worker",lambda:t.summarize(synthetic+[synthetic[0]]),ValueError)
    changes = {
        "noncollected":lambda x:x[0].update(status="FAILED"),
        "unplanned_case":lambda x:x[0].update(batch=2),
        "missing_correctness":lambda x:x[0].update(correctness_checks=189),
        "wrong_interval_calls":lambda x:x[0].update(calls_per_interval=99),
        "wrong_grouped_grid":lambda x:x[0]["actual_launches"]["grouped"][0].update(grid=[999]),
        "wrong_grouped_warps":lambda x:x[0]["actual_launches"]["grouped"][0]["parameters"].update(num_warps=1),
        "wrong_grouped_batch":lambda x:x[0]["actual_launches"]["grouped"][0]["parameters"].update(BATCH=7),
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
    expect_failure("unplanned_B2_D64",lambda:t.validate_worker_case(2,64,1),ValueError)
    expect_failure("unplanned_D192",lambda:t.validate_worker_case(16,192,1),ValueError)
    expect_failure("unplanned_round",lambda:t.validate_worker_case(16,64,4),ValueError)
    # Execute only the actual raw launch expression against a CPU recorder.
    grouped_node = next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name == "grouped_call")
    class TraceKernel:
        def __init__(self): self.calls = []
        def __getitem__(self,grid):
            def call(*args,**kwargs): self.calls.append(dict(grid=list(grid),parameters=kwargs))
            return call
    fixture_controls = []
    for batch,width in cases:
        kernel = TraceKernel()
        namespace = dict(grouped_kernel=kernel,batch=batch,width=width,src=width,dst=2*width,block=width,
            grouped_api=SimpleNamespace(NUM_WARPS=4,ROWS_PER_PROGRAM=4),kc=None,vc=None,indices=None,k=None,v=None)
        exec(compile(ast.Module(body=[grouped_node],type_ignores=[]),"actual-grouped-launch","exec"),namespace)
        namespace["grouped_call"]()
        assert kernel.calls == [dict(grid=[(batch+3)//4],parameters=t.launch_parameters(batch,width,width,2*width,"grouped"))]
        values = np.random.default_rng(32).integers(0,1024,(batch,width),dtype=np.int32).astype(np.float16)
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
        if batch >= 16:
            bad = {n:a.copy() for n,a in good.items()}
            bad["cache"][[0,8]] = bad["cache"][[8,0]]
            expect_failure(f"fixture_B{batch}_D{width}_wrong_rows0_8",lambda:t.h.verify_backing(oracle,bad),AssertionError)
        fixture_controls.append(dict(batch=batch,width=width,status="CPU_ORACLE_CONTROLS_PASS",raw_grouped_call=kernel.calls))
    expect_failure("fixture_unplanned_shape",lambda:t.fixture(np.zeros((2,64),dtype=np.float16)),ValueError)
    expect_failure("fixture_wrong_dtype",lambda:t.fixture(np.zeros((16,64),dtype=np.float32)),ValueError)
    accepted = HERE / "logs/m07-grouped-full-results" / t.ACCEPTED_RUN
    gate = t.accepted_gate(accepted)  # CPU read of saved31 arrays, not a new GPU run.
    assert (gate["positive_and_empty_cases"],gate["matched_pairs"],gate["invalid_rejections"]) == (196,98,36)
    with tempfile.TemporaryDirectory(prefix="m07-grouped-six-cpu-") as tmp:
        root,evidence,helpers = Path(tmp),Path(tmp)/"accepted",Path(tmp)/"helpers"
        evidence.mkdir()
        helpers.mkdir()
        accepted_hashes = json.loads((accepted/"source_hashes.json").read_text(encoding="utf-8"))
        for name in (*accepted_hashes,"source_hashes.json","status.json"):
            target = evidence/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(accepted/name,target)
        for name in t.HELPERS:
            shutil.copyfile(HERE/name,helpers/name)
        for directory in accepted.glob("reject-*"):
            shutil.copytree(directory,evidence/directory.name)
        t.verify_accepted_sources(evidence,helpers)
        status = json.loads((evidence/"status.json").read_text(encoding="utf-8"))
        records = status["rejected"]
        t.verify_invalid_records(evidence,records)
        def changed_file(name,path,payload,call,expected=RuntimeError):
            original = path.read_bytes()
            try:
                path.write_bytes(payload)
                expect_failure(name,call,expected)
            finally:
                path.write_bytes(original)
        bad_status = dict(status,status="FAILED")
        changed_file("gate_nonpass",evidence/"status.json",json.dumps(bad_status).encode(),lambda:t.accepted_gate(evidence))
        bad_status = dict(status,cases=status["cases"][:-1])
        changed_file("gate_missing_full_case",evidence/"status.json",json.dumps(bad_status).encode(),lambda:t.accepted_gate(evidence))
        changed_file("gate_grouped_snapshot_hash",evidence/t.r.GROUPED,b"changed",lambda:t.verify_accepted_sources(evidence,helpers))
        changed_file("gate_executed31_helper_changed",helpers/"31-check-kv-grouped-full.py",b"changed",lambda:t.verify_accepted_sources(evidence,helpers))
        bad_manifest = dict(accepted_hashes)
        bad_manifest["31-check-kv-grouped-full.py"] = "0"*64
        changed_file("gate31_runner_manifest_pin",evidence/"source_hashes.json",json.dumps(bad_manifest).encode(),lambda:t.verify_accepted_sources(evidence,helpers))
        expect_failure("gate_missing_invalid",lambda:t.verify_invalid_records(evidence,records[:-1]),RuntimeError)
        for name,key,value in (("nonzero_submission","submissions",1),("invalid_runtime_error","rejection",dict(type="RuntimeError",message="launch failed"))):
            bad = copy.deepcopy(records)
            bad[0][key] = value
            expect_failure(name,lambda:t.verify_invalid_records(evidence,bad),RuntimeError)
        first = records[0]
        result_path = evidence / f"reject-{first['implementation']}-{first['kind']}" / "result.json"
        saved_result = json.loads(result_path.read_text(encoding="utf-8"))
        payload = dict(saved_result,launches=[dict(grid=[1])])
        changed_file("invalid_launch_record",result_path,json.dumps(payload).encode(),lambda:t.verify_invalid_records(evidence,records))
        payload = dict(saved_result,values_after=[])
        changed_file("invalid_argument_view_changed",result_path,json.dumps(payload).encode(),lambda:t.verify_invalid_records(evidence,records))
        # Exercise unchanged worker's first-error file path before importing GPU libraries.
        snapshot = root / "missing-snapshot"
        snapshot.mkdir()
        directory = root / "failed-worker"
        expect_failure("worker_preflight_failure_saved",lambda:t.worker(root,snapshot,directory,16,128,1),FileNotFoundError)
        failed = json.loads((directory/"result.json").read_text(encoding="utf-8"))
        assert failed["status"] == "FAILED" and failed["error_type"] == "FileNotFoundError"
        assert failed["correctness_checks"] == 0 and "source_hashes.json" in (directory/"error.txt").read_text(encoding="utf-8")
    for name in ("32-compare-kv-grouped-full.py","32-grouped-six-timing-cpu-test.py"):
        ast.parse((HERE/name).read_text(encoding="utf-8"))
    for name in t.NEW_FILES:
        raw = (HERE/name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_SIX_CASE_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
        source_sha256={name:t.q.sha(HERE/name) for name in (*t.NEW_FILES,"32-grouped-six-timing-cpu-test.py")},
        worker_equivalence="32.worker AST equals accepted30.worker AST including all check/capture/timing/failure paths; local six-case globals used, no module monkeypatch",
        plan=plan,order_positions=positions,synthetic_summary=summary,negative_controls_rejected=rejected,
        accepted_gate=gate,fixture_controls=fixture_controls,
        failure_control="actual worker retains FAILED result and first traceback before GPU import",
        limitation="CPU source/record/fixture/summary controls and saved31 evidence reread; no new GPU correctness or timing result")
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_SIX_CASE_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 32-grouped-six-timing-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
