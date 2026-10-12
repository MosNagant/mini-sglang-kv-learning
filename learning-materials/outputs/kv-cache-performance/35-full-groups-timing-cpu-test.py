"""CPU-only controls for35's new arm, stride, reset, provenance and summary bindings."""
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
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_groups_timing_under_test",HERE/"35-compare-kv-full-groups.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def main(output):
    if output.exists():
        raise FileExistsError("REPORT_EXISTS: choose a new output")
    rejected = []
    def reject(label,call,error=ValueError):
        try:
            call()
        except error:
            rejected.append(label)
        else:
            raise AssertionError("NEGATIVE_CONTROL_NOT_REJECTED: "+label)
    plan = t.make_plan()
    assert t.CASES == tuple((b,d) for b in (5,256) for d in (64,128,192))
    assert (plan["fresh_processes"],plan["interval_averages"],len(plan["schedule"])) == (18,3240,18)
    assert {(x["round"],x["batch"],x["width"]) for x in plan["schedule"]} == {
        (round_id,b,d) for round_id in (1,2,3) for b,d in t.CASES}
    for round_id in (1,2,3):
        counts = Counter()
        for sample in range(30):
            order = t.measurement_order(sample,round_id)
            assert len(order) == 6 and set(order) == {(arm,policy) for arm in t.IMPLEMENTATIONS for policy in t.POLICIES}
            for position,item in enumerate(order):
                counts[(item,position)] += 1
        assert len(counts) == 36 and set(counts.values()) == {5}
    synthetic = []
    for item in plan["schedule"]:
        round_id,b,d = (item[k] for k in ("round","batch","width"))
        cfg = t.configuration(b,d)
        bases = dict(original_cuda=(6+2*round_id)/1000,accepted_grouped=.006,full_groups=2*round_id/1000)
        samples = {arm:{policy:[base*(2 if policy == "eager" else 1)*(1+(s-14.5)*.001) for s in range(30)]
                       for policy in t.POLICIES} for arm,base in bases.items()}
        measurements = [dict(sample=s,implementation=arm,policy=policy,ms_per_call=samples[arm][policy][s])
                        for s in range(30) for arm,policy in t.measurement_order(s,round_id)]
        launches = dict(original_cuda=dict(entry="upstream store_cache",public_calls=1),
            **{arm:[dict(grid=[(b+3)//4],parameters=t.launch_parameters(cfg))] for arm in t.IMPLEMENTATIONS[1:]})
        synthetic.append(dict(item,status="COLLECTED",config=cfg,capacity=512,calls_per_interval=100,
            correctness_checks=190,actual_launches=launches,samples_ms_per_call=samples,measurements=measurements))
    summary = t.summarize(synthetic)
    assert len(summary) == 6
    for row in summary:
        for policy in t.POLICIES:
            entry = row[policy]
            assert math.isclose(entry["speedup_original_over_full_groups"],2.5)
            assert math.isclose(entry["speedup_grouped_over_full_groups"],1.5)
            assert np.allclose(entry["paired_round_speedups_grouped_over_full_groups"],(3,1.5,1))
            assert np.allclose(entry["paired_round_speedups_original_over_full_groups"],(4,2.5,2))
            assert math.isclose(entry["full_groups"]["representative_us"],4*(2 if policy == "eager" else 1))
    reject("missing-worker",lambda:t.summarize(synthetic[:-1]))
    reject("duplicate-worker",lambda:t.summarize(synthetic+[synthetic[0]]))
    mutations = {
        "not-collected":lambda x:x[0].update(status="FAILED"),
        "wrong-case":lambda x:x[0].update(batch=16),
        "wrong-capacity":lambda x:x[0].update(capacity=1024),
        "missing-check":lambda x:x[0].update(correctness_checks=189),
        "wrong-call-count":lambda x:x[0].update(calls_per_interval=99),
        "wrong-layout":lambda x:x[0]["config"].update(layout="wrong"),
        "lost-gap":lambda x:x[0]["config"].update(gap=999),
        "wrong-grid":lambda x:x[0]["actual_launches"]["full_groups"][0].update(grid=[999]),
        "one-warp":lambda x:x[0]["actual_launches"]["accepted_grouped"][0]["parameters"].update(num_warps=1),
        "bad-block":lambda x:x[0]["actual_launches"]["full_groups"][0]["parameters"].update(BLOCK_SIZE=999),
        "short-samples":lambda x:x[0]["samples_ms_per_call"]["full_groups"]["graph"].pop(),
        "reordered":lambda x:x[0]["measurements"].reverse(),
        "raw-mismatch":lambda x:x[0]["measurements"][0].update(ms_per_call=999),
    }
    for value in (float("nan"),float("inf"),0,-1):
        mutations["invalid-sample-"+str(value)] = lambda x,value=value:x[0]["samples_ms_per_call"]["full_groups"]["graph"].__setitem__(0,value)
    for label,mutate in mutations.items():
        bad = copy.deepcopy(synthetic)
        mutate(bad)
        reject(label,lambda:t.summarize(bad))
    reject("unplanned-B1",lambda:t.validate_worker_case(1,64,1))
    reject("unplanned-round",lambda:t.validate_worker_case(5,64,4))
    tree = ast.parse((HERE/t.NEW_FILES[0]).read_text(encoding="utf-8"))
    worker = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "worker")
    raw_factory = next(n for n in ast.walk(worker) if isinstance(n,ast.FunctionDef) and n.name == "make_raw_call")
    resets = [n for n in ast.walk(worker) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name)
              and n.value.func.value.id == "pool" and n.value.func.attr == "copy_"]
    assert len(resets) == 2
    assert not any(isinstance(n,ast.Attribute) and n.attr in ("fill_","contiguous") for n in ast.walk(worker))
    class Kernel:
        def __init__(self,label): self.label,self.calls = label,[]
        def __getitem__(self,grid):
            def call(*args,**kwargs):
                self.calls.append(dict(grid=list(grid),parameters=kwargs,args=args))
                return self.label
            return call
    class Cache:
        def __init__(self,data): self.data = data.copy()
        def copy_(self,source): self.data[...] = source
    fixture_controls = []
    for b,d in t.CASES:
        cfg = t.configuration(b,d)
        before = t.v.fixture(cfg)
        params = t.launch_parameters(cfg)
        expected_strides = {64:(64,128,64),128:(192,128,128),192:(256,256,256)}[d]
        assert (params["SRC_ROW_STRIDE"],params["DST_ROW_STRIDE"],params["BLOCK_SIZE"]) == expected_strides
        good = t.c.correct_after(before)
        t.h.verify_backing(before,good)
        sentinels = [object() for _ in range(5)]
        namespace = dict(batch=b,params=params,**dict(zip(("kc","vc","indices","k","val"),sentinels)))
        exec(compile(ast.Module(body=[raw_factory],type_ignores=[]),"actual35-raw-factory","exec"),namespace)
        kernels = [Kernel(arm) for arm in t.IMPLEMENTATIONS[1:]]
        calls = [namespace["make_raw_call"](kernel) for kernel in kernels]
        assert [call() for call in calls] == list(t.IMPLEMENTATIONS[1:])
        for kernel in kernels:
            assert kernel.calls == [dict(grid=[(b+3)//4],parameters=params,args=tuple(sentinels))]
        for statement in resets:
            pool = Cache(before["cache"])
            pool.data[...] = 99
            exec(compile(ast.Module(body=[statement],type_ignores=[]),"actual35-cache-reset","exec"),
                 dict(pool=pool,initial_cache_gpu=before["cache"]))
            np.testing.assert_array_equal(pool.data,before["cache"])
        for name,index in (("k",(b-1,d-1)),("v",(b-1,d-1)),("indices",b-1)):
            bad = {key:a.copy() for key,a in good.items()}
            bad[name][index] = -99
            reject(f"B{b}-D{d}-readonly-{name}",lambda:t.h.verify_backing(before,bad),AssertionError)
        if cfg["gap"]:
            bad = {key:a.copy() for key,a in good.items()}
            bad["k"][0,d] = -99
            reject(f"B{b}-D{d}-input-gap",lambda:t.h.verify_backing(before,bad),AssertionError)
        if cfg["guard"]:
            bad = {key:a.copy() for key,a in good.items()}
            bad["cache"][0,-1,-1] = -99
            reject(f"B{b}-D{d}-cache-guard",lambda:t.h.verify_backing(before,bad),AssertionError)
        fixture_controls.append(dict(batch=b,width=d,source_stride=expected_strides[0],destination_stride=expected_strides[1],block=expected_strides[2]))
    accepted = HERE/"logs/m07-full-groups-results"/t.ACCEPTED_RUN
    gate = t.accepted_gate(accepted)
    with tempfile.TemporaryDirectory(prefix="m07-full-groups-timing-cpu-") as temporary:
        root = Path(temporary)
        project = root/"project"
        archived = project/"experiments/learning/m07-full-groups"/t.ACCEPTED_RUN
        archived.mkdir(parents=True)
        hashes = json.loads((accepted/"source_hashes.json").read_text(encoding="utf-8"))
        for name in (*hashes,"source_hashes.json","status.json"):
            shutil.copyfile(accepted/name,archived/name)
        for directory in accepted.glob("case-*"):
            destination = archived/directory.name
            destination.mkdir()
            for name in ("status.json","submissions.json"):
                shutil.copyfile(directory/name,destination/name)
        for name in (*t.q.PINNED,t.r.GROUPED,t.v.CANDIDATE):
            shutil.copyfile(accepted/name,project/"experiments"/name)
        prior = HERE/"logs/m07-grouped-six-timing-results"/t.v.ACCEPTED_TIMING_RUN
        archived_prior = project/"experiments/learning/m07-grouped-six-timing"/t.v.ACCEPTED_TIMING_RUN
        archived_prior.mkdir(parents=True)
        for name in (*json.loads((prior/"source_hashes.json").read_text(encoding="utf-8")),"source_hashes.json","status.json"):
            destination = archived_prior/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(prior/name,destination)
            if name.startswith("original_source/"):
                live = project/name.removeprefix("original_source/")
                live.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(prior/name,live)
        run = root/"snapshot"
        run.mkdir()
        t.snapshot(project,run)
        t.verify_snapshot(run,project)
        def changed_file(label,path,data,call):
            original = path.read_bytes()
            try:
                path.write_bytes(data)
                reject(label,call,RuntimeError)
            finally:
                path.write_bytes(original)
        changed_file("candidate-hash-changed",run/t.v.CANDIDATE,b"changed",lambda:t.verify_snapshot(run,project))
        changed_file("original-live-changed",project/"python/minisgl/kernel/store.py",b"changed",lambda:t.verify_snapshot(run,project))
        changed_file("34-archived-launch-changed",archived/"case-000-full_groups/submissions.json",b"[]",lambda:t.accepted_gate(archived))
        failed_state = json.loads((archived/"status.json").read_text(encoding="utf-8"))
        failed_state["status"] = "FAILED"
        changed_file("34-nonpass",archived/"status.json",json.dumps(failed_state).encode(),lambda:t.accepted_gate(archived))
        changed_file("34-frozen-helper-changed",archived/"34-check-kv-full-groups.py",b"changed",lambda:t.accepted_gate(archived))
        timer = root/"bad-timer.py"
        timer.write_text(t.p.m.p.p.TIMER_SOURCE.replace("range(100)","range(99)"),encoding="utf-8")
        reject("99-call-timer-rejected",lambda:t.load_reviewed_timer(timer),RuntimeError)
        failure = root/"failed-worker"
        reject("worker-preflight-first-error",lambda:t.worker(project,root/"missing-snapshot",failure,5,192,1),FileNotFoundError)
        failed = json.loads((failure/"result.json").read_text(encoding="utf-8"))
        assert failed["status"] == "FAILED" and failed["error_type"] == "FileNotFoundError" and failed["correctness_checks"] == 0
        assert "source_hashes.json" in (failure/"error.txt").read_text(encoding="utf-8")
    for name in (*t.NEW_FILES,"35-full-groups-timing-cpu-test.py"):
        raw = (HERE/name).read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        if name.endswith(".py"):
            ast.parse(raw.decode("utf-8"))
    assert "torch" not in sys.modules and "triton" not in sys.modules
    report = dict(status="CPU_FULL_GROUPS_TIMING_CONTROLS_PASS",GPU="NOT_RUN",WSL="NOT_RUN",
        plan=plan,negative_controls_rejected=rejected,synthetic_summary=summary,accepted34_gate=gate,
        fixture_controls=fixture_controls,raw_closures="both actual factories bind distinct mock kernels; full grouped kwargs and input identities preserved",
        reset_control="both actual pool.copy_ statements restore complete initial cache including guard=-7; no fill_/contiguous in worker",
        order_control="each arm/policy occupies each of six positions five times per process",
        source_sha256={name:t.q.sha(HERE/name) for name in (*t.NEW_FILES,"35-full-groups-timing-cpu-test.py")},
        limitation="CPU source/record/fixture/launch/reset/summary controls only; no GPU correctness or performance collection")
    output.parent.mkdir(parents=True,exist_ok=True)
    t.h.write_json(output,report)
    print(f"CPU_FULL_GROUPS_TIMING_CONTROLS_PASS negative_controls={len(rejected)} GPU_NOT_RUN REPORT={output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -B 35-full-groups-timing-cpu-test.py NEW_REPORT_PATH")
    main(Path(sys.argv[1]))
