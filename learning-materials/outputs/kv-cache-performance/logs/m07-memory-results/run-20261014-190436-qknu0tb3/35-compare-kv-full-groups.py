"""M07 same-session timing: original / accepted grouped / full-group branch.

Six exact34 configurations, three fresh processes each, 30 intervals x100 calls
per arm/policy. Graph primary, Eager separate; full backing resets/checks outside
timing. GPU collection is user-owned; --dry-run imports no GPU runtime.
"""
import argparse
import importlib.util
import inspect
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("full_groups_correctness_helpers",HERE / "34-check-kv-full-groups.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
p,r,q,c,h,COMMIT = v.t,v.r,v.q,v.c,v.h,v.COMMIT
load_reviewed_timer,measure_graph,telemetry,export_run = p.load_reviewed_timer,p.measure_graph,p.telemetry,p.export_run
IMPLEMENTATIONS = ("original_cuda","accepted_grouped","full_groups")
POLICIES = ("graph","eager")
CASES = tuple((b,d) for b in (5,256) for d in (64,128,192))
ACCEPTED_RUN = "run-20261010-213028-jgwjfp1z"
FULL_GROUPS_HASH = "6a53a5304566420c3ee0163f0b8c6ff157b4f9b16f99a83be20364b415a81156"
PINS = {
    "34-check-kv-full-groups.py":"1009ba8542c2a78047e822165fdab26b9a6c11ad89c794324e5d1450614dbe60",
    "34-check-kv-full-groups.sh":"62b3e241993cbd8f21994cf8451fdb6514b80fa8b2ebefdacf1c0759f76fd2b1",
    "34-full-groups-cpu-test.py":"f265d5514145a8543d2c5f33560299cbda0a4bffbfdb3fc8b331882252860e32",
}
HELPERS = tuple(v.HELPERS)+tuple(v.NEW_FILES)
NEW_FILES = ("35-compare-kv-full-groups.py","35-compare-kv-full-groups.sh")
REQUIRED_SOURCES = HELPERS+NEW_FILES+tuple(q.PINNED)+(r.GROUPED,v.CANDIDATE,"learner-timing.py",
    "hypothesis-card.md","accepted34-gate.json","prior34-status.json","prior34-source_hashes.json",
    "original_source/python/minisgl/kernel/store.py","original_source/python/minisgl/kernel/csrc/jit/store.cu")


def configuration(batch,width):
    if (batch,width) not in CASES:
        raise ValueError("UNPLANNED_FULL_GROUPS_TIMING_CASE")
    return next(cfg for cfg in v.configurations() if (cfg["batch"],cfg["width"]) == (batch,width))


def validate_worker_case(batch,width,round_id):
    configuration(batch,width)
    if round_id not in (1,2,3):
        raise ValueError("UNPLANNED_FULL_GROUPS_TIMING_ROUND")


def launch_parameters(cfg):
    d = cfg["width"]
    return dict(BATCH=cfg["batch"],WIDTH=d,SRC_ROW_STRIDE=d+cfg["gap"],
        DST_ROW_STRIDE=(d+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1),
        BLOCK_SIZE=1 << (d-1).bit_length(),ROWS_PER_PROGRAM=4,num_warps=4)


def measurement_order(sample,round_id):
    ring = [(impl,policy) for policy in POLICIES for impl in IMPLEMENTATIONS]
    offset = (sample+round_id-1) % len(ring)
    order = ring[offset:]+ring[:offset]
    return list(reversed(order)) if (sample//6+round_id-1) % 2 else order


def make_plan():
    plan = p.make_plan()
    for key in ("layout","input_gap","num_warps","rows_per_program","grouped_sha256"):
        plan.pop(key,None)
    rng,schedule = random.Random(107),[]
    for round_id in (1,2,3):
        ordered = list(CASES)
        rng.shuffle(ordered)
        schedule.extend(dict(round=round_id,batch=b,width=d) for b,d in ordered)
    plan.update(cases=CASES,configurations=[configuration(b,d) for b,d in CASES],schedule=schedule,
        capacity=512,input_seed=20261011,indices="(37*i+11)%512; exact34 unique int32 mapping",
        fresh_processes=18,interval_averages=3240,num_warps=dict(accepted_grouped=4,full_groups=4),
        rows_per_program=dict(accepted_grouped=4,full_groups=4),accepted_correctness_run=ACCEPTED_RUN,
        grouped_sha256=p.GROUPED_HASH,full_groups_sha256=FULL_GROUPS_HASH,
        timed_entries=dict(original_cuda="upstream store_cache Python/JIT entry",
            accepted_grouped="trusted raw accepted grouped launch; prevalidated metadata and indices",
            full_groups="trusted raw full-group branch launch; same four rows/program and four warps"),
        correctness="each arm first call and reset Graph replay checked against independent34 full backing; reset copies original cache including -7 guards",
        observation="same-session ratios only; tail groups are an unchanged-path control; no automatic speedup acceptance",
        scope="B5/256 x three exact34 layout/stride cases at C512; not a full model, DRAM, occupancy or AMD measurement")
    return plan


def summarize(results):
    expected = {(round_id,b,d) for round_id in (1,2,3) for b,d in CASES}
    keyed = {}
    for result in results:
        key = (result["round"],result["batch"],result["width"])
        if key not in expected or key in keyed or result.get("status") != "COLLECTED":
            raise ValueError("RESULT_ID_OR_STATUS_MISMATCH")
        cfg = configuration(result["batch"],result["width"])
        if result.get("config") != cfg or result.get("capacity") != 512 or result.get("calls_per_interval") != 100:
            raise ValueError("TIMING_POLICY_MISMATCH")
        if result.get("correctness_checks") != 190:
            raise ValueError("INCOMPLETE_ACTUAL_INPUT_CORRECTNESS")
        launches = result.get("actual_launches",{})
        if launches.get("original_cuda") != dict(entry="upstream store_cache",public_calls=1):
            raise ValueError("ORIGINAL_ENTRY_RECORD_MISMATCH")
        for arm in IMPLEMENTATIONS[1:]:
            if launches.get(arm) != [dict(grid=[(cfg["batch"]+3)//4],parameters=launch_parameters(cfg))]:
                raise ValueError("ACTUAL_TRUSTED_LAUNCH_MISMATCH")
        records = result["measurements"]
        expected_order = [(s,i,policy) for s in range(30) for i,policy in measurement_order(s,result["round"])]
        if [(x["sample"],x["implementation"],x["policy"]) for x in records] != expected_order:
            raise ValueError("MEASUREMENT_ORDER_MISMATCH")
        for arm in IMPLEMENTATIONS:
            for policy in POLICIES:
                values = result["samples_ms_per_call"][arm][policy]
                if len(values) != 30 or any(not math.isfinite(x) or x <= 0 for x in values):
                    raise ValueError("INVALID_INTERVAL_SAMPLES")
                if [x["ms_per_call"] for x in records if x["implementation"] == arm and x["policy"] == policy] != values:
                    raise ValueError("RAW_SAMPLE_MISMATCH")
        keyed[key] = result
    if set(keyed) != expected:
        raise ValueError("INCOMPLETE_18_PROCESS_MATRIX")
    summary = []
    for b,d in CASES:
        row = dict(batch=b,width=d,config=configuration(b,d))
        for policy in POLICIES:
            entry = {}
            for arm in IMPLEMENTATIONS:
                medians = [statistics.median(keyed[(round_id,b,d)]["samples_ms_per_call"][arm][policy])*1000 for round_id in (1,2,3)]
                representative = statistics.median(medians)
                entry[arm] = dict(round_medians_us=medians,representative_us=representative,
                                  round_span_pct=100*(max(medians)-min(medians))/representative)
            candidate = entry["full_groups"]
            for arm,label in (("original_cuda","original"),("accepted_grouped","grouped")):
                base = entry[arm]
                entry["speedup_"+label+"_over_full_groups"] = base["representative_us"]/candidate["representative_us"]
                entry["time_reduction_pct_vs_"+label] = 100*(1-candidate["representative_us"]/base["representative_us"])
                entry["paired_round_speedups_"+label+"_over_full_groups"] = [a/b for a,b in zip(base["round_medians_us"],candidate["round_medians_us"])]
            row[policy] = entry
        summary.append(row)
    return summary


def verify_core_sources(directory):
    grouped = v.verify_baseline(directory)
    if q.sha(directory/v.CANDIDATE) != FULL_GROUPS_HASH:
        raise RuntimeError("ACCEPTED_FULL_GROUPS_SOURCE_CHANGED")
    v.review_source((directory/v.CANDIDATE).read_text(encoding="utf-8-sig"),grouped)
    for name,expected in PINS.items():
        if q.sha(directory/name) != expected:
            raise RuntimeError("ACCEPTED34_HELPER_CHANGED: "+name)


def accepted_gate(directory):
    state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    if state.get("status") != "PASS" or state.get("commit") != COMMIT or state.get("plan") != v.make_plan():
        raise RuntimeError("ACCEPTED34_PASS_REQUIRED")
    v.verify_snapshot(directory)
    verify_core_sources(directory)
    hashes = json.loads((directory/"source_hashes.json").read_text(encoding="utf-8"))
    for name in HELPERS:
        if q.sha(HERE/name) != hashes.get(name):
            raise RuntimeError("ACCEPTED34_EXECUTED_HELPER_CHANGED: "+name)
    configs = v.configurations()
    if state.get("configurations") != configs or len(state["cases"]) != 46 or len(state["pairs"]) != 23:
        raise RuntimeError("ACCEPTED34_MATRIX_INCOMPLETE")
    for index,cfg in enumerate(configs):
        for offset,arm in enumerate(v.ARMS):
            record = state["cases"][2*index+offset]
            expected = dict(status="PASS" if cfg["batch"] else "EMPTY_WRAPPER_PASS",implementation=arm,
                            config=cfg,submissions=int(cfg["batch"] > 0))
            directory_case = directory/f"case-{index:03d}-{arm}"
            if record != expected or json.loads((directory_case/"status.json").read_text(encoding="utf-8")) != record:
                raise RuntimeError("ACCEPTED34_CASE_CHANGED")
            launches = json.loads((directory_case/"submissions.json").read_text(encoding="utf-8"))
            expected_launches = [dict(grid=[(cfg["batch"]+3)//4],parameters=launch_parameters(cfg))] if cfg["batch"] else []
            if launches != expected_launches:
                raise RuntimeError("ACCEPTED34_LAUNCH_CHANGED")
        if state["pairs"][index] != dict(config=cfg,status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"):
            raise RuntimeError("ACCEPTED34_PAIR_CHANGED")
    return dict(status="ACCEPTED34_SOURCE_AND_CORRECTNESS_RECORD_GATE_PASS",source_run=str(directory),
        status_sha256=q.sha(directory/"status.json"),source_hashes_sha256=q.sha(directory/"source_hashes.json"),
        full_groups_sha256=FULL_GROUPS_HASH,grouped_sha256=p.GROUPED_HASH,
        cases=46,nonempty_submissions=42,empty_wrapper_returns=4,pairs=23,
        scope="accepted34 sources/status/launch records checked; all three arms' actual timing inputs checked again per worker")


def verify_snapshot(run,project):
    hashes = json.loads((run/"source_hashes.json").read_text(encoding="utf-8"))
    if any(name not in hashes for name in REQUIRED_SOURCES):
        raise RuntimeError("INCOMPLETE_FULL_GROUPS_TIMING_SNAPSHOT")
    p.m.p.p.verify_snapshot(run,project)  # Reviewed25 manifest and live-original source comparison.
    verify_core_sources(run)
    load_reviewed_timer(run/"learner-timing.py")


def snapshot(project,run):
    for name in (*HELPERS,*NEW_FILES):
        shutil.copyfile(HERE/name,run/name)
    for name in (*q.PINNED,r.GROUPED,v.CANDIDATE):
        shutil.copyfile(project/"experiments"/name,run/name)
    h.write_json(run/"input_source_hashes.json",{x.name:q.sha(x) for x in run.iterdir() if x.is_file()})
    verify_core_sources(run)
    accepted = project/"experiments/learning/m07-full-groups"/ACCEPTED_RUN
    h.write_json(run/"accepted34-gate.json",accepted_gate(accepted))
    for name in ("status.json","source_hashes.json"):
        shutil.copyfile(accepted/name,run/("prior34-"+name))
    prior = project/"experiments/learning/m07-grouped-six-timing"/v.ACCEPTED_TIMING_RUN
    p.verify_snapshot(prior,project)
    prior_state = json.loads((prior/"status.json").read_text(encoding="utf-8"))
    if prior_state.get("status") != "COLLECTION_COMPLETE" or prior_state.get("commit") != COMMIT:
        raise RuntimeError("ACCEPTED32_ORIGINAL_SOURCE_REQUIRED")
    prior_hashes = json.loads((prior/"source_hashes.json").read_text(encoding="utf-8"))
    for name in prior_hashes:
        if name.startswith("original_source/") or name == "learner-timing.py":
            destination = run/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(prior/name,destination)
    h.write_json(run/"original32-source-gate.json",dict(run=v.ACCEPTED_TIMING_RUN,
        source_hashes_sha256=q.sha(prior/"source_hashes.json"),status_sha256=q.sha(prior/"status.json")))
    shutil.copyfile(HERE/"lessons/m07-10-full-groups-timing.md",run/"hypothesis-card.md")
    h.write_json(run/"source_hashes.json",{x.relative_to(run).as_posix():q.sha(x) for x in run.rglob("*") if x.is_file()})
    verify_snapshot(run,project)


def worker(project,run,directory,batch,width,round_id):
    directory.mkdir(parents=True)
    result = dict(status="STARTED",batch=batch,width=width,round=round_id,capacity=512,calls_per_interval=100,
        seed=20261011,samples_ms_per_call={arm:{policy:[] for policy in POLICIES} for arm in IMPLEMENTATIONS},
        measurements=[],correctness_checks=0)
    h.write_json(directory/"result.json",result)
    try:
        validate_worker_case(batch,width,round_id)
        cfg = configuration(batch,width)
        result["config"] = cfg
        verify_snapshot(run,project)
        eager_timer = load_reviewed_timer(run/"learner-timing.py")
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__,torch.version.cuda,triton.__version__) != ("2.9.1+cu128","12.8","3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project/"python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT")
        support = h.load_module(run/"kv_contract_support.py","kv_contract_support")
        apis = {arm:h.load_module(run/name,"timing35_"+arm) for arm,name in
                (("accepted_grouped",r.GROUPED),("full_groups",v.CANDIDATE))}
        if any((api.NUM_WARPS,api.ROWS_PER_PROGRAM) != (4,4) for api in apis.values()):
            raise RuntimeError("TRUSTED_GROUP_PARAMETERS_CHANGED")
        kernels = {arm:api.store_kv_grouped for arm,api in apis.items()}
        result.update(commit=COMMIT,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,
            gpu=torch.cuda.get_device_name(0),device_properties=str(torch.cuda.get_device_properties(0)),telemetry_before=telemetry())
        before = v.fixture(cfg)
        np.savez_compressed(directory/"before.npz",**before)
        buffers,inputs = c.gpu_packet(torch,cfg,before)
        kc,vc,indices,k,val = inputs
        pool = buffers["cache"]
        params = launch_parameters(cfg)
        b,d,src,dst,block = support.validate_metadata(*inputs)
        if (b,d,src,dst,block) != (batch,width,params["SRC_ROW_STRIDE"],params["DST_ROW_STRIDE"],params["BLOCK_SIZE"]):
            raise RuntimeError("FIXED_BENCHMARK_LAYOUT_MISMATCH")
        support.validate_index_values(indices,b,512)  # Public CPU/index checks stay outside timing/capture.
        h.write_json(directory/"tensors.json",[p.w.tensor_info(x) for x in inputs])
        initial_cache_gpu = torch.from_numpy(before["cache"].copy()).cuda()
        expected_gpu = {name:torch.from_numpy(before["expected_cache" if name == "cache" else name].copy()).cuda() for name in buffers}
        def save_arrays(stage):
            np.savez_compressed(directory/(stage+".npz"),**{name:x.cpu().numpy().copy() for name,x in buffers.items()})
        def check(stage,save=False):
            try:
                for name in buffers:
                    if not torch.equal(buffers[name],expected_gpu[name]):
                        raise AssertionError(stage+": "+name+" mismatch")
                result["correctness_checks"] += 1
            except Exception:
                try:
                    save_arrays("failed-"+stage)
                except Exception as error:
                    result.setdefault("failure_capture_errors",[]).append(dict(stage=stage,error=str(error)))
                raise
            if save:
                save_arrays(stage)
        def original_call():
            store_cache(kc,vc,indices,k,val)
        def make_raw_call(kernel):
            def call():
                return kernel[((batch+3)//4,)](kc,vc,indices,k,val,**params)
            return call
        calls = dict(original_cuda=original_call,**{arm:make_raw_call(kernel) for arm,kernel in kernels.items()})
        graphs = {}
        offset = (round_id-1) % len(IMPLEMENTATIONS)
        setup_order = list(IMPLEMENTATIONS[offset:]+IMPLEMENTATIONS[:offset])
        result.update(setup_order=setup_order,actual_launches={})
        for arm in setup_order:
            call = calls[arm]
            pool.copy_(initial_cache_gpu)  # Restore all guard bytes too; do not fill with -1.
            if arm == "original_cuda":
                call()
                result["actual_launches"][arm] = dict(entry="upstream store_cache",public_calls=1)
            else:
                probe = r.GroupedProbe(kernels[arm])
                probe.reset(True)
                probe[((batch+3)//4,)](*inputs,**params)
                result["actual_launches"][arm] = probe.attempts
            torch.cuda.synchronize()
            check(arm+"-initial",save=True)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(20):
                    call()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=side):
                for _ in range(100):
                    call()
            pool.copy_(initial_cache_gpu)
            graph.replay()
            torch.cuda.synchronize()
            check(arm+"-graph-replay",save=True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            check(arm+"-warmup")
            graphs[arm] = graph
        start,end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        print(f"PAIRED_READY round={round_id} B={batch} D={width}: three initial and three reset-Graph checks passed",flush=True)
        for sample in range(30):
            for arm,policy in measurement_order(sample,round_id):
                ms = float(eager_timer(calls[arm],start,end) if policy == "eager" else measure_graph(graphs[arm],start,end))
                if not math.isfinite(ms) or ms <= 0:
                    raise ValueError(f"INVALID_TIMING {arm}/{policy}: {ms}")
                result["samples_ms_per_call"][arm][policy].append(ms)
                result["measurements"].append(dict(sample=sample,implementation=arm,policy=policy,ms_per_call=ms))
                check(f"sample-{sample}-{arm}-{policy}")
            h.write_json(directory/"result.json",result)
        check("final",save=True)
        result["telemetry_after"] = telemetry()
        verify_snapshot(run,project)
        result.update(status="COLLECTED",policy=make_plan(),scope="actual inputs passed; timing collection requires review")
    except BaseException as error:
        result.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(directory/"result.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path)
    parser.add_argument("--shared-results",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--worker-dir",type=Path)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--batch",type=int,choices=(5,256))
    parser.add_argument("--width",type=int,choices=(64,128,192))
    parser.add_argument("--round",type=int,choices=(1,2,3))
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    if args.dry_run:
        print(json.dumps(make_plan(),indent=2))
        return
    if args.project is None:
        parser.error("--project required")
    project = args.project.resolve()
    if args.worker_dir:
        if None in (args.snapshot,args.batch,args.width,args.round):
            parser.error("worker requires snapshot/batch/width/round")
        worker(project,args.snapshot,args.worker_dir,args.batch,args.width,args.round)
        return
    if args.shared_results is None:
        parser.error("--shared-results required")
    def git(*words):
        return subprocess.check_output(["git","-C",str(project),*words],text=True).strip()
    parent = project/"experiments/learning/m07-full-groups-timing"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 full-group branch timing",commit=COMMIT,plan=make_plan(),results=[],
        ownership="learner supplied accepted constexpr condition; coach supplied branch structure, prior grouped expressions and timing harness")
    print(f"RUN_DIRECTORY={run}\nSCOPE=B5/256 x three exact34 configurations; Graph primary",flush=True)
    try:
        if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
            raise RuntimeError("PINNED_TRACKED_SOURCE_CHANGED")
        snapshot(project,run)
        h.write_json(run/"plan.json",state["plan"])
        environment = None
        for item in state["plan"]["schedule"]:
            round_id,b,d = (item[k] for k in ("round","batch","width"))
            directory = run/f"round-{round_id}-B{b}-D{d}"
            console = run/f"round-{round_id}-B{b}-D{d}.log"
            command = [sys.executable,"-B",str(run/NEW_FILES[0]),"--project",str(project),"--snapshot",str(run),
                "--worker-dir",str(directory),"--batch",str(b),"--width",str(d),"--round",str(round_id)]
            state.update(status="RUNNING",current=item)
            h.write_json(run/"status.json",state)
            h.write_json(run/(console.stem+"-command.json"),command)
            print(f"RUN round={round_id}/3 B={b} D={d} three implementations",flush=True)
            with console.open("x",encoding="utf-8") as log:
                completed = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                print(console.read_text(encoding="utf-8"),flush=True)
                raise RuntimeError("FULL_GROUPS_TIMING_WORKER_FAILED: "+str(console))
            result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
            if result["status"] != "COLLECTED" or any(result[k] != value for k,value in item.items()):
                raise ValueError("WORKER_RESULT_MISMATCH")
            identity = tuple(result[k] for k in ("commit","torch","cuda","triton","gpu","device_properties"))
            if environment is None:
                environment = identity
            if identity != environment:
                raise RuntimeError("ENVIRONMENT_CHANGED_BETWEEN_WORKERS")
            state["results"].append(result)
            h.write_json(run/"partial-results.json",state["results"])
            for policy in POLICIES:
                original,grouped,full = [statistics.median(result["samples_ms_per_call"][arm][policy])*1000 for arm in IMPLEMENTATIONS]
                print(f"COLLECTED {policy} original={original:.6f} us grouped={grouped:.6f} us full_groups={full:.6f} us "
                      f"grouped/full_groups={grouped/full:.4f}x original/full_groups={original/full:.4f}x",flush=True)
        if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
            raise RuntimeError("TRACKED_SOURCE_CHANGED_DURING_RUN")
        summary = summarize(state["results"])
        state.update(status="COLLECTION_COMPLETE",current=None,summary=summary)
        h.write_json(run/"summary.json",dict(status=state["status"],plan=state["plan"],summary=summary))
        for row in summary:
            entry = row["graph"]
            print(f"GRAPH_SUMMARY B={row['batch']} D={row['width']} "
                f"original={entry['original_cuda']['representative_us']:.6f} us/call "
                f"grouped={entry['accepted_grouped']['representative_us']:.6f} us/call "
                f"full_groups={entry['full_groups']['representative_us']:.6f} us/call "
                f"grouped/full_groups={entry['speedup_grouped_over_full_groups']:.4f}x "
                f"original/full_groups={entry['speedup_original_over_full_groups']:.4f}x",flush=True)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run/"status.json",state)
        export_run(run,args.shared_results,state["status"] == "COLLECTION_COMPLETE")
    print("FULL_GROUPS_TIMING_COLLECTION_COMPLETE: 18 processes; 3240 interval averages; review every paired round before a performance claim")
    print("LIMIT: six scoped reused-address cases; Graph/Eager separate; no automatic speedup acceptance or model/AMD claim")


if __name__ == "__main__":
    main()
