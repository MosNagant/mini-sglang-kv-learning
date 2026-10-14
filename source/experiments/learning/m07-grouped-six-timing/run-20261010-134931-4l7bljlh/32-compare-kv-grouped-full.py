"""M07 full six-case grouped efficiency after accepted31 full correctness.

The audited30 worker is copied unchanged; policy helpers are local to this
module. No old module globals, kernels, scripts or exploration data are edited.
18 fresh workers, 3240 interval averages; Graph primary, Eager separate.
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
spec = importlib.util.spec_from_file_location("six_case_full_correctness_helpers",HERE / "31-check-kv-grouped-full.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
m,r,q,c,h,w,COMMIT = f.m,f.r,f.q,f.c,f.h,f.w,f.COMMIT
load_reviewed_timer,measure_graph,telemetry,export_run = m.load_reviewed_timer,m.measure_graph,m.telemetry,m.export_run
measurement_order,launch_parameters = m.measurement_order,m.launch_parameters
IMPLEMENTATIONS = m.IMPLEMENTATIONS
POLICIES = m.POLICIES
CASES = tuple((batch,width) for batch in (1,16,256) for width in (64,128))
ACCEPTED_RUN = "run-20261010-133620-jeh8_est"
GROUPED_HASH = m.GROUPED_HASH
PINNED_31 = {
    "31-check-kv-grouped-full.py":"c966eee8eccb67488dd1c83411d641abee008eb0ef0458c769c3fb50d0b732eb",
    "31-check-kv-grouped-full.sh":"81c9a9612137095fd6035af0eb628823cdb5da3a3a55a52ef494aa1f2dbaff4d",
}
HELPERS = tuple(f.HELPERS) + tuple(f.NEW_FILES)
NEW_FILES = ("32-compare-kv-grouped-full.py","32-compare-kv-grouped-full.sh")
REQUIRED_SOURCES = HELPERS + NEW_FILES + tuple(q.PINNED) + (r.GROUPED,"learner-timing.py",
    "hypothesis-card.md","accepted_full_grouped_gate.json","original_source/python/minisgl/kernel/store.py",
    "original_source/python/minisgl/kernel/csrc/jit/store.cu")


def validate_worker_case(batch,width,round_id):
    if (batch,width) not in CASES or round_id not in (1,2,3):
        raise ValueError("UNPLANNED_SIX_CASE_WORKER")


def make_plan():
    plan = m.make_plan()
    rng,schedule = random.Random(plan["order_seed"]),[]
    for round_id in (1,2,3):
        ordered = list(CASES)
        rng.shuffle(ordered)
        schedule.extend(dict(round=round_id,batch=b,width=d) for b,d in ordered)
    plan.update(cases=CASES,schedule=schedule,fresh_processes=18,interval_averages=3240,
        accepted_correctness_run=ACCEPTED_RUN,
        observation="full six-case run;30 exploration remains separate; all ratios use the current same-session workers",
        scope="six-case NVIDIA FP16 C1024 microbenchmark; not arbitrary-layout, model, DRAM or AMD measurement")
    return plan


def fixture(k_values):
    batch,width = k_values.shape
    if (batch,width) not in CASES or k_values.dtype != np.float16:
        raise ValueError("TIMING_FIXTURE_SHAPE_OR_DTYPE")
    before = dict(cache=np.full((1024,2,width),-1,dtype=np.float16),k=k_values.copy(),
                  v=k_values+np.float16(1024),indices=np.arange(batch,dtype=np.int32))
    expected = before["cache"].copy()
    for row,position in enumerate(before["indices"]):
        expected[position,0] = before["k"][row]
        expected[position,1] = before["v"][row]
    return before,expected


def summarize(results):
    expected = {(r, b, d) for r in (1, 2, 3) for b, d in CASES}
    keyed = {}
    for result in results:
        key = (result["round"], result["batch"], result["width"])
        if key not in expected or key in keyed or result.get("status") != "COLLECTED":
            raise ValueError("RESULT_ID_OR_STATUS_MISMATCH")
        if result.get("calls_per_interval") != 100 or result.get("capacity") != 1024 or result.get("config") != dict(layout="interleaved",input_gap=0):
            raise ValueError("TIMING_POLICY_MISMATCH")
        batch,width = result["batch"],result["width"]
        launches = result.get("actual_launches",{})
        if launches.get("original_cuda") != dict(entry="upstream store_cache",public_calls=1):
            raise ValueError("ORIGINAL_ENTRY_RECORD_MISMATCH")
        for arm in ("frozen_v1","grouped"):
            grid = [(batch+3)//4 if arm == "grouped" else batch]
            if launches.get(arm) != [dict(grid=grid,parameters=launch_parameters(batch,width,width,2*width,arm))]:
                raise ValueError("ACTUAL_TRUSTED_LAUNCH_MISMATCH")
        if result.get("correctness_checks") != 190:
            raise ValueError("INCOMPLETE_ACTUAL_INPUT_CORRECTNESS")
        expected_order = [(sample, impl, policy) for sample in range(30)
                          for impl, policy in measurement_order(sample, result["round"])]
        records = result["measurements"]
        if [(x["sample"], x["implementation"], x["policy"]) for x in records] != expected_order:
            raise ValueError("MEASUREMENT_ORDER_MISMATCH")
        for impl in IMPLEMENTATIONS:
            for policy in POLICIES:
                values = result["samples_ms_per_call"][impl][policy]
                if len(values) != 30 or any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError("INVALID_INTERVAL_SAMPLES")
                ordered_values = [x["ms_per_call"] for x in records
                                  if x["implementation"] == impl and x["policy"] == policy]
                if ordered_values != values:
                    raise ValueError("RAW_SAMPLE_MISMATCH")
        keyed[key] = result
    if set(keyed) != expected:
        raise ValueError("INCOMPLETE_18_PROCESS_MATRIX")
    summary = []
    for b, d in CASES:
        row = dict(batch=b, width=d)
        for policy in POLICIES:
            entry = {}
            for impl in IMPLEMENTATIONS:
                medians = [statistics.median(keyed[(r,b,d)]["samples_ms_per_call"][impl][policy])*1000
                           for r in (1, 2, 3)]
                representative = statistics.median(medians)
                entry[impl] = dict(round_medians_us=medians, representative_us=representative,
                                   round_span_pct=100*(max(medians)-min(medians))/representative)
            grouped = entry["grouped"]["representative_us"]
            for reference in ("original_cuda", "frozen_v1"):
                label = "original" if reference == "original_cuda" else "v1"
                base = entry[reference]
                entry["speedup_" + label + "_over_grouped"] = base["representative_us"] / grouped
                entry["time_reduction_pct_vs_" + label] = 100*(1-grouped/base["representative_us"])
                entry["paired_round_speedups_" + label + "_over_grouped"] = [
                    a/b for a,b in zip(base["round_medians_us"], entry["grouped"]["round_medians_us"])]
            row[policy] = entry
        summary.append(row)
    return summary


def verify_accepted_sources(directory,helper_directory=HERE):
    hashes = json.loads((directory / "source_hashes.json").read_text(encoding="utf-8"))
    for name,expected in PINNED_31.items():
        if hashes.get(name) != expected:
            raise RuntimeError("ACCEPTED_31_RUNNER_MISMATCH: " + name)
    for name,expected in hashes.items():
        if q.sha(directory / name) != expected:
            raise RuntimeError("ACCEPTED_31_SNAPSHOT_CHANGED: " + name)
    for name in HELPERS:
        if q.sha(helper_directory / name) != hashes.get(name):
            raise RuntimeError("ACCEPTED_31_HELPER_CHANGED: " + name)
    f.verify_bound_sources(directory)
    return hashes


def verify_invalid_records(directory,records):
    expected_keys = {(arm,kind) for arm in f.ARMS for kind in w.INVALID_KINDS}
    if len(records) != 36 or {(x["implementation"],x["kind"]) for x in records} != expected_keys:
        raise RuntimeError("ACCEPTED_31_INVALID_MATRIX_INCOMPLETE")
    for record in records:
        arm,kind = record["implementation"],record["kind"]
        if record["status"] != "REJECTED_BEFORE_KV_LAUNCH" or record["submissions"] != 0:
            raise RuntimeError("ACCEPTED_31_INVALID_REJECTION_REQUIRED")
        if record["rejection"]["type"] not in ("ValueError","TypeError"):
            raise RuntimeError("UNEXPECTED_EXCEPTION_NOT_A_CONTRACT_REJECTION")
        case = directory / f"reject-{arm}-{kind}"
        saved_status = json.loads((case / "status.json").read_text(encoding="utf-8"))
        if saved_status != {k:v for k,v in record.items() if k not in ("implementation","submissions")}:
            raise RuntimeError("ACCEPTED_31_INVALID_STATUS_CHANGED")
        saved_input = json.loads((case / "input.json").read_text(encoding="utf-8"))
        saved_result = json.loads((case / "result.json").read_text(encoding="utf-8"))
        if saved_result["launches"] != [] or saved_result["rejection"] != record["rejection"]:
            raise RuntimeError("ACCEPTED_31_INVALID_LAUNCH_OR_REJECTION_CHANGED")
        if saved_input["kind"] != kind or len(saved_input["values"]) != 5 or saved_input["values"] != saved_result["values_after"]:
            raise RuntimeError("ACCEPTED_31_INVALID_ARGUMENT_VIEW_CHANGED")


def accepted_gate(directory):
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("commit") != COMMIT or status.get("plan") != f.make_plan():
        raise RuntimeError("ACCEPTED_31_FULL_PASS_REQUIRED")
    verify_accepted_sources(directory)
    configs = c.configurations()
    if status.get("configurations") != configs or len(status["cases"]) != 196 or len(status["pairs"]) != 98:
        raise RuntimeError("ACCEPTED_31_FULL_MATRIX_INCOMPLETE")
    verify_invalid_records(directory,status["rejected"])
    for index,cfg in enumerate(configs):
        oracle,outputs = c.fixture(cfg),[]
        for offset,arm in enumerate(f.ARMS):
            record = status["cases"][2*index+offset]
            expected_status = "PASS" if cfg["batch"] else "EMPTY_WRAPPER_PASS"
            if (record["implementation"],record["config"],record["status"],record["submissions"]) != (arm,cfg,expected_status,int(cfg["batch"] > 0)):
                raise RuntimeError("ACCEPTED_31_CASE_MISMATCH")
            case = directory / f"case-{index:03d}-{arm}"
            if json.loads((case / "status.json").read_text(encoding="utf-8")) != record:
                raise RuntimeError("ACCEPTED_31_CASE_STATUS_CHANGED")
            grid = [(cfg["batch"]+3)//4 if arm == "grouped" else cfg["batch"]]
            params = launch_parameters(cfg["batch"],cfg["width"],cfg["width"]+cfg["gap"],
                (cfg["width"]+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1),arm)
            launches = json.loads((case / "submissions.json").read_text(encoding="utf-8"))
            if launches != ([dict(grid=grid,parameters=params)] if cfg["batch"] else []):
                raise RuntimeError("ACCEPTED_31_CASE_LAUNCH_CHANGED")
            with np.load(case / "before.npz") as saved:
                if set(saved.files) != set(oracle):
                    raise RuntimeError("ACCEPTED_31_BEFORE_ARRAYS_MISSING")
                for name,expected in oracle.items():
                    np.testing.assert_array_equal(saved[name],expected,err_msg="ACCEPTED_31_BEFORE_"+name)
            with np.load(case / "after.npz") as saved:
                actual = {name:saved[name].copy() for name in saved.files}
            h.verify_backing(oracle,actual)
            outputs.append(actual)
        for name in ("cache","k","v","indices"):
            np.testing.assert_array_equal(outputs[0][name],outputs[1][name],err_msg="ACCEPTED_31_PAIR_"+name)
        if status["pairs"][index] != dict(config=cfg,status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"):
            raise RuntimeError("ACCEPTED_31_PAIR_STATUS_CHANGED")
    return dict(status="ACCEPTED_31_FULL_EVIDENCE_GATE_PASS",source_run=str(directory),
        status_sha256=q.sha(directory / "status.json"),source_hashes_sha256=q.sha(directory / "source_hashes.json"),
        grouped_sha256=GROUPED_HASH,positive_and_empty_cases=196,matched_pairs=98,
        nonempty_calls=168,empty_returns=28,invalid_rejections=36,invalid_submissions=0,
        scope="31 saved source/status and NPZ full backings rechecked on CPU; invalid checks cover five argument views; actual C1024 timing inputs checked again per worker")


def verify_snapshot(run,project):
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    if any(name not in hashes for name in REQUIRED_SOURCES):
        raise RuntimeError("INCOMPLETE_SIX_CASE_TIMING_SNAPSHOT")
    m.p.p.verify_snapshot(run,project)  # Existing25 manifest and live-original checks.
    f.verify_bound_sources(run)
    for name,expected in PINNED_31.items():
        if q.sha(run / name) != expected:
            raise RuntimeError("UNACCEPTED_FULL_CORRECTNESS_RUNNER")
    load_reviewed_timer(run / "learner-timing.py")


def snapshot(project,run,git):
    accepted = project / "experiments/learning/m07-grouped-full" / ACCEPTED_RUN
    gate = accepted_gate(accepted)
    h.write_json(run / "accepted_full_grouped_gate.json",gate)
    for name in ("status.json","source_hashes.json"):
        shutil.copyfile(accepted / name,run / ("prior_full_grouped_"+name))
    for name in HELPERS + NEW_FILES:
        shutil.copyfile(HERE / name,run / name)
    for name in (*q.PINNED,r.GROUPED):
        shutil.copyfile(project / "experiments" / name,run / name)
    f.verify_bound_sources(run)
    shutil.copyfile(HERE / "lessons/m07-07-grouped-six-case-timing.md",run / "hypothesis-card.md")
    for relative in git("ls-files","--","python/minisgl/kernel").splitlines():
        destination = run / "original_source" / relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(project / relative,destination)
    timer = project / "experiments/learning/m05-event-first/run-vtqqu_54/submitted_timing.py"
    shutil.copyfile(timer,run / "learner-timing.py")
    load_reviewed_timer(run / "learner-timing.py")
    hashes = {file.relative_to(run).as_posix():q.sha(file) for file in run.rglob("*") if file.is_file()}
    h.write_json(run / "source_hashes.json",hashes)
    verify_snapshot(run,project)

def worker(project, run, directory, batch, width, round_id):
    directory.mkdir(parents=True)
    result = dict(status="STARTED", batch=batch, width=width, round=round_id, capacity=1024,
                  calls_per_interval=100, seed=20261004, config=dict(layout="interleaved", input_gap=0),
                  samples_ms_per_call={n:{p:[] for p in POLICIES} for n in IMPLEMENTATIONS},
                  measurements=[], correctness_checks=0)
    h.write_json(directory / "result.json", result)
    try:
        validate_worker_case(batch, width, round_id)
        verify_snapshot(run, project)
        eager_timer = load_reviewed_timer(run / "learner-timing.py")
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_NOT_AVAILABLE")
        if (torch.__version__, torch.version.cuda, triton.__version__) != ("2.9.1+cu128", "12.8", "3.5.1"):
            raise RuntimeError("FIXED_ENVIRONMENT_CHANGED: review without automatic upgrade")
        if Path(inspect.getsourcefile(store_cache)).resolve() != (project/"python/minisgl/kernel/store.py").resolve():
            raise RuntimeError("WRONG_MINISGL_IMPORT")
        support = h.load_module(run / "kv_contract_support.py", "kv_contract_support")
        kernel = h.load_module(run / "triton_store_mask.py", "triton_store_mask").store_kv_v1
        grouped_api = h.load_module(run / r.GROUPED, "grouped_timing_module")
        grouped_kernel = grouped_api.store_kv_grouped
        if (grouped_api.NUM_WARPS,grouped_api.ROWS_PER_PROGRAM) != (4,4):
            raise RuntimeError("TRUSTED_GROUPED_PARAMETERS_MISMATCH")
        result.update(commit=COMMIT, torch=torch.__version__, cuda=torch.version.cuda,
                      triton=triton.__version__, gpu=torch.cuda.get_device_name(0),
                      device_properties=str(torch.cuda.get_device_properties(0)), telemetry_before=telemetry())
        generator = torch.Generator().manual_seed(20261004)
        k_cpu = torch.randint(0, 1024, (batch, width), generator=generator, dtype=torch.int32).to(torch.float16)
        before,expected = fixture(k_cpu.numpy())
        np.savez_compressed(directory / "before.npz", **before, expected_cache=expected)
        buffers = {n:torch.from_numpy(a.copy()).cuda() for n,a in before.items()}
        pool, k, v, indices = (buffers[n] for n in ("cache", "k", "v", "indices"))
        kc, vc = pool[:,0,:], pool[:,1,:]
        inputs = (kc,vc,indices,k,v)
        b,d,src,dst,block = support.validate_metadata(*inputs)
        if indices.shape != (b,) or (b,d,src,dst,block) != (batch,width,width,2*width,width):
            raise RuntimeError("FIXED_BENCHMARK_LAYOUT_MISMATCH")
        support.validate_index_values(indices,b,1024)  # Outside all timed/capture paths.
        h.write_json(directory / "tensors.json", [w.tensor_info(t) for t in inputs])
        expected_gpu = dict(cache=torch.from_numpy(expected.copy()).cuda(),
                            **{n:torch.from_numpy(before[n].copy()).cuda() for n in ("k","v","indices")})
        def save_arrays(stage):
            np.savez_compressed(directory/(stage+".npz"), **{n:t.cpu().numpy().copy() for n,t in buffers.items()})
        def check(stage, save=False):
            try:
                for n in buffers:
                    if not torch.equal(buffers[n],expected_gpu[n]):
                        raise AssertionError(stage+": "+n+" mismatch")
                result["correctness_checks"] += 1
            except Exception:
                try:
                    save_arrays("failed-"+stage)
                except Exception as capture_error:
                    result.setdefault("failure_capture_errors", []).append(dict(
                        stage=stage, error_type=type(capture_error).__name__, error=str(capture_error)))
                raise
            if save:
                save_arrays(stage)
        def original_call():
            store_cache(kc,vc,indices,k,v)
        def v1_call():
            return kernel[(batch,)](kc,vc,indices,k,v,WIDTH=width,SRC_ROW_STRIDE=src,
                                     DST_ROW_STRIDE=dst,BLOCK_SIZE=block,num_warps=4)
        def grouped_call():
            return grouped_kernel[((batch+3)//4,)](kc,vc,indices,k,v,
                BATCH=batch,WIDTH=width,SRC_ROW_STRIDE=src,DST_ROW_STRIDE=dst,
                BLOCK_SIZE=block,ROWS_PER_PROGRAM=grouped_api.ROWS_PER_PROGRAM,
                num_warps=grouped_api.NUM_WARPS)
        calls = dict(original_cuda=original_call,frozen_v1=v1_call,grouped=grouped_call)
        graphs = {}
        offset = (round_id - 1) % len(IMPLEMENTATIONS)
        setup_order = list(IMPLEMENTATIONS[offset:] + IMPLEMENTATIONS[:offset])
        result["setup_order"] = setup_order
        result["actual_launches"] = {}
        for impl in setup_order:
            call = calls[impl]
            pool.fill_(-1)
            if impl == "original_cuda":
                call()  # First compilation/setup is outside capture and measurement.
                result["actual_launches"][impl] = dict(entry="upstream store_cache", public_calls=1)
            elif impl == "frozen_v1":
                probe = q.LaunchProbe(kernel,4)
                probe.reset(True)
                probe[(batch,)](kc,vc,indices,k,v,WIDTH=width,SRC_ROW_STRIDE=src,
                                 DST_ROW_STRIDE=dst,BLOCK_SIZE=block,num_warps=4)
                result["actual_launches"][impl] = probe.attempts
            else:
                probe = r.GroupedProbe(grouped_kernel)
                probe.reset(True)
                probe[((batch+3)//4,)](kc,vc,indices,k,v,BATCH=batch,WIDTH=width,
                    SRC_ROW_STRIDE=src,DST_ROW_STRIDE=dst,BLOCK_SIZE=block,
                    ROWS_PER_PROGRAM=grouped_api.ROWS_PER_PROGRAM,num_warps=grouped_api.NUM_WARPS)
                result["actual_launches"][impl] = probe.attempts
                # Warmup/capture/timing use raw calls without probe or public validation.
            torch.cuda.synchronize()
            check(impl+"-initial", save=True)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(20):
                    call()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                for _ in range(100):
                    call()
            pool.fill_(-1)  # Replay must produce correct data from a reset cache.
            graph.replay()
            torch.cuda.synchronize()
            check(impl+"-graph-replay", save=True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            check(impl+"-warmup")
            graphs[impl] = graph
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        end.record()
        end.synchronize()
        print(f"PAIRED_READY round={round_id} B={batch} D={width}: three initial and three reset-Graph checks passed", flush=True)
        for sample in range(30):
            for impl, policy in measurement_order(sample,round_id):
                ms = float(eager_timer(calls[impl],start,end) if policy == "eager"
                           else measure_graph(graphs[impl],start,end))
                if not math.isfinite(ms) or ms <= 0:
                    raise ValueError(f"INVALID_TIMING {impl}/{policy}: {ms}")
                result["samples_ms_per_call"][impl][policy].append(ms)
                result["measurements"].append(dict(sample=sample, implementation=impl, policy=policy, ms_per_call=ms))
                check(f"sample-{sample}-{impl}-{policy}")
            # Save partial samples outside measurement; a later error never hides them.
            h.write_json(directory / "result.json", result)
        check("final",save=True)
        result["telemetry_after"] = telemetry()
        verify_snapshot(run,project)
        result.update(status="COLLECTED", policy=make_plan(),
                      scope="three implementations passed this benchmark input; timings require review")
    except BaseException as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (directory/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(directory/"result.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--batch", type=int, choices=(1,16,256))
    parser.add_argument("--width", type=int, choices=(64,128))
    parser.add_argument("--round", type=int, choices=(1,2,3))
    args = parser.parse_args()
    if sys.flags.optimize:
        raise RuntimeError("ASSERTS_REQUIRED")
    if args.dry_run:
        print(json.dumps(make_plan(), indent=2))
        return
    if args.project is None:
        parser.error("--project required")
    project = args.project.resolve()
    if args.worker_dir:
        if None in (args.snapshot,args.batch,args.width,args.round):
            parser.error("worker requires snapshot/batch/width/round")
        validate_worker_case(args.batch,args.width,args.round)
        worker(project,args.snapshot,args.worker_dir,args.batch,args.width,args.round)
        return
    if args.shared_results is None:
        parser.error("--shared-results required")
    def git(*words):
        return subprocess.check_output(["git","-C",str(project),*words],text=True).strip()
    if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("PINNED_TRACKED_SOURCE_CHANGED: preserve evidence and review")
    parent = project / "experiments/learning/m07-grouped-six-timing"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 grouped six-case timing",commit=COMMIT,plan=make_plan(),results=[],
                 ownership="coach demonstrated grouped rows/mask/grid; learner applied and ran accepted correctness; four warps held fixed; reviewed Event body reused; orchestration/Graph/records by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE=three arms; six cases x three fresh processes; Graph primary", flush=True)
    try:
        snapshot(project,run,git)
        h.write_json(run / "plan.json",state["plan"])
        environment = None
        for item in state["plan"]["schedule"]:
            r,b,d = (item[k] for k in ("round","batch","width"))
            directory = run / f"round-{r}-B{b}-D{d}"
            console = run / f"round-{r}-B{b}-D{d}.log"
            command = [sys.executable,"-B",str(run / "32-compare-kv-grouped-full.py"),"--project",str(project),
                       "--snapshot",str(run),"--worker-dir",str(directory),"--batch",str(b),"--width",str(d),"--round",str(r)]
            state.update(status="RUNNING",current=item)
            h.write_json(run / "status.json",state)
            h.write_json(run / (console.stem+"-command.json"),command)
            print(f"RUN round={r}/3 B={b} D={d} three implementations",flush=True)
            with console.open("x",encoding="utf-8") as log:
                completed = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                print(console.read_text(encoding="utf-8"),flush=True)
                raise RuntimeError("GROUPED_SIX_TIMING_WORKER_FAILED: " + str(console))
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if result["status"] != "COLLECTED" or any(result[k] != v for k,v in item.items()):
                raise ValueError("WORKER_RESULT_MISMATCH")
            identity = tuple(result[k] for k in ("commit","torch","cuda","triton","gpu","device_properties"))
            if environment is None:
                environment = identity
            if identity != environment:
                raise RuntimeError("ENVIRONMENT_CHANGED_BETWEEN_WORKERS")
            state["results"].append(result)
            h.write_json(run / "partial-results.json",state["results"])
            for policy in POLICIES:
                a,v,k = [statistics.median(result["samples_ms_per_call"][n][policy])*1000 for n in IMPLEMENTATIONS]
                print(f"COLLECTED {policy} original={a:.6f} us v1={v:.6f} us grouped={k:.6f} us "
                      f"v1/grouped={v/k:.4f}x original/grouped={a/k:.4f}x",flush=True)
        if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
            raise RuntimeError("TRACKED_SOURCE_CHANGED_DURING_RUN")
        summary = summarize(state["results"])
        state.update(status="COLLECTION_COMPLETE",current=None,summary=summary)
        h.write_json(run / "summary.json",dict(status=state["status"],plan=state["plan"],summary=summary))
        for row in summary:
            entry = row["graph"]
            print(f"GRAPH_SUMMARY B={row['batch']} D={row['width']} "
                  f"original={entry['original_cuda']['representative_us']:.6f} us/call "
                  f"v1={entry['frozen_v1']['representative_us']:.6f} us/call "
                  f"grouped={entry['grouped']['representative_us']:.6f} us/call "
                  f"v1/grouped={entry['speedup_v1_over_grouped']:.4f}x "
                  f"original/grouped={entry['speedup_original_over_grouped']:.4f}x",flush=True)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        export_run(run,args.shared_results,state["status"] == "COLLECTION_COMPLETE")
    print("GROUPED_SIX_TIMING_COLLECTION_COMPLETE: 18 processes; 3240 interval averages; review all rounds before a performance claim")
    print("LIMIT: six-case reused-address FP16 microbenchmark; Graph/Eager separate; not model or AMD speedup")


if __name__ == "__main__":
    main()
