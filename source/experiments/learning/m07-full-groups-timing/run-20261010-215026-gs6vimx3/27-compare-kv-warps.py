"""M07 first timing experiment: original CUDA / frozen v1=4 / candidate=1.

Only B1/D64 and B256/D128; 3 fresh processes per case, 1080 interval averages.
Actual benchmark-input correctness precedes timing. Graph is primary; no score
or performance conclusion is automatic. Learner sources and old runs are read only.
"""
import argparse
import hashlib
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
spec = importlib.util.spec_from_file_location("warps_paired_helpers", HERE / "25-compare-kv-v1.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
c, h, w, m, COMMIT = p.c, p.h, p.w, p.m, p.COMMIT
q = c.helper("warps_correctness_helpers", "26-check-kv-warps.py")
load_reviewed_timer, measure_graph, telemetry = p.load_reviewed_timer, p.measure_graph, p.telemetry
IMPLEMENTATIONS = ("original_cuda", "frozen_v1", "candidate")
POLICIES = ("graph", "eager")
CASES = ((1, 64), (256, 128))
ACCEPTED_RUN = "run-20261010-115255-6bwgq_qo"
CANDIDATE_HASH = "f2320f7ede79b9e4c7164ae0d91b87a53a718b42757b0c64e9bfb64bf070092b"
REQUIRED_SOURCE_NAMES = tuple(q.HELPERS) + ("25-compare-kv-v1.py", "27-compare-kv-warps.py", "27-compare-kv-warps.sh", "learner-timing.py", "hypothesis-card.md", "accepted_correctness_gate.json", "original_source/python/minisgl/kernel/store.py", "original_source/python/minisgl/kernel/csrc/jit/store.cu") + tuple(q.PINNED) + (q.CANDIDATE,)


def measurement_order(sample, round_id):
    ring = [(impl, policy) for policy in POLICIES for impl in IMPLEMENTATIONS]
    offset = (sample + round_id - 1) % len(ring)
    order = ring[offset:] + ring[:offset]
    return list(reversed(order)) if (sample // 6 + round_id - 1) % 2 else order


def make_plan():
    rng, schedule = random.Random(107), []
    for round_id in (1, 2, 3):
        ordered = list(CASES)
        rng.shuffle(ordered)
        schedule.extend(dict(round=round_id, batch=b, width=d) for b, d in ordered)
    return dict(cases=CASES, schedule=schedule, capacity=1024, dtype="float16",
                layout="interleaved", input_gap=0, input_seed=20261004, order_seed=107,
                indices="0..B-1, int32, unique", rounds=3, fresh_processes=6,
                intervals_per_impl_policy=30, calls_per_interval=100, interval_averages=1080,
                eager_warmup_calls=20, graph_warmup_replays=5,
                primary="graph", reference="eager", num_warps=dict(frozen_v1=4, candidate=1),
                timed_entries=dict(original_cuda="upstream store_cache Python/JIT entry",
                                   frozen_v1="trusted raw kernel launcher with prevalidated fixed arguments",
                                   candidate="trusted raw kernel launcher; NUM_WARPS from accepted candidate snapshot"),
                excluded="allocation, public validation, first JIT/setup, warmup, checks, snapshots, telemetry",
                order="rotate six arm/policy measurements; reverse alternate six-sample blocks; each item occupies each position five times per process",
                address_policy="same shared buffers for all three implementations; repeated addresses, no cache flush",
                statistic="median of 30 interval averages per process; median of three process medians",
                correctness="each arm resets cache before initial call and before 100-call Graph replay; all backings checked",
                accepted_correctness_run=ACCEPTED_RUN, candidate_sha256=CANDIDATE_HASH,
                scope="two-case NVIDIA FP16 microbenchmark; not all-six-case regression, model, DRAM, or AMD measurement",
                limits="Eager includes distinct host submission paths; Graph primary; no automatic speedup acceptance")


def validate_worker_case(batch, width, round_id):
    if (batch, width) not in CASES or round_id not in (1, 2, 3):
        raise ValueError("UNPLANNED_WORKER_CASE")


def summarize(results):
    expected = {(r, b, d) for r in (1, 2, 3) for b, d in CASES}
    keyed = {}
    for result in results:
        key = (result["round"], result["batch"], result["width"])
        if key not in expected or key in keyed or result.get("status") != "COLLECTED":
            raise ValueError("RESULT_ID_OR_STATUS_MISMATCH")
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
        raise ValueError("INCOMPLETE_6_PROCESS_MATRIX")
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
            candidate = entry["candidate"]["representative_us"]
            for reference in ("original_cuda", "frozen_v1"):
                label = "original" if reference == "original_cuda" else "v1"
                base = entry[reference]
                entry["speedup_" + label + "_over_candidate"] = base["representative_us"] / candidate
                entry["time_reduction_pct_vs_" + label] = 100*(1-candidate/base["representative_us"])
                entry["paired_round_speedups_" + label + "_over_candidate"] = [
                    a/b for a,b in zip(base["round_medians_us"], entry["candidate"]["round_medians_us"])]
            row[policy] = entry
        summary.append(row)
    return summary


def accepted_gate(directory):
    """Recheck accepted source identity and all saved status/launch records; no GPU."""
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("commit") != COMMIT or status.get("arms") != q.ARMS:
        raise RuntimeError("ACCEPTED_CORRECTNESS_STATUS_REQUIRED")
    hashes = json.loads((directory / "source_hashes.json").read_text(encoding="utf-8"))
    required = dict(q.PINNED, **{q.CANDIDATE: CANDIDATE_HASH})
    for name, expected in required.items():
        if hashes.get(name) != expected or q.sha(directory / name) != expected:
            raise RuntimeError("ACCEPTED_CORRECTNESS_SOURCE_MISMATCH: " + name)
    for name, expected in hashes.items():
        if q.sha(directory / name) != expected:
            raise RuntimeError("ACCEPTED_CORRECTNESS_SNAPSHOT_CHANGED: " + name)
    configs = c.configurations()
    if status.get("configurations") != configs or len(status["cases"]) != 196 or len(status["pairs"]) != 98:
        raise RuntimeError("ACCEPTED_CORRECTNESS_MATRIX_INCOMPLETE")
    for index, cfg in enumerate(configs):
        for offset, arm in enumerate(q.ARMS):
            record = status["cases"][2*index+offset]
            expected_status = "PASS" if cfg["batch"] else "EMPTY_WRAPPER_PASS"
            if (record["implementation"], record["config"], record["status"], record["submissions"]) != (arm, cfg, expected_status, int(cfg["batch"] > 0)):
                raise RuntimeError("ACCEPTED_CORRECTNESS_CASE_MISMATCH")
            case_dir = directory / f"case-{index:03d}-{arm}"
            if json.loads((case_dir / "status.json").read_text(encoding="utf-8")) != record:
                raise RuntimeError("ACCEPTED_CASE_STATUS_CHANGED")
            launches = json.loads((case_dir / "submissions.json").read_text(encoding="utf-8"))
            expected_parameters = dict(WIDTH=cfg["width"], SRC_ROW_STRIDE=cfg["width"]+cfg["gap"],
                                       DST_ROW_STRIDE=(cfg["width"]+cfg["guard"])*(2 if cfg["layout"] == "interleaved" else 1),
                                       BLOCK_SIZE=1 << (cfg["width"]-1).bit_length(), num_warps=q.ARMS[arm])
            expected_launches = [dict(grid=[cfg["batch"]], parameters=expected_parameters)] if cfg["batch"] else []
            if launches != expected_launches:
                raise RuntimeError("ACCEPTED_CASE_LAUNCH_CHANGED")
        if status["pairs"][index] != dict(config=cfg, status="BOTH_MATCH_REFERENCE_AND_EACH_OTHER"):
            raise RuntimeError("ACCEPTED_PAIR_STATUS_CHANGED")
    expected_rejections = {(arm, kind) for arm in q.ARMS for kind in w.INVALID_KINDS}
    records = status["rejected"]
    if len(records) != 36 or {(x["implementation"],x["kind"]) for x in records} != expected_rejections or any(x["status"] != "REJECTED_BEFORE_KV_LAUNCH" for x in records):
        raise RuntimeError("ACCEPTED_INVALID_CHECKS_INCOMPLETE")
    return dict(status="ACCEPTED_SOURCE_AND_STATUS_GATE_PASS", source_run=str(directory),
                status_sha256=q.sha(directory / "status.json"), source_hashes_sha256=q.sha(directory / "source_hashes.json"),
                required_sources=required, positive_cases=196, invalid_rejections=36,
                scope="accepted records/source identity only; actual timing inputs checked again in every worker")


def verify_snapshot(run, project):
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    if any(name not in hashes for name in REQUIRED_SOURCE_NAMES):
        raise RuntimeError("INCOMPLETE_TIMING_SNAPSHOT")
    p.verify_snapshot(run, project)
    q.check_pinned(run)
    if q.sha(run / q.CANDIDATE) != CANDIDATE_HASH:
        raise RuntimeError("UNACCEPTED_CANDIDATE_SOURCE")
    q.candidate_review((run / q.CANDIDATE).read_text(encoding="utf-8-sig"))
    load_reviewed_timer(run / "learner-timing.py")


def snapshot(project, run, git):
    gate = accepted_gate(project / "experiments/learning/m07-warps" / ACCEPTED_RUN)
    h.write_json(run / "accepted_correctness_gate.json", gate)
    shutil.copyfile(HERE / "lessons/m07-02-warps-timing.md", run / "hypothesis-card.md")
    for name in tuple(q.HELPERS) + ("25-compare-kv-v1.py", "27-compare-kv-warps.py", "27-compare-kv-warps.sh"):
        shutil.copyfile(HERE / name, run / name)
    for name in (*q.PINNED, q.CANDIDATE):
        shutil.copyfile(project / "experiments" / name, run / name)
    q.check_pinned(run)
    if q.sha(run / q.CANDIDATE) != CANDIDATE_HASH:
        raise RuntimeError("UNACCEPTED_LIVE_CANDIDATE: run correctness and review before timing")
    q.candidate_review((run / q.CANDIDATE).read_text(encoding="utf-8-sig"))
    for relative in git("ls-files", "--", "python/minisgl/kernel").splitlines():
        destination = run / "original_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, destination)
    archived = project / "experiments/learning/m05-event-first/run-vtqqu_54/submitted_timing.py"
    shutil.copyfile(archived, run / "learner-timing.py")
    load_reviewed_timer(run / "learner-timing.py")
    hashes = {str(f.relative_to(run)).replace(chr(92), "/"): q.sha(f) for f in run.rglob("*") if f.is_file()}
    h.write_json(run / "source_hashes.json", hashes)
    verify_snapshot(run, project)


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
        candidate_api = h.load_module(run / "triton_store_candidate.py", "warps_timing_candidate")
        warps = {"frozen_v1": 4, "candidate": candidate_api.NUM_WARPS}
        if warps != {"frozen_v1": 4, "candidate": 1}:
            raise RuntimeError("TRUSTED_WARPS_MISMATCH")
        result.update(commit=COMMIT, torch=torch.__version__, cuda=torch.version.cuda,
                      triton=triton.__version__, gpu=torch.cuda.get_device_name(0),
                      device_properties=str(torch.cuda.get_device_properties(0)), telemetry_before=telemetry())
        generator = torch.Generator().manual_seed(20261004)
        k_cpu = torch.randint(0, 1024, (batch, width), generator=generator, dtype=torch.int32).to(torch.float16)
        v_cpu = k_cpu + 1024
        before = dict(cache=np.full((1024,2,width), -1, dtype=np.float16),
                      k=k_cpu.numpy().copy(), v=v_cpu.numpy().copy(), indices=np.arange(batch, dtype=np.int32))
        expected = before["cache"].copy()
        for pid, pos in enumerate(before["indices"]):
            expected[pos,0] = before["k"][pid]
            expected[pos,1] = before["v"][pid]
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
        def trusted_call(num_warps):
            def call():
                return kernel[(batch,)](kc,vc,indices,k,v,WIDTH=width,SRC_ROW_STRIDE=src,
                                         DST_ROW_STRIDE=dst,BLOCK_SIZE=block,num_warps=num_warps)
            return call
        calls = dict(original_cuda=original_call,
                     frozen_v1=trusted_call(warps["frozen_v1"]),
                     candidate=trusted_call(warps["candidate"]))
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
            else:
                probe = q.LaunchProbe(kernel, warps[impl])
                probe.reset(True)
                probe[(batch,)](kc,vc,indices,k,v,WIDTH=width,SRC_ROW_STRIDE=src,
                                 DST_ROW_STRIDE=dst,BLOCK_SIZE=block,num_warps=warps[impl])
                result["actual_launches"][impl] = probe.attempts
                # Timed calls use the same fixed values without probe recording overhead.
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


def export_run(run, destination, successful):
    try:
        destination.mkdir(parents=True, exist_ok=True)
        shared = destination / run.name
        shutil.copytree(run, shared)
        print(f"SHARED_ARTIFACTS={shared}", flush=True)
    except OSError as error:
        print(f"SHARED_COPY_FAILED={error}; WSL evidence retained at {run}", flush=True)
        if successful:
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--batch", type=int, choices=(1,256))
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
    parent = project / "experiments/learning/m07-warps-timing"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M07 first timing experiment",commit=COMMIT,plan=make_plan(),results=[],
                 ownership="learner changed NUM_WARPS; accepted kernel and reviewed Event body reused; orchestration/Graph/records by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE=three arms; two cases x three fresh processes; Graph primary", flush=True)
    try:
        snapshot(project,run,git)
        h.write_json(run / "plan.json",state["plan"])
        environment = None
        for item in state["plan"]["schedule"]:
            r,b,d = (item[k] for k in ("round","batch","width"))
            directory = run / f"round-{r}-B{b}-D{d}"
            console = run / f"round-{r}-B{b}-D{d}.log"
            command = [sys.executable,"-B",str(run / "27-compare-kv-warps.py"),"--project",str(project),
                       "--snapshot",str(run),"--worker-dir",str(directory),"--batch",str(b),"--width",str(d),"--round",str(r)]
            state.update(status="RUNNING",current=item)
            h.write_json(run / "status.json",state)
            h.write_json(run / (console.stem+"-command.json"),command)
            print(f"RUN round={r}/3 B={b} D={d} three implementations",flush=True)
            with console.open("x",encoding="utf-8") as log:
                completed = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                print(console.read_text(encoding="utf-8"),flush=True)
                raise RuntimeError("WARPS_TIMING_WORKER_FAILED: " + str(console))
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
                print(f"COLLECTED {policy} original={a:.6f} us v1={v:.6f} us candidate={k:.6f} us "
                      f"v1/candidate={v/k:.4f}x original/candidate={a/k:.4f}x",flush=True)
        if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
            raise RuntimeError("TRACKED_SOURCE_CHANGED_DURING_RUN")
        summary = summarize(state["results"])
        state.update(status="COLLECTION_COMPLETE",current=None,summary=summary)
        h.write_json(run / "summary.json",dict(status=state["status"],plan=state["plan"],summary=summary))
        for row in summary:
            entry = row["graph"]
            print(f"GRAPH_SUMMARY B={row['batch']} D={row['width']} "
                  f"v1/candidate={entry['speedup_v1_over_candidate']:.4f}x "
                  f"original/candidate={entry['speedup_original_over_candidate']:.4f}x",flush=True)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run / "status.json",state)
        export_run(run,args.shared_results,state["status"] == "COLLECTION_COMPLETE")
    print("WARPS_TIMING_COLLECTION_COMPLETE: 6 processes; 1080 interval averages; review all rounds before a performance claim")
    print("LIMIT: two-case reused-address FP16 microbenchmark; Graph/Eager separate; not full regression, model or AMD speedup")


if __name__ == "__main__":
    main()
