"""M06-06 coach orchestration: same-session original CUDA / accepted Triton v1.

Six M05 configurations, three fresh processes each; both implementations share
the same buffers within a process. Graph is primary, Eager reported separately.
The reviewed learner 100-call Event timer is reused without executing its file.
No changes to learner kernels, no optimization or performance acceptance here.
"""
import argparse
import ast
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
spec = importlib.util.spec_from_file_location("paired_common_helpers", HERE / "23-check-kv-common.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
h, w, m, COMMIT = c.h, c.w, c.m, c.COMMIT
IMPLEMENTATIONS = ("original_cuda", "learner_triton")
POLICIES = ("graph", "eager")
CASES = [(b, d) for b in (1, 16, 256) for d in (64, 128)]
TIMER_SOURCE = """
def measure_once(call, start, end):
    start.record()
    for _ in range(100):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100
"""


def load_reviewed_timer(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "measure_once"]
    expected = ast.parse(TIMER_SOURCE).body[0]
    if len(found) != 1 or ast.dump(found[0]) != ast.dump(expected):
        raise RuntimeError("REVIEWED_TIMER_CHANGED: preserve original 100-call measurement")
    namespace = {}
    exec(compile(ast.Module(body=[found[0]], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["measure_once"]


def measurement_order(sample, round_id):
    forward = [("original_cuda", "graph"), ("learner_triton", "graph"),
               ("original_cuda", "eager"), ("learner_triton", "eager")]
    return forward if (sample + round_id) % 2 == 0 else list(reversed(forward))


def measure_graph(graph, start, end):
    start.record()
    graph.replay()  # This graph contains exactly 100 calls.
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


def make_plan():
    rng, schedule = random.Random(104), []
    for round_id in (1, 2, 3):
        ordered = CASES.copy()
        rng.shuffle(ordered)
        schedule.extend(dict(round=round_id, batch=b, width=d) for b, d in ordered)
    return dict(cases=CASES, schedule=schedule, capacity=1024, dtype="float16",
                layout="interleaved", input_gap=0, input_seed=20261004, order_seed=104,
                indices="0..B-1, int32, unique", rounds=3, intervals_per_impl_policy=30,
                calls_per_interval=100, eager_warmup_calls=20, graph_warmup_replays=5,
                primary="graph", reference="eager", num_warps=4,
                timed_entries=dict(original_cuda="upstream store_cache Python/JIT entry",
                                   learner_triton="trusted raw kernel launcher with prevalidated fixed arguments"),
                excluded="allocation, validation, first JIT/setup, warmup, checks, snapshots, telemetry",
                order="alternate forward/reverse four-measurement blocks; 15 each per process",
                address_policy="same shared buffers for both implementations; repeated addresses, no cache flush",
                statistic="median of 30 interval averages per process; median of three process medians",
                scope="same-session six-case microbenchmark; not model/DRAM/AMD measurement",
                limits="Eager includes distinct host submission paths; Graph comparison is primary; no automatic speedup acceptance")


def summarize(results):
    expected_keys = {(r, b, d) for r in (1, 2, 3) for b, d in CASES}
    keyed = {}
    for result in results:
        key = (result["round"], result["batch"], result["width"])
        if key not in expected_keys or key in keyed or result.get("status") != "COLLECTED":
            raise ValueError("RESULT_ID_OR_STATUS_MISMATCH")
        for impl in IMPLEMENTATIONS:
            for policy in POLICIES:
                values = result["samples_ms_per_call"][impl][policy]
                if len(values) != 30 or any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError("INVALID_INTERVAL_SAMPLES")
        keyed[key] = result
    if set(keyed) != expected_keys:
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
            original, candidate = (entry[n]["representative_us"] for n in IMPLEMENTATIONS)
            entry.update(speedup_original_over_triton=original/candidate,
                         time_reduction_pct=100*(1-candidate/original),
                         paired_round_speedups=[a/b for a,b in zip(entry["original_cuda"]["round_medians_us"],
                                                                 entry["learner_triton"]["round_medians_us"])])
            row[policy] = entry
        summary.append(row)
    return summary


def verify_snapshot(run, project):
    hashes = json.loads((run / "source_hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("SNAPSHOT_CHANGED: " + name)
        if name.startswith("original_source/"):
            live = project / name.removeprefix("original_source/")
            if hashlib.sha256(live.read_bytes()).hexdigest() != expected:
                raise RuntimeError("LIVE_ORIGINAL_CHANGED: " + name)


def telemetry():
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return dict(command=command, returncode=out.returncode, stdout=out.stdout, stderr=out.stderr,
                    scope="outside measurement, before/after snapshot; not continuous monitoring")
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status="UNAVAILABLE", reason=str(error))


def worker(project, run, directory, batch, width, round_id):
    directory.mkdir(parents=True)
    result = dict(status="STARTED", batch=batch, width=width, round=round_id, capacity=1024,
                  calls_per_interval=100, seed=20261004, config=dict(layout="interleaved", input_gap=0),
                  samples_ms_per_call={n:{p:[] for p in POLICIES} for n in IMPLEMENTATIONS},
                  measurements=[], correctness_checks=0)
    h.write_json(directory / "result.json", result)
    try:
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
                save_arrays("failed-"+stage)
                raise
            if save:
                save_arrays(stage)
        def original_call():
            store_cache(kc,vc,indices,k,v)
        def triton_call():
            kernel[(batch,)](kc,vc,indices,k,v,WIDTH=width,SRC_ROW_STRIDE=src,
                             DST_ROW_STRIDE=dst,BLOCK_SIZE=block,num_warps=4)
        calls = dict(original_cuda=original_call, learner_triton=triton_call)
        graphs = {}
        setup_order = list(IMPLEMENTATIONS) if round_id % 2 else list(reversed(IMPLEMENTATIONS))
        result["setup_order"] = setup_order
        for impl in setup_order:
            call = calls[impl]
            pool.fill_(-1)
            call()  # First compilation/setup is outside capture and measurement.
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
        print(f"PAIRED_READY round={round_id} B={batch} D={width}: two initial and two reset-Graph checks passed", flush=True)
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
                      scope="both implementations passed this benchmark input; timings require review")
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
    if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
        raise RuntimeError("PINNED_TRACKED_SOURCE_CHANGED: preserve evidence and review")
    parent = project/"experiments/learning/m06-paired-timing"
    parent.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"),dir=parent))
    state = dict(status="PREPARING",lesson="M06-06",commit=COMMIT,plan=make_plan(),results=[],
                 ownership="learner accepted kernel and reviewed Event body; orchestration/Graph/records by coach")
    print(f"RUN_DIRECTORY={run}\nSCOPE=same-session original/Triton; six cases x three fresh processes; Graph primary",flush=True)
    try:
        hashes = c.snapshot(project,run,git)
        for name in ("25-compare-kv-v1.py","25-compare-kv-v1.sh"):
            shutil.copyfile(HERE/name,run/name)
            hashes[name] = hashlib.sha256((run/name).read_bytes()).hexdigest()
        archived = project/"experiments/learning/m05-event-first/run-vtqqu_54/submitted_timing.py"
        shutil.copyfile(archived,run/"learner-timing.py")
        load_reviewed_timer(run/"learner-timing.py")
        hashes["learner-timing.py"] = hashlib.sha256((run/"learner-timing.py").read_bytes()).hexdigest()
        h.write_json(run/"source_hashes.json",hashes)
        h.write_json(run/"source_review.json",dict(wrapper=w.review_api((run/"triton_store_api.py").read_text(encoding="utf-8-sig")),
                                                  mask=m.validate_mask_source((run/"triton_store_mask.py").read_text(encoding="utf-8-sig"))))
        h.write_json(run/"plan.json",state["plan"])
        environment = None
        for item in state["plan"]["schedule"]:
            r,b,d = (item[k] for k in ("round","batch","width"))
            directory = run/f"round-{r}-B{b}-D{d}"
            console = run/f"round-{r}-B{b}-D{d}.log"
            command = [sys.executable,"-B",str(run/"25-compare-kv-v1.py"),"--project",str(project),
                       "--snapshot",str(run),"--worker-dir",str(directory),"--batch",str(b),"--width",str(d),"--round",str(r)]
            state.update(status="RUNNING",current=item)
            h.write_json(run/"status.json",state)
            h.write_json(run/(console.stem+"-command.json"),command)
            print(f"RUN round={r}/3 B={b} D={d} both implementations",flush=True)
            with console.open("w",encoding="utf-8") as log:
                completed = subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:
                print(console.read_text(encoding="utf-8"),flush=True)
                raise RuntimeError("PAIRED_WORKER_FAILED: "+str(console))
            result = json.loads((directory/"result.json").read_text(encoding="utf-8"))
            if result["status"] != "COLLECTED" or any(result[k] != v for k,v in item.items()):
                raise ValueError("WORKER_RESULT_MISMATCH")
            identity = tuple(result[k] for k in ("commit","torch","cuda","triton","gpu","device_properties"))
            if environment is None:
                environment = identity
            if identity != environment:
                raise RuntimeError("ENVIRONMENT_CHANGED_BETWEEN_WORKERS")
            state["results"].append(result)
            h.write_json(run/"partial-results.json",state["results"])
            for policy in POLICIES:
                a,v = [statistics.median(result["samples_ms_per_call"][n][policy])*1000 for n in IMPLEMENTATIONS]
                print(f"COLLECTED {policy} original={a:.6f} us triton={v:.6f} us ratio={a/v:.4f}x",flush=True)
        if git("rev-parse","HEAD") != COMMIT or git("status","--porcelain","--untracked-files=no"):
            raise RuntimeError("TRACKED_SOURCE_CHANGED_DURING_RUN")
        summary = summarize(state["results"])
        state.update(status="COLLECTION_COMPLETE",current=None,summary=summary)
        h.write_json(run/"summary.json",dict(status=state["status"],plan=state["plan"],summary=summary))
        for row in summary:
            a = row["graph"]["original_cuda"]["representative_us"]
            v = row["graph"]["learner_triton"]["representative_us"]
            print(f"GRAPH_SUMMARY B={row['batch']} D={row['width']} original={a:.6f} us triton={v:.6f} us ratio={a/v:.4f}x",flush=True)
    except BaseException as error:
        state.update(status="FAILED",error_type=type(error).__name__,error=str(error))
        (run/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        h.write_json(run/"status.json",state)
        args.shared_results.mkdir(parents=True,exist_ok=True)
        shared = args.shared_results/run.name
        try:
            shutil.copytree(run,shared)
            print(f"SHARED_ARTIFACTS={shared}",flush=True)
        except OSError as error:
            print(f"SHARED_COPY_FAILED={error}; WSL evidence retained at {run}",flush=True)
            if state["status"] == "COLLECTION_COMPLETE":
                raise
    print("PAIRED_COLLECTION_COMPLETE: 18 processes; 2160 interval averages; review all rounds before a performance claim")
    print("LIMIT: reused-address FP16 microbenchmark; Eager and Graph separate; no automatic M06 acceptance or model speedup")


if __name__ == "__main__":
    main()
