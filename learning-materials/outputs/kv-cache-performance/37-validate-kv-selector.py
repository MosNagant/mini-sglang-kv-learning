"""Coach-written M07-37: frozen conservative selection, independent A/A validation.

Three previously unused configurations x three fresh processes; original and
auto-selected entry are BOTH original. Graph/Eager: 30 intervals x100 calls.
No candidate optimization is measured or promoted. --dry-run is CPU-only.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime

import numpy as np

HERE = Path(__file__).resolve().parent
COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
POLICY_NAME = "37-kv-selection-policy.json"
NEW_FILES = (POLICY_NAME, "37-kv-selector.py", "37-validate-kv-selector.py",
             "37-validate-kv-selector.sh", "37-selector-cpu-test.py")
ACCEPTED35 = "m07-full-groups-timing-results/run-20261010-215026-gs6vimx3"
ARMS, POLICIES = ("original_cuda", "selected"), ("graph", "eager")
CASES = ((7, 64), (64, 128), (257, 192))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


s = module(HERE / "37-kv-selector.py", "frozen37_selector")


def configuration(b, d):
    require((b, d) in CASES, "UNPLANNED_HOLDOUT_CONFIGURATION")
    return dict(batch=b, width=d, capacity=512, layout="interleaved" if d == 64 else "separate",
                gap=0 if d == 64 else 64, guard=64 if d == 192 else 0,
                rows=[(53 * i + 19) % 512 for i in range(b)])


def measurement_order(sample, round_id):
    ring = [(a, p) for p in POLICIES for a in ARMS]
    offset = (sample + round_id - 1) % 4
    order = ring[offset:] + ring[:offset]
    return order[::-1] if (sample // 4 + round_id - 1) % 2 else order


def make_plan():
    rng, schedule = random.Random(137), []
    for rr in (1, 2, 3):
        cases = list(CASES)
        rng.shuffle(cases)
        schedule.extend(dict(round=rr, batch=b, width=d) for b, d in cases)
    return dict(cases=[list(x) for x in CASES], configurations=[configuration(*x) for x in CASES],
                schedule=schedule, capacity=512, dtype="float16", index_dtype="int32",
                input_seed=20261014, indices="(53*i+19)%512; unique int32", rounds=3,
                fresh_processes=9, arms=list(ARMS), policies=list(POLICIES),
                intervals_per_arm_policy=30, calls_per_interval=100, interval_averages=1080,
                recorded_checks_per_worker=127, saved_after_stages_per_worker=5,
                eager_warmup_calls=20, graph_warmup_replays=5, primary="graph", reference="eager",
                order="rotate four entries, reverse alternate four-sample blocks; each position appears7/8 times per worker",
                address_policy="same shared buffers and repeated addresses; no flush",
                timed_entries="both arms use the exact same upstream store_cache closure; selector and validation run before capture/timing",
                statistic="median30 interval averages per process; median3 process medians; paired ratios also retained",
                policy_fingerprint=s.POLICY_CANONICAL_SHA256,
                selection="frozen historical27/30/32/35 gate; auto original;36/37 cannot modify policy",
                scope="independent B7/D64, B64/D128, B257/D192 cases; A/A default-entry validation, no candidate speedup or model claim")


def cache_views(cache, layout):
    return (cache[:, 0, :], cache[:, 1, :]) if layout == "interleaved" else (cache[0], cache[1])


def fixture(cfg):
    require(cfg == configuration(cfg["batch"], cfg["width"]), "FIXTURE_CONFIG_CHANGED")
    b, d, c = (cfg[x] for x in ("batch", "width", "capacity"))
    pitch = d + cfg["guard"]
    shape = (c, 2, pitch) if cfg["layout"] == "interleaved" else (2, c, pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    for view in cache_views(cache, cfg["layout"]):
        view[:, d:] = -7
    values = np.random.default_rng(20261014).integers(0, 1024, (b, d), dtype=np.int32).astype(np.float16)
    values[:, 0] = np.arange(b, dtype=np.float16)
    k, v = (np.full((b, d + cfg["gap"]), -9, dtype=np.float16) for _ in range(2))
    k[:, :d], v[:, :d] = values, values + np.float16(1024)
    expected = cache.copy()
    ek, ev = cache_views(expected, cfg["layout"])
    for pid, pos in enumerate(cfg["rows"]):
        ek[pos, :d], ev[pos, :d] = k[pid, :d], v[pid, :d]
    return dict(cache=cache, k=k, v=v, indices=np.asarray(cfg["rows"], dtype=np.int32), expected_cache=expected)


def check_cpu_arrays(cfg, before, after):
    expected = before["cache"].copy()
    ek, ev = cache_views(expected, cfg["layout"])
    for pid, pos in enumerate(before["indices"]):
        ek[pos, :cfg["width"]] = before["k"][pid, :cfg["width"]]
        ev[pos, :cfg["width"]] = before["v"][pid, :cfg["width"]]
    require(np.array_equal(expected, before["expected_cache"]), "FIXTURE_ORACLE_MISMATCH")
    require(set(after) == {"cache", "k", "v", "indices"}, "ARRAY_SET_MISMATCH")
    for name, value in after.items():
        ref = expected if name == "cache" else before[name]
        require(value.dtype == ref.dtype and value.shape == ref.shape and np.array_equal(value, ref),
                "FULL_BACKING_MISMATCH:" + name)


def historical_gate(base, policy):
    """Read exact archived files; new36/37 outputs are not inputs to this gate."""
    counts = dict(sessions=0, workers=0, files_hashed=0, screened_entries=0, promoted=0)
    for session in policy["historical_sessions"]:
        directory = base / "logs" / session["run"]
        for name, digest in {**session["files"], **session["workers"]}.items():
            require(sha(directory / name) == digest, "HISTORICAL_SOURCE_CHANGED:" + name)
            counts["files_hashed"] += 1
        records = [read(directory / name) for name in session["workers"]]
        require(all(x["status"] == "COLLECTED" for x in records), "HISTORY_NOT_COLLECTED")
        for row in (x for x in policy["screening"] if x["session"] == session["id"]):
            b, d = row["config"]["batch"], row["config"]["width"]
            workers = sorted((w for w in records if (w["batch"], w["width"]) == (b, d)), key=lambda w: w["round"])
            require([w["round"] for w in workers] == [1, 2, 3], "HISTORY_ROUNDS_CHANGED")
            for arm, field in (("original_cuda", "original_us"), (row["arm"], "arm_us")):
                medians = [statistics.median(w["samples_ms_per_call"][arm][row["execution_policy"]]) * 1000 for w in workers]
                require(medians == row[field], "HISTORY_RAW_MEDIANS_CHANGED")
            require(not s.meets_promotion(row["original_us"], row["arm_us"]), "UNEXPECTED_PROMOTION")
            counts["screened_entries"] += 1
        counts["sessions"] += 1
        counts["workers"] += len(records)
    require(counts["sessions"] == 4 and counts["workers"] == 48 and counts["screened_entries"] == 64,
            "HISTORY_MATRIX_INCOMPLETE")
    return dict(status="HISTORICAL_ONLY_GATE_PASS", policy_fingerprint=s.POLICY_CANONICAL_SHA256, **counts)


def verify_snapshot(run, project=None):
    hashes = read(run / "source_hashes.json")
    require(all(name in hashes for name in NEW_FILES), "INCOMPLETE_NEW_SOURCE_SNAPSHOT")
    for name, digest in hashes.items():
        path = (run / name).resolve()
        require(path.is_relative_to(run.resolve()) and sha(path) == digest, "SNAPSHOT_CHANGED:" + name)
        if project is not None and name.startswith("original_source/"):
            require(sha(project / name.removeprefix("original_source/")) == digest, "LIVE_ORIGINAL_CHANGED:" + name)
    policy = s.load_policy(run / POLICY_NAME)
    for name, digest in policy["source_pins"].items():
        require(hashes.get(name) == digest and sha(run / name) == digest, "PINNED_ENTRY_CHANGED:" + name)
    return policy


def snapshot(project, run):
    policy = s.load_policy()
    gate = historical_gate(HERE, policy)
    accepted = HERE / "logs" / ACCEPTED35
    hashes = read(accepted / "source_hashes.json")
    for name, digest in hashes.items():
        require(sha(accepted / name) == digest, "ACCEPTED35_SOURCE_CHANGED:" + name)
        destination = run / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(accepted / name, destination)
    run27 = HERE / "logs" / policy["historical_sessions"][0]["run"]
    shutil.copyfile(run27 / "triton_store_candidate.py", run / "triton_store_candidate.py")
    for name in NEW_FILES:
        shutil.copyfile(HERE / name, run / name)
    write(run / "historical-gate.json", gate)
    write(run / "original-support-review.json", dict(
        status="SOURCE_SUPPORTED_PENDING_NEW_GPU_CALLS",
        basis="Pinned CUDA matches dynamic L and shared source/cache row strides; grid is ceil(L/4) with warp_id<L guard. D64/128/192 FP16 byte widths previously passed35; these B7/64/257 cases are new.",
        scope="Source review is not a new GPU pass; first and reset-Graph checks gate every measurement."))
    write(run / "source_hashes.json", {x.relative_to(run).as_posix(): sha(x) for x in run.rglob("*") if x.is_file()})
    verify_snapshot(run, project)


def tensor_record(x):
    return dict(shape=list(x.shape), stride=list(x.stride()), dtype=str(x.dtype), device=str(x.device), data_ptr=x.data_ptr())


def validate_decisions(result, policy):
    cfg = configuration(result["batch"], result["width"])
    for mode in POLICIES:
        expected = s.context_for_config(policy, cfg, mode)
        require(result["selection_contexts"][mode] == expected, "ACTUAL_SELECTION_CONTEXT_CHANGED")
        require(result["decisions"][mode] == s.select(policy, expected), "DECISION_OR_PROMOTION_CHANGED")
    require(result["actual_entries"] == {
        "original_cuda": dict(entry="upstream store_cache", initial_calls=1),
        "selected": dict(entry="upstream store_cache", initial_calls=1, same_callable_as_original=True)},
        "ACTUAL_ENTRY_CHANGED")


def summarize(results, policy):
    expected = {(rr, b, d) for rr in (1, 2, 3) for b, d in CASES}
    keyed = {}
    for result in results:
        key = (result["round"], result["batch"], result["width"])
        require(key in expected and key not in keyed and result["status"] == "COLLECTED", "WORKER_ID_OR_STATUS")
        require(result["config"] == configuration(result["batch"], result["width"]), "WORKER_CONFIG_CHANGED")
        require(result["calls_per_interval"] == 100 and result["correctness_checks"] == 127 and result["policy"] == make_plan(), "WORKER_POLICY_CHANGED")
        validate_decisions(result, policy)
        records = result["measurements"]
        order = [(i, a, p) for i in range(30) for a, p in measurement_order(i, result["round"])]
        require([(x["sample"], x["arm"], x["policy"]) for x in records] == order, "MEASUREMENT_ORDER_CHANGED")
        require(set(result["samples_ms_per_call"]) == set(ARMS), "ARM_SET_CHANGED")
        for arm in ARMS:
            require(set(result["samples_ms_per_call"][arm]) == set(POLICIES), "POLICY_SET_CHANGED")
            for mode in POLICIES:
                values = result["samples_ms_per_call"][arm][mode]
                require(len(values) == 30 and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values), "INVALID_SAMPLES")
                require(values == [x["ms_per_call"] for x in records if (x["arm"], x["policy"]) == (arm, mode)], "RAW_SAMPLES_CHANGED")
        keyed[key] = result
    require(set(keyed) == expected, "INCOMPLETE_HOLDOUT_MATRIX")
    summary = []
    for b, d in CASES:
        row = dict(batch=b, width=d, config=configuration(b, d), selection="original_cuda", interpretation="A/A variation; not candidate speedup")
        for mode in POLICIES:
            entry = {}
            for arm in ARMS:
                medians = [statistics.median(keyed[(rr, b, d)]["samples_ms_per_call"][arm][mode]) * 1000 for rr in (1, 2, 3)]
                entry[arm] = dict(round_medians_us=medians, representative_us=statistics.median(medians))
            orig, selected = (entry[a] for a in ARMS)
            paired = [x / y for x, y in zip(orig["round_medians_us"], selected["round_medians_us"])]
            ratio = orig["representative_us"] / selected["representative_us"]
            entry.update(original_over_selected_representative=ratio, paired_original_over_selected=paired,
                         median_paired_ratio=statistics.median(paired), selected_time_change_pct=100 * (1 / ratio - 1),
                         note="same implementation; numerical differences are not an optimization result")
            row[mode] = entry
        summary.append(row)
    return summary


def worker(project, run, directory, b, d, round_id):
    directory.mkdir(parents=True)
    result = dict(status="STARTED", batch=b, width=d, round=round_id, config=configuration(b, d),
                  calls_per_interval=100, correctness_checks=0, policy=make_plan(), measurements=[],
                  samples_ms_per_call={a: {p: [] for p in POLICIES} for a in ARMS})
    write(directory / "result.json", result)
    try:
        require(round_id in (1, 2, 3), "INVALID_ROUND")
        policy = verify_snapshot(run, project)
        timer = module(run / "25-compare-kv-v1.py", "selector37_timer")
        eager_timer = timer.load_reviewed_timer(run / "learner-timing.py")
        import torch
        import triton
        from minisgl.kernel.store import store_cache
        require(Path(inspect.getsourcefile(store_cache)).resolve() == (project / "python/minisgl/kernel/store.py").resolve(), "WRONG_ORIGINAL_IMPORT")
        environment = dict(gpu=torch.cuda.get_device_name(0), compute_capability=list(torch.cuda.get_device_capability(0)),
                           torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__, commit=COMMIT)
        require(environment == policy["environment"], "ENVIRONMENT_CHANGED")
        support = module(run / "kv_contract_support.py", "selector37_contract")
        cfg = configuration(b, d)
        before = fixture(cfg)
        np.savez_compressed(directory / "before.npz", **before)
        buffers, inputs = timer.c.gpu_packet(torch, cfg, before)
        kc, vc, indices, k, v = inputs
        support.validate_metadata(*inputs)
        require(indices.shape == (b,), "INDEX_COUNT")
        support.validate_index_values(indices, b, 512)
        require(indices.cpu().tolist() == cfg["rows"], "INDICES_CHANGED")
        tensor_info = [tensor_record(x) for x in inputs]
        write(directory / "tensors.json", tensor_info)
        result.update(environment=environment, telemetry_before=timer.telemetry(), selection_contexts={}, decisions={})
        for mode in POLICIES:
            ctx = s.context_for_config(policy, cfg, mode, environment)
            for name, tensor in zip(("k_cache", "v_cache", "indices", "k", "v"), inputs):
                ctx[name + "_shape"] = list(tensor.shape)
                ctx[name + "_stride"] = list(tensor.stride())
            ctx["device"] = str(k.device)
            result["selection_contexts"][mode] = ctx
            result["decisions"][mode] = s.select(policy, ctx)
            require(result["decisions"][mode]["selected"] == "original_cuda", "UNEXPECTED_CANDIDATE_SELECTED")
        pool = buffers["cache"]
        initial = torch.from_numpy(before["cache"].copy()).cuda()
        expected = {n: torch.from_numpy(before["expected_cache" if n == "cache" else n].copy()).cuda() for n in buffers}
        def save(stage):
            np.savez_compressed(directory / (stage + ".npz"), **{n: x.cpu().numpy().copy() for n, x in buffers.items()})
        def check(stage, retain=False):
            try:
                for n, x in buffers.items():
                    require(torch.equal(x, expected[n]), "FULL_BACKING_MISMATCH:" + stage + ":" + n)
                result["correctness_checks"] += 1
            except BaseException:
                try:
                    save("failed-" + stage)
                except Exception as capture_error:
                    result.setdefault("failure_capture_errors", []).append(dict(stage=stage, error=str(capture_error)))
                raise
            if retain:
                save(stage)
        def original_call():
            store_cache(kc, vc, indices, k, v)
        calls = dict(original_cuda=original_call, selected=original_call)
        require(calls["selected"] is calls["original_cuda"], "SELECTED_CALLABLE_CHANGED")
        graphs = {}
        setup = list(ARMS) if round_id % 2 else list(reversed(ARMS))
        result.update(setup_order=setup, actual_entries={})
        for arm in setup:
            call = calls[arm]
            pool.copy_(initial)
            call()
            result["actual_entries"][arm] = dict(entry="upstream store_cache", initial_calls=1)
            if arm == "selected":
                result["actual_entries"][arm]["same_callable_as_original"] = True
            torch.cuda.synchronize()
            check(arm + "-initial", True)
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
            pool.copy_(initial)
            graph.replay()
            torch.cuda.synchronize()
            check(arm + "-graph-replay", True)
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            check(arm + "-warmup")
            graphs[arm] = graph
        validate_decisions(result, policy)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record(); end.record(); end.synchronize()
        print(f"DEFAULT_ENTRY_CHECKED B={b} D={d} round={round_id}: selected=original_cuda; A/A", flush=True)
        for sample in range(30):
            for arm, mode in measurement_order(sample, round_id):
                value = float(eager_timer(calls[arm], start, end) if mode == "eager" else timer.measure_graph(graphs[arm], start, end))
                require(math.isfinite(value) and value > 0, "INVALID_TIMING")
                result["samples_ms_per_call"][arm][mode].append(value)
                result["measurements"].append(dict(sample=sample, arm=arm, policy=mode, ms_per_call=value))
                check(f"sample-{sample}-{arm}-{mode}")
            write(directory / "result.json", result)
        check("final", True)
        require(result["correctness_checks"] == 127, "CHECK_COUNT_CHANGED")
        result["telemetry_after"] = timer.telemetry()
        verify_snapshot(run, project)
        result.update(status="COLLECTED", selection_validated=True, candidate_speedup_claim=False)
    except BaseException as error:
        result.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write(directory / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--shared-results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    require(not sys.flags.optimize, "ASSERTS_MUST_REMAIN_ENABLED")
    if args.dry_run:
        policy = s.load_policy()
        print(json.dumps(dict(plan=make_plan(), decisions=[s.select(policy, s.context_for_config(policy, configuration(b, d), p)) for b, d in CASES for p in POLICIES]), indent=2))
        return
    if args.project is None:
        parser.error("--project is required")
    project = args.project.resolve()
    if args.worker_dir:
        if None in (args.snapshot, args.batch, args.width, args.round):
            parser.error("worker arguments are incomplete")
        worker(project, args.snapshot, args.worker_dir, args.batch, args.width, args.round)
        return
    if args.shared_results is None:
        parser.error("--shared-results is required")
    def git(*words):
        return subprocess.check_output(["git", "-C", str(project), *words], text=True).strip()
    parent = project / "experiments/learning/m07-selector-validation"
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run-%Y%m%d-%H%M%S-"), dir=parent))
    state = dict(status="PREPARING", commit=COMMIT, plan=make_plan(), results=[],
                 ownership="coach-written frozen policy, selector, fixture, checks and timing; user authorized autonomous implementation",
                 interpretation="both measured arms selected original; no candidate performance or learner mastery claim")
    print(f"RUN_DIRECTORY={run}\nSCOPE=three independent configurations; default original A/A; no candidate promotion", flush=True)
    try:
        require(git("rev-parse", "HEAD") == COMMIT and not git("status", "--porcelain", "--untracked-files=no"), "TRACKED_UPSTREAM_CHANGED")
        snapshot(project, run)
        policy = s.load_policy(run / POLICY_NAME)
        write(run / "plan.json", state["plan"])
        for item in state["plan"]["schedule"]:
            rr, b, d = (item[x] for x in ("round", "batch", "width"))
            directory = run / f"round-{rr}-B{b}-D{d}"
            command = [sys.executable, "-B", str(run / "37-validate-kv-selector.py"), "--project", str(project), "--snapshot", str(run), "--worker-dir", str(directory), "--batch", str(b), "--width", str(d), "--round", str(rr)]
            write(run / (directory.name + "-command.json"), command)
            state.update(status="RUNNING", current=item)
            write(run / "status.json", state)
            print(f"RUN round={rr}/3 B={b} D={d} original/selected(original)", flush=True)
            console = run / (directory.name + ".log")
            with console.open("x", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                print(console.read_text(encoding="utf-8"), flush=True)
                raise RuntimeError("SELECTOR_WORKER_FAILED")
            result = read(directory / "result.json")
            require(result["status"] == "COLLECTED" and all(result[k] == v for k, v in item.items()), "WORKER_RESULT_CHANGED")
            state["results"].append(result)
            write(run / "partial-results.json", state["results"])
        require(git("rev-parse", "HEAD") == COMMIT and not git("status", "--porcelain", "--untracked-files=no"), "TRACKED_UPSTREAM_CHANGED_DURING_RUN")
        summary = summarize(state["results"], policy)
        state.update(status="COLLECTION_COMPLETE", current=None, summary=summary)
        write(run / "summary.json", dict(status=state["status"], plan=state["plan"], summary=summary))
        for row in summary:
            g = row["graph"]
            print(f"GRAPH_AA B={row['batch']} D={row['width']} original={g['original_cuda']['representative_us']:.6f} us selected={g['selected']['representative_us']:.6f} us selected_entry=original_cuda", flush=True)
    except BaseException as error:
        state.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        write(run / "status.json", state)
        try:
            args.shared_results.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run, args.shared_results / run.name)
            print(f"SHARED_ARTIFACTS={args.shared_results / run.name}", flush=True)
        except OSError as export_error:
            print(f"SHARED_COPY_FAILED={export_error}; original evidence retained at {run}", flush=True)
            if state["status"] == "COLLECTION_COMPLETE":
                raise
    print("SELECTOR_VALIDATION_COLLECTION_COMPLETE: nine processes;1080 interval averages; review selection and full backing")
    print("LIMIT: default original A/A; no candidate speedup, generalization, model or AMD claim; frozen policy unchanged")


if __name__ == "__main__":
    main()
