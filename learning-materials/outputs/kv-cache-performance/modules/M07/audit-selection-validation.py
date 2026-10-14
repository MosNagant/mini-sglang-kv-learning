"""Rebuild the independent CPU audit/report for the frozen37 actual run.

Reads archived JSON, AST and NPZ only; never imports a runner, Torch or Triton.
Does not write the archived run. Generated Markdown/JSON stay in --out.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN_REL = "logs/m07-selector-validation-results/run-20261014-191004-v7b1lswv"
CASES = ((7, 64), (64, 128), (257, 192))
ARMS = ("original_cuda", "selected")
MODES = ("graph", "eager")
COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
POLICY_FINGERPRINT = "8a6f451a0abd47280e7139c3181bdb081504bad0e68ea28c2bb282f63c0028cd"
PINS = {
    "37-kv-selection-policy.json": "75e447b0754a0e4f4cd07bd2808781a12b6b97eec097b472cedfdb432ef8dae6",
    "37-kv-selector.py": "e90a0f1c4e76884bd01b2e5085b09f2d8841c3e1d3d2c8d235486095517aa1e9",
    "37-validate-kv-selector.py": "ad5bdcc2a97d984a9964c64f4355642f217bf0d6b9b7cf711a48595ad352f255",
    "37-validate-kv-selector.sh": "58b1c93a10f4e7505ca9310f2b55a57b373b2600b7c95317a7caa80c6db20859",
    "37-selector-cpu-test.py": "cb580ef908929b3ce09797c8671941bf5305275a51044e2bc123970d337743b0",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def equal(saved, expected, path="root"):
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and set(saved) == set(expected), "OBJECT_KEYS:" + path)
        for k in expected:
            equal(saved[k], expected[k], path + "/" + k)
    elif isinstance(expected, list):
        require(isinstance(saved, list) and len(saved) == len(expected), "ARRAY_LENGTH:" + path)
        for i, (a, b) in enumerate(zip(saved, expected)):
            equal(a, b, path + "/" + str(i))
    elif isinstance(expected, float):
        require(type(saved) in (float, int) and math.isfinite(saved) and
                math.isclose(saved, expected, rel_tol=1e-12, abs_tol=1e-12), "NUMBER:" + path)
    else:
        require(saved == expected and type(saved) is type(expected), "VALUE:" + path)


def config(b, d):
    return dict(batch=b, width=d, capacity=512, layout="interleaved" if d == 64 else "separate",
                gap=0 if d == 64 else 64, guard=64 if d == 192 else 0,
                rows=[(53 * i + 19) % 512 for i in range(b)])


def views(array, layout):
    return (array[:, 0, :], array[:, 1, :]) if layout == "interleaved" else (array[0], array[1])


def independent_before(cfg):
    """Independent NumPy reconstruction, not an import/call of fixture37."""
    b, d, c = (cfg[k] for k in ("batch", "width", "capacity"))
    pitch = d + cfg["guard"]
    shape = (c, 2, pitch) if cfg["layout"] == "interleaved" else (2, c, pitch)
    cache = np.full(shape, -1, dtype=np.float16)
    for view in views(cache, cfg["layout"]):
        view[:, d:] = -7
    k = np.full((b, d + cfg["gap"]), -9, dtype=np.float16)
    v = k.copy()
    values = np.random.default_rng(20261014).integers(0, 1024, (b, d), dtype=np.int32).astype(np.float16)
    values[:, 0] = np.arange(b, dtype=np.float16)
    k[:, :d] = values
    v[:, :d] = values + np.float16(1024)
    indices = np.asarray(cfg["rows"], dtype=np.int32)
    expected = cache.copy()
    ek, ev = views(expected, cfg["layout"])
    for pid, pos in enumerate(indices):
        ek[pos, :d], ev[pos, :d] = k[pid, :d], v[pid, :d]
    return dict(cache=cache, k=k, v=v, indices=indices, expected_cache=expected)


def check_npz(path, expected):
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == set(expected), "NPZ_KEYS:" + str(path))
        for name, ref in expected.items():
            got = arrays[name]
            require(got.dtype == ref.dtype and got.shape == ref.shape and np.array_equal(got, ref),
                    "NPZ_VALUE:" + str(path) + ":" + name)
    return dict(path=path.name, sha256=sha(path), arrays=len(expected), elements=sum(int(x.size) for x in expected.values()))


def order(sample, rr):
    ring = [(arm, mode) for mode in MODES for arm in ARMS]
    offset = (sample + rr - 1) % 4
    result = ring[offset:] + ring[:offset]
    return result[::-1] if (sample // 4 + rr - 1) % 2 else result


def expected_context(policy, cfg, mode):
    b, d, c = (cfg[x] for x in ("batch", "width", "capacity"))
    src = d + cfg["gap"]
    dst = (d + cfg["guard"]) * (2 if cfg["layout"] == "interleaved" else 1)
    return dict(environment=policy["environment"], execution_policy=mode,
                address_policy="reused_addresses_no_flush", dtype="float16", index_dtype="int32", device="cuda:0",
                batch=b, width=d, capacity=c, layout=cfg["layout"], input_gap=cfg["gap"], cache_guard=cfg["guard"],
                k_shape=[b, d], v_shape=[b, d], k_cache_shape=[c, d], v_cache_shape=[c, d],
                indices_shape=[b], indices_stride=[1], k_stride=[src, 1], v_stride=[src, 1],
                k_cache_stride=[dst, 1], v_cache_stride=[dst, 1], metadata_and_aliases_validated=True,
                unique_in_range_indices_validated=True, source_hashes=policy["source_pins"])


def audit_source(run):
    tree = ast.parse((run / "37-validate-kv-selector.py").read_text(encoding="utf-8"))
    worker = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "worker")
    original = next(x for x in ast.walk(worker) if isinstance(x, ast.FunctionDef) and x.name == "original_call")
    require(ast.dump(original.body[0]) == ast.dump(ast.parse("store_cache(kc, vc, indices, k, v)").body[0])
            and len(original.body) == 1, "ORIGINAL_CLOSURE_CHANGED")
    assignments = [x for x in ast.walk(worker) if isinstance(x, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "calls" for t in x.targets)]
    require(len(assignments) == 1 and ast.dump(assignments[0].value) ==
            ast.dump(ast.parse("dict(original_cuda=original_call, selected=original_call)", mode="eval").body),
            "CALLABLE_ALIAS_CHANGED")
    require(not any(isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id == "calls"
                    and isinstance(x.ctx, ast.Store) for x in ast.walk(worker)), "CALLABLE_REASSIGNED")
    selection = [x for x in ast.walk(worker) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                 and isinstance(x.func.value, ast.Name) and x.func.value.id == "s" and x.func.attr == "select"]
    measurement = next(x for x in ast.walk(worker) if isinstance(x, ast.For)
                       and isinstance(x.target, ast.Name) and x.target.id == "sample")
    require(len(selection) == 1 and selection[0].lineno < assignments[0].lineno < measurement.lineno,
            "SELECTOR_ENTERED_TIMING")
    require(not any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "select"
                    for x in ast.walk(measurement)), "SELECTOR_CALLED_IN_MEASUREMENT")
    learner = ast.parse((run / "learner-timing.py").read_text(encoding="utf-8"))
    timer = next(x for x in learner.body if isinstance(x, ast.FunctionDef) and x.name == "measure_once")
    reference = ast.parse("def measure_once(call, start, end):\n start.record()\n for _ in range(100):\n  call()\n end.record()\n end.synchronize()\n return start.elapsed_time(end)/100\n").body[0]
    require(ast.dump(timer) == ast.dump(reference), "EAGER_TIMER_CHANGED")
    helper = ast.parse((run / "25-compare-kv-v1.py").read_text(encoding="utf-8"))
    graph = next(x for x in helper.body if isinstance(x, ast.FunctionDef) and x.name == "measure_graph")
    reference = ast.parse("def measure_graph(graph, start, end):\n start.record()\n graph.replay()\n end.record()\n end.synchronize()\n return start.elapsed_time(end)/100\n").body[0]
    require(ast.dump(graph) == ast.dump(reference), "GRAPH_TIMER_CHANGED")
    return dict(same_closure_ast_confirmed=True, selector_outside_capture_and_measurement=True,
                original_closure_line=original.lineno, calls_alias_line=assignments[0].lineno,
                selection_line=selection[0].lineno, measurements_line=measurement.lineno,
                original_source_sha256=sha(run / "original_source/python/minisgl/kernel/store.py"),
                timer_boundary="CUDA Event interval /100; end synchronize after end.record; same selected callable")


def audit(base=BASE):
    run = base / RUN_REL
    state, saved_summary, plan, manifest = (read(run / x) for x in ("status.json", "summary.json", "plan.json", "source_hashes.json"))
    require(state["status"] == saved_summary["status"] == "COLLECTION_COMPLETE" and state["current"] is None, "RUN_NOT_COMPLETE")
    require(state["commit"] == COMMIT and not any(k in state for k in ("error", "error_type")), "RUN_ERROR_OR_COMMIT")
    require(state["plan"] == saved_summary["plan"] == plan and state["summary"] == saved_summary["summary"], "ROOT_COPIES_DIFFER")
    for name, digest in manifest.items():
        path = (run / name).resolve()
        require(path.is_relative_to(run.resolve()) and sha(path) == digest, "SOURCE_HASH:" + name)
    for name, digest in PINS.items():
        require(manifest.get(name) == digest, "PRE_RUN_SOURCE_PIN:" + name)
    policy = read(run / "37-kv-selection-policy.json")
    require(fingerprint(policy) == POLICY_FINGERPRINT and policy["default"] == "original_cuda"
            and policy["promoted_candidates"] == [], "FROZEN_POLICY_CHANGED")
    require(policy["promotion_rule"]["per_round_max_candidate_latency_ratio"] == .97
            and policy["promotion_rule"]["new36_or37_results_may_change_policy"] is False, "FROZEN_RULE_CHANGED")
    for name, digest in policy["source_pins"].items():
        require(manifest[name] == digest, "CONTEXT_SOURCE_PIN:" + name)
    historical_files = 0
    require([x["id"] for x in policy["historical_sessions"]] == ["27", "30", "32", "35"], "HISTORY_SESSIONS_CHANGED")
    for session in policy["historical_sessions"]:
        directory = base / "logs" / session["run"]
        for name, digest in {**session["files"], **session["workers"]}.items():
            require(sha(directory / name) == digest, "HISTORY_CHANGED:" + session["id"] + "/" + name)
            historical_files += 1
    require(len(policy["screening"]) == 64, "HISTORY_SCREENING_COUNT")
    for row in policy["screening"]:
        require(not all(x <= .97 * y for x, y in zip(row["arm_us"], row["original_us"])), "UNEXPECTED_HISTORICAL_PROMOTION")
        require(row["promoted"] is False and row["config"]["batch"] not in [7, 64, 257], "HOLDOUT_LEAKAGE")
    gate = read(run / "historical-gate.json")
    require(gate == dict(status="HISTORICAL_ONLY_GATE_PASS", policy_fingerprint=POLICY_FINGERPRINT,
                         sessions=4, workers=48, files_hashed=60, screened_entries=64, promoted=0), "HISTORY_GATE_CHANGED")
    source_review = audit_source(run)
    require(plan["cases"] == [list(x) for x in CASES] and plan["configurations"] == [config(*x) for x in CASES], "PLAN_CASES")
    for key, expected in dict(capacity=512, dtype="float16", index_dtype="int32", input_seed=20261014,
                              rounds=3, fresh_processes=9, intervals_per_arm_policy=30, calls_per_interval=100,
                              interval_averages=1080, recorded_checks_per_worker=127, saved_after_stages_per_worker=5,
                              eager_warmup_calls=20, graph_warmup_replays=5, primary="graph", reference="eager",
                              policy_fingerprint=POLICY_FINGERPRINT).items():
        require(plan[key] == expected, "PLAN_FIELD:" + key)
    rng, schedule = random.Random(137), []
    for rr in (1, 2, 3):
        ordered = list(CASES); rng.shuffle(ordered)
        schedule.extend(dict(round=rr, batch=b, width=d) for b, d in ordered)
    require(plan["schedule"] == schedule and len(state["results"]) == 9, "SCHEDULE_CHANGED")
    require(read(run / "partial-results.json") == state["results"], "PARTIAL_RESULTS_DIFFER")
    require(len(list(run.glob("round-*/result.json"))) == 9 and not list(run.rglob("error.txt"))
            and not list(run.rglob("failed-*.npz")), "WORKER_COUNT_OR_ERROR")
    counts = Counter(workers=0, interval_averages=0, recorded_checks=0, decisions=0, before_npz=0,
                     before_arrays=0, before_elements=0, after_npz=0, after_arrays=0, after_elements=0)
    worker_reports, worker_data = [], {}
    for item, root_copy in zip(schedule, state["results"]):
        rr, b, d = (item[k] for k in ("round", "batch", "width"))
        cfg = config(b, d); directory = run / f"round-{rr}-B{b}-D{d}"
        w = read(directory / "result.json")
        require(w == root_copy and w["status"] == "COLLECTED" and w["config"] == cfg, "WORKER_STATE")
        require(all(w[k] == v for k, v in item.items()) and w["policy"] == plan, "WORKER_ID_OR_POLICY")
        require(w["environment"] == policy["environment"] and w["correctness_checks"] == 127
                and w["calls_per_interval"] == 100 and w["selection_validated"] is True
                and w["candidate_speedup_claim"] is False, "WORKER_CONTRACT")
        require(not any(k in w for k in ("error", "error_type", "failure_capture_errors")), "WORKER_ERROR")
        for mode in MODES:
            ctx = expected_context(policy, cfg, mode)
            require(w["selection_contexts"][mode] == ctx, "CONTEXT_CHANGED")
            expected = dict(policy_id=policy["id"], policy_fingerprint=POLICY_FINGERPRINT,
                            context_fingerprint=fingerprint(ctx), request="auto", selected="original_cuda",
                            reason="no_historical_candidate_passed_frozen_gate", automatic_promotion=False,
                            performance_claim=False, source=policy["source_pins"]["original_source/python/minisgl/kernel/store.py"])
            require(w["decisions"][mode] == expected, "DECISION_CHANGED")
            counts["decisions"] += 1
        require(w["actual_entries"] == {"original_cuda": dict(entry="upstream store_cache", initial_calls=1),
                                        "selected": dict(entry="upstream store_cache", initial_calls=1, same_callable_as_original=True)}, "ENTRY_RECORD_CHANGED")
        require(w["setup_order"] == (list(ARMS) if rr % 2 else list(reversed(ARMS))), "SETUP_ORDER")
        tensors = read(directory / "tensors.json")
        src = d + cfg["gap"]; dst = (d + cfg["guard"]) * (2 if cfg["layout"] == "interleaved" else 1)
        require(len(tensors) == 5, "TENSOR_COUNT")
        for i, tensor in enumerate(tensors):
            shape, stride, dtype = ([512, d], [dst, 1], "torch.float16") if i < 2 else (([b], [1], "torch.int32") if i == 2 else ([b, d], [src, 1], "torch.float16"))
            require(tensor["shape"] == shape and tensor["stride"] == stride and tensor["dtype"] == dtype
                    and tensor["device"] == "cuda:0" and tensor["data_ptr"] > 0, "ACTUAL_TENSOR_METADATA")
        require(tensors[1]["data_ptr"] - tensors[0]["data_ptr"] == (d if d == 64 else 512 * (d + cfg["guard"])) * 2,
                "CACHE_LAYOUT_POINTER_DIFFERENCE")
        command = read(run / (directory.name + "-command.json"))
        require(command[1] == "-B" and command[2].endswith("/37-validate-kv-selector.py"), "WORKER_COMMAND")
        for flag, value in (("--batch", str(b)), ("--width", str(d)), ("--round", str(rr))):
            require(command[command.index(flag) + 1] == value, "WORKER_COMMAND_ARGUMENT")
        require(command[command.index("--worker-dir") + 1].endswith("/" + directory.name), "WORKER_COMMAND_DIRECTORY")
        expected_order = [(sample, a, p) for sample in range(30) for a, p in order(sample, rr)]
        require([(x["sample"], x["arm"], x["policy"]) for x in w["measurements"]] == expected_order, "RAW_ORDER")
        positions = Counter((a, p, pos) for sample in range(30) for pos, (a, p) in enumerate(order(sample, rr)))
        require(len(positions) == 16 and set(positions.values()) == {7, 8}, "POSITION_BALANCE")
        require(set(w["samples_ms_per_call"]) == set(ARMS), "SAMPLE_ARMS")
        for a in ARMS:
            require(set(w["samples_ms_per_call"][a]) == set(MODES), "SAMPLE_MODES")
            for mode in MODES:
                series = w["samples_ms_per_call"][a][mode]
                require(len(series) == 30 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in series), "SAMPLE_VALUES")
                require(series == [x["ms_per_call"] for x in w["measurements"] if x["arm"] == a and x["policy"] == mode], "RAW_SAMPLES_DIFFER")
                counts["interval_averages"] += len(series)
        before = independent_before(cfg)
        before_record = check_npz(directory / "before.npz", before)
        counts.update(before_npz=1, before_arrays=before_record["arrays"], before_elements=before_record["elements"])
        after = {n: before["expected_cache" if n == "cache" else n] for n in ("cache", "k", "v", "indices")}
        after_records = []
        stages = [a + "-" + phase for a in ARMS for phase in ("initial", "graph-replay")] + ["final"]
        require({p.name for p in directory.glob("*.npz")} == {"before.npz", *(x + ".npz" for x in stages)}, "NPZ_SET")
        for stage in stages:
            a = check_npz(directory / (stage + ".npz"), after)
            after_records.append(a)
            counts.update(after_npz=1, after_arrays=a["arrays"], after_elements=a["elements"])
        counts.update(workers=1, recorded_checks=w["correctness_checks"])
        worker_reports.append(dict(worker=directory.name, result_sha256=sha(directory / "result.json"),
                                   command_sha256=sha(run / (directory.name + "-command.json")),
                                   config=cfg, tensor_metadata=tensors, decisions=w["decisions"],
                                   before=before_record, after=after_records,
                                   samples_ms_per_call=w["samples_ms_per_call"], measurements=w["measurements"],
                                   telemetry_before=w["telemetry_before"], telemetry_after=w["telemetry_after"]))
        worker_data[(rr, b, d)] = w
    require(counts["workers"] == 9 and counts["interval_averages"] == 1080 and counts["decisions"] == 18, "TOTAL_COUNTS")
    recomputed, detailed = [], []
    for b, d in CASES:
        row = dict(batch=b, width=d, config=config(b, d), selection="original_cuda", interpretation="A/A variation; not candidate speedup")
        extra = dict(batch=b, width=d)
        for mode in MODES:
            entry = {}
            for arm in ARMS:
                medians = [statistics.median(worker_data[(rr, b, d)]["samples_ms_per_call"][arm][mode]) * 1000 for rr in (1, 2, 3)]
                entry[arm] = dict(round_medians_us=medians, representative_us=statistics.median(medians))
            orig, chosen = entry["original_cuda"], entry["selected"]
            ratios = [x / y for x, y in zip(orig["round_medians_us"], chosen["round_medians_us"])]
            representative = orig["representative_us"] / chosen["representative_us"]
            entry.update(original_over_selected_representative=representative, paired_original_over_selected=ratios,
                         median_paired_ratio=statistics.median(ratios), selected_time_change_pct=100 * (1 / representative - 1),
                         note="same implementation; numerical differences are not an optimization result")
            row[mode] = entry
            extra[mode] = dict(**entry, paired_selected_time_change_pct=[100 * (1 / x - 1) for x in ratios],
                              numerical_lower_equal_higher=[sum(x > 1 for x in ratios), sum(x == 1 for x in ratios), sum(x < 1 for x in ratios)])
        recomputed.append(row); detailed.append(extra)
    equal(saved_summary["summary"], recomputed)
    counts["total_saved_arrays"] = counts["before_arrays"] + counts["after_arrays"]
    counts["total_saved_elements"] = counts["before_elements"] + counts["after_elements"]
    require("torch" not in sys.modules and "triton" not in sys.modules, "UNEXPECTED_GPU_RUNTIME_IMPORT")
    return dict(status="INDEPENDENT_SAVED_EVIDENCE_AUDIT_PASS", report_date="2026-10-14", run=RUN_REL,
                blocking_findings=[], source_manifest_entries=len(manifest), source_manifest=manifest,
                source_review=source_review, frozen_policy_fingerprint=POLICY_FINGERPRINT,
                historical_files_hashed=historical_files, historical_screened_entries=64, promoted_candidates=[],
                source_archive_hashes={name: sha(run / name) for name in ("status.json", "summary.json", "source_hashes.json", "plan.json")},
                environment=policy["environment"], counts=dict(counts), results=detailed, workers=worker_reports,
                ownership=state["ownership"], conclusion="Frozen auto decision correctly uses original for all18 decisions and all saved arrays match an independent full-backing oracle. A/A differences are observed variability, not Triton speedup or selector latency.",
                limits=["Only three holdout configurations; not arbitrary inputs, model inference or AMD.",
                        "127 checks per worker are runtime counts; offline audit checks all five saved after stages, not127 independent saved snapshots.",
                        "1080 interval averages comprise100 calls each; they are not108000 independent experimental rounds.",
                        "No candidate is measured or promoted, and policy is not updated using36/37 results.",
                        "This audit verifies saved source, recorded calls and arrays; it does not reconstruct GPU hardware execution traces or utilization.",
                        "Three paired process rounds and exact numerical differences do not establish statistical significance."])


def markdown(report):
    n = report["counts"]
    lines = ["# M07 保守选版与独立配置验证：37号实际结果", "",
             "**默认选择原版的验证通过；没有候选晋升。** 2026-10-14实际运行包含9个新进程、1080个区间均值。两臂都调用同一个原版CUDA入口，测到的是A/A波动，不是Triton收益，也不是选择器执行耗时。", "",
             "## 选择依据与冻结边界", "",
             "策略只读取已归档27/30/32/35的数据；按同一会话、配置、执行政策逐轮比较，要求3轮均满足 `candidate_us / original_us <= 0.97`。这是耗时至少低3%，等价速度比至少 `1/0.97`，不是 `1.03`。64条历史筛选记录包含每会话两条Triton臂及Graph/Eager，均未满足该门槛。该筛选规则是对历史结果作出的保守工程选择；在新36/37结果出现前冻结，不回称旧实验事前已注册该阈值。", "",
             "受支持的完整上下文默认返回original_cuda；未知设备/环境、dtype、布局、stride、来源版本或缺少合法性校验会明确拒绝。显式research选项仅提供研究入口名，不能自动晋升；本次未调用任何研究候选。B7、B64、B257未出现在选版历史配置中。", "",
             "[冻结策略](../../37-kv-selection-policy.json) · [纯CPU选择器](../../37-kv-selector.py) · [实际run](../../" + report["run"] + "/status.json)", "",
             "## 实际验证范围", "",
             "| B / D | 缓存布局 | 输入gap / 缓存guard | 源 / 目标行距（元素） |", "| --- | --- | --- | --- |",
             "| 7 / 64 | interleaved | 0 / 0 | 64 / 128 |",
             "| 64 / 128 | separate | 64 / 0 | 192 / 128 |",
             "| 257 / 192 | separate | 64 / 64 | 256 / 256 |", "",
             "共同C512、FP16、唯一int32索引 `(53*i+19)%512`，输入seed20261014。每配置3个独立进程；每臂每政策30个区间，区间含100次调用。Graph和Eager分开；选择、分配、校验、首次编译、预热与存档均在计时外，同一进程两臂复用同一批地址，无cache flush。", "",
             "源码审查确认 `calls = dict(original_cuda=original_call, selected=original_call)`，两者绑定同一闭包；闭包仅调用原版store_cache。记录中的same_callable声明与冻结源码一致。选择发生在闭包构造和Graph捕获之前，计时内部没有选择器调用。", "",
             "## 同场A/A耗时", "",
             "单位µs/次。代表耗时＝3个进程中位数的中位数；每进程中位数来自30个区间均值。变化为selected相对original的代表耗时变化，正号表示本次记录更慢。", "",
             "| 政策 | B / D | original | selected（实际原版） | 代表比 original/selected | selected耗时变化 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for mode in MODES:
        for row in report["results"]:
            e = row[mode]
            lines.append(f"| {mode} | {row['batch']} / {row['width']} | {e['original_cuda']['representative_us']:.6f} | {e['selected']['representative_us']:.6f} | {e['original_over_selected_representative']:.6f} | {e['selected_time_change_pct']:+.3f}% |")
    lines += ["", "| 政策 | B / D | 第1 / 2 / 3轮 original/selected | 配对中位比 | selected数值更低 / 相等 / 更高 |",
              "| --- | --- | --- | ---: | --- |"]
    for mode in MODES:
        for row in report["results"]:
            e = row[mode]
            ratios = " / ".join(f"{x:.6f}" for x in e["paired_original_over_selected"])
            directions = " / ".join(str(x) for x in e["numerical_lower_equal_higher"])
            lines.append(f"| {mode} | {row['batch']} / {row['width']} | {ratios} | {e['median_paired_ratio']:.6f} | {directions} |")
    lines += ["", "Graph代表变化依次为+2.436%、+0.040%、−0.821%；既有正也有负。B64/D128第三轮比值达到1.180909，但两臂代码相同，不能把这一个数说成优化18.09%或选择器提速。单轮与代表值可不同，代表比也不等于三轮配对比的中位数。本次没有使用统计显著性检验，不给差异指定未经测量的硬件原因。", "",
              "## 独立证据核验", "",
              f"- {report['source_manifest_entries']}个源码快照条目SHA通过，5份37号文件与GPU运行前冻结哈希一致；25个选择上下文source pins一致。",
              "- 历史依据的60个文件SHA通过；18个实际决策的完整上下文及指纹全部复算，均选择original_cuda，策略未使用36/37数据更新。",
              "- 9份worker原始JSON、根汇总副本、执行命令、实际tensor形状/stride/布局指针差与测量顺序一致；1080个区间原始值均为有限正值，所有代表值及配对比独立复算相符。",
              f"- 调用前9份NPZ包含{n['before_arrays']}个数组、{n['before_elements']:,}个元素；调用后45份NPZ包含{n['after_arrays']}个数组、{n['after_elements']:,}个元素。合计{n['total_saved_arrays']}个数组、{n['total_saved_elements']:,}个元素与独立CPU期望完全相符。",
              "- 保存的after包括每臂首次调用、完整缓存重置后的Graph replay及final。核对全部有效目标、未命中缓存、guard、输入gap、只读K/V和indices；不是只看前几列。",
              "- 运行时每worker记录127次全backing检查，共1143次；离线核对的是实际保存的5个after阶段，不冒称保存并复验了1143份独立结果。", "",
              "[机器审查与原始样本](selection-validation.json) · [可重生成审查脚本](audit-selection-validation.py) · [CPU控制归档复跑](../../reviews/20261014-m07-selector-cpu.json)", "",
              "CPU控制记录于本次补档时重新运行同哈希脚本，65个负控通过，保留实际stdout与复跑说明；它不是第一次未保存stdout的原件，也没有新增GPU运行。", "",
              "## 能说明什么，以及不能说明什么", "",
              "本次支持：冻结决策按预定规则选择原版；三个独立配置的实际原版调用、重置后Graph replay和完整保存数组正确；调用入口和统计可以复核。继续默认原版，研究候选保持显式选择。", "",
              "本次不支持：Triton提速、选择器延迟大小、模型端到端收益、未知输入自动适配、GPU瓶颈原因或AMD性能。选择器本身没有放进计时区间，A/A也不是原版相对自身的优化。", "",
              "## 归因与复习", "",
              "本轮策略、选择器、测试运行器与审计由教练在用户授权下完成；GPU运行由教练通过已恢复的入口执行。没有新增本人独立实现/口述掌握证据。复习重点是：为什么负结果可以决定保留原版，为什么选择与独立验证要分开，以及为什么同一实现的计时仍有差异。", "",
              "## 重生成", "",
              "在项目根目录运行下列一行。只读冻结run和历史依据，不导入Torch/Triton，不调用WSL/GPU；只更新本模块生成报告。也可用 `--out 新目录` 对比确定性输出。", "",
              "```powershell", "& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B outputs/kv-cache-performance/modules/M07/audit-selection-validation.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    report = audit()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (args.out / "RESULTS-SELECTION.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], source_hashes=report["source_manifest_entries"],
                          counts=report["counts"], output=str(args.out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
