"""Read-only CPU reconstruction of the archived 36 memory experiment.

Does not import the runner, torch, triton, or contact WSL/GPU/network. Rebuilds
fixtures, geometry, launch records and statistics independently from raw files.
Writes only derived reports in --output; the original run remains untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUN_NAME = "run-20261014-190436-qknu0tb3"
DEFAULT_RUN = BASE / "logs/m07-memory-results" / RUN_NAME
ARMS = ("original_cuda", "frozen_v1", "grouped")
POLICIES = ("graph", "eager")
CASES = tuple("ABCDEF")
BUFFERS = ("cache", "k", "v", "indices")
COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
PINS = {
    "36-compare-kv-memory.py": "d0de1e650c4e615798d72b1f963b944ca7ebe4bb10ea8a855ea1cd4a44c25c2f",
    "36-compare-kv-memory.sh": "15d9dfa8a1cfd1ac18d773d8289f91eba1f1afb0473ee560e45c2308ad806a73",
    "36-memory-cpu-test.py": "7d35cf2f7d418aac2b3a258cf9b9eb9dd63da0deee29ed91f937df43d0d630e1",
    "triton_store_mask.py": "6e76ef9eb319f4e57e4a55923fcb5655fa5fb26f29c8c7af00c49f0c125bb19d",
    "triton_store_grouped.py": "5b190ce52e0f6c25f99b739346c2787f194ed7e75c759144cf25fcb06c8a1bd1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts, "unsafe archive path")
    target = root / p
    require(target.is_file() and not target.is_symlink(), "missing/nonregular archive file: " + str(target))
    return target


def cfg_for(condition):
    rows = np.arange(256, dtype=np.int32)
    if condition == "D":
        rows = np.random.default_rng(20261014).permutation(rows)
    return dict(condition=condition, batch=256, width=128, capacity=1024,
                layout="separate" if condition == "C" else "interleaved",
                gap=64 if condition == "B" else 0, guard=0,
                source_offset_elements=68 if condition == "E" else 64,
                input_address_mod128=8 if condition == "E" else 0,
                packets=4 if condition == "F" else 1, rows=rows.tolist())


def expected_arrays(cfg, slot):
    """Independent oracle: vectorized row scatter, not archived runner calls."""
    stride = 128 + cfg["gap"]
    offset = cfg["source_offset_elements"]
    indices = np.array(cfg["rows"], dtype=np.int32)
    values = np.random.default_rng(20261014 + 101 * slot).integers(0, 1024, (256, 128), dtype=np.int32).astype(np.float16)
    values[:, 0] = np.arange(256) + 256 * slot
    shape = (2, 1024, 128) if cfg["layout"] == "separate" else (1024, 2, 128)
    original = {"cache": np.full(shape, -1, dtype=np.float16), "indices": indices}
    for name, data in (("k", values), ("v", values + np.float16(1024))):
        backing = np.full(64 + 256 * stride + 68, -9, dtype=np.float16)
        locations = offset + np.arange(256)[:, None] * stride + np.arange(128)[None, :]
        backing[locations] = data
        original[name] = backing
    expected = original["cache"].copy()
    if cfg["layout"] == "separate":
        expected[0, indices, :] = values
        expected[1, indices, :] = values + np.float16(1024)
    else:
        expected[indices, 0, :] = values
        expected[indices, 1, :] = values + np.float16(1024)
    original["expected_cache"] = expected
    return original


def geometry(cfg, records):
    require(len(records) == cfg["packets"], "packet record count")
    allocations = []
    for slot, record in enumerate(records):
        require(record["slot"] == slot, "packet ordinal")
        arrays = expected_arrays(cfg, slot)
        require(set(record["buffers"]) == set(BUFFERS), "buffer keys")
        for name in BUFFERS:
            item, array = record["buffers"][name], arrays[name]
            require(item["shape"] == list(array.shape) and item["stride"] == [x // array.itemsize for x in array.strides], "backing dimensions")
            require(item["dtype"] == ("torch.int32" if name == "indices" else "torch.float16") and item["device"] == "cuda:0", "dtype/device")
            require(item["storage_offset"] == 0 and item["bytes"] == array.nbytes, "backing offset/size")
            require(isinstance(item["data_ptr"], int) and item["data_ptr"] > 0 and item["data_ptr"] % 128 == 0, "backing alignment")
            allocations.append((item["data_ptr"], item["data_ptr"] + array.nbytes))
        destination_stride = 128 if cfg["layout"] == "separate" else 256
        specs = [("cache", 0, [1024, 128], [destination_stride, 1]),
                 ("cache", 1024 * 128 if cfg["layout"] == "separate" else 128, [1024, 128], [destination_stride, 1]),
                 ("indices", 0, [256], [1]),
                 ("k", cfg["source_offset_elements"], [256, 128], [128 + cfg["gap"], 1]),
                 ("v", cfg["source_offset_elements"], [256, 128], [128 + cfg["gap"], 1])]
        require(len(record["views"]) == 5, "view count")
        for item, (name, offset, shape, stride) in zip(record["views"], specs):
            base = record["buffers"][name]
            size = 4 if name == "indices" else 2
            require(item == dict(shape=shape, stride=stride, dtype=base["dtype"], device=base["device"],
                                 data_ptr=base["data_ptr"] + offset * size, storage_offset=offset, bytes=math.prod(shape) * size), "actual view metadata")
            require(item["data_ptr"] % (4 if name == "indices" else 8) == 0, "original CUDA natural package alignment")
        require(all(x["data_ptr"] % 128 == cfg["input_address_mod128"] for x in record["views"][3:]), "source offset condition")
    allocations.sort()
    require(all(left[1] <= right[0] for left, right in zip(allocations, allocations[1:])), "recorded packet storage aliases")
    return len(allocations)


def order_for(round_id):
    ring = [(arm, policy) for policy in POLICIES for arm in ARMS]
    output = []
    for sample in range(30):
        start = (sample + round_id - 1) % 6
        sequence = ring[start:] + ring[:start]
        if (sample // 6 + round_id - 1) % 2:
            sequence.reverse()
        output.extend((sample, arm, policy) for arm, policy in sequence)
    return output


def launches_for(cfg):
    records = {}
    for arm in ARMS:
        records[arm] = []
        for slot in range(cfg["packets"]):
            if arm == "original_cuda":
                record = dict(slot=slot, entry="upstream store_cache", public_calls=1)
            else:
                parameters = dict(WIDTH=128, SRC_ROW_STRIDE=128 + cfg["gap"],
                                  DST_ROW_STRIDE=128 if cfg["layout"] == "separate" else 256, BLOCK_SIZE=128, num_warps=4)
                if arm == "grouped":
                    parameters.update(BATCH=256, ROWS_PER_PROGRAM=4)
                record = dict(slot=slot, grid=[256 if arm == "frozen_v1" else 64], parameters=parameters)
            records[arm].append(record)
    return records


def signal(ratios):
    if min(ratios) >= 1.03:
        return "THREE_ROUND_SLOWER_SIGNAL_ONLY"
    if max(ratios) <= .97:
        return "THREE_ROUND_FASTER_SIGNAL_ONLY"
    return "SMALL_OR_INCONSISTENT_NO_PROMOTION"


def recompute(workers):
    result = []
    for condition in CASES:
        entry = dict(condition=condition, config=cfg_for(condition))
        for policy in POLICIES:
            arms = {}
            for arm in ARMS:
                medians = [statistics.median(workers[(round_id, condition)]["samples_ms_per_call"][arm][policy]) * 1000 for round_id in (1, 2, 3)]
                base = [statistics.median(workers[(round_id, "A")]["samples_ms_per_call"][arm][policy]) * 1000 for round_id in (1, 2, 3)]
                ratios = [a / b for a, b in zip(medians, base)]
                arms[arm] = dict(round_medians_us=medians, representative_us=statistics.median(medians),
                                 round_span_pct=100 * (max(medians) - min(medians)) / statistics.median(medians),
                                 condition_over_A_round_time_ratios=ratios,
                                 condition_signal=signal(ratios) if condition != "A" else "BASELINE")
            for arm in ARMS[1:]:
                ratios = [a / b for a, b in zip(arms["original_cuda"]["round_medians_us"], arms[arm]["round_medians_us"])]
                arms[arm].update(original_over_arm=arms["original_cuda"]["representative_us"] / arms[arm]["representative_us"],
                                 paired_original_over_arm=ratios, paired_ratio_median=statistics.median(ratios), wins_vs_original=sum(x > 1 for x in ratios))
            entry[policy] = arms
        result.append(entry)
    return result


def audit(run):
    state, summary_file, plan = (read(run / name) for name in ("status.json", "summary.json", "plan.json"))
    require(state["status"] == summary_file["status"] == "COLLECTION_COMPLETE" and state["commit"] == COMMIT, "run status/commit")
    require(state["plan"] == summary_file["plan"] == plan, "plan copies")
    require(plan["configurations"] == [cfg_for(x) for x in CASES] and plan["implementations"] == list(ARMS), "planned scope")
    require((plan["fresh_processes"], plan["intervals_per_arm_policy"], plan["calls_per_interval"], plan["interval_averages"], plan["primary"], plan["reference"]) == (18, 30, 100, 3240, "graph", "eager"), "timing policy")
    expected_schedule, rng = [], random.Random(20261014)
    for round_id in (1, 2, 3):
        conditions = list(CASES)
        rng.shuffle(conditions)
        expected_schedule.extend(dict(round=round_id, condition=x) for x in conditions)
    require(plan["schedule"] == expected_schedule, "fresh-worker order")
    source_manifest = read(run / "source_hashes.json")
    for name, expected in source_manifest.items():
        require(sha(safe(run, name)) == expected, "source digest: " + name)
    for name, expected in PINS.items():
        require(source_manifest[name] == expected, "independently pinned source: " + name)
    require(read(run / "accepted35-gate.json")["status_sha256"] == sha(run / "prior35-status.json"), "accepted35 status receipt")
    require(read(run / "accepted35-gate.json")["source_hashes_sha256"] == sha(run / "prior35-source_hashes.json"), "accepted35 source receipt")
    require(re.search(r"range\(100\)", (run / "learner-timing.py").read_text(encoding="utf-8")) is not None, "100-call timer body")
    workers, environments, arrays_inventory, compiler_inventory, worker_inventory = {}, set(), [], [], []
    counts = Counter(source_files=len(source_manifest))
    for item in expected_schedule:
        condition, round_id = item["condition"], item["round"]
        directory = run / f"round-{round_id}-{condition}"
        worker = read(directory / "result.json")
        require(worker["status"] == "COLLECTED" and worker["condition"] == condition and worker["round"] == round_id, "worker identity")
        cfg = cfg_for(condition)
        require(worker["config"] == cfg and worker["policy"] == plan and worker["calls_per_interval"] == 100, "worker configuration/policy")
        require(worker["correctness_checks"] == 190 and worker["packet_checks"] == 190 * cfg["packets"] and worker["buffer_checks"] == 760 * cfg["packets"] and worker["per_packet_checks"] == [190] * cfg["packets"], "recorded runtime check counts")
        require(worker["actual_launches"] == launches_for(cfg), "first actual submissions")
        require(read(directory / "packets.json") == worker["packets"], "packet manifest copy")
        counts["recorded_allocations"] += geometry(cfg, worker["packets"])
        table = [0, 1, 2, 3] if condition == "F" else [0, 0, 0, 0]
        require(worker["slot_table"] == table and worker["capture_traces"] == {arm: table * 25 for arm in ARMS}, "actual capture slot sequence")
        counts["capture_sequences"] += 3
        counts["recorded_capture_calls"] += 300
        records = worker["measurements"]
        require([(x["sample"], x["implementation"], x["policy"]) for x in records] == order_for(round_id), "measurement rotation")
        positions = Counter((arm, policy, i % 6) for i, (_, arm, policy) in enumerate(order_for(round_id)))
        require(set(positions.values()) == {5}, "position balance")
        for arm in ARMS:
            for policy in POLICIES:
                samples = worker["samples_ms_per_call"][arm][policy]
                require(len(samples) == 30 and all(math.isfinite(x) and x > 0 for x in samples), "invalid original sample")
                require(samples == [x["ms_per_call"] for x in records if (x["implementation"], x["policy"]) == (arm, policy)], "raw sample copy")
                counts["interval_averages"] += len(samples)
        stages = ["before", "final"] + [arm + suffix for arm in ARMS for suffix in ("-initial", "-graph-replay")]
        expected_npz = set()
        for slot in range(cfg["packets"]):
            packet_dir = directory / f"packet-{slot}"
            expected = expected_arrays(cfg, slot)
            require(sorted(expected["indices"].tolist()) == list(range(256)), "changed destination slot set")
            for stage in stages:
                path = packet_dir / (stage + ".npz")
                expected_npz.add(path)
                require(path.exists(), "missing saved packet stage")
                names = set(expected) if stage == "before" else set(BUFFERS)
                details = []
                with np.load(path, allow_pickle=False) as archive:
                    require(set(archive.files) == names, "saved array names")
                    for name in sorted(names):
                        array = archive[name]
                        reference = expected["expected_cache" if name == "cache" and stage != "before" else name]
                        require(array.dtype == reference.dtype and array.shape == reference.shape and np.array_equal(array, reference), f"saved full backing mismatch: {path}/{name}")
                        details.append(dict(name=name, dtype=str(array.dtype), shape=list(array.shape), elements=int(array.size), bytes=int(array.nbytes)))
                        counts["saved_arrays"] += 1
                        counts["saved_elements"] += int(array.size)
                        counts["saved_uncompressed_bytes"] += int(array.nbytes)
                arrays_inventory.append(dict(path=path.relative_to(run).as_posix(), sha256=sha(path), compressed_bytes=path.stat().st_size, arrays=details))
                counts["npz_files"] += 1
                counts["before_npz" if stage == "before" else "after_npz"] += 1
            counts["unique_packets"] += 1
        require(set(directory.rglob("*.npz")) == expected_npz, "unexpected/missing saved NPZ")
        for arm in ARMS[1:]:
            artifact_dir = directory / (arm + "-first-packet")
            compiled = read(artifact_dir / "compiled.json")
            require(compiled["status"] == "ALL_REQUESTED_ARTIFACTS_RETAINED" and set(compiled["artifacts"]) == {"ttir", "ttgir", "llir", "ptx", "cubin"}, "compiler artifacts unavailable")
            require(compiled["metadata"]["value"]["num_warps"] == 4 and compiled["metadata"]["value"]["arch"] == "sm89", "compiled target/warps")
            artifacts = {}
            for kind, metadata in compiled["artifacts"].items():
                path = safe(artifact_dir, metadata["path"])
                require(sha(path) == metadata["sha256"] and path.stat().st_size == metadata["bytes"], "compiled artifact integrity")
                if kind != "cubin":
                    text = path.read_text(encoding="utf-8")
                    require(len(text.splitlines()) == metadata["lines"] and len(text) > 0, "compiled text readability")
                    counts["compiled_text_files"] += 1
                else:
                    require(path.read_bytes().startswith(b"\x7fELF"), "cubin ELF signature")
                artifacts[kind] = dict(path=path.relative_to(run).as_posix(), sha256=metadata["sha256"], bytes=metadata["bytes"])
                counts["compiled_artifacts"] += 1
            ptx_path = artifact_dir / "compiled/kernel.ptx"
            ptx = ptx_path.read_text(encoding="utf-8")
            require(".target sm_89" in ptx and ".reqntid 128" in ptx, "PTX target/threads")
            memory = [dict(line=i, text=line.strip()) for i, line in enumerate(ptx.splitlines(), 1) if "ld.global" in line or "st.global" in line]
            ttgir = (artifact_dir / "compiled/kernel.ttgir").read_text(encoding="utf-8")
            layout_and_parameters = [dict(line=i, text=line) for i, line in enumerate(ttgir.splitlines(), 1) if line.startswith("#blocked =") or "tt.func public" in line]
            opcodes = Counter(re.search(r"(?:ld|st)\.global\.[a-z0-9.]+", x["text"])[0] for x in memory)
            compiler_inventory.append(dict(condition=condition, round=round_id, arm=arm, report_sha256=sha(artifact_dir / "compiled.json"),
                artifacts=artifacts, static_resources=compiled["static_resources"], global_memory_instructions=memory,
                global_memory_opcode_counts=dict(opcodes), ttgir_layout_and_parameters=layout_and_parameters))
            counts["compiled_returns"] += 1
        environment = tuple(worker[x] for x in ("commit", "torch", "cuda", "triton", "gpu", "device_properties"))
        environments.add(environment)
        require(environment[:5] == (COMMIT, "2.9.1+cu128", "12.8", "3.5.1", "NVIDIA GeForce RTX 4070 Ti"), "fixed software/hardware identity")
        command = read(run / (directory.name + "-command.json"))
        require(command[command.index("--condition") + 1] == condition and int(command[command.index("--round") + 1]) == round_id, "recorded fresh-process command")
        require("MEMORY_READY" in (run / (directory.name + ".log")).read_text(encoding="utf-8"), "worker ready console")
        counts["recorded_runtime_check_stages"] += worker["correctness_checks"]
        counts["recorded_runtime_packet_checks"] += worker["packet_checks"]
        counts["recorded_runtime_buffer_checks"] += worker["buffer_checks"]
        worker_inventory.append(dict(path=directory.name, result_sha256=sha(directory / "result.json"), packets_sha256=sha(directory / "packets.json"),
            command_sha256=sha(run / (directory.name + "-command.json")), console_sha256=sha(run / (directory.name + ".log")),
            packets=worker["packets"], recorded_packet_checks=worker["per_packet_checks"], telemetry_before=worker["telemetry_before"], telemetry_after=worker["telemetry_after"]))
        workers[(round_id, condition)] = worker
    require(len(environments) == 1 and len(workers) == 18 and counts["interval_averages"] == 3240, "complete single-environment matrix")
    ordered_results = [workers[(x["round"], x["condition"])] for x in expected_schedule]
    require(ordered_results == state["results"] == read(run / "partial-results.json"), "worker/state/partial copies")
    summary = recompute(workers)
    require(summary == state["summary"] == summary_file["summary"], "independently recomputed summary differs")
    errors = [p.relative_to(run).as_posix() for p in run.rglob("error.txt")] + [p.relative_to(run).as_posix() for p in run.rglob("failed-*.npz")]
    require(not errors, "unexpected failure artifacts in this successful run")
    cross = []
    for row in summary:
        for policy in POLICIES:
            for reference, candidate in (("original_cuda", "frozen_v1"), ("original_cuda", "grouped"), ("frozen_v1", "grouped")):
                a, b = row[policy][reference], row[policy][candidate]
                ratios = [x / y for x, y in zip(a["round_medians_us"], b["round_medians_us"])]
                cross.append(dict(condition=row["condition"], policy=policy, reference=reference, candidate=candidate,
                    representative_ratio=a["representative_us"] / b["representative_us"], paired_round_ratios=ratios,
                    paired_ratio_median=statistics.median(ratios), wins=sum(x > 1 for x in ratios), ties=sum(x == 1 for x in ratios), losses=sum(x < 1 for x in ratios)))
    observations = []
    for arm in ARMS[1:]:
        for round_id in (1, 2, 3):
            selected = {x["condition"]: x for x in compiler_inventory if x["arm"] == arm and x["round"] == round_id and x["condition"] in ("A", "E")}
            expected_opcodes = {"ld.global.b32":1, "ld.global.b16":2, "st.global.b16":2}
            for condition in ("A", "E"):
                if arm == "grouped":
                    expected_opcodes = ({"ld.global.b32":1, "ld.global.v2.b32":2, "st.global.v2.b32":2} if condition == "A" else
                                        {"ld.global.b32":1, "ld.global.b16":8, "st.global.v2.b32":2})
                    layout = selected[condition]["ttgir_layout_and_parameters"][0]["text"]
                    require("sizePerThread = [1, 4]" in layout and "threadsPerWarp = [1, 32]" in layout and "warpsPerCTA = [4, 1]" in layout, "A/E grouped layout observation")
                    header = selected[condition]["ttgir_layout_and_parameters"][1]["text"]
                    for pointer in ("k_ptr", "v_ptr"):
                        needle = "%" + pointer + ": !tt.ptr<f16> " + ("{tt.divisibility = 16 : i32}" if condition == "A" else "loc(")
                        require(needle in header, "A/E input alignment metadata observation")
                require(selected[condition]["global_memory_opcode_counts"] == expected_opcodes, "A/E instruction width observation")
            observations.append(dict(arm=arm, round=round_id, A_E_ptx_byte_identical=selected["A"]["artifacts"]["ptx"]["sha256"] == selected["E"]["artifacts"]["ptx"]["sha256"],
                A_E_memory_instructions_equal=[x["text"] for x in selected["A"]["global_memory_instructions"]] == [x["text"] for x in selected["E"]["global_memory_instructions"]],
                A_E_register_values=[selected[c]["static_resources"]["registers_per_thread"]["value"] for c in ("A", "E")],
                opcode_counts={c:selected[c]["global_memory_opcode_counts"] for c in ("A", "E")}))
    require(all(row["graph"][arm]["condition_signal"] == "SMALL_OR_INCONSISTENT_NO_PROMOTION" for row in summary[1:] for arm in ARMS), "frozen Graph conclusion changed")
    return dict(status="ARCHIVED_MEMORY_RUN_AUDIT_PASS", run=RUN_NAME, run_status="COLLECTION_COMPLETE", new_gpu_execution=False,
        source_manifest_sha256=sha(run / "source_hashes.json"), run_status_sha256=sha(run / "status.json"), raw_summary_sha256=sha(run / "summary.json"),
        audit_script_sha256=sha(Path(__file__)), independent_scope="No runner imports; fixture, geometry, launch records and statistics reconstructed from archived data on CPU.",
        counts=dict(counts), failure_scope=dict(failed_workers_this_run=0, failure_artifacts_this_run=errors, historical_other_runs="not counted or changed; this zero does not erase prior failures"),
        checks_limitation="Saved 7 after stages per unique packet verified offline; 190 runtime check stages per worker are recorded counters, not 190 separately archived NPZ stages.",
        environment=list(next(iter(environments))), plan=plan, summary=summary, cross_arm_comparisons=cross,
        A_E_compiler_observations=observations, sources=source_manifest, workers=worker_inventory, arrays=arrays_inventory, compiled=compiler_inventory)


def fmt(values, digits=4):
    return " / ".join(f"{x:.{digits}f}" for x in values)


def markdown(report):
    counts, summary = report["counts"], report["summary"]
    lines = ["# M07 固定形状访存实验：36号结果", "",
        "本轮六种输入条件、三种冻结实现全部完成。Graph下，B—F的任何一臂都没有达到事前约定的“三轮同向且每轮至少3%”敏感信号；这表示本次证据不足以确认稳定变化，不能改写成“内存条件没有影响”。没有据此晋升grouped或修改内核。", "",
        f"原始运行：[{RUN_NAME}](../../logs/m07-memory-results/{RUN_NAME}/status.json)。GPU由主任务实际执行；本报告由[独立CPU审查脚本](audit-memory-results.py)重建，[详细核验JSON](memory-validation.json)保留数组、地址、编译文件及统计证据。未重跑GPU。", "",
        "## 范围与计时口径", "",
        "固定RTX 4070 Ti、Torch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1；B256/D128/C1024，FP16，唯一int32索引，原版提交`9a91cfafe754aa85daee49998176275667eb58f2`。三臂为原版CUDA、4warp单行v1、4warp四行grouped，源码保持冻结。", "",
        "| 条件 | 相对A的改变 | 源/目标行距（FP16元素） | 输入起点 mod128 | 实际packet数 |",
        "| --- | --- | --- | ---: | ---: |",
        "| A | 基准：交错缓存、顺序目标槽 | 128 / 256 | 0 | 1 |",
        "| B | 输入gap64 | 192 / 256 | 0 | 1 |",
        "| C | 缓存K/V分离 | 128 / 128 | 0 | 1 |",
        "| D | 将相同0..255目标槽集合固定置换 | 128 / 256 | 0 | 1 |",
        "| E | 仅输入K/V起点额外偏移8字节 | 128 / 256 | 8 | 1 |",
        "| F | 四套独立实际地址逐调用循环 | 128 / 256 | 0 | 4 |", "",
        "每条件3个fresh worker，共18个；每worker三臂同场，Graph主结果、Eager单列，每臂/政策30个区间，每区间100次调用。先求每进程30个区间均值的中位数，再取3个进程中位数的中位数作为代表值。原始3240个区间均值全部保留，没有删异常点。", "",
        "三臂配对比来自同一worker；条件/A只按轮次对应，来自不同进程，存在跨进程设备波动限制。分配、校验、首次JIT、预热、重置、完整backing检查和编译导出均在计时区间外。Eager使用统一四项Python选择表，原版走公开store_cache，Triton走预验证raw launch，因此包含不同主机提交路径。Graph capture保留100次槽选择记录。", "",
        "F每套在每个区间被调用25次；有效触及约257KiB/套、四套约1MiB，实际分配更大。没有cache flush，不是冷缓存或DRAM带宽测试。C同时改变K/V相对位置及目标行距；E仍满足原版8字节包的自然对齐。", "",
        "## 全部保存证据核验", "",
        f"核验{counts['source_files']}项来源哈希、18份worker与状态/partial副本、{counts['unique_packets']}套实际packet、{counts['recorded_allocations']}个记录的独立backing分配。检查了data_ptr、storage_offset、完整尺寸/stride、8字节自然对齐和同worker所有packet无重叠；E的输入offset为68个FP16，其他为64。", "",
        f"逐项读取{counts['npz_files']}份NPZ（{counts['before_npz']}份before、{counts['after_npz']}份after），共{counts['saved_arrays']:,}个数组、{counts['saved_elements']:,}个元素、{counts['saved_uncompressed_bytes']:,}字节解压数据。独立重建随机输入和scatter期望，完整核对目标缓存、未命中区域、输入gap及prefix/suffix、indices；没有只检查被写槽。", "",
        f"运行记录累计{counts['recorded_runtime_check_stages']:,}个完整检查阶段、{counts['recorded_runtime_packet_checks']:,}次packet检查、{counts['recorded_runtime_buffer_checks']:,}次buffer比较。离线实际验证的是每packet保存的7个after阶段；没有把190个运行时阶段宣称为190份离线保存数组。", "",
        f"{counts['capture_sequences']}条capture序列包含{counts['recorded_capture_calls']:,}条记录：A—E为0重复100次，F为0/1/2/3循环25次。全部实际首次launch参数、6项测量顺序、每项在每位置出现5次及3240个正有限原始样本均通过。", "",
        f"36份真实首次Triton返回对应{counts['compiled_artifacts']}份编译产物：{counts['compiled_text_files']}份TTIR/TTGIR/LLIR/PTX可读，36份cubin具备ELF签名，全部长度与SHA匹配。编译元数据只代表静态特化，不是实际occupancy或性能原因。", "",
        "本轮成功run内失败worker为0，error.txt与failed-*.npz为0。这个0仅描述本轮；历史其他运行的失败未删除、未改写，也未纳入本轮统计。", ""]
    for policy, title in (("graph", "GRAPH：主结果"), ("eager", "EAGER：单列参考")):
        lines += ["## " + title, "", "单位µs/次；代表值是三个进程中位数的中位数。", "",
            "| 条件 | 原版代表 | v1代表 | grouped代表 |", "| --- | ---: | ---: | ---: |"]
        for row in summary:
            lines.append("| " + row["condition"] + " | " + " | ".join(f"{row[policy][arm]['representative_us']:.5f}" for arm in ARMS) + " |")
        lines += ["", "条件/A耗时比>1表示该条件更慢。r1/r2/r3逐轮报告，不以代表值掩盖方向变化。", "",
            "| 条件 | 实现 | 三轮耗时 µs | 三轮条件/A耗时比 | 达到3%三轮同向门槛 |",
            "| --- | --- | --- | --- | --- |"]
        for row in summary:
            for arm in ARMS:
                item = row[policy][arm]
                label = {"BASELINE":"基准", "SMALL_OR_INCONSISTENT_NO_PROMOTION":"未达到", "THREE_ROUND_SLOWER_SIGNAL_ONLY":"变慢信号，仅本轮", "THREE_ROUND_FASTER_SIGNAL_ONLY":"变快信号，仅本轮"}[item["condition_signal"]]
                lines.append(f"| {row['condition']} | {arm} | {fmt(item['round_medians_us'],5)} | {fmt(item['condition_over_A_round_time_ratios'],5)} | {label} |")
        lines += ["", "同场比值＝参考耗时/候选耗时，>1表示候选更快；代表比与逐轮比的中位数分别列出。", "",
            "| 条件 | 参考/候选 | 代表比 | 三轮配对比 | 配对比中位数 | 候选胜/平/负 |",
            "| --- | --- | ---: | --- | ---: | --- |"]
        for item in report["cross_arm_comparisons"]:
            if item["policy"] == policy:
                lines.append(f"| {item['condition']} | {item['reference']}/{item['candidate']} | {item['representative_ratio']:.5f} | {fmt(item['paired_round_ratios'],5)} | {item['paired_ratio_median']:.5f} | {item['wins']}/{item['ties']}/{item['losses']} |")
        lines.append("")
    f = summary[5]["graph"]["frozen_v1"]
    lines += ["## 可以得出的结论", "",
        "- Graph全部条件敏感性比较均未跨过冻结门槛。F下v1三个轮次都比A慢，但变化分别为" + fmt([(x-1)*100 for x in f["condition_over_A_round_time_ratios"]], 3) + "%：其中约2.64%的一轮未达到3%。只记作三轮同向的观察，不能把阈值降到2%后称通过。",
        "- 输入条件、有限工作集和主机调度都可能影响结果。当前数据没有硬件计数器，不能确认DRAM带宽、cache miss、实际occupancy或向量事务是瓶颈；也不能由未达到门槛断言这些因素无影响。",
        "- Eager下grouped的D、E相对A达到三个轮次均慢至少3%的工程信号。它仍包含主机提交路径，只能按Eager口径陈述；不能替代Graph结论，也不能据此给出DRAM或cache根因。",
        "- A/E确实出现了有教学价值的编译差异：grouped的输入K/V从每份1条`ld.global.v2.b32`变为每份4条`ld.global.b16`；输出仍为每份1条`st.global.v2.b32`。第一条`ld.global.b32`是indices读取，不计入K/V向量宽度。三个轮次均一致，寄存器仍18/线程；v1仍为K/V的b16读写、14寄存器/线程。",
        "- 这些是PTX中的每线程指令宽度和静态条数，不是DRAM事务数。E保持8字节自然对齐，但本次grouped特化的输入指针未携带A中的`tt.divisibility=16`，布局仍为sizePerThread=[1,4]、threadsPerWarp=[1,32]、warpsPerCTA=[4,1]；不能把结果推广为任意8字节地址必然标量化。静态差异成立，稳定Graph性能因果仍未证实。",
        "- 本轮输入敏感性实验不改变公共支持范围，不自动生成shape dispatch，不晋升grouped，不与10-10会话合并计算收益。实现与运行按10-14授权由教练完成，学员独立解释另记。", "",
        f"具体证据（r1，r2/r3也已逐项核对）：[A的grouped TTGIR](../../logs/m07-memory-results/{RUN_NAME}/round-1-A/grouped-first-packet/compiled/kernel.ttgir)第1、9行与[E的TTGIR](../../logs/m07-memory-results/{RUN_NAME}/round-1-E/grouped-first-packet/compiled/kernel.ttgir)第1、9行；[A的PTX](../../logs/m07-memory-results/{RUN_NAME}/round-1-A/grouped-first-packet/compiled/kernel.ptx)第52行为indices、73/88行为K/V向量读、80/94行为向量写；[E的PTX](../../logs/m07-memory-results/{RUN_NAME}/round-1-E/grouped-first-packet/compiled/kernel.ptx)第53行为indices、76/80/84/88与107/111/115/119行为两份K/V标量读、97/127行为向量写。", "",
        "| 实现/轮次 | A/E PTX字节相同 | A/E全局访存指令文本相同 | A/E寄存器/线程 |",
        "| --- | --- | --- | --- |"]
    for item in report["A_E_compiler_observations"]:
        lines.append(f"| {item['arm']}/r{item['round']} | {'是' if item['A_E_ptx_byte_identical'] else '否'} | {'是' if item['A_E_memory_instructions_equal'] else '否'} | {fmt(item['A_E_register_values'],0)} |")
    lines += ["", "## 复现离线审查", "", "在已有NumPy的Python环境运行：", "", "```text", "python -B outputs/kv-cache-performance/modules/M07/audit-memory-results.py", "```", "", "脚本只读取固定归档run，重新生成本报告和memory-validation.json；不导入实验runner、不调用torch/Triton/WSL/GPU或网络。可用`--output 新目录`将复算结果另存。所有原始run保持不变。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    require(args.output.resolve() != args.run.resolve() and args.run.resolve() not in args.output.resolve().parents, "reports cannot modify archived run")
    report = audit(args.run)
    require("torch" not in sys.modules and "triton" not in sys.modules, "GPU framework imported")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "memory-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    (args.output / "RESULTS-MEMORY.md").write_text(markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=report["status"], counts=report["counts"], output=str(args.output), GPU="NOT_RUN")))


if __name__ == "__main__":
    main()
