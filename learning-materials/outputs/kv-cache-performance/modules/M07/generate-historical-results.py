#!/usr/bin/env python3
"""CPU-only, read-only reconstruction of four archived M07 timing sessions.

Run from any directory. JSON/statistics use only the standard library. ReportLab
renders SVG/PDF; optional --png-node uses an existing Node + sharp installation.
No archived runner, torch, triton, GPU, WSL, or network is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
POLICIES = ("graph", "eager")
GROUPED_SHA = "5b190ce52e0f6c25f99b739346c2787f194ed7e75c759144cf25fcb06c8a1bd1"
# A new experiment must be a NEW session with its own complete run and contract.
# Do not replace an existing run or pool rounds across sessions with the same B/D.
SESSIONS = [
    dict(id="27", name="单行 4→1 warp", run="m07-warps-timing-results/run-20261010-121112-p7ip91zk",
         candidate="candidate", prior="frozen_v1", prior_key="v1", cases=[(1, 64), (256, 128)],
         source="triton_store_candidate.py", sha="f2320f7ede79b9e4c7164ae0d91b87a53a718b42757b0c64e9bfb64bf070092b",
         review="20261010-m07-warps-timing-review.md"),
    dict(id="30", name="4行分组：两配置探索", run="m07-grouped-timing-results/run-20261010-131625-od4rcwyb",
         candidate="grouped", prior="frozen_v1", prior_key="v1", cases=[(1, 64), (256, 128)],
         source="triton_store_grouped.py", sha=GROUPED_SHA, review="20261010-m07-grouped-timing-review.md"),
    dict(id="32", name="4行分组：六配置回归", run="m07-grouped-six-timing-results/run-20261010-134931-4l7bljlh",
         candidate="grouped", prior="frozen_v1", prior_key="v1", cases=[(b, d) for b in (1, 16, 256) for d in (64, 128)],
         source="triton_store_grouped.py", sha=GROUPED_SHA, review="20261010-m07-grouped-six-timing-review.md"),
    dict(id="35", name="满组 constexpr 分支", run="m07-full-groups-timing-results/run-20261010-215026-gs6vimx3",
         candidate="full_groups", prior="accepted_grouped", prior_key="grouped", cases=[(b, d) for b in (5, 256) for d in (64, 128, 192)],
         source="triton_store_full_groups.py", sha="6a53a5304566420c3ee0163f0b8c6ff157b4f9b16f99a83be20364b415a81156",
         review="20261010-m07-full-groups-timing-review.md"),
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def configuration(spec, b, d):
    require((b, d) in spec["cases"], "unexpected case")
    full = spec["id"] == "35"
    layout, gap, guard = ("interleaved", 0, 0) if not full or d == 64 else ("separate", 64, 64 if d == 192 else 0)
    capacity = 512 if full else 1024
    return dict(batch=b, width=d, capacity=capacity, layout=layout, input_gap=gap, guard=guard,
                src_row_stride=d + gap, dst_row_stride=2 * d if layout == "interleaved" else d + guard,
                block_size=1 << (d - 1).bit_length(), input_seed=20261011 if full else 20261004,
                indices=[(37 * i + 11) % 512 if full else i for i in range(b)])


def validate_worker(spec, worker, plan):
    b, d, r = (worker[k] for k in ("batch", "width", "round"))
    cfg = configuration(spec, b, d)
    require(type(r) is int and r in (1, 2, 3), "unexpected round")
    require(worker["status"] == "COLLECTED", "worker not collected")
    require(worker["capacity"] == cfg["capacity"] and worker["seed"] == cfg["input_seed"], "capacity/seed mismatch")
    require(worker["calls_per_interval"] == 100 and worker["correctness_checks"] == 190, "measurement/check count changed")
    require(worker["commit"] == COMMIT and worker["policy"] == plan, "worker source/policy mismatch")
    expected_cfg = dict(layout="interleaved", input_gap=0)
    if spec["id"] == "35":
        expected_cfg = dict(batch=b, width=d, capacity=512, layout=cfg["layout"], gap=cfg["input_gap"], guard=cfg["guard"], rows=cfg["indices"])
    require(worker["config"] == expected_cfg, "layout/gap/guard/indices mismatch")
    arms = ("original_cuda", spec["prior"], spec["candidate"])
    require(set(worker["samples_ms_per_call"]) == set(arms), "missing/extra arm")
    for arm in arms:
        series = worker["samples_ms_per_call"][arm]
        require(set(series) == set(POLICIES), "missing/extra policy")
        for samples in series.values():
            require(len(samples) == 30, "interval count changed")
            require(all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in samples), "invalid timing sample")
    launches = worker["actual_launches"]
    require(set(launches) == set(arms), "launch arm mismatch")
    require(launches["original_cuda"] == {"entry": "upstream store_cache", "public_calls": 1}, "original entry mismatch")
    for arm in arms[1:]:
        grouped = arm in ("grouped", "accepted_grouped", "full_groups")
        params = dict(WIDTH=d, SRC_ROW_STRIDE=cfg["src_row_stride"], DST_ROW_STRIDE=cfg["dst_row_stride"], BLOCK_SIZE=cfg["block_size"], num_warps=1 if arm == "candidate" else 4)
        if grouped:
            params.update(BATCH=b, ROWS_PER_PROGRAM=4)
        require(launches[arm] == [dict(grid=[(b + 3) // 4 if grouped else b], parameters=params)], "grid/stride/block/warps mismatch")
    measurements = worker["measurements"]
    require(len(measurements) == 180, "measurement list incomplete")
    positions = Counter()
    items = {(a, p) for a in arms for p in POLICIES}
    for sample in range(30):
        chunk = measurements[sample * 6:sample * 6 + 6]
        require({(x["implementation"], x["policy"]) for x in chunk} == items, "measurement duplicate/missing item")
        for pos, item in enumerate(chunk):
            arm, policy = item["implementation"], item["policy"]
            require(item["sample"] == sample and item["ms_per_call"] == worker["samples_ms_per_call"][arm][policy][sample], "measurement/raw sample mismatch")
            positions[(arm, policy, pos)] += 1
    require(len(positions) == 36 and set(positions.values()) == {5}, "measurement order not balanced")
    return cfg


def comparison(reference_medians, candidate_medians):
    representative = statistics.median(reference_medians) / statistics.median(candidate_medians)
    paired = [a / b for a, b in zip(reference_medians, candidate_medians, strict=True)]
    return dict(representative_ratio=representative, paired_round_ratios=paired,
                median_paired_ratio=statistics.median(paired),
                candidate_time_change_pct=(1 / representative - 1) * 100,
                wins=sum(x > 1 for x in paired), ties=sum(x == 1 for x in paired), losses=sum(x < 1 for x in paired),
                direction="3/3 faster in this session" if all(x > 1 for x in paired) else "no 3/3 faster direction")


def summarize(spec, workers, archived_summary):
    expected = {(r, b, d) for r in (1, 2, 3) for b, d in spec["cases"]}
    actual = [(w["round"], w["batch"], w["width"]) for w in workers]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "duplicate/missing worker")
    archived = {(x["batch"], x["width"]): x for x in archived_summary}
    require(len(archived) == len(archived_summary) and set(archived) == set(spec["cases"]), "archived summary cases mismatch")
    rows = []
    for b, d in spec["cases"]:
        ordered = sorted((w for w in workers if (w["batch"], w["width"]) == (b, d)), key=lambda w: w["round"])
        row = dict(session=spec["id"], config=configuration(spec, b, d))
        for policy in POLICIES:
            values = {}
            saved = archived[(b, d)][policy]
            for arm in ("original_cuda", spec["prior"], spec["candidate"]):
                medians = [statistics.median(w["samples_ms_per_call"][arm][policy]) * 1000 for w in ordered]
                representative = statistics.median(medians)
                values[arm] = dict(round_medians_us=medians, representative_us=representative)
                require(all(close(a, b) for a, b in zip(medians, saved[arm]["round_medians_us"], strict=True)), "saved process medians mismatch")
                require(close(representative, saved[arm]["representative_us"]), "saved representative mismatch")
            refs = {}
            for ref, key in (("original_cuda", "original"), (spec["prior"], spec["prior_key"])):
                stats = comparison(values[ref]["round_medians_us"], values[spec["candidate"]]["round_medians_us"])
                require(close(stats["representative_ratio"], saved[f"speedup_{key}_over_{spec['candidate']}"]), "saved ratio mismatch")
                require(all(close(a, b) for a, b in zip(stats["paired_round_ratios"], saved[f"paired_round_speedups_{key}_over_{spec['candidate']}"], strict=True)), "saved paired ratio mismatch")
                require(close(-stats["candidate_time_change_pct"], saved[f"time_reduction_pct_vs_{key}"]), "saved time change mismatch")
                refs[ref] = stats
            row[policy] = dict(arms=values, comparisons=refs)
        rows.append(row)
    return rows


def load_session(spec, base=BASE):
    run = base / "logs" / spec["run"]
    status, manifest, saved_summary = (read_json(run / f) for f in ("status.json", "source_hashes.json", "summary.json"))
    require(status["status"] == saved_summary["status"] == "COLLECTION_COMPLETE", "session incomplete")
    require(status["commit"] == COMMIT, "wrong upstream commit")
    require(status["summary"] == saved_summary["summary"], "summary copies differ")
    plan = status["plan"]
    require(plan == saved_summary["plan"] == read_json(run / "plan.json"), "plan copies differ")
    require(plan["cases"] == [list(x) for x in spec["cases"]], "case plan mismatch")
    n = len(spec["cases"]) * 3
    require((plan["rounds"], plan["fresh_processes"], plan["intervals_per_impl_policy"], plan["calls_per_interval"], plan["interval_averages"]) == (3, n, 30, 100, n * 180), "sampling plan mismatch")
    require((plan["primary"], plan["reference"], plan["dtype"], plan["eager_warmup_calls"], plan["graph_warmup_replays"]) == ("graph", "eager", "float16", 20, 5), "policy changed")
    schedule = [(s["round"], s["batch"], s["width"]) for s in plan["schedule"]]
    require(len(schedule) == len(set(schedule)) == n and set(schedule) == {(r, b, d) for r in (1, 2, 3) for b, d in spec["cases"]}, "schedule mismatch")
    require(manifest.get(spec["source"]) == spec["sha"], "candidate source pin mismatch")
    for name, sha in manifest.items():
        path = (run / name).resolve()
        require(path.is_relative_to(run.resolve()), "source escapes archive")
        require(digest(path) == sha, f"source hash mismatch: {name}")
    paths = sorted(run.glob("round-*/result.json"))
    require(len(paths) == n, "missing/extra worker file")
    workers, worker_sources = [], []
    for path in paths:
        w = read_json(path)
        validate_worker(spec, w, plan)
        require(path.parent.name == f"round-{w['round']}-B{w['batch']}-D{w['width']}", "worker directory mismatch")
        workers.append(w)
        worker_sources.append(dict(path=path.relative_to(base).as_posix(), sha256=digest(path)))
    require(sorted(status["results"], key=lambda x: (x["round"], x["batch"], x["width"])) == sorted(workers, key=lambda x: (x["round"], x["batch"], x["width"])), "root and worker copies differ")
    rows = summarize(spec, workers, status["summary"])
    environment = {k: workers[0][k] for k in ("gpu", "torch", "cuda", "triton", "commit")}
    require(all({k: w[k] for k in environment} == environment for w in workers), "environment differs within session")
    sources = {f: digest(run / f) for f in ("status.json", "plan.json", "source_hashes.json", "summary.json", "hypothesis-card.md")}
    # Keep raw samples and their actual interleaving, without duplicating plan in every worker.
    preserved = [{k: v for k, v in w.items() if k != "policy"} for w in workers]
    return dict(id=spec["id"], name=spec["name"], run=spec["run"], candidate=spec["candidate"], prior=spec["prior"],
                environment=environment, ownership_as_recorded=status["ownership"], plan=plan,
                source_manifest=manifest, source_file_count=len(manifest), archive_file_sha256=sources,
                worker_sources=worker_sources, workers=preserved, cases=rows, review=spec["review"])


def chart(data, policy, kind):
    """ReportLab standard LinePlot scatter charts; session identity is never pooled."""
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.graphics.widgets.markers import makeMarker
    from reportlab.lib import colors
    rows = [(s, c) for s in data["sessions"] for c in s["cases"]]
    width, height = 1220, 780
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=colors.white, strokeColor=None))
    title = f"M07 historical results | {policy.upper()} {'(primary)' if policy == 'graph' else '(host submission reference)'}"
    drawing.add(String(28, 748, title, fontName="Helvetica-Bold", fontSize=20, fillColor=colors.HexColor("#152638")))
    subtitle = "Within-session ratios; circles pair the same round. > 1 means candidate faster. Both panels share one scale." if kind == "ratios" else "Microseconds per call: median of 3 process medians; 30 intervals x 100 calls per process. Lower is faster."
    drawing.add(String(28, 723, subtitle, fontName="Helvetica", fontSize=12, fillColor=colors.HexColor("#526174")))
    plot_y, plot_h = 95, 550
    n = len(rows)
    for i, (session, case) in enumerate(rows):
        y = plot_y + plot_h * (n - i - 0.5) / n
        if i % 2 == 0:
            drawing.add(Rect(25, y - plot_h / n / 2, width - 50, plot_h / n, fillColor=colors.HexColor("#f4f6f8"), strokeColor=None))
        cfg = case["config"]
        drawing.add(String(32, y - 3, f"#{session['id']}  B{cfg['batch']:<3} D{cfg['width']}", fontName="Helvetica", fontSize=11))
        if i and session["id"] != rows[i - 1][0]["id"]:
            drawing.add(Line(25, y + plot_h / n / 2, width - 25, y + plot_h / n / 2, strokeColor=colors.HexColor("#9faabb"), strokeWidth=1.3))
    panels = [(205, 445, "Prior Triton / candidate", "prior"), (705, 445, "Original CUDA / candidate", "original")] if kind == "ratios" else [(220, 925, "All three arms in each session", "latency")]
    palette = [colors.HexColor(x) for x in ("#1877a5", "#df7832", "#7c58a3", "#182a3c")]
    for x, panel_width, heading, role in panels:
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = x, plot_y, panel_width, plot_h
        if kind == "ratios":
            series = [[] for _ in range(4)]
            for i, (s, c) in enumerate(rows):
                stats = c[policy]["comparisons"][s["prior"] if role == "prior" else "original_cuda"]
                for r, value in enumerate(stats["paired_round_ratios"]):
                    series[r].append((value, n - i - 0.5 + (r - 1) * .16))
                series[3].append((stats["representative_ratio"], n - i - 0.5))
        else:
            series = [[] for _ in range(3)]
            for i, (s, c) in enumerate(rows):
                for a, arm in enumerate(("original_cuda", s["prior"], s["candidate"])):
                    series[a].append((c[policy]["arms"][arm]["representative_us"], n - i - 0.5 + (a - 1) * .21))
        values = [v[0] for seq in series for v in seq]
        if kind == "ratios":
            values = [value for s, c in rows for stats in c[policy]["comparisons"].values()
                      for value in stats["paired_round_ratios"] + [stats["representative_ratio"], 1]]
        low = math.floor((min(values) - .04) * 10) / 10 if kind == "ratios" else 0
        high = math.ceil((max(values) + .04) * 10) / 10 if kind == "ratios" else math.ceil(max(values) * 1.05)
        plot.data = series
        plot.joinedLines = 0
        plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = low, high
        plot.xValueAxis.valueSteps = [round(low + (high - low) * i / 5, 2) for i in range(6)]
        plot.xValueAxis.labelTextFormat = "%.2f" if kind == "ratios" else "%.1f"
        plot.xValueAxis.labels.fontName, plot.xValueAxis.labels.fontSize = "Helvetica", 10
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = 0, n
        plot.yValueAxis.visible = 0
        for j in range(len(series)):
            marker = makeMarker("Diamond" if kind == "ratios" and j == 3 else "Circle")
            marker.size = 8 if kind == "ratios" and j == 3 else 6
            marker.fillColor, marker.strokeColor = palette[j], colors.white
            marker.strokeWidth = .5
            plot.lines[j].symbol = marker
        if kind == "ratios":
            equal_x = x + panel_width * (1 - low) / (high - low)
            drawing.add(Line(equal_x, plot_y, equal_x, plot_y + plot_h, strokeColor=colors.HexColor("#667085"), strokeDashArray=[4, 3]))
        drawing.add(plot)
        drawing.add(String(x, 669, heading, fontName="Helvetica-Bold", fontSize=13))
    legend = "Blue circle: round 1    Orange circle: round 2    Purple circle: round 3    Dark diamond: ratio of separate representative medians" if kind == "ratios" else "Blue: original CUDA    Orange: prior Triton    Purple: candidate    Dots: median of three process medians"
    drawing.add(String(28, 54, legend, fontName="Helvetica", fontSize=11))
    drawing.add(String(28, 33, "#27: 1 warp | #30/#32: grouped 4 rows | #35: full_groups. #35 also changes layout/stride with D; see the report.", fontName="Helvetica", fontSize=10, fillColor=colors.HexColor("#526174")))
    return drawing


def markdown(data):
    lines = ["# M07 历史性能结果：四次会话分别复算", "", "数据截止2026-10-10；本报告于2026-10-14由教练脚本从已归档原始样本重新生成。没有新增GPU运行。", "",
             "三条优化路线均未在记录范围内证实相对原版CUDA的稳定收益。规模更大并不自动意味着grouped更快；正确性、静态编译变化和性能收益分别判断。", "",
             "## 统一读法", "", "所有比值只除同一次会话、同一配置、同一政策的耗时。27/30/32/35互不合并；30与32虽然使用同一grouped源码，仍保留为两次独立会话。Graph是主结果，Eager单列，后者包含不同的主机提交路径。", "",
             "- 每个样本是100次调用的CUDA Event区间耗时÷100；每臂/政策/进程30个区间均值。",
             "- 进程中位数：该进程30个区间均值的中位数。每配置3个新进程。",
             "- 代表耗时：3个进程中位数的中位数。代表比＝参考臂代表耗时÷候选代表耗时。",
             "- 逐轮配对比：同轮参考进程中位数÷候选进程中位数；配对中位比是这3个比值的中位数。它通常不等于代表比。",
             "- 比值>1表示候选更快；候选代表耗时变化＝(1/代表比−1)×100%。不能把speedup−1当作耗时降幅。",
             "- 胜/平/负按未四舍五入的逐轮比与1比较。3/3同向只描述该会话，不能视为统计显著性或普遍保证。",
             "", "这些是重复同一批地址、三臂共享buffer、无cache flush的NVIDIA FP16微基准。分配、公开索引校验、首次JIT、预热、检查和存档不计时。Triton使用预验证raw launch；Graph包含100次调用，Eager分别走各自提交入口。不能由此推出模型吞吐、请求P99、实际DRAM带宽、硬件计数器或AMD结果。", "",
             "每worker的历史记录声明190次全backing检查，但只保存7个完整after阶段；本次只复核计时JSON、来源哈希、启动元数据和统计，不重复声称离线验证了190份数组。原始正确性数组核验见各会话既有review。", "",
             "## 配置与来源", "", "| 会话 | 同场三臂 | 配置 | 进程/区间均值 | 原始数据与原审查 |", "| --- | --- | --- | ---: | --- |"]
    for s in data["sessions"]:
        cfg = "B1/D64、B256/D128；C1024" if s["id"] in ("27", "30") else "B1/16/256 × D64/128；C1024" if s["id"] == "32" else "B5/256 × 三种布局/行距；C512"
        lines.append(f"| {s['id']}：{s['name']} | original_cuda / {s['prior']} / {s['candidate']} | {cfg} | {len(s['workers'])} / {len(s['workers']) * 180} | [原始run](../../logs/{s['run']}/status.json) · [review](../../reviews/{s['review']}) |")
    lines += ["", "27/30/32：交错缓存、gap0、guard0、int32索引0..B−1、seed20261004。35：seed20261011、唯一索引(37i+11)%512；D64为interleaved/gap0/guard0，D128为separate/gap64/guard0，D192为separate/gap64/guard64；源/目标行距分别64/128、192/128、256/256。因此35不同D之间也改变了布局/行距，不能解释为纯宽度效应。", "",
              "27候选是1warp/1行，参考v1是4warp/1行；30/32候选是4warp/4行，参考仍为v1；35两条Triton均4warp/4行，只比较满组constexpr条件分支。35的B5属于尾组对照；不能把其波动归因于满组分支加速。", "",
              "共同环境：RTX 4070 Ti，Torch 2.9.1+cu128、CUDA 12.8、Triton 3.5.1；上游commit `" + COMMIT + "`。各进程真实环境、测量次序、启动参数、前后telemetry与完整raw样本均保留在[明细JSON](historical-results.json)。", ""]
    for policy in POLICIES:
        lines += [f"## {policy.upper()}：{'主结果' if policy == 'graph' else '主机提交参考'}", "", f"![{policy}逐轮与代表速度比](figures/{policy}-ratios.svg)", "", f"[独立PDF](figures/{policy}-ratios.pdf) · [绝对耗时图](figures/{policy}-latency.svg) · [绝对耗时PDF](figures/{policy}-latency.pdf)", "",
                  "图中的三个彩色圆点是逐轮配对比，深色菱形是代表比。表内单位为µs/次；“旧Triton”指本会话参考臂。", "",
                  "| 会话/配置 | 原版 | 旧Triton | 候选 | 旧/候选代表比 | 原版/候选代表比 | 候选相对旧臂耗时变化 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for s in data["sessions"]:
            for c in s["cases"]:
                a, comp = c[policy]["arms"], c[policy]["comparisons"]
                label = f"{s['id']} · B{c['config']['batch']}/D{c['config']['width']}"
                lines.append(f"| {label} | {a['original_cuda']['representative_us']:.6f} | {a[s['prior']]['representative_us']:.6f} | {a[s['candidate']]['representative_us']:.6f} | {comp[s['prior']]['representative_ratio']:.6f} | {comp['original_cuda']['representative_ratio']:.6f} | {comp[s['prior']]['candidate_time_change_pct']:+.3f}% |")
        lines += ["", "| 会话/配置 | 参考臂 | 第1轮比 | 第2轮比 | 第3轮比 | 配对中位比 | 胜/平/负 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for s in data["sessions"]:
            for c in s["cases"]:
                for ref in (s["prior"], "original_cuda"):
                    comp = c[policy]["comparisons"][ref]
                    values = " | ".join(f"{v:.6f}" for v in comp["paired_round_ratios"])
                    lines.append(f"| {s['id']} · B{c['config']['batch']}/D{c['config']['width']} | {ref} | {values} | {comp['median_paired_ratio']:.6f} | {comp['wins']}/{comp['ties']}/{comp['losses']} |")
        lines.append("")
    lines += ["## 保留负结果与复习重点", "",
              "27的1warp候选在两配置的Graph均未稳定获益。30的B256/D128相对v1三轮同向更快，但同场对原版仅1/3轮获胜。32相同grouped源码的六配置测试中，B1/B16均慢于两条参考臂，B256有涨有跌。35满组分支的B256三个宽度仍没有相对旧grouped的3/3同向收益；尾组对照也有波动。四次会话所有Eager轮次的候选均慢于原版。", "",
              "32 · B256/D128是统计反例：Graph原版/候选代表比1.066589，看起来候选更快；但逐轮比为1.066589、0.913304、0.938370，配对中位比0.938370，只有1/3轮获胜。代表比、逐轮方向必须同时看。", "",
              "编译观察曾看到满组分支改变静态寄存器记录/行判断，但这些不是运行时硬件计数器，也不能单凭它们解释性能。需要新的对照时创建新run，写明待验证假设，再以该run同场数据判断。", "",
              "## 帮助与归因", "", "用户实际执行GPU实验。27的NUM_WARPS修改由用户完成；grouped的行号、mask、grid三式经教练示范后由用户应用并运行；full_groups分支结构由教练准备，用户完成被接受的constexpr条件。计时入口、归档、独立检查与本汇编图表由教练辅助。工程结果不等于独立掌握，不因此新增口试/独立优化信用。", "",
              "每次run原始`ownership`原文保存在明细JSON的`ownership_as_recorded`；报告保留全部负结果。", "",
              "## 重生成与追加", "", "从项目根目录运行（只读原始档案；输出限定本模块目录）：", "", "```powershell", "& 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B outputs/kv-cache-performance/modules/M07/generate-historical-results.py --self-test", "```", "",
              "统计/校验使用Python标准库，SVG/PDF使用现有ReportLab，无需Torch/Triton。可用`--data-only`只生成JSON/Markdown；`--out`可指定新的输出目录以比较重生成结果；可选`--png-node`传现有Node路径，通过相邻node_modules里的sharp生成PNG。", "",
              "新增实验时在生成器的`SESSIONS`新增一项，保留旧run、source pin与独立session id；若合同改变，新增相应配置验证，不能静默套用旧范围。先验证新run完整，再增补图表，始终按session分组。", "",
              f"本次CPU复算：{data['counts']['sessions']}会话、{data['counts']['cases']}个会话内配置、{data['counts']['workers']}个worker、{data['counts']['interval_averages']}个区间均值；{data['counts']['source_files_verified']}个归档源码条目哈希通过；所有已存代表值/逐轮比/耗时变化与原始样本复算一致。这里只统计归档记录，不代表新增GPU工作。", "",
              "[生成器](generate-historical-results.py) · [CPU自检报告](historical-validation.json) · [产物哈希](historical-artifact-manifest.json)", ""]
    return "\n".join(lines)


def self_test(data):
    spec, session = SESSIONS[0], data["sessions"][0]
    original = deepcopy(session["workers"][0])
    original["policy"] = session["plan"]
    passed = []
    def reject(name, mutate):
        w = deepcopy(original)
        mutate(w)
        try:
            validate_worker(spec, w, session["plan"])
        except (ValueError, KeyError):
            passed.append(name)
        else:
            raise AssertionError(f"negative control accepted: {name}")
    for name, mutate in [
        ("wrong capacity", lambda w: w.update(capacity=512)),
        ("wrong layout", lambda w: w["config"].update(layout="separate")),
        ("wrong grid", lambda w: w["actual_launches"]["candidate"][0].update(grid=[99])),
        ("wrong warp", lambda w: w["actual_launches"]["candidate"][0]["parameters"].update(num_warps=4)),
        ("wrong stride", lambda w: w["actual_launches"]["candidate"][0]["parameters"].update(SRC_ROW_STRIDE=17)),
        ("missing arm", lambda w: w["samples_ms_per_call"].pop("candidate")),
        ("missing policy", lambda w: w["samples_ms_per_call"]["candidate"].pop("eager")),
        ("short sample", lambda w: w["samples_ms_per_call"]["candidate"]["graph"].pop()),
        ("NaN", lambda w: w["samples_ms_per_call"]["candidate"]["graph"].__setitem__(0, float("nan"))),
        ("Inf", lambda w: w["samples_ms_per_call"]["candidate"]["graph"].__setitem__(0, float("inf"))),
        ("zero", lambda w: w["samples_ms_per_call"]["candidate"]["graph"].__setitem__(0, 0)),
        ("negative", lambda w: w["samples_ms_per_call"]["candidate"]["graph"].__setitem__(0, -1)),
        ("raw/list mismatch", lambda w: w["measurements"][0].update(ms_per_call=999)),
        ("unbalanced order", lambda w: w["measurements"].__setitem__(slice(0, 6), w["measurements"][1:6] + w["measurements"][:1])),
    ]:
        reject(name, mutate)
    saved = read_json(BASE / "logs" / spec["run"] / "summary.json")["summary"]
    for name, workers in (("missing worker", session["workers"][:-1]), ("duplicate worker", session["workers"] + session["workers"][:1])):
        try:
            summarize(spec, workers, saved)
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError(name)
    changed = deepcopy(saved)
    changed[0]["graph"]["speedup_original_over_candidate"] += .01
    try:
        summarize(spec, session["workers"], changed)
    except ValueError:
        passed.append("changed archived ratio")
    else:
        raise AssertionError("changed ratio accepted")
    control = comparison([10, 9, 30], [9, 10, 31])
    require(control["representative_ratio"] == 1 and control["median_paired_ratio"] < 1 and control["wins"] == 1, "statistic distinction lost")
    actual = next(c for s in data["sessions"] if s["id"] == "32" for c in s["cases"] if (c["config"]["batch"], c["config"]["width"]) == (256, 128))["graph"]["comparisons"]["original_cuda"]
    require(actual["representative_ratio"] > 1 and actual["median_paired_ratio"] < 1 and actual["wins"] == 1, "historical counterexample lost")
    require("torch" not in sys.modules and "triton" not in sys.modules, "GPU module loaded")
    return dict(status="CPU_CHECKS_PASSED", negative_controls=passed, negative_control_count=len(passed),
                positive_controls=["all actual raw archives and saved summaries", "ratio of medians differs from median of paired ratios", "real32 representative>1 but one win", "no torch/triton import"],
                scope="No GPU rerun or revalidation of all correctness arrays; no statistical significance claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--png-node", type=Path, help="Existing Node executable with ../node_modules/sharp")
    args = parser.parse_args()
    sessions = [load_session(spec) for spec in SESSIONS]
    data = dict(schema="m07-historical-results-v1", historical_cutoff="2026-10-10", compiled_on="2026-10-14",
                generator_sha256=digest(Path(__file__)), policy="same-session ratios only; graph/eager separate; raw samples are 100-call interval averages",
                counts=dict(sessions=len(sessions), cases=sum(len(s["cases"]) for s in sessions), workers=sum(len(s["workers"]) for s in sessions),
                            interval_averages=sum(len(s["workers"]) * 180 for s in sessions), source_files_verified=sum(s["source_file_count"] for s in sessions)), sessions=sessions)
    tests = self_test(data) if args.self_test else dict(status="ARCHIVE_RECONSTRUCTION_PASSED", negative_controls="not requested")
    tests.update(counts=data["counts"], generator_sha256=data["generator_sha256"])
    out = args.out.resolve()
    require(not out.is_relative_to(BASE / "logs"), "output must not overwrite evidence")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "historical-results.json", data)
    write_json(out / "historical-validation.json", tests)
    (out / "RESULTS-HISTORICAL.md").write_text(markdown(data), encoding="utf-8", newline="\n")
    artifacts = ["historical-results.json", "historical-validation.json", "RESULTS-HISTORICAL.md"]
    if not args.data_only:
        from reportlab.graphics import renderSVG, renderPDF
        from reportlab import rl_config
        rl_config.invariant = 1
        (out / "figures").mkdir(exist_ok=True)
        for policy in POLICIES:
            for kind in ("ratios", "latency"):
                drawing = chart(data, policy, kind)
                stem = f"figures/{policy}-{kind}"
                renderSVG.drawToFile(drawing, str(out / (stem + ".svg")))
                renderPDF.drawToFile(drawing, str(out / (stem + ".pdf")))
                artifacts += [stem + ".svg", stem + ".pdf"]
                if args.png_node:
                    sharp = args.png_node.resolve().parent.parent / "node_modules" / "sharp"
                    subprocess.run([str(args.png_node), "-e", "require(process.argv[1])(process.argv[2]).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)})", str(sharp), str(out / (stem + ".svg")), str(out / (stem + ".png"))], check=True)
                    artifacts.append(stem + ".png")
    write_json(out / "historical-artifact-manifest.json", {name: dict(sha256=digest(out / name), bytes=(out / name).stat().st_size) for name in artifacts})
    print(json.dumps(dict(status=tests["status"], counts=data["counts"], negative_controls=tests.get("negative_control_count", 0), output=str(out)), ensure_ascii=False))


if __name__ == "__main__":
    main()
