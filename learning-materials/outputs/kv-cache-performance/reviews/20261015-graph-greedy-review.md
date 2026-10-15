# 55 号实现审查与实验结论

2026-10-15。候选 [graph_greedy.py](../stable/graph_greedy.py) 已完成实际正确性检查与固定三轮计时，**数值通过，性能晋升门槛未通过，不替换 53 号稳定默认方案**。

## 实施范围

- 保留原版模型、KV、FP32 Graph logits buffer、`torch.argmax(..., dim=-1)` 与 `int32` 转换。将后两项纳入原版 decode Graph 的捕获，不是把它们融合成一个新 GPU kernel。
- 固定 `graph.py`、`engine.py`、`sample.py` SHA256。原 capture 方法只在两处原有 logits 赋值后增加 greedy 运算：一次 warmup、一次实际 capture。没有额外 logits copy。
- 原版和候选均使用同一份 53 号 runtime 的受控副本，只插入一个导入后、Engine 初始化前的显式 hook。原始 minisgl 与 53/54 文件没有被修改，差异保存在每次运行的 `runtime-hook.diff`。
- captured token tensor 由 runner 保活，每次 replay 更新。Sampler 仅消费上一次实际 replay 的一次性 ticket，并检查 shape、stride、dtype、device、data pointer、storage offset、sampler 配置和当前 stream。消费后清空 ticket；未消费又开始 replay 会拒绝。
- 限定 B1、TP1、禁用 overlap、单 runtime 顺序请求。prefill 和非 greedy 等不匹配调用进入原版并记录原因；验收要求所有目标 decode 命中，不能用静默回退充当候选结果。
- Engine 的既有 `.to(int32)` 在候选返回 int32 时不再发起类型转换。原 scheduler 在同一 engine stream 处理 token pool 和 D2H，完成同步后才进行下一轮，支持本实验中持久 token buffer 的生命周期。

候选命中路径同时绕过了原 `Sampler.sample` 的装饰器与内部 NVTX range。因此本实验比较的是“采样捕获＋分发守卫＋NVTX 路径变化”的整体实现。不能把最终时间差唯一归因为 CPU 提交次数减少。

## CPU 与实际正确性证据

[26 项 CPU 控制](20261015-graph-greedy-controls.json) 检查原源码 SHA、两处 AST 插入及还原、metadata 与 stream 守卫、一次性票据消费。[8 项协议控制](20261015-graph-greedy-protocol-controls.json) 包含单个正负零位差、非有限值、B 请求 token 不同、两臂虽然相同但 A→B→A 的第二个 A 改变等拒绝用例。

实际 [strict 运行](../logs/m08-graph-greedy-check-results/run-20261015-053822-8uw6n0md/result.json) 在同一 runtime 顺序运行 A→B→A，B 是固定 128 个输入 ID 的逆序。每请求保留 32 步全部 logits 和 token；跨臂比较 **14,585,856 个 FP32 元素，位差为 0**，三个请求 token 均一致，两臂各自 A 回归也一致。

实际路由为 93 replay、93 次 captured-token 命中、3 次 prefill 原版采样，无遗留 ticket。独立 CPU 审计还确认 B 确实改变 logits 与 token，覆盖了改变输入时的旧结果复用风险。该结果仅说明测试输入与实现范围内等价，不是广泛的模型质量评估。

## 固定三轮性能结果

每臂每轮一个新进程，2 次 warmup 后计时 5 次 A 请求；同场原版 whole-decode Graph 为比较基线。计时覆盖持久 runtime 的完整 generate 和 GPU 完成，不包含加载、源文件核验、编译或 capture。两臂都使用相同轻量 replay 计数包装。

| 轮次 | 原版 Graph 中位数 ms | 候选中位数 ms | 候选延迟降低 |
|---|---:|---:|---:|
| 1 | 130.001557 | 131.723237 | -1.324% |
| 2 | 135.592818 | 130.224344 | +3.959% |
| 3 | 133.888234 | 131.739459 | +1.605% |

[原始计时运行](../logs/m08-graph-greedy-timing-results/run-20261015-054044-oru9f4py/result.json) 保存 6 个进程、30 个样本及逐次输出和路由。预先声明的门槛为每轮中位延迟至少降低 3%；本次未达到。没有重选有利轮次、放宽门槛或继续反复跑到正结果。

[源码一致性检查](20261015-graph-greedy-source-match.json) 确认当前候选、55 runner/shell、53 原版快照与 strict/timing 两份冻结源码共 8 项 SHA 比较一致，CPU 控制记录也匹配当前源码。完整 worker stdout/stderr 保存在各运行目录，控制器核心输出另存为 [console 摘录](20261015-graph-greedy-console.txt)。

这次实验证明可以在保持本范围数值不变的条件下扩大 Graph 捕获范围，但尚未证明它比已经启用整段 decode Graph 的原版稳定更快。53 号默认路径保持原样，55 号作为可复现研究候选保留。
