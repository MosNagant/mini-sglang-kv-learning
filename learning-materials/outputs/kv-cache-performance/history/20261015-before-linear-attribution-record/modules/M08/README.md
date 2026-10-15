# M08：真实模型接入与同条件对照

**2026-10-15更新：调用层优化已完成一轮验证。** 元信息缓存让CPU封装消融约降60%，真实模型没有稳定改善，默认原版保持。先读[本轮完整结果](RESULTS-HOST-PATH-20261015.md)、[整个项目的性能诊断与面试拆解](PERFORMANCE-DIAGNOSIS-20261015.md)，然后按[本轮复现入口](REPRODUCE-HOST-PATH-20261015.md)查原始证据。新增工作由教练完成，73/100和个人掌握状态不变；本次新增资料尚未同步GitHub。

| 新节点 | 产物 | 实际验收 |
| --- | --- | --- |
| 44 完整元信息缓存 | [新适配器](../../44-m08-kv-cached-metadata.py)、[CPU契约检查](../../reviews/20261015-m08-metadata-cache-cpu-check.json) | 同一40号GPU内核；完整签名、64项FIFO、不缓存旧指针 |
| 45 CPU消融与GPU检查 | [360区间均值](../../logs/m08-host-diagnostic-results/run-bqojhqoc/summary.json)、[独立审计](../../logs/m08-host-diagnostic-results/run-bqojhqoc/evidence-audit.json) | CPU路径六组约降60%；完整Graph与替换对象检查通过，不是模型或GPU加速 |
| 46 真实模型检查 | [结果](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/result.json)、[独立审计](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/host-path-evidence-audit-v2.json) | 两臂1792次KV检查；完整缓存、保存logits和token一致 |
| 46 三臂模型计时 | [90样本](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/result.json)、[独立审计](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/host-path-evidence-audit-v2.json) | 18新进程；Eager/Graph均未稳定改善；候选不晋升 |

下方38–43号为10-14已验收的历史基础，原始数据与范围保留。

2026-10-14：**真实模型接入和固定负载对照已完成，尚未证明稳定性能优势。默认原版CUDA保持。** M08-01～05工程9/10分，M08-06本人独立解释1分待复习；工程完成与掌握分开。新增实现、执行和报告由教练完成。

先读[完整结果](RESULTS.md)，再读[今日独立开发日志](../../obsidian/2026-10-14-开发日志-M08真实模型接入.md)。[复现与资料入口](REPRODUCE.md)记录文件和单行命令；已有结果无需重跑。

| 节点 | 实际产物 | 本轮结论 |
| --- | --- | --- |
| 38 环境/真实权重 | [安装回执](../../logs/m08-setup-results/dependency-install-ir_ipwxe/result.json)、[权重来源与哈希](../../logs/m08-setup-results/model-0d7ke3_g/result.json) | 新增11个缺失依赖，既有包保持；固定Qwen3-0.6B revision真实权重 |
| 39 原版模型 | [输出](../../logs/m08-model-smoke-results/run-20261014-143514-x0uhkeqq/result.json)、[KV trace](../../logs/m08-model-smoke-results/run-20261014-143514-x0uhkeqq/kv-trace.jsonl)、[审计](../../logs/m08-model-smoke-results/run-20261014-143514-x0uhkeqq/evidence-audit.json) | 5输入/8生成；224次真实KV调用，原版运行成功 |
| 40/41 候选接入 | [源码](../../40-m08-kv-backends.py)、[检查结果](../../logs/m08-model-check-results/run-20261014-144754-mmdz752v/result.json)、[审计](../../logs/m08-model-check-results/run-20261014-144754-mmdz752v/evidence-audit.json) | 真实D1024、源行距4096、缓存行距1024；三臂共2688调用检查通过 |
| 42 同场对照 | [全部60次样本](../../logs/m08-model-timing-results/run-o_ltqzpd/summary.json)、[冻结计划](../../logs/m08-model-timing-results/run-o_ltqzpd/plan.json) | 12新进程；Graph无稳定收益，Eager三轮均慢 |
| 43 实际份额 | [结果](../../logs/m08-model-profile-results/run-uml9cxd9/result.json)、[原始跟踪](../../logs/m08-model-profile-results/run-uml9cxd9/original-model.nsys-rep) | 896次KV内核累计0.812ms，占GPU内核累计时长0.782%，不等于墙钟占比 |

主工作负载：RTX4070Ti、Qwen3-0.6B、FP16、TP1、1请求、128输入/32生成、FlashInfer、naive prefix cache、无overlap；Graph仅decode尺寸1。39号更小的首次运行单独记。41号为Eager严格数据诊断，42号Graph复验的是实际输出token，未完成同等级Graph全缓存快照复核。无模型质量、全框架、高并发或AMD提速声明。

下一项：把主要GEMV与KV相邻操作对应到实际调用/形状，再选择优化对象；RoPE/写入融合仅是待评估方案。M05/M06/M07历史证据与失败记录保留。
