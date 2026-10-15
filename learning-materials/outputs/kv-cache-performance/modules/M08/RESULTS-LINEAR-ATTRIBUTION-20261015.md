# 47号：把主要GEMV对应到模型层和实际矩阵形状

日期：2026-10-15。**归因验证完成：3616次线性调用全部关联到各自GPU内核。** 本次没有新增加速候选；原版模型与KV内核保持原样。现在能据实测选择下一项优化对象，不再只按内核名称或调用次数猜测。

## 1. 测了什么

与42号相同的Qwen3-0.6B权重revision c1899de289a04d12100db370d81485cdf75e47ca、RTX4070Ti、FP16、TP1、单请求128输入/32生成、FlashInfer和naive缓存。新进程使用原版KV，关闭模型CUDA Graph，先两次正常预热，再进行一次独立Eager Nsight诊断。

包装器只在这次诊断生成期间给F.linear添加唯一NVTX标记，并读取实际输入、权重、输出的形状/stride及源码调用位置；数学计算委托原F.linear一次。每个调用不增加同步、GPU数值复制或磁盘写入。生成结束的全部32个原始token与已验收原版精确一致；没有宣称本次重新做了完整KV/logits逐元素验证。

本次来源、脚本、模型哈希及依赖均冻结。插桩和记录可能影响时间；**本报告是归因诊断，不是正式性能对照，更不是新版本提速结果。**

## 2. 怎样知道GPU内核来自哪一层

```text
加载后的真实权重对象 → 模型层名称
  → 本次F.linear的唯一NVTX标签＋实际形状
  ← 同CPU线程NVTX范围内的CUDA提交API
  ← 相同进程及correlation ID对应的GPU内核
```

使用CPU提交的时间归属，不能把GPU执行时刻直接塞进CPU NVTX时间范围：GPU可能在CPU函数返回后才开始执行。所有3616条线性记录各对应一个NVTX范围、一次实际GPU内核；没有未归属的已知GEMV/GEMM/CUTLASS内核。8329个其它GPU内核另存，不能误称“未识别矩阵运算”。总共11945个内核，其中896次为原版KV写入。

分析器已通过6个正例和14个反例的CPU检查，包括GPU时间在CPU范围外、不同进程同correlation ID、相关性歧义和混入Graph节点等情况。实际数据另由独立SQL审计复核。

方法依据：[Nsight Systems SQLite说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)和[NVIDIA的CPU API/NVTX关联说明](https://forums.developer.nvidia.com/t/nvtx-ranges-in-cuda-context-in-sqllite-file/221776/3)；实际实现以本机2025.1.3导出schema为准。

## 3. 实际各层的数据量与耗时

F.linear计算`Y = X @ W.T`，表中的权重为`[输出维度, 输入维度]`。这两个维度是矩阵乘的N/K，不是之前KV算子的B/D，不能直接套用。单请求decode的输入只有一行，因此属于矩阵与向量相乘的情形；prefill通常一次处理128行。

下列占比的分母是本次所有GPU内核持续时间之和 **103.501109ms**。它不是整个generate墙钟时间或关键路径，也不是旧43号Graph数据。

| 阶段 | 模型运算 | 实际输入shape | 实际权重shape | 调用数 | GPU累计ms | 本次内核时长和占比 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| decode | gate_up_proj | [1, 1024] | [6144, 1024] | 868 | 24.357800 | 23.534% |
| decode | lm_head | [1, 1024] | [151936, 1024] | 31 | 21.709375 | 20.975% |
| decode | qkv_proj | [1, 1024] | [4096, 1024] | 868 | 16.737100 | 16.171% |
| decode | down_proj | [1, 3072] | [1024, 3072] | 868 | 13.368121 | 12.916% |
| decode | o_proj | [1, 2048] | [1024, 2048] | 868 | 9.323264 | 9.008% |
| prefill | gate_up_proj | [128, 1024] | [6144, 1024] | 28 | 1.024650 | 0.990% |
| prefill | qkv_proj | [128, 1024] | [4096, 1024] | 28 | 0.792425 | 0.766% |
| prefill | lm_head | [1, 1024] | [151936, 1024] | 1 | 0.665895 | 0.643% |
| prefill | down_proj | [128, 3072] | [1024, 3072] | 28 | 0.574087 | 0.555% |
| prefill | o_proj | [128, 2048] | [1024, 2048] | 28 | 0.425604 | 0.411% |

所有线性调用累计88.978321ms，占85.968%；原版KV累计0.870507ms，占0.841%。这能支持优先研究主要线性路径的工程取舍；不能直接推出消灭KV后墙钟最多快0.841%，也不能据此宣布某层受DRAM带宽限制。

次数也已由运行核对：

- 每步28层，每层QKV、O、Gate/Up、Down四次线性，共112次，再加一次LMHead，共113次。
- 1次prefill＋31次decode，共113×32=3616次。
- 每种decoder投影：prefill28次，decode28×31=868次。
- LMHead：prefill1次、decode31次，总32次。

这次直接关联表明：四种decode投影使用相同的`gemvx ... (int)6`内核名称，但对应不同矩阵形状；LMHead使用`gemvx ... (int)7`；四种prefill decoder投影使用GEMM。不同模型运算可以共享一个内核名字，所以仅按名字汇总会隐藏优化对象。

## 4. 与旧43号的关系

旧43号是decode Graph诊断，内核创建阶段没有保留对应模块的NVTX；其Graph节点只能关联到GraphLaunch，无法可靠逐节点连到线性层。离线审查可以直接确认旧43中一次prefill LMHead，但不能把其余31次仅按次数猜为已经完成归因。

本次47在Eager模式建立了直接调用链，支持相同模型中上述层的语义与形状；**不把新Eager百分比抄成旧Graph百分比，也不冒称旧Graph每个节点已独立验证。** 两次trace的内核数量及持续时间可以不同。

## 5. 由结果决定下一项实验

优先选择**decode Gate/Up投影**做下一项有界研究：本次它累计24.358ms、868次调用、占23.534%，是累计成本最高的一类线性；形状固定为输入[1,1024]、权重[6144,1024]，便于先建立可讲清楚的GEMV基线。LMHead是第二研究对象，虽然只有31次decode调用，但权重形状更大、单次成本高。

已有实现事实：QKV已合并三个投影，Gate/Up也已合并两个投影；LMHead在prefill先选最后token，再做词表投影。因此“合并已有QKV”或“只算最后token的LMHead”不能当成本项目的新优化。

下一项计划，尚未执行：

1. 保留原版F.linear，采集实际Gate/Up输入与权重，建立独立算子对照；固定FP16权重与输出语义。
2. 区分重复同一权重与轮换28层权重的测量。单层12MiB权重可能形成不同的缓存条件，不能仅用同一地址反复跑的最好成绩替代模型场景。
3. 据真实访存/计算观察决定实现简单Triton GEMV还是研究Gate/Up GEMV与SiLU门控乘法（SwiGLU）的融合；实际后继是`silu_and_mul`，不是单独SiLU。一次只改一项，不先保证超过cuBLAS。
4. 新浮点计算须先冻结数值检查标准及高精度参照，再做正确性和多进程配对计时；最后接回模型验证。不可在看到失败后随意放宽误差界限。

Gate/Up每层FP16权重逻辑大小为6144×1024×2=12MiB，28层合计336MiB；LMHead约296.75MiB。这是逻辑存储规模，**不是本次测得的DRAM传输量**。若后续引入量化，应另立精度/模型质量契约，不能混成当前FP16等价优化。

## 6. 这一步怎样用于面试

| 层次 | 题目 | 回答要点 |
| --- | --- | --- |
| 基础 | GEMV与GEMM有什么关系？为什么prefill/decode不同？ | 看实际X和W形状；本次prefill128行、decode1行，LMHead因选最后token始终1行 |
| 机制 | 一个GPU内核为什么能出现在不同模型层里？如何定位它？ | 库函数可用同一模板处理多种形状；按相关ID连接CPU提交、NVTX和实际参数，不能只看内核名 |
| 项目 | 为什么没有继续优化KV的warp？ | 已有KV候选未稳定赢；本次KV仅占累计GPU内核时长0.841%，Gate/Up和LMHead更值得先研究；这仍不是硬件瓶颈最终归因 |
| 追问 | 为什么这份profile不能当加速证据？ | 本次只有原版、带插桩、一次Eager运行；无新候选对照，也不能把Eager比例移到Graph或墙钟 |

可按实际参与程度复述：项目先完成KV正确性、编译与性能对照，再接入真实模型。我参与分析为什么局部优化没变成模型收益；通过独立trace把主要GEMV定位到具体层和形状，随后选择成本较大的固定形状做下一轮实验。实现和采集由教练完成，本人尚需独立解释上述证据链。

## 7. 全部证据与复现

- [运行结果](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/result.json)、[逐类统计](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/attribution.json)、[独立审计](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/evidence-audit.json)。
- [3616条实际调用元信息](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/linear-calls.json)、[逐内核关联](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/attributed-kernels.jsonl)、[其它内核](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/unmatched-kernels.jsonl)。
- [原始Nsight报告](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/original-model.nsys-rep)、[原始SQLite](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/original-model.sqlite)、[工作负载输出](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/workload.json)、[冻结方案](../../logs/m08-linear-attribution-results/run-20261015-041937-ijssdrjr/plan.json)。
- [47运行器](../../47-m08-linear-attribution.py)、[SQL分析器](../../47-analyze-linear-attribution.py)、[事前计划](../../reviews/20261015-m08-linear-attribution-plan.md)、[CPU正反例检查](../../reviews/20261015-m08-linear-attribution-cpu-test.json)、[旧43号归因限制](../../reviews/20261015-m08-prior-attribution-limits.json)。

复习直接读证据即可；需要复现时，在PowerShell运行这一行，不重装环境、不改历史目录：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/47-m08-linear-attribution.sh --timing-run /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-timing-results/run-o_ltqzpd
```

本次不新增项目验收分，73/100混合权重及个人掌握状态保持；默认仍为原版，无AMD实测、硬件计数器或新提速声明。本次新增资料仅本地保存，尚未同步GitHub。
