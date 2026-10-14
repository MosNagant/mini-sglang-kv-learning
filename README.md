# mini-sglang KV Cache 学习项目

**2026-10-14：M07工程收尾，默认保留原版CUDA。** Triton v1、1-warp、grouped和full_groups保留为研究版本；已有证据未支持候选晋升为稳定更快的通用默认。36号访存实验与37号独立配置验证已经实际运行，完整数据和报告分别归档。M08真实模型接入与端到端验证仍未完成。

按原任务表，本次仅验收M07-06报告的3分：工程总分**64/100，M07为11/25**；其余14分个人复盘待验收。这些是工程验收权重，不是知识掌握百分比，也不表示学员独立完成了本轮实现和分析。

## 从这里复习

| 入口 | 内容 |
| --- | --- |
| [M07模块总览](learning-materials/outputs/kv-cache-performance/modules/M07/README.md) | 实验路线、最后的选版决定、支持范围与后续依赖 |
| [实现变化](learning-materials/outputs/kv-cache-performance/modules/M07/IMPLEMENTATION.md) | 原版、v1及三种候选分别改了什么 |
| [历史完整成绩](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-HISTORICAL.md) | 27/30/32/35四次会话、48个worker、8640区间均值与图表 |
| [36号访存实验](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-MEMORY.md) | 固定形状下的布局、stride、对齐与地址复用对照 |
| [37号选版验证](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-SELECTION.md) | 保守选择规则、独立配置与原版A/A波动 |
| [面试复习框架](learning-materials/outputs/kv-cache-performance/modules/M07/INTERVIEW-REVIEW.md) · [复现与离线核验](learning-materials/outputs/kv-cache-performance/modules/M07/REPRODUCE.md) | 用实际代码和数据回答问题；重建报告和运行检查 |
| [10-14独立开发日志](learning-materials/outputs/kv-cache-performance/obsidian/2026-10-14-开发日志-M07收尾.md) | 环境恢复、各节点的执行、核验与归档 |
| [M06冻结区](learning-materials/outputs/kv-cache-performance/modules/M06/README.md) · [M06性能报告](learning-materials/outputs/kv-cache-performance/modules/M06/RESULTS.md) | 冻结v1、原始对照与复现范围 |

[当前进度](learning-materials/outputs/kv-cache-performance/PROGRESS.md) · [任务表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md) · [连续日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md) · [笔记索引](learning-materials/outputs/kv-cache-performance/obsidian/学习笔记索引.md)

## 本轮实际结果

| 节点 | 已完成的实测与结论 |
| --- | --- |
| 27/30/32/35历史汇编 | 重新读取旧原始样本，四次会话分别计算；没有为汇编重跑GPU。三条候选路线均未证实稳定超过原版。 |
| 36：固定B256/D128/C1024的六条件访存实验 | **18个新进程、3240个区间均值**。Graph变化未达到本课事前“三轮同向且每轮至少3%”信号门槛；8字节输入偏移改变了编译读取形态，但未定位稳定速度因果。 |
| 37：冻结选版与新配置验证 | **9个新进程、1080个区间均值**，新B为7/64/257。默认决策均为原版，两臂调用同一原版入口；完整保存数组正确。A/A波动不是Triton加速，也不是选择器执行耗时。 |
| 默认与后续 | 保留原版默认，候选不晋升。融合需要M08真实调用链；真实模型接入、端到端验证独立待完成。 |

10-14两批实测合计27个新进程、4320个计时区间均值，每区间100次调用。Graph为主要政策，Eager单列；预热、首次编译、完整检查与存档在计时外。代表中位数比和逐轮配对比分别阅读，不跨会话、跨配置或混合Graph/Eager计算收益。37的选择器在计时外执行，3%是工程筛选门槛，不是统计显著性检验。

所有结论限定于报告列明的FP16、int32合法唯一索引、设备、形状、行距与布局。重复地址和四套轮换均不等于冷DRAM测试。没有新增AMD实机、NCU运行时计数器、模型tokens/s、请求P99或整模型加速结论。

## 源码、快照与证据

- `learning-materials/outputs/kv-cache-performance/`保存完整学习材料，包括36/37原始数据、源码快照、NPZ、编译产物、独立审查、报告和旧失败记录。当前研究源码入口在[M07 src/](learning-materials/outputs/kv-cache-performance/modules/M07/src/)。
- `source/`以**2026-10-14 10:53:20 UTC / 18:53:20 Asia/Shanghai**的真实WSL只读导出更新：**6457个普通文件、85,646,985字节，29项排除、1个符号链接**。逐项SHA-256、大小和Unix模式已经核验；Git保留可执行位与`.dockerignore → .gitignore`的120000模式。
- 这个快照**早于36/37两批新实验**。之后的运行器、策略、结果与审查以`learning-materials/`对应归档为准；不能称为两批实验之后的完整WSL状态。若补入快照后的学习文件，其来源单列于[本次源码来源记录](source/SOURCE_SNAPSHOT_20261014.json)。
- [10-14原始导出清单](source/EXPORT_MANIFEST_20261014T105320Z.json)描述此次快照。未出现在新导出中的旧路径和10-06、10-12来源记录保留为历史，不能因此认定它们都是10-14现状。范围见[同步状态与历史](SYNC_STATUS.md)。
- 本轮恢复GPU入口的[存储修复日志](learning-materials/outputs/logs/repair-storage-20261014-185022.log)独立归档；成功恢复并执行本次实验，不代表反复启动问题已永久解决。

原版入口：[store.py](source/python/minisgl/kernel/store.py)、[store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)、[mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)。上游为[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，固定提交`9a91cfafe754aa85daee49998176275667eb58f2`；[MIT许可证](source/LICENSE)与原文件声明保留。

## 归因与复现边界

10-14的新增实现、GPU执行、统计、核验和报告由教练根据用户明确授权完成。此前用户实际运行、参数修改与条件填写的贡献仍按原记录保留；grouped三处核心表达式曾由教练示范。不能把本轮工程产物直接记为本人独立实现或口述掌握。后续按[教练约定](learning-materials/outputs/kv-cache-performance/COACHING_GUIDE.md)记录回答与提示程度，再验收个人复盘的14分。

阅读无需GPU。已保存实测环境为RTX4070Ti、Torch2.9.1+cu128、CUDA12.8、Triton3.5.1；执行前按[M07复现说明](learning-materials/outputs/kv-cache-performance/modules/M07/REPRODUCE.md)核对环境、固定上游版本与原机路径。本整理仓库HEAD不等于上游提交，不能删掉版本校验来假装一致。虚拟环境、模型权重和整机镜像不在本次备份范围；M07微基准不能替代M08。
