# mini-sglang KV Cache 学习项目

**最新同步：2026-10-09。M06 v1工程产物已冻结，M07已有第一条优化预测。** 当前工程进度53/100、M06为23/25；最后的本人独立串讲待记录。限定范围内正确性已通过，尚无稳定提速结论。

## 今晚复习从这里开始

| 顺序 | 入口 | 看什么 |
| --- | --- | --- |
| 1 | [今天的开发日志](learning-materials/outputs/kv-cache-performance/obsidian/2026-10-09-开发日志-M06冻结与M07起步.md) | 今天做了什么、成绩如何、本人参与和待巩固内容 |
| 2 | [M06独立区块](learning-materials/outputs/kv-cache-performance/modules/M06/README.md) | 冻结源码、设计、正确性、编译产物与原始证据 |
| 3 | [M06性能报告](learning-materials/outputs/kv-cache-performance/modules/M06/RESULTS.md) | 全六配置、Graph/Eager、三轮波动与结论限制 |
| 4 | [M07第一课](learning-materials/outputs/kv-cache-performance/lessons/m07-01-num-warps-hypothesis.md) | num_warps从4改1的假设，候选尚未修改或运行 |
| 下载 | [M06完整ZIP](learning-materials/outputs/kv-cache-performance/modules/M06-v1-20261009.zip) | 可离线查证的完整阶段归档，包含源码和数据 |

[最新进度](learning-materials/outputs/kv-cache-performance/PROGRESS.md) · [任务表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md) · [连续日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md) · [笔记索引](learning-materials/outputs/kv-cache-performance/obsidian/学习笔记索引.md)

## 目录与当前状态

- `learning-materials/outputs/kv-cache-performance/`：本次更新到10月9日的代码、讲义、日志、M06完整原始数据与归档区。最新接受的Triton代码在[modules/M06/src/](learning-materials/outputs/kv-cache-performance/modules/M06/src/)。
- `source/`：保留2026-10-06 13:14:51 UTC（北京时间21:14:51）的WSL只读导出。它不是10月9日的新WSL导出，不能把其中早期练习当成最新v1。
- [同步范围与限制](SYNC_STATUS.md)：来源、排除项和换机前提。

| 阶段 | 证据支持的结论 |
| --- | --- |
| M05 | 六配置原版baseline v0工程交付已完成，旧成绩与真实KV时间线保持冻结 |
| M06 | 公共入口检查、共同正确性196结果（168非空GPU调用+28空返回）、实际编译产物、18进程同场对照已归档 |
| M06性能 | 没有稳定快于原版的证据；不能挑最快一轮或用Graph对Eager计算优化收益 |
| M07 | 本人预测减少warp可能更快；尚未实现或测量候选，未产生新优化成绩 |
| 模型/其他硬件 | 尚无模型端到端、AMD实机或NCU运行时计数器结论 |

本人参与K/V断言、指导下的核心表达式修改、实验执行、问题推导和初步假设；部分参数化/mask按本人请求由教练示范。环境、辅助校验、矩阵、计时组织和文档由教练协助，不能写成独立开发全部工程。分工见[教练约定](learning-materials/outputs/kv-cache-performance/COACHING_GUIDE.md)。

## 源码与运行

上游：[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，固定提交 `9a91cfafe754aa85daee49998176275667eb58f2`。[source/LICENSE](source/LICENSE)保留MIT许可证及原文件声明。

原版阅读入口：[store.py](source/python/minisgl/kernel/store.py)、[store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)、[mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)。最新v1：[内核](learning-materials/outputs/kv-cache-performance/modules/M06/src/triton_store_mask.py)、[公共入口](learning-materials/outputs/kv-cache-performance/modules/M06/src/triton_store_api.py)。

阅读无需GPU。复现需要完整上游固定版本及兼容环境，见[M06复现说明](learning-materials/outputs/kv-cache-performance/modules/M06/REPRODUCE.md)。本机证据来自RTX4070Ti、Torch2.9.1+cu128、Triton3.5.1；路径/激活脚本仍含原机器约定。本仓库Git提交不等于上游固定提交，不能删去版本检查来假装一致。

上传只同步文件，没有重新运行GPU。冻结包说明中的“尚未同步GitHub”描述打包时状态；本次发布状态以本页与SYNC_STATUS为准。
