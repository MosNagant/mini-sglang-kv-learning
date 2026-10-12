# mini-sglang KV Cache 学习项目

**本次更新：2026-10-12；实验与验收状态截至2026-10-10第35号。** 当前工程进度61/100，M06为23/25、M07为8/25。三条优化路线均有正确性与计时记录，但未证明稳定超过原版CUDA；没有晋升新的默认加速版本。M08真实模型接入与端到端验证尚未完成。

## 从这里复习

| 入口 | 内容 |
| --- | --- |
| [10-10独立开发日志](learning-materials/outputs/kv-cache-performance/obsidian/2026-10-10-开发日志-M07三条优化路线与结果复盘.md) | 三条路线、35号完整成绩、本人参与和下一步 |
| [35号完整组特化计时复盘](learning-materials/outputs/kv-cache-performance/reviews/20261010-m07-full-groups-timing-review.md) | 六配置、18新进程、3240区间均值与逐轮配对结论 |
| [32号grouped完整六配置复盘](learning-materials/outputs/kv-cache-performance/reviews/20261010-m07-grouped-six-timing-review.md) | 小规模退化、B256局部信号及不稳定性 |
| [27号1-warp复盘](learning-materials/outputs/kv-cache-performance/reviews/20261010-m07-warps-timing-review.md) | 第一条路线的否定结果 |
| [M06冻结区](learning-materials/outputs/kv-cache-performance/modules/M06/README.md) · [M06性能报告](learning-materials/outputs/kv-cache-performance/modules/M06/RESULTS.md) | 冻结v1、支持范围、原始对照与复现说明 |

[当前进度](learning-materials/outputs/kv-cache-performance/PROGRESS.md) · [任务表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md) · [连续日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md) · [笔记索引](learning-materials/outputs/kv-cache-performance/obsidian/学习笔记索引.md)

## 已验证结果与限制

| 阶段 | 当前结论 |
| --- | --- |
| M05 | 原版六配置baseline v0已完成并冻结 |
| M06 | v1共同正确性、编译观察与六配置同场计时已归档；独立串讲待记录 |
| 1-warp候选 | 未获稳定收益，保留负结果，不替换v1 |
| grouped候选 | 部分B256配置有局部改善，小B退化；尚不足以支持稳定超过原版，不全局替换 |
| full_groups候选 | 完整组编译简化已观察；35号未支持稳定收益，不替换旧grouped |
| M08及其他硬件 | 真实模型接入、端到端验证未完成；无AMD实机、NCU运行时计数器或整模型加速结论 |

Graph为主要计时口径，Eager单列。代表中位数比与逐轮配对比分别阅读，不用不同轮次或Graph/Eager交叉计算收益。静态寄存器、PTX与布局变化不自动说明性能原因。61/100是工程验收权重，不是知识掌握百分比。

本人运行GPU实验、修改参数和完整组条件，并参与问题分析；grouped三处核心表达式曾由教练示范后应用。运行器、检查、计时组织、证据审查与文档由教练协助，不能描述为独立完成全部工程。见[教练约定](learning-materials/outputs/kv-cache-performance/COACHING_GUIDE.md)。

## 源码和证据放在哪里

- `learning-materials/outputs/kv-cache-performance/`同步本地完整学习材料，保留历史失败、原始NPZ、编译产物、日志和旧归档；本次没有新建巨型ZIP。
- `source/`保留10-06的完整WSL导出，并在`source/experiments/`补入七份已验证学习源码。六份来自35号快照，1-warp候选来自26号快照；[覆盖清单与逐文件来源](source/SOURCE_OVERLAY_20261012.json)明确列出哈希和日期。
- 这七份文件是可追溯的源码更新，**不是10-12重新导出的完整WSL状态**。其他`source/`历史文件不因本次同步成为最新版本。
- [同步范围与历史记录](SYNC_STATUS.md)保留10-06、10-09说明；旧文档中的当时状态按其日期阅读。

最新阅读入口：[公共入口](source/experiments/triton_store_api.py)、[v1内核](source/experiments/triton_store_mask.py)、[1-warp候选](source/experiments/triton_store_candidate.py)、[grouped](source/experiments/triton_store_grouped.py)、[full_groups](source/experiments/triton_store_full_groups.py)。候选留档不代表默认采用或稳定加速。

上游：[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，固定提交`9a91cfafe754aa85daee49998176275667eb58f2`；[MIT许可证](source/LICENSE)与原文件声明保留。原版入口：[store.py](source/python/minisgl/kernel/store.py)、[store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)、[mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)。

阅读无需GPU。执行需完整固定版本上游及兼容环境，见[M06复现说明](learning-materials/outputs/kv-cache-performance/modules/M06/REPRODUCE.md)和各课运行记录。已保存证据来自RTX4070Ti、Torch2.9.1+cu128、Triton3.5.1；部分路径仍按原机器约定。此整理仓库的Git提交不等于上游固定提交，不能删去版本校验来假装一致。虚拟环境、模型权重、整机镜像不在备份范围。

本次同步未重新运行GPU，也不增加工程分数或个人掌握记录。下一阶段需完成真实模型与真实KV输入验证，已有微基准结果不能替代M08。
