# mini-sglang KV Cache 学习项目

**当前同步范围：官方固定提交的上游源码（121 个文件）＋已有 Windows 学习材料与证据（894 个文件）。WSL 当前启动挂起，最新本人源码、完整 `experiments/` 目录及 M05 试测原始逐次样本仍待恢复后导出。** 这里的本人代码入口是已保存的历史运行快照，不能视为当前 WSL 工作区。完整范围、来源和补齐方式见 [SYNC_STATUS.md](SYNC_STATUS.md)。

围绕 mini-sglang 的 KV Cache 写入路径，学习张量布局、原版 CUDA 实现、Triton 算子、正确性检查和性能测量。本仓库用于跨设备阅读源码、复习学习记录，以及追溯实验依据。

项目还在进行中。按截至 **2026-10-04** 的学习记录，M03 布局理解与 M04 原版契约检查已验收，M05 已完成首批原版性能试测采集；通用 Triton 实现、正式完整基线、优化收益和模型端到端验证仍待完成。

## 从哪里开始

| 想看什么 | 入口 |
| --- | --- |
| 最近做到了哪一步、哪些结论有证据 | [学习日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md)；先读顶部最新记录 |
| 整个项目要交付什么 | [项目蓝图](learning-materials/outputs/kv-cache-performance/PROJECT-BLUEPRINT-20261002.md)、[任务与验收表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md) |
| 官方上游代码与本人历史快照 | [source/](source/)、[2026-10-02 固定 64 宽 Triton 快照](learning-materials/outputs/kv-cache-performance/logs/kv-three-rows-results/run-20261002-140817-qyuxk3bo/triton_store_first.py) |
| 跟着讲义回顾基础与源码 | [lessons/](learning-materials/outputs/kv-cache-performance/lessons/) |
| 正确性审查与试测解释 | [reviews/](learning-materials/outputs/kv-cache-performance/reviews/) |
| 首批六配置试测的范围、数据和限制 | [M05 试测审查](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-review.md) |
| 教练脚手架与实验记录 | [kv-cache-performance/](learning-materials/outputs/kv-cache-performance/)、[logs/](learning-materials/outputs/kv-cache-performance/logs/) |
| CUDA / WSL 环境记录 | [CUDA 开发环境说明](learning-materials/outputs/CUDA开发环境说明.md)、[项目环境恢复记录](learning-materials/outputs/kv-cache-performance/ENVIRONMENT_RECOVERY.md) |

`PROGRESS.md` 和较早的 `SESSION-*.md` 保留了历史安排；判断最近状态时，以学习日志顶部及对应日期的审查记录为准。历史文档中的“下一步”不一定仍是当前任务。

## 目录与源码阅读路线

```text
source/                         官方固定提交的 mini-sglang 源码
learning-materials/
  outputs/
    kv-cache-performance/       讲义、教练脚手架、学习日志与证据
    ...                         CUDA / WSL 安装和修复说明
SYNC_STATUS.md                  同步范围、待补文件与导出步骤
```

上游基于 [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，学习与基线记录固定在提交 [`9a91cfafe754aa85daee49998176275667eb58f2`](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。上游源码及其附带第三方代码保留各自许可证和原有声明；本仓库中的学习整理不改变其归属。

建议按下面顺序阅读：

1. [tests/kernel/test_store.py](source/tests/kernel/test_store.py)：缓存、K/V 输入、indices、调用、检查和上游计时入口。
2. [python/minisgl/kernel/store.py](source/python/minisgl/kernel/store.py)：Python 到原版 CUDA 内核的调用封装。
3. [python/minisgl/kernel/csrc/jit/store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)：目标行选择及复制分工。
4. [python/minisgl/kernel/csrc/include/minisgl/warp.cuh](source/python/minisgl/kernel/csrc/include/minisgl/warp.cuh)：warp 内的复制方式。
5. [python/minisgl/kvcache/mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)：模型侧 KV Cache 分配与写入调用；注意其布局与交错缓存练习的差别。
6. [triton_store_first.py 历史快照](learning-materials/outputs/kv-cache-performance/logs/kv-three-rows-results/run-20261002-140817-qyuxk3bo/triton_store_first.py)及[同次运行归档](learning-materials/outputs/kv-cache-performance/logs/kv-three-rows-results/run-20261002-140817-qyuxk3bo/)：查看当时本人固定 64 宽的读写代码、检查与结果。它保留的是该次运行状态，最新 WSL 本人代码尚未导出。

## 当前成果与贡献范围

| 部分 | 当前记录支持的结论 |
| --- | --- |
| 上游原版 | 已固定版本并运行原版 store 测试；CUDA store 内核属于上游实现。 |
| 本人学习与编程 | 已有指导下的固定尺寸 Triton K/V 读写原型、K/V 比较检查和 Event 计时练习。固定 64 / 128 宽度分别通过的小例，不能合并表述为同一个通用内核已经完成。 |
| M03 / M04 | 布局与调用关系已按课堂记录验收；原版检查覆盖记录中的指定 dtype、宽度、布局和索引范围。范围与原始依据见对应审查。 |
| M05 | 首批 6 配置 × 3 轮原版试测采集完成，可作限定范围的初步参考；M05 整项仍进行中。 |
| 教练辅助 | 环境排障、讲义、模板、批量运行器、检查框架及审查整理由助手协助；这些产物不计为本人独立手写全部工程。 |

后续本人核心编写与验收约定见[手写项目执行规则](learning-materials/outputs/kv-cache-performance/手写项目执行规则-20261002.md)和[本人手写补充清单](learning-materials/outputs/kv-cache-performance/本人手写补充清单-20261002.md)。

## 怎样理解目前的性能记录

首批 M05 试测使用原版 CUDA store，范围是 B=1/16/256、D=64/128、容量 1024、FP16、交错 K/V 缓存和连续唯一索引。各进程反复访问同一批张量及目标地址，每个计时区间包含 100 次调用；它没有覆盖大工作集、冷缓存、随机索引、模型接入或本人 Triton 的性能对照。

CUDA Graph 重放是主口径，eager Python 循环提交是参考口径。两者测量的是同一原版的不同执行方式，时间差不能写成本人优化算子的加速比。首批 Graph 轮间相对跨度约 12.8%–37.5%；目前不据此判断几个百分点的提速或配置排名。

FP16 K/V 载荷的逻辑读写量为 `8 × B × D` 字节/调用，不含 indices。由该字节量除以时间得到的是有效读写带宽，不能直接当作实测 DRAM 带宽或显存峰值利用率。

已有[试测审查](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-review.md)依据用户回传控制台结果和本地脚本整理；[summary-transcribed.json](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-summary-transcribed.json)明确属于结果转录。原始逐次样本、运行快照与转录的证据层级应分别保留，不能因为同步进一个仓库就自动视为已重新验证。

## 在另一台设备上阅读与继续

只阅读源码和笔记时，可以直接在 GitHub 打开上述相对链接，或克隆仓库后用编辑器打开根目录，不需要安装 CUDA。

继续运行实验时，需要为新设备单独配置兼容的 Linux / WSL、NVIDIA 驱动、CUDA、Python 与项目依赖。先核对[环境记录](learning-materials/outputs/kv-cache-performance/ENVIRONMENT-20260929.md)和[依赖清单](learning-materials/outputs/kv-cache-performance/kernel-requirements.in)。完整本人实验目录及环境文件尚待导出，不能把当前上游 `source/` 当作已经恢复的本人运行环境；补齐流程见 [SYNC_STATUS.md](SYNC_STATUS.md)。

历史文档和脚本中仍可能包含原机器的 `/home/...`、`/mnt/c/...`、Windows 绝对路径、WSL 分发名及用户名。它们用于追溯当时环境；在另一台设备执行前，需要改为实际的仓库位置和环境配置。`source/` 是这里的上游源码根目录；将来补齐本人实验后，命令的工作目录、`PYTHONPATH` 与激活脚本中的路径也应重新核对。

仓库同步范围不包含 `.venv/`、Python / 编译缓存、编辑器缓存及 WSL 安装包、下载分片；这些内容应在目标设备重新建立。已有实验结果用于阅读和审查，不代表本次整理过程中重新执行了 GPU 测试。
