# mini-sglang KV Cache 学习项目

**已补齐 2026-10-06 21:14:51（北京时间）的 WSL 项目快照。** [source/](source/) 现在包含实际项目源码、本人练习、正确性检查记录和 M05 原始试测样本；原有 Windows 讲义与记录继续保留在 [learning-materials/](learning-materials/)。同步范围和排除项见 [SYNC_STATUS.md](SYNC_STATUS.md)。

**最新进展（2026-10-06）：M05 六配置原版基线的工程交付已验收。** 已补齐原始数据审计、代表配置复测和真实 KV 时间线；[基线成绩单与后续优化路线](learning-materials/outputs/kv-cache-performance/M05基线收尾与后续优化路线-20261006.md)是本次复习入口。个人理解与独立实现能力单独记录，M06 草稿已准备、尚未运行。21:23 后的新证据在 learning-materials/，source/ 仍保持上面的 21:14:51 快照。

围绕 mini-sglang 的 KV Cache 写入路径，学习张量布局、原版 CUDA 实现、Triton 算子、正确性检查和性能测量。本仓库用于跨设备阅读与复习，不是 WSL 系统镜像。

## 阅读入口

| 想看什么 | 入口 |
| --- | --- |
| 最新导出的本人 Triton 练习 | [triton_store_first.py](source/experiments/triton_store_first.py) |
| 本人 K/V 检查与计时练习 | [check_kv_targets.py](source/experiments/check_kv_targets.py)、[bench_kv_first.py](source/experiments/bench_kv_first.py) |
| M05 六配置 × 三进程原始试测 | [run-d9gjhywx/](source/experiments/learning/m05-original-pilot/run-d9gjhywx/)、[summary.json](source/experiments/learning/m05-original-pilot/run-d9gjhywx/summary.json) |
| 全部导出的实验与历史运行 | [experiments/](source/experiments/) |
| 进度、讲义和任务安排 | [学习日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md)、[任务与验收表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md)、[lessons/](learning-materials/outputs/kv-cache-performance/lessons/) |
| 试测解释与历史审查 | [reviews/](learning-materials/outputs/kv-cache-performance/reviews/)、[M05 试测审查](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-review.md) |
| 环境恢复记录 | [ENVIRONMENT_RECOVERY.md](learning-materials/outputs/kv-cache-performance/ENVIRONMENT_RECOVERY.md) |
| 文件来源与哈希清单 | [EXPORT_MANIFEST.json](source/EXPORT_MANIFEST.json)、[SOURCE_SNAPSHOT.json](source/SOURCE_SNAPSHOT.json) |

本次补充同步了 M05 收尾笔记、进度表、原始数据审计、复测与 Nsight 证据，以及 M06-01 草稿。其余历史材料保留原日期；当前状态以[最新收尾笔记](learning-materials/outputs/kv-cache-performance/M05基线收尾与后续优化路线-20261006.md)及对应审查为准。source/ 内的 SOURCE_SNAPSHOT.json 记录导出当时状态，不代表后续验收状态。

## 项目状态与贡献范围

| 部分 | 当前证据支持的结论 |
| --- | --- |
| 原版 CUDA | 基于上游固定版本，原版 store 属于 mini-sglang；已保存原版正确性检查结果。 |
| 本人代码 | 指导下完成固定尺寸 Triton K/V 读写、K/V 比较检查和 Event 计时练习。64 / 128 宽分别通过的历史例子，不等于同一通用内核已经完成。 |
| M03 / M04 | 课堂记录中的布局理解和原版契约检查已有验收依据；覆盖范围以对应运行与审查为准。 |
| M05 | 六配置原版 baseline v0 工程交付已验收：18 组原始数据审计、代表配置复测、真实 KV 时间线及复现记录已完成；具体支持范围见新报告。个人独立掌握另列。 |
| M06 / 优化 | M06-01 参数化宽度草稿及运行器已准备，尚未运行。统一正确性覆盖、公平性能对比、优化收益和模型端到端验证仍待后续。 |
| 教练辅助 | 环境排障、脚手架、批量运行器、检查框架和审查文档由助手协助，不计为本人独立编写全部工程。 |

本人编写约定见[手写项目执行规则](learning-materials/outputs/kv-cache-performance/手写项目执行规则-20261002.md)和[本人手写补充清单](learning-materials/outputs/kv-cache-performance/本人手写补充清单-20261002.md)。

## 源码阅读路线

1. [tests/kernel/test_store.py](source/tests/kernel/test_store.py)：准备数据、调用、检查及上游计时入口。
2. [python/minisgl/kernel/store.py](source/python/minisgl/kernel/store.py)：Python 到原版 CUDA 的封装。
3. [store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)：选择目标行与复制分工。
4. [warp.cuh](source/python/minisgl/kernel/csrc/include/minisgl/warp.cuh)：warp 内复制方式。
5. [mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)：模型侧缓存分配与写入调用。
6. [triton_store_first.py](source/experiments/triton_store_first.py)：对照本人固定尺寸读写原型及同目录的检查、运行档案。

上游来源：[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，WSL 导出提交为 [9a91cfafe754aa85daee49998176275667eb58f2](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。[MIT 许可证](source/LICENSE)及文件中的第三方声明保留。

## 性能数据的解释边界

M05 首批试测范围：原版 CUDA store，B=1/16/256，D=64/128，容量 1024，FP16，交错缓存，连续唯一索引；重复访问同一批张量和地址，每区间 100 次调用。CUDA Graph 重放是主口径，eager Python 循环提交是参考口径。

Graph 与 eager 是同一算子的不同执行方式，其差值不能写成本人优化的加速比。现有结果未覆盖大工作集、冷缓存、随机索引或模型接入。FP16 K/V 逻辑读写量为 8 × B × D 字节/调用，不含 indices；由此计算的有效带宽不能直接称为 DRAM 实测带宽。

此前 [summary-transcribed.json](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-summary-transcribed.json)是控制台结果转录；新补齐的 source/experiments 保存原始样本与运行快照。二者分开保留。同步过程本身没有运行 GPU 测试。用户随后执行的 M05 收尾已归档在[run-k84pq53x](learning-materials/outputs/kv-cache-performance/logs/m05-closeout-results/run-k84pq53x/)。真实 KV 时间线共定位 41 次 store 调用，其中 20 次关联到标记范围内的 CPU 启动；这项观察用于定位内核，profile 时长不替换六配置 Graph 基线。代表配置 B256/D128 复测 Graph=1.095680 微秒/次，处于旧三轮范围内，也不覆盖旧代表值。

## 换设备阅读与运行

阅读时可直接在 GitHub 浏览或克隆本仓库，无需 CUDA。运行时需要重新建立 Linux / WSL、NVIDIA 驱动、CUDA、Python 与依赖环境；虚拟环境、安装工具及缓存没有同步。

**本仓库未导入原项目 .git 历史。** 克隆后的 Git HEAD 是本备份仓库提交，source/ 也不是独立的上游 Git 仓库。历史 M05 脚本含固定上游提交校验，因此不要把它们在这里的报错误认为源码变更。要复现实验，可另行检出上述 mini-sglang 固定提交，再将对应 experiments 和教练脚本放入约定位置，核对环境与路径后运行。

历史脚本保留原机器 /home/...、/mnt/c/...、Windows 绝对路径、WSL 名称和用户名；换机时需要调整工作目录、激活脚本、PYTHONPATH 及外部脚本路径。完整排除说明见 [SYNC_STATUS.md](SYNC_STATUS.md)。
