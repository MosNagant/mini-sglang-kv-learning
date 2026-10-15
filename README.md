# mini-sglang KV Cache / Triton 性能学习工程

从一个 KV Cache 写入算子出发，逐步学习地址计算、GPU 并行分工、正确性验证、微基准、真实模型接入和性能归因。仓库同时保存实现、实验原始记录、失败结果、学习笔记与面试复习材料。

**2026-10-15 当前交付：原版数学 + 整段 Decode CUDA Graph 稳定运行入口。** 在固定 Qwen3-0.6B 工作负载中，三轮完整请求时延相对相同 API 的原版 Eager 降低 **40.23%～44.03%**；全部 32 步 logits 和生成 token 验证一致。

收益来自启用并验证 mini-sglang 已有的整段 Decode Graph 功能。自写 Triton KV/GEMV 候选保留为研究版本；本项目没有证据支持“自写 Triton 通用地超过原版 CUDA/Graph”。

## 先看什么

| 入口 | 内容 |
| --- | --- |
| [稳定版本使用说明](learning-materials/outputs/kv-cache-performance/stable/README.md) | 可直接运行的入口、环境约束、复用方式和验收结果 |
| [最新完整结果报告](learning-materials/outputs/kv-cache-performance/modules/M08/RESULTS-STABLE-RUNTIME-20261015.md) | 53/54 稳定版与 55 号研究候选的实现、检查、计时和决定 |
| [当前进度](learning-materials/outputs/kv-cache-performance/PROGRESS.md) · [任务表](learning-materials/outputs/kv-cache-performance/KV算子项目任务表.md) | 当前验收状态；带日期的旧条目保留为历史 |
| [学习笔记索引](learning-materials/outputs/kv-cache-performance/obsidian/学习笔记索引.md) | 从概念、代码和实验反向复习 |
| [连续开发日志](learning-materials/outputs/kv-cache-performance/LEARNING_LOG.md) | 每个节点做了什么、保存了什么、为什么继续或停止 |

阅读这些材料不需要 GPU。工程验收与个人独立掌握分别记录，任务表中的权重不作为“知识掌握百分比”。

## 已验证的完整请求收益

比较对象是**同一个持久化运行接口**的两种执行方式：原版 Eager，以及原版数学/KV 不变的整段 Decode Graph。Prefill 都使用原版 Eager。

固定条件：RTX 4070 Ti、Qwen3-0.6B、FP16、TP=1、单请求、128 个输入 token、32 步 greedy 生成、`ignore_eos=True`；Graph batch size 为 `[1]`。

| 独立进程配对 | 原版 Eager，毫秒/请求 | 稳定入口 Graph，毫秒/请求 | 时延下降 | Eager / Graph |
| --- | ---: | ---: | ---: | ---: |
| 1 | 223.293499 | 133.471155 | 40.23% | 1.6730× |
| 2 | 231.264929 | 129.428888 | 44.03% | 1.7868× |
| 3 | 227.794499 | 134.785158 | 40.83% | 1.6901× |

每格为该新进程五次完整请求的中位数；共三对新进程、30 个原始样本，每个进程预热两次。计时包含调度、采样、原始 token 记录、输出处理与 GPU 完成等待；模型加载、文件校验、编译、Graph 捕获及预热均在计时外。

严格检查在独立进程中保存双方全部 32 步 logits，共比较 **4,861,952 个值**。统一保存为 FP32 后逐位相同，最大绝对误差为 0；32 个生成 token 完全一致。Graph 实际执行 31 次 decode replay。真实 WSL 项目的默认入口也已运行通过。

- [严格检查结果](learning-materials/outputs/kv-cache-performance/logs/m08-stable-runtime-check-results/run-20261015-052859-ufwymynd/result.json) · [独立审计](learning-materials/outputs/kv-cache-performance/logs/m08-stable-runtime-check-results/run-20261015-052859-ufwymynd/independent-evidence-audit.json)
- [全部时间样本与逐轮结果](learning-materials/outputs/kv-cache-performance/logs/m08-stable-runtime-timing-results/run-20261015-053033-nlwp0fei/result.json) · [独立审计](learning-materials/outputs/kv-cache-performance/logs/m08-stable-runtime-timing-results/run-20261015-053033-nlwp0fei/independent-evidence-audit.json)
- [实际项目入口运行](learning-materials/outputs/kv-cache-performance/logs/m08-stable-runtime-live-smoke/run-20261015-053233/result.json) · [发布清单](learning-materials/outputs/kv-cache-performance/stable/release.json)

这组数据证明的是本设备、固定负载下的执行方式收益；不能据此声称超过原版 Graph、改善任意输入、提高多请求服务吞吐，或在 AMD GPU 上获得相同效果。

## 项目如何走到这里

| 阶段 | 工程内容 | 要掌握的知识与结论 |
| --- | --- | --- |
| M01–M04 | 环境恢复、真实 KV 路径、固定案例与输入契约 | `B` 是输入行数、`D` 是有效列数；stride、gap、目标 indices 决定真实地址；正确写入还需验证未写区域和输入不变 |
| M05 | 建立原版 CUDA 算子基线，分开 Eager/Event 与 Graph 测量 | 首次编译和预热不进计时；小工作集重复地址测试不等于 DRAM 带宽测试；历史基线保持冻结 |
| [M06](learning-materials/outputs/kv-cache-performance/modules/M06/README.md) | Triton v1 参数化宽度/行距、mask、空输入与非法输入处理、编译观察和同场对照 | 能搬对数据，再讨论是否更快；设备 kernel 与带检查的 Python 接口需要分开测量 |
| [M07](learning-materials/outputs/kv-cache-performance/modules/M07/README.md) | Warp 数、每 program 多行、完整组特化、访存布局与保守选版 | 线程分工、任务粒度、静态编译资源和运行效率之间需要实测；候选未获通用默认晋升 |
| [M08](learning-materials/outputs/kv-cache-performance/modules/M08/README.md) | 真实模型接入、全链路检查、Nsight 归因、调用层优化、Gate/Up 实验与稳定交付 | 从真实工作负载定位主要成本；微基准收益、模型数值要求和完整请求收益分别验收 |

M06 的学习形状与 M08 模型实际输入不同。真实模型中的 KV 输入是 `D=1024`，需要按实际行距适配，不能直接把早期 `D=64/128/192` 的练习成功当作模型适配完成。

## 研究结果，包括没有晋升的版本

### KV 与 M07：保留对照和负结果

1-warp、四行 grouped、full-groups 等候选在部分配置中相对 v1 有改善，但没有形成稳定胜过原版 CUDA 的通用结论。完整组特化曾减少静态寄存器数，访存布局也改变过 PTX 指令形态；这些现象都不能单独证明运行时瓶颈已经消除。

原版因此保持默认，所有候选和失败记录仍可用于复习。这里的价值是提出假设、控制变量、保留不利结果，并明确优化适用范围。

- [M07 实现变化](learning-materials/outputs/kv-cache-performance/modules/M07/IMPLEMENTATION.md)
- [历史结果](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-HISTORICAL.md) · [访存实验](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-MEMORY.md) · [独立选版验证](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-SELECTION.md)

### M08：先找主要成本，再优化

[47 号实际调用归因](learning-materials/outputs/kv-cache-performance/modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)将 3616 次 `F.linear` 调用对应到实际形状与 GPU 内核。在这次独立 Eager 跟踪中，decode Gate/Up 占 GPU 内核累计时长约 23.53%，KV 约 0.84%。这是累计 GPU 内核时长份额，不是整个请求的墙钟占比，也不直接移用于 Graph。

[调用层元信息缓存](learning-materials/outputs/kv-cache-performance/modules/M08/RESULTS-HOST-PATH-20261015.md)减少过 CPU 封装成本，但模型对照没有获得稳定收益。项目随后转向实际占时较高的 Gate/Up，而非继续盲目扫 KV 参数。

### Gate/Up：局部通过，模型数值拒绝

[七路线实验](learning-materials/outputs/kv-cache-performance/modules/M08/RESULTS-GATE-UP-20261015.md)覆盖行块、warp、Split-K、点积与激活融合、真实权重重排；使用 28 层真实权重和 84 组真实输入，保存全部 588 项局部数值检查。

融合 r4 在**单层同权重复用的 Graph 微基准**中降低约 33%～41% 耗时；在预先确定的 28 层权重轮换主目标中，收益明显缩小。选中的 r8 在留出数据六轮中五轮更慢。

接入真实模型后，32 个 greedy token 相同，但 **989,536 / 4,861,952 个 logits** 超过事前设定的 `atol=0.002, rtol=0.002` 门槛。因此该候选被拒绝，模型计时没有继续，容差没有放宽。

这说明局部数值通过、argmax 相同、完整 logits 一致、模型质量等价是不同结论。数值分解与舍入审查保留了进一步证据，但没有冒认模型误差传播的唯一原因已经确定。

### 55：采样捕获候选没有替换稳定版

55 号将原版 greedy argmax 和 int32 转换纳入 decode Graph。A→B→A 三次请求的 96 步、14,585,856 个 logits 跨臂逐位一致，实际路由检查通过。

它相对原版 Graph 的三轮时延变化为 **+1.324%、−3.959%、−1.605%**，没有达到预设的逐轮收益门槛，所以不启用为默认。候选同时改变了 Python 分发/NVTX 路径，不能将差异唯一归因于提交次数。

详情与所有数据见[稳定版及 55 号报告](learning-materials/outputs/kv-cache-performance/modules/M08/RESULTS-STABLE-RUNTIME-20261015.md)。54 的 40%～44% 收益不能记到 55 或自写 Triton 名下。

## 目录与证据

```text
source/                                      上游源码与学习项目快照
learning-materials/outputs/kv-cache-performance/
├── stable/                                  已验收运行入口、使用说明、发布清单
├── modules/M06/、M07/、M08/                  阶段报告、冻结实现与复现说明
├── lessons/                                 小步骤练习与教学目标
├── obsidian/                                独立笔记、开发日志和复习索引
├── logs/                                    逐次运行记录、源码快照、数据与编译产物
├── reviews/                                 代码审查、CPU 核验、独立审计
├── releases/                                精简可用版本压缩包
├── PROGRESS.md / LEARNING_LOG.md             当前入口与连续历史
└── KV算子项目任务表.md                       工程验收与个人复盘状态
SYNC_STATUS.md                               同步范围、来源和历史
```

`source/` 是快照，最新实验的运行器和证据以 `learning-materials/` 对应归档为准。快照日期、补入文件及本次同步边界见 [SYNC_STATUS.md](SYNC_STATUS.md)；本仓库 HEAD 与上游提交不是同一概念。

模型权重、虚拟环境与整机镜像不随仓库分发。48 号夹具中的 28 个 `weights/layerNN.npy` 是模型派生权重，本次不上传；其它实验数组保留。精确排除路径和哈希见根目录 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。[稳定版 ZIP](learning-materials/outputs/kv-cache-performance/releases/stable-runtime-20261015.zip)只包含运行入口、53/54 脚本、说明和 JSON 结果，不含大型原始 logits NPZ。精简包可以复习结果，重新验证时应使用完整实验目录或重新运行检查。

## 复现：先恢复记录的环境

当前实测环境是 WSL Ubuntu 24.04、RTX 4070 Ti、Torch `2.9.1+cu128`、CUDA `12.8`、Triton `3.5.1`、FlashInfer `0.5.3`。其它依赖版本由发布入口和实验计划核验。

- 上游固定提交：`9a91cfafe754aa85daee49998176275667eb58f2`。
- 模型：`Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，真实 FP16 权重。
- 已验证项目路径：`/home/weltlaw/projects/mini-sglang-kv`，环境激活入口 `experiments/activate.sh`。
- 已验证模型路径：`/home/weltlaw/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca`。

下面在**已具备上述环境的 WSL 终端、此仓库根目录**执行。克隆资料仓库不会自动安装依赖或下载权重；入口会检查固定源码与版本，环境不同需重新验证。

```bash
LAB="$PWD/learning-materials/outputs/kv-cache-performance"
source /home/weltlaw/projects/mini-sglang-kv/experiments/activate.sh
python "$LAB/stable/stable_infer.py"
```

默认运行保存的 128/32 固定输入并核对生成 token。输出是推理结果，不是计时报告。可在启动前明确使用 `--execution eager`；Graph 初始化失败会报错，不会在 GPU 初始化后静默重建另一个 Engine。

也可以复用一次初始化得到的运行对象，避免重复加载模型和捕获 Graph：

```python
# 先把 learning-materials/outputs/kv-cache-performance/stable 加入模块路径。
from stable_infer import StableRuntime

with StableRuntime() as runtime:
    first = runtime.generate()
    second = runtime.generate()
    runtime.synchronize()
```

只核对文件/包版本、不创建 GPU Engine：

```bash
python "$LAB/stable/stable_infer.py" --verify-only
```

完整复测先运行正确性检查：

```bash
bash "$LAB/54-verify-stable-runtime.sh" --mode check
```

只有末尾出现 `STATUS=STABLE_RUNTIME_CORRECTNESS_PASS` 才继续。将下面的路径替换为**这次新生成且通过检查的 `RUN_DIRECTORY`**：

```bash
CHECK_RUN="/home/weltlaw/projects/mini-sglang-kv/experiments/m08-stable-runtime-check/本次通过的run目录"
bash "$LAB/54-verify-stable-runtime.sh" --mode timing --accepted-check "$CHECK_RUN"
```

脚本每次创建独立运行目录，保存首个错误及全部样本。GPU 实验应串行运行。其它设备需自行恢复路径和记录的环境；本仓库没有宣称完成跨设备复现。详细步骤见 [M07 复现](learning-materials/outputs/kv-cache-performance/modules/M07/REPRODUCE.md)与 [M08 复现](learning-materials/outputs/kv-cache-performance/modules/M08/REPRODUCE.md)。

## 如何用这个项目复习面试

建议每轮只选一个真实现象，按“基础概念 → 机制追问 → 代码与证据 → 适用边界”组织回答。

| 问题 | 本项目可以拿出的依据 |
| --- | --- |
| B、D、stride、gap、indices 各自决定什么？ | 地址公式、交错/分离缓存、padding 检查和真实模型布局 |
| 为什么减少 warp 或 program 数量不一定更快？ | M07 同场计时、尾组 mask、TTGIR/PTX 和静态资源记录 |
| GPU Event 与 Graph 测量有什么区别？ | M05 提交间隙诊断，Graph/Eager 分开的完整样本 |
| 微基准为什么不能直接代表模型速度？ | 同权重复用与 28 层轮换、局部收益和模型留出结果 |
| token 相同为什么不能代替数值检查？ | Gate/Up 的模型拒绝结果与 54 的全 logits 一致结果 |
| 优化没成功时，项目价值在哪里？ | 假设、控制变量、数值门、负结果、版本决定与最终可用入口 |

[M07 面试复习](learning-materials/outputs/kv-cache-performance/modules/M07/INTERVIEW-REVIEW.md) · [M08 性能与面试拆解](learning-materials/outputs/kv-cache-performance/modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md) · [知识地图](learning-materials/outputs/kv-cache-performance/KNOWLEDGE_MAP.md)

静态寄存器数、PTX 访存形态和时间变化不能替代运行时硬件计数器。本机相关 Nsight Compute 查询曾返回 `ERR_NVGPUCTRPERM`，未据此宣称实测 occupancy、DRAM 带宽或缓存命中率；目前没有 AMD 实机结果。

## 来源、许可与作者分工

上游为 [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)，固定提交见上文。随附上游代码保留 [MIT License](source/LICENSE)及原版权声明。原版入口可从 [store.py](source/python/minisgl/kernel/store.py)、[store.cu](source/python/minisgl/kernel/csrc/jit/store.cu)、[mha_pool.py](source/python/minisgl/kvcache/mha_pool.py)开始阅读。

学习者参与了早期地址、检查、mask/grid/参数练习、实际运行与结果讨论；具体帮助程度按原始日志记录。后续经明确授权，教练完成了实验脚手架、部分核心示范、诊断、GPU 执行、统计、审计和整理。

上游实现、教练代办工程和学习者独立掌握不会混记。个人面试准备继续依照[教练约定](learning-materials/outputs/kv-cache-performance/COACHING_GUIDE.md)，通过自己的解释、推导和修改来核验；工程产物完成不等于个人已能独立复现全部工作。
