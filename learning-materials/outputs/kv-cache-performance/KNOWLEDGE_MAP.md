# AI Infra 知识地图：围绕 Triton 算子项目学习

最后更新：2026-10-02，统一执行版 v2。配套排期为 [每周任务安排](ROADMAP-TRITON-20261024.md)，实际证据见 [学习日志](LEARNING_LOG.md)。调整前的知识地图保存在 history/20261001-before-operator-replan/KNOWLEDGE_MAP.md。

## 最终要能做什么

在固定 mini-sglang 提交上，用自己编写的 Triton 内核完成 KV Cache 写入，通过独立正确性验证，与原版 CUDA 实现公平比较，依据数据迭代，最后接入单卡推理并讲清效果与限制。学到的每个知识点应落在源码、测试、数据或解释里。

算子是操作的约定：输入是什么、如何产生结果。内核是 GPU 上执行这一操作的实现。当前操作就是根据 indices 保存已有 K/V；它可以用 CUDA C++ 或 Triton 实现。不是先完成一门 CUDA 课程才有资格写算子。

## 从底层到模型的学习深度

| 层次 | 本项目需要做到 | 深度边界 |
| --- | --- | --- |
| GPU 执行与存储 | 用并行分工、地址分布、缓存和启动开销解释真实实验 | 不学芯片电路设计；指标未采到就不冒充测量 |
| Triton 内核 | 自己写 program 分工、偏移、mask、load/store；换尺寸后仍能解释 | 本月最主要的原创代码；初稿和错误由用户提出，教练审查 |
| CUDA 环境与原版 CUDA C++ | 使用既有 NVIDIA 环境；读懂原版数据契约、关键地址/映射；保存对照 | 不要求重写整套 CUDA 实现或逐行掌握辅助模板；独立 CUDA 变体后移 |
| 编译与运行时 | 区分源码、JIT/特化、生成 GPU 代码、启动执行；展示本人内核的一种实际编译产物或元信息 | 不将使用 Triton 或查看 PTX 称为修改 LLVM/MLIR 编译器 |
| Python/PyTorch 实验层 | 准备 GPU 数据，构造独立期望值，调用内核，验证和测量 | 不展开自动求导、训练循环等与当前写入任务无关的课程 |
| mini-sglang 与 SGLang 背景 | 找到 KV 分配、索引来源、store 调用和后续使用；对照完整 SGLang 的请求/批处理/KV 管理 | mini-sglang 实验不能写成完整 SGLang 工程经验 |
| 模型推理 | 解释 KV 为何复用、prefill/decode，实际测局部变化与整机 tokens/s | 不做预训练、多卡服务或新模型研究；四周不等于掌握全部 AI Infra |

## 必须保留、随代码补齐的知识

| 主题 | 为什么直接影响自己的算子 | 当前证据与下一次验证 |
| --- | --- | --- |
| Tensor、shape、dtype、device | 决定数据在哪、每行多少元素、占多少字节 | 已跑 (8,2,128) float16 小例；单位及任意形状推导仍待应用 |
| view、stride、指针偏移 | 逻辑相邻行不一定在存储中紧邻；算错会读写错误元素 | 用户能解释当前输入K连续、缓存K/V交错及两个cache指针不同起点；任意布局仍待应用 |
| indices 与写入契约 | 决定源行对应哪个目标；不同实现必须做同一件事 | 10-02经反馈完成通用断言，原版与本人Triton的[5,1]、[7,3]、[7,0]六组实测通过；当前固定形状下的目标映射已有数据 |
| program、块状索引、mask | 决定一次工作实例负责哪些元素、尾部是否越界 | 10-01本人首版固定用例运行通过，本例分工/行距解释在指导后通过；mask尚未实现；不把program当成固定32线程warp |
| thread/warp/block 与访存 | 用于理解原版及生成执行方式、分析并行与地址分布 | 用户已有当前 CUDA 分组的局部解释；随实际性能问题深化，不重开整门课 |
| 异步、同步、预热与 JIT | 调用返回不等于 GPU 完成，首次编译会污染时间 | 已用 synchronize 等结果；计时方案和核心仍需用户先写 |
| 正确性与反例 | 目标写对和其他位置未被破坏都要证明 | 10-02六组目标/未命中/输入不变检查通过，五个CPU故意错误被拒绝；形状/布局边界套件与独立解释继续推进 |
| 重复测量与公平对照 | 才能区分真正收益、波动、布局或缓存条件差异 | 无正式基线；必须保存原始样本和统一协议，报告全部形状 |
| 性能假设与生成代码 | 代码改动必须对应可验证预测，参数变化不能代替机制解释 | 第3周结合实际数据；代码观察与硬件计数器、真实时间分别标注 |
| 框架接入与模型结果 | 独立微基准提速不等于模型真的调用或整体提速 | 第4周 Qwen3-0.6B；需验证实际路径、输出及 tokens/s |

## 当前已有的工程成果

- 现有 WSL/GPU/开发环境和上游测试可运行；固定提交为 9a91cfafe754aa85daee49998176275667eb58f2。
- 用户已在同一完整实验中完成四条目标整行断言，未命中循环经初稿反馈和指导修正后通过。
- 证据为 logs/complete-kv-20261001-152947-V3FHSB.log；限 [5,1]、容量8、batch=2、宽度128这一组。
- 10-01 16:59指导下完成的本人Triton首版已通过一个固定用例，见logs/first-triton-20261001-165942-W4powA.log。通用正确性套件、可信性能基线和真实模型验证仍未完成；运行通过不等于独立掌握设计。
- 10-02 12:46同一组用户通用断言用于原版CUDA与本人Triton，各三组索引全部通过，包含缓存位置0与7；全量结果和22个源码哈希已复核。见logs/kv-index-check-20261002-124631-JPVPBT.log及reviews/20261002-kv-index-check-review.json。仍仅batch2/width128/当前布局，没有计时。

## 三行已验证；当前围绕真实布局、统一检查与计时推进

10-02 13:06，原版CUDA与本人Triton各通过一组三行[7,0,3]，grid=(3,)、width128及既有布局保持。用户正确判断pid0/1/2与行距不变，定义混入调用参数的问题已修正。两种实现全量目标/未命中/输入值和22个源码哈希已复核，见reviews/20261002-kv-three-rows-review.json。支持证据目前是batch2和3的列举用例，不等于任意形状。

14:08用户用13号`--width 64`完成原版/Triton固定64宽两组GPU检查；全部值及22个源码哈希已复核，见reviews/20261002-width64-pass-review.json。之前64内核配128数据的失败已定位并修复，用户内核在修复前后相同。64与128仍分别是固定常量版本，14号参数模板没有GPU结果。下一组工程成果为真实width/stride参数接口→统一正确性→计时入口。已核验固定版本MHAKVCache的K/V分离布局，练习中的缓存2×WIDTH行距不能照搬；输入实际布局和精度待模型运行确认。详见[项目蓝图](PROJECT-BLUEPRINT-20261002.md)。当前无正式计时、性能迭代或模型运行证据。

## 四周内后移的内容

独立 CUDA C++ 变体、复杂模板与内联汇编、手写 GEMM/FlashAttention、LLVM/MLIR Pass、ROCm 后端、多 GPU、分布式训练和生产集群都不设为本月作品前置要求。C++ 基础/面试训练按用户原安排继续，遇到本项目阻塞再补对应概念。

了解编译链是当前必需知识；编译器研发是另一种成果。只有 RTX 4070 Ti 的实测数据，不能证明 AMD/ROCm 性能或兼容性。具体岗位是否要求更深入编译器能力，仍须结合完整 JD 校准。

## 10-01 面经阅读：把术语对应到实际问题

用户提供摩尔线程 AI Infra 面经截图，反馈读来像天书。该记录把内存管理、GPU 性能诊断、算子设计和特定架构压缩在一起，不能直接作为零基础教学顺序，也不能代表所有岗位的统一考纲。当前已有指导下固定用例正确性证据，尚无正式计时和独立性能诊断证据；讲解以下概念不计为掌握。

| 面经主题 | 用普通话解释它在问什么 | 与当前作品的关系及顺序 |
| --- | --- | --- |
| H2D、pinned/pageable memory | CPU 内存里的数据如何交给 GPU，传送时 CPU 能否继续做事 | 当前 KV 源与目标已在 GPU，内核本身不是 H2D；后续用单独传输实验理解，不加为10月5日基线前置任务 |
| memory throughput、memory-bound、roofline | 慢在搬数据、计算还是别的开销；离可达到的速度还有多远 | 当前最直接相关：先数有效读写量、明确计时范围、重复测量；小 batch 可能受启动等开销影响，不能仅因代码主要复制就认定所有形状都受带宽限制 |
| register、occupancy、active warps、latency hiding | 一批工作等数据时，是否有其他工作可以执行，资源够不够 | 测到具体性能问题后学习；occupancy 是驻留 warp 数相对硬件上限的比例，不是 GPU 利用率，也不是越高一定越快；当前 NCU 计数器仍不可用 |
| GEMV、reduction、tiling、shared memory | 矩阵的每一行如何与同一向量逐项相乘后求和，怎样分工并减少不必要的数据搬运 | 是另一类计算任务，可作为 KV 后的扩展；本月不新增手写 GEMV 必需交付。shared memory 除复用外还可重排数据改善访存，不能将截图的简写理解成只有复用才有用途 |
| Hopper/Blackwell、TMA、pipeline | 特定硬件如何把批量搬运和计算组织成重叠执行的流程 | 当前4070 Ti为Ada，Hopper引入的TMA后移；不把“普通LDG一定打不满带宽、必须TMA”作为通用结论 |
| Amdahl's Law | 只加速整段流程的一小部分，最终能省多少总时间 | 后期模型验证直接相关。假设总耗时100ms，其中KV写入10ms，且其他部分保持不变；写入减半后总时间95ms，而非50ms。这是解释用假设，不是实测 |

后续每组学习采用“提出实际问题 → 用户写代码或预测 → 运行并保留证据 → 用户解释结果”，避免只逐个背词。10月5日前主线仍是通用检查和原版KV写入可信基线，之后推进自己的通用Triton实现、相同条件的对照与优化；当前首版已是DIY起点。

核对资料（2026-10-01）：[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)（页锁定传输、合并访存及共享内存的作用；页锁定本身不保证传输与计算重叠）；[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)（occupancy与roofline）；[Ada调优指南](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)及[Hopper调优指南](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)（架构差别）。在线文档用于概念核对，不代表升级本机软件或取得新的硬件指标。

## 理解与进度的记录方式

逐项记录“未学 → 已讲待验证 → 能解释当前例子 → 能独立实现 → 换输入也能推导”。交付权重见 LEARNING_LOG，不按对话条数或知识术语计分。阶段结束用3–5个与本人代码/数据相关的问题检查；日常以一组可运行产物推进，不把每行代码拆成一个漫长阶段。

## 参考资料

- [Triton 官方介绍](https://triton-lang.org/main/index.html)及[首个内核教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)：2026-10-01核对，用于 program、偏移、mask、读写和验证；只取当前需要的部分。文档 main 不等于本机3.5.1，具体行为以本机实测为准。
- [PyTorch 2.9 CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)：异步、同步与计时的背景，贴近现有2.9.1版本。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：采集成功时才解释对应指标；当前 NCU 计数器不可用，Nsight Systems 时间戳精度限制继续记录。
