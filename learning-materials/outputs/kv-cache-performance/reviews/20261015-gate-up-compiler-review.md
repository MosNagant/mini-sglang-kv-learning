# 49 号 Gate-Up 实验：实际编译结果与机制解释

审查日期：2026-10-15。只读分析已有源码、TTGIR、PTX、编译资源和独立 CPU 审计；本审查没有运行 GPU，也没有修改候选。

证据来自 [49 号完整运行目录](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/)。固定输入为 FP16 `(1,1024)`，权重为 FP16 `(6144,1024)`，前 3072 行为 gate、后 3072 行为 up；设备为 RTX 4070 Ti，Triton 3.5.1。本次编译快照取自第 0 层、decode step 1 的实际调用；结论只适用于保存的特化。

## 1. 先看实际资源，而不是猜测

| 候选 / kernel | 寄存器/线程 | 共享内存/CTA（字节） | `n_spills_raw` | 资源记录 |
|---|---:|---:|---:|---|
| direct r4 / w4 | 38 | 64 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/compiled.json#L5) |
| direct r8 / w4 | 42 | 128 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w4/compiler/compiled.json#L5) |
| direct r8 / w8 | 38 | 2048 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/compiled.json#L5) |
| split-K4 / partials | 22 | 512 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/compiled.json#L5) |
| split-K4 / reduce | 20 | 2048 | 0 | [compiled.json:19](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/compiled.json#L19) |
| fused r4 / w4 | 41 | 64 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/compiled.json#L5) |
| fused r8 / w4 | 47 | 128 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r8_w4/compiler/compiled.json#L5) |
| packed fused r4 / w4 | 42 | 64 | 0 | [compiled.json:5](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_packed_r4_w4/compiler/compiled.json#L5) |

这里已经存在真实的共享内存指令；此前 KV 搬运内核的共享内存为 0，与当前 GEMV 归约内核是不同事实。当前共享内存由编译器用于归约或布局转换，源码没有显式声明 shared 数组也可以产生这些指令。**存在 shared memory 不等于存在 bank conflict。**

这些是加载后特化的静态资源，不是 achieved occupancy、实际 spill 流量或带宽计数器。`n_spills_raw=0` 按运行时原始字段记录，不臆测其单位；split-K 两个 kernel 的共享内存不能相加成一个 CTA 的占用。

## 2. Direct：row tile 与 warp 并不是同一个旋钮

`direct r4/w4` 和 `r8/w4` 的 [TTGIR 布局:1](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ttgir#L1) 都是：

```text
sizePerThread = [1,8]
threadsPerWarp = [1,32]
warpsPerCTA = [1,4]
order = [1,0]
```

两个轴是输出行和归约维 K。这一布局把 4 个 warp 放在 K 维协作；不能把“4 行、4 warp”解释成一个 warp 固定负责一行。r4→r8 增加每个 program 的输出行数，program 数从 `6144/4=1536` 降到 `6144/8=768`，同时增加每个 program 的工作和中间结果。

实际 PTX 中，r4 的输入 `ld.global.v4.b32` 在 [59 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L59)，权重同类加载在 [101 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L101)，随后出现 [shuffle:209](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L209)、[shared store:295](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L295)、[barrier:309](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L309) 和 [shared load:314](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r4_w4/compiler/kernel0.ptx#L314)。这里同时观察到了线程内累加、warp 内交换以及 shared/barrier 协作。

`r8/w8` 的 [布局:1](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/kernel0.ttgir#L1) 则为 `warpsPerCTA=[2,4]`。它的输入加载后还有 [convert_layout:31](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/kernel0.ttgir#L31)：PTX 先 [global v2.b32:62](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/kernel0.ptx#L62)，再 [shared store:72](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/kernel0.ptx#L72)、barrier、[shared load:77](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_direct_r8_w8/compiler/kernel0.ptx#L77)。因此加到 8 warp 改变了数据分工和布局转换，并非仅增加一倍相同的工人。

可调查的假设是：program 数、每线程工作、归约同步和资源驻留之间存在权衡。当前静态结果只能证明这些机制改变了，不能认定其中哪一项主导了实测时间。尤其不能把每线程负责更多列直接换算成同倍数的机器指令。

## 3. Split-K：缩短单次归约，同时增加另一个归约阶段

[源码:185](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/candidates.py#L185) 将 K=1024 分成 4 份，每份 256。第一阶段 grid 为 `(1536,4)`，第二阶段 grid 为 `(24,)`，把 FP32 partials 合起来再舍入成 FP16。它使用额外 `(4,6144)` FP32 scratch：每层 98,304 字节、28 层共 2,752,512 字节。

第一阶段 [TTGIR:1](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel0.ttgir#L1) 是 `warpsPerCTA=[4,1]`，与 direct 的 `[1,4]` 不同。其 [PTX:137](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel0.ptx#L137) 有 shuffle，且 [176 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel0.ptx#L176) 将 FP32 partial 写回 global memory。第二阶段从 [PTX:62](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel1.ptx#L62) 加载 partials，经 shared/shuffle，再在 [158 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel1.ptx#L158) 转为 FP16并写回。

注意第一阶段 `st.global.b32` 是一个 FP32 partial，第二阶段 [176 行的 b32 store](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/gemv_splitk4_r4_w4/compiler/kernel1.ptx#L176) 则存两个打包的 FP16；`.b32` 表示 32 位，单看后缀不能判断数值类型。

这里可证实的代价是额外 kernel、scratch 读写和第二次归约；可证实的改变是更短的分片归约及更多 program。哪一方占优取决于输入与实测，不能由“寄存器从 38 降到 22”直接推断。对完整 Gate-Up+SiLU 路径，direct 仍需独立激活 kernel，split-K 则是两次 GEMV 阶段加一次激活；编译资源表没有把 FlashInfer 激活计入这些 Triton kernel。

## 4. Fused：少一次独立激活与中间张量往返，但单 kernel 做更多事

fused 同一个 program 求 gate/up 两组点积，并完成 SiLU(gate)×up；r4/r8 分别启动 `3072/4=768`、`3072/8=384` 个 program。相较 direct，它消除了独立激活调用及用于该调用的 Z global 写回/再读取，但仍需读取两部分权重。

数值语义没有偷偷把中间 FP16 舍入去掉。fused r4 的 [PTX:574](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L574) 将 gate 的 FP32 归约结果转为 FP16，[579 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L579) 再转回 FP32；[591 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L591) 使用 `ex2.approx.f32`，[604 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L604) 有除法；up 同样先 [614 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L614) 舍入，再在 [624 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L624) 相乘。最终输出为 FP16。

r4→r8 的寄存器从 41 增至 47，共享内存从 64 增至 128 字节；同时保存的 PTX 中出现更多展开的归约和激活指令。它减少 program 数，但提高每个 program 的工作量。编译结果支持“融合确实发生”，不支持“融合一定更快”或“数值逐位等价”。

## 5. Prepacking：物理权重布局改变，不等于自动得到更宽访存

[源码:164](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/candidates.py#L164) 新建 `(3072,2,1024)` 的连续权重并复制，将原来整块 gate、整块 up 改成每行 gate/up 配对。它是实际分配与复制，不只是修改 view。计时区间外的 setup 约 21.6–21.8 ms / 28 层，增加权重副本 336 MiB；不能在系统收益分析中忽略这部分常驻空间。

packed 的 [PTX 权重加载:100](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_packed_r4_w4/compiler/kernel0.ptx#L100) 与普通 fused 的 [105 行](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/numerical/swiglu_fused_r4_w4/compiler/kernel0.ptx#L105) 都是 `ld.global.v4.b32`；两者都使用 64 字节共享内存，寄存器为 42 对 41。实际观察没有显示 packed 引入了更宽的权重加载。布局可能影响相邻地址和缓存行为，但需要运行时证据，不能把它直接描述成访存合并已经改善。

## 6. 指令形态汇总及解释边界

下表统计保存 PTX 中的**静态指令位置数**，不代表动态执行次数、内存事务数或每次调用耗时；展开、谓词、循环及最终 SASS 都会影响实际执行。

| 候选 / kernel | global load | global store | shared load / store | barrier | shuffle | ex2 |
|---|---:|---:|---:|---:|---:|---:|
| direct r4/w4 | 5 | 1 | 6 / 6 | 4 | 22 | 0 |
| direct r8/w4 | 9 | 1 | 10 / 10 | 4 | 42 | 0 |
| direct r8/w8 | 5 | 1 | 8 / 10 | 6 | 22 | 0 |
| split-K partials | 2 | 1 | 5 / 2 | 3 | 5 | 0 |
| split-K reduce | 2 | 1 | 9 / 10 | 4 | 4 | 0 |
| fused r4/w4 | 9 | 1 | 11 / 11 | 7 | 44 | 4 |
| fused r8/w4 | 17 | 1 | 19 / 19 | 7 | 84 | 8 |
| packed fused r4/w4 | 9 | 1 | 11 / 11 | 7 | 44 | 4 |

这 8 份候选 PTX 没有检出 MMA/WMMA/WGMMA/TCGEN 指令；保存的实现主要是 FP32 乘积和归约。该观察不描述原版 F.linear 的实现，也不表示所有 GEMV 都不能使用 Tensor Core。

## 7. 编译证据怎样与实测结论衔接

[独立 CPU 审计](../logs/m08-gate-up-experiments/run-20261015-044242-mca3s3gj/summaryaudit.json) 已重算全部 588 项数值验收、原始数组及 4320 个计时间隔。七个候选均通过预先规定的数值门槛；这是容差验收，不是逐位一致。`np.array_equal` 本身也不会区分正零和负零，因此不能把普通值相等报告改成 bitwise equality。

在预定主目标“28 层权重轮换、Graph、step 1”中，选出的 fused r8/w4 每轮相对原版仅降低约 2.16%、2.51%、3.10% 延迟，没有达到预定的明显优势标准；step 16/31 的 6 次 holdout 中 5 次更慢。反过来，重复第 0 层同一权重的 hot-L0 诊断中，fused r4/w4 的延迟降低约 33.2%–40.5%。两种执行条件可以同时成立，不能用 hot-L0 替换预先声明的主指标。

机制层面可以说：融合确实减少了独立激活和中间 Z 往返；测试范围内它的收益强烈依赖权重复用场景。尚不能说：轮换权重时已经测出了 DRAM 带宽瓶颈、bank conflict 或 occupancy 不足。后续是否补采计数器应由要区分的具体假设决定。

面试表达可以用三句完成：

1. 我比较了行分块、warp 分工、Split-K、激活融合及权重预排布，保存了实际编译产物和资源。
2. 我能指出优化改变了哪些计算、同步、访存和 setup 成本，同时区分编译证据与硬件瓶颈假设。
3. 融合在重复同一层权重时有明显局部收益，但真实层轮换与未参与性能选择的步数没有稳定优势，因此没有把候选宣称为通用替换或模型加速。
