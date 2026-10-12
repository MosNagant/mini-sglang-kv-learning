# M07-01：先写假设，再改变 num_warps

**12:11新结果：** 26号正确性之后，27号两配置三方计时也已运行并核验，1-warp未稳定快于v1；后续先对照行分工和D128编译差异。见[首轮性能结果](../reviews/20261010-m07-warps-timing-review.md)，不重复下方准备或运行流程。

**10-10当前断点：** 本人已改NUM_WARPS 4→1并运行26号，196结果及36非法拒绝经独立复核通过，见[审查](../reviews/20261010-m07-warps-correctness-review.md)。无需重复下方准备/运行步骤。下一步记录B1/D64与B256/D128的预测和否定条件，准备同场性能对照；尚无候选计时或提速结论。M07-04交付4分，总57/100，独立解释另记。

## 10-10实作入口：做一次改动，再逆向复盘

用户选择项目先行、分轮复习。沿用昨天“更快，因为一个warp工作量不算多”的初稿，先准备候选并做正确性，不要求先重学清单上所有知识。下方完整假设卡可逐步补；计时前只需留下两个重点配置的预测和判定口径，允许回答不确定。

本轮候选只改变每个program的`num_warps`：4→1。仍是一次启动、`grid=(B,)`、一个program搬一条K和一条V，D、BLOCK_SIZE、indices、地址和mask保持相同。`num_warps`控制每个程序实例使用的warp数量；不决定总输入行数，也不让每行只搬32个元素。[Triton官方参数说明](https://triton-lang.org/main/python-api/generated/triton.Config.html)。本机3.5.1是否编译/执行通过以实际运行结果为准。

第一轮只要能对应这三层即可：`B=5`→5个program；`num_warps=1`→每个program配置1个warp；`D=192`→每条K/V仍搬192个有效元素。减少warp可能减少某些资源开销，也可能降低并行度，目前只是待测方向，不能写成已证明提高occupancy。

之后按产物复盘：

1. 正确性输出：解释一次launch与多个program，为什么改warp后仍要检查mask尾部。
2. 编译/地址：观察线程如何分担逻辑列，再补coalescing。`indices=[7,0,3]`决定不同program去哪个槽；是否合并访存要看同一warp同次访问的地址关系，不能仅凭槽不连续下结论。[NVIDIA访存说明](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)。
3. 同场成绩：比较4-warp与1-warp的Graph时间及三轮变化，再复盘Event/Graph、有效带宽和判定口径。
4. 面试串讲：从真实代码、一次失败/正确性反例和成绩说明“改了什么、如何判断、有哪些局限”。

以下保留实验设计和面试提醒，均不是额外的动手门槛。

### 本轮实际操作

教练已准备26号入口与候选模板，完成CPU辅助检查及独立审查；没有运行GPU。原始清单和课堂约定已保存，本轮不自动发布GitHub。

在Windows PowerShell依次执行。第一条只创建候选（已有文件保留），不运行GPU：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/26-check-kv-warps.sh --prepare
```

打开候选：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_candidate.py'
```

把`NUM_WARPS = 4`改成`NUM_WARPS = 1`，保存。模板中的公共检查与M06一致，真正的启动参数是`num_warps=NUM_WARPS`。第一次只改这一处；不改原来的mask内核、v1入口或逻辑列块。

运行正确性：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/26-check-kv-warps.sh
```

这次比较`frozen_v1=4`与`candidate=1`，不调用原版CUDA，不计时。计划98配置×两组，196项=168次非空提交+28次空返回，另36项非法调用拒绝。检查完整目标、未写存储、输入/gap、缓存guard和indices，记录真实grid/num_warps、源码与失败。B0不启动。范围为FP16/int32、C32、B0/1/2/3/4/5/17、D64/128/192及M06选定布局；后续B256计时输入还要另行正确性检查。

只有实际出现`WARPS_CORRECTNESS_PASS`且保存结果审查通过，才接受本轮正确性。通过不说明更快；下一步才是原版/v1/候选的同场测量。候选未改仍为4时，脚本在GPU导入前提示`LEARNER_CHANGE_PENDING`，不会冒认优化实验完成。

准备记录：[CPU辅助检查](../reviews/20261010-m07-warps-scaffold-cpu.json)。源码：[候选模板](../26-triton-store-candidate.py)、[运行器](../26-check-kv-warps.py)。

M06 已完成限定范围的正确性、编译观察与同场对照，v1 暂无已确认的稳定提速。证据与范围见 [M06 归档入口](../modules/M06/README.md)。本课把你的初步猜想补成一张**可检验的假设卡**，并准备一个只改变启动配置的候选；本页不代表候选已经实现或运行。

## 从已看到的分工开始

把“一行 K 和一行 V 搬到同一个目标槽”视作一份工作。目前两种实现的分工不同：

| 当前实现 | 一份启动单元怎样分工 |
| --- | --- |
| Triton v1 | 一个 program 处理一条输入 K/V 行，grid 为 `(B,)`，每个 program 使用 `num_warps=4` |
| 固定原版 CUDA | 一个 block 有 128 个线程、4 个 warp；每个 warp 处理一条输入 K/V 行，一个 block 最多处理 4 行，末尾受行数条件保护 |

这里的 NVIDIA GPU 一个 warp 有 32 个线程，因此 4 个 warp 对应 128 个线程。**program、warp、线程和逻辑列块是不同层次。** 原版依据是 [固定 store.cu](../logs/m06-paired-timing-results/run-20261009-145144-n1lwky52/original_source/python/minisgl/kernel/csrc/jit/store.cu) 中的 `warp_id`、`warp_id < length` 与 `div_ceil(length, kWarpPerBlock)`；Triton 依据是 [已运行启动器](../logs/m06-paired-timing-results/run-20261009-145144-n1lwky52/25-compare-kv-v1.py) 的 `triton_call`。

`BLOCK_SIZE` 是逻辑列位置数量，不是线程数。例如 D192 的逻辑块为 256，列 mask 保留前 192 列。改变 warp 数时，逻辑列范围仍由原来的 `tl.arange` 决定；编译器重新分配这些列由哪些线程处理，不能把 BLOCK_SIZE 跟着缩成 32。新映射仍须编译并通过正确性检查。

## 第一个实验只动一项

先比较冻结 v1 的 **4 个 warp** 与候选的 **1 个 warp**；读完结果后，如有明确问题再追加 2 个 warp。这称为启动配置调优，不把它写成已经完成的内核分工改造。

重点先看 **B1/D64** 和 **B256/D128**：一个只有一条输入行，另一个有很多较宽的输入行。已有 Graph 代表值中，v1 分别比同场原版耗时高约 5.42% 与 11.85%；后者逐轮方向还不一致。这些观察不直接告诉我们改成 1 个 warp 会怎样。

你之后在独立候选启动器中改 `num_warps`，保留冻结 v1。以下保持一致：同一批张量与地址复用方式、dtype、shape、布局、indices、grid、WIDTH/BLOCK_SIZE、地址表达式、读写 mask，以及计时边界和统计方法。B 仍是输入总行数，`pid` 才是某一行的编号。

你的第一版假设（本人原话，2026-10-09）：

> 更快，因为一个warp工作量不算多

状态：**初步假设，尚未测量或证明**。这足以选定第一候选 4→1；两个重点配置的分别预测、否定条件仍待你填写。

教练提示（不计作你已完成的解释）：可以把机制说得更具体——每行搬运工作较少，1 个 warp 可能已经够用；减少 warp 也可能降低行内并行度，所以不能保证更快。哪种作用在本例占主导，要看实验。

继续补充短卡，每格一两句即可；不确定可以写“不确定”并指出要查的证据。

| 项目 | 你的初稿 |
| --- | --- |
| 观察：我依据哪项结果和波动？ | ____ |
| 假设：warp 数可能通过什么影响本例？ | 已记录上方本人初稿；机制解释待补 |
| 单因素改动与必须保持的条件 | ____ |
| 预测：B1/D64 会怎样？ | ____ |
| 预测：B256/D128 会怎样？ | ____ |
| 否定条件：怎样的结果会让我放弃或缩小假设？ | ____ |
| 正确性风险与检查；还缺哪种证据？ | ____ |

## 卡片之后怎样推进

1. 基于你的初稿准备 4→1 候选：教练准备运行器，你完成关键启动参数改动；两个配置的预测与否定条件在测量前补齐。
2. 候选先过完整约定正确性，含空输入、完整 K/V、未命中存储、只读输入/索引及 D192 尾部 guard，再预热与计时。
3. 两个重点配置中，同场保留原版、v1 与候选。Graph 对 Graph 为主，Eager 单列；沿用每区间 100 次调用、每轮 30 个区间样本、3 轮新进程和平衡顺序。
4. 有继续研究价值时，再看全部六配置；最终保留与否要包含回退和轮间变化，不能只选最快一轮。只有针对问题的下一步才追加 `num_warps=2`。

速度比先说清比较对象：`v1 耗时 / 候选耗时` 回答这次改动的效果，`同场原版耗时 / 候选耗时` 回答相对原版的位置。100 次调用除以 100 是一个区间平均样本。

没有稳定收益也可以放弃候选并保留有效的否定结论。少几个 warp、寄存器元信息变化或约 1 微秒的耗时，都不能单独证明 DRAM、L2、occupancy 或模型端到端原因；当前没有相应硬件计数器证据，也没有 AMD 结果。

## 面试知识提醒卡

- **相关概念与项目证据：** program/warp/线程/逻辑列块；对照上面的固定源码和 M06 同场全表。编译元信息是静态信息，运行时并发需另取证据。
- **What（基础问法）：** `num_warps=4` 与 `BLOCK_SIZE=256` 分别描述什么？
- **Why（联系本项目）：** 为什么值得把 warp 数作为一个实验因素？哪些现象还不足以支持你的原因解释？
- **What if（条件变化）：** 改成 1 个 warp 而 BLOCK_SIZE 不变，剩余逻辑列会怎样分配？怎样验证没有少写？
- **易错说法：** “warp 越少一定越快”“256 个逻辑列就是 256 个线程”“占用率高就一定快”“只搬数据就一定受 DRAM 带宽限制”。
- **目标深度：** 会做单因素改动和正确性复查；会解释上述分工与证据边界；寄存器/occupancy 模型暂了解。实际是否独立解释，按你的回答与实验另记，不因读完本页自动验收。
