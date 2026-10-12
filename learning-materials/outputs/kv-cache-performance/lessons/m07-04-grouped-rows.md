# M07第二个假设：一个program处理4行

日期：2026-10-10。**运行后更新：12:59的29号34项限定正确性已独立核验，无需重复运行。** 23源码哈希、实际参数及136数组389,300元素通过；三处表达式由教练示范后用户应用，独立掌握未验收，性能尚未测量。见[结果复盘](../reviews/20261010-m07-grouped-correctness-review.md)。以下保留教学设计及操作步骤，运行前假设卡另保存在本次run中；本课不修改冻结v1或1-warp候选。

## 为什么开始改这里

上一轮只把每program的warp数从4改1，Graph结果没有稳定收益。固定原版源码中一个block最多处理4行，而当前Triton v1每program只处理1行。这给出一个新的可检验方向：**保留4个warp，将每program的任务扩大到4行。** 原版的分工是灵感来源，不能直接认定这是原版较快的原因，也不声称新Triton与原版具有相同线程映射。

| 比较项 | 冻结v1 | 新grouped候选 |
| --- | --- | --- |
| 每program配置的warp | 4 | 4 |
| 每program负责的输入K/V行 | 1 | 最多4 |
| B=256的program数 | 256 | 64 |
| B=5的program数 | 5 | 2 |
| 本次总搬运量 | 相同 | 相同 |

两者一次非空公共调用仍各提交一次KV内核。program减少不等于Python调用次数减少，也不表示所有program一定同时运行。

这是改变逻辑工作分组的一项实验。分组可能连带改变编译器安排的线程映射、指令和资源用量；这些不能假定保持不变。

## 先看5行输入

假设B=5，输入行编号为0、1、2、3、4。组内编号为0、1、2、3：

| program编号pid | 生成的输入行号rows | 可以读写的行 |
| --- | --- | --- |
| 0 | 0、1、2、3 | 全部 |
| 1 | 4、5、6、7 | 只有4 |

以前`pid`恰好也是输入行号；现在`pid`是**组编号**。实际输入行号要由组编号和组内编号共同计算。每一行仍通过`indices[该输入行号]`找自己的目标槽，不要求四个目标槽挨在一起。

5、6、7这三行根本不存在：既不能读取它们的indices，也不能读取K/V或写缓存。原有列mask继续保护D=192时的第192～255列；新增行mask保护最后不足4行的情况。

## 本人改三处，其他由教练接好

新文件：WSL项目的`experiments/triton_store_grouped.py`。

| 待填写项 | 要表达的规则 | 提示 |
| --- | --- | --- |
| `rows` | 这一组的起点加上组内编号 | 用`pid`、`ROWS_PER_PROGRAM`和`tl.arange` |
| `row_mask` | 只允许真实存在的输入行 | 行编号从0开始，与`BATCH`比较 |
| `grid` | 有几组就开几个program，剩余不足4行也算一组 | `triton.cdiv(a, b)`表示向上取整的除法；一维grid保留逗号 |

教练提供二维地址广播、受保护的indices读取、行列mask组合、四个load/store及沿用的公共合同。三处之外无需改动；二维广播稍后结合实际代码解释，不要求这一步默写整个内核。若本人填写与检查器预期写法不同但语义合理，先审查源码，不为了通过检查盲改。

核心所有权明确：本人完成三处关键表达式，教练实现其余新代码；看到完整代码或CPU模拟通过不表示已独立实现、GPU编译或性能验收。

## 本课检查什么

先跑小范围边界检查，不计时：

- B=1、3、4、5、17：覆盖不足一组、恰好一组、多一行、多个整组加尾行。
- D64交错无gap；D128分离、input gap64；D192分离、input gap64及cache guard64，各配上面的五种B，共15个非空配置。
- 两个B0配置：公共入口直接返回，不提交KV内核。
- 每配置比较冻结4-warp v1与新4-warp grouped候选，共34条结果，计划30次非空KV提交和4次空返回。
- 两臂各用独立分配、同内容输入和调用前独立期望；检查全部缓存、未写槽、guard、完整输入含gap及indices，并记录实际grid/参数。

mask源码审查与结果检查互补。本课没有运行时非法读取检测工具，也没有全98配置回归或计时；不能因为结果相等就声称证明所有读取都合法。后续性能配置B256/C1024仍需另做其真实输入的正确性检查。

## 操作顺序

在自己的PowerShell中，先创建新文件（已有文件会保留，不启动GPU）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/29-check-kv-grouped.sh --prepare
```

再打开新文件：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_grouped.py'
```

找到三处`TODO`，只替换三个`...`，保留缩进。先把这三行贴回审查，确认后运行下列检查：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/29-check-kv-grouped.sh
```

实际启动参数保存在各case的`submissions.json`中。输出沿用旧helper的`WARPS_CASE_PASS`前缀，臂名`frozen_v1`/`grouped`区分两份实现；`submissions=1`表示调用一次KV内核，并非只处理一行。只有全部完成才会出现`GROUPED_CORRECTNESS_PASS: 34/34`。发生错误就保留首错与本次目录，不靠重复运行改变结果。

## 事前假设与判读

本人对B256的事前预测原文：**“暂时不确定，但我认为应该更快。”** 尚未给出具体机制理由，B1方向未定；不自动代填为已经解释了瓶颈。

教练提出：B较大时减少program数量可能有益，但每program工作增加，不能保证更快；B很小时没有program数量收益，且可能有更多无效逻辑行。这是教练的待测解释，不记为本人独立推导。

本课先证明搬对。通过后，才以同场Graph对照检查是否比v1快、是否胜原版；Eager单列，保留全部轮次和退化。若再次无稳定收益，保留否定结果，不强行宣称优化成功。

## 面试知识提醒卡

来源：教练根据本项目设计的迁移题。

- **What：** program编号、输入行号、grid、行mask与列mask分别管什么？
- **Why：** 为什么B=5每组4行时不能只启动一个program？为什么只保护store还不够？
- **What if：** B=4与B=5分别启动几组？最后一组哪些行能读取indices？
- **易错：** 4个warp不自动等于每warp一行；program少不自动更快；空输入返回不是内核处理了空输入。
- **目标：** 能修改行号、mask、grid并解释尾行；线程的实际分配待编译证据，不提前记为掌握。

机制核对：[Triton官方分块与grid教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[tl.load的mask行为](https://triton-lang.org/main/python-api/generated/triton.language.load.html)。沿用本机Triton3.5.1，网页用于解释，不据此升级环境。
