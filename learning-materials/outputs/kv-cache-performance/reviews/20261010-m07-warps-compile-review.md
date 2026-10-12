# M07编译观察：每线程1列与每线程4列

日期：2026-10-10。接受的实际运行：`run-20261010-123123-pgndtykl`，入口28号。两次GPU调用正确、编译证据与源码核验通过。本轮不计时；性能结论仍以[27号同场实验](20261010-m07-warps-timing-review.md)为准，未确认1-warp稳定加速。

## 证据核验

独立审计记录：[机器审计JSON](20261010-m07-warps-compile-evidence-audit.json)。47项源码哈希、10份编译产物、4份运行时源码及30条摘录原文/行号匹配。独立重建全部期望，8个after数组共655,872元素一致，包含768个未写缓存行、输入K/V和indices。

固定B256、D128、C1024、FP16、交错缓存、gap0、guard0，索引0..255。两边实际grid=[256]，WIDTH=128、SRC_ROW_STRIDE=128、DST_ROW_STRIDE=256、BLOCK_SIZE=128；只有num_warps分别为4和1。它们来自各自实际调用返回的编译对象，均保留TTIR/TTGIR/LLIR/PTX/cubin。GPU为4070 Ti，Triton3.5.1，PyTorch2.9.1+cu128。

输入是28号独立fixture，两臂内容一致、分配独立；与27号形状布局相同但不冒称数值或地址相同。本轮没有重跑性能或证明首次编译；PTX的调试路径仍指向早先相同源码，符合保留/复用编译特化的可能性，不把路径当作新编译时间证据。

## 已经看清的分工

| 本次静态事实 | v1：4 warp | 候选：1 warp |
| --- | --- | --- |
| program数量 | 256 | 256 |
| 每program的线程数 | 128 | 32 |
| 每program处理的输入K/V行 | 1行K及对应1行V | 1行K及对应1行V |
| sizePerThread | [1] | [4] |
| threadsPerWarp | [32] | [32] |
| warpsPerCTA | [4] | [1] |
| 每线程负责的K列数/V列数 | 各1列 | 各4列 |
| 每线程寄存器 | 14 | 14 |
| shared_memory_bytes | 0 | 0 |
| n_spills_raw | 0 | 0 |

以单份K的一行128元素为例，V采用同样列分工：

```text
4 warp：线程0→列0，线程1→列1，…，线程127→列127
1 warp：线程0→列0..3，线程1→列4..7，…，线程31→列124..127

4 × 32 × 1 = 128 个K元素
1 × 32 × 4 = 128 个K元素
```

这是本次D128的实际映射，不推广到所有宽度/布局。TTGIR记录了分配方式，PTX同时验证了地址计算：v1使用`tid.x & 127`作为列；候选使用`(tid.x << 2) & 124`作为连续四列的起点。元素偏移再在编译代码中乘2得到FP16字节偏移，Triton源码的指针偏移单位依然是元素。

定义参照[Triton布局说明](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)；本课只借其解释layout字段，不将Gluon新API引入当前固定版本代码。实际依据是本次保存的[4-warp TTGIR](../logs/m07-warps-compile-results/run-20261010-123123-pgndtykl/frozen_v1/compiled/kernel.ttgir)与[1-warp TTGIR](../logs/m07-warps-compile-results/run-20261010-123123-pgndtykl/candidate/compiled/kernel.ttgir)。

## 读写指令怎样看

v1的K/V读写使用`ld.global.b16`、`st.global.b16`，本次每个活动线程一次读/写16位，即2字节、1个FP16。

候选使用`ld.global.v2.b32`、`st.global.v2.b32`，本次每个活动线程一次读/写2×32位，即8字节、4个连续FP16。数据类型仍是FP16；b32描述此处操作的位宽。向量长度与基本位宽相乘确定访问大小，参照[NVIDIA PTX ISA 8.7](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#addresses-as-operands)。

两边最前面的`ld.global.b32`是读取`indices[pid]`这个int32值，与后面的K/V数据读写分开看。其地址来自indices_ptr+pid×4字节。

PTX原文：[4-warp](../logs/m07-warps-compile-results/run-20261010-123123-pgndtykl/frozen_v1/compiled/kernel.ptx)、[1-warp](../logs/m07-warps-compile-results/run-20261010-123123-pgndtykl/candidate/compiled/kernel.ptx)。PTX访问宽度不等于DRAM事务大小，也不是最终机器指令数的直接证明。

两个版本的`@%p1`在本次特化中对应前面的`mov.pred %p1, -1`，之后未修改，条件恒真。WIDTH=BLOCK_SIZE=128，没有无效尾列；因此这些谓词不能当作D192动态尾部检查证据。源代码保留mask逻辑，具体特化可化简它。

## 和性能结果怎么联系

1-warp确实减少了每program线程数，同时增加每线程列数，生成了更宽的K/V PTX读写。每行K/V总有效数据量没变，每program仍然只搬一行。

27号B256/D128的Graph代表耗时为v1 1.144960微秒、候选1.195360微秒，候选三轮都较慢。**更宽的每线程访存指令没有在这次实验中转化为提速。** 这能校准“向量化一定更快”的说法，却不能单独定位为什么慢。

两边寄存器都是14/线程，不支持“候选每线程寄存器数增加”的具体猜测。但线程数不同，不能据此推出每block资源相同或occupancy相同；没有运行时硬件计数器。shared为0和n_spills原字段0，也不冒称测到了全部存储流量或缓存命中率。

## 下一项候选方向与教学状态

已确认原版默认每warp一行、每block最多4行；当前Triton每program一行。下一项值得设计的单因素实验是：固定4 warp，尝试每program处理4行，grid按行组数量设置，并保护最后不足4行的情况。这是教练提出的后续候选方向，尚未有本人预测、实现、正确性或计时结果，也不承诺提速。B1可能产生空闲行位，B256可减少program数量；这些是需要验证的取舍。

当前学习重点为“源码参数→实际列映射→PTX访问宽度→实测结果”，不要求逐行背汇编。本人先前关于4-warp和小规模代表耗时的观察已有记录，本次编译解释由教练整理，尚未算本人独立复述。M07工程总分保持57/100；不重新运行28号，不修改既有算子或自动同步GitHub。
