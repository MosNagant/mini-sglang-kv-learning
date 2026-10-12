# M07：看4个warp和1个warp怎样搬同一行

日期：2026-10-10。**运行后更新：12:31实际观察已核验，勿重复运行下面命令。** 本次D128中4-warp每线程1列、1-warp每线程连续4列，PTX访问宽度不同而每线程寄存器同为14。实际数据、编译文件与来源均通过独立复核，见[运行后复盘](../reviews/20261010-m07-warps-compile-review.md)。本轮无计时，性能仍见[首轮实验复盘](../reviews/20261010-m07-warps-timing-review.md)。以下保留运行前教学设计，事前快照也已在run中留存；本课不改冻结v1，不把编译观察自动算成性能提升。

## 这一步解决什么问题

27号结果未支持1-warp稳定提速。先保留4-warp v1作为起点，弄清“只改num_warps后，实际执行分工发生了什么变化”，再选下一项修改。

本课由教练准备观察、检查和保存脚本。学习者先运行，再结合实际输出理解两种分工；不要求手写导出编译器中间文件的代码。本次不计时，不重复26号完整矩阵或27号性能实验。

## 三个数字各管什么

两边固定 **B=256、D=128、C=1024、FP16、交错缓存、输入gap=0、indices=0..255**，与27号较大配置相同。K/V输入行距128元素，K/V缓存视图行距256元素，BLOCK_SIZE=128。

| 参数 | 当前值 | 用班组比喻说明 |
| --- | --- | --- |
| grid中的B | 256 | 总共安排256个program，每个负责一行K及对应的一行V |
| WIDTH / BLOCK_SIZE | 128 / 128 | 每个program分别搬128个K元素、128个V元素；两者均指元素数 |
| num_warps | v1为4，候选为1 | 改的是每个program配置的warp数；不把256行变成64行或1024行 |

当前NVIDIA设备的一个warp有32个线程，因此两边每个program配置128或32个线程。**线程数不直接给出每个线程搬哪几列、每条指令搬多宽。** 编译器会安排逻辑元素到线程的映射；以保留下来的实际特化产物为准，不能在看到结果前断言4-warp就是每线程一个元素。

两边内核仍按同一份规则寻址：

```python
pid = tl.program_id(0)
pos = tl.load(indices_ptr + pid)
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < WIDTH
src_offsets = pid * SRC_ROW_STRIDE + cols
dst_offsets = pos * DST_ROW_STRIDE + cols
```

同样的源码配上不同num_warps，可以得到不同的编译结果。D128、BLOCK_SIZE128时逻辑列都有效，编译器可能简化mask；不得把D192旧实验中的谓词指令直接当作本次结果。

## 原版为什么也值得对照源码

已固定的原版store.py默认128线程/block，store.cu用`warp_id`为一行编号，因此一个block最多处理4行。B256时原版启动64个block。当前Triton两边则都启动256个program，一个program一行。

| 实现 | 每个block/program处理的行数 | B256对应数量 |
| --- | --- | --- |
| 原版CUDA默认配置 | 每warp一行，每block最多4行 | 64个block |
| Triton v1，4 warp | 每program一行 | 256个program |
| Triton候选，1 warp | 每program一行 | 256个program |

这是从实际项目源码得到的分工区别，**还不是原版较快的原因证明**。28号先观察Triton两种配置，不新增原版GPU采样，也不直接改成多行program。若下一轮选分行方式作为变量，需要新的预测、尾行mask与正确性检查。

源码依据：[冻结原版CUDA](../modules/M06/evidence/06-paired/original_source/python/minisgl/kernel/csrc/jit/store.cu)、[冻结原版调用入口](../modules/M06/evidence/06-paired/original_source/python/minisgl/kernel/store.py)、[Triton内核](../modules/M06/src/triton_store_mask.py)。

## 28号会保存什么

对v1与候选分别创建初始数据，执行一次实际调用并核对全部缓存、未命中位置、输入和indices。本次数据用固定随机种子且每行可区分，与27号形状/布局相同、数值不同；两臂使用相同数据内容与独立分配。确认实际启动参数与接受版本一致，再保存这次调用返回的编译对象：

- **TTGIR：** 观察编译器记录的线程/warp布局，保留原文供解释。
- **PTX：** GPU虚拟指令，观察实际全局内存读写指令；不是每个tl.load必然对应一条指令。
- **cubin：** 已保留的设备二进制；不是要求学习者手写机器码。
- **静态资源元信息：** 记录可取得的寄存器、共享内存等字段；缺失就标记不可用。它们不是运行时occupancy、DRAM利用率或L2命中率。

同时保存TTIR、LLIR、源码哈希、独立期望和实际结果、环境与启动参数。允许复用现有编译缓存，不删cache，不声称观察到了首次编译过程。

## 运行与结果阅读

脚本准备审查已通过，在自己的PowerShell运行一整行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/28-compare-kv-warps-compile.sh
```

先看两边实际调用与正确性，再看保留产物。把末尾结果贴回，教练会直接读取共享文件一起解释。首次只抓住一个问题：**num_warps改了，但每program搬的一行没变，编译器怎样重新分配这行数据？** 不要求先读懂整份汇编。

若两边产物有变化，先记录变化，再提出可能影响；不能只凭某一边寄存器少或指令行数少就宣布更快。若产物不足以确定原因，如实保留不确定性，仍可设计下一轮有控制变量的实验。

## 面试知识提醒

来源：教练围绕本项目设计的迁移题，呼应用户面经中的register、occupancy和访存问题；不是声称某岗位必考。

- 相关概念：program、warp、逻辑列块、编译特化、静态资源与运行时指标。
- 基础问法：为什么改num_warps之后，输入行数和每行有效元素数没有变化？
- 条件变化：如果每个program改成处理多行，grid与最后不足整组的行该如何处理？这是下一假设的入口，尚未实现。
- 易错说法：warp少一定快；128个逻辑元素就是128个线程；寄存器少就证明occupancy高；读过PTX就等于写过编译器。
- 目标深度：会解释参数分工与实际产物中的一处差异；完整编译器和硬件资源模型暂了解。个人掌握状态在解释后另记。
