# M06-02：宽度不变，行距可以改变

2026-10-07更新：两次gap小例后，用户运行--layouts，run-20261007-164300-ejjr_j5x的16/16组合GPU通过。完整存储与24项源码哈希独立复核一致，M06-02交付4/4，总37/100；见[矩阵审查](../reviews/20261007-m06-02-matrix-review.md)。个人局部布局推导为辅导后正确，不等同独立掌握。接下来[mask概念](m06-03-mask-intro.md)，无mask或性能GPU结果。以下保留本课操作与教学顺序。

## 先只改变输入gap

继续使用现有 /home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_v1.py。先保留参数化三行，不把参数换回固定数字。

本次B=3、D=64、capacity=8、indices=[7,0,3]，缓存仍为交错布局；唯一变化是输入K、V每行64个有效数后都留64个元素空隙。

~~~text
K输入底层： [K0的64个数][64个gap][K1的64个数][64个gap][K2的64个数][64个gap]
V输入底层： [V0的64个数][64个gap][V1的64个数][64个gap][V2的64个数][64个gap]
缓存底层：  [K槽0][V槽0][K槽1][V槽1]……
~~~

K/V的shape仍为(3,64)。WIDTH控制有效列数，stride(0)控制相邻行开头的距离；gap不属于有效数据，也不是新增一行。测试把gap填为-9，缓存初始值仍为-1；这些数是为了发现意外改写，不代表真实模型必须使用这些填充值。

## 本人先预测，再运行

把WIDTH、SRC_ROW_STRIDE、DST_ROW_STRIDE写出，并用自己的话说明哪个量与上一课64宽不同。仍使用元素单位，不额外乘FP16的2字节。

先运行一个配置（两种实现，共2例），在Windows PowerShell执行：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/14-check-kv-widths.sh --width 64 --input-gap 64
~~~

观察CASE_SETUP中的input_shape、input_gap、input_stride与cache_stride；对照TRITON_LAUNCH的三个参数。本次已返回KV_LAYOUT_CASES_PASS: 2/2 cases，run-20261007-161319-4d7j4cip。WIDTH=64、SRC_ROW_STRIDE=128、DST_ROW_STRIDE=128；两个128的来源不同，输入为有效64+gap64，缓存为K64+V64。

检查包含目标K/V全部元素、完整缓存及未写槽、完整K/V输入底层存储（含gap）和indices；原始输入/输出与源码按唯一run保存，错误时先保留再停止。16例本地模拟只是教练检查运行器，不是用户GPU测试，也不是Triton内核通过。

## 之后再改变缓存布局

先读完首个gap用例，再保持输入不变、改为分离缓存；讨论K/V视图基址和行距的变化。仍限定输入K/V共享源行距、缓存K/V共享目标行距、列距1，暂不支持两侧K/V分别不同的行距或任意列步长。

分离缓存可由同一(2,8,D)分配得到K/V视图，不意味着必需两个独立显存分配。gap与缓存交错是相互独立的配置。

接下来只改缓存布局，输入继续保留gap64；先预测哪个传入参数变化，再在PowerShell运行：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/14-check-kv-widths.sh --width 64 --input-gap 64 --layout separate
~~~

此命令已实际两例通过，run-20261007-162505-wba7gh_x。输入仍为(3,64)有效视图、行距128；缓存分配(2,8,64)，K/V各自视图(8,64)、行距64。源码与上一课完全相同，旧结果不覆盖。

两种变化能解释后，用同一个入口做本课完整回归：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/14-check-kv-widths.sh --layouts
~~~

已实际通过16例 = 两种实现 × 两种宽度64/128 × 两种缓存布局 × 两种input_gap0/64；包含上一课配置。单个2/2不代表全部16例通过。默认不加参数仍为原有M06-01四例；--layouts不可与单配置选项混用。

## 随课面试知识提醒

- 来源：教练设计的项目迁移题；不是招聘方保证会问的原题。
- 概念与代码：shape是有效元素的组织，stride是索引到实际地址的间距；对应WIDTH和SRC_ROW_STRIDE。
- 基础问法：两个shape同为(3,64)的张量，内存布局一定相同吗？
- 变式追问：源gap变化时，为什么目标缓存行距可以不变？
- 易错点：把gap算入要搬的列数；认为缓存行距总是2*WIDTH；把.contiguous()转换成本悄悄藏在候选实现前。
- 目标深度：会手算与解释布局；会用同一公式验证变化。片上存储/缓存性能暂了解，此处没有共享内存优化或性能结论。
- 本人当前证据：已把函数参数与地址表达式联系起来；本课gap与分离缓存小例GPU通过；先前混淆源gap与目标交错，经提示后已正确给出分离缓存下标1的地址64*1+col，并澄清行起点间距与列偏移。记录为本局部辅导后正确；源gap/其它布局的独立迁移继续随实验巩固，GPU证据照常保留。

准备审查见[辅助检查记录](../reviews/20261007-m06-02-harness-review.md)。不重做M05，mask与性能仍在后续课。
