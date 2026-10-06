# 原版 store.cu：线程分工与行起点

日期：2026-09-30。依据：[用户贴出的带行号源码](../logs/store-source-user-output-20260930.txt) 与 [08真实GPU观察](../logs/real-store-20260930-182741-GPaQSw.log)。当前为源码讲解，用户推导尚待反馈，无新内核或计时。

## 本次只连接三个事实

1. 第27–28行 __global__ 标记GPU入口；第60–119行run负责主机端校验和启动。当前不扩展模板、PDL或launch bounds。
2. 当前配置每block128线程，NVIDIA warp为32线程，每block4个warp。第37–38行warp_id是跨block的warp编号；第42–49行一个warp负责一行输入K及对应V。注释element在这里指一组token数据，不是一个FP16标量。
3. 第96–97行已将张量元素步长乘dtype_size，传入内核的stride单位为字节。08输入行步长128元素即256字节，缓存行步长256元素即512字节。第91行结合store.py可知本次kElementSize为单行K的256字节，V另外复制相同字节数。

## 两组输入对应哪些线程

用户随后问为何n/32：第38行的整数除法只计算block内warp编号，threadIdx.x自身不变，硬件分组也不是这条语句创建的。CUDA block内线程以32个一组组织，定义已核对 [NVIDIA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#warps-and-simt)。例如线程47/32得到1；后面的block偏移将局部warp编号转为本次启动中的全局编号。同组线程据同一warp_id选择一个输入行，实际每lane复制分工仍待查帮助函数。

length=2时，第110–112行给出ceil(2/4)=1个block，启动128线程。block0内部：

| threadIdx.x | warp_id | 输入行 | 目标行 |
| --- | --- | --- | --- |
| 0–31 | 0 | 0 | indices[0]=5 |
| 32–63 | 1 | 1 | indices[1]=1 |
| 64–95 | 2 | 无 | warp_id < length为假，跳过写入分支 |
| 96–127 | 3 | 无 | warp_id < length为假，跳过写入分支 |

后两个warp也包含在启动的block内；不能把跳过写入说成没有启动这些线程。此小例子不能证明任何优化会提速。

## 源和目标

第43行 pos=indices[warp_id]。第45行源K行起点按输入顺序warp_id计算，第44行目标K行起点按pos计算；V使用各自基址和相同行步长。此处字节步长由主机端转换而来，不能再重复乘2。

warp::copy在第46/49行调用。09-30后续已收到用户的warp.cuh，真实实现的搬运单位、循环次数与lane分工见[下一课](05-warp-copy.md)。本例每线程8字节现有源码依据，不能将此前的256/32猜测作为证明。指针offset帮助函数实现后续必要时核对。

## 历史理解检查（当前任务见05-warp-copy.md）

用户最新要求从当前代码干中学。当前先连接第118–119行启动和第37–38行：解释const/auto、threadIdx.x、blockIdx.x及局部warp_id，只让用户代入线程47算warp_id并给算式；下列其余追踪题之后沿代码接续，不一次发完。不要另起玩具程序或要求先复述类比。

保持length=2、blockIdx.x=0、indices=[5,1]、128线程/block：

1. threadIdx.x=47属于哪个warp？该warp负责哪一行输入、哪个缓存行？
2. 该warp拿到的K源行、K目标行起点，相对各自K基址偏移多少字节？给算式。这里问行起点，不是该lane在warp::copy内的独立访存地址。
3. threadIdx.x=96会进入写入分支吗？用第42行条件说明。

尚无用户回答；完成后接warp::copy实现或一个受控形状实验，再由用户写独立正确性检查。保留用户先提出CUDA映射和计时方案的约定，不提前代写优化内核。
