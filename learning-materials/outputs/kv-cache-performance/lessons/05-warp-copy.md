# 从warp::copy推导每个线程搬运的字节

日期：2026-09-30。依据：[用户贴回的warp.cuh原始输出](../logs/warp-header-user-output-20260930.txt)、已保存store.cu及08真实GPU观察。原始输出为粘贴产生的长行，保留用户所带的1–80行编号。SHA256：52EC652E6D835F24ABD70929C0B8806978C58EA7C85541E30D13659A2AF3246D。

当前是原版源码解释，不是新内核实现或一次新GPU测试。用户对下面推导的独立掌握尚未验收。

## 本次数据

单行K含128个FP16，kBytes=256；V是另外一次同样大小的copy。当前block0包含128线程，只有warp0和warp1进入length=2的写入分支。

## 源码推导

- 第40–41行的默认模板参数调用resolve_unit_size(kBytes)。第25–32行先检查256能否被16×32=512整除，不能；再检查能否被8×32=256整除，可以，返回8。
- 第15–16行因此选择uint2，Package是类型别名而不是新分配的内存。uint2含两个32位unsigned int，sizeof为8字节。本次将原有FP16位模式作为8字节单位复制，不做FP16到整数的数值转换。
- 第45行kBytesPerLoop=8×32=256；第46行kLoopCount=256/256=1。第55行只有i=0这一轮。
- 第52行lane_id=threadIdx.x%32，是线程在warp内的编号。第56行j=i×32+lane_id，本例j=lane_id。
- 第50–51行将指针按Package来索引；第57行dst_packed[j]=src_packed[j]复制一个8字节Package。源和目标都在各自行起点基础上增加j×8字节。
- 因此对本例进入分支的warp，32个线程各复制8字节，恰好覆盖单行K的256字节；V另外复制。行内每8字节对应4个FP16。以上是源码工作量，不能据此断言具体SASS指令条数、一次显存事务大小或实际DRAM流量。

CUDA内建向量类型参考：[NVIDIA CUDA 12.8 Programming Guide](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#built-in-vector-types)。表中的alignment是对齐要求，不能把对齐值当作所有类型的sizeof通用规则。

## 512与1024字节：用户要求的完整对照演算

范围为单行K的一次copy；V另外调用copy。两者均通过第26行的x%(16×32)==0，返回16；Package为uint4、sizeof为16。因此每warp每轮都是16×32=512字节，kLoopCount分别为512/512=1和1024/512=2。

第55–57行逻辑为每个参与线程执行for循环，j=i×32+lane_id，复制src_packed[j]到dst_packed[j]。对本例16字节Package，区间为从j×16到j×16+15的字节偏移，分别相对于src和dst行起点。

| warp内lane_id | 第一轮i=0，j=lane_id | 第二轮i=1，j=32+lane_id（仅1024字节例子） |
| --- | --- | --- |
| 0 | 字节0–15 | 字节512–527 |
| 1 | 字节16–31 | 字节528–543 |
| 31 | 字节496–511 | 字节1008–1023 |

第一轮32线程共同覆盖0–511；第二轮相同线程继续覆盖512–1023。512字节例子每线程共搬16字节，1024字节例子每线程共搬32字节。第二轮没有增加线程或创建新warp。这里讨论源码逻辑迭代，#pragma unroll可展开循环；不能据此断言GPU时钟周期数、硬件事务数或实际耗时必然翻倍。

用户主动要求这份展开演算，当前属于教练示范，尚无用户独立推导1024字节地址的证据。

## 后续练习参考（当前先完成上述对照）

保持blockIdx.x=0、length=2、num_threads=128、kBytes=256，沿原版源码代入threadIdx.x=35：

1. 它属于哪个warp？写整数除法算式。
2. 它的lane_id是多少？写取余算式。
3. 它复制K时，在所负责K行内的起始字节偏移是多少？写包下标到字节的换算。

先由用户给出答案，再判断是否能独立区分线程号、warp号、lane号和行内字节偏移。此处不要求启动新程序。收到回答后再接输入行与indices目标行的地址，避免一次堆叠模板、宏、汇编和基准计时。
