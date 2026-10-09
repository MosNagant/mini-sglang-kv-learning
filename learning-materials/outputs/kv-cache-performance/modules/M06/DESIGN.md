# v1设计：按索引搬运一行K和一行V

## 操作目的和调用链

输入K/V已经在GPU显存中。对每个输入行 `pid`，读取 `pos = indices[pid]`，把 `k[pid, :]` 和 `v[pid, :]` 分别复制到缓存的 `pos` 行。这里不计算注意力、不生成K/V，也不做模型推理。

```text
外部张量 → 元数据检查 → 索引数检查 → B=0直接返回
                               → 索引值检查 → grid=(B,)启动一次
                                                   → B个program各搬一行K/V
完成后：独立期望与完整存储比较
```

`B`是输入总行数，`pid`才是其中的行编号；`D/WIDTH`是每条K或V的有效元素数。一次launch可以包含多个program，Python检查循环不负责逐行启动内核。

## 核心地址和mask

```python
pid = tl.program_id(0)
pos = tl.load(indices_ptr + pid)
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < WIDTH
src_offsets = pid * SRC_ROW_STRIDE + cols
dst_offsets = pos * DST_ROW_STRIDE + cols
```

偏移按各指针的元素类型计算；FP16指针不再手动乘2。换算字节时，整个元素偏移再乘2。K与V有各自基址，因此同一个dst_offsets不表示写到同一地址。

例如D192、输入gap64、分离缓存guard64：源/目标行距都为256元素，BLOCK_SIZE=256，只有0…191列有效。两次load和两次store均带mask。BLOCK_SIZE是逻辑列数，不是线程数。

输入gap与缓存布局是独立条件：D64输入gap64的源行距为128；目标分离且无guard时为64，交错且无guard时为128。当前内核使用实际stride，不写死 `DST=2*SRC` 或 `DST=2*WIDTH`。

## 当前公共接口范围

| 项目 | 约定 |
| --- | --- |
| 设备/类型 | 同一CUDA设备，K/V及缓存FP16，indices为int32 |
| 形状 | K/V同为(B,D)，缓存同为(C,D)，C>0，0≤B≤C；indices形状恰为(B,) |
| 宽度 | D为64、128、192；不是任意宽度 |
| 列步长 | 均为1；indices连续一维 |
| 输入行步长 | K/V相同，D或D+64 |
| 缓存行步长 | K/V相同；D或2D；D192另支持256/512的guard布局 |
| 索引 | 唯一，且0≤pos<C；当前不支持重复目的地 |
| 重叠 | K/V缓存全部C行的有效列区域不得重叠，源/索引不得与这些区域冲突；只读K/V源允许彼此重叠 |
| 空输入 | 封装直接返回，无KV内核提交 |

合同由 [kv_contract_support.py](src/kv_contract_support.py) 实现。允许的合同范围与实际测试矩阵分别记录：共同矩阵用C32、B={0,1,2,3,4,5,17}、三个D及选定布局/gap/guard；不意味着穷举所有合法输入。纯复制的有限数值测试使用精确相等；未做NaN载荷等逐位语义专项。

## 正确性、编译与计时各证明什么

共同正确性为每配置先建立一份独立CPU期望，两实现各新分配缓存并分别与其比较，检查目标所有元素、未命中缓存、guard、输入完整底层存储/gap和indices。98配置×两实现共196结果；B0的28例是适配器返回。反例验证检查器会拒绝损坏。guard未变不等于运行时检测了所有非法读取。

编译观察对应B3/D192、独立缓存、源gap64、缓存guard64的实际调用。保存源码到PTX/cubin的产物与元信息；PTX是虚拟指令表示，cubin是设备二进制。该次特化可来自缓存，不能据此声称新编译耗时。元信息18寄存器/线程和shared=0是该特化的静态资源信息，不是实测occupancy。

性能使用已验证的受信任调用；公共接口检查GPU索引值可能同步，不放进单方计时。v1一次启动B个program，每program配置4个warp。原版固定128线程block内，每warp处理一行，最多4行；这两种工作分配不同，但不能仅凭分配方式证明快慢原因。

## 冻结边界

v1内核SHA256：`6e76ef9eb319f4e57e4a55923fcb5655fa5fb26f29c8c7af00c49f0c125bb19d`。

M07只在新候选文件/入口中调整配置；原版、v1、失败结果和本包保持不变。当前没有模型端到端、AMD硬件或NCU运行时计数器结果。
