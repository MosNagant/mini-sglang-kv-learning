# M08 真实 KV 后端适配：CPU 准备审查

2026-10-14，教练实现。新增 [40-m08-kv-backends.py](../40-m08-kv-backends.py)，没有修改上游、M06/M07 源码或旧证据，没有安装软件、启动模型或运行 GPU。

入口为 `make_store_cache(backend, *, original_store=None, observe=False)`；返回上游五张量签名的 `store_cache`。backend 必须显式选 `original` 或 `triton`，不提供自动选择或回退。支持同 CUDA 设备上的 FP16 K/V 与缓存，二维或尾维连续的三维视图，展开 `1 <= D <= 4096`，非重叠行；索引限定一维连续 int32，长度 B，要求 B≤C。三维转二维仅在原版调用路径使用已验证的 `view`，没有复制。

真实模型的 `(B,1024)`、输入行距4096和缓存 `(C,8,128)`、行距1024由此合同接收。Triton 分别传入 K/V 输入和 K/V 缓存的四个行步长，每行一个 program，4 warps，D 补至2的幂后 load/store 均加列 mask；行/索引先转 int64 再乘步长。原版按固定 `store.cu`/`warp.cuh` 的合同，额外拒绝 K/V 行距不同、行字节数非128倍数以及实际向量宽度所需对齐不满足的配置。

索引值唯一、范围有效以及输入/输出不重叠由调用方负责；本入口不在 host 读取 GPU 值，也不在 Graph 中检查索引值。Graph 必须先对实际特化预热；replay 不重新执行 Python 元数据检查，调用方须保持地址/元数据和索引合同。模块导入只用标准库，不初始化 CUDA；依赖在创建后端时加载。

诊断用 `observe=True`。`call_count` 和 `launch_count` 仅在非空底层调用成功返回后增加；拒绝、异常、空输入不增加。`last_launch` 含 backend、B/C/D、四个行步长、grid、BLOCK_D、num_warps、counter_scope，`compiled_kernel` 保留 Triton 实际 launch 的返回对象。计数含 Python capture 调用，**不代表 Graph replay 次数或 replay 正确性**。默认 `observe=False` 不构造逐次字典或更新计数。

[CPU 检查脚本](20261014-m08-backends-cpu-check.py)由宿主现有 Python 3.12.14 / NumPy 2.3.5 执行，[30项结果](20261014-m08-backends-cpu-check.json)全部通过。通过模拟 Triton 执行内核 Python 函数体，检查 D64/65/192/1024、四个独立步长、二维/三维混合输入、完整缓存及未命中/间隔元素、只读输入/索引；另检查空调用、诊断开关、16种非法元数据、宽度边界、原版显式调用与对齐/步长限制，以及 AST 中无值回传/连续化/同步操作。该模拟验证地址表达式和 host 控制，不验证真实 Triton 编译、GPU 访存、Graph、模型输出或性能。

本次源码 SHA256：`cc4305af9cf89985f494b90f990070c5c042a328722866aad4d67a5353f81c15`。后续41号须对真实模型输入做 GPU 复制与模型输出/Graph replay 核验，再决定是否进入计时；M07 默认原版和不晋升决定保持不变。不得据此增加学员独立实现或掌握分。
