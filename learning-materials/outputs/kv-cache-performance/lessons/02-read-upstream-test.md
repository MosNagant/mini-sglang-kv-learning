# 阅读上游 test_store.py

日期：2026-09-29。用户希望从理解这份测试开始。状态：已讲解，待用户解释；没有修改实现或测量代码。

固定提交：9a91cfafe754aa85daee49998176275667eb58f2。

## 整体目的

准备缓存 → 对不同写入规模构造数据 → 调用原版 store_cache → 检查选中行 → 与 torch.compile 索引赋值实现比较性能。

输入是随机合成数据，无模型下载或真实模型推理。16 个 bs 为 2^0 至 2^15，即 1 至 32768。

## 1–10 行：入口

- from __future__ import annotations 是 Python 的类型注解相关设置，先略过细节。
- torch 提供张量及 GPU 操作；store_cache 是被测写入入口；compare_memory_kernel_perf 是测量辅助函数。
- def 定义函数，后续缩进内容属于函数体。
- @call_if_main(__name__)：直接运行该文件时执行函数；pytest 导入文件时保留函数供测试发现。固定提交的 utils/misc.py 已核对，当前无需实现装饰器。

## 11–17 行：缓存

- HEAD_SIZE=128 是这个测试每个 K/V 向量的宽度。
- NUM_TOKENS=1048576 是缓存槽位总数，不是每轮输入数。
- 创建并设置当前 CUDA stream；可先理解为有序提交 GPU 工作的队列，CPU 与 GPU 异步细节在计时时再讲。
- randn 构造标准正态分布随机数，device='cuda' 把张量放在 GPU，float16 每元素 2 字节。
- kv_cache 形状 [1048576,2,128]：槽位、K/V、向量元素。
- [:,0,:] 与 [:,1,:] 是基本切片视图，分别选 K/V，形状均为 [1048576,128]；它们共享底层 kv_cache 存储，不凭二维形状判断连续性。

## 19–24 行：输入

- range(0,16) 给出 0 至 15；** 是乘方；列表推导式生成 16 种规模。
- randperm(C) 将 0 至 C-1 排成随机排列；[:bs] 取前 bs 个；to(int32) 转成 32 位整数。索引有效且互不重复。
- 重复索引会让不同输入行争写同一槽位；当前测试不接受此条件。
- qkv 形状 [bs,512]，只是本测试随机输入缓冲区的变量名，未计算真实模型的 Q/K/V。
- Python 切片起点包含、终点不包含：k 取列 0–127，v 取列 128–255。两者均 [bs,128]，剩余 256 列未作为这次 store 的输入。

## 25–34 行：调用与检查

- store_cache 接收两个输出缓存、目标索引、两个输入张量，更新缓存。
- indices 决定每个输入行对应的目标行，K 与 V 使用相同索引；下标从 0 开始。
- k_cache[indices] 按 indices 的顺序取出目标行，与 k 对齐。
- == 逐元素比较；torch.all 检查是否全部相等；assert 在条件失败时报错，逗号后的 bs 是错误信息。
- 此路径搬运数据，不涉及累加，原版使用精确相等比较。
- 测试只检查被索引行；未命中行保持不变应在后续正式测试设计中补齐。该缺口仍需用户自行解释。

## 37–40 行：字节量与布局变化

- MEM=bs×HEAD_SIZE×2×kv_cache.element_size()：本次 token 数 × 每个向量的元素数 × K/V 两份 × 每元素字节数。
- MEM 是逻辑有效写入量，不是缓存分配总量或实测 DRAM 流量。
- 当 bs>1 时，这里的 k/v 切片跨行有空隙；contiguous() 创建连续布局所需的副本，已经连续时则直接返回自身。
- 正确性调用在 contiguous 之前，性能比较在它之后。正式实验必须记录并覆盖相应布局，不把两阶段视作同一个输入布局。

## 42–53 行：比较

- baseline 内的两次索引赋值表达与 store 相同的目标写入；torch.compile 对该 Python 函数中的张量操作进行编译优化，GPU 张量的写入在 GPU 上执行。
- our_impl 的 lambda 是一个延后调用的无参数小函数，方便测量器反复调用 store_cache。
- memory_footprint 传入有效写入字节数；description 中 BS 标签显示当前规模，:6d 控制整数显示宽度；extra_kwargs 的 init_stream=False 让辅助函数选择当前 stream。
- 固定提交的 perf.py 已核对：先调用预热，再默认捕获 10 次操作到 CUDA Graph，预热 replay，CUDA Event 记录 10 次 replay 的总耗时后除以 100，得到平均每次操作的 GPU 时间；输出单位 ms，保留三位小数。
- 这不是 Python 完整调用延迟、包含 JIT 的首次延迟或多次独立实验的中位数。图捕获与计时实现细节留待测量课。
- 日志 Baseline 是编译后的 PyTorch 索引赋值；Our Impl 是上游原版 CUDA，用户变体尚未实现。

## 本轮只需反馈

1. bs=4 时，qkv、k、v 的形状分别是多少？
2. indices=[6,2,0,5] 时，输入第 1 行 K/V 应写到哪里？
3. 若上面所有目标行正确，但未命中第 7 行被意外修改，现有 assert 是否能发现？根据代码解释。

若有卡住的具体行，先继续讲清该行；不以用户粘贴源码代替掌握验证。

## 已核对资料

- [固定版本测试](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/tests/kernel/test_store.py)
- [固定版本测量辅助函数](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/benchmark/perf.py)
- [固定版本 call_if_main](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/utils/misc.py)
- [PyTorch 2.9 randperm](https://docs.pytorch.org/docs/2.9/generated/torch.randperm.html)
- [PyTorch 2.9 Tensor Views](https://docs.pytorch.org/docs/2.9/tensor_view.html)
- [PyTorch 2.9 torch.compile](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html)
