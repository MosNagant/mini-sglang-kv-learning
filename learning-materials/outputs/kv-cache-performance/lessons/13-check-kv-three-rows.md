# 从两行到三行：内核定义与启动调用

日期：2026-10-02。状态：13:06用户实际运行完成，两种实现的三行用例共2/2通过；七个CPU故意错误被拒绝。保存的全量结果及22个源码哈希已独立复核，见reviews/20261002-kv-three-rows-review.json。

## 这次改什么

保持 FP16、width=128、capacity=8、输入 stride=(128,1)、缓存 K/V 视图 stride=(256,1)。目标改为 `[7,0,3]`，输入 K/V 各有 3 行，Triton 启动 3 个 program：0→7、1→0、2→3。

128 是输入相邻行起点的元素间距，256 是当前交错缓存同类行起点的元素间距。增加输入行数不改变它们；更换宽度或布局时不能继续默认这两个常量。

## 修正用户贴出的定义

用户保留了正确的地址/读写逻辑，但把调用参数追加进 def，并将 def 与函数名分行。定义仍然只有五个指针参数。下面是同一内核的语法整理版，教练没有将它覆盖到用户文件：

```python
import triton
import triton.language as tl

@triton.jit
def store_kv_first(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    pos = tl.load(indices_ptr + pid)
    src_offsets = pid * 128 + cols
    dst_offsets = pos * 256 + cols
    k_values = tl.load(k_ptr + src_offsets)
    tl.store(k_cache_ptr + dst_offsets, k_values)
    v_values = tl.load(v_ptr + src_offsets)
    tl.store(v_cache_ptr + dst_offsets, v_values)
```

保存位置：`/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_first.py`。只复制代码框内容，保留函数体四个空格缩进。文件原开头“两行输入”说明是第一版实验范围，不能用它判断本次实际启动数量。

定义类似写工作说明：每个 program 按自己的 pid 搬一行。外部调用才决定这次派多少个 program、交给它们哪些实际张量。测试入口里的调用示意如下；无需再粘进内核文件：

```python
batch = k.shape[0]
store_kv_first[(batch,)](k_cache, v_cache, indices, k, v, num_warps=4)
```

`num_warps=4` 是每个 program 的 warp 配置，不是本次输入行数。参数名末尾的 `_ptr` 不构成特殊语法，Triton 在调用时取得传入张量的指针。

## 执行

在 Windows PowerShell 运行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/13-check-kv-three-rows.sh
```

教练提供新的测试输入、动态启动封装与记录；继续使用用户已有内核和 check_kv_targets.py，不覆盖它们。原版 CUDA 与用户 Triton 各运行一次三行用例，检查全部目标元素、未命中行、输入 K/V 和 indices 不变。CPU 正例与七个故意错误检验检查器。五指针签名和 K/V/indices 行数在启动前核对。

只有两种实现均检查通过且共享归档成功，才会输出 `KV_THREE_ROWS_PASS: 2/2 cases`。这是两种实现的同一个三行用例，不是两个 program，不是 benchmark 或任意 batch/布局验收。

原始证据写入 `/home/weltlaw/projects/mini-sglang-kv/experiments/learning/kv-three-rows/run-*`；共享副本位于本 Windows 工作区 `outputs/kv-cache-performance/logs/kv-three-rows-results/run-*`。保存运行器、入口、用户源码、原版源码、输入、结果、状态与失败信息。

教练侧验证记录：`verification/13-checks-local-result.json`。使用 NumPy 替身验证三行输入/三个 program 的组织、漏开第三个 program 会被拒绝且保留结果、行数不匹配及混合参数定义会被拒绝。它不代表 PyTorch/Triton 编译或 GPU 执行。Bash 未在教练本地执行。

## 本次结果怎么读

CASE_SETUP和TRITON_LAUNCH说明准备了什么输入、启动了多少program；CASE_PASS表示该实现本次检查通过。最后KV_THREE_ROWS_PASS: 2/2是原版CUDA和本人Triton两个用例都通过且归档成功，不是只检查了两个program。CHECKER_CONTROLS_PASS表示检查器接受正确样本、拒绝七个故意错误。COVERAGE_LIMIT限制为本次形状、布局和索引，没有速度结论。出现Traceback、AssertionError、FAILED或CHECK_STOPPED时，先保留首条错误和日志，不能仅凭回到PS提示符判断通过。

## 与昨天最后版本准确对照

昨天11-triton-store-run.py已经是`module.store_kv_first[(batch,)](...)`，其中batch在输入准备时固定为2。今天13号从`k.shape[0]`取到3；中间12号使用字面量`[(2,)]`。所以今天不是第一次引入batch启动语法，而是把输入准备和启动数量接到新的三行场景。

输入K/V由两行变三行、indices由[5,1]变[7,0,3]，每行128元素和缓存交错布局保持。现有内核给每份program同一套规则，pid选择自己的输入行和索引项。因此增加第三份program会自然读取pid2并搬运第三行，不需要额外if分支或在def中追加张量参数。该kernel搬的是GPU显存里的已有输入；CPU到GPU的准备属于外部步骤。

概念依据：[Triton官方入门](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Triton语言/编译器说明](https://triton-lang.org/main/index.html)、[num_warps配置](https://triton-lang.org/main/python-api/generated/triton.Config.html)、[CUDA编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。官方main用于概念核对，本机实际版本仍Triton3.5.1。
# 10月2日入口更新：选择与内核匹配的宽度

13号入口现在支持`--width 64`或`--width 128`，默认128。此选项统一改变测试数据、期望值、缓存分配、检查和记录，不修改用户内核。当前用户保存的是64/64/128，使用：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/13-check-kv-three-rows.sh --width 64
```

匹配时应看到`KERNEL_LAYOUT_MATCH: cols=64 input_stride=64 cache_stride=128`及`input_shape=(3, 64)`；成功结束应为`KV_THREE_ROWS_PASS: 2/2 cases; width=64; ...`。这些是预期标记，不是已经取得的GPU结果。任何异常都不计通过。

若选择的width与已保存的三条常量地址式不同，会在导入GPU库/启动内核前拒绝，并保存源码和错误。这不是通用内核分析器；后续参数化内核需要相应的调用接口。旧128实验说明保留如下，其源码原样版本位于对应run归档。
