# Gate/Up 数值与测试边界审查（2026-10-15）

本次为独立 CPU 审查。审查者没有运行 GPU，也没有修改已冻结的 50 号候选。实际运行记录由主实验保存，下面只说明源码与测试规则能支持哪些判断。

## 已安装 FlashInfer 的实际数学路径

来源为本机虚拟环境中的 `flashinfer-python 0.5.3`，归档在 [20261015-installed-flashinfer/manifest.json](20261015-installed-flashinfer/manifest.json)。归档的四个文件都重新计算 SHA256，全部与清单一致。

1. [activation.py](20261015-installed-flashinfer/sources/00-activation.py) 第 69–110 行把输入前半作为 gate、后半作为 up；没有传入 `out` 时会分配与输入同 dtype 的输出，并调用 `get_act_and_mul_module("silu")`。
2. [jit/activation.py](20261015-installed-flashinfer/sources/03-activation.py) 第 77–81 行生成的 SiLU 函数为 `val / (1.0f + __expf(-val))`，返回 FP32。这里调用的是 CUDA `__expf`。
3. [activation.cuh](20261015-installed-flashinfer/sources/01-activation.cuh) 第 43–51 行使用 `vec_t<float,...>` 接收输入，计算 `Activation(gate) * up`，再写回输出类型。尾部路径第 58–61 行同样先转成 float 做激活与乘法。
4. [math.cuh](20261015-installed-flashinfer/sources/02-math.cuh) 确实定义了 `ptx_exp2` 等辅助函数，但上述 SiLU 生成代码没有调用 `math::ptx_exp2`。不能仅因为头文件包含这些函数，就宣称此次激活实际走了该辅助函数。

因此，50 号 fused 候选保留“线性结果先舍入为 FP16，再将 gate/up 转为 FP32 计算 SiLU 与乘法，最后写 FP16”的边界，与被比较的两算子路径一致。候选的 `tl.exp`、归约顺序与原版 `__expf`/cuBLAS 仍可能产生差异，不能据此宣称逐位相等，也不能凭已安装源码推断最终 SASS、全部编译选项或硬件计数器。

## 溢出与数值判定

- 对有限 FP16 输入和权重，K=1024 时 FP32 点积的最坏绝对上界约为 `1024 × 65504² ≈ 4.4e12`，远小于 FP32 最大有限数。
- 点积随后转换成 FP16 仍可能溢出。实际输出、参考输出或 oracle 出现 NaN/Inf 时，数值判定必须失败；本轮通过的真实样本不能推广到所有可能的 FP16 数值。
- 有限的大负 gate 可能使 FP32 `exp(-gate)` 溢出为正无穷，表达式趋于有符号零。这个现象不等同于最终结果必然出错；实际误差仍由预先冻结的规则判定。
- 49 号 oracle 对原始 GEMV 使用 FP64 点积；对激活输出，先将该点积舍入到 FP16，再用稳定的 FP64 SiLU 与乘法计算，保留最终 FP16 输出舍入前的值。它用于误差衡量，不是被计时实现。
- 两道门同时成立才通过：全部元素满足 `abs(candidate-reference) <= 0.002 + 0.002*abs(reference)`；candidate NRMSE 不超过 `1.10 × original NRMSE + 1e-6`。NRMSE 分母是 `max(norm(oracle), 1e-12)`，零 oracle 不会触发除零。

## 已完成的 CPU 控制测试

[CPU 控制脚本](20261015-m08-gate-up-cpu-controls.py) 与 [保存结果](20261015-m08-gate-up-cpu-controls.json)覆盖：

- 七个候选的 Python 接口、当前输入转发、输出复用、1/2 次 kernel 提交数量、split-K workspace 大小、真实 gate/up packing 的逐元素还原，以及原始权重未变。
- 49 个非法当前输入与 9 个 prepare 非法条件被拒绝；准备阶段拒绝 Graph capture；实际调用不新增分配或同步。这里使用 NumPy 假对象，不证明 GPU 编译、地址安全或算术正确。
- 对 49 号真实 `numerical_check` 函数的 25 项控制：零 oracle、NaN/正负无穷分别出现在三个数组、形状/维数不一致、元素误差与 NRMSE 各自单独失败、容差边界和空数组拒绝。
- FlashInfer 源码归档器没有执行被检测包的 `__init__`；归档文件 hash 正确，已有证据目录不会被覆盖。

## 49 号计时复核

计时分母为 `replays × calls_per_replay`：hot 配置每次 replay 一个完整 Gate/Up+SiLU*Up，rotate28 每次 replay 28 个完整流水段。raw GEMV 候选仍接原版激活；fused 候选输出激活结果。输出复用与原版分配 API 的差异已记录，因此 Eager 结果是完整调用实现成本，不能称为纯 GPU 内核差异。

运行前提出的“计时结束后还要验证当前结果”问题已修复：冻结的 49 号脚本会对每个候选/原版重新执行 Eager 与 rotate28 Graph，Graph 输出先填 NaN，再 replay；逐层检查 FP16 dtype、全部元素容差与 oracle NRMSE。计时前还包含改变捕获输入内容后重放的控制，防止正确的旧输出掩盖没有执行或读取旧值的问题。

上述审查不增加学习者独立掌握评分，不把 CPU 假对象测试当成 GPU 结果，不构成模型加速或 AMD 性能结论。
