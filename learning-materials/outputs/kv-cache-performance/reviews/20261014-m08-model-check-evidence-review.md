# M08 41号真实模型检查：独立离线审计

审计对象：[run-20261014-144754-mmdz752v](../logs/m08-model-check-results/run-20261014-144754-mmdz752v/)。结论：**本次限定 Eager 工作负载证据通过独立 CPU 审计**。审计没有运行 GPU、模型或重新执行旧用例，没有改变原始结果。

- [evidence-audit.json](../logs/m08-model-check-results/run-20261014-144754-mmdz752v/evidence-audit.json)：102项上游清单文件与4项 helper/adapter/入口文件，共106项源码哈希吻合；三臂15项已登记 NPZ/JSONL 文件大小与 SHA256 全部吻合。`runner.py` 是39号 helper 的保留副本；实际执行的41号入口以 `41-m08-model-check.py` 和 request 的 runner_sha256 核对。
- 三臂 original_repeat/original/triton 各有1792行 JSONL，即896对调用前/成功记录。逐条核对调用号、28层、28次 prefill + 868次 decode、索引/位置、真实 token、FP16/shape/stride、独立只读分配关系与实际启动参数，总2688条成功记录。Triton 的 original_calls=0，896条记录均带 `CompiledKernel`、同一有效编译hash及 `[B]`/D1024/BLOCK_D1024/4warps；两原版臂各有896次原版调用。
- 六份已保存的 layer0 样本（每臂首个 prefill 与首个 decode）均从 `before + K/V + indices` 重新构造整层期望，逐字节核对 expected/after，并检查相邻样本的缓存衔接。每臂完整最终缓存的未命中行159～512（含dummy槽）保持 sentinel；layer0 已保存的前129行与最终缓存一致。
- original_repeat↔original、original↔triton 两对的全部保存 NPZ，共44个数组、每侧145,765,896字节／72,882,690元素，dtype、shape和全部字节精确一致。全部返回 token/text 和32步 raw token 序列也一致。每臂保存的首 prefill、首 decode、末 decode logits 均有限，argmax与对应实际输出 token 一致。

工作负载为 Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`，RTX4070Ti，FP16，单请求，128输入token、32生成步，28层，naive cache，512实际页+1dummy页。真实输入 K/V `(B,1024)` 行距4096，缓存视图 `(513,8,128)` 行距1024；本次 Graph 关闭。

边界必须随结论说明：896次/臂的全缓存与只读分配检查由实际运行日志和已核对源码提供证据，不能称全部中间数组已离线复算；离线独立重建仅覆盖六份已保存层样本，其他中间K/V未保存。完整最终缓存与三份采样logits已离线逐字节比较，但不是全部32步logits归档。41记录了实际编译对象类型/hash，没有保存PTX/cubin或profiler轨迹。本审计未重新读取WSL权重文件；权重SHA来自原运行记录。

**41不覆盖 CUDA Graph replay，不提供计时、加速、模型质量或学员独立掌握结论。** 后续Graph政策须在自身实际捕获/预热/重放链路验证输出，性能按预定独立进程政策另记；M07默认原版与不晋升决定不变。

审计脚本：[20261014-m08-model-check-independent-audit.py](20261014-m08-model-check-independent-audit.py)。教练实现／审计，宿主现有Python与NumPy执行。
