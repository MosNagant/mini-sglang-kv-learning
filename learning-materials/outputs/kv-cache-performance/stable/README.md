# 原版数学与整段 Decode Graph 运行入口

`stable_infer.py` 提供可重复使用的推理对象，默认启用 mini-sglang 已有的整段 decode CUDA Graph。K/V 写入、Gate/Up、SiLU、Down Projection 和其余模型计算均使用原版实现；prefill 仍由原版 Eager 路径执行。

这是运行方式的工程整理，使用已有的 CUDA Graph 功能和原版算子。2026-10-15 已完成严格正确性检查、三轮配对计时及真实项目入口运行；结果见下文。

## 支持范围

- NVIDIA GeForce RTX 4070 Ti；现有 Torch 2.9.1+cu128、Triton 3.5.1 及项目配套环境。
- 本地 Qwen3-0.6B，固定模型 revision，FP16、TP=1。
- 同时一个请求，每个请求恰好 128 个输入 token ID，生成 32 步，greedy、ignore_eos。
- 默认 decode Graph batch size 为 `[1]`，保留已有的 512 槽 KV 配置和原版调度。
- 一个 `StableRuntime` 对象可多次调用，避免重复加载模型和捕获 Graph；不支持并发调用。
- `close()` 也必须在请求完成后由同一使用方调用，不支持请求运行中从其他线程关闭对象。
- 初始化前核对源码、模型字节和依赖版本；GPU 查询在 Engine 初始化后执行。
- 初始化失败会直接报错，不会自动换实现或重新启动另一个 Engine。若需要 Eager，必须在启动前明确选择。

## 在现有 WSL 环境运行

PowerShell 一行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --cd /home/weltlaw/projects/mini-sglang-kv --exec bash -lc 'source experiments/activate.sh && python /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/stable/stable_infer.py'
```

默认运行已留存的 128/32 固定输入，并核对 32 个生成 token。输出是推理结果，不是性能测试数据。

在 WSL 已激活的 Python 环境中，可以使用 `--execution eager` 选择对照方式，或用 `--input-json /path/to/ids.json` 输入恰好 128 个 token ID。自定义输入只检查范围和运行步数；固定实验的输出正确性结论不会自动推广到其他输入。

## Python 复用

将本目录加入 Python 模块路径后：

```python
from stable_infer import StableRuntime

with StableRuntime() as runtime:
    first = runtime.generate()       # 固定实验输入
    second = runtime.generate()      # 复用模型和已捕获的 Graph
    runtime.synchronize()
```

必须在新进程中、导入 mini-sglang 之前创建运行对象。`--verify-only` 仅核对文件和包版本，不创建 Engine，也不能据此声称 GPU 已验证。

## 如何核验

54 号脚本先在两个新进程中比较原版 Eager 与原版 Graph：保存全部 32 步 logits，按既定 `atol=0.002, rtol=0.002` 逐元素检查，并要求 32 个生成 token 完全一致。Graph 的实际 replay 次数也会记录。

只有这一检查通过，才运行三组独立进程配对，每种方式两次预热、五次完整请求，保留全部 30 个时间样本。比较基线是相同接口下的**原版 Eager**。这些结果不代表新算子超过原版 Graph，也不代表服务并发、其他 GPU 或 AMD 性能。

## 本次验收结果

严格检查保存了双方全部 32 步 logits，共比较 4,861,952 个值。统一保存为 FP32 后，全部值逐位相同，最大绝对误差为 0；32 个生成 token 完全一致。Graph 路径实际执行 31 次 decode replay，prefill 保持 Eager。

| 独立进程配对 | 原版 Eager，毫秒/请求 | 原版 Decode Graph，毫秒/请求 | 时延下降 |
| --- | ---: | ---: | ---: |
| 1 | 223.293499 | 133.471155 | 40.23% |
| 2 | 231.264929 | 129.428888 | 44.03% |
| 3 | 227.794499 | 134.785158 | 40.83% |

每格为该进程五次完整请求的中位数，包含调度、采样、输出处理和 GPU 完成等待。模型加载、源码/权重校验、编译、Graph 捕获和两次预热均在计时外。所有 30 个原始样本保留；本结论限于上述固定工作负载。

- [严格检查及独立审计](../logs/m08-stable-runtime-check-results/run-20261015-052859-ufwymynd/result.json)
- [计时原始结果及配对数据](../logs/m08-stable-runtime-timing-results/run-20261015-053033-nlwp0fei/result.json)
- [真实项目默认入口运行](../logs/m08-stable-runtime-live-smoke/run-20261015-053233/result.json)
- [发布信息与文件校验值](release.json)

`stable_infer.py` 与 53 号源文件保持逐字节一致。54 号验证脚本在开始时校验两者哈希，避免发布入口与被验证源码不同步。

当前推荐执行方式是 `decode_graph`。本次发布只包含上述已验收入口；其他实验候选没有纳入默认执行路径。

## 本地压缩包

`releases/stable-runtime-20261015.zip` 打包了运行入口、53/54 脚本、本文和发布清单，以及本次验收的 JSON 结果和独立审计。压缩包没有包含模型权重、完整 mini-sglang 源码、Python 环境或大型 logits NPZ；运行仍需上文列出的现有环境，原始 NPZ 保留在工程日志目录。

压缩包里的 JSON 供复习和核对。若要重新进行 54 号计时，应先运行新的 `--mode check`，再将新生成的完整检查目录交给 `--mode timing --accepted-check`，不能拿缺少 NPZ 的精简发布证据代替完整检查目录。
