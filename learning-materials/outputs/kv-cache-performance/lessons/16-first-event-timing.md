# M05：第一次 CUDA Event 计时练习

2026-10-03。用户已恢复 WSL，返回固定提交、项目虚拟环境 Python 与 RTX 4070 Ti。尚无本轮计时结果。

本轮沿用户最新要求进入 M05，复用教练提供的数据与检查。之前的独立最小验证仍未验收，不把复用模板记为独立写出测试框架。交付分保持20/100。

模板：[16-event-timing-template.py](../16-event-timing-template.py)。用户补 `warmup` 与 `measure_once` 两个函数体，不新增嵌套同名函数，不改变数据区。

## 在 WSL 的项目根目录准备

先加载项目环境（不仅是 `.venv`）：它还设置源码搜索路径 `PYTHONPATH` 和 CUDA 工具路径。2026-10-03 用户仅使用虚拟环境 Python 直接运行时出现 `ModuleNotFoundError: No module named 'minisgl'`；应复用旧测试入口使用的激活脚本，不重新安装同名包。

```bash
source /home/weltlaw/projects/mini-sglang-kv/experiments/activate.sh
```

```bash
cp -n /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/16-event-timing-template.py experiments/bench_kv_first.py
/mnt/c/Windows/System32/notepad.exe '\\wsl.localhost\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\bench_kv_first.py'
```

`cp -n` 保留同名已有文件。模板有明确的 NotImplementedError，不补函数体就会停止，不会产生有效计时结论。命令在 WSL 执行；不要把命令粘进编辑器保存框。

## 本轮 API

| 写法 | 用途 |
| --- | --- |
| `call()` | 调用已封装好的原版算子一次 |
| `for _ in range(repeats):` | 重复指定次数；循环体缩进 |
| `torch.cuda.synchronize()` | 等待当前设备之前提交的 GPU 工作完成 |
| `start.record()` / `end.record()` | 在当前流提交开始/结束标记 |
| `end.synchronize()` | 等待结束标记完成 |
| `start.elapsed_time(end)` | 读取两标记之间的毫秒数 |
| `return ...` | 把数值返回给外部收集循环 |

来源：[PyTorch 2.9 Event 文档](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html)。

第一次提交先补 warmup，再补计时函数；可以只贴这两个函数供审查。保留函数定义，只替换占位异常。

## 测量边界与限制

- 原版第一次调用、检查、事件资源初始化与预热在正式样本区间外。
- 20次预热是练习配置，不代表所有设备及配置都因此稳定。
- 30个逐次同步的 Event 样本；复用同一输入、索引与缓存，不在样本内部重置缓存。
- 数据极小，事件区间可能包含主机提交不及时造成的 GPU 空隙，不能声称它等于纯内核指令执行时间。
- 不在这个结果上推导真实 DRAM 带宽或直接比较不同计时方法；M05后续还需检查方法开销、扩配置与重复轮次。
- 模板检查全部小缓存和只读输入；它不代替48+4套件或任意输入安全证明。
- 本轮只做计时入门，正式基线还未形成。辅助框架未在GPU运行；已做本地Python语法检查。

保存并完成审查后运行：

```bash
source /home/weltlaw/projects/mini-sglang-kv/experiments/activate.sh
python experiments/bench_kv_first.py
```

回传两个函数、运行输出，以及你认为计时包含哪些工作的解释。测得数字不自动证明计时核心正确，需审查事件顺序、调用次数、等待与返回值。
