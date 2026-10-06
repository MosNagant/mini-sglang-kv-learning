# 从固定64到WIDTH参数

日期：2026-10-02。状态：用户已正确提交64宽地址稿；新版模板/入口已准备，等待用户填写参数表达式。没有新的GPU运行或计时。

## 已完成的地址推导

用户提交：

```python
cols = tl.arange(0, 64)
src_offsets = pid * 64 + cols
dst_offsets = pos * 128 + cols
```

对本次连续输入、交错缓存、width64成立。列编号0..63，输入行距64元素，缓存同类行距64+64=128元素。仍搬三行，program数仍3；输入行数与每行宽度是两个不同参数。这里只审查聊天稿，未声称已经保存或运行；粘贴中的缩进和Markdown格式不等于磁盘文件一定如此。

## 新参数是什么意思

目标：同一份新版源码用于width64和128。五个地址形参后新增`WIDTH: tl.constexpr`，它表示行宽，不是再追加一遍调用所用的张量。

`tl.constexpr`说明编译这份内核时宽度已知，可用来确定`tl.arange`的范围。本次64和128可分别产生编译特化；同一份源码不代表同一个机器代码版本。仅计划这两种宽度及当前布局，任意宽度、mask、任意stride另行处理。

教练已提供新版函数头：

```python
@triton.jit
def store_kv_width(k_cache_ptr, v_cache_ptr, indices_ptr, k_ptr, v_ptr,
                   WIDTH: tl.constexpr):
    ...
```

参数化第一行示范：`cols = tl.arange(0, WIDTH)`。用户将自己64宽稿中的两条行距表达式改成使用WIDTH，删掉模板中的pass，保留四个空格缩进。不再粘贴一层def或整份旧文件，模板里的读写语句继续使用。

外部测试脚本读取`batch, width = k.shape`，通过`store_kv_width[(batch,)](..., WIDTH=width, num_warps=4)`传入。WIDTH决定每行多少元素，batch决定启动多少份program，num_warps仍是每份program内部的warp配置。

## 准备和编辑

在Windows PowerShell运行以下两条。第一条只创建缺失的新版模板，不运行GPU；已存在文件会保留：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/14-check-kv-widths.sh --prepare
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_width.py'
```

新版位置：`/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_width.py`。旧版first文件不会被入口覆盖，历史运行快照保留。若用户已手动将旧文件改成64，不用旧11/13号固定128输入入口来验证它；新实验独立读取width文件。

将pass替换为三行地址代码，先贴回审查。以后正式运行使用相同14号命令但去掉`--prepare`。

## 运行范围与记录

两种实现分别测试width64、128，共四个GPU用例，batch3、capacity8、FP16、indices=[7,0,3]固定。两种宽度均用全新输入与缓存，运行前保存独立CPU期望值；运行后先保存全量结果，再用用户检查循环验证目标行，另核对未命中行和输入K/V/indices不变。CPU正例各一组，三行K/V末列损坏与交换映射共七个反例各测一遍，两种宽度共14个反例。

只会在全部检查及共享归档成功后输出`KV_WIDTH_CASES_PASS: 4/4 cases`，不包含计时。执行时原版仍固定commit 9a91cfafe754aa85daee49998176275667eb58f2并核对导入位置。原版warp复制要求行字节数为128的倍数，FP16宽64和128满足这项静态约束；不是预先宣布运行通过。

原始输出：`/home/weltlaw/projects/mini-sglang-kv/experiments/learning/kv-width-check/run-*`。共享归档位于Windows工作区`outputs/kv-cache-performance/logs/kv-width-check-results/run-*`，失败也保存可取得的源码、数据与错误。

教练本地验证见`verification/14-checks-local-result.json`：仅NumPy替身验证两种宽度的输入/布局/参数传递、14个检查反例、固定只写64列的错误实现被128测试拒绝、源码未完成/缺constexpr拒绝及已存在学习文件保留。未执行PyTorch/Triton GPU编译或本地Bash。

资料：[Triton官方入门的编译期参数](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[arange范围](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)。概念参照官方main，实际版本仍以本机Triton3.5.1为准。
