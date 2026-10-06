# L00：找到项目，亲手写、保存和运行一个文件

状态：2026-09-29，用户通过记事本将同一 WSL 文件首项改为 99、保存并执行原 Python 命令，实际输出 K 首项为 99、V 保持原样。修改—保存—运行闭环已完成，证据在 logs/first-kv-user-output-20260929.txt，当前进入 01a-first-indexed-write.md；不要重做本页练习。VS 对两种入口的保存故障未修复，下面 VS 新建步骤保留作历史，不应解释为已恢复的编辑方案。

## 固定位置

- Ubuntu 项目：`/home/weltlaw/projects/mini-sglang-kv`，简写 `~/projects/mini-sglang-kv`（在 weltlaw 账户下）。
- 本次手写练习：`/home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py`。沿用已有 experiments 文件夹；此前 experiments/learning/first_kv.py 方案尚未确认执行，本次不依赖创建额外文件夹。
- 原版 Python/CUDA 源码：项目下 `python/minisgl/`；正确性测试：`tests/kernel/`。正式微基准与变体按后续任务建立，不移动或重建原项目。
- 当前学习日志主页：Windows 的 `C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\kv-cache-performance\LEARNING_LOG.md`。这些 Windows 教练文档尚未自动同步进 WSL 项目或 GitHub。
- Ubuntu 中已保存的文件正常重启后仍在。终端关闭不等于删除源文件；重新打开时需要进入项目并激活环境。尚未保存的编辑内容与 GPU 内存中的运行状态不能当作源文件保存。

Windows 文件管理器地址栏可输入：

```text
\\wsl.localhost\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv
```

这是访问同一份 Ubuntu 项目文件的入口，不是另建一份副本。[Microsoft WSL 文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)介绍了在 Linux 文件系统中保存项目和通过文件管理器访问的方法。

## 本次唯一练习

目的：把两组数字写进源文件，运行显示，再亲手修改一处确认理解。仅为 Python 文件操作与最小起稿练习，不使用 GPU，也不测试 store。

在完整 Visual Studio 中选择“文件 → 新建 → 文件 → 文本文件”，建立新标签；不要把练习写进原版 store.py，也不要覆盖或关闭其他尚未保存的编辑内容。

在编辑器中手写四行，用英文标点：

```python
k = [10, 11, 12, 13]
v = [20, 21, 22, 23]
print("K:", k)
print("V:", v)
```

前两行给两组数字取名字，后两行显示它们。中文需求是“保存两组数字，显示出来”，这里只实现这个最小需求。

在新标签按 Ctrl+S，保存对话框“文件名”输入下面的完整路径：

```text
\\wsl.localhost\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\first_kv.py
```

确认文件名为 first_kv.py，不是 first_kv.py.txt；保存类型若有“所有文件”可选择它。若再次报错，保留编辑标签，提供包含完整保存路径和错误的截图；不要重复重装或修复 WSL。不要把“文件名”误当作终端输入框。

保存成功后，在 VS 的同一个 PowerShell（PS 提示符）执行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /home/weltlaw/projects/mini-sglang-kv/.venv/bin/python /home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py
```

此命令调用 Ubuntu 项目已有的 Python 来运行保存的文件；无需另外安装 Windows Python，也无需按 VS 的 F5。命令使用绝对路径，不依赖当前 PowerShell 位于哪个目录。

预期：

```text
K: [10, 11, 12, 13]
V: [20, 21, 22, 23]
```

上述实际输出已收到。现在只把 K 的第一个数字 10 改成 99，Ctrl+S 保存并重跑同一命令；预期 K 首项为 99，V 不变。请用户说明执行命令用的是 Windows 的 Python 还是 Ubuntu 的 Python，并指出判断依据。此次结果尚待反馈；出错时记录第一处错误，不同时修改多项。

完成标准：用户贴回运行输出，知道代码存在哪、如何重开文件、修改和运行，并能说明自己的改动影响哪项输出。收到结果后才决定是否接回布局实验。

## 重启后的入口

在 Windows PowerShell：

```powershell
wsl -d CUDA-Ubuntu-24.04 --cd /home/weltlaw/projects/mini-sglang-kv
```

进入 Ubuntu 后：

```bash
source experiments/activate.sh
```

在 VS 中通过“文件 → 打开 → 文件”重新打开上述保存路径即可继续编辑。也可直接使用前面的 PowerShell 运行命令，不必先交互进入 Ubuntu。Python 的 .py 文件可交给 Python 运行；后续 CUDA 的 .cu 文件还要编译，这一步随 CUDA 任务讲解。

Git 提交用于记录代码版本，推送到自己的 GitHub 仓库可保存一份远端副本；这两个动作尚未为本次练习执行，不把已保存文件等同于已备份到远端。到项目保存步骤时一起建立，不在第一次写文件前堆叠工具配置。
