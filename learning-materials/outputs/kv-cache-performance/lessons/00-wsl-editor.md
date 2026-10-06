# L00a：Windows 编辑器连接 WSL 项目

2026-09-29。用户希望使用 Windows 上的 VS 类编辑器，暂缓 nano 路线。用户已确认本地编辑运行和从需求起稿两方面都不熟，先建立图形编辑器中的写—存—跑流程。

## 当前已核实与待确认

**当前操作以本段为准，以下较早条目保留为历史。** 记事本编辑同一 WSL first_kv.py、保存后用 Ubuntu Python 运行已由用户实际输出验证（K 首项 99，V 不变）。当前使用这个临时方式进入 01a-first-indexed-write.md，不再要求重复修复或数字修改。VS 对两种入口的 URI 保存故障仍未解决，临时编辑器不等于改变长期偏好；后续需要建立适合更大 Python/CUDA 文件的持续编辑流程。

```powershell
Get-Item -LiteralPath '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv'
```

这条验证同一终端上下文的 Windows 文件入口，不写项目文件，也不能替代实际保存验证。

- 最新验证：修复后用户的 wsl/ls 输出成功列出 `/home/weltlaw/projects/mini-sglang-kv`（属主 weltlaw），Ubuntu 启动与原目录访问已恢复。现在重新执行本页 VS 打开文件夹步骤；不要重复之前的路径/账户诊断或修复。VS 文件树仍待确认，GPU 和网络未在本轮复测。
- 21:27 更新：存储设备修复成功并有日志与驱动状态复核；接下来先在现有 PowerShell 执行 `wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec ls -ld /home/weltlaw/projects/mini-sglang-kv`，验证 Ubuntu 可启动且原项目可访问。确认后再回 VS 打开原文件夹；未重装软件，自动删除设备的来源仍待调查。
- 21:24 更新：用户实际 wsl 输出确认虚拟机启动失败，且说明中途重启过。只读证据确认 ROOT\STORVSP\0000 再次缺失，21:11:34 DevManView 删除该设备。当前先按学习日志恢复 WSL 存储组件，暂停后面的编辑器路径诊断；不再把不同时点的 Explorer 成功和 VS 失败当成进程权限差异证据。
- 最新故障：Windows 文件资源管理器曾确认能打开项目，但 VS 选择文件夹对话框访问 wsl$ 与 wsl.localhost 都失败。根因尚未确定，不再重复别名尝试。下一步在 VS 内的 PowerShell 终端执行下方只读检查；不能继续断言只是用户没选文件夹，也不因为“网络错误”默认重装 Ubuntu、调整 Linux 权限或更改系统 UAC。
- Windows 的 vswhere 返回 **Visual Studio Community 2026**。
- 检查了 Windows 用户安装与 Program Files 的常见 VS Code 路径，未找到 Code.exe。不能据此排除非标准位置或便携版安装。
- 用户已确认使用紫色图标的完整 Visual Studio；沿用已安装的 Community 2026，不要求切换 VS Code。
- vswhere 查询 Linux C++ 工作负载 Microsoft.VisualStudio.Workload.NativeCrossPlat 没有匹配结果；本次基础文件编辑加 WSL 终端流程不以该原生构建集成为前提，不声称已配置 F5、Linux IntelliSense 或 CUDA 调试。
- 没有安装或启动编辑器、扩展，没有修改 WSL 项目；不得把下面的计划当作连接成功。

## 同一份代码与两边的分工

- 源文件持续保存在 Ubuntu `/home/weltlaw/projects/mini-sglang-kv`。
- Windows 显示编辑器窗口；使用 WSL 连接后，编辑该 Ubuntu 文件夹，并使用 Ubuntu 的 Python、CUDA 和终端运行。
- 保存后的源码正常重启后仍在；关闭窗口和停止程序不等于删除源码。当前尚未建立个人 GitHub 远端副本。
- 现有 Ubuntu、CUDA 和项目虚拟环境沿用。当前 Windows 学习日志路径见 LEARNING_LOG.md，不声称已经同步到 WSL。

## 当前工作方式及边界

按用户选择，使用完整 Visual Studio 的文件夹视图打开 Ubuntu 源文件；在 VS 的终端运行 wsl.exe，进入已有 Ubuntu 环境，再使用项目 Python/JIT 启动。当前只建立编辑、保存、终端运行流程，不等于完成 VS 原生项目构建与调试配置。

Visual Studio 有原生 WSL C++ 构建与调试支持；其 C++/CMake 项目配置和 VS Code 的 WSL 扩展不同。需根据用户选择检查相应组件，不把 Code 的命令面板操作指给完整 Visual Studio，也不为了 IDE 改写 mini-sglang 的构建系统。

## 当前唯一任务：在 Visual Studio 打开文件夹并进入 WSL 终端

当前先处理文件夹打开失败。取消文件选择与错误对话框，在 VS 的“视图 → 终端”选择 PowerShell，然后逐行运行：

```powershell
whoami
wsl --list --verbose
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec ls -ld /home/weltlaw/projects/mini-sglang-kv
Get-Item -LiteralPath '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv'
```

保留全部输出，包括错误。先确定 VS 的子进程能否识别同一分发、Linux 目录及 Windows 共享入口，再决定修复。WSL 执行 ls 可能启动现有分发，但不会修改项目文件。只有路径问题解决后才继续以下原定打开步骤。

1. 启动紫色图标 Visual Studio，在启动页选择打开本地文件夹，或进入主界面的 文件 → 打开 → 文件夹。
2. 输入下面的 Windows 访问路径，选择此文件夹：

```text
\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv
```

应在文件夹视图看到 python、tests、experiments。这里编辑的是 Ubuntu 中的同一份源码，保存后正常重启仍保留。若打不开路径，先贴错误，不重新克隆或新建替代仓库。

3. 打开 视图 → 终端。若它显示 Windows 的 PowerShell/命令提示符，在其中执行：

```powershell
wsl -d CUDA-Ubuntu-24.04 --cd /home/weltlaw/projects/mini-sglang-kv
```

4. 出现 Ubuntu 的 weltlaw 提示符后执行：

```bash
source experiments/activate.sh
pwd
command -v python
```

后两行预期输出：

```text
/home/weltlaw/projects/mini-sglang-kv
/home/weltlaw/projects/mini-sglang-kv/.venv/bin/python
```

VS 中若未找到终端菜单，已有外部 PowerShell/Ubuntu 终端也可执行相同步骤；先反馈具体界面，不因为终端入口阻塞源码编辑。完整 Visual Studio 不使用 VS Code 的 `code .` 或左下角 WSL 扩展指示器作为本次验收。

完成标准：用户在 VS 文件夹视图看到项目，并贴回以上目录/解释器输出。随后才创建 experiments/learning/first_kv.py，手写、Ctrl+S 保存并从 WSL 终端运行。自动补全、F5 调试或正式 C++ 构建配置按实际需要增加。

## 此前备选 VS Code 方案（当前不执行）

1. 若 Windows 未安装 VS Code，从官方 Windows 安装入口安装；安装时保留加入 PATH。已安装则沿用。
2. Windows VS Code 扩展页安装微软发布的 WSL，扩展 ID 为 `ms-vscode-remote.remote-wsl`。
3. 在 Ubuntu 终端运行：

```bash
source ~/projects/mini-sglang-kv/experiments/activate.sh
code .
```

4. 首次连接可能需要下载 VS Code Server。待左下角显示 `WSL: CUDA-Ubuntu-24.04`，侧栏打开 mini-sglang-kv。若 code 找不到，先重新打开 Ubuntu 终端；仍失败则贴回错误检查 PATH，不重装 Ubuntu。
5. 在该 WSL 窗口打开新终端，运行：

```bash
source /home/weltlaw/projects/mini-sglang-kv/experiments/activate.sh
pwd
command -v python
```

预期分别是 `/home/weltlaw/projects/mini-sglang-kv` 和 `/home/weltlaw/projects/mini-sglang-kv/.venv/bin/python`。新终端不假定继承启动 code 时的虚拟环境。

本步骤只验收编辑器连接与环境位置。接着才在侧栏创建 experiments/learning/first_kv.py，手写几行、Ctrl+S 保存，在该终端运行；不先要求用户独立写完整 CUDA 或测试框架。

## 依据

- [Visual Studio：无项目或解决方案开发](https://learn.microsoft.com/en-us/visualstudio/ide/develop-code-in-visual-studio-without-projects-or-solutions)：文件夹打开方式。
- [Microsoft：WSL 文件系统](https://learn.microsoft.com/en-us/windows/wsl/filesystems)：通过 `\\wsl$` 访问 Linux 文件。
- [Visual Studio 官方博客：集成终端](https://devblogs.microsoft.com/visualstudio/a-more-integrated-terminal-experience/)：内置终端工作方式。
- [VS Code：Developing in WSL](https://code.visualstudio.com/docs/remote/wsl)：Windows 安装、微软 WSL 扩展、code .、服务端组件及 WSL 状态标志。
- [Visual Studio：Connect to WSL](https://learn.microsoft.com/en-us/cpp/linux/connect-to-your-remote-linux-computer?view=msvc-170#connect-to-wsl)：完整 Visual Studio 的原生 WSL C++ 支持。

编辑器功能核对日期：2026-09-29。实际连接是否成功仍由用户输出确认。
