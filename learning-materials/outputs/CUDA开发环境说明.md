# CUDA 开发环境配置

2026-09-29 17:07，CUDA 开发环境验证通过。Ubuntu 使用普通用户 `weltlaw`；GPU 为 RTX 4070 Ti，Windows 驱动 576.52。

已确认 `cuda-toolkit-12-8` 包版本 12.8.2-1、nvcc 12.8.93、GCC 13.3.0、CMake 3.28.3，以及 Ninja、Git 等构建工具。此次安装日志显示 CUDA Toolkit 包已经是已安装的最新 12.8 系列版本；修复 Windows 设备没有重建 Ubuntu 或删除其中的软件。

项目位于 `/home/weltlaw/projects/cuda-starter`。程序已完成 CUDA 编译及 GPU 执行，1,048,589 个结果全部与 CPU 参考值一致，CTest 1/1 通过。证据见 `logs/cuda-setup-20260929-170426.log`；0.18 秒是整个 CTest 测试耗时，不是单独的 GPU kernel 性能测量。

WSL 是独立的 Linux 环境。CUDA C++ 的基础构建环境已配置，无需每个项目重复安装；项目额外依赖仍应按需求配置。此次未配置项目专用的 Python/Conda/PyTorch 环境，也未验证图形编辑器与 WSL 的连接。

## 执行入口

以下命令保留作为维护入口，现在无需重复执行。2026-09-29 曾发现 Hyper-V 虚拟磁盘服务设备再次被外部设备清理操作删除；17:04 已恢复。需要再次修复该设备时，在管理员 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\Start-CUDA-Setup.ps1" -RepairStorage
```

证据及修复范围见同目录《WSL故障复发记录.md》。下方普通模式适用于设备正常时。

在你自己的普通 Windows PowerShell 中执行，无需 Windows 管理员模式：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\Start-CUDA-Setup.ps1"
```

脚本会进入 `CUDA-Ubuntu-24.04` 的 `weltlaw` 账户；出现 sudo 提示时输入先前设置的 Linux 密码，输入不可见。无需把密码发给 Codex。

安装可能下载数 GB 文件，取决于已有软件包；脚本要求 Linux 和 Windows C 盘各有至少 20 GiB 可用空间。保留终端窗口直到结束。

## 执行内容与成功标准

1. 检查发行版、用户、GPU、驱动及磁盘空间。
2. 安装编译工具，通过 HTTPS 获取官方仓库 keyring；APT 校验仓库签名。
3. 安装固定 12.8 系列的 Toolkit；APT 计划若包含 Linux 显卡驱动则停止。
4. 备份并向 `~/.bashrc` 添加 CUDA 编译器路径。
5. 创建 `~/projects/cuda-starter`；已有同名目录时改用带时间戳的新目录。
6. 编译 GPU 向量加法程序，执行 kernel，并逐项检查 1,048,589 个计算结果。

只有最后程序通过，日志才会写入 `PASS`。安装日志和包计划保存在同目录 `logs`；最新结果为 `logs/cuda-latest-status.txt`，WSL 既有安装状态另行保留。

启动器会先运行 `bash -n` 检查脚本语法。本次安装及 GPU 验证由用户在自己的 Windows 账户下执行，Codex 已读取落盘日志确认成功。

## 网络与后续使用

本次 Ubuntu APT 软件源及 NVIDIA HTTPS 下载均成功。脚本没有修改 Windows 代理或 WSL 网络模式；localhost 代理提示未阻止本次安装。

完成后重新打开 Ubuntu，让编译器路径生效：

```powershell
wsl -d CUDA-Ubuntu-24.04 --cd ~
```

然后在 Ubuntu 中执行（若安装输出了带日期的目录，使用那个目录）：

```bash
cd ~/projects/cuda-starter
nvcc --version
./build/vector_add
```

图形编辑器与 WSL 的连接将在编译验证通过后配置。

官方依据：[CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)、[CUDA 12.8 Update 1 驱动版本表](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html)、[CUDA 12.8 WSL 安装步骤](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#windows-subsystem-for-linux)。
