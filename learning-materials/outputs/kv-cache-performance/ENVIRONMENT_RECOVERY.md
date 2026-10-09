# 环境问题与恢复手册

**2026-10-09最新恢复确认：** 用户执行既有-SkipSetup修复后返回STORAGE DEVICE OK，并实际运行echo得到WSL_OK；见[用户输出记录](logs/wsl-recovery-user-output-20261009.json)。本次WSL启动阻塞解除，localhost代理警告未阻止命令执行。此步骤未验证项目导入/GPU或长期重启稳定性，不重复修复。继续当前PowerShell运行22号--prepare并编辑M06-04外层入口。下方同日“恢复待确认”为修复前历史。

**2026-10-09最新状态：启动故障复发，恢复待确认。** 用户进入项目报CreateVm/HCS/ERROR_PATH_NOT_FOUND；本次只读检查storvsp STOPPED/1077、ROOT/STORVSP/0000注册缺失、storvsp.sys及既有修复脚本存在。见[本次诊断](logs/wsl-recurrence-20261009-20261009053622245.json)。当前故障条件与之前一致，设备移除来源未调查，不能把旧DevManView日志当作本次原因证据。下一步用户管理员PowerShell沿用Repair-WSL-Storage.ps1 -SkipSetup后验证WSL_OK；恢复后回M06-04三空位入口练习，不重做M05。工具未执行修复或GPU，工程进度不变。

**2026-10-07最新状态：本次启动已恢复。** 15:42用户运行既有-SkipSetup脚本，日志确认ROOT/STORVSP/0000重新注册、Present=True、ConfigManagerErrorCode=0、storvsp Running、STORAGE DEVICE OK；随后15:46 M06-01四例GPU运行通过。见[恢复日志](../logs/repair-storage-20261007-154213.log)与[M06证据](logs/kv-width-check-results/run-20261007-154657-9g3dxths/status.json)。此前只读故障检查保留在[环境证据](logs/wsl-recurrence-20261007.json)。本次设备移除来源未调查，不能称为长期故障已根治；现在继续M06-02，无需再次修复。

> 以下原有日期的恢复状态为历史；当前是否恢复以上方最新修复日志及用户GPU运行证据为准。

整理日期：2026-09-29；更新：2026-10-02，Asia/Shanghai。当前状态：12:28沿用既有-SkipSetup脚本恢复设备，Present=True、ConfigManagerErrorCode=0、storvsp=Running；用户已返回实际WSL_OK，本次启动阻塞解除。系统日志此前记录10月1日22:40:12再次被DevManView卸载；上级调用来源和长期稳定性仍未解决。[最新复发证据](logs/wsl-recurrence-20261002.txt)、[恢复用户输出](logs/wsl-recovery-user-output-20261002.txt)。学习任务以LEARNING_LOG.md顶部及当前快照为准。

## 下次从这里回来

代码一直保存在 Ubuntu 的 `/home/weltlaw/projects/mini-sglang-kv`。已验证的临时方式是 **Windows 记事本编辑同一份 Linux 文件，PowerShell 调用 Ubuntu Python 运行**。VS 的保存问题尚未修好。

在 Windows PowerShell（看到 `PS C:\...>` 的地方）打开已经存在的练习文件：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\first_kv.py'
```

代码写在记事本中，修改后 Ctrl+S。运行命令放回 PowerShell：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /home/weltlaw/projects/mini-sglang-kv/.venv/bin/python /home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py
```

上面的first_kv.py命令保留为编辑/运行方式的历史示例。该列表练习已结束；09-30随后已运行真实CUDA store并完成首条K检查的正反例。10-01恢复后回到完整真实KV实验的数据准备→原版调用→成组检查，详见[今日计划](obsidian/2026-10-01-CUDA学习日志.md)，不重做旧列表练习。

源码、终端和文件名框是不同地方：代码写进编辑器；命令写在 PowerShell 的 PS 提示符后；保存窗口“文件名”框只放路径或文件名。保存文件能跨重启保留，终端历史和未保存的编辑内容不能代替保存；目前未建立 GitHub 远端备份。

## 状态总表

| 现象 | 已确认的处理与证据 | 当前状态 |
| --- | --- | --- |
| WSL 下载返回 403 | 用经过微软签名和 SHA-256 校验的本地官方 WSL MSI 安装 | 已安装，无需重复 |
| CreateVm/HCS/ERROR_PATH_NOT_FOUND | 10-07 15:42设备Present、驱动Running；15:46用户四例GPU通过 | 本次启动恢复；最新移除来源与长期稳定性未解决 |
| Linux 用户名含大写或空格被拒绝 | 创建并使用 `weltlaw` | 已解决 |
| Triton/Inductor 编译找不到 Python.h | 安装 Ubuntu `python3.12-dev` 后，原版测试通过 | 已解决 |
| 指定 Nsight Systems APT 版本不存在 | 从实际 apt-cache policy 读取完整 Version，修正包版本假设 | 已安装 |
| Systems 有报告但没有 GPU kernel 数据 | 2025.1.3 配合时间戳转换设置，vector_add kernel 事件出现 | 有限制的绕过，非 KV 性能基线 |
| Compute 性能计数器不可用 | Windows 允许全部用户访问后仍遇到资源/权限错误 | 未解决 |
| VS 保存报 URI 主机名解析错误 | 两种 WSL 入口都失败；记事本保存同一文件并运行得到 99 | VS 未修复，临时编辑方式已验证 |
| PowerShell 一直显示 >> | Ctrl+C 退出未完成输入，在 PS 提示符后重新输入完整命令 | 本次输入问题已解决 |
| notepad.exe 命令被当作文件名 | 取消保存对话框，在底部 PowerShell 执行命令 | 输入位置已纠正 |
| localhost 代理未镜像警告 | 本地 WSL、Python 和已验证 CUDA 工作仍成功 | 警告仍在，不能据此判定命令失败 |

## 1. 重启后 WSL 再次报找不到路径

10-02恢复结果：[12:28修复日志](../logs/repair-storage-20261002-122833.log)确认设备Present=True、ConfigManagerErrorCode=0、storvsp=Running；用户随后WSL_OK验证成功，见[用户输出](logs/wsl-recovery-user-output-20261002.txt)。当前无需再执行修复。本步没有验证项目文件或GPU，也未验证长期重启稳定性。

10-02复发：用户WSL_OK命令失败；只读检查得到storvsp STOPPED、退出码1077、注册项缺失、storvsp.sys存在。系统日志第32098–32102行记录2026/10/01 22:40:12.045通过DevManView的`/uninstall *Disk* /use_wildcard`删除ROOT\STORVSP\0000并成功；这是昨晚的删除记录，不能称为已证实今天开机时删除。沿用下方管理员PowerShell修复，结果尚待用户返回，未更改启动项或执行GPU任务。见[10-02证据](logs/wsl-recurrence-20261002.txt)。

10-01复发及恢复：用户进入项目的wsl命令报CreateVm/HCS/ERROR_PATH_NOT_FOUND；只读检查得到storvsp STOPPED、注册项缺失、驱动及INF文件存在。setupapi.dev.log第31886–31890行记载12:52:11.685再次通过同一DevManView命令删除设备，详见[新证据](logs/wsl-recurrence-20261001.txt)。随后[13:07修复日志](../logs/repair-storage-20261001-130723.log)确认Present=True、ConfigManagerErrorCode=0、驱动Running；用户实际WSL_OK验证成功，本次启动已恢复，不再重复运行修复。项目/GPU在这一步未重测，长期稳定性未验证。

本机已确认的故障链：开机附近运行 `C:\DevManView.exe /uninstall *Disk* /use_wildcard`，删除了 `ROOT\STORVSP\0000`，使 WSL 创建虚拟机失败。该实例对应 Microsoft Hyper-V 虚拟磁盘服务器，驱动服务为 storvsp。

2026-09-29 21:11:34 的删除有 setupapi.dev.log 证据。21:27 修复后，设备 Present=True、ConfigManagerErrorCode=0、驱动 Running；随后原 Ubuntu 目录可访问，Python 可运行。谁启动了 DevManView 仍未知，不能归因或停用某个尚无证据关联的启动项。设备被卸载不等于 Linux 文件被删除。

2026-09-30 新证据：用户运行 WSL_OK 返回相同的 CreateVm/HCS 错误；Windows 只读检查确认 storvsp=STOPPED、退出码 1077、设备注册项缺失。setupapi.dev.log 第 31681–31686 行记录 17:41:44.455，由同一 DevManView 命令成功删除 ROOT\STORVSP\0000。只读证据保存在 [09-30 复发记录](logs/wsl-recurrence-20260930-175734.txt)。时间是否对应开机及上级调用来源均未验证。

09-30 恢复结果：用户沿用下方 -SkipSetup 修复，于 17:59:09 完成；[修复原始日志](../logs/repair-storage-20260930-175907.log) 显示 Present=True、ConfigManagerErrorCode=0、storvsp=Running。用户随后执行 WSL_OK 成功，[用户输出](logs/wsl-recovery-user-output-20260930.txt) 已留存。确认本次启动恢复；文件内容、GPU 和长期启动稳定性未在这一步重测。恢复后继续原练习，不需要再次运行修复脚本。

只在出现相同故障时，先在 PowerShell 做只读检查：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /bin/echo WSL_OK
sc.exe query storvsp
Test-Path -LiteralPath 'Registry::HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Enum\ROOT\STORVSP\0000'
```

如果 WSL 正常输出 WSL_OK，就不运行修复。仅当再次出现对应 CreateVm/HCS 错误，并确认这个设备缺失时，使用已经验证过的修复：打开**用户自己的管理员 PowerShell**。管理员权限是因为需要恢复 Windows 设备注册。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\Repair-WSL-Storage.ps1" -SkipSetup
```

`-SkipSetup` 保留：只修复设备，不重新进入 Ubuntu/CUDA 安装流程。预期 STORAGE DEVICE OK，再运行上面的 WSL_OK 检查。这个特定修复不是所有 ERROR_PATH_NOT_FOUND 的通用解法；设备存在而启动失败时保存新错误再诊断。

证据：[复发说明](../WSL故障复发记录.md)、[21:24 修复前记录](../logs/wsl-recurrence-20260929-212408.txt)、[21:27 修复日志](../logs/repair-storage-20260929-212727.log)。当晚没有通过再次重启来验证长期稳定性。

## 2. VS 保存失败与“代码已改、输出没变”

实际顺序：VS 中第一行改为 99，标签有星号；cat 读取的文件仍为 10；用户按保存后，VS 弹出“无效的 URI：未能分析主机名”。因此不能简单说用户忘记保存。

已试过 `\\wsl$\...` 和 `\\wsl.localhost\...` 两种入口，实际 VS 保存均失败。本机 System.Uri 解析 wsl.localhost 成功，只是字符串解析测试，不能证明 VS 保存成功。不要重复要求用户换这两个地址。

已验证的临时方法是本页开头的记事本命令。用户保存后运行同一个 Python 文件，实际得到 99，说明本次修改已经写入运行目标。VS 的内部失败点未查明，尚未建立适合长期写 Python/CUDA 的完整 IDE 流程；没有安装 VS Code，也没有更改用户长期编辑器偏好。

如果以后编辑内容和结果不一致，在 PowerShell 查看运行目标的已保存内容：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /bin/cat /home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py
```

cat 只读文件。先比较保存内容和编辑器内容，再检查是否是同一路径。不能在没有证据时解释成 Python 缓存或 CUDA 故障。

证据：[用户两次运行输出](logs/first-kv-user-output-20260929.txt)、[URI 测试与实际失败补记](logs/vs-wsl-uri-probe-20260929.txt)、[临时编辑步骤](lessons/00-notepad-save-check.md)。

## 3. 终端输入与保存窗口混淆

- `PS C:\...>`：Windows PowerShell，可以执行本文 powershell 代码框中的命令。
- `weltlaw@...:~$`：Ubuntu 的 shell，适合运行后面的 bash 命令。
- `>>`：PowerShell 正在等待输入补完；先 Ctrl+C 返回 PS，再输入完整命令，不要复制提示符或 Markdown 围栏。不能仅凭 >> 判断 WSL 挂起。
- “另存文件为”的文件名框：只放文件名或路径。在这里放整条 `notepad.exe '路径'`，保存窗口会把命令当文件名并报错；这不代表记事本已运行并失败。

本次分别以 WSL_OK 输出和记事本成功编辑验证上述输入位置已纠正；无需重装环境。

## 4. Python.h 缺失导致原版测试失败

首次失败发生在 torch.compile / Inductor / Triton 编译辅助 C 扩展时，GCC 找不到 `/usr/include/python3.12/Python.h`。这是开发头文件缺失，不是当时 CUDA store 结果比较失败。

当时在 Ubuntu 内安装了 `python3.12-dev`（含对应 libpython3.12-dev）。已验证的修复操作是：

```bash
sudo apt-get update
sudo apt-get install python3.12-dev
```

这些包目前已装好，不需要每日执行。随后原版测试 `1 passed in 17.78s`；17.78 秒是整次 pytest，不是内核延迟。

证据：[首次失败日志](logs/upstream-run-20260929-180451-GTVqUB.log)、[通过日志](logs/upstream-run-20260929-181115-pFBvSx.log)、[环境版本记录](ENVIRONMENT-20260929.md)。

## 5. Nsight Systems 安装版本与 GPU 时间线

错误安装字符串曾是 `nsight-systems-2025.1.3=2025.1.3.140-1`；实际 APT Version 为 `2025.1.3.140-251335620677v0`。05 脚本已改为读取实际候选版本并核对预定构建，不能用仓库 .deb 文件名推测 Version。工具已安装，不需要重新安装 CUDA。

本次使用的 Systems 是 `/opt/nvidia/nsight-systems/2025.1.3/bin/nsys`。默认 PATH 中的 nsys 可能仍指向旧版，因此后续采集明确指定工具路径并记录版本。

升级工具后仍没有 GPU kernel 事件。随后在 `/home/weltlaw/.config/NVIDIA Corporation/nsys-config.ini` 设置：

```ini
CuptiUseRawGpuTimestamps=false
```

已有 [06 脚本](06-nsys-wsl-timestamps.sh) 负责配置记录及可恢复处理。本次配置之前不存在，记录 backup=null 是正常情况。设置目前已应用，不要作为每日启动操作重跑。

已验证：vector_add 报告中出现 1 个正时长 GPU kernel 事件（5824 ns）。限制：这个转换方式时间戳精度较低，单个事件不是稳定微基准；vector_add 也不是 KV 内核。未采到 NVTX 和 Unified Memory/P2P 提示不能抵消已存在的普通 kernel 事件，也不能据此宣称那些跟踪功能可用。

证据：[成功摘要](logs/nsys-wsl-timestamps-run-20260929-191618-eHnaZP/summary.txt)、[配置记录](logs/nsys-wsl-timestamps-run-20260929-191618-eHnaZP/configuration-change.json)、[GPU 事件统计](logs/nsys-wsl-timestamps-run-20260929-191618-eHnaZP/nsys-kernels.csv)。

## 6. Nsight Compute 计数器：仍未解决

用户已在 Windows NVIDIA 控制面板允许所有用户访问 GPU 性能计数器并应用。之后最小采集仍报 driver resource unavailable 或 permission 错误，当前状态为 DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE。

不要重复要求用户开启已经开启的选项，也不要把帮助 URL 中出现 ERR_NVGPUCTRPERM 当成错误正文。现阶段不能报告测得的 DRAM 吞吐、硬件 occupancy 或缓存命中率。后续先使用可获得的计时和 Systems 时间线证据，计时核心仍由用户先设计；缺失指标需要如实写入报告。

证据：[NCU 原始错误](logs/profiler-compatibility-run-20260929-190613-i0a4w2/ncu-profile.log)。

## 7. 初始安装和其他已知提示

- WSL 下载 403：已用 `outputs/packages/wsl.2.7.14.0.x64.msi` 完成安装，安装脚本同时校验微软 Authenticode 签名和预期 SHA-256；原包与校验逻辑仍保留在 [Install-WSL.ps1](../Install-WSL.ps1)。不把重新下载或安装 Ubuntu 当作每天启动步骤。
- 用户名：最终已创建 `weltlaw`。之前混用大小写、空格的用户名不符合当时创建工具的规则；已有账户无需重建。
- localhost 代理：提示仍存在，但 WSL 启动、Python 执行、已验证的 CUDA 计算和当时软件包下载成功。若之后有明确需要代理的下载失败，再单独排查网络；本次不据此改 .wslconfig 或宣称代理已修复。
- CUDA：已安装 Toolkit 12.8.93，显卡接口来自 Windows 驱动；nvidia-smi 的 CUDA Version 12.9 与 nvcc 12.8 表示不同信息，不是本次已确认的故障。WSL 不另装 Linux NVIDIA 驱动。

详细历史保留在 [环境观察](ENVIRONMENT-20260929.md)、[安装说明](../安装说明.md) 和 [WSL 故障复发记录](../WSL故障复发记录.md)。

## 次日续学与证据边界

项目固定 SHA：`9a91cfafe754aa85daee49998176275667eb58f2`。原版测试通过；正式 KV benchmark、用户 CUDA 变体、优化收益和整机推理均未完成。不要因为花时间排障而虚增作品进度。

当前教学任务见LEARNING_LOG.md：恢复启动后接完整真实KV实验，PyTorch从零解释，用户先编写关键检查。VS故障和设备删除的上级调用来源仍未解决；不在用户休息期间自动运行安装、修复、GPU压测或代写核心实现。
