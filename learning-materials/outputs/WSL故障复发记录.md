# 2026-09-29 WSL 故障复发

## 21:11 后再次复发（21:27 设备已恢复，随后 Ubuntu 启动验证通过）

用户先在 Windows 文件管理器成功访问 mini-sglang-kv，后在 Visual Studio 的文件夹选择器中访问 wsl$ 和 wsl.localhost 均失败。实际 PowerShell 输出随后确认分发为 Stopped，启动时报 Wsl/Service/CreateInstance/CreateVm/HCS/ERROR_PATH_NOT_FOUND，用户说明期间重启过电脑。先前可访问和后来无法访问属于不同时间点，不能据此断定 VS 与 Explorer 的身份差异是原因。

2026-09-29 21:24 的只读检查确认：

- ROOT\STORVSP\0000 的注册项不存在，pnputil 显示没有对应设备。
- storvsp 内核驱动 STOPPED，WIN32_EXIT_CODE=1077。
- setupapi.dev.log 第 31476–31481 行记录：21:11:34.225，C:\DevManView.exe /uninstall *Disk* /use_wildcard 删除 ROOT\STORVSP\0000，返回 SUCCESS。
- C:\Windows\INF\wstorvsp.inf 与 C:\Windows\System32\drivers\storvsp.sys 均仍存在，微软签名 Valid。
- 原始证据：logs/wsl-recurrence-20260929-212408.txt。删除程序的上游调用方仍未确定，不修改未知自启动项。

已有修复脚本和 Restore-HyperVStorage.cs 已复核；本次使用 -SkipSetup，仅恢复存储设备，不进入已有 Ubuntu/CUDA 的安装流程。在用户自己的管理员 PowerShell 运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\Repair-WSL-Storage.ps1" -SkipSetup
```

2026-09-29 21:27 用户已执行上述命令，结果 STORAGE DEVICE OK，设备 Present=True、ConfigManagerErrorCode=0，storvsp=Running。助手已读取 logs/repair-storage-20260929-212727.log 并通过 sc query 复核驱动 RUNNING、退出码 0。未重新安装 Ubuntu/CUDA，脚本未要求重启。

下一步在用户 PowerShell 验证原分发启动与项目目录：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec ls -ld /home/weltlaw/projects/mini-sglang-kv
```

用户随后贴回该命令的实际输出：`drwxr-xr-x 13 weltlaw weltlaw 4096 Sep 29 21:03 /home/weltlaw/projects/mini-sglang-kv`。确认 Ubuntu 已启动、原项目目录可访问；localhost 代理警告未阻止该命令完成。VS 文件夹打开仍待用户重新验证，本轮未再次测试 GPU 或软件包。设备和启动恢复不代表已经消除自动卸载的触发来源。

## 之前两次开机附近的故障与 17:04 恢复

CUDA 入口在首次进入 Ubuntu 时失败，Linux 安装脚本尚未执行。旧启动器把 WSL 输出捕获到路径变量，失败时没有打印其中内容，因此只显示了笼统的“无法进入 weltlaw”提示。

## 已确认的证据

- Windows 系统启动事件记录：2026-09-29 11:54:46、16:45:03 启动。
- `C:\Windows\INF\setupapi.dev.log` 第 30989 行附近记录：11:55:16.036，`C:\DevManView.exe /uninstall *Disk* /use_wildcard` 删除 `ROOT\STORVSP\0000`，结果为 `SUCCESS`，距离第一次开机约 30 秒。
- 同日日志在 16:45:18 再次记录 DevManView 批量卸载磁盘/存储设备，距离第二次开机约 15 秒。
- `ROOT\STORVSP\0000` 是 2026-09-27 已恢复并成功启动 WSL 的同一个 Hyper-V 虚拟磁盘服务设备。
- 本次修复前，注册表中该设备项不存在；`pnputil /enum-devices /instanceid ROOT\STORVSP\0000` 显示找不到设备，`storvsp` 驱动处于 `STOPPED`，退出码 1077。

这确认了设备再次被删除，而且操作紧随两次开机发生，强烈指向某个开机触发的软件或脚本。尚未确认具体调用方，不能归因于任何一个普通自启动软件。

日志中的设备卸载不等于删除磁盘文件或卸载所有软件，但会破坏依赖这些设备的功能。当前隔离账户无权读取计划任务，需要由用户在管理员 PowerShell 中进行针对性只读检查。

## 恢复并继续 CUDA 安装

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2\outputs\Start-CUDA-Setup.ps1" -RepairStorage
```

该入口先只读保存开机任务、相关进程审计和脚本引用证据，再恢复已知缺失设备，验证 Ubuntu 启动并继续安装 CUDA。不会自动停用任务或修改自启动设置，现有发行版和文件保留。

设备恢复需要 Windows 管理员权限；进入 Ubuntu 后安装软件仍可能要求 `weltlaw` 的 Linux sudo 密码。启动阶段起即保存 `logs/cuda-launcher-*.log`，调查结果保存为 `logs/device-cleanup-investigation-*.json`。

如只想先收集启动来源证据，在管理员 PowerShell 中执行 `Find-DeviceCleanup.ps1`，它不会进行系统修复或软件安装。

2026-09-29 17:04 已恢复该设备，日志显示 `Present=True`、`ConfigManagerErrorCode=0`、`storvsp=Running`；17:07 CUDA 编译及 GPU 计算验证通过。

本次只读调查已完成，结果保存在 `logs/device-cleanup-investigation-20260929-170422.json`。两个时间窗口的进程审计没有匹配到 DevManView，直接引用该程序的脚本也没有找到，因此具体调用方仍未确认。登录任务中包含 `MyMionTask`，指向 `C:\mion.exe`；文件只做了静态元数据和字符串检查，尚无证据把它与设备卸载关联，不应据此停用或归因。

触发设备批量卸载的软件尚未确定；本次恢复成功并不等于已经消除复发原因。未修改任何自启动任务。

进程来源调查使用 Windows [4688 进程创建事件](https://learn.microsoft.com/en-us/previous-versions/windows/it-pro/windows-10/security/threat-protection/auditing/event-4688)中的父进程字段；如果此前未启用该审计，历史来源可能无法从此日志还原。脚本不会更改审计设置。
