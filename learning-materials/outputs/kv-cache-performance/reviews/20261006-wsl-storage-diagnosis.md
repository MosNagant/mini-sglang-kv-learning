# 2026-10-06 WSL 虚拟磁盘设备再次被卸载

本轮目标：恢复 WSL，复核 M05 原始数据并收尾，再进入 M06/M07。当前尚未恢复，不把环境排障计为实验验收。

## 已确认事实

- Windows SetupAPI 日志第 32887 行：2026-10-06 18:01:52.149，`C:\DevManView.exe /uninstall *Disk* /use_wildcard` 删除 `ROOT\STORVSP\0000`，结果 SUCCESS。
- 本轮只读检查：storvsp STOPPED，退出码 1077；设备注册键缺失；pnputil 无对应设备。Windows 驱动和 INF 文件仍存在。
- WslService 与 vmcompute 状态 RUNNING。服务状态并不证明请求正常完成；用户的 WSL echo 和 terminate 都挂起。设备缺失是已确认故障，恢复后仍须实际探测是否有剩余挂起。
- 当前工具身份为 CodexSandboxOffline，无法代替用户的 Administrator 会话恢复系统设备；读取受保护的 HCS / Security 日志也被拒绝。工具身份拒绝访问与用户的 WSL 挂起是两个不同观测。

## 复发来源的证据边界

直接执行卸载的是 DevManView。谁启动它尚未确认。旧调查只覆盖 9 月 29 日；本轮发现任务计划程序历史当前未启用，Sysmon 通道不存在。不得据无关联证据停用 MyMionTask 或其他启动项。下一项定向证据是用户管理员上下文中 10 月 6 日 18:00–18:04 的 Security 4688 父进程记录；若当时未记录，不能从现有日志虚构上级调用来源。

## 已准备的恢复动作

[recover-wsl-and-check.py](../../recover-wsl-and-check.py) 由用户自己的管理员 PowerShell 执行，调用此前复核的 Repair-WSL-Storage.ps1 -SkipSetup。修复等待最多 120 秒，超时保留修复进程避免中断设备安装；WSL echo 最多 25 秒，超时只停止本脚本启动的客户端。成功后导出当前项目以复核 M05，不执行 GPU benchmark。

本轮没有更改服务、自启动项或审计策略；没有删除、重装 Ubuntu。设备恢复结果、WSL_OK、导出文件仍待回传。

## 恢复后的顺序

1. 读取 run-d9gjhywx 的 plan/summary、18 份 result.json 和冻结源码；先复核已有数据，避免重跑整批。
2. 复核 100 次/区间、30 个样本/方法、两种执行方式、8BD 字节量及波动范围；完成 KV 时间线尝试、代表配置复现与范围明确的基线说明。
3. M06：同一 Triton 源码参数化 width / 实际 stride，统一正确性和公平计时。
4. M07：根据原版和 v1 数据提出可证伪假设，实施并验证一次优化迭代。

原始 Windows 诊断目录：`C:\Users\Administrator\Documents\Codex\2026-09-27\cuda-wsl-wsl-2/outputs/logs/wsl-diagnosis-20261006-114155`。调用方调查：[报告](../../logs/device-cleanup-caller-readonly-20261006.json)。
