# 临时验证：记事本保存同一份 Ubuntu 练习文件

2026-09-29，已通过：用户提供原 WSL Python 命令的实际输出 K: [99, 11, 12, 13]、V: [20, 21, 22, 23]，并返回 PS 提示符。该输出验证记事本编辑同一 WSL 文件、保存并运行的方法已完成一次；不再重复本页检查。当前进入 01a-first-indexed-write.md，继续用此临时编辑方式。VS 的 URI 保存故障仍未修复。

1. 错误弹窗点“确定”，保存对话框点“取消”；保留 VS 的编辑标签。
2. 在 VS 的 PowerShell 中执行：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\first_kv.py'
```

这是直接打开已有 Ubuntu 文件的 Windows 入口，此入口此前 Get-Item 已成功。若提示找不到文件或要求新建，取消并贴错误，不另建一份位置不明的文件。

3. 在记事本将第一行改成 `k = [99, 11, 12, 13]`，Ctrl+S 保存。其他三行保留；若保存失败，停在错误并贴回，不执行后面的验收命令。
4. 保存成功后回到 PowerShell 执行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /home/weltlaw/projects/mini-sglang-kv/.venv/bin/python /home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py
```

预期 K: [99, 11, 12, 13] 与 V: [20, 21, 22, 23]。贴回实际输出；通过后再安排项目数据与索引练习，并处理长期编辑器选择。不要把本次临时方法写成 VS 已修复。

Windows 与 Linux 文件互访参考：[Microsoft WSL 文件系统说明](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。
