# 如何阅读、核验和复现

## 离线核验

阅读README、DESIGN和RESULTS无需GPU。用Python标准库即可核验包内文件哈希：

```powershell
python verify_package.py
```

命令在解压后的M06目录中运行；校验通过证明文件与打包清单一致，不代表重新运行GPU正确性或性能测试。NPZ数组另需NumPy读取。

## 现有WSL环境复现

不需要为了进入M07再次跑完M06。下面只用于主动复现，均为一行PowerShell命令。

共同正确性：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/modules/M06/runners/23-check-kv-common.sh
```

编译观察：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/modules/M06/runners/24-inspect-triton-compile.sh
```

同场六配置计时：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/modules/M06/runners/25-compare-kv-v1.sh
```

这些入口仍从现有WSL项目读取experiments里的学员文件，不自动覆盖它们。运行前应确认与本包src四份已验收源码一致；若后来编辑过，新的实验只证明新版本。shell会在runners/logs中写入新记录；不要将新记录混作冻结证据。

## 换设备的前提

需另有完整mini-sglang固定提交、兼容NVIDIA/CUDA驱动、Torch2.9.1+cu128/Triton3.5.1等既有依赖，并确认项目Python能导入minisgl。包内original_source只存kernel子树，不能直接pip安装成完整仓库。

shell使用 `$HOME/projects/mini-sglang-kv` 及其 `experiments/activate.sh`。Python入口可查看 `--help` 并指定project，但不会替你创建环境。需要把src四份文件放入新环境的experiments；先备份现有文件，不能自动覆盖个人修改。

25号入口还读取 `experiments/learning/m05-event-first/run-vtqqu_54/submitted_timing.py`。本包 `src/learner-timing.py` 是该份经审核的计时快照；在新环境恢复此对应路径，或明确记录运行器改动。不能只复制25号单文件便声称可复现。

硬件不同应重新测量；本次4070Ti数据不是其他GPU的预期保证。当前操作只完成Windows工作区归档，没有宣称WSL/GitHub已同步。
