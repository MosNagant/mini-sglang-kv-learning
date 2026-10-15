# 44–46调用层优化：复现入口

2026-10-15。使用既有WSL项目/venv和已下载模型，不安装或升级包。下面每个代码块是一条PowerShell命令；已有结果可以直接复习，不需要重复测量。源码或模型变化会被哈希门槛拒绝，不能删掉门槛强行对比。

45号CPU消融及独立GPU正确性：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/45-m08-host-path-diagnostic.sh --accepted-timing /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-timing-results/run-o_ltqzpd
```

46号真实模型严格正确性：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/46-m08-host-path-model.sh --mode check --accepted-check /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-check-results/run-20261014-144754-mmdz752v --accepted-timing /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-timing-results/run-o_ltqzpd --candidate-sha256 1efe2013e8d18381701c11f90daa54b3d227e9ab32803ca07dca24fa4ff85356
```

46号三臂计时（下面使用本次已通过检查；若重新检查，应替换`--accepted-new-check`为新通过目录）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/46-m08-host-path-model.sh --mode timing --accepted-check /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-check-results/run-20261014-144754-mmdz752v --accepted-timing /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-timing-results/run-o_ltqzpd --candidate-sha256 1efe2013e8d18381701c11f90daa54b3d227e9ab32803ca07dca24fa4ff85356 --accepted-new-check /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-host-path-check-results/run-20261015-033200-qh26bon6
```

运行器均保存源码、计划、原始结果、每进程日志、失败原因和共享副本。45为3新进程的CPU空启动消融，46检查2新进程、计时18新进程90样本。源码、计时范围、数值契约与所有比较见[完整结果](RESULTS-HOST-PATH-20261015.md)。

CPU离线审计脚本：[45审计](../../reviews/20261015-m08-host-diagnostic-audit.py)、[46审计](../../reviews/20261015-m08-host-model-audit.py)。审计只读归档，不重跑GPU，不将运行时报告扩大为未保存的快照证据。
