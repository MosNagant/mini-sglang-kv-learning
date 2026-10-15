# M08：已有资料与复现入口

**10-15稳定交付：** 直接使用[stable入口说明](../../stable/README.md)中的单行命令；[实际结果](RESULTS-STABLE-RUNTIME-20261015.md)已保存正确性、30个计时样本和独立审计，无须为复习重跑。54号先`--mode check`，通过后才允许`--mode timing --accepted-check <新的WSL检查run目录>`；它比较原版Eager与原版整段decode Graph。55号已完成原版Graph与采样捕获候选的完整数值和30样本计时；数值通过、性能未达逐轮门槛，不作为stable默认。

**10-15最新：** [48–52号结果](RESULTS-GATE-UP-20261015.md)已保留84真实fixture、588数值检查、4320区间和模型拒绝证据。复习直接读取结果，不必重新采样。入口依次为`48-m08-capture-gate-up.sh --attribution-run <47run>`、`49-m08-gate-up-experiments.sh --fixture-run <48run>`、`51-m08-gate-up-model.sh --mode check ...`。50是候选模块，不独立运行。51选中候选严格模型数值未通过，不能运行其timing模式绕过验收；52记录的是计数器权限拒绝。

**10-15最新：** [47号主要线性层归因报告](RESULTS-LINEAR-ATTRIBUTION-20261015.md)末尾提供单行复现及原始Nsight/SQLite/逐调用记录入口。它是独立Eager诊断，不并入42/46正式计时样本。

**10-15新增：** 44–46调用层候选、CPU消融与真实模型三臂对照见[独立复现说明](REPRODUCE-HOST-PATH-20261015.md)；本页保留38–43号原始复现入口。新增实际结果见[报告](RESULTS-HOST-PATH-20261015.md)，无需为复习重复GPU测试。

以下为本机已验证环境的复现命令。正常复习先读保存结果，无需再次运行。所有脚本新建run目录，失败证据保留；不把未执行的新命令记成结果。

## 环境

- WSL：CUDA-Ubuntu-24.04 / weltlaw；项目 `/home/weltlaw/projects/mini-sglang-kv`，沿用 `.venv` 和 `experiments/activate.sh`。
- 上游固定提交 `9a91cfafe754aa85daee49998176275667eb58f2`。39/41/42/43的模型run各保留实际102份源码副本/哈希；38的环境记录另记，不依赖日后工作区未记录的修改。
- [依赖安装记录](../../logs/m08-setup-results/dependency-install-ir_ipwxe/result.json)固定已有包后只补11个缺失包；没有升级Torch/Triton。
- 真实模型位于 `/home/weltlaw/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca`。1.50GB权重仅保存在本机，资料包保存revision、下载和哈希清单，不包含权重。不要用dummy权重替代。
- 38号是环境/下载准备脚本，已完成后不必重复安装下载。离线推理进程保持HF/Transformers离线；只在第一次真实下载时单独联网。

## 阅读顺序

1. [事前测量方案](../../reviews/20261014-m08-measurement-plan.md)：为什么固定模型/负载、分开Graph/Eager和诊断/计时。
2. [40号实际适配](../../40-m08-kv-backends.py)：输入D、四个行距、mask及显式后端。
3. [41号精确检查](../../logs/m08-model-check-results/run-20261014-144754-mmdz752v/result.json)与[独立审查](../../reviews/20261014-m08-model-check-evidence-review.md)。
4. [42号样本](../../logs/m08-model-timing-results/run-o_ltqzpd/summary.json)：每进程5次、每方式3轮，两层统计分别看。
5. [43号trace](../../logs/m08-model-profile-results/run-uml9cxd9/result.json)：GPU kernel时长份额与整个generate墙钟区别。
6. [结论](RESULTS.md)：记录负结果、范围和下一项假设。

## 单行PowerShell命令

先确认WSL能启动；本轮已启动成功，不无故重复修复系统。

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /bin/echo WSL_OK
```

精确正确性复现（创建新run、会运行模型；不计时）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/41-m08-model-check.sh --model-path /home/weltlaw/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca --model-revision c1899de289a04d12100db370d81485cdf75e47ca
```

固定负载同场计时（12新进程，GPU串行；不可与其它GPU测试同时执行）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/42-m08-model-timing.sh --accepted-check /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-check-results/run-20261014-144754-mmdz752v
```

独立Nsight跟踪（不是正式时间样本）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --cd /home/weltlaw/projects/mini-sglang-kv --exec bash -lc 'source experiments/activate.sh && python -B -u /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/43-m08-model-profile.py --timing-run /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/m08-model-timing-results/run-o_ltqzpd'
```

其它设备需要恢复记录的依赖、项目、模型revision及路径，不能保证直接复用本机绝对路径。没有声称做过另一台设备的复现。

## 留下的故障记录

43号首个run `run-rv7o7fmk`：诊断worker命名 `profile.py`，遮蔽Python标准库同名模块，使Transformers间接导入失败。更名 `m08_profile_worker.py` 后新run成功。旧源码/错误仍在[失败记录](../../logs/m08-model-profile-results/run-rv7o7fmk/result.json)；未改模型、放宽比较或隐藏失败。
