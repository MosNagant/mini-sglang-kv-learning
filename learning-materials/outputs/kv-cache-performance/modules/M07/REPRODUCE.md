# M07 阅读、核验与复现

## 只复习资料

打开本模块README，按实现→历史结果→36→37→面试框架阅读即可，不需要GPU。GitHub上的源码和数据目录均可展开，原始NPZ可由NumPy读取；SVG/PDF图表已生成。

## 在已保存资料上离线复算

进入本模块目录，使用已有Python环境。下列命令不执行GPU；统计和数组审查需NumPy，历史图表需ReportLab。未安装这些依赖时可直接阅读已保存报告，不需要为了复习重新部署CUDA。

```text
python -B package_sources.py --verify
python -B generate-historical-results.py --self-test
python -B audit-memory-results.py
python -B audit-selection-validation.py
```

37的审查重生成入口见[选版报告](RESULTS-SELECTION.md)。重生成默认更新本模块对应派生报告；原始`logs/run-*`不会被改写。若要保留派生文件的前一版本，先使用各脚本`--help`确认输出参数或另复制材料目录。

`package_sources.py`不带`--verify`用于从原机导出ZIP重建选定源码，需要原机本地已核验导出包；跨设备阅读只使用已保存src和`--verify`，不用下载该ZIP。

## 在当前原机主动重跑新实验

本次已经完成36和37，不要求再次运行来复习。主动复现才使用下面的一行PowerShell命令；每次会创建新run，不覆盖旧证据。

```powershell
& 'C:\Program Files\WSL\wsl.exe' -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/36-compare-kv-memory.sh
```

```powershell
& 'C:\Program Files\WSL\wsl.exe' -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/37-validate-kv-selector.sh
```

两次应串行运行。shell激活现有`/home/weltlaw/projects/mini-sglang-kv/experiments/activate.sh`；无需重装。36/37会检查固定commit、tracked源码状态、所需旧证据和源码哈希，发现漂移即停止并保存首个错误，不删去版本保护继续测。

WSL若再次出现HCS路径错误，先按[恢复手册](../../ENVIRONMENT_RECOVERY.md)做只读诊断，不把重装当默认处理。恢复设备的脚本只在确认相同设备缺失时使用。

## 换机器执行需要什么

私有整理仓库的Git提交不是mini-sglang上游提交。应准备固定上游完整工作树、兼容GPU/驱动及Torch/Triton环境，再恢复学习源码和运行器需要的历史证据路径；不要在整理仓库中删掉`rev-parse`检查假装上游版本相同。

新36依赖WSL项目内`experiments/learning/m07-full-groups-timing/run-20261010-215026-gs6vimx3`及经审查计时脚本，37依赖学习材料`logs/`里的27/30/32/35完整来源。只复制36或37单文件不能运行。完整依赖已随本仓库source/和learning-materials/保留，但路径仍按原项目约定；异机须显式配置并记录改动。

旧37选择策略固定到已验证设备/软件/源码上下文；新机器若不符应拒绝旧策略，并重新建立来源和门槛，不能宣称沿用4070Ti结论。需要适配时创建新实验版本，保留当前冻结结果。

本次实际复现环境仍是同一台4070Ti；没有做异机执行验收。
