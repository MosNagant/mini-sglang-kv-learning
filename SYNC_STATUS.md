# 同步范围与来源

## 2026-10-14：M07工程收尾与本次来源边界

当前工程结论：M07研究材料和选版决定收尾，默认保留原版CUDA，Triton v1及三条候选均保留研究用途。按旧任务表仅新增M07-06报告3分，工程总分64/100、M07为11/25；14分个人复盘待验收。工程权重不是知识掌握百分比。M08真实模型接入和端到端验证独立未完成。

本轮实际新增GPU测试为36号18个新进程/3240区间均值、37号9个新进程/1080区间均值，合计27进程/4320区间均值。每区间100次调用；Graph主结果、Eager单列。36在固定B256/D128/C1024下比较六种内存条件，保留完整数组、实际地址与编译产物；37验证冻结默认选择在新B7/64/257上的正确性，两臂实际调用同一原版，A/A差值不构成候选或选择器加速。没有候选晋升默认。

入口：[M07模块](learning-materials/outputs/kv-cache-performance/modules/M07/README.md)、[历史完整成绩](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-HISTORICAL.md)、[36号访存报告](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-MEMORY.md)、[37号选版报告](learning-materials/outputs/kv-cache-performance/modules/M07/RESULTS-SELECTION.md)、[10-14节点日志](learning-materials/outputs/kv-cache-performance/obsidian/2026-10-14-开发日志-M07收尾.md)。27/30/32/35的48worker/8640区间均值来自历史实测，本次仅重新汇编。全部旧run、负结果与失败记录保留。

### 此次真实WSL快照

源码导出时刻为**2026-10-14 10:53:20 UTC / 18:53:20 Asia/Shanghai**，路径`/home/weltlaw/projects/mini-sglang-kv`；上游固定提交`9a91cfafe754aa85daee49998176275667eb58f2`，导出时tracked_status为空。归档`wsl-project-20261014T105320Z.zip`包含6457个普通文件、85,646,985字节，记录29项排除与1个symlink。ZIP的SHA-256为`a973db75652b953e5fba7f274d393a8802058efbaead323b11f6045863c87093`。

导出文件逐项核验SHA-256、字节数、ZIP CRC与Unix模式后更新到`source/`。普通文件的Git模式依据原可执行位映射为100644/100755，`.dockerignore → .gitignore`保留120000模式；原始644/664/666/755权限仍列在[本次导出清单](source/EXPORT_MANIFEST_20261014T105320Z.json)。上游[LICENSE](source/LICENSE)和已有文件声明保留。

**该快照早于36/37的实际运行。** 两批新增运行器、策略、所有测试数据、源码快照和独立审查在`learning-materials/outputs/kv-cache-performance/`完整归档。快照后的学习源码若单独补入`source/experiments/`，在[本次来源记录](source/SOURCE_SNAPSHOT_20261014.json)的post_export_overlays中逐项列出来源与哈希，不把它们冒称为10:53:20导出内容。

新导出存在的路径覆盖其当前`source/`内容；新导出未包含的既有路径和10-06/10-12来源元数据继续保留为历史。本次SOURCE_SNAPSHOT列出保留路径，原`source/EXPORT_MANIFEST.json`、`source/SOURCE_SNAPSHOT.json`和`source/SOURCE_OVERLAY_20261012.json`仍按各自日期理解。仓库source目录是可追溯的现行快照与历史保留文件，不能仅凭目录存在就把每个旧文件判为10-14实况。

### 本次学习材料与恢复证据

完整同步当前`outputs/kv-cache-performance/`到`learning-materials/outputs/kv-cache-performance/`，包括modules/M07实现说明、面试复习、复现指南、三份性能/决策报告、原始run与可重生成审查脚本。排除`__pycache__`、`.pyc/.pyo`、`.git`及同类可再生缓存；NPZ、PTX、cubin、既有Nsight资产和ZIP按原字节保留。导出的ZIP保存在本地同步工作区，仓库源码以逐文件内容为入口。

另归档存储恢复的[修复日志](learning-materials/outputs/logs/repair-storage-20261014-185022.log)、[恢复后状态](learning-materials/outputs/logs/storage-after-20261014-185022.json)和[服务注册前态](learning-materials/outputs/logs/storvsp-service-before-20261014-185022.reg)。同时间`storage-before-20261014-185022.json`未发现，不补造；已发布的[Repair-WSL-Storage.ps1](learning-materials/outputs/Repair-WSL-Storage.ps1)和[Restore-HyperVStorage.cs](learning-materials/outputs/Restore-HyperVStorage.cs)沿用。此次恢复后实际GPU执行成功，不等于反复启动故障根因已解决。

本轮新增代码、执行和报告由教练在用户授权下完成；此前个人贡献与提示程度按原记录保留。同步不会额外增加工程得分或个人掌握记录，也不会替代M08。普通可解码UTF-8文件使用有限已知凭据/私钥模式与敏感文件名检查；该检查不是对二进制和所有可能凭据的穷尽保证。私有仓库范围不变，虚拟环境、模型权重与整机镜像不在范围内。

本节说明待发布材料的范围；上传提交、远端读回与完成状态由独立同步回执记录，不从本地stage推断远端成功。原始导出、材料冻结、Git对象上传和ref更新分别留证。下方保留前次同步原文，其中“最新”“当前”等措辞仅指相应历史日期。

## 以下为10-12及更早同步的历史说明

## 2026-10-12更新，证据截至10-10第35号

本次将Windows工作区完整`outputs/kv-cache-performance/`同步到`learning-materials/outputs/kv-cache-performance/`，保留既有文件与历史证据。新增M07三条候选的源码、正确性、编译观察、计时原始数据、独立审查、教学记录和[10-10独立日志](learning-materials/outputs/kv-cache-performance/obsidian/2026-10-10-开发日志-M07三条优化路线与结果复盘.md)。不删除旧失败记录，不改写历史实验。

工程61/100，M06为23/25、M07为8/25。35号已核验18进程、3240区间均值；三条路线均不足以支持稳定胜过原版CUDA，候选不晋升为默认新版本。M08真实模型接入与端到端验证未完成。同步不代表新的GPU运行、优化收益或独立掌握。

`source/`底座仍是2026-10-06 13:14:51 UTC的WSL导出。仅`source/experiments/`七份有明确来源的学习文件补至已验证版本，详见[source/SOURCE_OVERLAY_20261012.json](source/SOURCE_OVERLAY_20261012.json)：api/mask/support/check/grouped/full_groups来自35号快照，1-warp candidate来自26号快照。check_kv_targets.py与旧导出字节相同。其余原导出文件、EXPORT_MANIFEST和SOURCE_SNAPSHOT逐字节保留；旧manifest描述原导出，overlay清单描述本次增量，两者共同解释当前source目录。这不是新WSL全量导出。

排除`__pycache__`、`.pyc/.pyo`、`.git`及同类可再生缓存；NPZ、PTX、cubin、Nsight资产和既有ZIP按原字节保留。本次不创建新ZIP、不上传.env或凭据文件。普通文本做有限已知凭据模式扫描，无命中不构成穷尽安全保证。绝对路径和机器环境信息保留用于复现。

本次保持私有仓库范围，不包含虚拟环境、模型权重或整机镜像。运行前仍需核对固定上游提交、环境与路径；不要用本整理仓库HEAD替代上游版本检查。上传提交与核验回执由同步操作记录，本文仅说明文件范围与证据边界。

## 以下是前次同步的历史说明（不代表本次当前状态）

## 2026-10-09 最新增量

本次将Windows工作区的 `outputs/kv-cache-performance/` 同步到仓库 `learning-materials/outputs/kv-cache-performance/`，保留已有其他路径；没有重新导出WSL或修改source/快照。

新增/更新范围：今日独立开发日志、连续日志与进度/任务表、教学指南、M06-01～07脚本及原始记录、M06冻结区/ZIP/哈希清单、M07首课与本人初步假设。首页提供直接复习入口。

M06 23/25、总53/100；M06-08独立串讲待记录，M07候选未改/未运行。同步成功不代表新的GPU验收或优化收益。

排除可再生成的__pycache__、.pyc等缓存。NPZ、PTX/cubin、Nsight归档和ZIP属于实验资产，随对应材料保留。冻结M06包保持逐字节完整，未改写其中打包时的未同步状态描述。本次不改变仓库私有可见性。

source/仍为10月6日快照；最新接受的学员源码以M06/src为准。完整原版仓库/虚拟环境/模型权重不由本次材料增量提供。换机阅读直接使用GitHub；执行时按M06复现说明核对固定上游版本、环境及路径。

## 以下为10月6日同步时的历史记录

此次 WSL 快照时间：2026-10-06 13:14:51 UTC / 21:14:51 Asia/Shanghai。source/ 保持该快照不变。

随后补充发布了 21:23 起的 M05 收尾证据与最终笔记到 learning-materials/。当前工程结论：**M05 六配置原版 baseline v0 已验收**；个人独立掌握另列，M06 草稿已准备而未运行。最新入口：[M05基线收尾与后续优化路线](learning-materials/outputs/kv-cache-performance/M05基线收尾与后续优化路线-20261006.md)、[详细基线报告](learning-materials/outputs/kv-cache-performance/M05原版基线-v0.md)。

## 已同步内容

- source/ 来自用户实际 WSL 项目 /home/weltlaw/projects/mini-sglang-kv 的只读导出；包含源码、本人练习及 experiments 中的历史结果。
- 原始导出共 1029 个普通文件，7,419,517 字节；每项已按 EXPORT_MANIFEST.json 校验 SHA-256。
- 发布其中 1027 个普通文件，另附原始 EXPORT_MANIFEST.json、SOURCE_SNAPSHOT.json，并将 .dockerignore 记录恢复为指向 .gitignore 的 Git 符号链接。
- M05 pilot run-d9gjhywx 的 78 个文件已包含在 source/experiments/learning/m05-original-pilot/ 内，包括计划、逐次样本、运行脚本和汇总。
- learning-materials/ 原有材料保留，并定向更新 M05/M06 文件：收尾笔记、六配置基线报告、进度与任务表、原始数据审计、代表配置复测、真实 KV Nsight 记录以及 M06-01 模板和运行器。

WSL 能完成导出是这次恢复后的实际证据。同步不修改原 WSL 源码，也不运行 GPU 测试。M05 工程验收来自单独完成的原始数据审查、用户实际复测与时间线证据：共 41 次真实 store 调用，20 次对应标记区间。profile 时长不替代六配置 Graph 基线，未声明 NCU/DRAM 计数或整模型提速。

## 来源与版本

- 上游：[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)。
- WSL Git HEAD：9a91cfafe754aa85daee49998176275667eb58f2。
- 导出时 tracked_status 为空；这只描述已跟踪文件状态，experiments 中新增的未跟踪文件也按导出清单保留。
- 归档名：wsl-project-20261006T131451Z.zip。
- 归档 SHA-256：49b3af56900e2ca2226f03593d8476607f6f711cc58a9812fe4511173e41ce65。
- 原始文件与排除清单：[EXPORT_MANIFEST.json](source/EXPORT_MANIFEST.json)。
- 发布差异：[SOURCE_SNAPSHOT.json](source/SOURCE_SNAPSHOT.json)。
- [source/LICENSE](source/LICENSE)保留上游 MIT 许可证；文件内第三方声明照旧保留。

未导入上游或用户本地 Git 历史；本仓库的提交只记录整理和同步。source/ 是快照子目录，不能用本仓库 Git HEAD 代替上游固定提交。普通文件可执行位按导出 mode 保存；不保证 Linux 用户、时间戳和其他文件系统元数据。

## 明确排除

原导出清单记录 21 项排除：.git、.venv、Python / pytest 缓存，以及超过 20 MiB 的 .lab-tools/uv（安装工具）。导出器还设置凭据文件与模型权重等排除规则，实际命中项以清单为准。

发布额外跳过两项可重建内容：.vs/slnx.sqlite（0 字节编辑器缓存）、.lab-tools/uvx（346,992 字节安装工具）。它们仍列在原始导出清单中，发布清单另记排除原因。普通文本经有限凭据模式扫描未发现匹配；这不表示对所有可能的敏感信息作出穷尽保证。

导出只记录 .dockerignore → .gitignore 符号链接，本次发布将该链接恢复为 Git 的 120000 模式。此前备份将它展开为普通文件的状态仍保留在旧提交中。虚拟环境、模型权重、工具安装包和完整 WSL 系统镜像不在此次备份范围。

## 跨设备使用

可直接阅读源代码和实验记录。要执行历史脚本，先建立兼容环境，核对原始记录中的驱动、CUDA、PyTorch 与 Triton 版本，再调整原机绝对路径与激活脚本。部分计时脚本明确校验上游 Git 提交，建议在单独检出的上游固定版本中恢复相应实验文件；不要删去版本校验来假装环境一致。

本次成功备份是截至快照时间的文件状态；之后在 WSL 或 Windows 中的修改需要再次同步。本次列出的最终审查与收尾记录已补充发布；其余历史笔记保留各自日期。之后的新实验仍需再次同步。
