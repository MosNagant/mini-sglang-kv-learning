# 同步范围与来源

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
