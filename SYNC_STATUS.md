# 同步范围与待补内容

记录日期：2026-10-06。本页描述此次同步所整理的文件范围，不构成 WSL 已恢复或实验已重新运行的证明。

## 当前整理内容

| 内容 | 来源与数量 | 范围 |
| --- | --- | --- |
| [source/](source/) | GitHub 官方上游固定提交；121 个文件，905,126 字节 | 包含上游代码、文档、资源和许可证；没有本人 WSL 的 `experiments/`。 |
| [learning-materials/](learning-materials/) | 已有 Windows 工作区；894 个文件 | 讲义、教练脚手架、学习记录、既有实验快照与证据，以及 CUDA / WSL 说明。 |
| 本人历史源码示例 | [2026-10-02 Triton 快照](learning-materials/outputs/kv-cache-performance/logs/kv-three-rows-results/run-20261002-140817-qyuxk3bo/triton_store_first.py) | 原样保存当次固定 64 宽实现，未重建为最新本人源码。对应[运行归档](learning-materials/outputs/kv-cache-performance/logs/kv-three-rows-results/run-20261002-140817-qyuxk3bo/)与[审查](learning-materials/outputs/kv-cache-performance/reviews/20261002-width64-pass-review.json)可一并阅读。 |

没有导入上游或用户本地项目的 Git 历史；`.git` 不在复制/导出范围。此仓库的提交记录只记录本次整理及后续同步，不能替代原项目历史。

用户原 WSL 项目 `/home/weltlaw/projects/mini-sglang-kv` 保持原状。本次整理未改写该项目、未恢复或覆盖本人文件，也未重新运行 benchmark 或 GPU 测试。

## 尚待补齐

当前 WSL 连最小 `/bin/echo WSL_OK` 调用及停止分发命令也出现挂起，导出脚本尚未开始生成项目压缩包。因此以下内容仍待 WSL 恢复后读取：

- 本人当前工作区中的最新源码与未提交改动。
- 完整 `experiments/`，包括实际激活脚本、依赖锁定、本人实验和原始运行目录。
- M05 首批试测 `experiments/learning/m05-original-pilot/run-d9gjhywx/` 中的原始逐次样本、配置与控制台证据。

当前已有的 [M05 结果转录](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-summary-transcribed.json)是用户控制台输出的转录；[试测审查](learning-materials/outputs/kv-cache-performance/reviews/20261004-original-pilot-review.md)明确记录其依据。它们不能替代尚未导出的原始样本。历史检查归档也不能证明 WSL 最新代码与当时相同。

## 上游来源与校验

- 官方仓库：[sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)。
- 固定提交：[`9a91cfafe754aa85daee49998176275667eb58f2`](https://github.com/sgl-project/mini-sglang/tree/9a91cfafe754aa85daee49998176275667eb58f2)。
- 获取来源：[GitHub 官方 codeload 归档](https://codeload.github.com/sgl-project/mini-sglang/tar.gz/9a91cfafe754aa85daee49998176275667eb58f2)。
- 归档大小：587,654 字节。
- 归档 SHA-256：`576860aa97a8c844393bc753830efbc81c44563ad929989d7dadc2621b2afb95`。
- 已检查归档路径安全，并将解包后的 121 个文件逐一与归档内容核对。
- [source/LICENSE](source/LICENSE)为 MIT，原有版权及文件内第三方声明保留。
- 官方归档中 `.dockerignore` 是指向 `.gitignore` 的符号链接。本次 Windows 整理将其展开为与目标等字节的普通文件，其余文件内容按归档保留；不声称保留全部 Linux 文件系统元数据。

## WSL 恢复后的导出步骤

以下命令只在原电脑上、WSL 已恢复后执行。当前挂起期间不要反复提交导出任务。

先确认最小命令能正常输出并返回 PowerShell 提示符：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /bin/echo WSL_OK
```

成功后，运行原电脑 Windows 工作区里已经准备好的导出脚本：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /usr/bin/python3 /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/github-upload/export-wsl-project.py
```

脚本读取 `/home/weltlaw/projects/mini-sglang-kv`，在 Windows 的 `outputs/github-upload/` 下创建带时间戳的 `wsl-project-*.zip` 与 `wsl-export-ready.json`；出现 `EXPORT_READY=` 才表示完成。压缩包内的 `EXPORT_MANIFEST.json` 记录版本、已跟踪改动状态、文件哈希及排除项。导出不等于已经补传到 GitHub；之后仍需核对清单、合并文件并提交。

该脚本保留在原电脑的上述准备目录，本仓库当前不包含它。换机后不要照抄这个原电脑路径。导出跳过 `.git`、虚拟环境、缓存、模型权重和可能包含凭据的指定文件；超过 20 MiB 的文件及符号链接会记入清单而不直接复制，需要另外审查。它不是完整系统镜像。

## 跨设备使用说明

阅读源码和笔记无需 CUDA。重新运行实验则需为新设备建立兼容环境，并修正历史记录中的 Windows / Linux 绝对路径、WSL 分发名、用户名、工作目录及 `PYTHONPATH`。当前 `source/` 是官方上游代码，不能视为原 WSL 环境的完整备份。

虚拟环境、编译/Python 缓存、WSL 安装包及下载分片均未同步。安装和修复说明保留了原机器上下文，其中指向未同步安装包或系统日志的历史路径在新设备上可能不存在；先阅读说明，再按目标设备实际环境调整。
