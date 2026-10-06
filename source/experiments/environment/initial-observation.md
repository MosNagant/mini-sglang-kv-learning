# 环境观察：2026-09-29 17:17

来源：用户在 WSL 中执行第一组检查后的输出。这里是整理记录，不能替代 Linux 主目录中的原始 `cuda-env-check-*.txt` 日志。

| 项目 | 当前证据 |
| --- | --- |
| 发行版 | 本轮片段显示 Ubuntu noble；VERSION_ID/内核行未贴出。此前成功日志为 Ubuntu 24.04 / WSL 2；安装导入检查会补录完整值 |
| GPU | NVIDIA GeForce RTX 4070 Ti |
| 显存 | 总计 12282 MiB，检查时使用 816 MiB；不是正式实验可分配预算 |
| Windows 驱动 | 576.52；WSL NVIDIA-SMI 为 575.57.04 |
| 驱动显示的 CUDA 支持能力 | 12.9 |
| CUDA 编译器 | `/usr/local/cuda-12.8/bin/nvcc`，release 12.8，V12.8.93 |
| Python | `/usr/bin/python3`，3.12.3 |
| Git | `/usr/bin/git`，2.43.0 |
| uv | 当前 PATH 中不可见 |
| Nsight Compute | `/usr/local/cuda-12.8/bin/ncu`，2025.1.1.0 |
| Nsight Systems | `/usr/local/cuda-12.8/bin/nsys`，2024.6.2.225 |
| Linux 文件系统可用 | 945G（虚拟文件系统视图） |
| Windows C 盘可用 | 162G；需要同时考虑承载虚拟磁盘的物理空间 |
| GPU 当时状态 | 52°C，P8，13W，利用率 37%；不能视为稳定空闲性能基线 |
| 实时 `refs/heads/main` | `9a91cfafe754aa85daee49998176275667eb58f2` |

Nsight 仅确认命令存在及版本，尚未确认计数器权限或实际采集能力。

## 下一组环境选择

- 克隆到 `~/projects/mini-sglang-kv`，创建 `kv-cache-study` 分支并检出上述提交；原有 `cuda-starter` 保留。
- 独立 `.venv` 使用系统 Python 3.12；项目本地安装 uv 0.12.20，不重装 Ubuntu/CUDA。
- 固定 PyTorch 2.9.1+cu128、apache-tvm-ffi 0.1.4、Transformers 4.57.3；其余内核测试导入依赖经 uv 解析后写入带哈希的锁定文件，再安装。
- 当前使用原始源码的 `PYTHONPATH` 导入，只准备内核测试环境；完整推理服务依赖尚未安装。避免把部分依赖环境误称为完整 mini-sglang 服务安装。
- 激活脚本为前期测试设置模型离线模式；末期模型验证时再显式调整。
- 只检查导入、PyTorch GPU 分配与原版测试收集，不执行 KV 内核、计时或模型推理。

依据：[固定提交的依赖声明](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/pyproject.toml)、[固定提交的原版测试](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/tests/kernel/test_store.py)、[PyTorch 官方历史版本安装](https://pytorch.org/get-started/previous-versions/)、[uv 官方安装](https://docs.astral.sh/uv/getting-started/installation/)。
