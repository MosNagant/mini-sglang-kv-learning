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

## 安装验证结果：2026-09-29 17:58

证据：`logs/setup-kernel-env-20260929-172950.log` 和 `logs/environment-20260929-172950.json`，已直接读取保存日志并与用户粘贴的成功输出核对。

- 本地 checkout SHA 与选定提交一致，源码位于 `/home/weltlaw/projects/mini-sglang-kv`。
- Ubuntu 24.04.5 LTS，内核 `6.18.33.2-microsoft-standard-WSL2`。
- 项目 Python `/home/weltlaw/projects/mini-sglang-kv/.venv/bin/python`，3.12.3。
- PyTorch `2.9.1+cu128`，`torch.version.cuda=12.8`；GPU 分配与同步检查通过。
- GPU RTX 4070 Ti，计算能力 8.9，显存 12878086144 字节。
- TVM-FFI 0.1.4、Transformers 4.57.3、Triton 3.5.1、pytest 9.1.1、pyzmq 27.2.0、msgpack 1.2.3。
- `uv pip check` 报告 50 个包兼容；ENV_IMPORT_PASS、1 test collected、SETUP_PASS。收集不等于执行，尚无 KV 正确性或性能结果。
- 保存了可选 torch C DLPack 扩展 JIT 失败、EnvTensorAllocator 未启用的警告。当前导入和 GPU 检查通过，先观察 KV 实际执行；尚未据此判断调用开销或添加依赖。

## 首次原版测试失败：2026-09-29 18:04–18:05

- 原始日志：`logs/upstream-run-20260929-180451-GTVqUB.log`，未覆盖。
- pytest：1 failed，1 warning，11.13s；没有得到完整性能对照结果。
- 直接原因：GCC 编译 Triton 的 cuda_utils.c 时找不到 `Python.h`，include 参数为 `/usr/include/python3.12`。
- 失败路径：test_store_cache → compare_memory_kernel_perf → perf_cuda(baseline) → torch.compile / Inductor / Triton 编译辅助 C 扩展。
- 按调用顺序，首个 batch=1 的 store_cache 调用及 K/V 写入断言已通过；不能推广成全套形状正确性通过。
- 修复方案：Ubuntu 内安装 python3.12-dev（含 libpython3.12-dev 依赖），然后原样复测。该包提供 Python 3.12 开发头文件：[Ubuntu 官方包说明](https://packages.ubuntu.com/noble/python3.12-dev)。安装实际结果待用户输出。
- 助手修复脚手架：02/03 入口增加 Python.h 预检，03 保存系统 Python 包版本和 include 路径；没有变更依赖锁、CUDA 内核、原版断言或性能计时。

## 原版测试通过与 profiler 限制：2026-09-29 18:11

- 修复：python3.12、python3.12-dev、libpython3.12-dev 均为 3.12.3-1ubuntu0.17，Python.h 预检通过。
- 原版测试：`logs/upstream-run-20260929-181115-pFBvSx.log`，1 passed in 17.78s。这个数字包含编译/分配/检查/计时等 pytest 工作，不是单次 kernel 时间。16 个 batch 均完成；仍只有原版断言覆盖，未命中位置/更多宽度等尚待后续补齐。
- 本轮完整日志没有再出现 torch C DLPack 扩展警告；不凭此日志单独推断 allocator 功能是否启用。
- GPU 开始时 54°C、P8、13W、38% 利用率、881MiB 显存占用；此轮为探查，不具备正式重复测量的控制条件。
- Nsight Compute 探针：`logs/away-run-20260929-181102-OHBHvP/ncu-profile.log`，ERR_NVGPUCTRPERM，未获硬件计数器。已核对 [NVIDIA WSL 要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#linux-and-wsl) 和 [权限设置说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)：WSL 计数器权限需在 Windows 宿主机 NVIDIA 控制面板开启。此机器为驱动 576.52，采用该版本可用的控制面板路径；不套用新版 R610 Linux capabilities 配置。
- Nsight Systems 探针：nsys-vector-add.nsys-rep 已生成，但 nsys-kernels.csv（含 stats 原始输出）明确写 does not contain CUDA kernel data。报告文件存在不代表 CUDA 采集成功，原因仍未确定。
- 两个探针使用旧 cuda-starter 的 vector_add；不得把任何探针结果作为 KV 内核性能数据。
- 下一步仅改已确认的 Windows 权限设置，使用 `04-away-run.sh --profilers-only` 单独复查，原测试和安装不重复。新增无 kernel 数据的明确状态并保存 SQLite，以便必要时进一步排查。

## 权限设置后复查：2026-09-29 18:18

- 用户明确确认已在 Windows 控制面板选择允许所有用户访问并应用。
- 原始证据目录：`logs/away-run-20260929-181808-pa4Ovc/`，保留原摘要，包括当时的误分类。
- Compute 原始日志这次为 driver resource unavailable OR permission 错误；ERR_NVGPUCTRPERM 只出现在帮助 URL 中。04 旧版宽泛 grep 导致错误摘要仍称权限被拒绝；已修正为只匹配 `==ERROR== ERR_NVGPUCTRPERM` 错误行，另设 DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE。
- 本轮以只读方式打开 Systems SQLite：14 条 CUPTI runtime API 记录、1 条同步记录，没有 kernel 事件表。诊断记录显示驱动 CUDA 12.9 不受此工具构建支持而回退至 CUPTI 12.8；还有 Unified Memory/P2P trace 限制提示。没有把 CPU runtime API 延迟当作 GPU kernel 时间。
- 最小下一步：单独安装 nsight-systems-2025.1.3（2025.1.3.140-1），版本来自 [CUDA 12.9 GA 官方配套表](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)，包已在 [NVIDIA Ubuntu 24.04 仓库](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/) 核实。保持 nvcc 12.8；以新 Systems 的包专属路径采集。Compute 原有版本先请求单个 duration 指标来缩小问题范围。
- [NVIDIA Compute FAQ](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq) 将 driver resource unavailable 与性能监视资源获取失败相关联，可能存在竞争；当前没有证据指向特定占用程序，未停止任何用户进程，也未重启系统或 WSL。

## Systems 安装版本字符串修正：2026-09-29 19:02

- 原始证据：`logs/profiler-compatibility-run-20260929-190206-cEaAVQ/package-policy.txt`、console.log、install-plan.txt。退出码 100，NCU_MINIMAL 和 NSYS_PROBE 均为 NOT_RUN。
- 已更新的软件源是 NVIDIA wsl-ubuntu/x86_64，APT 候选版本为 `2025.1.3.140-251335620677v0`；助手脚本此前指定的 `2025.1.3.140-1` 不在 APT 版本表中。
- NVIDIA 仓库网页的 .deb 文件名虽含 `-1`，但不能用文件名代替实际 APT Version 字段。此次错误来自助手安装脚手架的版本假设，不是 CUDA 计算失败。
- 修复后的 05 从实际 `apt-cache policy` 读取候选版本，限制为预定上游构建 2025.1.3.140，将完整选择写入 package-selection.txt，并用精确版本参数安装、核对安装后 Version。用户原始失败日志保留不改。

## Systems 安装后采集仍失败：2026-09-29 19:09

- 证据目录：`logs/profiler-compatibility-run-20260929-190613-i0a4w2/`。Systems 2025.1.3.140-251335620677v0 已实际安装并调用；CUPTI 12.9 加载成功，旧版的驱动接口不支持警告消失。
- 结果：NCU_MINIMAL=DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE；Systems 仍为 14 条 runtime API、无 kernel 事件表。版本不匹配因素已消除，但未恢复 GPU 采集。
- 下一项单独假设：WSL GPU→CPU 时间戳换算问题导致活动未正确出现在报告中。NVIDIA 工程师在 [缺少 GPU kernel 数据讨论](https://forums.developer.nvidia.com/t/nsys-doesnt-show-cuda-kernel-and-memory-data/315536) 给出 CuptiUseRawGpuTimestamps=false 的绕过并说明时间精度降低；[另一官方讨论](https://forums.developer.nvidia.com/t/nsight-system-error-unified-memory-cannot-be-traced-on-devices-that-dont-support-peer-to-peer-transfers/294742) 说明 Unified Memory 提示只影响相关跟踪功能，不应据此否定所有其他 GPU 数据。
- 06 入口备份原用户配置、修改单个设置、只复测 Systems；以 SQLite 正时长 kernel 事件为采集成功条件。此配置一旦应用会保留供后续采集使用，原始备份路径写入 configuration-change.json；若设置已经为 false 则停止并记录现状。
- 如果成功，报告中必须记录时间戳绕过及其精度限制；这里仍只是 vector_add 工具能力验证。Compute 硬件计数器没有恢复前，不报告相关占用率/访存指标。

## Systems GPU 时间线恢复：2026-09-29 19:16

- 原始证据：`logs/nsys-wsl-timestamps-run-20260929-191618-eHnaZP/`。用户输出与镜像 SQLite 已核对：14 条 CUDA runtime API；1 条 CUPTI_ACTIVITY_KIND_KERNEL，正时长 5824 ns，名称为 vector_add(const float *, const float *, float *, int)。NSYS_WSL_TRACE=GPU_KERNEL_EVENTS_PRESENT。
- 唯一此次配置变动：Systems 用户配置中的 CuptiUseRawGpuTimestamps=false。configuration-change.json 记录目标 `/home/weltlaw/.config/NVIDIA Corporation/nsys-config.ini`；原文件不存在，backup=null，不是备份失败。
- 结论范围：在已安装 Systems 2025.1.3 的当前环境中，该配置使此探针的 GPU kernel 事件出现在报告里。保留 NVIDIA 官方说明的较低时间戳精度限制，不据此宣称所有 WSL profiling 问题均已解决。
- 5824 ns = 5.824 µs 只是一条向量加法事件，不是 KV baseline。Instances=1，因此 StdDev=0 没有提供跨次波动证据。正式测量需预热、独立重复和清楚的计时范围，由用户先设计计时核心。
- 日志提醒 device-side CUDA Event completion trace 开启，可能增加开销及产生虚假流依赖；未来无需该功能的采集加入 --cuda-event-trace=false。未采集 NVTX 是此程序没有相关标记的预期情况。Unified Memory/P2P 跟踪限制保留，但普通 kernel 已可见。
- NCU 本次未重测，仍为 DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE；不声称计数器已可用。该限制不阻塞调用链阅读和微基准设计。
- 下一步进入固定源码的数据契约与正确性覆盖阅读。本轮未修改 CUDA 内核、Python glue 或计时核心。

## 初始环境选择（历史计划，安装与执行进展见上文）

- 克隆到 `~/projects/mini-sglang-kv`，创建 `kv-cache-study` 分支并检出上述提交；原有 `cuda-starter` 保留。
- 独立 `.venv` 使用系统 Python 3.12；项目本地安装 uv 0.12.20，不重装 Ubuntu/CUDA。
- 固定 PyTorch 2.9.1+cu128、apache-tvm-ffi 0.1.4、Transformers 4.57.3；其余内核测试导入依赖经 uv 解析后写入带哈希的锁定文件，再安装。
- 当前使用原始源码的 `PYTHONPATH` 导入，只准备内核测试环境；完整推理服务依赖尚未安装。避免把部分依赖环境误称为完整 mini-sglang 服务安装。
- 激活脚本为前期测试设置模型离线模式；末期模型验证时再显式调整。
- 只检查导入、PyTorch GPU 分配与原版测试收集，不执行 KV 内核、计时或模型推理。

依据：[固定提交的依赖声明](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/pyproject.toml)、[固定提交的原版测试](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/tests/kernel/test_store.py)、[PyTorch 官方历史版本安装](https://pytorch.org/get-started/previous-versions/)、[uv 官方安装](https://docs.astral.sh/uv/getting-started/installation/)。
