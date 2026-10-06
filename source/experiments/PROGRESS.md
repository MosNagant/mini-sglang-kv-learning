# KV Cache 写入性能项目：进度与教练约定

最后更新：2026-09-29 19:09，阶段 1，新版 Systems 已安装但 GPU 事件仍缺失；下一步验证 NVIDIA 给出的 WSL 时间戳绕过配置。
目标日期：2026-10-31。
项目目标：基于 mini-sglang 的 KV Cache 写入路径，完成用户能独立解释的 CUDA C++ 变体、正确性测试、可复现 benchmark、profiling、性能报告和申请作品。面向 AMD GPU 性能分析实习；本机 NVIDIA 结果不能冒充 AMD 硬件测量。

## 当前进度

- 已完成：用户已提供 GPU、CUDA、Python、Git、Nsight 和空间检查；远端 main 和本地 checkout SHA 已核对。已读取固定提交的依赖声明、store 内核与测试导入链。用户运行安装入口，完整日志显示 ENV_IMPORT_PASS、1 test collected、SETUP_PASS；PyTorch 2.9.1+cu128、TVM-FFI 0.1.4，GPU 分配检查通过。
- 历史证据：2026-09-29 CUDA 向量加法示例成功，RTX 4070 Ti、驱动 576.52、nvcc 12.8.93、GCC 13.3.0、CMake 3.28.3；这不是 mini-sglang baseline，也不证明用户已掌握 CUDA。
- 新完成：用户安装 python3.12-dev 后，2026-09-29 18:11 的原版测试为 1 passed in 17.78s，16 个 batch（1 至 32768 的 2 次幂）均执行完并打印对照数据；该耗时是整次 pytest 时间，不是内核延迟。TVM-FFI 可选扩展警告在这次日志未再出现。首跑 Python.h 错误已解决并保留原始失败证据。
- 运行时长澄清：本轮入口目录时间 18:11:02、结束时间 18:11:38，整体约 36 秒。用户询问为何这么快；30 分钟只是原版测试超时上限，流程完成即退出。运行仅做一次原版测试及小规模 profiler 探针，编译缓存保留，NCU 权限失败迅速退出；此前数分钟到十几分钟的估计偏长。下一轮不为延长时间而无目的重复，正式基准矩阵待用户参与设计。
- 当前阻塞：用户确认已在 Windows 允许所有用户访问 GPU 性能计数器并应用。18:18 的 Compute 原始错误已变为“driver resource was unavailable or ... permission”，不是最初的显式 ERR_NVGPUCTRPERM。脚本曾因匹配错误帮助链接中的字符串而误分类，现已修正。资源竞争/驱动会话/兼容性等尚未确定，不重复要求用户改同一个设置。
- Systems 新证据：18:18 的 SQLite 含 14 条 CPU 侧 CUDA runtime API 和 1 条同步事件，但缺 GPU kernel 事件表。DIAGNOSTIC_EVENT 明确警告已安装驱动 CUDA 12.9 接口不受当前 Systems 构建支持、回退使用 12.8 库；还存在 Unified Memory/P2P tracing 限制信息。兼容警告是排障线索，不是已证明的唯一原因。
- 本轮安装阻塞已定位：19:02 的 05 入口在 APT 模拟安装前后失败，退出码 100，未运行两项 profiler。实际 `package-policy.txt` 的 Candidate 为 `2025.1.3.140-251335620677v0`（NVIDIA wsl-ubuntu 源）；助手此前误将仓库 .deb 文件名的 `-1` 后缀当作 APT 的内部 Version。新 Systems 尚未安装，旧环境成功结果仍有效。
- 最新验证：`logs/profiler-compatibility-run-20260929-190613-i0a4w2/` 已确认 Systems 2025.1.3.140-251335620677v0 安装成功，使用 CUPTI 12.9，驱动版本不支持警告消失。仍只有 14 条 runtime API 记录、无 kernel 事件表；Compute 单 duration 指标也报 DRIVER_RESOURCE_OR_PERMISSION_UNAVAILABLE。版本更新未解决采集缺失，不能继续将其当作已确认根因。
- 下一项任务：运行 `06-nsys-wsl-timestamps.sh`，验证 NVIDIA 工程师公布的 WSL 时间戳换算绕过 `CuptiUseRawGpuTimestamps=false`。入口通过新 Systems 的 `-z` 获取用户配置路径，备份已有文件、保留其他配置，再只采集一次现有 vector_add；若配置已经为 false，则报告并停止，不原样重试。验收查看 SQLite 中是否有正时长的 GPU kernel 事件，而非只看报告文件存在。此轮不安装工具、不重跑 Compute、不重启系统或 WSL。
- 时间线精度边界：该绕过使用较低精度的时间换算方式，成功后用于观察活动顺序和时间线；正式微秒级延迟结论仍需后续由用户设计的可复现计时和交叉验证。Compute 指标不可用单独记录，不能把 CPU API 时间冒充 GPU kernel 时间，也不能把 Unified Memory/P2P 限制提示直接当作所有采集失败的根因。
- 脚手架修正：02/03 入口已补 Python.h 预检且用户实际复测成功；03 保存 Python 系统包版本和 include 路径。04 明确区分错误行中的 ERR_NVGPUCTRPERM 与驱动资源/权限不确定错误。05 APT 版本选择和安装已由用户执行验证；06 静态检查与配置备份/保留行为验证后交给用户执行，实际时间线恢复仍待结果。所有新运行另建目录保留原日志。
- 已掌握：尚未通过本项目解释或小实验评估；已有终端操作经历不等于掌握 GPU 性能分析。
- 待补：此前 3 个环境解释问题尚未回答（驱动/Toolkit/PyTorch CUDA 的关系、收集与执行的区别、系统 Python 与项目 .venv 的差别），排障期间不重复追加问题。当前错误补充一个实际知识点：Python 解释器和已安装 wheel 可运行，不代表编译 Python C/C++ 扩展所需的 Python.h 已安装。后续按实际代码和数据补线程映射、访存、计时及性能归因。

## 上游版本

- 仓库：https://github.com/sgl-project/mini-sglang
- 用户此前核对：`9a91cfafe754aa85daee49998176275667eb58f2`。
- 2026-09-29 用户执行 `git ls-remote ... refs/heads/main`，实时返回 `9a91cfafe754aa85daee49998176275667eb58f2`，已完成远端确认。
- 选定 baseline commit：`9a91cfafe754aa85daee49998176275667eb58f2`。安装日志与环境 JSON 已确认本地 HEAD 和源码路径；锁定后不随 main 漂移。
- 本轮通过 GitHub connector 读取上述 commit 的源文件；之前网页缓存不能实时验证的问题已不再阻塞源码检查。

## 阶段 1 依赖与检查范围

- 详见 `ENVIRONMENT-20260929.md`，本轮依赖选择及测试边界已记录。
- 使用最小内核实验依赖和原始源码 PYTHONPATH 导入，尚未安装完整推理服务。锁定文件、激活脚本、原始环境日志和导入结果将放入 WSL 项目的 `experiments/`。
- 当前原版测试含正确性检查、`torch.compile` 对照和 CUDA Graph 计时；不能把其打印数据直接当成覆盖所有形状/波动统计的项目 baseline。
- 原版形状：cache [1048576, 2, 128]、FP16，batch 2^0 到 2^15。正确性部分输入 K/V 是 qkv 切片，随后计时前转为 contiguous；比较时必须记录这种布局差别。上游断言只验证目标行，未覆盖未命中位置不变。原版输出的 Baseline 是 torch.compile 索引赋值，Our Impl 是上游原版 CUDA，均不是用户优化版。以 ms 保留三位小数的显示不能支撑细小延迟变化的结论。
- 原版运行入口固定种子 0 并保留现有编译缓存；该次运行只是正确性和初步性能探查，不作为已控制全部条件的正式 baseline。
- 成功证据：`logs/upstream-run-20260929-181115-pFBvSx.log`；profiler 证据目录：`logs/away-run-20260929-181102-OHBHvP/`。原版表格另提取到 `observations/upstream-20260929-181115.csv`，明确标记延迟来自舍入后的 ms 显示值；原始日志保留，不能把此数据当作用户变体或正式重复测量结果。
- 没有替用户编写 CUDA kernel、线程映射、地址计算或微基准计时核心。

## 教练约定

- 每阶段先说明目的、问题及证据；每轮只给一组可执行任务，收到用户输出再推进。
- 核心 CUDA kernel、warp/线程映射、地址与边界处理、计时核心、profiler 解释和结论，由用户先提方案或写初稿。助手提供问题、伪代码、局部提示与审查；未经请求不代写完整核心实现。
- 环境排障、脚手架和记录工作由助手积极完成；不因非阻塞理论拖延项目。
- 每个里程碑用 3–5 个基于刚完成代码/数据的问题检查理解；能从证据推导才标记掌握。
- 每次对话更新本文件。WSL 仓库建立后把项目记录放入项目内，当前 Windows 输出目录作为起步记录。
- Codex 当前不能直接操作用户账户的 WSL；以用户输出和保存的日志为依据，不假装已执行命令。

## 实验边界

- 四周顺序：原版测试/调用链/微基准/首次 profiling → 瓶颈假设及用户 CUDA 变体 → 重复测量/消融/收益与回退 → Qwen3-0.6B 整机验证及 README/图表/报告/面试讲解。
- 前期不下载模型。已读取固定版本 store.cu，warp/token 映射将在原版测试后由用户对照源码解释；不预设多个 warp/token 更优。
- 覆盖 batch：1、4、16、64、256、1024、4096、16384。每 token 字节宽度按真实 KV 布局与显存可行性确定。
- 正确性包含互不重复目标索引、未写位置不变、边界尺寸；哨兵/空 batch/布局约束等在读固定版本源码后确定。
- 修改前保存原始数据；记录 commit、输入、随机种子、硬件、软件版本及运行条件。
- 分开报告 JIT、Python 完整调用、GPU kernel、整机吞吐；复测报告中位数、波动、回退及有效写入带宽定义。
- Nsight 能力需实测。工具存在不等于有硬件计数器权限；不可用指标明确记录，不编造。
- 在数据驱动选定且预先登记的至少两个中大 batch 形状上挑战约 15% 稳定 kernel 延迟改善；未达到要如实分析，不挑最佳点或把 kernel 加速等同整机加速。

## 尚未解决的环境风险

此前 Windows 开机附近有 DevManView 批量卸载磁盘设备，造成 WSL 的 Hyper-V 存储设备缺失。已两次恢复，但调用来源未确定。若 WSL 启动再次失败，先核对该证据，不默认重装 Ubuntu。当前任务按用户可进入 WSL 的实际输出继续。
