# KV Cache 写入性能项目：进度与教练约定

最后更新：2026-09-29，阶段 1，原版测试首跑失败，已定位缺少 Python 开发头文件，等待安装并复测。
目标日期：2026-10-31。
项目目标：基于 mini-sglang 的 KV Cache 写入路径，完成用户能独立解释的 CUDA C++ 变体、正确性测试、可复现 benchmark、profiling、性能报告和申请作品。面向 AMD GPU 性能分析实习；本机 NVIDIA 结果不能冒充 AMD 硬件测量。

## 当前进度

- 已完成：用户已提供 GPU、CUDA、Python、Git、Nsight 和空间检查；远端 main 和本地 checkout SHA 已核对。已读取固定提交的依赖声明、store 内核与测试导入链。用户运行安装入口，完整日志显示 ENV_IMPORT_PASS、1 test collected、SETUP_PASS；PyTorch 2.9.1+cu128、TVM-FFI 0.1.4，GPU 分配检查通过。
- 历史证据：2026-09-29 CUDA 向量加法示例成功，RTX 4070 Ti、驱动 576.52、nvcc 12.8.93、GCC 13.3.0、CMake 3.28.3；这不是 mini-sglang baseline，也不证明用户已掌握 CUDA。
- 当前阻塞：首跑日志 `logs/upstream-run-20260929-180451-GTVqUB.log` 的第 36 行报 `fatal error: Python.h: No such file or directory`。Triton 用 GCC 编译 cuda_utils.c，include 指向 /usr/include/python3.12；失败在 torch.compile 对照实现首次编译。调用栈已越过首个 batch=1 的原版 CUDA 调用与两条写入结果断言，但剩余形状和性能对照尚未通过。TVM-FFI 可选扩展警告仍保留，是否同因需复测，Nsight 计数器仍未验证。
- 下一项任务：用户要外出，已将修复与验证串成 `04-away-run.sh`：开始时验证 sudo，安装 python3.12-dev，再调用原版测试入口（30 分钟上限）。仅在原版测试通过后，使用此前通过的 cuda-starter vector_add 检查 Nsight Systems / Compute 采集，各 3 分钟上限。每项状态、原始日志、报告和摘要同时保存到 WSL 实验目录与 Windows 输出目录。Nsight 失败独立记录，不能当作原版测试失败或伪造指标；此探针不是 KV 性能数据。用户回传摘要后再决定下一步。
- 脚手架修正：02/03 入口已补 Python.h 预检；03 保存 Python 系统包版本和 include 路径。新的运行各建独立目录，原失败日志保持原样。当前只做静态语法检查，实际修复和采集结果仍待用户运行。
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
