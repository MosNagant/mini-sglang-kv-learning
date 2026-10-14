# M07 实现变化与研究范围

本目录`src/`为2026-10-14从真实WSL导出中逐文件校验后保存的学习源码。来源与SHA256见[source-manifest.json](source-manifest.json)，可运行`python package_sources.py --verify`离线核验。原版参考的两份文件与许可证在`original_source/`，完整上游和Python环境仍是运行前提。

## 原版、v1与三个候选

| 实现 | 代码 | 改动点 | 状态 |
| --- | --- | --- | --- |
| 原版CUDA | [store.py](original_source/python/minisgl/kernel/store.py)、[store.cu](original_source/python/minisgl/kernel/csrc/jit/store.cu) | 固定上游实现，作为比较对象 | 当前保守默认 |
| 冻结v1 | [mask内核](src/triton_store_mask.py)、[公开入口](src/triton_store_api.py) | 一行/program；参数化宽度与stride；行宽向上取块；列mask | 正确性已验收，保留对照 |
| 1-warp | [候选入口](src/triton_store_candidate.py) | 复用v1内核，仅num_warps由4改1 | 27号无稳定收益，不晋升 |
| grouped | [四行内核和入口](src/triton_store_grouped.py) | 每program最多4行，二维广播，独立行/列mask；grid向上取整 | 正确性已验收，研究候选 |
| full_groups | [完整组特化](src/triton_store_full_groups.py) | 编译期满足B整除RPP时，行条件为全True；尾组仍用rows<BATCH | 35号未支持稳定收益，不晋升 |

帮助归因：1-warp参数由学员修改；grouped的三式经教练示范后由学员应用；full_groups框架由教练提供、学员在提示后填写整除条件。检查、计时组织和归档由教练辅助。10-14新增访存运行器、选择策略、独立验证、统计和文档由教练按用户明确委托完成。

## 为什么分组需要两种mask

逻辑输入行号是`pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)`，目标槽从这些行对应的indices取。输入行号与目标槽号分别乘各自真实stride，不能用pos当作输入行号。

行条件保护尾部多出来的逻辑行，列条件保护向上取整后多出来的列。必须先用行mask保护indices读取，再以行列条件的交集保护K/V读写。B5、RPP4、D192、BLOCK256是同时存在尾行和尾列的反例；若只有store加mask，非法load仍可能发生。

`BATCH % ROWS_PER_PROGRAM == 0`依赖每program行数，和warp数没有直接代换关系。编译期条件消失与实际运行变快分别验证；34号编译证据不能替代35号计时。

## 为什么36不另改一个新kernel

旧35号在改变D时也改变了layout/gap/guard，因此不能把跨D的差异全归因于规模。36固定B256、D128、C1024，让相同源码分别面对不同内存条件：这是隔离变量的实验，而不是给候选增加功能。

输入宽度与行距分别控制有效搬运量和寻址。实验中的8字节偏移保持原版向量包的合法对齐；四套地址轮换保留实际独立分配和完整backing。逻辑字节数与实际显存事务数不同；没有硬件计数器便不报告“实测DRAM带宽”或“合并效率”。

## 为什么37可以选择继续使用原版

自动选版不应要求必定选中一个Triton候选。当前历史数据没有稳定胜过原版的候选，因此冻结规则保留原版。选版元数据、来源和未知情形要明确；不得通过重复测量、挑最好的一轮或在验证集上调门槛制造提升。

## 融合与M08的边界

M07没有真实模型调用链的完整测量。把RoPE、量化等与cache写入融合需要确认相邻操作、依赖、dtype、布局、同步和模型输出一致性。它属于M08真实输入/调用点验收后的工作，当前只记录该依赖，不实施一个与实际模型脱离的“融合加速”来凑方法数量。

因此M07收尾的交付边界是：受控的算子改动、正确性、性能对照、编译证据、访存敏感性、保守选版和可复习报告。模型吞吐、请求延迟、AMD实机及完整编译器开发不在已完成结论中。
