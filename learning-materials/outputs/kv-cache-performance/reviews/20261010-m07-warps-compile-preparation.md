# 28号：4/1-warp编译观察准备审查

日期：2026-10-10。结论：运行入口准备就绪，本地CPU检查与独立代码审查通过；WSL/GPU未运行，不增加工程分或编译观察验收。

## 目的与范围

对已接受的4-warp v1与1-warp候选各执行一次实际调用，固定B256、D128、C1024、FP16、交错缓存、输入gap0、int32索引0..255。捕获各自实际launch返回的compiled对象，保留可用中间表示、PTX、设备二进制及静态资源字段。无计时，不删编译缓存，不声称首次编译。

复用24号编译导出的概念及JSON转换、23号GPU数据封装与完整存储oracle、26号参数化启动探针/正确性运行、27号来源核验/安全导出。未修改这些既有脚本，也未修改用户内核或封装。新增文件为28号Python、shell、CPU检查脚本；讲义见[本课](../lessons/m07-03-warps-compile.md)。

本次fixture使用NumPy seed20261010，有限且可精确表示的FP16整数，首列固定为行号，V=K+1024。与27号形状/布局相同，但数值和分配地址不冒称相同；两臂使用相同fixture、独立GPU分配及初始状态。

## 核验与修正

- 实际执行的导入helper与27号接受快照核对哈希；接受的26/27运行器、v1、mask、支持模块、本人检查和candidate均有来源约束。当前文件变化时停止留档，不自动重写。
- B256/C1024使用新fixture，没有误用23号限定C32/B17的配置生成器。每臂独立检查完整缓存、未命中位置、K/V输入和indices，并保留before/after数组；两臂再互相比对。
- 探针由底层实际启动的返回值捕获compiled对象，reset清除旧对象；实际grid、width、stride、block与warp数受检查。没有从缓存目录任挑一个特化版本。
- 初稿使用周期数据，审查发现每8行重复，已改为可区分行的fixture，并用行0/8互换反例验证。
- 初稿当前helper与归档旧helper可能不一致，独立审查后补哈希门禁，并验证被修改的helper在GPU之前拒绝。
- 编译产物按实际字节保存；字段缺失或不可读标记UNAVAILABLE。样例行号来自本次保留的原文。寄存器/共享内存是静态字段，n_spills只记录原始值且不猜单位。
- 新建唯一结果目录；旧证据不覆盖，异常保留首错与可取得的现场，保存失败不伪装成GPU成功。

## CPU检查证据

最终记录：[CPU准备检查](20261010-m07-warps-compile-cpu.json)。11项负控制通过，包括错行交换、目标K/V与未命中缓存损坏、输入/indices损坏、错误warp、helper变更和禁止覆盖已有导出目录。旧M06编译产物仅用来验证序列化字节哈希和原始行号；不是新D128的编译证据。

额外检查包括缺失compiled对象/属性/API及不支持产物类型、探针实际返回与reset行为、Python语法、shell的LF/无BOM、无torch/Triton导入或GPU调用。独立审查确认最终来源边界、fixture、探针和输出解释无阻断问题。真实CUDA/Triton运行仍待用户执行。

| 文件 | SHA256 |
| --- | --- |
| 28-compare-kv-warps-compile.py | 767c3c5adfe9ea353229c2df928af432fdfc48da28b731d386c05de549fbf055 |
| 28-compare-kv-warps-compile.sh | c128383ba1b2b6f9249a0fb013f8d8ac6188c37245bafd394b144dbb4dfe42f2 |
| 28-warps-compile-cpu-test.py | 052be69af5af2671c164ef0f7c8343cb4cf4647787da2438bc3ec840ce446f0d |

## 下一步

用户在自己的PowerShell运行28号，随后审查两臂实际参数、完整正确性与编译文件。输出可支持关于静态分工的观察，不能单独证明变慢原因、运行时occupancy、DRAM瓶颈或未来加速。工程仍57/100；未自动同步GitHub。
