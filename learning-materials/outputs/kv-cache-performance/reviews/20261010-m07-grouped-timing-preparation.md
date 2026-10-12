# 30号grouped计时准备审查

日期：2026-10-10。结论：入口准备完成，可以交用户在既有WSL环境运行。**GPU和性能尚未运行。**

## 变更与依据

新增 `30-compare-kv-grouped.py`、`.sh` 和有界CPU检查入口 `30-grouped-timing-cpu-test.py`。沿用27号计时政策，三臂改为原版CUDA、冻结4-warp单行v1、已接受4-warp四行grouped；旧1-warp候选不进入计时。旧运行器、用户内核与历史证据未改。

事前政策见[本课](../lessons/m07-05-grouped-timing.md)。两配置B1/D64、B256/D128各三新进程，C1024/交错/gap0，Graph主口径，Eager单列；每臂每政策30个100调用区间，1080个区间平均值。相同地址复用，无缓存清空。

29号限定正确性来源为 `run-20261010-125938-pe9pljcg`；当前源码与保存的已接受grouped哈希必须一致，辅助依赖也校验并保存快照。29号最大B17/C32，因此每个新worker都先用本次C1024输入检查三方初始调用与重置后的Graph replay。启动探针只用于区间外首个调用，计时直接使用已验证参数的内核入口；grouped传BATCH、ROWS_PER_PROGRAM=4、num_warps=4及ceil(B/4)的grid。

## 实际完成的准备检查

实现者执行的[CPU报告](20261010-m07-grouped-timing-cpu.json)状态为 `CPU_ORCHESTRATION_CONTROLS_PASS`，36个负控制被拒绝。范围包括六worker计划、六位置平衡顺序、汇总/原始记录一致性、真实grouped调用表达式的模拟启动、C1024夹具破坏反例、29号来源与helper改动拒绝、旧100-call计时体绑定，以及worker在GPU前失败时保留首错。

独立审查者只读核对真实导入与快照依赖、三方计时入口、grouped参数、实际输入与Graph重置检查、100-call归一化、顺序和失败留档，无阻断发现。主教练复核关键worker、来源门禁、最终汇总参数检查和CPU报告哈希一致；不重复运行已通过的旧正确性套件。未执行GPU/WSL，不把CPU模拟写成GPU通过。

最终文件SHA256：

| 文件 | SHA256 |
| --- | --- |
| 30-compare-kv-grouped.py | f575c7b08d321639386bc1e62544878ca7098e0eec1e29579f5e9cde7ca16e5b |
| 30-compare-kv-grouped.sh | 0e1b014ca1ed78686f3e611b5185ea554a8074a1be2491c3a662ab5278992f70 |
| 30-grouped-timing-cpu-test.py | c952ad1d8afa04227565f66f1126ec9c74ddfdc15bc4d5ecd02752917e9d05ab |

## 教学与范围

本人此前预测B256“暂时不确定，但我认为应该更快”；B1方向没有代填。grouped三条核心表达式是教练示范后本人应用并运行，本次owner记录已明确这点。运行器、Graph组织和记录由教练提供；不推断独立掌握。

运行后检查所有三轮Graph结果，再看代表比值v1/grouped和original/grouped。大于1仅表示对应口径下grouped耗时更短，是否稳定须结合轮次；Eager不替代Graph结论。两个配置不替代全六配置回归，不作模型、AMD或DRAM带宽收益声明。工程57/100保持，未自动同步GitHub。
