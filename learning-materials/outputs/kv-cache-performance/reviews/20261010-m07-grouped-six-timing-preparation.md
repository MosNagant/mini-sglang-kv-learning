# 32号六配置grouped效率测试：准备审查

日期：2026-10-10。结论：新入口完成CPU检查与独立审查，可以交用户运行。**本轮GPU计时尚未执行，无新速度结论。**

31号完整正确性已经[离线核验通过](20261010-m07-grouped-full-review.md)。32号按[事前实验卡](../lessons/m07-07-grouped-six-case-timing.md)测B1/16/256×D64/128，每配置三个新进程；原版CUDA、v1、grouped三方，Graph主口径，Eager单列。计划18个worker、3240个区间平均样本，每区间100次调用。保持当前内核，保留旧30号探索记录。

## 实现和实际检查

新增32号Python/Shell入口与 `32-grouped-six-timing-cpu-test.py`；未改26～31号脚本、已接受内核或历史运行数据。

GPU worker逐字功能复用30号，经AST比较完全相同；新模块独立定义六配置校验、输入夹具、计划、完整18进程汇总和31号证据门禁，没有改旧模块全局状态。导入的辅助文件与本轮源码进入快照。31号门禁重新核对196结果、98配对、36拒绝、实际启动、完整保存数组和文件身份；计时现场再核对实际源码及固定上游。

每个worker仍在测量外检查实际C1024输入、每臂初次调用及重置缓存后的Graph重放；每个区间后检查结果，合计190次运行时检查，保留七份调用后快照。公共索引值检查与启动探针不进入计时。32号新增B16等输入的受理、grouped实际grid与参数已通过CPU控制检查，实际GPU工作仍须由本次运行验证。

[实现者CPU报告](20261010-m07-grouped-six-timing-cpu.json)为 `CPU_SIX_CASE_CONTROLS_PASS`，包含72项负控制：缺失/重复worker、原始样本与顺序不符、错误grouped参数、非有限或非正耗时、六配置数据损坏、来源变更、36拒绝记录异常以及GPU导入前失败留档。真实31保存证据可通过新门禁。报告中的synthetic_summary是CPU合成数据，不能用作GPU成绩。

独立审查者另行核对worker AST、六配置受理、18唯一worker计划、平衡顺序、已知数值的汇总、实际31门禁、快照/导入依赖与首错保存，无阻断发现。最后仅增加Graph汇总中三方代表微秒数的显示；审查者确认除此之外源码未变，无需再重复测试。主教练核对最终哈希与CPU报告一致。所有本地检查均未运行GPU/WSL。

## 最终文件身份

| 文件 | SHA256 |
| --- | --- |
| 32-compare-kv-grouped-full.py | 75b55e6bcd966904ea000f854aa0e0822f6fae2a8adee6133c68794e45e50a97 |
| 32-compare-kv-grouped-full.sh | c60b498630fa45c96f0f57c39fbd5ead5089c8477225b98cf7c17c7ffabc5dda |
| 32-grouped-six-timing-cpu-test.py | 6c10145d7b16edc1b2daafefff49f9f906dac6746d593f3186dc3bc52f850d2c |

## 接续

用户在既有PowerShell执行32号Shell入口。输出18次RUN与逐轮成绩，最后六行GRAPH_SUMMARY给出三方代表微秒数、v1/grouped与original/grouped；结果保存在 `experiments/learning/m07-grouped-six-timing`，共享到 `logs/m07-grouped-six-timing-results`。

运行后审查所有三轮、所有配置和原始数据，再决定收益/退化。Eager与Graph分列；重复地址、交错缓存、gap0、FP16六配置微基准的结论不扩展到所有布局、DRAM带宽、模型或AMD。当前工程57/100保持，M07-05待实际采集验收，个人解释另记。未同步GitHub。
