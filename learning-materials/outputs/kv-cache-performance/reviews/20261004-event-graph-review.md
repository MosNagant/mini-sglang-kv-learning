# M05：图重放对照用户输出复核与首批矩阵准备

日期：2026-10-04。依据是用户回传控制台与此前17号脚本源码；本轮没有直接读取WSL结果JSON或运行GPU，不冒称完成原始数组独立复核。

用户运行目录：`/home/weltlaw/projects/mini-sglang-kv/experiments/learning/m05-event-graph/run-y8zj4_6o`。

```text
REPLAY_CORRECTNESS_OK: targets, untouched storage and inputs
EAGER_US_PER_CALL: {'median': 8.588320016860962, 'min': 7.765759825706482, 'max': 21.083199977874756}
GRAPH_US_PER_CALL: {'median': 1.051199994981289, 'min': 0.8579199761152267, 'max': 2.014719992876053}
DIAGNOSTIC_COMPLETE: compare execution policies; no final baseline PASS.
```

当时17号默认B1、D64、C1024、FP16、交错缓存、indices=[0]，同进程交替测30对样本。每个样本均为100次操作区间的平均每次时间。图捕获后先重置缓存为-1，再重放检查，不把之前eager的正确缓存内容当成重放证据。

可以确认用户回传该诊断完成，结果受执行方式明显影响。不能把两种方式的时间差归结为精确的Python开销、不能称为优化了算子，也不能用一次对照判定最终稳定值。Graph结果仍包含图执行相关成本，不是单条内核指令的纯计数，更不是DRAM带宽。

## 下一步：6配置×3轮原版试测

- 主口径：CUDA Graph重放；参考口径：eager循环提交。两者分别保存，不混在一起比较原版/Triton。
- B=1/16/256，D=64/128，容量1024，FP16，交错缓存，连续合法唯一indices=0..B-1。
- 为大尺寸提供可区分、精确有限的输入：CPU固定seed=20261004生成0..1023整数K，转换到GPU FP16；V=K+1024。准备不计时。该输入模式与旧B1 arange诊断不同，旧结果只用于方法诊断。
- 20次side-stream预热；捕获100次调用；额外5次图重放预热；每进程30对交替顺序的eager/graph样本。
- 每配置每轮新建进程，共18个；三轮间打乱配置顺序。所有轮次仍会受到机器状态影响，不宣称统计独立或预热已保证稳定。
- 各方法复用同一小输入/缓存。最大KV池有效分配524288字节，最大输入K/V共131072字节；不含快照、图与运行时资源。该小工作集不是显存峰值带宽实验。
- 每例检查完整缓存与只读输入；保存源码、配置、版本、全部样本、控制台与失败；冻结17号运行器后执行。
- 这是原任务大矩阵前的首批试测。仍不包括大batch、分离布局、随机索引、冷缓存、模型接入或Triton验证，不悄悄将M05整项范围缩减为六例。

入口：[18-original-baseline-pilot.py](../18-original-baseline-pilot.py)。17号新增参数但保留旧默认；旧运行的归档源码不会被更新。批量工具为教练提供，不增加用户独立手写成果。

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --cd /home/weltlaw/projects/mini-sglang-kv --exec bash -lc 'source experiments/activate.sh && python /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/18-original-baseline-pilot.py'
```

先记录采集完成与轮间波动，之后再验收基线。新增代码仅本地语法/计划检查和静态审查，未在本轮执行GPU。项目正式验收计分暂维持20/100。
