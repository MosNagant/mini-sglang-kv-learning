"""Generate review documents from saved 45/46 evidence; no GPU execution."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT/'logs/m08-host-diagnostic-results/run-bqojhqoc'
CHECK = ROOT/'logs/m08-host-path-check-results/run-20261015-033200-qh26bon6'
TIMING = ROOT/'logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v'
FAILED = 'run-20261015-033031-7rgjj7fy'

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

host, checked, timing = read(HOST/'summary.json'), read(CHECK/'result.json'), read(TIMING/'result.json')
assert host['status']=='HOST_DIAGNOSTIC_COMPLETE'
assert checked['status']=='HOST_PATH_MODEL_CORRECTNESS_PASS'
assert timing['status']=='HOST_PATH_MODEL_TIMING_COLLECTION_COMPLETE'

cpu_rows=[]
for rnd in host['rounds']:
    for c in rnd['cases']:
        m=c['host_medians_us']
        cpu_rows.append(f"| {rnd['round']} | {c['B']} | {m['old']:.4f} | {m['cached']:.4f} | {c['host_change_pct']:+.3f}% |")
model_rows=[]
for policy in ['eager','decode_graph']:
    for rnd in [1,2,3]:
        workers={w['host_path_arm']:w for w in timing['workers'] if w['policy']==policy and w['round']==rnd}
        original,old,new=[workers[label]['median_seconds']*1000 for label in ['original','checked_triton','cached_triton']]
        model_rows.append(f'| {policy} | {rnd} | {original:.3f} | {old:.3f} | {new:.3f} | {(new/old-1)*100:+.3f}% | {(new/original-1)*100:+.3f}% |')

report='''# M08 调用层优化：局部CPU开销下降，整模型收益未稳定

日期：2026-10-15。结论：**相同GPU内核下，缓存元信息使CPU封装消融耗时下降约60%；真实模型未获得稳定加速，不晋升默认后端。** 这是一次有局部收益、端到端尚未成功的优化，不能写成“模型加速60%”或“已经超过CUDA”。

## 1. 为什么选择这个改动

旧42号真实模型中，Triton Eager明显慢于原版，而decode Graph接近原版。源码40号每次调用都执行五张量元信息检查、四次行布局解析。固定128输入/32生成负载，每次generate有896次KV写入，因此提出可验证假设：重复Python适配工作可能解释一部分退化。

原版也有检查：Python创建cache view，C++ TensorMatcher检查参数。不能说原版完全不验证，更不能人为给原版加重封装来制造胜利。本次原版仍走raw upstream入口。

## 2. 改了什么、如何保持语义

[44号新适配器](../../44-m08-kv-cached-metadata.py)保留40号同一个`m08_store_rows`内核、四warp、一行/program、全部地址和mask规则。

- 每次读取五个当前张量的type、shape、完整stride、layout、dtype、device，形成完整元信息键。
- 未见过的键仍执行40号原验证器；通过才缓存B/C/D/行距和BLOCK_D，最多64项，FIFO淘汰。
- 键变化重新验证；非法输入不入缓存；没有缓存旧tensor对象、数据指针或CUDA stream。每次传当前张量。
- 索引值唯一/合法、只读与写区域不重叠仍是旧入口已有的调用方契约。两版均不在此入口验证GPU索引值；这些约束继续由调用方保证。
- Graph replay仍绕过Python，必须满足固定地址/元信息及每次有效索引的原有契约。

这是调用层缓存优化，不是warp、访存合并、GPU带宽或shared-memory优化。实现和实验由教练完成，个人独立掌握另记。

## 3. 第一层：CPU成本是否真的降低

[45号全部原始样本](../../logs/m08-host-diagnostic-results/run-bqojhqoc/summary.json)与[独立审计](../../logs/m08-host-diagnostic-results/run-bqojhqoc/evidence-audit.json)。

真实CUDA张量，D1024、源行距4096、缓存行距1024、C513；B1和B128。**两臂都仅将GPU启动器替换为空操作**，计时适配路径；GPU不参与这些计时区间。3新进程，每配置/臂30区间×200调用，保存360个区间均值。交替旧/新顺序，提前预热；每格为30区间中位数，单位微秒/次。

| 轮次 | B | 旧封装μs | 新封装μs | 新版耗时变化 |
| --- | ---: | ---: | ---: | ---: |
'''+ '\n'.join(cpu_rows)+'''

六组均下降59.57%–60.83%。这是CPU封装消融的局部证据，既不是GPU内核时间，也不是精确的整模型耗时归因。不能把“约4微秒×896”直接从generate耗时扣掉：真实提交、设备执行、其它工作及CPU/GPU重叠都不在这个空操作测量中。

45号另行运行真实GPU正确性，覆盖旧/新/原版的Eager、改值后Graph replay，以及新tensor对象/新地址的缓存命中。独立审计重建18份Graph与6份替换对象NPZ，逐字节核对全部缓存及只读数据；原始输入/索引和独立scatter期望均保存。18次Eager完整检查是运行时检查，没有冒称保存并重建了18份Eager快照。两配置旧/新PTX及TTGIR一致。

[CPU契约检查](../../reviews/20261015-m08-metadata-cache-cpu-check.json)另覆盖错误dtype/device/layout/stride/shape、空输入、缓存上限及淘汰；该项是假张量/启动器测试，不能冒称GPU测试。

## 4. 第二层：真实模型的结果是否仍正确

[46号严格模型结果](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/result.json)与[独立审计](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/host-path-evidence-audit-v2.json)。

原版和新候选各一个新进程，复用41号原检查，每臂896次KV调用，共1792次逐调用完整backing与只读输入检查。跨臂最终缓存、保存的局部样本、首prefill/首decode/末decode三份logits及32个token精确一致；新原版还与旧41原版作复现对照。没有放宽容差或静默回退原版。

严格全缓存/logits检查是Eager；模型Graph验证仍为全部实际生成token。45号孤立KV的Graph完整检查不能冒充完整模型Graph缓存验收。

## 5. 第三层：真实模型是否更快

[46号全部90次计时](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/result.json)、[冻结方案](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/experiment.json)、[独立审计](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/host-path-evidence-audit-v2.json)。

固定RTX4070Ti、Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca、FP16、TP1、单请求128输入/32生成，FlashInfer/naive缓存/page1/无overlap。三臂：原版CUDA、40号旧Triton、44号新Triton；两模式×三轮，共18新进程、90次完整generate计时。每个结果均检查原始生成token。

复用42号墙钟范围：前后等待GPU，包含调度、采样、D2H、token记录及detokenize；排除模型加载、JIT、预热、输入tokenization和网络。decode Graph仅覆盖decode，prefill仍Eager。每臂两次观察入口预热后，**统一再对实际计时入口预热一次**，避免新工厂的元信息缓存首次建立混入计时。原42文件保留，新派生脚本和精确diff归档。

每格是新进程5次计时中位数，单位ms。变化=(新/对照−1)×100%，正数表示变慢。只使用本次同场参照，不把昨天不同会话的最低时间拼进来。

| 模式 | 轮 | 原版ms | 旧Triton ms | 新Triton ms | 新/旧耗时变化 | 新/原版耗时变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(model_rows)+'''

相对旧Triton，Eager只有第二轮更快，第一轮更慢、第三轮几乎持平；相对原版三轮均慢7.45%–17.28%。Graph对旧/原版均两慢一快。四组比较均未达到事前“各轮至少快3%”的工程筛选门槛；该门槛不是统计显著性或置信区间。本轮按固定计划结束，不重跑不利轮次寻找正结果。

新候选计时前Eager缓存有2种键(B128/B1)，decode Graph有1种键(仅Python-dispatched prefill)。六对最后观察到的预热PTX去除路径/调试信息后相同；45号同一进程两配置PTX/TTGIR也完全相同。46的最后预热特化不等于所有模型Graph节点的逐项编译审计。

## 6. 这次归因到哪一步

| 说法 | 当前判断 |
| --- | --- |
| 每次重复元信息处理可以减少 | 有实现和CPU消融证据 |
| 缓存新对象、新地址时仍按当前数据工作 | 孤立GPU/Graph及真实模型限定范围已验证 |
| GPU内核因这次改动变快 | 没有这样的改动或计时证据 |
| 元信息校验就是旧Eager退化的全部原因 | 不成立为已证实结论；仍缺真实调用层分解 |
| 新版已经让完整模型稳定变快 | 当前三轮不支持 |
| shared bank conflict或occupancy导致退化 | 未测相关计数器，不能归因 |

默认保留原版，新缓存封装保留研究候选。局部优化成功不等于产品性能成功，也不能因为端到端未稳定变快就抹去CPU局部证据。

## 7. 下一步及面试怎么讲

下一项优先把43号中占主要累计GPU内核时长的GEMV与具体线性层、矩阵形状及prefill/decode对应；KV相邻操作是否可融合须据真实调用链再判断。仅按调用次数相似推断LM head还不是完整映射。若继续解释Eager退化，应针对真实Python/Triton提交路径做独立profile，区分适配、JIT dispatch、驱动提交和GPU执行，不把所有差值都指给校验器。

停止条件：没有新的瓶颈证据就不继续扫warp/group；主要计算路径优化应单独命名、冻结正确性条件和对照，不能把模型其它算子的收益记成KV收益。AMD迁移只涉及方法，当前没有AMD实机结果。

三层复习：

1. 基础：shape、stride和tensor数据地址为什么是不同信息？缓存shape是否等于缓存数据？
2. 机制：为什么Graph replay不执行Python验证，却仍要求每次索引合法？为什么CPU少4μs不保证总耗时少4μs？
3. 项目：如何回答“你说封装快了60%，为什么模型没有快60%”？指着45的空操作计时和46的完整计时边界解释。

可据证据表达：我发现真实模型集成存在重复Python元信息工作，保持设备内核不变实现有完整签名保护的有界缓存。局部消融约降60%，全数据与模型检查通过，但三轮端到端对照未稳定获益，因此未替换原版，并转向主要耗时路径调查。具体实现/运行由教练完成，个人项目表达须如实说明自己的参与部分。

## 8. 故障与复现

本次WSL启动复发ERROR_PATH_NOT_FOUND，实际storvsp停止且设备项缺失；运行既有签名校验修复脚本后STORAGE DEVICE OK及WSL_OK，未重装环境。记录位于workspace的outputs/logs/repair-storage-20261015-112007.log；长期复发来源未解决。

46第一次运行在引擎初始化之前失败：设备信息探针提前初始化CUDA，触发Engine的not-initialized断言。失败目录'''+FAILED+'''保留；仅把设备验收移到worker执行之后，使用新目录重新运行。该失败不是KV数据校验失败，也没有删去不利计时样本。

复现入口：[45号](../../45-m08-host-path-diagnostic.sh)、[46号](../../46-m08-host-path-model.sh)。具体命令见[本轮复现说明](REPRODUCE-HOST-PATH-20261015.md)。当前证据已保存到本地共享目录；本次新增资料尚未再次同步GitHub。
'''
(ROOT/'modules/M08/RESULTS-HOST-PATH-20261015.md').write_text(report,encoding='utf-8')

log='''---
title: 开发日志：M08调用层优化与归因
date: 2026-10-15
tags: [KV-Cache, Triton, Profiling, 面试复盘]
---

# M08调用层优化：先找到成本，再验证改动

今天按用户提出的“瓶颈归因→针对性优化→实验验证→面试表达”推进。教练实施一项有界的元信息缓存优化，保留所有失败和负结果；没有继续扫warp/group参数。个人独立解释与验收分数保持原状。

| 节点 | 实际完成 | 结论 |
| --- | --- | --- |
| 历史复盘 | 六条优化路线、1warp映射、证据/假设分类 | 向量化发生不等于已定位加速或慢因 |
| 44实现 | 完整元信息签名、64项FIFO；相同40号GPU内核 | 每次仍核对当前metadata，未缓存旧指针 |
| 45隔离 | 3新进程、360CPU区间均值；独立GPU/Graph/替换对象检查 | CPU封装6.6–6.8μs→2.6–2.7μs，约降60%；不是模型提速 |
| 46正确性 | 两臂1792次逐调用检查，完整缓存/样本/logits/token对照 | 限定真实模型检查通过 |
| 46计时 | 18新进程、90生成样本，Graph/Eager分别比较 | 对旧版Eager一快一慢一近似持平；未稳定超过原版 |
| 决定 | 原版默认保持，候选及完整证据保留 | 优先调查主要计算路径及真实提交成本 |

新封装对旧版Eager三轮耗时变化为+0.976%、−5.740%、+0.019%；对原版为+7.448%、+12.152%、+17.276%。Graph对原版为+4.098%、+2.026%、−1.933%。不能用一轮最好成绩替代三轮。

先读[本轮完整结果与数据](../modules/M08/RESULTS-HOST-PATH-20261015.md)，再读[整个项目性能诊断与面试拆解](../modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md)。[复现说明](../modules/M08/REPRODUCE-HOST-PATH-20261015.md)保留可复制命令，已有实验不需重复执行。

今天的复习重点是“计时边界决定数字的含义”。CPU空操作消融、GPU Event与完整generate墙钟回答不同问题。约60%的局部收益不能换成约60%的模型收益；KV占GPU累计内核时长0.782%也不能直接当墙钟关键路径占比。

故障另记：WSL既有存储设备恢复成功；46首次探针提前初始化CUDA导致引擎断言，保留失败目录后修正调用顺序并新建run。没有改动旧40/41/42或历史基线，没有重装依赖，没有新的AMD/occupancy/带宽计数器结果。

所有实现、运行、审计和报告由教练按本次授权完成；用户提出了以归因推动优化的方法。工程证据新增，不自动记为个人已掌握或提高73/100混合验收分。本次新增材料本地保存，尚未再次同步GitHub。
'''
(ROOT/'obsidian/2026-10-15-开发日志-M08调用层优化与归因.md').write_text(log,encoding='utf-8')

prefix='wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/'
shared='/mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/logs/'
flags=' --accepted-check '+shared+'m08-model-check-results/run-20261014-144754-mmdz752v --accepted-timing '+shared+'m08-model-timing-results/run-o_ltqzpd --candidate-sha256 '+read(CHECK/'experiment.json')['candidate_sha256']
repro='''# 44–46调用层优化：复现入口

2026-10-15。使用既有WSL项目/venv和已下载模型，不安装或升级包。下面每个代码块是一条PowerShell命令；已有结果可以直接复习，不需要重复测量。源码或模型变化会被哈希门槛拒绝，不能删掉门槛强行对比。

45号CPU消融及独立GPU正确性：

```powershell
'''+prefix+'45-m08-host-path-diagnostic.sh --accepted-timing '+shared+'m08-model-timing-results/run-o_ltqzpd'+'''
```

46号真实模型严格正确性：

```powershell
'''+prefix+'46-m08-host-path-model.sh --mode check'+flags+'''
```

46号三臂计时（下面使用本次已通过检查；若重新检查，应替换`--accepted-new-check`为新通过目录）：

```powershell
'''+prefix+'46-m08-host-path-model.sh --mode timing'+flags+' --accepted-new-check '+shared+'m08-host-path-check-results/'+CHECK.name+'''
```

运行器均保存源码、计划、原始结果、每进程日志、失败原因和共享副本。45为3新进程的CPU空启动消融，46检查2新进程、计时18新进程90样本。源码、计时范围、数值契约与所有比较见[完整结果](RESULTS-HOST-PATH-20261015.md)。

CPU离线审计脚本：[45审计](../../reviews/20261015-m08-host-diagnostic-audit.py)、[46审计](../../reviews/20261015-m08-host-model-audit.py)。审计只读归档，不重跑GPU，不将运行时报告扩大为未保存的快照证据。
'''
(ROOT/'modules/M08/REPRODUCE-HOST-PATH-20261015.md').write_text(repro,encoding='utf-8')
print('WROTE 3 standalone records from saved results')
