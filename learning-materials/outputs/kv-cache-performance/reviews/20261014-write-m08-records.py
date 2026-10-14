"""Build dated M08 documentation from saved results; no GPU or network calls."""
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parents[1]
MOD = BASE / 'modules/M08'
TIMING = 'logs/m08-model-timing-results/run-o_ltqzpd'
CHECK = 'logs/m08-model-check-results/run-20261014-144754-mmdz752v'
SMOKE = 'logs/m08-model-smoke-results/run-20261014-143514-x0uhkeqq'
PROFILE = 'logs/m08-model-profile-results/run-uml9cxd9'

def read(name):
    return json.loads((BASE / name).read_text(encoding='utf-8'))

timing = read(TIMING + '/summary.json')
profile = read(PROFILE + '/result.json')
assert timing['status'] == 'MODEL_TIMING_COLLECTION_COMPLETE'
assert profile['status'] == 'REAL_MODEL_KV_SHARE_RECORDED'
assert profile['kv_count'] == 896

rows = []
representatives = []
for policy in ('decode_graph', 'eager'):
    pairs = timing['comparisons'][policy]['pairs']
    assert len(pairs) == 3
    for pair in pairs:
        rows.append(f"| {policy} | {pair['round']} | {pair['original_ms']:.6f} | {pair['triton_ms']:.6f} | {pair['candidate_time_change_pct']:+.3f}% |")
    a, b = [statistics.median(p[k] for p in pairs) for k in ('original_ms', 'triton_ms')]
    representatives.append(f'| {policy} | {a:.6f} | {b:.6f} | {(b/a-1)*100:+.3f}% | {32000/a:.3f} / {32000/b:.3f} |')

def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + '\n', encoding='utf-8')

write(MOD / 'RESULTS.md', f'''# M08 首轮真实模型结果：接入通过，尚无稳定提速

日期：2026-10-14。本轮完成真实模型接入、精确正确性检查、固定工作负载的原版/Triton对照与独立性能跟踪。**现有数据不支持“我们的算子明显优于原版CUDA”。默认仍保留原版，不晋升研究候选。**

## 1. 本轮比较的具体对象

- 原版：mini-sglang 提交 `9a91cfafe754aa85daee49998176275667eb58f2` 中的 `store_cache` 原始路径。
- 研究版：[40号适配器](../../40-m08-kv-backends.py)中的 `m08_store_rows`，一行/program、4 warp、列mask、四个独立源/目标行距；由教练基于既有行搬运内核实现真实模型适配。它不是冻结M07的grouped/full_groups版本，不能把结果说成那些候选已在模型中提速。
- 真实接入：在模型初始化/Graph捕获前显式替换 `minisgl.kernel.store_cache`；不支持的元信息报错，没有静默回退原版或隐含连续化复制。
- 测量原版使用原始入口，Triton使用 `observe=False` 适配入口；候选必要的Python元信息校验仍属于集成成本。跟踪计数器只用于初始化/预热，计时前移除。
- 设备：RTX 4070 Ti；Torch 2.9.1+cu128、Triton 3.5.1、CUDA 12.8、FlashInfer 0.5.3。没有AMD结果。
- 模型：Qwen3-0.6B真实权重，revision `c1899de289a04d12100db370d81485cdf75e47ca`，加载精度FP16、TP1。权重SHA256：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

这是两个具体实现及其调用路径的比较，不是对 CUDA 与 Triton 两种技术的普遍排名。

## 2. 从教学KV进入真实模型后，哪里变了

| 项目 | 本次真实输入 | 意义 |
| --- | --- | --- |
| 请求数 | 1 | 不等于每次KV写入的B |
| prefill输入 | K/V `(128,1024)` | B=128个输入token行 |
| decode输入 | K/V `(1,1024)` | 之后31次前向各新增1行 |
| D | 1024 = 8个KV头 × 每头128元素 | 每个token的一条K或V展开宽度；不可把D定义成hidden_size |
| K/V输入行距 | 4096元素 | 来自合并QKV投影，Q2048+K1024+V1024；相邻K行之间包含其它有效数据，并非全是padding |
| 单层K/V缓存 | `(513,8,128)`，stride `(1024,128,1)` | 512可用槽+1 dummy槽，展平后的目标行距1024 |
| 全缓存 | `(2,28,513,1,8,128)`，58,834,944字节 | 按K/V及层分区保存，不是教学样例的逐行KV交错 |

地址逻辑保持：`src = row * 4096 + col`，`dst = indices[row] * 1024 + col`。这些是元素偏移，FP16类型化指针寻址不再手乘2。

32个生成token来自1次prefill前向加31次decode前向，每次经过28层，所以一次生成的KV写入调用数是 `28 × (1+31) = 896`。最后一个采样token结束请求后，不再多做一次decode前向。39号最初冒烟使用较小的5输入/8输出、257槽；不要与正式对照的128/32、513槽混在一起。

## 3. 已验证的正确性

[41号原始结果](../../{CHECK}/result.json)及[独立审计](../../{CHECK}/evidence-audit.json)：

- 三个新进程：原版重复自检、原版、Triton。每臂896次调用，共2688次；逐调用在运行时核对全KV backing、未写区域、只读输入/索引。
- 跨进程比较最终全缓存、保存的layer0首prefill/首decode样本、首prefill和首/末decode三份logits、32个原始生成token，均精确一致；没有事后放宽容差。
- Triton路径真实执行 `m08_store_rows`，记录896次实际启动；未调用原版fallback。
- 独立CPU审计核对106项来源哈希、15项产物哈希、两对合计44项数组比较（每对22项）逐字节相同，另重建6份保存的局部样本。896次运行时检查不是896份全缓存快照都已离线重建。

41号严格数据诊断为 **Eager**。42号对Graph真实重放检查所有生成token；尚未做Graph模式最终全缓存/中间logits的同等级逐位复核。因此只接受本次实际覆盖，不宣称框架全部Graph配置正确。

## 4. 计时方法与全部轮次

[事前方案](../../reviews/20261014-m08-measurement-plan.md)、[实际冻结计划](../../{TIMING}/plan.json)、[60次原始样本与汇总](../../{TIMING}/summary.json)。

冻结为1请求、128输入token、32生成步、greedy、ignore_eos、naive prefix cache、page1、512可用槽、无overlap。同一GPU串行执行：2后端 × 2执行方式 × 3轮 = 12新进程；每进程2次预热、5次计时，共60次计时。每次实际生成32个token，均与参考完全一致。

每次用 `perf_counter` 测完整 `LLM.generate`，前后等待GPU完成。包含调度、采样、结果D2H、原始token记录和detokenize；不包含模型加载、JIT、预热、输入tokenization、网络。它是固定离线调用的墙钟时间，不是纯KV内核时间，也不是在线服务时延。

`decode_graph`只对decode使用尺寸1的CUDA Graph，prefill仍Eager；`eager`全程普通提交。两类分别比较。Graph本身的收益不能记成自写KV内核的收益。

每格是该进程5次计时的中位数，单位毫秒；变化=`Triton/原版 - 1`，**正数代表变慢**。

| 执行方式 | 轮次 | 原版ms | Triton ms | Triton耗时变化 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

三轮中位数的中位数仅作代表值，不能替代上面的逐轮判断：

| 执行方式 | 原版代表ms | Triton代表ms | 代表耗时变化 | 输出tokens/s：原版 / Triton |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(representatives)}

主口径Graph两轮慢、一轮快0.836%，没有达到事前“各轮至少下降3%”的工程筛选条件。Eager三轮均慢12.803%–23.022%。3%是工程筛选门槛，不是置信区间；三轮也不足以声称统计显著性。候选的Python封装成本可能影响Eager，但本轮未完成独立因果分解，不能把全部差值直接归因于某一行检查代码。[42号独立复算](../../{TIMING}/evidence-audit.json)核对12进程、60次样本、1920个实际输出token、全部中位数及速度比；没有重新测量性能。

按预先停止规则保留全部结果，本轮不换负载或只重跑不利轮次来寻找正结果。

## 5. 独立跟踪：KV在模型里占多少

[43号完整结果](../../{PROFILE}/result.json)及[原始Nsight报告](../../{PROFILE}/original-model.nsys-rep)来自独立插桩运行，未混入上述60个时间样本。

原版、同一128/32负载、decode Graph、完成两次预热后才开始捕获，实际记录896个 `store_kv_cache` GPU内核：

- KV内核持续时间累计：`812245 ns = 0.812245 ms`。
- 全部GPU内核持续时间累计：`103927254 ns = 103.927254 ms`。
- 二者之比：`0.781551%`。

这是**GPU内核持续时间之和的份额**，不是整个generate墙钟的关键路径占比。Graph节点追踪、插桩开销及WSL时间戳设置都会影响精度；没有用它计算或宣称整机提速上限。它提示单独缩短KV搬运可影响的范围较小，值得先调查相邻操作/提交成本，再决定是否融合。

[SQLite独立审计](../../{PROFILE}/audit.json)重算11,976个内核、26种名称，并确认896次KV由28次prefill及868次decode图节点组成；KV两阶段累计分别0.035969ms、0.776276ms。这里没有把所有图外内核都当作prefill。

| 跟踪中的主要kernel（名称缩写） | 次数 | 累计ms | 占全部kernel累计时长 |
| --- | ---: | ---: | ---: |
| gemvx `(int)6` 特化 | 3472 | 64.975756 | 62.520420% |
| gemvx `(int)7` 特化 | 32 | 21.642346 | 20.824514% |
| BatchDecodeWithPagedKVCache | 868 | 4.634928 | 4.459781% |
| FusedAddRMSNorm | 1792 | 2.588559 | 2.490741% |
| QKRMSNorm | 1792 | 2.147839 | 2.066675% |

两类GEMV合计83.345%的累计时间值得优先核对其具体调用和矩阵形状；尚未建立每个模板与模型某层/LM head的对应关系，不能只凭名字归因。完整名称、计数和范围见[43号审查](../../reviews/20261014-m08-model-profile-review.md)。

本轮未采独立prefill/decode Event分段指标、TTFT/ITL、候选Nsight对照、带宽/occupancy计数器。事前方案中的这些扩展尚未完成；不能从总和冒认已获得分阶段瓶颈或因果解释。

## 6. 决定、贡献与下一步

1. 默认原版CUDA保持，40号研究后端保留用于可复现对照；M07历史选择不改。
2. M08-01～05在本次限定范围完成工程交付。M08-06个人独立解释仍待复习，不以教练写出答案代替本人掌握。
3. 下一项先将主要GEMV与KV相邻操作映射到真实调用/形状，再决定优化对象。若继续KV主线，审查RoPE/相邻写入融合能省什么、数值和布局契约、Graph下是否仍有收益；若目标是明显模型提速，需另立主要矩阵计算路径的优化范围。这些均是新假设，尚未实施，不承诺提速；不能把整个模型其它优化冒称KV算子加速。
4. 如要改变默认或扩大结论，仍需新版本、同条件正确性和完整对照。多请求、padding/dummy冲突、page_size>1、TP>1、多模型及AMD均未验收。

本轮环境、适配器、诊断/计时与文档由教练实施。学员前期地址、mask、grid和优化实验参与记录保持；未新增本人独立实现或解释的证据。
''')

write(MOD / 'README.md', f'''# M08：真实模型接入与同条件对照

2026-10-14：**真实模型接入和固定负载对照已完成，尚未证明稳定性能优势。默认原版CUDA保持。** M08-01～05工程9/10分，M08-06本人独立解释1分待复习；工程完成与掌握分开。新增实现、执行和报告由教练完成。

先读[完整结果](RESULTS.md)，再读[今日独立开发日志](../../obsidian/2026-10-14-开发日志-M08真实模型接入.md)。[复现与资料入口](REPRODUCE.md)记录文件和单行命令；已有结果无需重跑。

| 节点 | 实际产物 | 本轮结论 |
| --- | --- | --- |
| 38 环境/真实权重 | [安装回执](../../logs/m08-setup-results/dependency-install-ir_ipwxe/result.json)、[权重来源与哈希](../../logs/m08-setup-results/model-0d7ke3_g/result.json) | 新增11个缺失依赖，既有包保持；固定Qwen3-0.6B revision真实权重 |
| 39 原版模型 | [输出](../../{SMOKE}/result.json)、[KV trace](../../{SMOKE}/kv-trace.jsonl)、[审计](../../{SMOKE}/evidence-audit.json) | 5输入/8生成；224次真实KV调用，原版运行成功 |
| 40/41 候选接入 | [源码](../../40-m08-kv-backends.py)、[检查结果](../../{CHECK}/result.json)、[审计](../../{CHECK}/evidence-audit.json) | 真实D1024、源行距4096、缓存行距1024；三臂共2688调用检查通过 |
| 42 同场对照 | [全部60次样本](../../{TIMING}/summary.json)、[冻结计划](../../{TIMING}/plan.json) | 12新进程；Graph无稳定收益，Eager三轮均慢 |
| 43 实际份额 | [结果](../../{PROFILE}/result.json)、[原始跟踪](../../{PROFILE}/original-model.nsys-rep) | 896次KV内核累计0.812ms，占GPU内核累计时长0.782%，不等于墙钟占比 |

主工作负载：RTX4070Ti、Qwen3-0.6B、FP16、TP1、1请求、128输入/32生成、FlashInfer、naive prefix cache、无overlap；Graph仅decode尺寸1。39号更小的首次运行单独记。41号为Eager严格数据诊断，42号Graph复验的是实际输出token，未完成同等级Graph全缓存快照复核。无模型质量、全框架、高并发或AMD提速声明。

下一项：把主要GEMV与KV相邻操作对应到实际调用/形状，再选择优化对象；RoPE/写入融合仅是待评估方案。M05/M06/M07历史证据与失败记录保留。
''')

write(MOD / 'REPRODUCE.md', f'''# M08：已有资料与复现入口

以下为本机已验证环境的复现命令。正常复习先读保存结果，无需再次运行。所有脚本新建run目录，失败证据保留；不把未执行的新命令记成结果。

## 环境

- WSL：CUDA-Ubuntu-24.04 / weltlaw；项目 `/home/weltlaw/projects/mini-sglang-kv`，沿用 `.venv` 和 `experiments/activate.sh`。
- 上游固定提交 `9a91cfafe754aa85daee49998176275667eb58f2`。39/41/42/43的模型run各保留实际102份源码副本/哈希；38的环境记录另记，不依赖日后工作区未记录的修改。
- [依赖安装记录](../../logs/m08-setup-results/dependency-install-ir_ipwxe/result.json)固定已有包后只补11个缺失包；没有升级Torch/Triton。
- 真实模型位于 `/home/weltlaw/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca`。1.50GB权重仅保存在本机，资料包保存revision、下载和哈希清单，不包含权重。不要用dummy权重替代。
- 38号是环境/下载准备脚本，已完成后不必重复安装下载。离线推理进程保持HF/Transformers离线；只在第一次真实下载时单独联网。

## 阅读顺序

1. [事前测量方案](../../reviews/20261014-m08-measurement-plan.md)：为什么固定模型/负载、分开Graph/Eager和诊断/计时。
2. [40号实际适配](../../40-m08-kv-backends.py)：输入D、四个行距、mask及显式后端。
3. [41号精确检查](../../{CHECK}/result.json)与[独立审查](../../reviews/20261014-m08-model-check-evidence-review.md)。
4. [42号样本](../../{TIMING}/summary.json)：每进程5次、每方式3轮，两层统计分别看。
5. [43号trace](../../{PROFILE}/result.json)：GPU kernel时长份额与整个generate墙钟区别。
6. [结论](RESULTS.md)：记录负结果、范围和下一项假设。

## 单行PowerShell命令

先确认WSL能启动；本轮已启动成功，不无故重复修复系统。

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /bin/echo WSL_OK
```

精确正确性复现（创建新run、会运行模型；不计时）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/41-m08-model-check.sh --model-path /home/weltlaw/models/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca --model-revision c1899de289a04d12100db370d81485cdf75e47ca
```

固定负载同场计时（12新进程，GPU串行；不可与其它GPU测试同时执行）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/42-m08-model-timing.sh --accepted-check /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/{CHECK}
```

独立Nsight跟踪（不是正式时间样本）：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --cd /home/weltlaw/projects/mini-sglang-kv --exec bash -lc 'source experiments/activate.sh && python -B -u /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/43-m08-model-profile.py --timing-run /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/{TIMING}'
```

其它设备需要恢复记录的依赖、项目、模型revision及路径，不能保证直接复用本机绝对路径。没有声称做过另一台设备的复现。

## 留下的故障记录

43号首个run `run-rv7o7fmk`：诊断worker命名 `profile.py`，遮蔽Python标准库同名模块，使Transformers间接导入失败。更名 `m08_profile_worker.py` 后新run成功。旧源码/错误仍在[失败记录](../../logs/m08-model-profile-results/run-rv7o7fmk/result.json)；未改模型、放宽比较或隐藏失败。
''')

write(BASE / 'obsidian/2026-10-14-开发日志-M08真实模型接入.md', f'''---
title: 2026-10-14 开发日志 — M08真实模型接入
date: 2026-10-14
tags: [Triton, KV-Cache, M08, 性能分析, 真实模型]
---

# 从KV小实验走到真实生成

今天的目标来自本人要求：“开始M08，要让结果能证明明显优势。”我们实际完成了真实接入与对照；**测量尚未证明明显优势，结论按结果保留。** 不将想达到的目标写成已有成绩。

## 今天各节点做了什么

| 节点 | 问题与操作 | 实际结果 |
| --- | --- | --- |
| 38 环境 | 保留既有Torch/Triton，补缺失推理依赖，固定真实权重revision | 11个新增依赖；真实Qwen3-0.6B约1.50GB权重下载并核对哈希 |
| 39 首次模型 | 原版能否实际生成，真实KV是否同教学样例 | 5输入/8输出正常；224次KV调用；D1024、源行距4096、缓存行距1024 |
| 40 适配 | 如何支持实际三维层缓存与独立K/V行距 | 显式原版/Triton后端，无静默fallback；一行/program、4warp、列mask |
| 41 正确性 | 新路径是否只搬运正确数据、模型输出是否相同 | 3新进程各896次KV检查，最终缓存/32token/3份logits精确一致；Eager范围 |
| 42 性能 | 相同128输入/32生成，原版与Triton谁更快 | 12新进程/60次时间；Graph无稳定优势，Eager三轮均退化 |
| 43 跟踪 | 真实模型中KV写入是否占主要GPU时间 | 896次KV共0.812ms，占GPU内核累计时间0.782%；不能当墙钟关键路径占比 |

完整表、范围、原始数据链接在[M08结果报告](../modules/M08/RESULTS.md)，复习无需重跑全部测试。所有结果和失败run保留。今天没有改M05基线或M07旧结论。

## 本次最值得复习的四点

1. **B随阶段改变，D来自模型结构。** 单请求prefill有128行，decode通常1行；D=8个KV头×128=1024。本次并非把教学B/D随意改大来制造收益。
2. **行距不等于宽度，间隔也不全是空白。** QKV合并投影宽4096，K仅占1024；跨到下一行K仍走4096。缓存K/V分区，目标行距1024。
3. **局部正确不自动等于模型快。** 41先证明搬运、输出一致，42才测完整generate。原版与Triton都用相同Graph设置；Graph对Eager的提升不能据为自己的KV优化。
4. **性能要看所有轮次和份额。** Graph候选两轮慢、一轮快不到1%；Eager三轮全慢。0.782%只是kernel累计时长份额，不能直接代入Amdahl声称上限或整机收益。

## 当前工程进度与个人参与

- M08-01～05限定工程交付完成，9/10；M08-06本人独立解释待记录。全项目混合验收73/100；不是个人知识掌握73%，也不是任意场景软件已完成73%。
- 本轮环境、适配、运行器、实际执行、审计和文档由教练完成；本人提出目标，之前已参与的地址/mask/grid/warp/grouped学习记录不变。
- 模型正确性详细诊断只做Eager；Graph检查了实际输出token，没有同等级的最终全缓存/全步logits复核。多请求、模型质量、AMD、完整融合和其它框架功能未验收。
- 43首轮worker名称遮蔽标准库，留存错误后更名解决；未升级依赖或改模型掩盖问题。
- 本人实际学习时长未记录。工程运行耗时不能代替本人投入时长。

## 下一次第一步与面试追问

下一次先把profile中主要GEMV（合计83.345%累计kernel时长）与KV相邻操作对应到模型源码/形状，再选择优化对象；RoPE/相邻写入融合只是待评估方向。需要明确区分KV项目内改进与扩大到矩阵计算优化。尚未实现或宣称收益，默认原版继续使用。

复习时可以围绕这些实际材料提问：

- 为什么单请求的B有时128、有时1？D1024怎么算？
- 源行距4096为什么不能用 `2*WIDTH`，中间3072是否全是gap？
- 32生成token为什么只有31次decode前向？896次store怎么得到？
- 为什么先跑original_repeat？为什么KV复制检查不放宽容差？
- 60个样本为何不等于60个独立进程？哪层取中位数？
- 为什么Graph赢Eager不能证明自写算子赢原版？
- 0.782%能直接当整机份额吗？下一项优化为何考虑相邻融合？
- 哪些实现由本人完成、哪些由教练完成？怎样用源码和原始结果证明？

这些是待复习问题，未收到本人回答，不能自动标记已掌握。
''')

print(json.dumps({'written': ['modules/M08/README.md', 'modules/M08/RESULTS.md', 'modules/M08/REPRODUCE.md', 'obsidian/2026-10-14-开发日志-M08真实模型接入.md'], 'gpu_run': False}, ensure_ascii=False))
