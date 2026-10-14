# M08 最小模型 A/B 测量方案（2026-10-14，执行前方案）

**先运行原版、确认真实 KV 合同，再接入一个显式研究版。** M07 默认仍是原版；37号选择器在新模型上下文可能拒绝，不能改其冻结策略来迁就模型。若两臂最终都走原版，必须记为 A/A，不能宣称完成 Triton 模型 A/B。本文只读固定源码并提出方案，没有运行模型、安装依赖、修改核心或产生新的性能证据。

## 1. 已读接口与必须先确认的差异

依据当前导出 `outputs/github-upload/repository/source/`；实际运行还须保存 live 源码哈希及工作区差异，不能仅凭旧 commit 相同断言文件相同。

| 源码入口 | 与实验相关的事实 |
| --- | --- |
| [LLM.generate](../../github-upload/repository/source/python/minisgl/llm/llm.py) | 接受固定 token ID 列表与 SamplingParams；返回 token_ids/text；默认 dtype 是 BF16。构造 Engine 前 CUDA 不能已初始化，全局 Context 只能设置一次，因此不同版本/Graph 政策用独立进程。 |
| [Engine.forward_batch](../../github-upload/repository/source/python/minisgl/engine/engine.py) | 模型前向后还包含 sampling、异步 D2H 与 event。模型 GPU 时间、完整 forward 时间与 generate 墙钟时间不能混称。 |
| [GraphRunner](../../github-upload/repository/source/python/minisgl/engine/graph.py) | 只对 decode 使用 Graph；prefill 仍为 Eager。图在 Engine 初始化时捕获，必须在构造模型之前选择 KV 入口，事后替换 Python 函数不会改变已捕获图。 |
| [AttentionLayer](../../github-upload/repository/source/python/minisgl/layers/attention.py) / [RoPE](../../github-upload/repository/source/python/minisgl/layers/rotary.py) | K/V 来自 QKV split；RoPE 原地修改 Q/K并返回原视图。源行距可能是完整 QKV 行宽，不是 D 或 D+64。必须记录实际 k/v 的 shape/stride/storage_offset。 |
| [MHAKVCache](../../github-upload/repository/source/python/minisgl/kvcache/mha_pool.py) | 同一大分配的 K/V 分区，层缓存先以三维视图传给 store_cache；D 应为 local_kv_heads×head_dim。不是教学交错布局，也不等同 hidden_size。 |
| [FlashInferBackend](../../github-upload/repository/source/python/minisgl/attention/fi.py) | 每层先 store_kv 再执行 attention。图重放不重新调用 Python KV hook，不能用 hook 次数冒充重放内核次数。 |
| [Scheduler](../../github-upload/repository/source/python/minisgl/scheduler/scheduler.py) / [NaivePrefixCache](../../github-upload/repository/source/python/minisgl/kvcache/naive_cache.py) | 默认 radix 会缓存 prefix；重复相同 prompt 可能不再执行相同 prefill 工作。现有 cache_type='naive' 可关闭跨请求前缀命中，保留本请求 decode 所需 KV。 |
| [现有 offline benchmark](../../github-upload/repository/source/benchmark/offline/bench.py) | 默认256请求、随机长度与采样、page_size256、整体 time.time；不直接作为本轮最小实验，也不能输出其未记录的 TTFT/ITL。 |

真实模型 D、dtype、源行距、缓存容量可能超出 M06/M07 固定合同。须用本次真实样本另做适配/正确性验收；不能静默 contiguous()，也不能直接给旧 helper 放宽全部检查。若确需复制或转换，保留同样原版接口语义并把其成本计入候选路径。

Graph capture 使用 dummy 输入/目的槽，padding 也可能引入重复 dummy 槽；当前“所有 indices 唯一”的公开检查不能原样塞进捕获区。首轮只用真实请求数1、图尺寸[1]，避开多 dummy 重复的额外范围；capture 样本与真实运行样本仍分别标记。未来扩到 padding 时，必须单独解决和验证真实行/dummy 行的写入语义，不能默认重复写安全。

## 2. 首轮只选这一份小工作负载

| 项目 | 冻结建议 |
| --- | --- |
| 模型/设备 | 本地已备好的 Qwen3-0.6B 真实权重；单卡 RTX4070Ti、TP=1；不用 dummy weight |
| 请求 | 1个请求，固定128个输入 token，max_tokens=32，greedy：temperature=0、top_p=1、ignore_eos=True |
| 输入来源 | 用本地 tokenizer 将固定短文本重复后编码，取前128个 token；在运行前保存文本、实际 token_ids、tokenizer 版本与输入 SHA。计时直接传保存的 ID，避免不同 tokenization |
| 精度 | 显式 torch.float16；本轮不同时引入 BF16 |
| 调度/缓存 | cache_type='naive'、page_size=1、max_running_req=1、max_extend_tokens=256、max_seq_len_override=256、num_page_override=512；调度 overlap 环境值固定并记录 |
| 两种政策 | decode Graph：cuda_graph_bs=[1]；全 Eager：cuda_graph_bs=[]。两者各自 A/B，不能 Graph 对 Eager；前者的 prefill 仍是 Eager |
| 资源范围 | 分配512个真实页，Engine 另加1个 dummy 页；实际缓存容量须记录，不能误写为512行或沿用 M07 的 C1024 |

先只跑**原版一遍**得到输出与真实 shape trace，失败即保存首个错误。原版成功后，使用同一份输入完成研究版正确性；此后才进入下方三轮计时。不得因看到速度不理想临时把请求数、输出长度或 prompt 改成有利配置。此工作负载只证明小模型单请求接入，不能代表高并发 serving。

运行清单至少保存：模型 snapshot revision（无法查到则如实注明）、config/tokenizer/权重文件 SHA 或完整权重清单、源码 commit+dirty diff+哈希、GPU/driver/Torch/CUDA/Triton/FlashInfer/Transformers 版本、本地模型路径、backend、dtype、KV 页数/峰值显存、命令与环境。离线环境变量只给本次独立进程，不改永久激活配置。

## 3. 真实 KV 信息与局部正确性门槛

诊断运行记录每个阶段/层的：phase、真实请求数、padded_size、实际输入 token 行数 B、展开 D、K/V/cache 的完整 shape/stride、dtype/device、基址关系/storage_offset、indices dtype/shape/范围、捕获或真实执行标记。prefill B 是未缓存的输入 token 总数，decode B 通常对应参与该步的行数；不要只写“batch=请求数”。

诊断中的 KV 复制必须精确一致：在写入前保留输入与索引，维护独立参考缓存，按输入行→目标槽映射更新；核对目标全元素、未命中位置、只读 K/V/indices。研究版必须真正进入已冻结的 Triton 内核，保存入口源码与实际特化/launch 或对应 profiler 名称；没有进入的调用计为显式 fallback，不能算候选完成。

为控制归档量，可在**诊断模式**维护完整参考缓存，在 prefill 完成和最终 decode 完成时保存全层有效缓存及只读输入/索引证据；另保存首个 prefill、首个/末个 decode 的真实 KV 数据样本与期望。报告应明确实际保存多少快照，不能把运行时逐调用检查数当成逐次已离线复验的数组数。缓存未初始化区域须以调用前状态为参考或在诊断中统一填 sentinel，不能对未定义的初值直接作跨进程 equal。

正式性能运行不保留这些同步、CPU 拷贝、全量比较或逐层打印。Graph 必须验证重置/新请求后的真实 replay 输出；不能只检查 capture warmup 的值。若不能在不改变 Graph 语义的情况下逐调用验证，则以同合同离线复制检查+实际 replay 最终缓存/输出检查分别留证，准确写明覆盖。

## 4. 模型输出正确性：先看 token，再查 logits

同一政策下原版与候选使用相同真实权重、输入 ID、greedy 参数与生成步数，要求返回的全部 token ID 顺序精确一致，文本只作展示。每臂在本次三轮中都保留输出；先做一份原版重复作为确定性自检。KV 只做复制，输出差异不应靠宽松阈值直接“验收”。

在诊断模式于 sampler 前保存首个 prefill 和首/末 decode 的 logits，记录 shape/dtype、非有限值、最大绝对误差、argmax 及 top1-top2 间隔。先测试 bitwise 一致；若不一致，保存首个差异并定位，不能事后放宽 tolerance。若原版自身已不稳定，应先界定其原因/可重复性，模型 A/B 暂不宣称通过。未保存全步 logits，报告不得称全部 logits 均逐位一致。

源码 `offline_send_result` 会在 finished 且 token==EOS 时不把最后一个 EOS 放进返回列表，即使 ignore_eos=True；因此同时记录实际采样生成步数和返回 token 数。吞吐分子明确选择哪一种，不得仅凭请求 max_tokens 假定返回恰好32，也不得把 prefill 的128个输入 token混入“输出 tokens/s”。

## 5. 小而有界的正式计时

先冻结研究版本与适配源码；A=原版CUDA，B=显式研究Triton。若真实合同尚未通过，暂不计时；若B回退原版，只能报告A/A。

- 两种政策各3轮，每轮A、B各启动一个全新独立进程，共12个模型进程；同一GPU串行执行，禁止并发抢占。轮1 A→B、轮2 B→A、轮3 A→B，Graph/Eager轮内顺序交替，计划先写盘。三轮无法让独立进程先后次序完全等量，报告保留此限制，不称完全平衡。
- 每进程加载/初始化/图捕获/编译后先运行2次同负载预热，再收集5次完整 generate。naive cache 保证不跨请求复用 prefix；计时前后核对真实 prefill 行数、decode 步数及输出。请求结束后资源必须正确释放。
- 主指标：已准备 token IDs 的离线 generate 墙钟时间，用 perf_counter；计时前 synchronize 清空先前任务，generate 返回后 synchronize 再读结束。模型加载/JIT/预热排除，generate 内的调度、sampling、D2H、结果组装和 detokenize包含。此指标不含网络服务，不能叫在线服务端到端。
- 保存全部5次时间、真实输出/步数、峰值显存和环境状态；每进程取中位数。3轮中位数的中位数作为代表值，同时逐轮配对报告 A/B、B相对A耗时变化和胜/平/负。5次测量不是5个独立进程；不报告请求P99或统计显著性。
- GPU阶段指标另采：在实际 engine.stream 上为 prefill/decode 的约定前向边界记录 Event，结束后统一等待/读数，不每步 synchronize 改变调度。明确是否包含 sampling/D2H；Graph 重放的内部阶段需要 trace支持，不能用 Python hook 拆不存在的回调。阶段诊断与主墙钟计时分开，以免Event插桩影响微小差异。

原版接口没有现成逐token TTFT/ITL结果。需要时可在**单独诊断**中记录请求入队、offline_send_result 首token及后续token的host时间戳：命名为“离线首token回调延迟/回调间隔”；其包含D2H与调度，不能把 GPU prefill Event 直接叫TTFT。主报告最少给 generate墙钟、实际输出tokens/s、prefill/decode GPU时间及工作量，不为术语齐全追加多套性能测试。

## 6. 真实 KV 占比：单独 profile，不用 M05 的微秒数硬乘

在原版成功后，为同一负载单独采 Nsight Systems：先预热，在 generate 与 prefill/decode 边界加标记；排除模型加载、JIT、Graph捕获、热身。至少记录原版的真实 kernel 次数/名称/累计持续时间，Graph运行必须能看到图内节点；无法分解时明确“占比不可得”，不能以 Python wrapper 时间代替 GPU kernel 时间。

分阶段保存：

1. `KV GPU持续时间之和`：该阶段实际 store_kv_cache kernel 的 duration 相加。
2. `全部GPU kernel持续时间之和`：同阶段同统计口径的kernel duration；两者之比仅称“GPU kernel持续时间份额”。多流重叠会重复计时，不能等同墙钟贡献。
3. `阶段GPU时间窗/离线generate墙钟`：说明端点及包含内容。给出时间线，检查KV是否在关键路径；仅凭份额不能证明它是端到端瓶颈。

如研究版可正确运行，再采同样一次 trace作对照；profiler时长不混入正式性能表。各阶段总写入量按实际B/D/dtype/层数汇总，只是逻辑字节数，不能宣称DRAM实测带宽或硬件利用率。

Amdahl关系 `T_new/T_old = 1 - f + f/s` 只在假设其余时间不变、f是可改善且不重叠的墙钟份额时使用。kernel持续时间份额不自动满足此定义；若未得到关键路径f，只给标注假设的情景计算，或直接报告观察到的局部时间与实际整体变化。不能从“KV约1微秒”推出模型一定提速。

## 7. 收益验收与停止规则

接入正确、可重复并得到可信负结果也可以完成M08工程交付，不以必须提速为条件。

本轮主口径事前限定为decode-Graph政策下的离线 generate墙钟。只有全部正确性通过、B实际为候选、3轮各自 `B_us/A_us <= 0.97`，才称“该固定负载出现一致至少3%的耗时下降”；仍不是统计显著性或普适收益证明。Eager独立给出相同判据与全部退化，不得用Graph正结果覆盖Eager负结果。3%是工程筛选门槛，不是仪器精度或噪声置信区间。

任一轮反向、差异小于门槛或A/A波动可解释该差异，结论写“本轮未建立稳定收益”，保留所有轮次后停止，不追加试到出现漂亮结果。首次出现功能/配置错误可以修复后重跑完整预定轮次，旧失败保留；环境中断只补对应预先定义的未完成项，不能按速度删除完成轮次。

M07选择策略继续冻结、默认原版。M08即使出现正结果，也只支持这个真实模型配置的研究结论；若要改变默认，需要另立带来源/范围的新策略，不能回改37的历史判据。本轮不扩展第二个模型、并发扫描、融合或多GPU；这些均为后续任务。

## 8. 最短下一步与记录归属

现在由主线程先执行原版单请求小负载，保存真实输出、配置、prefill/decode shape与首个错误（如有）。拿到实测后，只补实际合同缺口，冻结一个显式研究后端，再按3/4/5节顺序推进。本文为教练基于源码拟定的方案，不是用户已掌握方法、模型已跑通或M08已验收的证据。

本次只读关键源 SHA256（便于方案对应代码，GPU运行须另记live SHA）：

```text
llm/llm.py          cafaa39d456d77f328867d44a2055e5f171d3bedc3b2b256fa1e3809fb4e7fce
engine/engine.py    82bd58030c78dbfe249eaef857fbd48db9cb403e179dc1f64ade89cb3f8c699f
engine/graph.py     2fc80c2d4debf56ccec70850dcc1060afa52da631f90f8172d6c87d5ca4b4115
layers/attention.py fc155cea18ce71f0753b110f8c37a961754d0f8471ccb402b5607685a8510811
layers/rotary.py    f3ab653703e90d1f06b09ad875058f2a1af098ffa15a23b97018accc5825c57d
attention/fi.py     2708806defcd919bd6c5fbe9fd3e7a0deb726dcbf109737424f5116c7a352de4
kvcache/mha_pool.py c22ce82ac132be5dc18a26b8fbb58239cff9baa6f1053aee896376059cb9c4fe
kvcache/naive_cache.py 6081459a29dffb4f8d602a563287b537a2268fe0f06bb1d509de9e0a24746dce
scheduler/scheduler.py decc817459aec97e8d6f3984add4b8bb80d9cc4c7f19e05469ead588d9e27b92
kernel/store.py     39e21e21d1de0cc2f14faa37384ea94e427c0aabc997a56c049884ed46fc0c01
benchmark/offline/bench.py d453a143e6801cf3cd848060ab4fdfe436d41b14a5325e460387459c270b5476
```
