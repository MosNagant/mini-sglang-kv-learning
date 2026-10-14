# M08 43号真实模型 profile：独立审计

审计对象：[run-uml9cxd9](../logs/m08-model-profile-results/run-uml9cxd9/)。[audit.json](../logs/m08-model-profile-results/run-uml9cxd9/audit.json)已保存直接SQLite查询、完整kernel名、哈希和限制。CPU只读重算通过，无GPU运行，原始result未改。

SQLite与Nsight导出的CSV、原result三者一致：全部 **11,976个kernel，26种名称，累计103,927,254 ns**；其中原版KV写入 **896次，累计812,245 ns**。比例为 **0.7815514879%**，分母明确是 `SUM(kernel.end-kernel.start)`，不是generate墙钟时间，也不是可用于Amdahl的关键路径份额。

累计kernel时间前五如下；较长的模板名在表中缩写，完整名称保存在audit.json的top5。

| kernel名称/特化 | 次数 | 累计ms | 占全部kernel累计时间 |
| --- | ---: | ---: | ---: |
| `internal::gemvx::kernel`，`(int)6`特化 | 3,472 | 64.975756 | 62.520420% |
| `internal::gemvx::kernel`，`(int)7`特化 | 32 | 21.642346 | 20.824514% |
| `flashinfer::BatchDecodeWithPagedKVCacheKernel` | 868 | 4.634928 | 4.459781% |
| `flashinfer::norm::FusedAddRMSNormKernel` | 1,792 | 2.588559 | 2.490741% |
| `flashinfer::norm::QKRMSNormKernel` | 1,792 | 2.147839 | 2.066675% |

这份证据适合下一步优先核对GEMV与具体模型调用/矩阵形状的对应关系。尚未做这一归因，不能仅凭模板名把某一项指定为某层投影或LM head，更不能据其累计份额直接承诺整机收益。

43源码先完成模型加载、decode Graph捕获和两次固定负载预热，再在cudaProfilerApi区间内运行一次generate。显式原版入口未被替换；实际128输入token、32步生成输出与冻结token序列吻合，包版本吻合。保存的102项上游源码哈希匹配，成功run的`m08_profile_worker.py`与当前43号一致。Nsight命令启用Graph节点追踪，并关闭CUDA event completion trace。全部kernel均处于唯一`M08_ORIGINAL_GENERATE` NVTX区间内，无非正持续时间。

KV本身可按数据库`graphNodeId`与实际grid分组：**28次非图prefill KV，gridX=32，累计35,969 ns；868次decode图内KV，gridX=1，累计776,276 ns**，两组blockX均128，合计精确等于896次/812,245 ns。这里只拆分KV累计时间。decode另有图外sampling等操作，不能把所有非图kernel都当作prefill；本次没有完整prefill/decode阶段时间划分。

限制：这是单次插桩的原版decode-Graph政策诊断，prefill仍Eager；未采Triton配对profile，不混入42号未插桩性能样本。保留`CuptiUseRawGpuTimestamps=false`设置，WSL时间戳精度和节点追踪开销随结果说明。SUM kernel时长不是墙钟/关键路径、带宽、占用率或统计置信结论；不据0.78%直接推导整机加速上限。

失败的[run-rv7o7fmk](../logs/m08-model-profile-results/run-rv7o7fmk/)完整保留：原入口名`profile.py`，日志为Transformers `GenerationMixin` ImportError，退出码1；修复记录将入口改为`m08_profile_worker.py`以避免遮蔽标准库profile。审计记录失败目录全部文件哈希并确认期间未变，没有用成功结果覆盖失败证据。

复算脚本：[20261014-m08-profile-independent-audit.py](20261014-m08-profile-independent-audit.py)。教练执行CPU审计，不增加学员独立掌握或AMD实测证据。
