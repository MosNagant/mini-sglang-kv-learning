# 47号：真实线性层与GPU内核归因，事前计划

2026-10-15，教练实现诊断；不计为学员已独立掌握，不修改模型、权重或算子。

问题：43号Graph trace中两类GEMV占较多累计GPU内核时长，但Graph节点的创建阶段未保留模块NVTX，不能只按内核名和调用次数认定每个节点的调用来源。[旧证据限制](20261015-m08-prior-attribution-limits.json)保存离线SQL核对。

## 固定范围

- 原版KV后端；与42号相同的Qwen3-0.6B固定权重、FP16、TP1、RTX4070Ti、128输入/32生成、单请求；依赖及上游源码哈希门槛保持。
- 独立新进程，Eager模式、两次未插桩预热，然后一次Nsight诊断生成；不记入42/46性能样本。
- 只在诊断生成期间包装F.linear：以实际已加载的权重对象识别113个线性模块，记录真实input/weight/output shape与stride、源代码位置、prefill/decode和生成步数，给每次调用唯一NVTX标签；实际运算仍调用原F.linear一次。
- 每个调用不增加同步、读取GPU数值或写磁盘。诊断结束同步并检查全部32个原始生成token，记录全部元信息；这不是新增算法的完整数值验收。
- Nsight CUDA kernel通过同进程correlation ID关联CPU提交，再通过同线程CPU时间包含关系关联唯一NVTX，最后连接实际形状记录；不用GPU执行时刻直接套CPU范围。

## 预期及失败条件

源代码推导：每步28层×4线性＋1 LMHead=113次，共3616次；112次prefill decoder线性、3472次decode decoder线性、32次LMHead。只有实测顺序/形状/阶段验证后才能写成观察事实。

每个已记录线性调用都应有GPU内核关联；同一内核存在无法解释的CPU相关ID歧义时停止；不允许凭猜测关联。未标记的已知GEMV/GEMM/CUTLASS名字拒绝完整通过，其余非线性内核也全部另存。Graph节点出现时拒绝用本方案推导。

失败保存目录和首个异常；不改历史实验，不重复直到得到某个期望的占比。保留完整SQLite、nsys-rep、调用元信息、逐内核关联、源码与权重来源/哈希、worker日志和输出token。

## 允许与不允许的结论

可以报告本次Eager trace中具体矩阵形状、调用来源、次数及累计内核持续时间。不能把该占比转给旧Graph实验，不能称墙钟关键路径占比、真实DRAM流量、带宽饱和、occupancy瓶颈或优化提速。阶段完成后据结果选择一项后续实验，不为制造优势改变原版或降低原契约。

方法参考：[NVIDIA Nsight Systems SQLite分析说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)、[NVIDIA关于CPU API与NVTX关联的说明](https://forums.developer.nvidia.com/t/nvtx-ranges-in-cuda-context-in-sqllite-file/221776/3)。实际代码以本机2025.1.3导出schema为准，不假设最新版Graph NVTX功能存在。
