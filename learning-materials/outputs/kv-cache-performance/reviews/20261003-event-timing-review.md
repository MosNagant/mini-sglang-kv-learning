# M05 首轮计时审查

依据：用户回传的结果JSON及归档源码函数；本轮未直接访问WSL、未重跑GPU。

- `run-vtqqu_54`：固定提交9a91cfafe754aa85daee49998176275667eb58f2，tracked_git_status为空，B=1/D=64/C=1024、FP16、indices=[0]、输入stride=(64,1)、缓存stride=(128,1)。空的tracked_git_status不表示未跟踪文件也已核验。
- 归档函数经用户sed回传：start.record→100次call→end.record→end.synchronize→elapsed_time/100，顺序和归一化正确；warmup先循环后统一同步。
- 中位数22.277119微秒/次，范围8.683520～24.483840微秒/次；这些是30个区间各自平均每次的统计。
- JSON的policy仍写one call per sample，是元数据未更新；不改写历史结果。按归档源码解释为100次区间均值，不能把它误当单次调用分布。
- 之前14.336微秒是B3/D64/C8的单调用计时；7.8792微秒是B3/D64/C8的100次区间均值，两者不能作为算子优化加速证据。
- 当前先做同配置eager/graph执行方式对照，再决定正式基线方法；未完成矩阵或正式基线，不增加项目验收分值。

新增教练诊断工具：[17-compare-event-graph.py](../17-compare-event-graph.py)。复用用户归档的已审查计时函数，不修改用户文件；B1/D64/C1024固定，对照Python逐次提交与包含100次调用的CUDA Graph重放，交替顺序采30对样本。捕获前在side stream预热；捕获后重置缓存再重放检查，避免把旧正确结果当作图执行成功。保存原始样本、环境、源码和异常。

此工具是教练提供的测量诊断，不记为用户独立手写；图重放结果是一种新执行口径，不能直接用来声称修改算子提速。尚只做Python语法检查，图兼容性与GPU结果待用户运行。若capture不支持，保存实际错误再分析，不推定故障原因。

原理：[PyTorch 2.9 CUDA Graphs](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs)。图重放减少逐次主机调度开销，也可能改变GPU执行开销；结果差异不是对所有主机开销的精确分解。
