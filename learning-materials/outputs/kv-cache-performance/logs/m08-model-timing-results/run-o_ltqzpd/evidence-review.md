# M08 42号真实模型计时证据复核

CPU审计通过。12个独立进程、60个请求耗时样本全部保留；1920个输出token ID逐个与41号原版基准一致。没有运行新GPU实验。

102份上游源码哈希与41号验收快照一致；39助手、40适配器、42运行器及计划版本均核对。每个worker与summary一致，全部中位数、吞吐量和配对比值独立复算通过。

|策略|轮次|原版中位数 ms|Triton中位数 ms|候选耗时变化|
|---|---:|---:|---:|---:|
|Decode Graph|1|132.537|136.715|+3.152%|
|Decode Graph|2|137.948|139.741|+1.300%|
|Decode Graph|3|135.723|134.588|-0.836%|
|Eager|1|234.877|279.663|+19.068%|
|Eager|2|243.895|275.121|+12.803%|
|Eager|3|226.683|278.871|+23.022%|

正数表示候选更慢。Graph两慢一快，Eager三轮全部更慢；均未满足预先约定的“三轮均至少降低3%耗时”，原版继续保持默认。

范围为Qwen3-0.6B真实权重、TP1/FP16、单请求128输入token/32步生成、naive cache、page1、512可用页（物理513行含dummy页）、关闭overlap。Graph仅用于decode，prefill仍为Eager；每进程2次warmup、5次实测。同进程样本不是额外独立进程。

12份后端记录均通过：Graph每worker记录112次初始化/捕获/warmup Python提交，最后为B128；Eager为1792次，最后为B1。D1024、输入行距4096、cache行距1024。6个Triton worker均保存实际m08_store_rows的PTX/TTGIR，sm_89、4 warps。该计数不包含Graph replay，不能当作硬件kernel总数。

TTGIR仍保留41号快照源位置，属于相同适配器字节的编译缓存复用，并非重新编译。计时包含调度、采样、D2H、detokenize；候选Eager还包含适配器检查与Triton分派，不能把全部差值归因于KV GPU指令。后续profile需单独观察KV占比。

Windows CPU审计未重新读取WSL中的1.5GB权重，只核对完整hash与41验收记录一致；冻结42源码含采集前全文件hash门禁。详细哈希、样本及复算见[evidence-audit.json](evidence-audit.json)。
