# 从 Python 列表进入真实 CUDA store

2026-09-30。用户指出列表练习过于基础，教练调整节奏：不继续列表语法题，也不把未回答的三个列表问题作为后续阻塞。L01a 单例按示范运行成功，独立掌握程度仍保留为待验证。

**执行结果已收到：** 用户于18:27运行成功，见 [共享原始日志](../logs/real-store-20260930-182741-GPaQSw.log)。RTX4070Ti上实际调用原版store；缓存5和1打印出的前四个K/V值分别对应输入0和1，其他六行打印出的前四个值仍为-1。完整结果在WSL experiments/learning/real-store/run-20260930-182741-8kevyxeq/result.json，助手未读该文件。当前不重跑，转入下方地址推导；不标完整正确性PASS。

## 当前练习：从实测步长推导地址

用户已用自己的话表述indices指导输入数据写到哪些行，随后表示理解并要求继续，尚未给出偏移算式。当前转入读取项目store.cu，结合实际线程与指针代码再验证理解；不要求先补答旧题。以下偏移推导保留作参考，不一次堆积，也不将用户自述理解记为独立验收。

- 输入K shape=(2,128)，stride=(128,1)；K缓存shape=(8,128)，stride=(256,1)。stride单位是元素，FP16每元素2字节。
- 本脚本底层缓存[8,2,128]按K0、V0、K1、V1排列，每段128个数；因此相邻K行起点跨256个元素。这里描述本次测试布局，不预设真实mha_pool也相同。
- 追踪输入[1,3]的141写到缓存[1,3]：用户算出输入与目标相对各自K基址的元素偏移、字节偏移，并判断目标误用行步长128会落到哪块区域。
- 提示：先跨过前面的行，再走到行内列；得到元素偏移后乘每元素字节数。尚无用户答案，不标掌握。下一步将其推导对应到固定版store.cu。

## 这一步的任务

运行一次已经通过上游测试的原版 CUDA store，观察两组输入按 [5,1] 写入8行缓存。宽度沿用原测试的128，dtype为FP16；K/V共用形状[8,2,128]的缓存，初值为-1。K输入为[2,128]，V=K+1000；目标索引唯一且合法。使用小数据观察，无模型、无新内核、无计时。

助手提供 08-observe-real-store.py/sh 作为准备数据、启动、记录脚手架。关键调用仍是上游 store_cache(k_cache, v_cache, indices, k, v)，不是 Python 列表赋值。脚本记录固定HEAD并保存调用相关源码；若这几个源码文件相对HEAD有改动则停止，保留diff，避免把改动版本标作原版。

用户在 Windows PowerShell 执行这一整行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/08-observe-real-store.sh
```

预期输出包含GPU名称、shape、stride、indices、两行输入与八行缓存的前四个值，以及 REAL_STORE_OBSERVATION_COMPLETE。首次调用可能进行JIT编译，耗时不作性能结论。完成标记只表示观察流程运行完成，不是完整正确性通过。

用户对照输入预测编号5和1的内容，检查输出与预测，观察其余六行前四个值是否保持-1。控制台只显示每行前四个数，完整128个数在result.json中；后续由用户写正式正确性检查，不能由控制台片段宣称全面通过。

原始结果、脚本副本与源码保存在 WSL 项目 experiments/learning/real-store/run-*，控制台日志同步保存到本工作区 logs/real-store-*.log。失败时保留第一个错误；若 SOURCE_EDITS_PRESENT，先检查保存的diff，不自动还原用户代码。若 WSL 创建虚拟机失败，使用环境手册中已验证的故障分流，不重复安装依赖。

准备检查：助手在 Windows 只做 Python AST 语法检查和 shell -n 检查，均通过；尚未代用户执行 WSL/GPU。本脚本不是微基准或测试框架。

后续依据实际输出学习：shape与stride如何对应GPU张量行起点；同样128个FP16数，输入和缓存的行间距为什么不同；继而对照固定版store.cu中的线程映射和指针偏移。核心CUDA、计时实现与性能解释仍由用户先尝试。
