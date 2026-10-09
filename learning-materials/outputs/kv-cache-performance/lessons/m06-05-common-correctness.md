# M06-05：原版与新入口参加同一套正确性检查

状态：23号入口及CPU审查已完成，待用户实际运行；尚无本课GPU结果。M06-04已交付，工程44/100、M06 14/25保持；本页不提前验收。

## 这一步解决什么问题

M06-04检查Triton入口接收合法输入、处理空输入、拒绝非法输入。现在让原版CUDA和这个新入口接收同一份合法数据，分别检查完整结果，确认它们完成同一个KV写入操作。旧M04原版通过不等于新入口已经通过；本课复用旧数据和检查，不重做M05计时。

调用链：生成CPU输入和独立期望 → 给原版新分配GPU输入/缓存并运行 → 等待并检查 → 给Triton另分配输入/缓存并运行 → 等待并检查 → 比较两者完整结果。

两种实现必须各自符合独立期望。只有两者相同还不够：它们可能犯同一个错误。每个实现开始时缓存都重新初始化，不能把原版写好的缓存直接交给Triton，让漏写被掩盖。

## 本人负责什么

继续使用本人已写、已审查的check_kv_targets：

~~~python
for pid, pos in enumerate(target_rows):
    assert torch.equal(k_cache[pos], expected_k[pid])
    assert torch.equal(v_cache[pos], expected_v[pid])
~~~

不再重复填写地址/mask/grid。教练负责矩阵、输入夹具、完整存储检查和归档。本人运行并理解一次结果；复盘时解释“如果漏搬最后一条V输入的最后一个元素，上面哪条断言应该失败”。这是一道针对本课的变式，不把完整套件当作本人独立编写。

B是本次输入总行数，pid是其中一条输入的编号。一次Triton启动包含B个program，不是B次Python调用。

## 配置表与计数

| 维度 | 本次取值 |
| --- | --- |
| 输入行数B | 0、1、2、3、4、5、17 |
| 有效行宽D | 64、128、192 |
| 缓存容量C | 32 |
| 目标缓存布局 | interleaved交错、separate分离 |
| 输入gap | 0、64 |
| 缓存尾部guard | 普通用例0；D192且输入gap64时额外检查guard64 |
| 类型 | K/V/cache为FP16，indices为int32 |
| 目标编号 | B1为[31]；B2为[31,0]；更多B继续追加1、2……，唯一且合法 |

普通布局配置为3种D×2种布局×2种输入gap=12，再加2种D192/guard64配置，合计14。14×7种B=98个配置，每个交给两实现，所以196个结果。

- 非空：14×6×2=168次KV调用/提交。
- 空输入：14×1×2=28次外层返回，不启动KV内核。
- 原M04的48非空+4空配置及D64/128输入数值完整保留；新范围明确包括更多B与D192组合。

原版通过教练的checked adapter统一校验并处理B0，不能称原版内核原生支持空启动。原版计数是调用原版算子的次数，没有用profiler测内部kernel数量；Triton记录真实提交的grid和参数。

## 运行（Windows PowerShell，一整行）

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/23-check-kv-common.sh
~~~

此次只读取已经保存的triton_store_api.py、triton_store_mask.py、kv_contract_support.py和check_kv_targets.py，不覆盖它们。现有WSL已恢复，不重复修复或重建环境。

预期末尾是COMMON_CORRECTNESS_PASS: 196/196；实际失败时先保留错误和对应run，再依据证据排查，不为了出现PASS改小覆盖。准备通过不等于实际GPU通过。

## 怎样读结果

~~~text
COMMON_CASE_PASS implementation=original_cuda B=5 D=192 layout=separate input_gap=64 cache_guard=64 submissions=1
COMMON_CASE_PASS implementation=learner_triton B=5 D=192 layout=separate input_gap=64 cache_guard=64 submissions=1
~~~

上面是预期格式示例，不是本次已运行结果。两行表示两种实现分别完成同一个配置。submissions=1是一次算子调用/内核提交；Triton内部有5个program，每条K/V行192个有效元素，输入行距256，目标分离缓存因guard64行距也是256。

检查包括目标K/V所有元素、未命中缓存、全部输入、源gap、缓存guard以及indices不变。期望在调用前由CPU构造；保存before.npz/after.npz完整数组，便于离线复核。控制实验故意损坏末行末列、错目标、K/V混用、未命中区域、输入、索引、gap/guard，确保检查器能抓到错误。

脚本把每个结果和源码快照存入experiments/learning/m06-common/run-*，并复制到共享logs/m06-common-results。它不测速度；无Triton提速、硬件利用率或模型收益结论。列guard检查写入，mask另经源码审查，不宣称运行时非法读取检测。未覆盖任意dtype、任意布局、任意索引模式或完整模型。

## 面试知识提醒

来源：教练设计的项目迁移题，不是招聘方原题。

- 概念：独立期望、回归测试、检查器负控制、测试覆盖范围。
- 代码：本人K/V断言与23号run_case中的完整存储检查。
- 基础问法：为什么两种实现输出相等还不足以确认正确？
- 变式：最后一条V输入的最后一个元素没有写入，哪个检查能发现？
- 易错：只看前四个数字；用同一份已写好的缓存验证另一个实现；把196个检查说成196次GPU内核启动；通过有限集合就声称支持任意输入。
- 深度：会读结果、会解释期望和覆盖；无需手写整套归档框架。个人答案仍待实际复盘。

通过并审查后，进入M06-07本机编译产物观察，再做M06-06同场公平计时与M06-08冻结。M05已完成，后面的速度比仍需同场重测原版控制组。
