# M06-03 mask示范及测试入口准备审查

2026-10-07；用户明确请求“帮我改一下我们测试一下”。核心改动由教练按请求示范，不能记为本人独立实现。当前工程37/100不变，M06-03 GPU尚未运行。

## 改动与入口

- [mask示范](../21-triton-store-mask.py)：新增BLOCK_SIZE，按cols<WIDTH生成mask，K/V两load加mask和other=0，两store加mask；源/目标行距公式不变。
- [测试运行器](../21-check-kv-mask.py)及[WSL入口](../21-check-kv-mask.sh)：仅创建新的experiments/triton_store_mask.py，已有新文件保留并审查；旧triton_store_v1.py和14号不改。
- 16旧范围回归+8个D192+4个cache_guard64，共28例。固定FP16 B3 C8 int32[7,0,3]，两布局、input_gap0/64；guard用例输入gap64，缓存每条K/V行尾实际分配64元素。
- 全量存储、原始源码/哈希、checker正负控制、失败输出和异常保存到唯一run，复制到Windows共享logs/m06-mask-results。
- 源码预检暂要求与示范函数体AST相同，以免未经审查的无mask版本进入无guard GPU用例；这是本课准备策略，不是通用算子接口或安全证明。后续本人改动需先审查，再相应调整本课入口。

## 已实际运行的本地验证

这些都是Windows工具侧CPU检查，没有CUDA/Triton编译和GPU成绩。

1. 两份Python语法解析通过。
2. [CPU存储控制](20261007-m06-03-cpu-controls.json)：14个正确fixture接受，122个损坏反例拒绝。
3. [运行器模拟](20261007-m06-03-local-check.json)：28次模拟调用通过，真实stride/BLOCK参数对应；guard故意破坏被拒绝且完整结果/异常留档；安装保留旧v1和已有新文件；7个源码变式被预检拒绝。
4. [独立语义审查](20261007-m06-03-independent-check.json)：使用NumPy/fake-tl执行实际函数主体，独立重建14组fixture全存储一致；D192的191列允许、192列屏蔽。7个坏版本均被预检拒绝。未发现阻断问题。
5. 旧14号Python及Shell哈希与已验收16例快照完全一致。

## 重要证据边界

原版FP16 D192为384字节，固定源码warp复制选择4字节/lane、每个K/V3轮；缓存允许匹配的真实行距，guard切片有效宽度仍为192。此为源码支持推导，实际GPU仍待运行。

独立CPU演示确认：去掉某个load的mask可以额外读取guard而不改变存储，因此“完整输入未改”并不能证明没有额外读。两个load安全性在本次由保存源码审查支持；guard检查主要发现不应发生的写入，未进行运行时非法读取检测，也未故意运行无mask的坏GPU内核。

## 下一步

用户在自己的PowerShell执行[本课命令](../lessons/m06-03-mask-intro.md)，脚本自动创建mask文件并运行28例；工具账户不能直接访问用户WSL。成功时应显示MASK_CASES_PASS:28/28，再结合保存源码/结果审查。失败保留原始记录后排错。本次没有性能或AMD结论，不修改M05冻结基线。
