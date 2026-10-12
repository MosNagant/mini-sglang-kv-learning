# 35号完整组/尾组效率对照：准备记录

2026-10-10。状态：运行入口及本地CPU检查已准备；没有用户WSL/GPU计时，不能写成性能通过。用户提出“现在要不要开始测试不同组别下的效率”，按现有教学顺序进入同场对照，无需再修改学员内核。

## 固定设计

三个版本为原版CUDA、已接受grouped、完整组条件版full_groups。两个Triton均为4行/program、4warp。本轮只测已接受if条件的影响，不同时扫分组大小或warp数。

B5/256分别取34号三种配置：D64交错/gap0/guard0、D128分离/gap64/guard0、D192分离/gap64/guard64，C512；数据种子与索引映射同34号。六配置各三新进程，共18worker、3240区间均值。Graph主口径、Eager单列；100调用/区间、每臂每口径30区间；六种arm/policy在每个测量位置各出现五次，设置顺序按轮次改变。

核心源锁定34实际提交：full_groups SHA `6a53a5304566420c3ee0163f0b8c6ff157b4f9b16f99a83be20364b415a81156`，grouped SHA `5b190ce52e0f6c25f99b739346c2787f194ed7e75c759144cf25fcb06c8a1bd1`。34已接受运行的来源/状态/启动记录作入口门禁，不重新运行34。原版通过32归档原版源文件与live哈希绑定，并核对实际导入路径；上游commit仍为`9a91cfafe754aa85daee49998176275667eb58f2`。

## 本地核验

[CPU控制报告](20261010-m07-full-groups-timing-cpu.json)完成52项负控，包含配置/顺序/统计、实际34来源门禁、临时项目新快照、独立raw闭包、完整guard恢复及worker前置失败留档。未导入Torch/Triton，GPU和WSL为NOT_RUN。root实际运行--dry-run，并核对报告中三文件SHA与交付文件完全相符。

独立审查确认：原版固定源码按真实源/目标行距和D复制，D192的有效范围不会因Triton BLOCK256而扩大；旧M06已有原版D192/gap64/guard64证据，但新增B256/C512仍须本轮首次调用与reset Graph正确性检查。每个worker对三臂分别检查实际完整输入、缓存、guard和indices；两处reset用完整初始cache复制，不会把guard=-7误改成-1。检查、记录、首次编译、预热、合同/索引转CPU验证都在计时区间外。

统计独立复核保留代表中位数之比与逐轮配对比，二者可能方向不同；不能择一报告。主流程保留部分样本、失败日志和唯一运行目录，不覆盖旧实验。详细独立检查见[独立报告](20261010-m07-full-groups-timing-independent.json)。

| 交付文件 | SHA256 |
| --- | --- |
| 35-compare-kv-full-groups.py | ae5efd06babab6296038f52e4385bf0295cb998b36a217b03274604f3fb84987 |
| 35-compare-kv-full-groups.sh | 624b8fb4cd0db14aca65b747ad102c888cf09e9d32bf91daa679f55c33a47fae |
| 35-full-groups-timing-cpu-test.py | 9e218de81618c123a2c58c15f946b2a97e40e9a2e51ad567930f767594248bfc |

## 交付与边界

PowerShell一行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/35-compare-kv-full-groups.sh
```

共享结果为`logs/m07-full-groups-timing-results`；完成标签`FULL_GROUPS_TIMING_COLLECTION_COMPLETE`仅表示收集完毕。所有比值都采用本轮同配置同场对照，不拼接32/34或M05旧耗时。

详见[本课与事前假设](../lessons/m07-10-full-groups-timing.md)。B5用作尾组对照，满组是否改善待测；跨D同时改变布局/gap/guard，不能直接作D单因素结论。重复地址、小工作集；没有DRAM、occupancy、模型或AMD速度主张。候选尚未晋升，完整回归另需验收。工程61/100、M07 8/25不变，无GitHub同步。
