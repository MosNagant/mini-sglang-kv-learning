# 29号多行program练习：准备审查

日期：2026-10-10。结论：检查入口、未填写模板和讲义已就绪，CPU检查及独立源码审查通过。没有运行WSL/GPU、Triton编译或性能；工程57/100保持。

## 改动与分工

固定4warp，将每program由1行改为4行，比较冻结v1与新grouped候选。新增文件为29号Python/shell和29-triton-store-grouped.py模板，安装目标为WSL experiments/triton_store_grouped.py。本人填写rows、row_mask、grid；教练提供二维广播、掩码读写、合同、检查及记录。模板仍有3个Ellipsis，没有把参考答案安装给用户。

教练提出第二个假设；用户对B256的预测为“暂时不确定，但我认为应该更快”，尚无理由或B1预测。运行时保存当时的[假设卡/讲义](../lessons/m07-04-grouped-rows.md)并纳入源码哈希。

复用23号fixture/GPU封装、26号完整存储检查与v1探针、27号接受来源门禁和失败导出；新分组探针核对ceil(B/4)、BATCH、真实stride、BLOCK_SIZE、4行/program及4warps。旧来源哈希核对后才导入GPU库；冻结v1、1-warp候选及历史证据均不改动。

## 范围与关键审查

17配置：三种指定D/layout/gap/guard组合各配B1/3/4/5/17，加两个B0配置。每配置两臂独立分配同内容输入，共34项，计划30次非空提交、4次封装空返回。完整缓存、未写槽、guard、输入含gap、indices对调用前独立期望核对；两臂结果再互比。不重跑非法输入集合或全98配置，也不包括B256/C1024及性能。

- 先用行mask保护indices load，再用行与列mask组合保护所有K/V load/store。
- 源行由组内输入rows确定，目的行由indices返回的positions确定；分别乘真实行距。
- B0在wrapper返回，不当成GPU处理空输入；program数量与KV提交次数分别记录。
- 源码AST要求三处经过审查的表达式；等价的不同写法需要人工复核，不能将门禁当成一般安全证明。
- 独立审查建议保持v1的Kload/Kstore/Vload/Vstore源码顺序，已修正，避免额外改变读写顺序。编译映射仍可能随分组变化，不能假定每warp一行。
- 新建唯一运行目录，先保存实际学习者源码再审查；失败保存首错、状态和可取得的结果，导出不覆盖旧目录。

## CPU验证

最终记录：[final CPU报告](20261010-m07-grouped-scaffold-final-cpu.json)。17夹具正例及完整存储损坏反例、15个正确模拟启动、53个错误模拟启动拦截、12类未填/源码变更拒绝、仅首次创建与保留已有编辑均通过。外部依赖来源由独立审查核对。shell为LF且无BOM，模板保留3处TODO。

前两份CPU报告保留其对应修订，不作为最终文件哈希：scaffold-cpu为初稿、root-cpu为加入假设快照后的版本；final-cpu对应恢复K/V顺序后的最终版。没有把这些CPU控制冒充GPU执行或非法读取检测。

| 文件 | 最终SHA256 |
| --- | --- |
| 29-check-kv-grouped.py | 3a8edafb19ac86eaba7769cbb77be24a4c7490e93b8dd322d882f8e4dadfffdb |
| 29-check-kv-grouped.sh | f79597a47d874520ffe08bb0cd0594a861ac0906482982d0f9a7a5346a677148 |
| 29-triton-store-grouped.py | 86890437f0125140db796db80109497a793e97e9ba7ab48ed0335f7189310646 |

下一步用户运行29号--prepare、填写三处、提交审查，再运行不带--prepare的检查。尚无grouped性能收益或独立掌握结论；本轮未同步GitHub。
