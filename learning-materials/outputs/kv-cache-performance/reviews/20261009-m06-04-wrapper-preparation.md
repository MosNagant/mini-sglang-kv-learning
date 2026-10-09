# M06-04 外层入口准备审查（2026-10-09）

状态：脚手架已准备，学员三个表达式未提交；WSL/GPU未执行。工程41/100、M06为11/25保持不变。

## 产物及分工

- 22-triton-store-api-template.py：本人填索引数量、B0、grid三处。按明确课题限制进行AST预检；其他等价写法需要重新审查，不等于被证明错误。
- 22-kv-contract-support.py：教练辅助元数据、索引值和有效地址区域别名校验。不替本人写上述三处。
- 22-check-kv-wrapper.py/sh：排他创建新学生入口与辅助模块，保留现有文件；实际加载已通过的triton_store_mask.py，不改旧内核。
- 新WSL目标为 experiments/triton_store_api.py 和 kv_contract_support.py；仅用户运行--prepare后才创建，不冒称已经安装。

## 待GPU运行的范围

正常4例：B3/D64交错，B5/D64交错，B5/D192分离且input_gap64/cache_guard64，B1/D128分离。另有B0/D64交错与B0/D192分离空输入2例，要求零KV启动。固定容量8，合法非重复int32目的地，FP16、支持行距与列距1。

18个非法调用分别涉及索引数量过多/过少、V形状、K类型、indices类型、CPU源/索引、宽度96、列/索引步长、K/V源行距不一致、不支持的源行距、缓存彼此重叠、源目标重叠、重复/负/越界索引、空输入却有一个索引。准备张量本身可使用GPU；要求的是在提交KV内核之前拒绝，不称完全没使用GPU。

每个正常例检查完整cache、输入及gap/guard、indices，再调用已有学员K/V检查。每个非法例须收到公共入口TypeError/ValueError，且LaunchProbe提交计数为0；探针自身RuntimeError不能伪装正常拒绝。探针对正常调用验证grid与全部启动参数再转交原mask内核。

## 已完成本地检查

辅助模块47项CPU假张量检查见20261009-m06-04-contract-cpu-review.json；fixture及源预检见20261009-m06-04-fixture-cpu-review.json。实际GPU编译/运行与学员提交均未发生，后续整条CPU模拟另存。Python新文件语法已检查。

独立只读审查发现并修正NumPy空数组stride问题：B0源与indices改用torch.empty显式构造连续空张量，不依赖from_numpy空数组的零步长。正常调用的launches.json放进finally，首次提交或同步失败也保存参数；不为留档读取已经出错的GPU上下文。旧结果和当前学生文件不会被覆盖。

## 支持与性能限制

元数据支持范围不等于已在所有B/C组合验证；本课只验上述小集合，共同M06-05仍待接入。源K/V只读区域彼此重叠允许；所有缓存有效行与源/另一cache/indices有效区间的写别名拒绝，同底层交错K/V不误判为重叠。不支持并发修改indices或源/目标。

含校验公共入口会通过indices.cpu().tolist()读取目的地值，发生传输/同步及主机检查成本。别名元数据检查也有主机工作成本；后续计时必须将该边界说清楚，不能直接把这个入口放进Graph/kernel-only比较。没有当前Triton计时或提速结论。

整条教练reference入口的27项CPU模拟已通过，见20261009-m06-04-wrapper-cpu-check.json：6有效/空调用、18非法拒绝、3错源码预检；实际grid/参数/完整backing与零启动检查通过。安装保留检查见20261009-m06-04-install-review.json。以上不含真实Triton/GPU，也不代表学员提交。
