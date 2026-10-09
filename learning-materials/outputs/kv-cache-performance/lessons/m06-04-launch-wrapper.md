# 2026-10-09：M06加速串联与调用封装第一步

当前状态（10-09 13:42验收）：M06-04限定范围交付通过，M06为14/25，总工程44/100。见[验收记录](../reviews/20261009-m06-04-wrapper-review.md)。以下备课/待填描述保留学习过程，最新结果及下一步以本段和文末为准。

## 今天按三个实作块推进

| 顺序 | 对应任务 | 要理解的问题 | 可见产物与验收 |
| --- | --- | --- | --- |
| 1：入口与统一检查 | M06-04/05 | 调用者交来张量后，怎样决定启动数量、拒绝不支持输入、确认搬对？ | 可导入的外部封装；B=0不launch；原版与v1跑共同正确性集合 |
| 2：编译与执行 | M06-07 | Python形式的Triton代码怎样成为GPU运行的代码？ | 本机一种编译产物或元信息；区分首次编译与后续执行 |
| 3：公平对照与冻结 | M06-06/08 | 同条件下v1与原版各用多久，这版到底支持什么？ | 同场六配置及原始样本；v1设计说明与复现入口 |

初始预算每块60–90分钟，合计约3–4.5小时，另留排错时间；这是学习安排而非完成承诺。第一块按已有资产复用，不重新搭环境或重讲全部基础。今天争取走完M06，以实际提交、运行和解释为准；若没有结果，保留断点不提前验收。M07优化方案须等本次对照后再选。

## 一条完整调用链

调用者给出 k/v、indices、k_cache/v_cache
→ 外层Python读取shape/stride并校验支持范围
→ 根据B启动grid，传WIDTH/真实行距/BLOCK_SIZE
→ GPU内核：pid→indices[pid]→pos→地址→mask→load/store
→ 正确性脚本等待完成，与独立期望比较
→ 正确性通过后，另用约定的计时入口比较原版和v1。

Python外层在CPU组织调用，shape/stride是张量元数据，读取它们不等于把整份GPU数据复制回CPU。这里K/V原本已在GPU显存，store做的是设备内的按索引复制，不是H2D上传。GPU提交与完成是两件事；正确性脚本等待完成后再观察结果。

## 本来就有的代码在哪里

[21号检查器](../21-check-kv-mask.py)第157–181行的make_launcher已经含有：

~~~python
batch, d = k.shape
src, dst = k.stride(0), kc.stride(0)
block = 1 << (d - 1).bit_length()
module.store_kv_v1[(batch,)](
    kc, vc, indices, k, v,
    WIDTH=d, SRC_ROW_STRIDE=src,
    DST_ROW_STRIDE=dst, BLOCK_SIZE=block, num_warps=4,
)
~~~

这是省略检查与打印后的原代码摘录，不是新的完整公共接口。block那行求不小于有效宽度的最小2次幂：D64→64，D128→128，D192→256。

目前启动器只接受教学B3/C8、选定width/stride。合法FP16、同GPU、int32及唯一合法[7,0,3]来自测试准备；它还不是对任意调用者完整校验的入口。M06-04要把调用部分抽为可导入函数，补齐契约和B0路径。已通过的mask内核保持原证据，新增封装独立验证。

| 取值 | 回答的问题 | 例子 |
| --- | --- | --- |
| k.shape[0] | 输入有几条K行，也对应几条V行？ | 5行→本设计需要5个program |
| k.shape[1] | 每条有效数据有几个元素？ | 192 |
| k.stride(0) | 相邻输入行开头距离多少元素？ | 若有64个input_gap，则256 |
| k_cache.stride(0) | 相邻目标K行开头距离多少元素？ | 分离无额外guard时192；实际仍读取张量 |
| BLOCK_SIZE | 本program生成多少逻辑列位置？ | 256 |
| indices[pid] | 这一输入行放哪个缓存槽？ | 按索引内容读取，不由B或stride决定 |

上表5行是教学推导，尚不是新GPU运行。grid=(B,)是本版“一program处理一条输入K/V”的设计，不是所有Triton内核的固定规则。一次launch有B个program，不是Python for循环调用B次；具体并发调度由GPU决定，不保证全部同时执行。

## 现在只补一个启动片段

假定前面的输入校验已通过。这里暂不要求默写全部校验框架。

~~~python
batch = k.shape[0]
if batch == ___:
    return

store_kv_v1[(___,)](...)
~~~

两个空分别是“没有任何输入行时”的判断，以及“每条输入行安排一个program”的数量。return结束外层函数，不再提交内核。请先在对话中补这两处；完整接口与验证随后接入，不把含省略号的片段当作可直接运行的算子。

需要接着补齐的边界：shape匹配、FP16/int32、同CUDA设备、列距1、各侧K/V共享行距、约定行距与宽度、indices唯一且合法、源目标不能重叠。合法的B0应直接返回，避免空indices的min/max或不必要launch。列mask只管列，不会自动防止错误pos或重复目标冲突。

## 和计时的关系

元数据检查与GPU索引内容检查分开说明。读取indices到CPU，或用GPU归约结果驱动Python判断，可能引入同步；唯一性检查也有工作成本。未来需要区分包含校验的接口与准备阶段已验证输入的内核调用。共同对照选择相同边界，不能只让Triton付额外校验成本。当前21号里的print、同步、完整拷回、JSON记录属于正确性流程，不直接作为benchmark入口。

M05冻结基线不重写。M06正式速度比用本次同场原版/v1；Graph对Graph、Eager对Eager；固定六配置、3进程轮次、30个区间、每区间100次/100。先验证再计时，任何提速目前都还没有证据。

## 面试知识提醒

来源：教练设计的项目迁移题；不是招聘方原题。

- 知识点：主机封装、grid与program、空输入、异步提交。
- 项目位置：21号make_launcher与mask内核中的program_id(0)。
- 基础问法：B从3变5，哪一层决定多启动两个program？
- 变式追问：B=0时，应该由谁结束工作？重复indices能否依赖program编号决定最终值？
- 易错：把一次launch说成Python调用B次；把BLOCK_SIZE或num_warps当作输入行数；用列mask处理越界槽号。
- 目标深度：会改grid/空输入片段，能解释调用和执行边界；尚未提交回答，不认定掌握。

本次记录：教练读现有源码并整理教学桥接，未修改WSL学生文件、未运行GPU、未准备完整封装或计时器。下一项仍是本人补启动片段，再接公共入口和检查。


## 实作入口已准备（10-09，尚未GPU验证）

刚才的三个表达式现在放入新的 triton_store_api.py 模板：索引数量用哪个量、空输入与谁比较、grid使用哪个量。本人先填，不重复改已经通过的mask内核。上述片段的答案在课堂提示中出现过，本轮完成记录为提示后练习，不冒称完全独立设计。

1. 在Windows PowerShell创建新文件（已有文件保留）：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/22-check-kv-wrapper.sh --prepare
~~~

2. 打开新的外层入口：

~~~powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_api.py'
~~~

只替换函数内三处 ... 表达式，保留四空格缩进。Ctrl+S后运行：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/22-check-kv-wrapper.sh
~~~

预期检查4个非空GPU用例、2个空输入零启动用例、18种非法调用在KV启动前拒绝。EXPECTED_REJECTION是故意错误被拦住，不是任务失败；最终是否通过仍看实际输出与保存结果。run目录在experiments/learning/m06-wrapper，Windows共享结果在logs/m06-wrapper-results。当前这些GPU用例均未运行，不提前计分。

辅助校验代码和运行器由教练完成。元数据辅助47项、本课完整入口27项CPU模拟（含6个有效/空输入、18非法及3源代码反例）、安装保留检查与语法检查已通过。CPU模拟实际执行的是教练填好的内存reference，不替学员填写文件，也不等于Triton编译或GPU通过。审查见[准备记录](../reviews/20261009-m06-04-wrapper-preparation.md)。


## 实际运行与本课结论（10-09）

4个非空调用通过，2个空输入无KV启动，18个非法调用均在KV启动前按预期拒绝；完整结果和11项哈希复核一致。本人初稿B0条件正确，索引数量与单元素grid元组在提示后修正并运行；独立设计能力不由PASS自动确认。

一次launch可以包含B个program。B5、KV_launches=1就是一次提交5个program；B0、KV_launches=0是外层直接返回。EXPECTED_REJECTION是故意错误被正确拦住。

本课新增外层入口与支持范围检查，内部mask搬运源码没有改动。调用含GPU索引值到CPU的检查，可能同步，没有计时或加速结论。下一步M06-05把原版与新入口接入共同正确性矩阵，无需再改这三个表达式或重跑M05。
