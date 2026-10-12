# M07：完整四行组的编译期特化

日期：2026-10-10。状态：21:30用户已填写条件并运行，46项正确性通过，四份同场编译产物已保存。完整组B256/D128行比较消失、寄存器18→14；尾组B5/D192保留行列保护、22→22。详见[实际复盘](../reviews/20261010-m07-full-groups-review.md)。尚未计时，不能声称加速；无需重复本轮。以下保留课前教学步骤和当时的计划措辞。

## 今天只解决这个问题

当前grouped每个program处理4个输入行位置，num_warps保持4。原实现每次使用`row_mask = rows < BATCH`保护行边界。如果B正好能被4整除，启动的每个program都负责完整四行，我们希望编译器利用这个已知事实，尝试省去运行时行比较。性能能否改善留待实际结果，不预设更快。

先用两个例子看清楚，行号从0开始：

| 输入总行数B | grid | program 0的行位置 | program 1的行位置 |
| --- | --- | --- | --- |
| 8 | 2 | 0、1、2、3，全有效 | 4、5、6、7，全有效 |
| 5 | 2 | 0、1、2、3，全有效 | 4、5、6、7，只有4有效 |

判定的是“这一批能否分成完整四行组”，不是“B是否很大”。B256满足条件，B257不满足。这里不能用`B>=4`代替整除，也不能因为第一组完整就取消所有program的行保护。

## 谁负责哪一行

新增文件为`/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_full_groups.py`。原grouped、v1和1-warp候选保留。

教练提供分支框架和全部外围检查；学员只填写一个条件，替换`if ...:`的省略号：

```python
rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
if ...:  # 判断BATCH除以ROWS_PER_PROGRAM的余数是否为0
    row_mask = tl.full((ROWS_PER_PROGRAM,), True, tl.int1)
else:
    row_mask = rows < BATCH
```

提示：Python的`%`表示求余数，例如`8 % 4`得到0，`5 % 4`得到1。表达式使用变量`BATCH`和`ROWS_PER_PROGRAM`，不要写死256或4。变量`BATCH`是本次总行数，`pid`是当前program编号。

`tl.full`这行生成4个True，保持row_mask是一维向量，后续`row_mask[:, None]`仍能和列mask组合；不要换成单个Python `True`。[Triton full文档](https://triton-lang.org/main/python-api/generated/triton.language.full.html)说明了shape、填充值与dtype的含义，实际兼容性仍待本机Triton3.5.1运行验证。

`BATCH`与`ROWS_PER_PROGRAM`已经是`tl.constexpr`参数，编译这一特化版本时知道它们的值。我们希望这段选择发生在编译阶段。没有完整组时保留原来的动态行比较，不降低支持范围。

后面的indices读入仍使用`row_mask`；K/V两次load与两次store仍使用行列合并的`valid`。D192时BLOCK_SIZE256中的最后64列继续被屏蔽。行边界和列边界是两件事。

帮助程度：假设、if/else框架、全True向量和运行器由教练提供；学员条件表达式及说明待提交。不能把本课记录为学员独立提出全部优化。

## 操作：三条各自是一整行的PowerShell指令

先创建独立练习文件；已有文件保留，`--prepare`不运行GPU：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/34-check-kv-full-groups.sh --prepare
```

打开它，填写上面的一个条件并保存：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_full_groups.py'
```

保存后检查新候选和原grouped：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/34-check-kv-full-groups.sh
```

未填或改变了其他关键表达式会先停止供审查，不靠运行未知内核来检查语法。出现真实WSL故障时再按环境恢复记录处理，不预先重复存储修复。

## 本次检查什么

容量C512，FP16 K/V、int32唯一合法indices；固定随机数据并保留可区分的行内容。B取1、3、4、5、8、17、256，每个B检查三种配置：

| D | 缓存布局 | input gap | cache guard | 源行距/目标行距，元素 |
| --- | --- | --- | --- | --- |
| 64 | interleaved | 0 | 0 | 64 / 128 |
| 128 | separate | 64 | 0 | 192 / 128 |
| 192 | separate | 64 | 64 | 256 / 256 |

合计21个非空配置×原grouped/新候选=42次KV提交；另两个B0配置×两臂=4次封装直接返回。计划共46项。每次独立检查目标全元素、未写缓存/guard、完整输入底层存储与indices，并核对真实grid/参数。每个非空调用仍只提交一次kernel，program数为ceil(B/4)，不能把program数叫启动次数。

这是一组针对本次分支的检查，不是全部参数回归或硬件非法读取检测。旧完整回归已留证，新版本若晋升仍需对应的完整验证。

同一次运行额外保留四份真实编译产物：两臂各观察B256/D128完整组和B5/D192尾组，不为观察再额外启动相同kernel。不清除编译缓存，不保证发生新编译。

编译差异以本次34号同配置的原grouped和新候选为对照。此前33号B256/D128是交错缓存、gap0；本次同宽为分离缓存、gap64，不能跨这两个布局直接把所有差别归因于本次if。

## 怎样判断值得继续计时

先看是否正确，再核对实际编译结果：完整组中的行比较是否消失或变成常量；尾组是否仍有相应保护；列mask和读写范围是否保持。仅看PTX里还有`@p`不能判定失败，常量谓词也可能这样显示。只数指令行数不能证明执行更快。

如果没有预期代码差异，就记录编译器可能已处理/候选未产生改进的事实，先分析再决定是否计时。如果有差异，再固定配置与对照做新的公平实验；本轮34号没有计时，不能推出速度比。原版CUDA和冻结v1性能参照保留，后续正式性能比较仍需同场控制。

## 面试知识提醒

- 概念：编译期特化、边界条件、行/列mask的独立性。
- 基础问法：为什么B已知时可以生成不同代码？代价可能是什么？
- 变式追问：B从256变257，grid和最后一组的mask怎样变？D从128变192又影响哪个mask？
- 易错说法：“B是4的倍数，所以所有mask都可以删掉”“看到少一条PTX就一定更快”。
- 目标：能填写条件、说明合法范围、看懂实际编译对照；个人掌握待回答后记录，不设为运行前的额外口试门槛。
