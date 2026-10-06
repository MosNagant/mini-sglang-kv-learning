# 第一条真实GPU写入结果检查

日期：2026-09-30。当前任务由地址纸上练习转为一个实际检查；完整地址和边界推导仍待验收，不因跳到本练习而视为已掌握。

## 目标

用09脚手架调用固定提交的原版store，用户编写一条assert，验证输入K第0行被完整写到indices指定的缓存行。沿用08布局：indices=[5,1]，K输入(2,128)，K缓存(8,128)，FP16，缓存初值-1。

Python断言检查实际CUDA调用产生的结果；用户目前没有编写新CUDA内核。此次仅检查第一条K，其他K、V、未命中行和边界之后逐步补齐。

## 文件与准备

- Windows共享脚手架：上一级09-check-real-store.py和09-check-real-store.sh。
- 模板：上一级09-check-real-store-template.py，仅check_first_k函数留一条NotImplementedError占位。
- 用户运行--prepare后才会在WSL创建：/home/weltlaw/projects/mini-sglang-kv/experiments/check_real_store.py。
- 同名文件已存在时保留，不覆盖。助手没有直接访问或创建用户WSL文件。

在Windows PowerShell依次运行：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/09-check-real-store.sh --prepare
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\check_real_store.py'
```

准备应显示LEARNER_FILE_CREATED或EXISTING_FILE_KEPT，以及PREPARE_COMPLETE。准备不运行GPU。

## 用户任务与局部提示

将函数中的raise NotImplementedError(...)一行替换成自己的assert，保留函数体的四个空格缩进。只检查输入K第0行对应的完整目标行。先由用户选择要比较的两行。

- k_cache[...]表示缓存中的一整行；k[...]表示输入中的一整行。
- torch.equal(A,B)比较两个张量的形状和全部元素是否相同，返回布尔值。本次输入有限、dtype相同，适合精确复制检查；它本身不验证dtype，也不将NaN视为相等。
- assert 条件：条件为假时抛出AssertionError。
- 不先填写行号或给出完整断言；用户尝试后审查。

参考：https://docs.pytorch.org/docs/2.9/generated/torch.equal.html

保存后在PowerShell运行同一脚本，去掉--prepare。脚手架会快照学生代码和原版源码，检查提交/源码差异，运行原版内核，在用户断言执行前保存全部K/V数据到独立run目录的result.json。共享控制台日志保存到Windows的logs/first-store-check-*.log。

## 输出解释与验收

- NotImplementedError：占位行尚未替换，或编辑结果未保存到目标文件。
- AssertionError：断言为假。应检查用户选择的两行及原始结果，不能直接断定CUDA有错。
- LEARNER_CHECK_RETURNED：函数返回且无异常，只是运行状态；须审查用户断言才可认可覆盖，空函数或assert True不算有效检查。
- 返回用户编写的函数和控制台输出后继续；首条检查通过不等于完整正确性套件或性能结果。

本机准备验证：09的两个Python文件AST语法检查通过；shell脚本sh -n通过且为LF换行。未在用户WSL执行准备、GPU内核或断言。

## 09-30 21:15首次用户运行

用户实际执行09，日志logs/first-store-check-20260930-211554-bEsM2p.log已由教练读取核对。GPU调用后保存result.json成功，学生函数的运行快照第15行仍为原始raise NotImplementedError占位，因此停止。未收到有效断言代码，不能认定用户已写对、CUDA写入错误或编辑器保存失败。

当前先让用户在聊天中写出输入K第0行的断言，提示比较k_cache[目标行号]与k[0]，由用户根据indices=[5,1]选择目标行。审查后再指导替换占位行、Ctrl+S和运行；不先再次运行同一占位代码。

## 用户初稿与局部纠正

用户随后提交assert torch.equal(0,5)，解释输入第0行写入缓存第5行。行映射正确；参数需为行张量而不是整数。torch.equal(0,5)会因参数类型不符合而报错，不会检查缓存。

在收到尝试后，教练给出的纠正函数如下：

```python
import torch


def check_first_k(k_cache, k):
    assert torch.equal(k_cache[5], k[0])
```

此断言检查两行全部128个元素；不执行写入，不重新计算indices，也未覆盖其他行。须移除原raise占位，保存后再运行09。当前没有修改后输出；指导下的纠正不计独立实现，后续扩展或故意引入错误的检查由用户先尝试。

## 09-30 21:19第二次运行：转为检查磁盘文件

共享日志logs/first-store-check-20260930-211954-AtaWvi.log再次显示运行快照第15行原始raise占位；GPU调用与结果保存已完成。不能据此认定整个文件没有修改，只能确认执行路径仍到达占位。

当前只执行一次只读诊断：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /usr/bin/nl -ba /home/weltlaw/projects/mini-sglang-kv/experiments/check_real_store.py
```

待用户贴回完整文件内容，再判断占位未删除、修改未保存或修改了其他文件。暂停重复GPU运行，不因聊天转录中的反斜杠转义直接断言UNC路径错误。

## 读回结果：嵌套同名函数，尚未执行断言

用户贴回文件第10行是外层def check_first_k，第15行又缩进四格定义同名内层函数，第16行在内层执行正确的整行比较表达式。原raise已移除，说明当前实际文件确有修改；不能把此代码的新状态归因为未保存。该版本尚无运行结果。

外层被调用时只创建内层函数，随后返回；没有调用内层，断言不会执行。因此LEARNER_CHECK_RETURNED仅表示无异常的设计边界在此十分重要，不能声称通过了检查。

下一项明确替换整个练习文件为上方三个有效代码行（import、单个顶层def、缩进四格的assert），不是把整段def追加进已有函数。保存后用nl读回并复跑09，审查代码和运行结果。随后再安排用户先尝试错误输入/目标行的负例，验证断言能发现问题；本轮不同时追加任务。

## 09-30 21:24只重复准备：简化运行操作

用户运行--prepare，输出EXISTING_FILE_KEPT和PREPARE_COMPLETE，未运行GPU、未覆盖既有文件。嵌套函数是否已修正未知。共享日志first-store-check-20260930-212423-xN1s3G.log已核对。

教练增强普通运行日志：复制学生文件快照后，在SAVED_CHECK_CODE_BEGIN/END之间带行号打印该快照，后续实际执行同一快照。这样只需运行一次不带--prepare的09命令，就能审查代码和执行结果。--prepare行为不变，原版内核与学生断言不变；不额外要求用户单独运行nl。源码是否真正包含可执行断言仍须审查，LEARNER_CHECK_RETURNED不自动代表完整正确性通过。

## 09-30 21:27首条K检查通过

日志logs/first-store-check-20260930-212733-UCLXuw.log已由教练直接核对：实际执行快照为单个顶层函数，断言assert torch.equal(k_cache[5], k[0])位于函数体内；真实GPU运行后函数无异常返回。本次K输入第0行与缓存第5行全部128个元素相同。WSL结果目录为/home/weltlaw/projects/mini-sglang-kv/experiments/learning/first-store-check/run-20260930-212733-7tqax13i，raw结果保留，助手未直接读取WSL中的result.json。

这是一条指导下完成的检查；未覆盖另一条K、任何V、未命中行或边界。下一项为反例验证：由用户从0–7中选择非5的合法缓存行号，只替换左侧下标，右侧k[0]不变，预测不相等并运行同一脚本。预期AssertionError，说明断言拒绝这次错误比较；它不证明完整套件质量或CUDA故障。记录用户选择、理由和输出，随后恢复下标5并保存。

## 09-30 21:32错误配对反例达到预期

用户实际选择k_cache[1]与k[0]比较，日志logs/first-store-check-20260930-213229-ZijqII.log包含执行快照和AssertionError，教练已直接核对。这个结果符合反例预期；不是出现新的环境故障，也不是GPU结果被有意破坏后的测试。用户尚未解释缓存1实际对应谁，尚未确认恢复正确断言。

下一组任务：恢复已有首条正确K检查，在同一函数中独立添加第二条K整行检查。只提示indices=[5,1]和检查对象是另一个输入行，第二条完整断言由用户先写。保存并运行09，期望两条有效断言执行后无异常；贴回输出并解释为何刚才缓存1与输入0的配对不正确。当前不同时要求V、未命中行或benchmark。

09的日志范围文案随练习扩展为learner-written K checks，具体覆盖以保存的用户源码为准；不再固定声称只检查第一条K。该调整仅涉及日志文字，不改变CUDA调用、数据或用户代码。
