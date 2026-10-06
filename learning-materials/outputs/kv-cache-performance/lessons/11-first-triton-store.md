# 第一份 Triton KV 写入：读写核心由你完成

这一步用已有小例验证本人内核。目的：从“知道地址”走到“在GPU上实际读写”。仍使用两行输入、每行128个float16、容量8的交错K/V缓存、indices=[5,1]。当前固定形状的索引全部合法，启动恰好两个program；不支持直接改宽度/布局，不声称有完整尾部处理或最优性能。

## 创建和打开

**先看终端提示符：** `PS C:\...>`是Windows PowerShell；`(.venv) weltlaw@...$`是Ubuntu。下方两套命令按当前提示符选一套，不把`/mnt/c/.../notepad.exe`直接输入PowerShell。16:42日志已确认文件创建成功，当前只需打开，不必重复准备。

如果当前是PowerShell，打开已创建文件用：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_first.py'
```

在用户自己的Ubuntu终端执行，脚本保留已有同名文件：

```bash
bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/11-triton-store.sh --prepare
/mnt/c/Windows/System32/notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_first.py'
```

出现PREPARE_COMPLETE只表示准备完成，没有运行GPU。用户文件是`/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_first.py`。此前`kv_write_lab.py`保持原样。

## 你写的部分

模板保留经讨论的pid、cols、pos和两行元素偏移。找到LEARNER_READ_WRITE_BEGIN，把pass替换为四行，仍在函数内缩进四个空格：

1. 读取输入K，存入变量k_values。
2. 读取输入V，存入变量v_values。
3. 将k_values写入缓存K。
4. 将v_values写入缓存V。

`tl.load(读取地址)`读出数据；`tl.store(写入地址, 数据)`写入。这里括号内中文是说明，不是可直接粘贴运行的代码。读取或写入地址由实际指针加对应的元素偏移组成。

| 指针 | 指向什么 | 对应偏移 |
| --- | --- | --- |
| k_ptr | 输入K的开头 | src_offsets |
| v_ptr | 输入V的开头 | src_offsets |
| k_cache_ptr | 缓存K视图的开头 | dst_offsets |
| v_cache_ptr | 缓存V视图的开头 | dst_offsets |

K/V指针已经分别指向各自开头，V不需要再额外加128。偏移按元素计数，不手动再乘2。cols的128个位置表示一组逻辑元素，不等于要求某个线程串行处理128次。

先贴四行代码审查，再保存运行。不要新增嵌套def或把终端命令写进源文件。

## 审查后的运行入口

```bash
bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/11-triton-store.sh
```

脚手架只调用学生Triton内核，不调用原版store代替它。保存学生源码快照、运行脚手架、输入/环境，以及完成GPU执行后的全量原始结果，再执行此前的目标K/V整行检查、未命中位置检查；教练额外检查输入和索引没有被改写。期望出现FIRST_TRITON_CASE_PASS。该标记只适用于当前固定小例，不是通用算子、完整测试套件或性能基线通过。

若出现LEARNER_CODE_PENDING，说明pass还未替换，未发起GPU内核；编译报错或AssertionError则保留首次错误和日志，提交代码审查。失败也保留源码与status/error文件；GPU未完成时可能没有result.json，不能假定输出已可用。

## 工具验证范围

教练已检查Python语法、UTF-8/LF、shell语法，并在Windows CPU环境确认未完成草稿会在导入GPU依赖前停止并留下失败记录。验证记录位于`verification/first-triton-scaffold-x13b9i9j`。未直接访问用户WSL、未编译或运行Triton GPU代码。

参考：[Triton load](https://triton-lang.org/main/python-api/generated/triton.language.load.html)、[store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)。2026-10-01核对；本机版本和实际行为以用户运行输出为准。
