# M06-03：先区分有效宽度、逻辑块和行距

2026-10-07 18:11验收：用户运行21号，run-20261007-181154-5q2k4kbh实际GPU 28/28通过，完整存储与27项源码哈希复核一致；见[mask验收](../reviews/20261007-m06-03-mask-review.md)。M06-03交付4/4，M06累计11/25，总41/100。核心由教练按用户请求示范、用户实际运行，个人独立掌握另记；没有性能测量。下一课M06-04调用接口。

## 从当前内核出发

上一版列下标由tl.arange(0, WIDTH)生成，D64/128已经检查。本课mask版在一行192个有效元素时，使用256个逻辑列位置来表达这一行的工作，并已通过限定范围GPU检查。BLOCK_SIZE描述逻辑元素位置数量，不代表256个线程，也不把实际张量宽度或行距改成256。

| 量 | 本课示例值 | 负责什么 |
| --- | ---: | --- |
| WIDTH | 192 | 这条K或V真正有多少有效元素 |
| BLOCK_SIZE | 256 | 本program生成多少个逻辑列位置 |
| SRC_ROW_STRIDE | 从张量读取 | 下一条输入行的实际起点距离 |
| DST_ROW_STRIDE | 从张量读取 | 下一条缓存K行或V行的实际起点距离 |

列0到191共192列有效；192到255共64列应禁止读写。mask就是逐列的允许/禁止条件，用于K/V的load与store。被禁止的位置不访问相应内存；不等于把缓存相应位置写成0。该机制对应[Triton官方向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)的mask用法。

这64个多余逻辑位置并不自动变成input_gap：gap是实际存储布局的额外间距，逻辑块只是代码生成的索引范围。是否存在gap仍由真实张量决定。地址公式仍为pid*源行距+cols及pos*目标行距+cols，不用BLOCK_SIZE替代行距。

## 本次已按请求完成的改动

共享示范源码：[21-triton-store-mask.py](../21-triton-store-mask.py)。实际新WSL文件为/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_mask.py；函数名保持store_kv_v1，由21号专用启动器加载。旧triton_store_v1.py及14号检查保留。

~~~python
# 函数参数新增 BLOCK_SIZE: tl.constexpr
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < WIDTH
src_offsets = pid * SRC_ROW_STRIDE + cols
dst_offsets = pos * DST_ROW_STRIDE + cols

k_values = tl.load(k_ptr + src_offsets, mask=mask, other=0)
tl.store(k_cache_ptr + dst_offsets, k_values, mask=mask)
v_values = tl.load(v_ptr + src_offsets, mask=mask, other=0)
tl.store(v_cache_ptr + dst_offsets, v_values, mask=mask)
~~~

调用方为D64/128/192分别传BLOCK_SIZE64/128/256，其余stride仍从有效视图读取。mask为false的load位置用0作局部占位，不写缓存；store必须同样带mask。indices的标量读取不套用列mask，因为本课grid精确为B3。

## 运行一次：自动创建新文件并测试

在自己的Windows PowerShell运行：

~~~powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/21-check-kv-mask.sh
~~~

入口只在新文件不存在时创建；已有内容会保留并在GPU前审查。无需打开记事本或修改旧内核。本次实际末尾已显示MASK_CASES_PASS:28/28，且保存结果已核验；无需为了验收重复运行。日志与全量数据自动保存到logs/m06-mask-results下的唯一run目录。失败结果不覆盖，成功的CPU模拟不算GPU证据。

| 检查组 | 配置 | 两实现合计 |
| --- | --- | ---: |
| 旧范围回归 | D64/128×两布局×input_gap0/64 | 16 |
| 非2幂宽度 | D192×两布局×input_gap0/64 | 8 |
| 行尾guard | D192、input_gap64、cache_guard64×两布局 | 4 |

固定FP16、B3、C8、GPU int32唯一合法indices[7,0,3]；K/V各侧共享行距，列距1。每条cache guard有64个实际分配的元素，填-7；输入gap填-9，有效缓存初始-1，检查完整底层。guard交错源行距256、目标512；分离源256、目标256。输入与缓存传入的是有效192列视图。

完整结果与guard检查可以发现额外写入；只读数据未改变不能证明没有额外读取。两个load的mask依赖保存源码审查，本次没有运行时非法读取检测器，也不运行去掉mask的坏GPU版本。CPU负控制及独立本地语义检查另记为辅助证据。

本地准备验证：14个CPU正例、122个损坏反例、28次运行器模拟、错误留档/安装保留检查通过；上述是准备阶段历史；随后用户实际GPU 28/28通过。准备时来源和限制见[准备审查](../reviews/20261007-m06-03-preparation-review.md)。

选择192的固定原版源码依据见[大纲第3课](../M06-M07教学大纲-20261006.md)；本次限定范围已GPU通过。新增192用于正确性，不扩大M05冻结性能矩阵。

## 面试知识提醒

来源：教练设计的项目迁移题，不是保证命中的面经原题。

- 概念及代码：有效列、逻辑块、内存访问边界，对应cols/mask/load/store。
- 基础问法：D192而逻辑块256，为什么需要mask？
- 变式追问：只给store加mask，load不加，会有什么问题？
- 易错说法：把BLOCK_SIZE当线程数；认为mask为false就向缓存写0；用256替换真实row stride；把多出的逻辑列当成实际gap。
- 目标深度：会写列边界条件并解释四个读写的保护；具体线程映射暂了解，后续结合生成代码。
- 当前本人状态：理解在提示后推进；本次核心由教练按明确请求代改，不认定独立手写。GPU验收与本人掌握分别记录。
