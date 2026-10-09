# M06-01：同一份内核支持64和128列

2026-10-06。M05原版测量与trace工程证据已齐，下一步是实现可配置Triton v1；这一课不宣称提速。

## 这次要交付什么

同一份源码，在一次检查中通过原版/Triton × D64/D128共4个用例。旧first内核保留；新文件已准备在：

/home/weltlaw/projects/mini-sglang-kv/experiments/triton_store_v1.py

PowerShell打开：

```powershell
notepad.exe '\\wsl$\CUDA-Ubuntu-24.04\home\weltlaw\projects\mini-sglang-kv\experiments\triton_store_v1.py'
```

## 为什么要改

旧内核把“每行多少数”“跨到下一输入行多远”“跨到下一缓存行多远”写死为64、64、128。外部输入变了，内核仍按旧间距寻址，就会重复之前宽度不匹配的错误。现在启动器把实际张量的信息传给内核，源码不随测试宽度手动修改。

| 参数 | 通俗含义 | 启动器从哪里读 |
| --- | --- | --- |
| WIDTH | 每行真正要搬多少个数 | k.shape[1] |
| SRC_ROW_STRIDE | 从输入某行开头跨到下一行开头，要走多少个元素 | k.stride(0) |
| DST_ROW_STRIDE | 从缓存K视图某行开头跨到下一行开头，要走多少个元素 | k_cache.stride(0) |

三个参数本课标为tl.constexpr：编译器为本次确定的参数生成代码。64和128可以产生不同特化版本，“同一份源码”不等于“相同机器码”。本课K/V输入各自的行距必须相等，两缓存视图的行距也必须相等；运行器会检查。全部偏移仍按元素数，无须手动乘FP16的2字节。

## 本人要改的三行

找到pass，删除它，自己填入定义cols、src_offsets、dst_offsets的三行。思路沿用已经解释过的“行编号×实际行距+列下标”。输入行由pid选，缓存行由indices中读出的pos选。后面的四行load/store无需改变。

可以从下面的空格稿开始，所有行缩进四个空格；不要把省略号保留到正式运行：

```python
cols = tl.arange(0, ___)
src_offsets = ___ * ___ + cols
dst_offsets = ___ * ___ + cols
```

先提交这三行供审查，再保存运行。没有必要重写整个数据准备、日志或正确性框架；这些外围来自教练。此前本人指导下四行搬运实现继续复用，不冒称独立从零重建。

## 保存后怎么验收

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/14-check-kv-widths.sh
```

预期打印KV_WIDTH_CASES_PASS: 4/4 cases。实际必须看源码快照和全部检查，而不仅看最后一行。范围限定B3、容量8、FP16、int32索引[7,0,3]、连续输入、交错缓存、D64/128；包含目标K/V、未命中行、只读输入检查。M06-02才扩布局与行间空隙，M06-03补非2幂宽度和mask。

当前状态：模板及启动器完成；Python语法与13项本地静态/模拟控制通过；用户20号已创建草稿，尚未提交三行，也未执行这份可配置内核的GPU检查。
