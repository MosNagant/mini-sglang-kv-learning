# L01a：把一组 K 写入指定缓存行

2026-09-29 制定，2026-09-30 更新。前置已通过：用户能在记事本修改 WSL first_kv.py 并保存，用 Ubuntu 项目 Python 运行。当前单例输出已符合预期：编号2行等于K，其余行零，见 [用户成功输出](../logs/first-indexed-write-pass-user-output-20260930.txt)。初稿漏赋值，教练在用户运行并贴源码后提供了一行局部示范；用户按示范完成，独立理解仍待验证。

当前理解检查：把slot改为1后从头运行文件，哪行收到K；为什么还要看另外三行；删除写入赋值而保留slot和print会输出什么。尚无回答。通过后接V和多个输入，不继续重复同一数字修改实验。

## 目的与边界

建立 store 操作的逻辑契约：输入数字放进目标行，其他行保持不变。此次只处理一组 K，V 随后用相同目标行练习；不使用 GPU，不计时。Python 的嵌套列表仅模拟逻辑槽位，不表示真实 KV cache 的连续内存、数据类型字节数或 GPU 写入方式。

四个槽位编号为 0、1、2、3，每行四个数。slot=2 表示第三行。k 使用 first_kv.py 中已经保存的 [99, 11, 12, 13]。

## 原练习步骤（保留作为复现说明）

在记事本保留 first_kv.py 原四行，在末尾追加下面的脚手架。用户在 TODO 下方亲手补一条赋值语句；助手不得提前填上这行。

```python
k_cache = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
slot = 2

# TODO: 把 k 的一份副本放进 k_cache 的 slot 行

print("K cache:", k_cache)
```

局部提示：`k_cache[slot]` 表示目标行；`k.copy()` 可复制当前这个数字列表。只提示表达式含义，赋值语句由用户连接。这里的 copy 对平坦数字列表足够，不延伸成一般深复制结论。

Ctrl+S 后在 PowerShell 运行同一条命令：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec /home/weltlaw/projects/mini-sglang-kv/.venv/bin/python /home/weltlaw/projects/mini-sglang-kv/experiments/first_kv.py
```

完成标准：除了原 K/V 输出，多一条 K cache；编号 2 的行为 [99, 11, 12, 13]，编号 0、1、3 的行仍全零。用户贴回亲手补的一行与实际输出。若没有写入行，输出会全零；若 NameError: k，检查是否误删原四行；语法错误先看报错指向的行。不要声称该观察等于正式正确性覆盖。

反馈后再决定下一步：检查目的行与输入行的关系，扩展 V 和两组输入，随后迁移到真实 Tensor 的 shape/stride。正式数据契约里输入 k/v 是 [B,D]；当前一组列表只是 B=1 的简化。
