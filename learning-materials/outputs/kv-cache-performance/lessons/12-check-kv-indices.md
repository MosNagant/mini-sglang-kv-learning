# 同一份目标列表，驱动搬运和结果检查

2026-10-02。范围：固定FP16、batch=2、width=128、capacity=8、输入stride=(128,1)、K/V交错缓存视图stride=(256,1)。这是多索引正确性实验，不是计时或通用算子验收。

## 谁负责什么

1. 测试程序的Python列表target_rows指定本次目标，例如[7,3]。
2. 测试程序用同一列表创建int32的GPU indices。
3. 昨天的Triton内核已经通过tl.load(indices_ptr + pid)读取当前输入行的目标，并执行load/store。
4. 搬运完成后，用户的循环根据target_rows比较目标缓存行与运行前保留的期望输入行。

这次新增的是自动跟随目标列表的检查，不是补一个内核之前没有的索引读取动作。下列语句验证结果，不搬运数据：

```python
for pid, pos in enumerate(target_rows):
    assert torch.equal(k_cache[pos], expected_k[pid])
    assert torch.equal(v_cache[pos], expected_v[pid])
```

两条断言由用户在指导后提交；教练只整理换行、加函数封装/import，并组织运行。运行时检查使用CPU结果快照，expected_k/v独立存在于GPU启动之前，避免把可能被改动的输入当作正确答案。

## 一次运行

Windows PowerShell：

```powershell
wsl -d CUDA-Ubuntu-24.04 -u weltlaw --exec bash /mnt/c/Users/Administrator/Documents/Codex/2026-09-27/cuda-wsl-wsl-2/outputs/kv-cache-performance/12-check-kv-indices.sh
```

入口会将已审查的用户检查保存到/home/weltlaw/projects/mini-sglang-kv/experiments/check_kv_targets.py；如果已有文件则保留，打印实际快照并运行它。继续读取用户已有的experiments/triton_store_first.py，不改写内核，不替换为教练参考内核。

运行前确认固定提交及原版kernel目录未修改，并核对导入的store_cache来自该项目。对原版CUDA、本人Triton分别以全新输入/缓存运行[5,1]、[7,3]、[7,0]，共六组。

目标整行由用户的断言检查；未命中行复用此前逻辑，另检查只读输入K/V和indices未变。运行前的CPU对照会接受正确样本，并要求拒绝两行K/V末列损坏及交换目标配对，共五个故意错误。此处反例不修改GPU内核。

## 看什么输出

- CHECKER_CONTROLS_PASS：CPU上的正确/故意错误样本按预期被接受/拒绝。
- 每个CASE_PASS只代表标出的实现和索引对通过。
- 全部六组通过并归档成功后才输出KV_INDEX_CASES_PASS。
- 任何错误保留首条错误、源码、已取得的全量结果和status.json；失败的运行也尝试复制到共享目录。

源码/结果主目录：项目experiments/learning/kv-index-check/run-*。Windows共享副本：outputs/kv-cache-performance/logs/kv-index-check-results/run-*；具体路径看SHARED_ARTIFACTS。控制台另有独立日志。

准备阶段的教练侧静态语法与NumPy替身验证见verification/12-checks-local-result.json，不是GPU实测。随后用户已于10-02 12:46在RTX 4070 Ti实际运行，两种实现共6/6通过，五个CPU故意错误被拒绝。共享原始目录为logs/kv-index-check-results/run-20261002-124631-acw5vrex，教练对保存数据和22个源码哈希的独立复核见reviews/20261002-kv-index-check-review.json。任意形状、非连续输入、重复/越界索引及性能仍不在这次验收范围。
