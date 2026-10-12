# 31号grouped完整约定回归：准备记录

日期：2026-10-10。入口已完成，独立审查及CPU辅助检查通过；GPU/WSL未运行。

新增31号Python/单行Shell入口与有界CPU检查，复用23号共同98配置、26号正例运行与完整底层存储校验、22号18种非法公共调用、29号grouped启动探针及30号已接受来源校验。没有改动26～30或学员内核。两臂为冻结4-warp单行v1和已接受4-warp四行grouped。

计划196项=168非空KV提交+28合法空返回、98配对，另有36项非法输入拒绝。覆盖C32、B0/1/2/3/4/5/17、D64/128/192、交错/分离及选定gap/guard；完整仅指既定集合。正例检查完整backing；非法项要求KV启动前拒绝、零KV提交、五个传入view值不变，不扩大成所有底层间隙都已验证。

来源快照保存真正导入的30→29→27/26等helper链、固定v1、实际grouped、检查与本课讲义。29号已接受源码/状态及current helper必须吻合；当前grouped哈希固定为 `5b190ce52e0f6c25f99b739346c2787f194ed7e75c759144cf25fcb06c8a1bd1`。新run独立创建，首错、当前用例与已有数据保留，不覆盖旧证据。

[CPU报告](20261010-m07-grouped-full-cpu.json)状态为 `CPU_FULL_GROUPED_BINDING_CONTROLS_PASS`：98个复用参考正例、168个模拟非空启动转发、28个空启动阻断、36个非法检查分发绑定通过；grouped/30源码改动被拒绝，意外RuntimeError不会被改记为正常合同拒绝并可保存首错。模拟分发不等于实际36项公共合同已通过，GPU内核与真实拒绝仍待用户运行。

独立审查者核对范围、两臂探针绑定、非法调用的异常种类、真实导入链与快照、完整数据检查及失败保留，没有阻断问题。主教练通读31运行路径并核对报告与以下最终SHA256一致；未重复旧CPU/GPU套件。

| 文件 | SHA256 |
| --- | --- |
| 31-check-kv-grouped-full.py | c966eee8eccb67488dd1c83411d641abee008eb0ef0458c769c3fb50d0b732eb |
| 31-check-kv-grouped-full.sh | 81c9a9612137095fd6035af0eb628823cdb5da3a3a55a52ef494aa1f2dbaff4d |
| 31-grouped-full-cpu-test.py | 00db4fd251b8db47c6661dbc35543bb42badce1f4066a1ee9f01a519eebcfd23 |

本人实际分工仍为：教练示范grouped三表达式后，本人应用并运行29/30；本轮回归脚手架由教练提供，不把查看结果记为独立手写。本课[讲义与模拟面试提醒](../lessons/m07-06-grouped-full-regression.md)已就绪。工程57/100保持，无新计时、模型/AMD结果或GitHub同步。
