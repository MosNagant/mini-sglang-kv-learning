"""Promote completed47 diagnostic to current record without changing old evidence."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
backup=root/'history/20261015-before-linear-attribution-record'

def insert(name, text, heading=None):
    p=root/name
    original=p.read_text(encoding='utf-8')
    if text.splitlines()[0] in original:
        print('KEPT',name); return
    dest=backup/name
    if dest.exists(): raise RuntimeError('Backup already exists: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    lines=original.splitlines(keepends=True)
    i=next(i for i,line in enumerate(lines) if (line.strip()==heading if heading else line.startswith('# ')))
    lines[i+1:i+1]=['\n',text.rstrip()+'\n']
    p.write_text(''.join(lines),encoding='utf-8')
    print('UPDATED',name)

checkpoint='''**10-15最新：主要线性层归因已实测完成。** [47号完整报告](modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)记录原版模型一次独立Eager诊断：3616次F.linear/唯一NVTX/对应GPU内核全部关联，32生成token精确一致。decode Gate/Up与LMHead分别占本次全部GPU内核累计时长23.534%/20.975%，KV为0.841%；不是墙钟占比，也不移用为旧Graph数据。

下一项优先固定decode Gate/Up的实际[1,1024]×[6144,1024]权重转置运算建立GEMV对照，区分单权重复用与轮换28层权重，再据证据选择候选；尚未实现GEMV优化或得到新提速。原版默认、73/100和个人掌握状态保持。教练完成诊断及归档，[独立开发日志](obsidian/2026-10-15-开发日志-M08主要计算路径归因.md)供逆向复习；新增资料尚未同步GitHub。下方44–46及更早日期为此前阶段。
'''
for name in ['PROGRESS.md','LEARNING_LOG.md']:
    insert(name,checkpoint)
insert('KV算子项目任务表.md',checkpoint,'## 现在在哪里')
insert('modules/M08/README.md','''**2026-10-15最新节点：47号线性层归因通过。** [完整报告](RESULTS-LINEAR-ATTRIBUTION-20261015.md)将3616次实际调用对应到模型层、shape和GPU内核；原版32token一致。主要成本为decode Gate/Up和LMHead，本次Eager内核时长和占比23.534%/20.975%，KV0.841%。下一项据此建立Gate/Up GEMV对照；本次没有修改数学计算或新增提速结论。报告包含所有原始数据、独立审计、面试题与单行复现。默认后端与验收分保持。
''')
insert('obsidian/学习笔记索引.md','''**10-15最新独立笔记：** [主要计算路径归因日志](2026-10-15-开发日志-M08主要计算路径归因.md)和[47号完整报告](../modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)回答“主要GEMV具体是哪一层、什么矩阵形状”。3616次实际调用全部关联；这是下一项优化的依据，不是新提速。与下方CPU调用层日志分文件保存。
''')
insert('COACHING_GUIDE.md','''**10-15最新复习节点：** [47号实际归因](modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)已完成，不能再把主要GEMV的调用映射说成尚未开始。重点让本人解释F.linear的X/W形状、prefill与decode、CPU提交与GPU执行的相关性，以及为什么内核名不能唯一代表模型层。下一项为实际Gate/Up GEMV对照；没有新算法或加速结果，勿把诊断通过计为优化胜利。以下保留前序教学安排。
''')
insert('M06-M07教学大纲-20261006.md','''**10-15最新扩展：** [47号线性层归因](modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)已实际完成；3616次调用对应具体形状与GPU内核。下一单元从主要decode Gate/Up建立GEMV参照，补充归约与数值验证方法；是否实现融合取决于证据，不当作已完成。个人解释与工程证据分开，73/100保持。
''')
insert('ROADMAP-TRITON-20261024.md','''**10-15最新执行节点：** 47号已完成[主要GEMV的实际调用/形状归因](modules/M08/RESULTS-LINEAR-ATTRIBUTION-20261015.md)。下一项固定decode Gate/Up建立真实输入的GEMV参照，保留原FP16语义和独立数值验收，区分重复单权重与轮换28层权重；候选和整模型新收益尚未发生。已有KV版本/模型实验可以用于复习，不因新增研究拖延既有材料的讲解。以下为前序排期。
''')
insert('modules/M08/REPRODUCE.md','''**10-15最新：** [47号主要线性层归因报告](RESULTS-LINEAR-ATTRIBUTION-20261015.md)末尾提供单行复现及原始Nsight/SQLite/逐调用记录入口。它是独立Eager诊断，不并入42/46正式计时样本。
''')
insert('modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md','''**后续实测更新：** [47号归因结果](RESULTS-LINEAR-ATTRIBUTION-20261015.md)已将主要GEMV对应到实际模型层和矩阵形状，3616次调用全部关联；接续44–46的CPU局部优化与模型负结果。下文“主要GEMV待映射”是47运行之前的调查计划，当前下一项为Gate/Up GEMV对照；尚无新的GPU或模型提速。
''')
