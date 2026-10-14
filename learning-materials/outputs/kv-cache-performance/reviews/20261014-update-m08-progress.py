"""Record the accepted, scoped M08 engineering evidence; no GPU/network."""
from pathlib import Path
import shutil

B = Path(__file__).resolve().parents[1]
headline = '''**10-14当前：M08真实模型接入与固定负载对照完成，尚无稳定提速。** [M08完整结果](modules/M08/RESULTS.md)保存Qwen3-0.6B/RTX4070Ti/FP16/TP1、单请求128输入/32生成的全部证据。41号三臂共2688次KV检查通过，Eager最终缓存、32token和三份logits精确一致；42号12新进程/60次计时及输出核验完成，Graph三轮候选耗时变化+3.152%、+1.300%、−0.836%，Eager全三轮更慢。候选真实执行，没有静默原版fallback。

43号独立Nsight跟踪记录896次KV内核、累计0.812245ms，占全部GPU内核累计时长0.781551%；不是整个generate墙钟关键路径占比。两类GEMV内核合计约83.3%的累计时长，下一项先映射真实调用/形状，再决定优化对象；尚无融合实现或模型明显优势。原版保持默认，M05—M07历史结论不改。

M08-01～05限定工程交付9/10，全项目混合验收73/100；M08-06本人独立解释待复习，不把教练实现/运行/报告计为本人掌握。Graph只核验实际生成token，尚无同等级Graph全缓存逐位复核；多请求/模型质量/AMD不在范围内。[今日独立日志](obsidian/2026-10-14-开发日志-M08真实模型接入.md) · [复现入口](modules/M08/REPRODUCE.md)。本轮新材料已保存在本地与运行归档，GitHub发布状态以新增回执为准。

以下日期条目保留历史状态，不是当前待办。
'''

paths = ['PROGRESS.md', 'LEARNING_LOG.md', 'KV算子项目任务表.md', 'COACHING_GUIDE.md', 'obsidian/学习笔记索引.md']
backup = B / 'history/20261014-before-m08-record'
for name in paths:
    src = B/name
    dst = backup/name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

for name in ('PROGRESS.md','LEARNING_LOG.md'):
    p=B/name
    text=p.read_text(encoding='utf-8')
    if '**10-14当前：M08真实模型' not in text:
        title,rest=text.split('\n',1)
        p.write_text(title+'\n\n'+headline+'\n'+rest.lstrip(),encoding='utf-8')

p=B/'KV算子项目任务表.md'
text=p.read_text(encoding='utf-8')
if '**10-14当前：M08真实模型' not in text:
    text=text.replace('## 现在在哪里\n','## 现在在哪里\n\n'+headline,1)
    lines=text.splitlines()
    updates={
       '全项目验收进度':'| 全项目验收进度 | **73/100分**；10-14新增M08-01～05工程9分，M08为9/10；混合权重不代表代码或个人掌握百分比 |',
       '已有实际运行':'| 已有实际运行 | M05—M07证据冻结；M08真实模型三臂正确性、12进程/60次对照及实际Nsight跟踪已完成 |',
       '最近一次完成':'| 最近一次完成 | 10-14 M08：2688次KV运行时检查、60次计时、896次KV trace；均独立核验；无稳定提速 |',
       '最近一次有效排错':'| 最近一次有效排错 | 43号诊断worker更名解决标准库profile遮蔽，旧失败保存；本轮WSL/GPU已实际运行 |',
       '尚缺的核心交付':'| 尚缺的核心交付 | 本人独立分析/串讲、M09最终作品与复现验收；Graph全缓存复核、融合及多请求尚未实现/验收 |',
    }
    for i,line in enumerate(lines):
        for label,new in updates.items():
            if line.startswith('| '+label+' |'):
                lines[i]=new
        if line.startswith('| M08 | 模型接入'):
            lines[i]='| M08 | 模型接入与端到端对照 | 10 | 9 | 10-14限定工程完成；个人解释待复习 |'
        if line.startswith('| M08-0'):
            cells=line.split('|')
            n=int(cells[1].strip()[-1])
            if n<=5:
                cells[2]=' ☑ 限定工程完成 '
                evidence={1:'39号真实权重原版生成已完成；环境/模型revision/输出保存。',2:'39/41实测prefill B128、decode B1、D1024，源4096/缓存1024行距。',3:'40号教练实现真实布局适配；41/42实际调用m08_store_rows，无fallback；Graph仅尺寸1。',4:'41原版重复/原版/候选精确一致；Eager缓存、token及3份logits通过，Graph输出token另查，非全步logits/Graph全缓存验收。',5:'42号12新进程/60次计时和输出核验；Graph无稳定收益、Eager全慢；全部数据保存。'}[n]
                cells[4]=' '+evidence+' 详见[限定结果](modules/M08/RESULTS.md)。 '
                cells[5]=' 教练实现/执行/独立审计；不计本人独立掌握 '
            else:
                cells[2]=' ◐ 证据齐，个人解释待复习 '
                cells[4]=' 43实测KV累计时长份额0.782%，报告已区分墙钟；本人独立解释局部与整机影响仍待记录，不因教练报告加分。 '
            lines[i]='|'.join(cells)
    text='\n'.join(lines)+'\n'
    text=text.replace('无任意布局、运行时非法读检测、模型或AMD性能结果。','该段仅列M06/M07范围：无任意布局、运行时非法读检测或AMD性能结果；新增M08固定模型对照范围见顶部及M08报告。',1)
    text=text.replace('## 当前冲刺排期与复盘（10月10日更新）','## 历史冲刺排期与复盘（10月10日制定；实际进度以上方10月14日为准）')
    p.write_text(text,encoding='utf-8')

p=B/'COACHING_GUIDE.md'
text=p.read_text(encoding='utf-8')
note='**2026-10-14 M08接续：** 用户要求开始真实模型并争取明显优势。本轮教练完成环境/适配/诊断/对照，限定工程交付9/10；真实结果未建立优势，保留原版和全部负结果。现在围绕[M08证据](modules/M08/RESULTS.md)逆向复习，个人解释未发生，不新增口试阻挡已完成工程。下一项先根据真实profile映射主要GEMV与KV调用，再决定融合或扩展优化对象；不能承诺“测到优势为止”。\n'
if '**2026-10-14 M08接续：**' not in text:
    title,rest=text.split('\n',1)
    p.write_text(title+'\n\n'+note+'\n'+rest.lstrip(),encoding='utf-8')

p=B/'obsidian/学习笔记索引.md'
text=p.read_text(encoding='utf-8')
note='**10-14新增M08：** [真实模型独立开发日志](2026-10-14-开发日志-M08真实模型接入.md)与[完整结果](../modules/M08/RESULTS.md)补充真实D/stride、模型正确性、12进程/60次对照和KV实际份额。结论为接入通过、尚无稳定提速；与下方早先M07日志分文件保存。\n'
if '**10-14新增M08：**' not in text:
    text=text.replace('# KV算子学习笔记索引\n','# KV算子学习笔记索引\n\n'+note,1)
    p.write_text(text,encoding='utf-8')
print('M08_PROGRESS_RECORDED; old documents preserved; no personal-mastery credit')
