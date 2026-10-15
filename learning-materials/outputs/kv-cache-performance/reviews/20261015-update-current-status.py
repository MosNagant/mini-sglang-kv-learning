"""Record completed 44–46 evidence in current entrances; preserve history."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
backup = root / 'history/20261015-before-host-path-record'
marker = '**10-15当前：调用层优化已验证，局部有效，模型未稳定获益。**'
checkpoint = marker + ''' [44–46完整结果](modules/M08/RESULTS-HOST-PATH-20261015.md)记录相同GPU内核下的有界元信息缓存：45号3新进程/360区间均值，CPU封装消融约6.6–6.8μs→2.6–2.7μs（约降60%）；46号1792次真实KV检查通过，18新进程/90次生成计时完整保留。Eager对旧版三轮耗时变化+0.976%、−5.740%、+0.019%，对原版均慢7.45%–17.28%；Graph也未稳定改善。正确性通过不等于模型提速，候选不晋升。

下一项优先把主要GEMV对应到真实调用/形状；若继续解释Eager退化，单独剖析真实提交路径。没有证据时不继续扫warp/group。实现、运行、审计由教练完成，全项目混合验收73/100及个人掌握状态保持；M05不重开。失败运行、全部样本、源码哈希及独立审计已保留，本次新增资料尚未同步GitHub。[今日独立日志](obsidian/2026-10-15-开发日志-M08调用层优化与归因.md) · [六路线性能与面试拆解](modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md)。

以下带日期条目为历史状态。
'''

def insert(relative, content, heading=None):
    path = root / relative
    text = path.read_text(encoding='utf-8-sig')
    if content.splitlines()[0] in text:
        print('KEPT', relative)
        return
    saved = backup / relative
    if saved.exists():
        raise RuntimeError(f'Refusing to overwrite backup: {saved}')
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, saved)
    lines = text.splitlines(keepends=True)
    idx = next(i for i,line in enumerate(lines) if (line.strip()==heading if heading else line.startswith('# ')))
    lines[idx+1:idx+1] = ['\n', content.rstrip()+'\n']
    result = ''.join(lines)
    if result.startswith('---\n'):
        front, body = result[4:].split('\n---', 1)
        front = front.replace('updated: 2026-10-14', 'updated: 2026-10-15')
        result = '---\n'+front+'\n---'+body
    path.write_text(result, encoding='utf-8')
    print('UPDATED', relative)

for relative in ['PROGRESS.md', 'LEARNING_LOG.md']:
    insert(relative, checkpoint)
insert('KV算子项目任务表.md', checkpoint, '## 现在在哪里')
insert('COACHING_GUIDE.md', '''**10-15教学接续：性能瓶颈归因→针对性优化→实验验证→面试表达。** 用户明确要求按已有实验诊断并推进有效优化；本次44–46由教练实现和执行，局部CPU约降60%，整模型未稳定提速，见[实际结果](modules/M08/RESULTS-HOST-PATH-20261015.md)。复习按“基础概念题＋机制追问＋项目证据题”逐层展开；先让本人解释再纠正，不把答案或教练代码算作独立掌握。原版默认、73/100保持，M05不重开。下一项根据真实profile映射主要调用；先区分观察与假设，不能把静态PTX或寄存器数量直接叫瓶颈。以下为历史约定。
''')
insert('M06-M07教学大纲-20261006.md', '''**10-15扩展课已执行：调用层成本与端到端收益。** [44–46结果](modules/M08/RESULTS-HOST-PATH-20261015.md)已保存完整实现、局部消融、严格正确性和三轮模型对照；CPU封装约降60%，模型未稳定获益。后续按[六路线诊断](modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md)逆向讲解，优先定位主要GEMV与真实提交开销，不要求先盲改更多参数。工程/个人掌握分开，总分73/100不变；下方M06/M07及10-14条目保留历史。
''')
insert('ROADMAP-TRITON-20261024.md', '''**10-15执行更新：真实模型对照已完成，调用层优化也已验证。** 10-14的[M08模型结果](modules/M08/RESULTS.md)和10-15的[44–46调用层结果](modules/M08/RESULTS-HOST-PATH-20261015.md)取代下方“M08未运行”的历史状态。本轮局部CPU收益约60%，整模型未稳定提速；默认原版、73/100及个人复习待办保持。下一次工程工作先映射实际主要GEMV/调用形状，或独立剖析真实提交路径；穿插按证据复盘。10-24作品/独立讲解与10-25演示仍是目标，不用日期或一次局部收益冒认全部完成。以下排期按当时状态留存。
''')
insert('obsidian/学习笔记索引.md', '''**10-15新增独立笔记：** [调用层优化开发日志](2026-10-15-开发日志-M08调用层优化与归因.md)记录44–46全部节点；[完整结果](../modules/M08/RESULTS-HOST-PATH-20261015.md)解释“CPU局部约60%改善，模型未稳定提速”；[六路线诊断与面试题](../modules/M08/PERFORMANCE-DIAGNOSIS-20261015.md)按事实/假设/缺失证据拆解整个工程。新笔记独立成文件，旧KV/M05/M06/M07笔记保留；不把教练产物当个人已掌握。
''')
insert('modules/M08/README.md', '''**2026-10-15更新：调用层优化已完成一轮验证。** 元信息缓存让CPU封装消融约降60%，真实模型没有稳定改善，默认原版保持。先读[本轮完整结果](RESULTS-HOST-PATH-20261015.md)、[整个项目的性能诊断与面试拆解](PERFORMANCE-DIAGNOSIS-20261015.md)，然后按[本轮复现入口](REPRODUCE-HOST-PATH-20261015.md)查原始证据。新增工作由教练完成，73/100和个人掌握状态不变；本次新增资料尚未同步GitHub。

| 新节点 | 产物 | 实际验收 |
| --- | --- | --- |
| 44 完整元信息缓存 | [新适配器](../../44-m08-kv-cached-metadata.py)、[CPU契约检查](../../reviews/20261015-m08-metadata-cache-cpu-check.json) | 同一40号GPU内核；完整签名、64项FIFO、不缓存旧指针 |
| 45 CPU消融与GPU检查 | [360区间均值](../../logs/m08-host-diagnostic-results/run-bqojhqoc/summary.json)、[独立审计](../../logs/m08-host-diagnostic-results/run-bqojhqoc/evidence-audit.json) | CPU路径六组约降60%；完整Graph与替换对象检查通过，不是模型或GPU加速 |
| 46 真实模型检查 | [结果](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/result.json)、[独立审计](../../logs/m08-host-path-check-results/run-20261015-033200-qh26bon6/host-path-evidence-audit-v2.json) | 两臂1792次KV检查；完整缓存、保存logits和token一致 |
| 46 三臂模型计时 | [90样本](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/result.json)、[独立审计](../../logs/m08-host-path-timing-results/run-20261015-033254-kxw4qv2v/host-path-evidence-audit-v2.json) | 18新进程；Eager/Graph均未稳定改善；候选不晋升 |

下方38–43号为10-14已验收的历史基础，原始数据与范围保留。
''')
insert('modules/M08/REPRODUCE.md', '''**10-15新增：** 44–46调用层候选、CPU消融与真实模型三臂对照见[独立复现说明](REPRODUCE-HOST-PATH-20261015.md)；本页保留38–43号原始复现入口。新增实际结果见[报告](RESULTS-HOST-PATH-20261015.md)，无需为复习重复GPU测试。
''')
