"""Reconcile current documentation; preserve historical entries and backups."""
from pathlib import Path
import shutil

base = Path(__file__).resolve().parents[1]
backup = base / 'history' / '20261009-before-m06-closeout'
backup.mkdir(parents=True, exist_ok=True)

def read(name):
    path = base / name
    if not (backup / name).exists():
        shutil.copyfile(path, backup / name)
    return path.read_text(encoding='utf-8-sig')

def write(name, value):
    (base / name).write_text(value, encoding='utf-8')

current = ('**10-09当前：M06工程证据已汇总，v1已冻结，开始M07。** '
           'M06-05共同正确性196结果（168非空调用+28空返回）、M06-07实际编译产物观察、M06-06六配置×三新进程同场对照均已审查。'
           '现有数据未证明v1稳定快于原版。新增[M06独立归档区](modules/M06/README.md)，含源码、设计、完整证据及成绩。'
           'M06累计23/25，总工程53/100；M06-08产物冻结完成、本人独立串讲待记录，2分暂不计。'
           'M05保持完成；不把复习变成重做基线。下一课[M07启动配置假设](lessons/m07-01-num-warps-hypothesis.md)。'
           '本人初步预测“更快，因为一个warp工作量不算多”，针对num_warps由4改1；尚未修改候选或取得新GPU结果，M07暂0分。')

name = 'PROGRESS.md'
s = read(name)
start = s.index('## 2026-10-09 当前接续')
end = s.index('## 2026-10-07 最近实际验收')
old = s[start:end].replace('## 2026-10-09 当前接续', '## 2026-10-09 早先断点（历史，已由上方收尾取代）', 1)
s = s[:start] + '## 2026-10-09 当前接续\n\n' + current + '\n\n本次仅离线整理、文件哈希核验与统计复算，没有重跑GPU；未自动同步GitHub或WSL。\n\n' + old + s[end:]
write(name, s)

name = 'LEARNING_LOG.md'
s = read(name)
anchor = '**10-09 M06-05准备并下发：**'
addition = '''**10-09 M06汇总冻结、M07开始：** 用户要求总结日志并把M06整理为独立区块。本次建立[modules/M06](modules/M06/README.md)，保存七阶段完整证据（省略__pycache__）、冻结内核/封装/检查/计时快照、运行器依赖、设计与复现说明、完整Graph/Eager成绩、文件哈希清单与ZIP。原目录和失败记录保留。只在Windows共享工作区生成，不声称WSL安装或GitHub同步。M06-08文档冻结已完成，个人独立串讲仍待记录，整格暂不计2分；M06累计23/25，总53/100。本人已提出M07初稿“更快，因为一个warp工作量不算多”，针对num_warps4→1；教练补充减少warp也可能降低并行度。两重点配置的分别预测和否定条件继续随课补齐；未代写候选改动或新GPU成绩，M07为进行中未计分。见[M07首课](lessons/m07-01-num-warps-hypothesis.md)。实际用时未记录。

**10-09 14:51 M06-06首次同场对照验收：** 用户运行25号，run-20261009-145144-n1lwky52。六配置×三新进程、两实现×Graph/Eager×30个100调用区间，共2160区间平均样本；全部原始样本、完整数组及源码已复核。打包时再次从18份result.json复算中位数/比值，与summary一致。Graph六个代表速度比为0.9486、0.9607、1.0283、0.9481、1.0531、0.8940；B256/D64虽代表值大于1，三轮仅一轮更快，不能宣称稳定提速。Eager全部配对v1更慢，不能把主机提交路径差异都归因于GPU内核。记录范围为C1024、B1/16/256、D64/128、交错FP16、gap0、连续唯一索引、重复地址；无DRAM/model/AMD收益结论。M06-06工程3分；计时组织和统计为教练辅助。见[M06成绩](modules/M06/RESULTS.md)。

**10-09 14:27 M06-07编译观察验收：** 用户运行24号，run-20261009-142759-l12p0a0c。一份B3/D192、C32、分离缓存、源gap64/缓存guard64的实际Triton调用通过；grid3，源/目标行距256，BLOCK256、num_warps4。保留TTIR/TTGIR/LLVM IR/PTX/cubin及编译元信息。允许复用编译缓存，未证明发生新编译；静态18寄存器/线程、shared0不是运行时硬件利用率。M06-07工程证据3分；查看产物不等于编译器开发或个人独立掌握。

**10-09 14:09 M06-05共同正确性验收：** 用户运行23号，run-20261009-140928-ht8v9mg8。98配置×两实现共196检查结果：168非空GPU调用+28空返回；C32、B0/1/2/3/4/5/17、D64/128/192、选定布局/gap/guard。两实现分别建立独立期望，检查目标、未命中存储、缓存guard、完整输入及gap、indices；控制夹具98正例/908损坏反例。保存结果与源码核验通过，M06-05记3分。本轮沿用本人检查和已验收内核/公共入口，矩阵与外围记录由教练提供。空返回不称GPU执行，有限矩阵不称任意布局支持。后续记录原始顺序为先编译观察、再同场计时。

以下为保留的早先过程；“待运行”只描述各条当时状态。

'''
assert anchor in s
s = s.replace(anchor, addition + anchor, 1)
write(name, s)

name = 'COACHING_GUIDE.md'
s = read(name)
a = s.index('项目主线：')
b = s.index('\n\n## 2.', a)
s = s[:a] + '项目主线：mini-sglang KV写入的Triton实现与性能分析。原版CUDA作参照，Triton为主要动手对象。M05完成并冻结；M06-01～07工程证据已验收，M06 23/25、总53/100；M06-08冻结/设计产物已整理，独立串讲待记录。首次同场计时未证明稳定提速。当前进入M07，用户提出num_warps4→1可能更快的初步预测；分别预测/否定条件、核心改动与新运行仍待完成。详见[M06归档](modules/M06/README.md)和[M07首课](lessons/m07-01-num-warps-hypothesis.md)。个人独立能力与工程证据分别记录。' + s[b:]
write(name, s)

name = 'M06-M07教学大纲-20261006.md'
s = read(name)
s = s.replace('当前只布置第一课，其余课程按前一课结果逐步展开。', '当前推进M07第一课，其余课程按实际证据展开。')
a = s.index('**M05已经完成')
b = s.index('\n\n现在要把', a)
s = s[:a] + current + s[b:]
a = s.index('## 本次状态与下一步')
b = s.index('## 阅读依据', a)
s = s[:a] + '''## 本次状态与下一步

- [x] M06-01～04实现、布局/mask与公共入口验收。
- [x] M06-05：共同正确性196结果通过。
- [x] M06-07：实际调用及编译产物留存、审查。
- [x] M06-06：六配置×三新进程，原版/v1同场Graph/Eager结果审查。
- [x] M06-08的工程产物：v1源码、设计、成绩及原始证据打包。
- [ ] M06-08的本人独立串讲尚未记录，2分暂不计；累计M06 23/25、全项目53/100。
- [ ] M07-01/02已开始：用户预测num_warps4→1可能更快；尚未形成完整两配置预测/否定条件，未修改/运行v2。

下一步使用[M07首课](lessons/m07-01-num-warps-hypothesis.md)，先解释“减少每个program的warp数，program数量仍由B决定”，再完善假设并进行单因素修改。M06版本保持冻结，M05不重做。没有稳定提速或模型收益结论。

''' + s[b:]
write(name, s)

name = 'KV算子项目任务表.md'
s = read(name)
a = s.index('**10-09当前待运行：**')
b = s.index('**教学方式历史', a)
s = s[:a] + current + '\n\n' + s[b:]
s = s.replace('**44/100分**；M01–M05及M06-01/02/03/04交付完成，个人独立解释能力另记', '**53/100分**；M01–M05及M06-01～07工程证据验收；M06-08独立串讲待记录')
s = s.replace('原版与mask版D64/128/192、两布局、输入gap及尾部guard，共28例GPU通过', '共同正确性168非空调用+28空返回；编译观察；18进程同场性能对照')
s = s.replace('10-09 13:42 M06-04：4非空+2空+18非法用例及归档复核通过；M05冻结基线不变', '10-09 14:51 M06-06同场对照采集/复核完成；v1无稳定提速证据')
s = s.replace('参数化Triton及统一检查、原版/v1公平对照、优化实验、模型接入、独立讲解', '独立串讲、M07优化迭代、模型接入、作品与复现交付')
a = s.index('**范围提醒：**')
b = s.index('\n\n## 本人手写', a)
s = s[:a] + '**范围提醒：** 正确性覆盖FP16/int32、C32、B0/1/2/3/4/5/17、D64/128/192、两布局和选定gap/guard，共享K/V各侧行距、列距1。计时只覆盖B1/16/256×D64/128、C1024、交错/gap0/重复地址。两个范围不能混称。无任意布局、运行时非法读检测、模型或AMD性能结果。见[M06设计与结果](modules/M06/README.md)。' + s[b:]
s = s.replace('- [ ] **当前M06-05：** 将原版与新公共入口接入共同正确性集合；尚未完成新矩阵。', '- [x] **M06-05/07/06：** 共同正确性、编译观察、首次同场对照已验收。\n- [x] **M06-08工程产物：** 冻结源码、设计说明与成绩归档。\n- [ ] **M06-08独立说明：** 本人串讲待记录，整格暂不计分。\n- [ ] **当前M07-01/02：** 已有本人初步预测，继续完善单因素假设并准备候选。')
s = s.replace('当前不需要重跑六配置baseline，也不同时代写mask、调优或整个模型接入。当前断点与已通过源码见[mask课](lessons/m06-03-mask-intro.md)，下一课按大纲M06-05共同正确性继续。', '当前不需要重做M05或M06全套。先推进[M07首课](lessons/m07-01-num-warps-hypothesis.md)：候选独立保存、本人改关键参数、正确性先于计时。')
s = s.replace('| 25 | 14 | 10月12日前 |', '| 25 | 23 | 10月12日前 |')
s = s.replace('| **100** | **44** |', '| **100** | **53** |')
updates = {
 'M06-05': ('☑ 工程验收', '10-09 run-20261009-140928-ht8v9mg8，168非空+28空返回，完整存储/控制夹具复核。'),
 'M06-06': ('☑ 首次对照验收', '10-09 run-20261009-145144-n1lwky52，18进程2160区间；没有稳定提速结论。'),
 'M06-07': ('☑ 工程证据验收', '10-09 run-20261009-142759-l12p0a0c，实际调用、五类产物；本人独立解释另记。'),
 'M06-08': ('◐ 已冻结，串讲待记录', '10-09源码/设计/成绩已打包；本人独立说明尚无记录，本格2分不提前计。'),
 'M07-01': ('◐ 首课已开始', '已给全表与波动分析；本人对大小B的独立分析待补。'),
 'M07-02': ('◐ 已有初步预测', '本人：“更快，因为一个warp工作量不算多”；两个配置的分别预测与否定条件待补。'),
}
lines = s.splitlines()
for i, line in enumerate(lines):
    for key, (status, evidence) in updates.items():
        if line.startswith('| '+key+' |'):
            parts = line.split('|')
            parts[2] = ' '+status+' '
            parts[4] = parts[4].rstrip()+'；'+evidence+' '
            lines[i] = '|'.join(parts)
write(name, '\n'.join(lines)+'\n')
print('UPDATED: PROGRESS, LEARNING_LOG, COACHING_GUIDE, syllabus, task board; backups retained')
