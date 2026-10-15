"""Read-only SQLite attribution: kernel correlation -> CPU API -> linear NVTX.

GPU time containment is NOT used to identify the CPU caller. CPU API records
must have matching process/thread, and be enclosed by one labelled NVTX range.
The saved F.linear shape record supplies the label's actual callsite metadata.
No hardware counters, critical-path fractions, or unprofiled speedup claims.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def analyze(run):
    run = Path(run).resolve()
    calls = read(run/'linear-calls.json')
    by_label = {c['label']: c for c in calls}
    if len(by_label) != len(calls) or not calls:
        raise RuntimeError('Duplicate or empty linear call records')
    db = sqlite3.connect((run/'original-model.sqlite').as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    ranges, thread_ranges = {}, defaultdict(list)
    for row in db.execute('SELECT * FROM NVTX_EVENTS'):
        row = dict(row)
        label = row.get('text') or strings.get(row.get('textId'))
        if label not in by_label:
            continue
        if label in ranges or row['end'] is None or row['end'] <= row['start']:
            raise RuntimeError('Duplicate or unfinished labelled NVTX range: '+label)
        if row.get('endGlobalTid') not in (None, row['globalTid']):
            raise RuntimeError('Cross-thread range is outside this push/pop scope')
        ranges[label] = dict(label=label, start=row['start'], end=row['end'], globalTid=row['globalTid'])
        thread_ranges[row['globalTid']].append(ranges[label])
    if set(ranges) != set(by_label):
        raise RuntimeError(f'NVTX/call record mismatch {len(ranges)}/{len(by_label)}')
    thread_starts = {}
    for tid, values in thread_ranges.items():
        values.sort(key=lambda x:x['start'])
        if any(a['end'] > b['start'] for a,b in zip(values,values[1:])):
            raise RuntimeError('Linear call ranges overlap/nest; refuse ambiguous attribution')
        thread_starts[tid] = [x['start'] for x in values]

    # Process IDs carry the same high bits as globalTid; the low 24 bits are TID.
    pid_mask = ~((1 << 24)-1)
    api_by_key = defaultdict(list)
    api_count = 0
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER']:
        if table not in tables:
            continue
        for row in db.execute('SELECT * FROM '+table):
            row = dict(row)
            tid, corr = row.get('globalTid'), row.get('correlationId')
            if tid is None or corr is None:
                continue
            api_count += 1
            label = None
            starts = thread_starts.get(tid, [])
            i = bisect_right(starts, row['start'])-1
            if i >= 0:
                nv = thread_ranges[tid][i]
                if nv['start'] <= row['start'] and row['end'] <= nv['end']:
                    label = nv['label']
            api_by_key[(tid & pid_mask, corr)].append(dict(
                source=table, start=row['start'], end=row['end'], globalTid=tid,
                name=strings.get(row.get('nameId'), str(row.get('nameId'))), label=label))

    matched, unmatched = [], []
    for seq, row in enumerate(db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start,end')):
        row = dict(row)
        if row['end'] <= row['start']:
            raise RuntimeError('Nonpositive kernel duration')
        if row.get('graphNodeId') not in (None, 0):
            raise RuntimeError('Graph node found in Eager attribution; direct API containment is insufficient')
        name = strings.get(row.get('demangledName'), strings.get(row.get('shortName'), 'UNKNOWN'))
        apis = api_by_key.get((row.get('globalPid'),row.get('correlationId')), [])
        labels = {a['label'] for a in apis if a['label'] is not None}
        if len(labels) > 1:
            raise RuntimeError('A kernel correlation maps to different labelled calls')
        if labels and len(apis) > 1:
            # A runtime wrapper and its nested driver API may describe the same
            # submission. Unrelated/reused correlation IDs must not be guessed.
            nested = (len(apis) == 2 and len({a['source'] for a in apis}) == 2
                      and len({a['globalTid'] for a in apis}) == 1
                      and all(a['label'] in labels for a in apis)
                      and ((apis[0]['start'] <= apis[1]['start'] <= apis[1]['end'] <= apis[0]['end'])
                           or (apis[1]['start'] <= apis[0]['start'] <= apis[0]['end'] <= apis[1]['end'])))
            if not nested:
                raise RuntimeError('Ambiguous CPU API correlation; refuse linear attribution')
        record = dict(sequence=seq, name=name, start_ns=row['start'], end_ns=row['end'],
                      duration_ns=row['end']-row['start'], globalPid=row.get('globalPid'),
                      correlationId=row.get('correlationId'), streamId=row.get('streamId'),
                      deviceId=row.get('deviceId'),
                      grid=[row.get('grid'+axis) for axis in 'XYZ'],
                      block=[row.get('block'+axis) for axis in 'XYZ'], api_matches=apis)
        if labels:
            label = next(iter(labels))
            record.update(label=label, call=by_label[label], cpu_nvtx=ranges[label])
            matched.append(record)
        else:
            record['reason'] = 'CPU API is outside a labelled linear range' if apis else 'No matching CPU API'
            unmatched.append(record)
    db.close()
    counts = Counter(r['label'] for r in matched)
    missing = sorted(set(by_label)-set(counts))
    unknown_matrix = [r for r in unmatched if any(s in r['name'].lower() for s in ['gemv','gemm','cutlass'])]
    if missing or unknown_matrix:
        save(run/'attribution-incomplete.json', dict(missing_linear_labels=missing, unmatched_matrix_kernels=unknown_matrix))
        raise RuntimeError(f'Incomplete linear attribution: {len(missing)} calls, {len(unknown_matrix)} named matrix kernels')
    all_ns = sum(r['duration_ns'] for r in matched+unmatched)
    grouped = {}
    for c in calls:
        role = c['module_path'].split('.')[-1]
        key = (c['phase'], role, tuple(c['input']['shape']), tuple(c['weight']['shape']))
        g = grouped.setdefault(key,dict(phase=c['phase'],role=role,input_shape=c['input']['shape'],
            weight_shape=c['weight']['shape'],call_count=0,kernel_count=0,total_ns=0,kernel_names={}))
        g['call_count'] += 1
    for r in matched:
        c = r['call']
        key = (c['phase'],c['module_path'].split('.')[-1],tuple(c['input']['shape']),tuple(c['weight']['shape']))
        g = grouped[key]
        g['kernel_count'] += 1
        g['total_ns'] += r['duration_ns']
        k = g['kernel_names'].setdefault(r['name'],dict(count=0,total_ns=0))
        k['count'] += 1
        k['total_ns'] += r['duration_ns']
    groups = sorted(grouped.values(),key=lambda g:g['total_ns'],reverse=True)
    for g in groups:
        g['share_of_all_kernel_duration_sum_pct'] = 100*g['total_ns']/all_ns
    kernel_totals = {}
    for r in matched+unmatched:
        k=kernel_totals.setdefault(r['name'],dict(count=0,total_ns=0))
        k['count']+=1
        k['total_ns']+=r['duration_ns']
    kv=[r for r in unmatched if 'store_kv_cache<' in r['name']]
    summary = dict(status='LINEAR_KERNEL_ATTRIBUTION_COMPLETE', policy='eager',
        method='Kernel (globalPid,correlationId) -> CPU API (globalTid high bits,correlationId) -> same-thread containing NVTX -> actual linear metadata',
        source_hashes={name:digest(run/name) for name in ['original-model.sqlite','linear-calls.json','workload.json','plan.json']},
        analyzer_sha256=digest(__file__),linear_calls=len(calls),linear_nvtx_ranges=len(ranges),
        cpu_api_records=api_count,kernel_count=len(matched)+len(unmatched),matched_linear_kernels=len(matched),
        unmatched_other_kernels=len(unmatched),all_kernel_total_ns=all_ns,
        linear_kernel_total_ns=sum(r['duration_ns'] for r in matched),
        kv_kernel_count=len(kv),kv_kernel_total_ns=sum(r['duration_ns'] for r in kv),
        per_call_kernel_count_histogram=dict(Counter(counts.values())),groups=groups,kernels=kernel_totals,
        limits=['One separate original-backend instrumented Eager workload; not formal timing or a speedup experiment.',
                'Summed GPU kernel durations are not generate wall time or critical-path fractions.',
                'This joins this run only; old 43 Graph nodes are not thereby individually attributed.',
                'All labelled linear calls have associated kernels; unlabelled known gemv/gemm/cutlass names are rejected, other unlabelled kernels remain explicitly saved.',
                'No occupancy, cache/bandwidth counters, bank conflicts, or AMD results measured.'])
    for filename, rows in [('attributed-kernels.jsonl',matched),('unmatched-kernels.jsonl',unmatched)]:
        with (run/filename).open('w',encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(run/'attribution.json',summary)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=analyze(args.run)
    print(json.dumps({k:result[k] for k in ['status','linear_calls','kernel_count','matched_linear_kernels','kv_kernel_count']},indent=2))
