"""CPU saved-array audit: FP16 rounding boundaries versus actual dot products.

Every disputed coordinate also gets an exact integer dot: every finite FP16
operand is an integer multiple of 2**-24, so each product is integral at 2**-48.
This diagnoses rounding, not GPU timing or full-model divergence causality.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok, message):
    if not ok: raise AssertionError(message)
def scaled_half(value): return int(float(value) * (2**48))
def bits(value): return int(np.asarray(value,dtype=np.float16).view(np.uint16))
def ordered(values):
    raw=np.asarray(values,dtype=np.float16).view(np.uint16).astype(np.int32)
    return np.where(raw & 0x8000, 0x8000-(raw & 0x7fff), 0x8000+raw)


def exact_dot(x, w):
    xi=np.ldexp(x.astype(np.float64),24).astype(np.int64)
    wi=np.ldexp(w.astype(np.float64),24).astype(np.int64)
    return sum(int(a)*int(b) for a,b in zip(xi,wi))


def correct_round(total):
    approximate=np.float16(total/(2**48))
    choices=[approximate,np.nextafter(approximate,np.float16(-np.inf)),
             np.nextafter(approximate,np.float16(np.inf))]
    choices=[x for x in choices if np.isfinite(x)]
    best=min(choices,key=lambda x:(abs(scaled_half(x)-total),bits(x)&1))
    if total==0: best=np.float16(0)
    nearest=scaled_half(best)
    if nearest <= total:
        lo,hi=best,np.nextafter(best,np.float16(np.inf))
    else:
        lo,hi=np.nextafter(best,np.float16(-np.inf)),best
    low,high=scaled_half(lo),scaled_half(hi)
    midpoint_numerator=abs(2*total-low-high)
    return best, {'exact_midpoint':midpoint_numerator==0,
                  'distance_to_nearest_midpoint_ulp':midpoint_numerator/(2*(high-low)),
                  'bracket_low':float(lo),'bracket_high':float(hi)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--fixture-run',type=Path,required=True)
    args=parser.parse_args();run,fixture_run=args.run,args.fixture_run
    result,manifest,fixture=read(run/'result.json'),read(run/'artifact-manifest.json'),read(fixture_run/'fixture.json')
    require(sha(run/'result.json')==manifest['result.json'],'Campaign changed')
    require(sha(fixture_run/'fixture.json')==result['fixture_sha256'],'Fixture binding changed')
    names=[name for name in result['numerical_eligible'] if name.startswith('gemv')]
    require(len(names)==4 and len(fixture['samples'])==84,'Unexpected saved scope')
    statistics={name:Counter() for name in names}
    weights={entry['layer']:entry for entry in fixture['weights']}
    loaded_weights={};details=[];total_elements=0;verified=0;oracle_exact_round_differences=0
    baseline_stats=Counter();ptx=[]
    for name in names:
        folder=run/'numerical'/name/'compiler'
        entries=read(folder/'compiled.json')
        for entry in entries:
            for filename,digest in entry['files'].items():require(sha(folder/filename)==digest,'Compiler artifact changed')
            path=folder/f"kernel{entry['kernel_index']}.ptx"
            text=path.read_text()
            ptx.append({'name':name,'kernel':entry['name'],'ptx_sha256':sha(path),
                        'fma_rn_f32_positions':len(re.findall(r'\bfma\.rn\.f32\b',text)),
                        'mul_f32_positions':len(re.findall(r'\bmul\.f32\b',text)),
                        'add_f32_positions':len(re.findall(r'\badd\.f32\b',text)),
                        'shfl_positions':len(re.findall(r'\bshfl\.sync',text)),
                        'limit':'Static PTX positions only; not dynamic instruction counts or original cuBLAS reduction order.'})
    for sample in fixture['samples']:
        layer,step=sample['layer'],sample['step']
        original_path=fixture_run/sample['path'];require(sha(original_path)==sample['sha256'],'Fixture sample changed')
        with np.load(original_path,allow_pickle=False) as data: x,z=data['x'].copy(),data['z'].copy()
        relative=f'numerical/oracle-step{step:02d}-layer{layer:02d}.npz'
        require(sha(run/relative)==manifest[relative],'FP64 oracle changed')
        with np.load(run/relative,allow_pickle=False) as data: oracle=data['z64'].copy()
        rounded=oracle.astype(np.float16);outputs={}
        require(z.shape==oracle.shape==(1,6144) and np.isfinite(oracle).all(),'Invalid fixed fixture')
        total_elements+=z.size
        baseline_stats['different_from_fp64_rounded']+=int(np.count_nonzero(z!=rounded))
        baseline_stats['bit_different_from_fp64_rounded']+=int(np.count_nonzero(z.view(np.uint16)!=rounded.view(np.uint16)))
        disputed=z!=rounded
        for name in names:
            rel=f'numerical/{name}/step{step:02d}-layer{layer:02d}.npz'
            require(sha(run/rel)==manifest[rel],'Saved candidate changed');verified+=1
            with np.load(run/rel,allow_pickle=False) as data:a=data['z'].copy()
            require(a.dtype==z.dtype==np.float16 and a.shape==z.shape and np.isfinite(a).all(),'Invalid candidate metadata')
            outputs[name]=a;s=statistics[name]
            unequal=a!=z;distance=np.abs(ordered(a)-ordered(z))
            s['different_from_original']+=int(unequal.sum())
            s['bit_different_from_original']+=int(np.count_nonzero(a.view(np.uint16)!=z.view(np.uint16)))
            s['different_from_fp64_rounded']+=int(np.count_nonzero(a!=rounded))
            s['equal_to_fp64_rounded']+=int(np.count_nonzero(a==rounded))
            s['one_ulp_difference_from_original']+=int(np.count_nonzero(distance==1))
            s['more_than_one_ulp_difference_from_original']+=int(np.count_nonzero(distance>1))
            s['max_ulp_difference_from_original']=max(s['max_ulp_difference_from_original'],int(distance.max()))
            disputed|=(a!=rounded)|(a!=z)
        if disputed.any() and layer not in loaded_weights:
            item=weights[layer];path=fixture_run/item['path'];require(sha(path)==item['sha256'],'Weight changed')
            loaded_weights[layer]=np.load(path,allow_pickle=False,mmap_mode='r')
        for row in np.flatnonzero(disputed.ravel()):
            total=exact_dot(x.ravel(),loaded_weights[layer][row]);best,mid=correct_round(total)
            if bits(best)!=bits(rounded[0,row]):oracle_exact_round_differences+=1
            original=z[0,row];bo=abs(scaled_half(original)-total)
            item={'step':step,'layer':layer,'output_column':int(row),'half':'gate' if row<3072 else 'up',
                  'original':float(original),'saved_fp64_oracle':float(oracle[0,row]),
                  'exact_integer_dot_numerator':str(total),'exact_integer_dot_denominator_power':48,
                  'correctly_rounded_fp16_from_exact_dot':float(best),
                  'original_correctly_rounded':bits(original)==bits(best),**mid,'candidates':{}}
            baseline_stats['exact_checked_coordinates']+=1
            baseline_stats['correct_on_exact_checked_coordinates']+=int(bits(original)==bits(best))
            baseline_stats['exact_dot_at_fp16_midpoint']+=int(mid['exact_midpoint'])
            for name,a in outputs.items():
                value=a[0,row];error=abs(scaled_half(value)-total);s=statistics[name]
                comparison='equal_error' if error==bo else ('candidate_closer' if error<bo else 'original_closer')
                good=bits(value)==bits(best)
                s['correct_on_exact_checked_coordinates']+=int(good)
                if value!=original:
                    s[comparison+'_on_original_disagreements']+=1
                    s['disagreement_near_midpoint_within_0.01_ulp']+=int(mid['distance_to_nearest_midpoint_ulp']<=.01)
                    s['disagreement_near_midpoint_within_0.001_ulp']+=int(mid['distance_to_nearest_midpoint_ulp']<=.001)
                    key=('both_correct' if good and bits(original)==bits(best) else
                         'candidate_only_correct' if good else 'original_only_correct' if bits(original)==bits(best) else 'neither_correct')
                    s[key+'_on_original_disagreements']+=1
                item['candidates'][name]={'value':float(value),'correctly_rounded':good,'closer_to_exact_dot':comparison,
                                        'absolute_exact_error':error/(2**48),
                                        'ulp_distance_from_original':abs(int(ordered(value))-int(ordered(original)))}
            details.append(item)
    output={'status':'SAVED_FP16_ROUNDING_AUDIT_PASS','gpu_execution':False,'source_sha256':sha(__file__),
            'fixture_sha256':sha(fixture_run/'fixture.json'),'campaign_result_sha256':sha(run/'result.json'),
            'sample_count':84,'z_elements_per_candidate':total_elements,'candidate_z_files_verified':verified,
            'baseline':dict(baseline_stats),'candidates':{k:dict(v) for k,v in statistics.items()},
            'exact_checked_coordinates':len(details),'saved_fp64_round_vs_exact_round_differences_on_checked_coordinates':oracle_exact_round_differences,
            'exact_method':'FP16 operands scaled by2^24 to signed integers; Python arbitrary-width integer products/sum; compare nearest FP16 and adjacent values in exact2^-48 units, ties-to-even.',
            'ptx_observations':ptx,
            'limit':'Exact dot recomputation covers union of disputed coordinates, not all516096 dots. Full-array comparisons use saved FP64 oracle. Original cuBLAS internal reduction order is not recovered; no full-model causal or accuracy-quality claim.',
            'details_file':'rounding-analysis-details.json'}
    for filename,document in [('rounding-analysis-details.json',details),('rounding-analysis.json',output)]:
        path=run/filename;require(not path.exists(),'Preserve existing analysis');path.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
