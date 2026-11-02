"""Offline H200 (sm_90a) compiler diagnostics. No H200 GPU or score produced.

Inputs are exact embedded CUDA device sources/defines extracted from local
candidates. Only the Torch host launcher is omitted; explicit kernel template
instantiations replace its references. Inspect register counts, spills, compiler
warnings and SASS. Runtime correctness and speed still require H200 checks.
"""
from pathlib import Path
import concurrent.futures,hashlib,json,re,subprocess,time,sys
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'hopper_build'
BUILD.mkdir(exist_ok=True)

def compile_case(case):
    folder=BUILD/case['name'];folder.mkdir(exist_ok=True)
    source=case['source'].split('\n#include <torch/extension.h>')[0]
    source+='\ntemplate __global__ void okc::moe_fused<false>(okc::Params);\n'
    if not case['tiny']:source+='template __global__ void okc::moe_fused<true>(okc::Params);\n'
    path=folder/'kernel.cu';path.write_text(source)
    binary=folder/'kernel.cubin'
    flags=list(case['defines'])
    for key,value in [('OKC_NS',3),('OKC_NSF',3),('OKC_DQ7',1),('OKC_PRMTSC',1)]:
        if not any(f.startswith('-D'+key+'=') for f in flags):flags.append(f'-D{key}={value}')
    flags=[v for v in flags if not v.startswith('@')]
    command=['nvcc','-std=c++17','-O3','-lineinfo','--expt-relaxed-constexpr','--cubin','-arch=sm_90a','-Xptxas=-v','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__',*flags,str(path),'-o',str(binary)]
    start=time.monotonic()
    run=subprocess.run(command,capture_output=True,text=True,timeout=240)
    (folder/'compiler.log').write_text(run.stdout+run.stderr)
    report={'name':case['name'],'compiled':run.returncode==0,'seconds':round(time.monotonic()-start,3),'device_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'defines':flags,'resource_lines':[line for line in run.stderr.splitlines() if any(s in line for s in ('registers','stack frame','spill','warning','error'))]}
    if run.returncode==0:
        sass=subprocess.run(['cuobjdump','--dump-sass',str(binary)],capture_output=True,text=True,check=True).stdout
        (folder/'kernel.sass').write_text(sass)
        counts={}
        for line in sass.splitlines():
            match=re.search(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]+)',line)
            if match:
                op=match.group(1).split('.')[0];counts[op]=counts.get(op,0)+1
        report['static_instruction_counts']=counts
        report['cubin_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    else:report['error_tail']=run.stderr[-5000:]
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report

def main():
    package=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'hopper_cases.json')
    cases=json.loads(package.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(compile_case,cases))
    (ROOT/(package.stem+'_result.json' if len(sys.argv)>1 else 'hopper_crosscompile_result.json')).write_text(json.dumps({'target':'sm_90a','timing_claim':None,'reports':reports},indent=2)+'\n')
    print('HOPPER_CROSSCOMPILE_DONE',sum(r['compiled'] for r in reports),'/',len(reports),flush=True)

if __name__=='__main__':main()
