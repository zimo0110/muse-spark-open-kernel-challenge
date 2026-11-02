"""Offline Hopper register redistribution probes; not a selected entry.

Four compute warpgroups request 112 or 104 registers/thread. The producer
warpgroup (two TMA warps plus two idle warps) donates down to 32 or 64. All
four warps in each warpgroup execute the same setmaxnreg, as PTX requires.
Entry budget96*640 == 112*512+32*128 == 104*512+64*128 ==61440 registers.

Primary references:
https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
tree=ast.parse((R/'cuda_persistent_aligned_safe.py').read_text())
def val(k):return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==k for t in n.targets))
s=val('CUDA_SRC').replace('NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0)','NT + (OKC_WGS ? 128 : OKC_WS ? 32 : 0)')
old='    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));'
assert s.count(old)==1
s=s.replace(old,'    else if(tid<NT+64) wgs_xproducer(p,S,reinterpret_cast<unsigned char*>(xbuf));\n    else { WS_JOIN(); } // idle warps complete the producer warpgroup')
cases=[]
for consumer,producer in ((112,32),(104,64)):
 code=s
 marker='#if OKC_WGS\n  if (tid >= NT) {'
 assert code.count(marker)==1
 code=code.replace(marker,marker+f'\n    asm volatile("setmaxnreg.dec.sync.aligned.u32 {producer};":::"memory");')
 # Compute allocation starts after the producer-only branch has exited;
 # no control-flow merge can require a common register budget afterward.
 marker='#endif\n#if FAST1_ON\n  // tiny build'
 assert code.count(marker)==1
 code=code.replace(marker,f'#endif\n#if OKC_WGS\n  asm volatile("setmaxnreg.inc.sync.aligned.u32 {consumer};":::"memory");\n#endif\n#if FAST1_ON\n  // tiny build')
 for gs in (2,4,"narrow4"):
  src=code if gs=="narrow4" else code.replace('#define OKC_WGSGS 2',f'#define OKC_WGSGS {gs}')
  if gs=="narrow4": src=src.replace('constexpr int GS = OKC_WGSGS;','constexpr int GS = NTT<=4 ? 4 : 2;')
  cases.append(dict(name=f'register_donation_{consumer}_{producer}_gs{gs}',source=src,defines=val('MAIN_DEFINES'),tiny=False))
(R/'research/spark/register_donation_cases.json').write_text(json.dumps(cases)+'\n')
print('Prepared branch-local register donation compiler probes. Runtime correctness/performance unverified.')
