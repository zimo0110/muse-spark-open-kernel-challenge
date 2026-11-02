"""Offline-only alternatives to reject register-pressure regressions early."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
cases=json.loads((R/'research/spark/fixed_width_cases.json').read_text());wide=next(c for c in cases if c['name']=='fixed_n64')
src=wide['source'];a=src.index('__device__ __forceinline__ void wgs_seg(');b=src.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase(',a);body=src[a:b]
body=body.replace('    const int valid_ntt = (S.jC[u >> UBL] + 7) >> 3;\n','').replace('nt < valid_ntt','nt < ((fixed_count(S,u>>UBL)+7)>>3)').replace('nt < ((S.jC[j]+7)>>3)','nt < ((fixed_count(S,j)+7)>>3)')
helper='''__device__ __forceinline__ int fixed_count(const Smem& S,int j) {
 int count;
 asm volatile("ld.volatile.shared.u32 %0,[%1];" : "=r"(count) : "r"(smem_u32(S.jC+j)) : "memory");
 return count;
}
'''
# Insert before the preceding template declaration.
prefix=src[:a];marker='template <bool PB, int NTT>\n';assert prefix.endswith(marker)
changed=prefix[:-len(marker)]+helper+marker+body+src[b:]
out=[]
for tag,s in [('fixed_n64_volatile',changed),('fixed_n64_gs1',src.replace('#define OKC_WGSGS 2','#define OKC_WGSGS 1')),('fixed_n64_volatile_gs1',changed.replace('#define OKC_WGSGS 2','#define OKC_WGSGS 1'))]:
 q=dict(wide,name=tag,source=s);out.append(q)
(R/'research/spark/fixed_width_register_cases.json').write_text(json.dumps(out)+'\n')
print('Prepared3 offline register-lifetime probes; NOT H200 dispatched.')
