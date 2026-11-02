"""Move activation-padding initialization to the weight-producer warp.

The producer clears shared padding while consumers build routing/gather data.
The existing full-CTA join orders those stores before either producer starts
TMA or consumers issue WGMMA. This also removes initialization temporaries
from the consumer function's register-allocation context.
"""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
base=next(q for q in json.loads((R/'research/spark/fixed_width_cases.json').read_text()) if q['name']=='fixed_n64')
src=base['source']
old='''  {
    auto zero_ring = reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(xbuf) + WGS_NS*UNIT_BYTES);
    for (int q=tid; q<WGS_NX*WGS_XS/16; q+=NT) zero_ring[q]=make_uint4(0,0,0,0);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    CTA_SYNC();
  }
'''
assert src.count(old)==1;src=src.replace(old,'')
marker='''__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();'''
insert='''__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  auto zero_ring = reinterpret_cast<uint4*>(ring + WGS_NS*UNIT_BYTES);
  for (int q=(threadIdx.x&31); q<WGS_NX*WGS_XS/16; q+=32) zero_ring[q]=make_uint4(0,0,0,0);
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
  __syncwarp();
  WS_JOIN();'''
assert src.count(marker)==1;src=src.replace(marker,insert)
noepi=src.replace('for (int nt = 0; nt < NTT; ++nt) if (nt < ((S.jC[j]+7)>>3)) wgs_epi_tile','for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile')
assert noepi!=src
cases=[]
for name,code in [('fixed_n64_producer_init',src),('fixed_n64_producer_init_noepi',noepi),('fixed_n64_producer_init_noepi_gs1',noepi.replace('#define OKC_WGSGS 2','#define OKC_WGSGS 1'))]:
 cases.append(dict(base,name=name,source=code))
(R/'research/spark/fixed64_producer_init_cases.json').write_text(json.dumps(cases)+'\n')
print('Prepared3 producer-initialization compiler experiments; H200 unmeasured.')
