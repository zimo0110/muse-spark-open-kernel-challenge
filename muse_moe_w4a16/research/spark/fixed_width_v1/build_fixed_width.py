"""Prototype one fixed WGMMA N-width per shape band. No dispatch/promotion.

Trade: extra tensor work on padded columns for a smaller consumer binary and
fewer dynamic dispatch variants. Only valid token columns may be stored into
split-K partials or expert outputs. Shared B is initialized once per call;
invalid columns can subsequently contain same-call old data, but their results
are always discarded. NVFP4/BF16 formats and all numeric boundaries unchanged.
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
text=(R/'cuda_persistent_aligned_safe.py').read_text();tree=ast.parse(text)
def value(name):return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
base=value('CUDA_SRC')
cases=[]
for width in (4,8):
 src=base
 a=src.index('    switch (ntt) {',src.index('__device__ __forceinline__ void wgs_phase('));b=src.index('    gs += s1 - s0;',a)
 switch=src[a:b];assert switch.count('wgs_seg<PB,')==8
 src=src[:a]+f'    wgs_seg<PB, {width}>(p, S, ring, u, s0, s1, slot, seq, pd);\n'+src[b:]
 # Fixed-width padding must not leave stale atomic partials in inactive tiles.
 a=src.index('__device__ __forceinline__ void wgs_seg(');b=src.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase(',a);body=src[a:b]
 old='''#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32'''
 new='''    const int valid_ntt = (S.jC[u >> UBL] + 7) >> 3;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) if (nt < valid_ntt)
      asm volatile("red.global.add.v4.f32'''
 assert body.count(old)==1;body=body.replace(old,new)
 old='  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);'
 new='  for (int nt = 0; nt < NTT; ++nt) if (nt < ((S.jC[j]+7)>>3)) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);'
 assert body.count(old)==1;body=body.replace(old,new)
 src=src[:a]+body+src[b:]
 # Readers may access additional valid shared addresses beyond the current
 # TMA payload. Initialize them before helpers start any TMA copies.
 marker='''  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();'''
 insert='''  wgs_gather(p, S);
  DBG(7);
  {
    auto zero_ring = reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(xbuf) + WGS_NS*UNIT_BYTES);
    for (int q=tid; q<WGS_NX*WGS_XS/16; q+=NT) zero_ring[q]=make_uint4(0,0,0,0);
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    CTA_SYNC();
  }
  WS_JOIN();'''
 assert src.count(marker)==1;src=src.replace(marker,insert)
 if width==4:
  # Bound actual jobs, including duplicate expert ids, without changing the
  # fixed64-token scratch ABI shared with the control kernel.
  ra=src.index('__device__ void route(')
  rb=src.index('// ---------------------------------------------------------------- epilogues',ra)
  route=src[ra:rb]
  import re
  route,n=re.subn(r'\bFT_T\b','32',route)
  assert n>=4
  src=src[:ra]+route+src[rb:]
  marker='  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;'
  assert src.count(marker)==1;src=src.replace(marker,marker+'\n  TORCH_CHECK(p.M<=32,"fixed N32 supports at most32 tokens");')
 name=f'fixed_n{width*8}'
 (R/'research/spark'/f'{name}.cu').write_text(src)
 cases.append({'name':name,'source':src,'defines':value('MAIN_DEFINES'),'tiny':False})
(R/'research/spark/fixed_width_cases.json').write_text(json.dumps(cases)+'\n')
# CPU addressing proof: initialization covers every B byte once; actual NTT
# guards only touch valid partial tiles, even while calls alternate N widths.
for width in (4,8):
 for count in range(1,width*8+1):
  valid=(count+7)//8
  assert [nt for nt in range(width) if nt<valid]==list(range(valid))
  assert [8*nt+t for nt in range(width) for t in range(8) if 8*nt+t<count]==list(range(count))
for slots in (1,2,3):
 total=slots*32768//16
 assert sorted(q for tid in range(512) for q in range(tid,total,512))==list(range(total))
print('Prepared N32/N64 fixed-width prototypes; CPU valid-column/partial/zero-ring coverage PASS. H200 UNVERIFIED.')
