"""Distribute M1 activation K across lanes, then gather B fragments with shuffles.

Original x0[16] uint4 is64registers/thread even though M1 has one live token
row. The same warp holds eight identical zero row groups. Store this warp's
512K values once, spread across all32lanes (two uint4/thread=8registers).
Gather group c from lane4*c+t, then broadcast that token row to valid n rows.
M1 duplicate routes to an expert share the same x/inter row, so remain valid.
Only the explicitly compiled M1 variant uses this layout. Full M1 original
math, BF16 boundaries, weight bytes and FP32 accumulators remain unchanged.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_occupancy_v3.py').read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  st=src.index('__device__ __forceinline__ void load_x_regs(')
  en=src.index('\n}\n',st)+3;body=src[st:en]
  marker='  const int cnt = S.jC[j], st = S.jS[j];'
  assert body.count(marker)==1
  body=body.replace(marker,marker+r'''
#if defined(OKC_M1DIST)
  static_assert(NW == 8 && NCH == 8, "M1 distributed warp layout");
  const __nv_bfloat16* row = PB ? p.inter + (size_t)st * ID : p.x;
  const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + t * 16);
  x[0] = PB ? __ldcg(src) : __ldg(src);
  x[1] = PB ? __ldcg(src + 1) : __ldg(src + 1);
  return;
#endif
''')
  src=src[:st]+body+src[en:]
  st=src.index('__device__ void skinny_ring_ws(');en=src.index('\n}\n',st)+3;body=src[st:en]
  old='''const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};'''
  assert body.count(old)==1
  new=r'''
#if defined(OKC_M1DIST)
      const int from = 4 * c + (lane & 3);
      uint32_t b[8] = {
        __shfl_sync(0xffffffffu,x0[0].x,from), __shfl_sync(0xffffffffu,x0[0].y,from),
        __shfl_sync(0xffffffffu,x0[0].z,from), __shfl_sync(0xffffffffu,x0[0].w,from),
        __shfl_sync(0xffffffffu,x0[1].x,from), __shfl_sync(0xffffffffu,x0[1].y,from),
        __shfl_sync(0xffffffffu,x0[1].z,from), __shfl_sync(0xffffffffu,x0[1].w,from)};
#pragma unroll
      for (int k = 0; k < 8; ++k) if ((lane >> 2) >= S.jC[j]) b[k] = 0;
#else
      '''+old+r'''
#endif
'''
  body=body.replace(old,new);src=src[:st]+body+src[en:]
  marker='p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;'
  assert src.count(marker)==1
  src=src.replace(marker,marker+'\n#if defined(OKC_M1DIST)\n  TORCH_CHECK(p.M == 1, "M1 distributed kernel requires one token");\n#endif')
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
old="small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2',)+sfx)";assert text.count(old)==1
text=text.replace(old,"small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2','-DOKC_M1DIST=1')+sfx)")
assert text.count('for m in (1,4):')==1
text=text.replace('for m in (1,4):','for m in (1,):')
(root/'cuda_persistent_m1_distributed.py').write_text(text)
# CPU address/fragment proof for arbitrary input row, all warps/K chunks,
# all lane groups,1..8duplicate routes, and distinct row/token positions.
import numpy as np
rng=np.random.default_rng(84017)
x=rng.integers(0,2**32,size=(4096//2,),dtype=np.uint32)
for warp in range(8):
 dist=np.stack([x[warp*256+(lane//4)*32+(lane%4)*8:warp*256+(lane//4)*32+(lane%4)*8+8] for lane in range(32)])
 for cnt in range(1,9):
  for c in range(8):
   for lane in range(32):
    got=dist[4*c+lane%4] if lane//4<cnt else np.zeros(8,dtype=np.uint32)
    start=warp*256+c*32+(lane%4)*8
    want=x[start:start+8] if lane//4<cnt else np.zeros(8,dtype=np.uint32)
    assert np.array_equal(got,want)
print('PASS distributed/gathered exact B-fragment bits:8warps x8chunks x32lanes x1..8route duplicates.')
