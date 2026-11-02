"""Prepare spill-checked1/2WG variants with coalesced BF16 final stores.

UNVERIFIED onH200. No submissions/checks are launched. Each thread covers
multiple8-column chunks striped byNT; each128-bit output store remains contiguous
across a warp, unlike contiguous16/32-column per-thread packing.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_two_wg.py').read_text();lines=text.splitlines(keepends=True)
node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
src=ast.literal_eval(node.value)
src=src.replace('static_assert(NW == 8 || NW == 16, "WGS physical warpgroup count");','static_assert(NW == 4 || NW == 8 || NW == 16, "WGS physical warpgroup count");')
src=src.replace('(WGS_R == 2 && NTT >= 5)', '(WGS_R >= 2 && NTT >= 5)')
a=src.index('#if OKC_WGS && OKC_WGSFINSS\n');b=src.index('#elif OKC_FINCOL',a);body=src[a:b]
body=body.replace('FCOLS == 8 || FCOLS == 16','FCOLS == 8 || FCOLS == 16 || FCOLS == 32')
old='const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * (FCOLS / 4);'
assert body.count(old)==2
body=body.replace(old,'const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;')
body=body.replace('const float4 a=__ldcg(src+vi), b=__ldcg(src+vi+1);','const float4 a=__ldcg(src+(vi/2)*NT*2), b=__ldcg(src+(vi/2)*NT*2+1);')
body=body.replace('const float4 a=__ldcg(src+vi);','const float4 a=__ldcg(src+(vi/2)*NT*2+(vi&1));')
old='''uint4* dst=reinterpret_cast<uint4*>(p.out+(size_t)tk*HD+tid*FCOLS);
#pragma unroll
      for(int vi=0;vi<FCOLS/8;++vi) dst[vi]=o[vi];'''
assert body.count(old)==1
body=body.replace(old,'''#pragma unroll
      for(int vi=0;vi<FCOLS/8;++vi)
        *reinterpret_cast<uint4*>(p.out+(size_t)tk*HD+(tid+vi*NT)*8)=o[vi];''')
src=src[:a]+body+src[b:]
a=sum(map(len,lines[:node.value.lineno-1]))+node.value.col_offset;b=sum(map(len,lines[:node.value.end_lineno-1]))+node.value.end_col_offset
out=text[:a]+repr(src)+text[b:]
(root/'cuda_persistent_two_wg_striped.py').write_text(out)
old='MAIN_DEFINES = ("-DOKC_NW=8",';assert out.count(old)==1
(root/'cuda_persistent_one_wg.py').write_text(out.replace(old,'MAIN_DEFINES = ("-DOKC_NW=4",'))
for nw in (4,8,16):
 nt=nw*32;r=16//nw;parts=4096//(nt*8)
 assert sorted(wg*r+ri for wg in range(nw//4) for ri in range(r))==list(range(4))
 columns=[(tid+part*nt)*8+c for tid in range(nt) for part in range(parts) for c in range(8)]
 assert sorted(columns)==list(range(4096))
 for warp in range(nw):
  for part in range(parts):
   starts=[(warp*32+lane+part*nt)*16 for lane in range(32)]
   assert starts==list(range(starts[0],starts[0]+512,16))
 assert sum(r for _ in range(nw//4))*2==8
print('Generated1WG/2WGstriped experiments; four logical tiles, ready counts, full-row coverage, coalesced16B stores PASS. H200 UNVERIFIED.')
