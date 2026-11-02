"""Four compute warps per tiny CTA, two CTAs/SM only if occupancy permits.

NCH doubles8->16, so explicitly extend scale-bank addressing. Weight codes and
activation K coverage already derive from NCH. The scale vectors were hardcoded
for8warps and must select the correct old eight-chunk bank for each c group.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_occupancy_v3.py').read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  start=src.index('__device__ void skinny_ring_ws(')
  end=src.index('\n}\n',start)+3
  body=src[start:end]
  old='const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);'
  assert body.count(old)==1
  body=body.replace(old,'// Scale bank is selected per eight K chunks below (supports NW=4 or8).')
  marker='const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);'
  assert body.count(marker)==1
  body=body.replace(marker,marker+'\n      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + (warp * (NCH / 8) + c / 8) * 512 + lane * 16);\n      const int sc_c = c & 7;')
  old='c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)'
  assert body.count(old)==2
  body=body.replace(old,'sc_c < 4 ? (sc_c < 2 ? sc.x : sc.y) : (sc_c < 6 ? sc.z : sc.w)')
  src=src[:start]+body+src[end:]
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
old="small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2',)+sfx)";assert text.count(old)==1
text=text.replace(old,"small=_ext(tuple(d for d in TINY_DEFINES if not d.startswith('-DOKC_NW='))+('-DOKC_NW=4','-DOKC_NS=2')+sfx)")
(root/'cuda_persistent_fourwarp.py').write_text(text)
# Every64-K slice and its scale bank match the original lossless packing.
for nw in (4,8):
 nch=64//nw
 seen=[]
 for w in range(nw):
  for c in range(nch):
   k=w*nch+c;bank=w*(nch//8)+c//8;word=(c&7)//2
   assert bank==k//8 and word==(k%8)//2
   seen.append(k)
 assert seen==list(range(64))
print('Built NW4/NS2 experiment; exact64-Kcode/scalebank coverage matches original packing.')
