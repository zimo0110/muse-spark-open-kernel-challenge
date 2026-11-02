"""Remove unreachable fat scratch from tiny builds; bound NS2 to two CTAs/SM."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_occupancy.py').read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  old='  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;'
  assert src.count(old)==1
  src=src.replace(old,'''#if OKC_TINY
  return base + xa; // all fat launches/paths are unreachable in TINY variants
#endif
'''+old)
  old='__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p)'
  assert src.count(old)==1
  src=src.replace(old,'__global__ void __launch_bounds__(NTL, (OKC_TINY && OKC_NS <= 2) ? 2 : 1) moe_fused(Params p)')
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
(root/'cuda_persistent_occupancy_v2.py').write_text(text)
print('Tiny scratch allocation excludes fat ring; NS2 bounds two resident blocks, then runtime occupancy-gates grid.')
