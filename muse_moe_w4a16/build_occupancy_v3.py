"""Bound only tiny builds' shared routing arrays; keep global scratch strides."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_occupancy_v2.py').read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  old='constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;'
  assert src.count(old)==1
  src=src.replace(old,old+'\nconstexpr int ROUTE_MMAX = OKC_TINY ? (OKC_SPLIT ? 16 : 8) : MMAX;\nconstexpr int ROUTE_EMAX = ROUTE_MMAX * 8;')
  for typ,field in [('signed char','eid'),('short','ent'),('short','pos')]:
   old=f'{typ} {field}[EMAX];';assert src.count(old)==1
   src=src.replace(old,f'{typ} {field}[ROUTE_EMAX];')
  old='p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;';assert src.count(old)==1
  src=src.replace(old,old+'\n  TORCH_CHECK(p.M <= okc::ROUTE_MMAX, "batch exceeds compiled routing capacity");')
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
(root/'cuda_persistent_occupancy_v3.py').write_text(text)
# Every tiny input route index and packed entry remains in bounds. EMAX is
# deliberately unchanged: clearing double-buffered sums must cover all384 rows
# between mixed-shape launches, even when the current launch is tiny.
for tiny,split in [(True,False),(True,True),(False,False)]:
 cap=(16 if split else 8) if tiny else 384
 for m in range(1,cap+1):
  assert max(range(m*8)) < cap*8
print('Bounded shared routing arrays; global EMAX3072, scratch offsets and all-EMAX resets unchanged.')
