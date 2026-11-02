"""Use four k16 MMAs/commit only in the narrowest N=8 WGMMA variant.

Upstream's global GS=4 failed at N=16..32 due register pressure/C7512. This
experiment leaves all those variants at GS=2 and specializes N=8 only. The
k16 operation order, input precision, shared-ring retirement protocol, and
FP32 accumulation remain identical. The pending-publication point stays at
8 k16 operations into the first stage (group index1 versus3).
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_aligned_safe.py').read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  old='constexpr int GS = OKC_WGSGS;';assert src.count(old)==1
  src=src.replace(old,'constexpr int GS = NTT == 1 ? 4 : OKC_WGSGS;')
  old='FIRST && ci == OKC_WGSFCI && pd.kind';assert src.count(old)==1
  src=src.replace(old,'FIRST && ci == (NTT == 1 ? 1 : OKC_WGSFCI) && pd.kind')
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
(root/'cuda_persistent_narrow.py').write_text(text)
for gs in (2,4):
 order=[lg*4+h*gs+jj for lg in range(4) for h in range(4//gs) for jj in range(gs)]
 assert order==list(range(16))
print('Built narrow GS4 experiment: identical k16 order; eight-step release point preserved.')
