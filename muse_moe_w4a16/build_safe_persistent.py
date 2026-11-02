"""Short content-addressed extension names and an explicit empty-job guard.

No change on nonempty routes. Cost-table indexing/division is undefined when
there are zero jobs: publish four empty producer/consumer ranges instead.
"""
import ast,re,sys
from pathlib import Path
root=Path(__file__).resolve().parent
name=sys.argv[1]
base=root/name
text=base.read_text();lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  old='''  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);'''
  new='''  const int nj = S.njF + S.njS;
  // No local experts: avoid indexing an absent job or dividing by its cost.
  if (nj == 0) {
    S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;
    return;
  }
  wgs_prefix(S, nj);'''
  assert src.count(old)==1;src=src.replace(old,new)
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
if 'import hashlib' not in text:text=text.replace('from typing import TypedDict','from typing import TypedDict\nimport hashlib')
text,n=re.subn(r'name=f"okc_moe[^\n]+",', 'name=f"okc_moe_{hashlib.sha256((src + repr(defs)).encode()).hexdigest()[:24]}_sm{arch}",',text)
assert n==1
out=root/(base.stem+'_safe.py');out.write_text(text);print(out.name)
