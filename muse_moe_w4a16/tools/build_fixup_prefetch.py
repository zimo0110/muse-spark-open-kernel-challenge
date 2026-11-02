"""Batch independent split-unit readbacks; no check or main-file mutation."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];text=(root/'cuda_persistent_aligned_safe.py').read_text();nodes=ast.parse(text)
node=next(n for n in nodes.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets));src=ast.literal_eval(node.value)
a=src.index('__device__ __forceinline__ void wgs_flush(');b=src.index('template <bool PB, int NTT>',a);body=src[a:b]
assert body.count('nt0 += 2')==1 and body.count('float4 v[2]')==1 and body.count('k < 2')==2
lines=text.splitlines(keepends=True);start=sum(map(len,lines[:node.value.lineno-1]))+node.value.col_offset;end=sum(map(len,lines[:node.value.end_lineno-1]))+node.value.end_col_offset
defs=next(ast.literal_eval(n.value) for n in nodes.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAIN_DEFINES' for t in n.targets))
cases=[]
for burst in (4,8):
 changed=body.replace('nt0 += 2',f'nt0 += {burst}').replace('float4 v[2]',f'float4 v[{burst}]').replace('k < 2',f'k < {burst}')
 source=src[:a]+changed+src[b:];out=text[:start]+repr(source)+text[end:];ast.parse(out)
 (root/f'cuda_persistent_fixup{burst}.py').write_text(out)
 cases.append({'name':f'fixup_prefetch{burst}','source':source,'defines':defs,'tiny':False})
 for ntt in range(1,9):assert [i+k for i in range(0,ntt,burst) for k in range(burst) if i+k<ntt]==list(range(ntt))
for name,expr in [('fixup_pb4','PB ? 4 : 2'),('fixup_pa4','PB ? 2 : 4')]:
    changed=body.replace('  for (int nt0 = 0; nt0 < ntt; nt0 += 2)',f'  constexpr int BURST = {expr};\n  for (int nt0 = 0; nt0 < ntt; nt0 += BURST)')
    changed=changed.replace('float4 v[2]','float4 v[BURST]').replace('k < 2','k < BURST')
    source=src[:a]+changed+src[b:];out=text[:start]+repr(source)+text[end:];ast.parse(out)
    (root/f'cuda_persistent_{name}.py').write_text(out)
    cases.append({'name':name,'source':source,'defines':defs,'tiny':False})
Path('/tmp/muse_fixup_cases.json').write_text(json.dumps(cases))
print('Prepared4/8-wide and phase-specific readbacks. Math, stores and synchronization preserved; H200 UNVERIFIED. Tile coverage PASS.')
