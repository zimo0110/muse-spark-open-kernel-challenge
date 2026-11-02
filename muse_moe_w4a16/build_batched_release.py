"""Batch only same-CTA, whole paired units' phase-A ready publications.

PTX release/acquire and named barrier semantics:
https://docs.nvidia.com/cuda/parallel-thread-execution/#release-and-acquire-patterns
The original 128-thread named barrier still orders ALL stores before release.
A +2 release replaces two +1 releases only when both units' writers are exactly
this warpgroup, both units are whole, and they feed the same phase-B stage.
Partial/split units, nonconsecutive units, job boundaries and phase-end flushes
keep the existing protocol. Prepared weights and all arithmetic are unchanged.
"""
import ast
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base=ROOT/'cuda_persistent_bf16.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='91766194b46a086f3d772238d25f86e87c4668109384798bb8245f4f66f7d9dd'
text=base.read_text(); lines=text.splitlines(keepends=True)
def patch(src):
    old='if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);'
    new='''// Coalesce the two whole phase-A units feeding one phase-B stage.
        // No wait can consume only half a stage. Never coalesce split units.
        if (FIRST && ci == OKC_WGSFCI && pd.kind &&
            !(!PB && slot < 0 && (u & 1) && pd.kind == 1 && pd.u == u - 1 && pd.nst == 1))
          wgs_flush<PB>(p, S, pd);'''
    assert src.count(old)==1;src=src.replace(old,new)
    old='if (!PB) pd = WgsPend{1, u, -1, 0};'
    new='if (!PB) pd = WgsPend{1, u, -1, pd.kind == 1 ? 2 : 1};'
    assert src.count(old)==1;src=src.replace(old,new)
    begin=src.index('if (pd.kind == 1) {');end=src.index('int* fc =',begin)
    segment=src[begin:end]
    assert segment.count('"r"(1)')==1
    segment=segment.replace('"r"(1)', '"r"(pd.nst)')
    src=src[:begin]+segment+src[end:]
    return src
repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(patch(ast.literal_eval(n.value)))))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
text=text.replace('okc_moe_w4a16_bf16stage_', 'okc_moe_w4a16_batchrelease_')
text=text.replace('# Local Apache-2.0 port: bf16stage;', '# Local Apache-2.0 port: paired phase-A ready releases;')
text+=(ROOT/'persistent_replay_stress.inc').read_text()
(ROOT/'cuda_persistent_batchrelease.py').write_text(text)
print('Built paired-release candidate with prepare-only replay stress.')
