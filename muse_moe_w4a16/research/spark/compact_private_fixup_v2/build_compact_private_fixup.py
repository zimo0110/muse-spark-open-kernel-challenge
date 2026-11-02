"""Compact exclusive partials to two banks per CTA (head and tail).

A static contiguous stream-K range has at most one partial unit at either end.
Contributors to a split unit are consecutive CTAs, in increasing K order.
This keeps disjoint writers and packed release-counter protocol, with 2 instead
of 16 banks per slot. No H200 check dispatched by this builder.
"""
import ast,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'cuda_persistent_private_fixup.py').read_text();t=ast.parse(s)
n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='CUDA_SRC' for a in n.targets));src=ast.literal_eval(n.value)
old='wgs_fix(p, PB) + ((size_t)(slot * 4 + wg) * 16 + s0) * WGS_FIXW + wt * 4'
new='wgs_fix(p, PB) + ((size_t)((int)blockIdx.x * 2 + (s0 != 0)) * 4 + wg) * WGS_FIXW + wt * 4'
assert src.count(old)==1;src=src.replace(old,new)
old='? 3 : 2, u, slot, s1 - s0, s0}'
assert src.count(old)==1;src=src.replace(old,'? 3 : 2, u, slot, s1 - s0, (int)blockIdx.x - slot}')
src=src.replace('WGS_FIXW * 16','WGS_FIXW * 2')
old='(size_t)start * WGS_FIXW + (nt0+k) * 512'
new='(size_t)(start * 8 + (start != 0 ? 4 : 0)) * WGS_FIXW + (nt0+k) * 512'
assert src.count(old)==1;src=src.replace(old,new)
# Bit0 always identifies the share containing K stage0. Initialize from that
# bank instead of looping over it and adding its values to zero. Remaining
# mask bits continue in increasing CTA/K order, with unchanged FP32 precision.
old_init = 'float4 value[1] = {make_float4(0.f,0.f,0.f,0.f)};\n    unsigned mask = (unsigned)S.wglast[wg];'
new_init = 'float4 value[1] = {__ldcg(reinterpret_cast<const float4*>(fb + nt0 * 512))};\n    unsigned mask = (unsigned)S.wglast[wg] & ~1u;'
assert src.count(old_init)==1;src=src.replace(old_init,new_init)
# Read base is the first contributor CTA's tail bank (2*slot).
src=src.replace('(size_t)(pd.slot * 4 + wg) * WGS_FIXW * 2 + wt * 4','((size_t)pd.slot * 8 + wg) * WGS_FIXW + wt * 4')
lines=s.splitlines(keepends=True);a=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;b=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
s=s[:a]+repr(src)+s[b:]
s=s.replace('WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW * 16 +','WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW * 2 +')
ast.parse(s);(r/'cuda_persistent_compact_private_fixup.py').write_text(s)
defs=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='MAIN_DEFINES' for a in n.targets))
Path('/tmp/muse_compact_private_fixup_cases.json').write_text(json.dumps([{'name':'compact_private_fixup','source':src,'defines':defs,'tiny':False}]))
print('Prepared compact exclusive partial banks: +20MiB workspace; H200 UNVERIFIED.')
