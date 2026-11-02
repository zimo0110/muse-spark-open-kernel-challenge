"""Generate a GB10-only developer port for tiny shapes. NEVER submit this file.

Keeps NVFP4 packing, BF16 boundaries and mma.sync math. Excludes the Hopper
WGMMA main path, uses a one-slot TMA ring within GB10 shared-memory capacity,
and keeps FAST1 preissue within that one slot. Large shapes are explicitly
rejected, rather than claiming this development subset solves the challenge.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_occupancy_v3.py').read_text()
text=text[:text.index('_codex_occ_base_kernel = kernel_fn')]
lines=text.splitlines(keepends=True);repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  old='__launch_bounds__(NTL, (OKC_TINY && OKC_NS <= 2) ? 2 : 1)';assert src.count(old)==1
  src=src.replace(old,'__launch_bounds__(NTL, 1)')
  start=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;end=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((start,end,repr(src)))
assert len(repls)==2
for start,end,value in reversed(repls):text=text[:start]+value+text[end:]
text=text.replace('# okc: problem=muse_moe_w4a16','# DEVELOPER ONLY: GB10 tiny-shape port. NEVER SUBMIT.',1)
text+='''
# GB10 developer subset; no Hopper WGMMA build, no large-shape fallback.
def _spark_tiny_flags(flags):
    return tuple(d for d in flags if not d.startswith(('-DOKC_NS=','-DOKC_FAST1='))) + ('-DOKC_NS=1','-DOKC_FAST1=1')
TINY_DEFINES = _spark_tiny_flags(TINY_DEFINES)
TINY2_DEFINES = _spark_tiny_flags(TINY2_DEFINES)
MID_DEFINES = _spark_tiny_flags(MID_DEFINES)
MAIN_DEFINES = TINY_DEFINES
MAIN_FALLBACK_DEFINES = TINY_DEFINES
_spark_native_kernel = kernel_fn

def kernel_fn(data,state):
    if not 1 <= data['hidden_states'].shape[0] <= 16:
        raise ValueError('GB10 developer subset supports1..16tokens only; not a challenge submission.')
    return _spark_native_kernel(data,state)
'''
out=root/'tools'/'spark_tiny_port.py';out.write_text(text)
print('Generated development-only GB10 port:3tiny compile variants,1-slot ring,1..16tokens,original global scratch strides.')
