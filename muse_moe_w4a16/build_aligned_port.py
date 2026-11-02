"""Match the pinned SGLang activation/norm boundaries rather than over-rounding.

Pinned SGLang a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f:
python/sglang/kernels/jit/csrc/elementwise/activation.cuh:
filtered silu_and_mul uses kRoundActivation=false: BF16 gate/up -> float SiLU
and multiply -> BF16 inter. No separate BF16 SiLU intermediate.
python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py:
BF16 down projection -> float multiply by effective BF16 norm weight -> float
RMSNorm and weighted sum -> BF16 final. No BF16 weighted intermediate.
Baseline prepare stores BF16(1 + original BF16 norm weight); do the same.
"""
import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'cuda_persistent_bf16.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='91766194b46a086f3d772238d25f86e87c4668109384798bb8245f4f66f7d9dd'
text=p.read_text();lines=text.splitlines(keepends=True)
def patch(src):
 old='const float activation = codex_bf16_round(SILU_DIV(g, 1.f + __expf(-g)));'
 assert src.count(old)==1
 src=src.replace(old,'const float activation = SILU_DIV(g, 1.f + __expf(-g));')
 old='return codex_bf16_round(projection * norm);'
 assert src.count(old)==1;src=src.replace(old,'return projection * norm;')
 assert src.count('1.f + __bfloat162float(')==14
 assert src.count('1.f + __uint_as_float(lnw')==2
 src=src.replace('1.f + __bfloat162float(', '__bfloat162float(')
 src=src.replace('1.f + __uint_as_float(lnw', '__uint_as_float(lnw')
 return src
repls=[]
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(patch(ast.literal_eval(n.value)))))
assert len(repls)==2
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
old='lut.contiguous(), weights["post_norm_w"].contiguous(),'
assert text.count(old)==1
text=text.replace(old,'lut.contiguous(), (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).contiguous(),')
text=text.replace('from typing import TypedDict','from typing import TypedDict\nimport hashlib')
text=text.replace('name=f"okc_moe_w4a16_bf16stage_{tag}_sm{arch}"','name=f"okc_moe_al_{hashlib.sha256(tag.encode()).hexdigest()[:16]}_sm{arch}"')
text+=(ROOT/'persistent_replay_stress.inc').read_text()
(ROOT/'cuda_persistent_aligned.py').write_text(text)
print('Built SGLang-aligned BF16 activation boundaries; physical FP32 weighted output scratch retained.')
