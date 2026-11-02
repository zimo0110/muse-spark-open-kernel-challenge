"""Build BF16 physical scratch experiment from immutable checked numeric port."""
import ast
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
base = ROOT / 'cuda_persistent_bf16.py'
text = base.read_text()
assert hashlib.sha256(base.read_bytes()).hexdigest() == '91766194b46a086f3d772238d25f86e87c4668109384798bb8245f4f66f7d9dd'
helper = r'''
// The phase-B values are already BF16-rounded. Store exactly those bits.
__device__ __forceinline__ void st_keep_f32(__nv_bfloat16* ptr, float value) {
  st_keep_bf16(ptr, __float2bfloat16_rn(value));
}
__device__ __forceinline__ float4 codex_load_y4(const __nv_bfloat16* ptr) {
  const uint2 bits = __ldcg(reinterpret_cast<const uint2*>(ptr));
  const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.x));
  const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}
'''
def patch(src):
    assert src.count('float* ybuf;') == 1
    src = src.replace('float* ybuf;', '__nv_bfloat16* ybuf;')
    # Preserve FP32 stream-K/fixup allocations and offsets after the original
    # full-size reserved ybuf region. The unused padding cannot alias ybuf.
    src = src.replace('p.ybuf + (size_t)EMAX * HD', 'reinterpret_cast<float*>(p.ybuf) + (size_t)EMAX * HD')
    src = src.replace('p.ybuf + (size_t)okc::EMAX * okc::HD', 'reinterpret_cast<float*>(p.ybuf) + (size_t)okc::EMAX * okc::HD')
    assert src.count('p.ybuf = p.ss + okc::EMAX;') == 1
    src = src.replace('p.ybuf = p.ss + okc::EMAX;', 'p.ybuf = reinterpret_cast<__nv_bfloat16*>(p.ss + okc::EMAX);')
    marker = 'constexpr int HD = 4096, ID = 4096;'
    assert src.count(marker)==1
    src = src.replace(marker, helper+'\n'+marker)
    old='const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;'
    assert src.count(old)==2
    src=src.replace(old,'const __nv_bfloat16* src = p.ybuf + (size_t)qv[sl] * HD + tid * 8;')
    assert src.count('const float4 a = __ldcg(src), b = __ldcg(src + 1);')==2
    src=src.replace('const float4 a = __ldcg(src), b = __ldcg(src + 1);','const float4 a = codex_load_y4(src), b = codex_load_y4(src + 4);')
    for idx in ('qvp[s]', 'q'):
        old=f'__ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t){idx} * HD + col))'
        assert src.count(old)==1
        src=src.replace(old,f'codex_load_y4(p.ybuf + (size_t){idx} * HD + col)')
    old='const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;'
    assert src.count(old)==1
    src=src.replace(old,'const __nv_bfloat16* src = p.ybuf + (size_t)q * HD + tid * 16;')
    assert src.count('float4 v = __ldcg(src + i);')==1
    src=src.replace('float4 v = __ldcg(src + i);','float4 v = codex_load_y4(src + i * 4);')
    assert 'reinterpret_cast<const float4*>(p.ybuf' not in src
    return src
lines=text.splitlines(keepends=True)
replacements=[]
for n in ast.parse(text).body:
    if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id=='CUDA_SRC') or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
        start=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset
        end=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
        replacements.append((start,end,repr(patch(ast.literal_eval(n.value)))))
assert len(replacements)==2
for start,end,new in reversed(replacements): text=text[:start]+new+text[end:]
text=text.replace('okc_moe_w4a16_bf16stage_', 'okc_moe_w4a16_bf16compact_')
text=text.replace('# Local Apache-2.0 port: bf16stage;', '# Local Apache-2.0 port: physical BF16 scratch;')
(ROOT/'cuda_persistent_compact.py').write_text(text)
print('Generated compact BF16 scratch candidate; unchanged arithmetic and FP32 fixup layout.')
text += (ROOT/'persistent_replay_stress.inc').read_text()
(ROOT/'cuda_persistent_compact.py').write_text(text)
print('Appended prepare-only mixed-shape graph replay validation.')

# Keep extension identifiers short: long MID names failed dynamic import on H200.
text=text.replace('from typing import TypedDict', 'from typing import TypedDict\nimport hashlib')
text=text.replace('name=f"okc_moe_w4a16_bf16compact_{tag}_sm{arch}"', 'name=f"okc_moe_cx_{hashlib.sha256(tag.encode()).hexdigest()[:16]}_sm{arch}"')
(ROOT/'cuda_persistent_compact_v2.py').write_text(text)
