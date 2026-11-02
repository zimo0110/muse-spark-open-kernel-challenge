"""Disjoint FP32 split-K partials: retain packed release-count protocol.

No dispatch. 16 possible K-start positions identify exclusive partial writers.
The counter packs sum(stages) in low5 bits and the distinct start bits above;
there is no low-field carry (sum<=16) or repeated start bit. The designated
finisher inserts its own start bit after acquiring the other contributors.
No temporary converted weights; additional workspace contains FP32 activations.
"""
import ast,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_aligned_safe.py').read_text();tree=ast.parse(text)
def assignment(name):
 return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
src=ast.literal_eval(assignment('CUDA_SRC').value)
src=src.replace('struct WgsPend { int kind, u, slot, nst; };','struct WgsPend { int kind, u, slot, nst, start; };')
# Every split-unit slot owns 16 banks, indexed by the writer's starting K stage.
src=src.replace('(size_t)pb * WGS_SLOTS * 4 * WGS_FIXW','(size_t)pb * WGS_SLOTS * 4 * WGS_FIXW * 16')
src=src.replace('(size_t)2 * WGS_SLOTS * 4 * WGS_FIXW','(size_t)2 * WGS_SLOTS * 4 * WGS_FIXW * 16')
src=src.replace('(size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW)', '(size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW * 16)')
a=src.index('__device__ __forceinline__ void wgs_flush(');b=src.index('template <bool PB, int NTT>',a)
body=src[a:b]
body=body.replace('  constexpr bool DF =', '  const int contribution = pd.nst + ((1 << pd.start) << 5);\n  constexpr bool DF =')
body=body.replace('"r"(pd.nst)', '"r"(contribution)')
old='  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }'
new='''  if (wt == 0) {
    int observed;
    while (((observed = ld_acquire(fc)) & 31) < 16 - pd.nst) __nanosleep(32);
    S.wglast[wg] = (observed >> 5) | (1 << pd.start);
  }'''
assert old in body;body=body.replace(old,new)
body=body.replace('S.wglast[wg] = old + pd.nst == 16;', 'S.wglast[wg] = ((old & 31) + pd.nst == 16) ? ((old >> 5) | (1 << pd.start)) : 0;')
body=body.replace('  const bool last = S.wglast[wg];', '  const bool last = S.wglast[wg] != 0;')
body=body.replace('(size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4','(size_t)(pd.slot * 4 + wg) * WGS_FIXW * 16 + wt * 4')
a2=body.index('  // two n-tiles');b2=body.index('  if (wt == 0) *fc = 0;',a2)
body=body[:a2]+'''  // Each bank has one writer. Acquire the contributors' packed release count,
  // then sum the live banks in increasing K order; no data atomics or zeroing.
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 value[2] = {make_float4(0.f,0.f,0.f,0.f),make_float4(0.f,0.f,0.f,0.f)};
    unsigned mask = (unsigned)S.wglast[wg];
    while (mask) {
      const int start = __ffs(mask)-1;
      float4 part[2];
#pragma unroll
      for (int k=0;k<2;++k)
        if (nt0+k<ntt) part[k] = __ldcg(reinterpret_cast<const float4*>(fb + (size_t)start * WGS_FIXW + (nt0+k) * 512));
#pragma unroll
      for (int k=0;k<2;++k) if (nt0+k<ntt) {
        value[k].x += part[k].x; value[k].y += part[k].y; value[k].z += part[k].z; value[k].w += part[k].w;
      }
      mask &= mask-1;
    }
#pragma unroll
    for (int k=0;k<2;++k) if (nt0+k<ntt) {
      const float d4[4] = {value[k].x,value[k].y,value[k].z,value[k].w};
      wgs_epi_tile<PB>(p,S,j,r16,nt0+k,d4);
    }
  }
'''+body[b2:]
burst=int(sys.argv[1]) if len(sys.argv)>1 else 1
assert burst in (1,2)
body=body.replace('nt0 += 2',f'nt0 += {burst}').replace('float4 value[2] = {make_float4(0.f,0.f,0.f,0.f),make_float4(0.f,0.f,0.f,0.f)};', 'float4 value[1] = {make_float4(0.f,0.f,0.f,0.f)};' if burst==1 else 'float4 value[2] = {make_float4(0.f,0.f,0.f,0.f),make_float4(0.f,0.f,0.f,0.f)};').replace('float4 part[2]',f'float4 part[{burst}]').replace('k<2',f'k<{burst}')
src=src[:a]+body+src[b:]
old='''    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");'''
new='''    float* fb = wgs_fix(p, PB) + ((size_t)(slot * 4 + wg) * 16 + s0) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(d[nt][0],d[nt][1],d[nt][2],d[nt][3]);'''
assert src.count(old)==1;src=src.replace(old,new)
src=src.replace('WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0}', 'WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0, s0}')
# Replace only embedded source and Python scratch formula (not other historical paths).
node=assignment('CUDA_SRC');lines=text.splitlines(keepends=True)
a=sum(map(len,lines[:node.value.lineno-1]))+node.value.col_offset;b=sum(map(len,lines[:node.value.end_lineno-1]))+node.value.end_col_offset
text=text[:a]+repr(src)+text[b:]
text=text.replace('WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW +', 'WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW * 16 +')
ast.parse(text);(root/'cuda_persistent_private_fixup.py').write_text(text)
defs=ast.literal_eval(assignment('MAIN_DEFINES').value)
Path('/tmp/muse_private_fixup_cases.json').write_text(json.dumps([{'name':'private_fixup','source':src,'defines':defs,'tiny':False}]))
Path('/tmp/muse_private_fixup.cu').write_text(src)
assert 'red.global.add.v4.f32' not in src
assert src.count('WGS_FIXW * 16')==5,src.count('WGS_FIXW * 16')
print('Prepared exclusive split partial banks; H200 UNVERIFIED. Additional workspace about300MiB.')
