"""Probe small-batch residency; retain full-K arithmetic and packed-weight ABI.

NW4 maps each compute warp to two original 512-K slices. Scale vectors stay
in their original eight-slice layout. TINY never enters the fat path, so its
launcher reserves only the skinny ring. A two-block grid MUST be gated by
runtime CUDA occupancy before launch because this kernel has a grid barrier.
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
t=ast.parse((R/'cuda_persistent_aligned_safe.py').read_text())
def val(k):return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id==k for v in n.targets))
s=val('CUDA_SRC')
# Correct NW4 scale addressing only inside the reachable WS compute path.
a=s.index('__device__ void skinny_ring_ws(');b=s.index('__device__ __forceinline__ void ws_issue(',a)
part=s[a:b]
old='    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);'
assert part.count(old)==1
part=part.replace(old,'''#if OKC_NW == 4
    const uint4 sc0=*reinterpret_cast<const uint4*>(slot+UNIT_CODES+warp*1024+lane*16);
    const uint4 sc1=*reinterpret_cast<const uint4*>(slot+UNIT_CODES+warp*1024+512+lane*16);
#else
    const uint4 sc=*reinterpret_cast<const uint4*>(slot+UNIT_CODES+warp*512+lane*16);
#endif''')
marker='    for (int c = 0; c < NCH; ++c) {'
assert part.count(marker)==1
part=part.replace(marker,marker+'''
#if OKC_NW == 4
      const uint4 sc=c<8?sc0:sc1;
      const int cc=c&7;
#else
      const int cc=c;
#endif''')
part=part.replace('c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)','cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w)')
s=s[:a]+part+s[b:]
marker='  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;'
assert s.count(marker)==1
s=s.replace(marker,'#if OKC_TINY\n  return base+xa;\n#endif\n'+marker)
# Add an ordinary host resource query. It is preparation-only.
marker='void okc_run(std::vector<torch::Tensor> w,'
query='''std::vector<int64_t> okc_resources() {
 size_t sm=okc_smem_bytes();
 cudaError_t e=cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
 TORCH_CHECK(e==cudaSuccess,"shared-memory attribute: ",cudaGetErrorString(e));
 e=cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
 TORCH_CHECK(e==cudaSuccess,"shared-memory carveout: ",cudaGetErrorString(e));
 int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,okc::moe_fused<false>,okc::NTL,sm);
 TORCH_CHECK(e==cudaSuccess,"occupancy: ",cudaGetErrorString(e));
 cudaFuncAttributes at;e=cudaFuncGetAttributes(&at,okc::moe_fused<false>);
 TORCH_CHECK(e==cudaSuccess,"kernel attributes: ",cudaGetErrorString(e));
 return{active,at.numRegs,(int64_t)sm,okc::NTL};
}
'''
assert s.count(marker)==1;s=s.replace(marker,query+'\n'+marker)
cases=[]
for nw,blocks in ((8,1),(8,2),(4,1),(4,2)):
 code=s.replace('__launch_bounds__(NTL, 1)',f'__launch_bounds__(NTL, {blocks})')
 defs=tuple(d for d in val('TINY_DEFINES') if not d.startswith('-DOKC_NW='))+(f'-DOKC_NW={nw}','-DOKC_NS=2','-DOKC_NSF=1','-DOKC_SPLIT=1')
 cases.append(dict(name=f'skinny_nw{nw}_b{blocks}',source=code,defines=defs,tiny=True))
(R/'research/spark/skinny_residency_cases.json').write_text(json.dumps(cases)+'\n')
print('Prepared four occupancy compiler probes. No target speed/correctness claim.')
