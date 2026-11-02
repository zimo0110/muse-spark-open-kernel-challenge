"""Experimental epilogue service warps; no selection or speed claim.

Four WGMMA consumer groups publish completed FP32 accumulator tiles into
single-slot shared mailboxes. Two helper warps drain those mailboxes and
perform the original BF16 epilogues. Split units use atomic last-arrival,
never a helper waiting on a future counter. Shared acquire/release and
warpgroup barriers protect both directions of mailbox ownership.

PTX ordering: https://docs.nvidia.com/cuda/parallel-thread-execution/
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
tree=ast.parse((R/'cuda_persistent_aligned_safe.py').read_text())
def val(key):
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
s=val('CUDA_SRC')
def sub(old,new):
 global s
 assert s.count(old)==1,old[:90]
 s=s.replace(old,new)
sub('NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0)','NT + (OKC_WGS ? 128 : OKC_WS ? 32 : 0)')
sub('  int wglast[4];', '  uint32_t ep_ready[4], ep_free[4], ep_done[4];\n  int ep_desc[4][5];\n  int wglast[4];')
sub('#define OKC_WGSNX 3','#define OKC_WGSNX 2')
sub('constexpr int WGS_NS = 3;','constexpr int WGS_NS = 2;')
sub('+ 136 * 16;  // + doneAs','+ JMAX * 16;  // + doneAs')
# Forward declaration before producer/consumer functions.
a=s.index('struct WgsPend'); b=s.index('template <bool PB>\n__device__ __forceinline__ void wgs_flush',a)
epi_end=b
# Retain wgs_epi_tile, replace legacy deferred fixup with a warp service.
end=s.index('// one segment (stages',b)
helpers=r'''
__device__ __forceinline__ uint32_t ep_acquire(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
  return v;
}
__device__ __forceinline__ void ep_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.cta.shared.u32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v) : "memory");
}
__device__ __forceinline__ float* ep_payload(unsigned char* ring, int wg) {
  return reinterpret_cast<float*>(ring + WGS_NS*UNIT_BYTES + WGS_NX*WGS_XS) + wg*WGS_FIXW;
}
template <bool PB>
__device__ __forceinline__ void ep_task(const Params& p, Smem& S, unsigned char* ring, int wg) {
  const int lane=threadIdx.x&31;
  const int u=S.ep_desc[wg][0], slot=S.ep_desc[wg][1], nst=S.ep_desc[wg][2], ntt=S.ep_desc[wg][3];
  constexpr int UBL=PB?4:5;
  const int j=u>>UBL, ub=u&((1<<UBL)-1);
  const float* payload=ep_payload(ring,wg);
  if(slot>=0) {
    float* fb=wgs_fix(p,PB)+(size_t)(slot*4+wg)*WGS_FIXW;
    for(int nt=0;nt<ntt;++nt) {
      float4 v[4];
#pragma unroll
      for(int wp=0;wp<4;++wp) v[wp]=*reinterpret_cast<const float4*>(payload+nt*512+(wp*32+lane)*4);
#pragma unroll
      for(int wp=0;wp<4;++wp)
        asm volatile("red.global.add.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(fb+nt*512+(wp*32+lane)*4),"f"(v[wp].x),"f"(v[wp].y),"f"(v[wp].z),"f"(v[wp].w):"memory");
    }
    // All data reductions precede publishing this share's stage count.
    __threadfence(); __syncwarp();
    int last=0; int* fc=wgs_cnt(p,PB)+slot*4+wg;
    if(lane==0) {
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old):"l"(fc),"r"(nst):"memory");
      last=old+nst==16;
    }
    last=__shfl_sync(0xffffffffu,last,0);
    if(!last) return;
    asm volatile("fence.acq_rel.gpu;":::"memory");
    for(int nt=0;nt<ntt;++nt) {
      float4 v[4];
#pragma unroll
      for(int wp=0;wp<4;++wp) v[wp]=__ldcg(reinterpret_cast<const float4*>(fb+nt*512+(wp*32+lane)*4));
#pragma unroll
      for(int wp=0;wp<4;++wp) {
        *reinterpret_cast<float4*>(fb+nt*512+(wp*32+lane)*4)=make_float4(0.f,0.f,0.f,0.f);
        const float q[4]={v[wp].x,v[wp].y,v[wp].z,v[wp].w};
        wgs_epi_tile<PB>(p,S,j,ub*16+wg*4+wp,nt,q);
      }
    }
    if(lane==0) *fc=0;
  } else {
    for(int nt=0;nt<ntt;++nt) {
      float4 v[4];
#pragma unroll
      for(int wp=0;wp<4;++wp) v[wp]=*reinterpret_cast<const float4*>(payload+nt*512+(wp*32+lane)*4);
#pragma unroll
      for(int wp=0;wp<4;++wp) {
        const float q[4]={v[wp].x,v[wp].y,v[wp].z,v[wp].w};
        wgs_epi_tile<PB>(p,S,j,ub*16+wg*4+wp,nt,q);
      }
    }
  }
  __threadfence(); __syncwarp();
  if constexpr(!PB) if(lane==0)
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p.doneAs+j*16+(ub>>1)),"r"(1):"memory");
}
__device__ void ep_helper(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();
  const int lane=threadIdx.x&31, base=((threadIdx.x-NT-64)>>5)*2;
  uint32_t consumed[2]={0,0};
  while(true) {
    int finished=0;
#pragma unroll
    for(int k=0;k<2;++k) {
      const int wg=base+k;
      // Acquire done BEFORE ready, so termination cannot miss its last task.
      uint32_t done=0,ready=0;
      if(lane==0) { done=ep_acquire(S.ep_done+wg); ready=ep_acquire(S.ep_ready+wg); }
      done=__shfl_sync(0xffffffffu,done,0); ready=__shfl_sync(0xffffffffu,ready,0);
      __syncwarp();
      if(ready!=consumed[k]) {
        if(S.ep_desc[wg][4]) ep_task<true>(p,S,ring,wg); else ep_task<false>(p,S,ring,wg);
        __syncwarp();
        consumed[k]=ready;
        if(lane==0) ep_release(S.ep_free+wg,ready);
      }
      finished += done && ready==consumed[k];
    }
    if(finished==2) break;
  }
  asm volatile("bar.sync 8, %0;" :: "n"(NT+64):"memory");
}
'''
s=s[:b]+helpers+s[end:]
# Only the consumer segment/phase context uses epseq (old producer unchanged).
a=s.index('__device__ __forceinline__ void wgs_seg(');b=s.index('// cost-weighted stage ranges:',a)
part=s[a:b].replace('int& seq, WgsPend& pd)','int& seq, int& epseq)')
part=part.replace('if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);','// Epilogues are drained independently by service warps.')
a2=part.index('  if (OKC_WGSX & 8)'); b2=part.index('\n}\ntemplate <bool PB>',a2)
part=part[:a2]+r'''
  if(wt==0) while(ep_acquire(S.ep_free+wg)!=(uint32_t)epseq) __nanosleep(32);
  asm volatile("bar.sync %0, 128;" ::"r"(3+wg):"memory");
  float* dst=ep_payload(ring,wg)+wt*4;
#pragma unroll
  for(int nt=0;nt<NTT;++nt) *reinterpret_cast<float4*>(dst+nt*512)=make_float4(d[nt][0],d[nt][1],d[nt][2],d[nt][3]);
  asm volatile("bar.sync %0, 128;" ::"r"(3+wg):"memory");
  if(wt==0) {
    S.ep_desc[wg][0]=u; S.ep_desc[wg][1]=slot; S.ep_desc[wg][2]=s1-s0;
    S.ep_desc[wg][3]=NTT; S.ep_desc[wg][4]=PB;
    ep_release(S.ep_ready+wg,epseq+1);
  }
  ++epseq;'''+part[b2:]
part=part.replace('int g0, int g1, int& seq)','int g0, int g1, int& seq, int& epseq)')
part=part.replace('  WgsPend pd{0, 0, 0, 0};\n','').replace('slot, seq, pd)','slot, seq, epseq)')
part=part.replace('  wgs_flush<PB>(p, S, pd);\n','')
s=s[:a]+part+s[b:]
sub('    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));','    else if(tid<NT+64) wgs_xproducer(p,S,reinterpret_cast<unsigned char*>(xbuf));\n    else ep_helper(p,S,reinterpret_cast<unsigned char*>(xbuf));')
sub('    S.preissuedB = 0;','    S.preissuedB = 0;\n#if OKC_WGS\n    for(int q=0;q<4;++q) S.ep_ready[q]=S.ep_free[q]=S.ep_done[q]=0;\n#endif')
sub('    int seq = 0;\n    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);','    int seq = 0, epseq=0;\n    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq, epseq);')
sub('    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);','    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq, epseq);\n    if((tid&127)==0) ep_release(S.ep_done+(tid>>7),1);\n    asm volatile("bar.sync 8, %0;" ::"n"(NT+64):"memory");')
sub('return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;','return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS + (size_t)4*okc::WGS_FIXW*sizeof(float);')
(R/'research/spark/epilogue_warps.cu').write_text(s)
cases=[dict(name='epilogue_warps',source=s,defines=val('MAIN_DEFINES'),tiny=False)]
(R/'research/spark/epilogue_warps_cases.json').write_text(json.dumps(cases)+'\n')
print('Prepared epilogue-warp prototype. Correctness and performance unverified.')
