// Apache-2.0. Native warp-sized MoE prototype.
// DQ7/KPERM packing and low-level primitives derive from vjeux/okc-2026
// and alygg77's Apache-2.0 challenge kernels; see ../research/alygg77_2_4015/NOTICE.
// New scheduling: independent one-warp tiles, TMA ring, separate GEMMs/finalize.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifndef WARP_NPAD
#define WARP_NPAD 8
#endif
#ifndef WARP_NS
#define WARP_NS 4
#endif
namespace native_moe {
constexpr int H=4096, ROWS=128, JOBS=128, NP=WARP_NPAD, NS=WARP_NS;
constexpr int STAGE=32, JE=STAGE+32*ROWS, JC=JE+JOBS, JR=JC+JOBS, META=JR+JOBS*16;
constexpr int BYTES=2304, NACC=NP/8;
static_assert(NP==8 || NP==16,"supported token tile");
static_assert(NS>=2 && NS<=8,"bounded TMA pipeline");
struct Params {
 const unsigned char *wa,*wb;
 const float *ga,*gb;
 const __nv_bfloat16 *norm,*x;
 const int *ids;
 const float *tw;
 int *meta;
 __nv_bfloat16 *inter;
 float *partial;
 __nv_bfloat16 *out;
 int m,lo,nloc,zero;
};
struct __align__(128) Shared {
 uint64_t bar[NS];
 __align__(128) unsigned char ring[NS][BYTES];
};
__device__ __forceinline__ uint32_t smaddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ int acquire(const int* p) {
 int v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ void wait(uint64_t* bar,uint32_t parity) {
 asm volatile("{ .reg .pred p; WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra WAIT_%=; }"::"r"(smaddr(bar)),"r"(parity):"memory");
}
__device__ __forceinline__ void issue(uint64_t* bar,void* dst,const void* src) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(smaddr(bar)),"r"(BYTES):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"::"r"(smaddr(dst)),"l"(src),"r"(BYTES),"r"(smaddr(bar)),"l"(0x12F0000000000000ull):"memory");
}
__device__ __forceinline__ uint32_t mullo(uint32_t a,uint32_t b) {
 const __nv_bfloat162 av=*reinterpret_cast<const __nv_bfloat162*>(&a),bv=*reinterpret_cast<const __nv_bfloat162*>(&b);
 const __nv_bfloat162 r=__hmul2(av,__low2bfloat162(bv));return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t mulhi(uint32_t a,uint32_t b) {
 const __nv_bfloat162 av=*reinterpret_cast<const __nv_bfloat162*>(&a),bv=*reinterpret_cast<const __nv_bfloat162*>(&b);
 const __nv_bfloat162 r=__hmul2(av,__high2bfloat162(bv));return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ void decode(uint32_t q,uint32_t &a,uint32_t &b,uint32_t &c,uint32_t &d) {
 constexpr uint32_t M=0x81C081C0u;
 a=q&M;b=(q*16u)&M;c=__funnelshift_l(q,q,10)&M;
 d=((q*64u)&0x80C080C0u)|(__funnelshift_l(q,q,14)&0x01000100u);
}
__device__ __forceinline__ void mma(float *d,uint32_t a,uint32_t b,uint32_t c,uint32_t e,uint32_t x,uint32_t y) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a),"r"(b),"r"(c),"r"(e),"r"(x),"r"(y));
}
__device__ __forceinline__ float bfround(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float gu(float g,float u) {
 g=bfround(g);u=bfround(u);return __fdividef(g,1.f+__expf(-g))*u;
}
__device__ __forceinline__ int colmap(int p) {
 const int v=p&255,s=v>>4,c=v&15;
 return (p&~255)+16*(4*((c>>1)&3)+(s&3))+4*(s>>2)+(c&1)+2*(c>>3);
}
// One elected, already-resident CTA produces fresh routing on every call.
// Finalize resets ready after the two GEMMs, including the no-local-work case.
__device__ void route(const Params &p) {
 const int lane=threadIdx.x;
 // Read first: CTAs arriving after routing starts need not join the single
 // address's CAS queue. Only contenders that observed ready0 attempt election.
 int won=0;
 if(lane==0 && acquire(p.meta)==0) won=atomicCAS(p.meta,0,1)==0;
 won=__shfl_sync(0xffffffffu,won,0);
 if(won) {
  const int E=p.m*8;
  const int4 v=4*lane<E?__ldg(reinterpret_cast<const int4*>(p.ids)+lane):make_int4(-1,-1,-1,-1);
  int cnt=0;
  for(int k=0;k<E/4;++k) {
   const int id[4]={__shfl_sync(0xffffffffu,v.x,k),__shfl_sync(0xffffffffu,v.y,k),__shfl_sync(0xffffffffu,v.z,k),__shfl_sync(0xffffffffu,v.w,k)};
#pragma unroll
   for(int h=0;h<4;++h) if(lane<p.nloc && id[h]==p.lo+lane) p.meta[STAGE+lane*ROWS+cnt++]=k*4+h;
  }
  const int nj=(cnt+NP-1)/NP;
  int inclusive=nj;
#pragma unroll
  for(int off=1;off<32;off*=2) {
   const int v=__shfl_up_sync(0xffffffffu,inclusive,off);
   if(lane>=off) inclusive+=v;
  }
  const int base=inclusive-nj,total=__shfl_sync(0xffffffffu,inclusive,31);
  for(int j=0;j<nj;++j) {
   const int count=min(NP,cnt-j*NP),q=base+j;
   p.meta[JE+q]=lane;p.meta[JC+q]=count;
   for(int r=0;r<count;++r) p.meta[JR+q*16+r]=p.meta[STAGE+lane*ROWS+j*NP+r];
  }
  if(lane==0) p.meta[1]=total;
  __threadfence();__syncwarp();
  if(lane==0) asm volatile("st.release.gpu.global.u32 [%0],2;"::"l"(p.meta):"memory");
 } else {
  while(acquire(p.meta)!=2) __nanosleep(32);
 }
 __syncwarp();
}

template<bool PB>
__global__ __launch_bounds__(32,16) void gemm(Params p) {
 __shared__ Shared S;
 const int lane=threadIdx.x,g=lane>>2,t=lane&3;
 if constexpr(!PB) route(p);
 const int nj=p.meta[1];
 constexpr int TILES=PB?256:512,SPLITS=PB?2:1;
 const unsigned char *weights=PB?p.wb:p.wa;
 if((int)blockIdx.x>=nj*TILES*SPLITS) return;
 if(lane==0) {
#pragma unroll
  for(int q=0;q<NS;++q)
   asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(S.bar+q)):"memory");
  asm volatile("fence.proxy.async.shared::cta;":::"memory");
 }
 __syncwarp();
 int sequence=0;
 for(int task=blockIdx.x;task<nj*TILES*SPLITS;task+=gridDim.x) {
  const int split=task%SPLITS,tile=(task/SPLITS)%TILES,job=task/(TILES*SPLITS);
  const int expert=p.meta[JE+job],count=p.meta[JC+job];
  const int s0=PB?split*8:0,s1=PB?s0+8:16;
  const unsigned char *base=weights+((size_t)expert*TILES+tile)*16*BYTES;
  int input_row[NACC];
#pragma unroll
  for(int nt=0;nt<NACC;++nt) input_row[nt]=(g+nt*8<count)?__ldg(p.meta+JR+job*16+g+nt*8):-1;
  float d[NACC][4];
#pragma unroll
  for(int nt=0;nt<NACC;++nt) {
#pragma unroll
   for(int k=0;k<4;++k) d[nt][k]=0.f;
  }
  if(lane==0) {
#pragma unroll
   for(int q=0;q<NS;++q) if(s0+q<s1) {
    const int slot=(sequence+q)%NS;
    issue(S.bar+slot,S.ring[slot],base+(size_t)(s0+q)*BYTES);
   }
  }
  const uint32_t k16c=16u+p.zero;
  for(int st=s0;st<s1;++st) {
   const int stage_sequence=sequence+st-s0,slot=stage_sequence%NS;
   wait(S.bar+slot,(stage_sequence/NS)&1);
   const unsigned char *r=S.ring[slot];
   uint4 next=*reinterpret_cast<const uint4*>(r+lane*16);
   const uint2 sf=*reinterpret_cast<const uint2*>(r+2048+lane*8);
#pragma unroll
   for(int lg=0;lg<4;++lg) {
    const uint4 codes=next;
    if(lg<3) next=*reinterpret_cast<const uint4*>(r+(lg+1)*512+lane*16);
    else {
     // Force every lane's final shared-memory payload into registers before
     // lane0 permits TMA to overwrite this slot. No device instruction emitted.
     asm volatile(""::"r"(codes.x),"r"(codes.y),"r"(codes.z),"r"(codes.w),"r"(sf.x),"r"(sf.y):"memory");
     __syncwarp();
     if(lane==0 && st+NS<s1) issue(S.bar+slot,S.ring[slot],base+(size_t)(st+NS)*BYTES);
    }
#pragma unroll
    for(int j=0;j<4;++j) {
     const uint32_t word=j==0?codes.x:j==1?codes.y:j==2?codes.z:codes.w;
     uint32_t a0,a1,a2,a3;decode(word,a0,a1,a2,a3);
     const uint32_t sw=j<2?sf.x:sf.y,sb=(j&1)*2;
     const uint32_t sc=__byte_perm(sw,0u,0x4040u|sb|((sb+1)<<8))*k16c+0x5C005C00u;
     a0=mullo(a0,sc);a1=mulhi(a1,sc);a2=mullo(a2,sc);a3=mulhi(a3,sc);
     const int kbase=st*256+64*t+16*j+4*lg;
#pragma unroll
     for(int nt=0;nt<NACC;++nt) {
      uint2 b=make_uint2(0,0);
      if(input_row[nt]>=0) {
       const __nv_bfloat16 *x=PB?p.inter+(size_t)input_row[nt]*H:p.x+(size_t)(input_row[nt]>>3)*H;
       b=__ldg(reinterpret_cast<const uint2*>(x+kbase));
      }
      mma(d[nt],a0,a1,a2,a3,b.x,b.y);
     }
    }
   }
  }
#pragma unroll
  for(int nt=0;nt<NACC;++nt) {
#pragma unroll
   for(int h=0;h<2;++h) {
    const int token=nt*8+2*t+h;
    if(token<count) {
     const int row=__ldg(p.meta+JR+job*16+token);
     if constexpr(PB) {
      float *dst=p.partial+((size_t)split*ROWS+row)*H+tile*16+g;
      dst[0]=d[nt][h];dst[8]=d[nt][2+h];
     } else {
      const float gs=__ldg(p.ga+expert)*0x1p62f;
      const float value=gu(d[nt][h]*gs,d[nt][2+h]*gs);
      p.inter[(size_t)row*H+colmap(tile*8+g)]=__float2bfloat16_rn(value);
     }
    }
   }
  }
  __syncwarp();
  sequence+=s1-s0;
 }
}

__global__ __launch_bounds__(256) void finalize(Params p) {
 __shared__ float red[8],scale;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,token=blockIdx.x;
 float acc[16],norm[16];
#pragma unroll
 for(int i=0;i<16;++i) {acc[i]=0.f;norm[i]=__bfloat162float(__ldg(p.norm+tid*16+i));}
#pragma unroll
 for(int slot=0;slot<8;++slot) {
  const int row=token*8+slot,expert=__ldg(p.ids+row)-p.lo;
  if(expert>=0 && expert<p.nloc) {
   const float gs=__ldg(p.gb+expert)*0x1p62f;
   float value[16],sum=0.f;
   const float *a=p.partial+(size_t)row*H+tid*16,*b=a+(size_t)ROWS*H;
#pragma unroll
   for(int i=0;i<16;++i) {
    value[i]=bfround((__ldg(a+i)+__ldg(b+i))*gs)*norm[i];
    sum+=value[i]*value[i];
   }
#pragma unroll
   for(int off=16;off;off>>=1) sum+=__shfl_xor_sync(0xffffffffu,sum,off);
   if(lane==0) red[warp]=sum;
   __syncthreads();
   if(tid==0) {
    float total=0.f;
#pragma unroll
    for(int w=0;w<8;++w) total+=red[w];
    scale=__ldg(p.tw+row)*rsqrtf(total*(1.f/H)+1e-8f);
   }
   __syncthreads();
#pragma unroll
   for(int i=0;i<16;++i) acc[i]+=scale*value[i];
   __syncthreads();
  }
 }
 uint4 v[2];uint32_t *words=reinterpret_cast<uint32_t*>(v);
#pragma unroll
 for(int i=0;i<8;++i) {
  const __nv_bfloat162 z=__floats2bfloat162_rn(acc[2*i],acc[2*i+1]);words[i]=*reinterpret_cast<const uint32_t*>(&z);
 }
 auto dst=reinterpret_cast<uint4*>(p.out+(size_t)token*H+tid*16);dst[0]=v[0];dst[1]=v[1];
 if(token==0 && tid==0) p.meta[0]=0;
}
} // namespace native_moe
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void warp_run(std::vector<torch::Tensor> w,std::vector<torch::Tensor> sc,torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor out,int64_t lo,int64_t nloc,int64_t grid) {
 TORCH_CHECK(x.size(0)>0 && x.size(0)<=WARP_NPAD,"native warp band bound");
 TORCH_CHECK(nloc<=32 && ((uintptr_t)ids.data_ptr()&15)==0,"native route bound/alignment");
 native_moe::Params p{};
 p.wa=(const unsigned char*)w[0].data_ptr();p.wb=(const unsigned char*)w[1].data_ptr();
 p.ga=(const float*)w[2].data_ptr();p.gb=(const float*)w[3].data_ptr();p.norm=(const __nv_bfloat16*)w[4].data_ptr();
 p.x=(const __nv_bfloat16*)x.data_ptr();p.ids=(const int*)ids.data_ptr();p.tw=(const float*)tw.data_ptr();
 p.meta=(int*)sc[0].data_ptr();p.inter=(__nv_bfloat16*)sc[1].data_ptr();p.partial=(float*)sc[2].data_ptr();p.out=(__nv_bfloat16*)out.data_ptr();
 p.m=(int)x.size(0);p.lo=(int)lo;p.nloc=(int)nloc;p.zero=0;
 auto stream=at::cuda::getCurrentCUDAStream();
 native_moe::gemm<false><<<(int)grid,32,0,stream>>>(p);
 native_moe::gemm<true><<<(int)grid,32,0,stream>>>(p);
 native_moe::finalize<<<p.m,256,0,stream>>>(p);
}
std::vector<int64_t> warp_resources() {
 int a=0,b=0;cudaFuncAttributes attr{};
 cudaError_t e=cudaFuncSetAttribute(native_moe::gemm<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 e=cudaFuncSetAttribute(native_moe::gemm<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a,native_moe::gemm<false>,32,0);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,native_moe::gemm<true>,32,0);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 e=cudaFuncGetAttributes(&attr,native_moe::gemm<false>);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
 return {(int64_t)a,(int64_t)b,(int64_t)attr.numRegs,(int64_t)attr.sharedSizeBytes};
}
