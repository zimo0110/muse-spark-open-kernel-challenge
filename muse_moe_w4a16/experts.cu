#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ uint32_t loadq(const void* p) {
  uint32_t r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 loadx(const void* p) {
  uint4 r;
  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"l"(p));
  return r;
}
__device__ __forceinline__ uint4 loadp(const void* p) {
 uint4 r;
 asm("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"l"(p));
 return r;
}
__device__ __forceinline__ uint32_t smem32(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void initbar(uint64_t* b) {
 asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smem32(b)));
 asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
}
__device__ __forceinline__ void copybulk(void* d,const void* p,int n,uint64_t* b) {
 asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(smem32(b)),"r"(n):"memory");
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"::"r"(smem32(d)),"l"(p),"r"(n),"r"(smem32(b)):"memory");
}
__device__ __forceinline__ void waitbar(uint64_t* b) {
 uint32_t done=0;
 while(!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }":"=r"(done):"r"(smem32(b)):"memory");
}
__device__ __forceinline__ uint16_t bits4(uint32_t c) {
  uint32_t m = c & 7;
  return (uint16_t)((m == 0 ? 0 : m == 1 ? 0x3f00 : (m + 252) << 6) | ((c & 8) << 12));
}
__device__ __forceinline__ __nv_bfloat162 scale2(uint8_t b) {
  union {uint32_t u;__nv_bfloat162 b;} v;
  uint32_t bits;
  if ((b & 0x78) != 0) bits = ((uint32_t)(b & 127) << 4) + (131u << 7) + ((b & 128u) << 8);
  else bits = __bfloat16_as_ushort(__float2bfloat16_rn((float)(b & 7) * 4.f * ((b & 128) ? -1.f : 1.f)));
  v.u=bits | (bits<<16);
  return v.b;
}
// FP4 is embedded as a tiny BF16 value. A power-of-two bias is moved into the
// block/global scales, as in Marlin, so decode uses masks/shifts and native mul.
__device__ __forceinline__ void decode8(uint32_t q, uint8_t s, uint32_t* d) {
  union { __nv_bfloat162 b; uint32_t u; } sc;
  sc.b=scale2(s);
  #pragma unroll
  for (int z=0;z<4;++z) {
    uint32_t a=q<<(12-4*z);
    uint32_t v=(a & 0x80008000u)|((a & 0x70007000u)>>6);
    asm("mul.rn.bf16x2 %0, %1, %2;":"=r"(d[z]):"r"(v),"r"(sc.u));
  }
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b,
                                     uint32_t x0,uint32_t x1,int j) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[j]),"r"(b[j]),"r"(a[j+1]),"r"(b[j+1]),"r"(x0),"r"(x1));
}

template<int NT, int WARPS, bool FIRST, int U>
__global__ void __launch_bounds__(WARPS*32) experts(
 const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
 const uint8_t* __restrict__ SF, const float* __restrict__ GS,
 const int* __restrict__ MAP,const int* __restrict__ COUNT,const int* __restrict__ ENDS,
 __nv_bfloat16* __restrict__ OUT,int M,int E,int start,int N,int K) {
 constexpr int COLS=NT*8;
 __shared__ float red[WARPS][16*COLS];
 __shared__ float redu[WARPS][16*COLS];
 const int total=ENDS[E-1]*(N/16);
 for(int work=blockIdx.x;work<total;work+=gridDim.x) {
 const int block=work/(N/16);
 int e=0;
 while(block >= ENDS[e]) ++e;
 const int localrow = (block - (e ? ENDS[e-1] : 0))*COLS;
 const int cnt=COUNT[e];
 const int eg=e+start;
 const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
 const int g=lane>>2,t=lane&3;
 const int slab=work%(N/16);
 const int tilebytes=(K/128)*1280;
 const uint8_t* buf=W+((size_t)eg*(FIRST ? N*2/16 : N/16)+slab)*tilebytes;
 float acc[NT][4], accu[NT][4];
 const __nv_bfloat16* xr[NT];
 bool valid[NT];
 #pragma unroll
 for(int j=0;j<NT;++j) {
   #pragma unroll
   for(int z=0;z<4;++z) acc[j][z]=accu[j][z]=0;
   int r=localrow+j*8+g;
   valid[j]=r<cnt;
   int route=valid[j] ? MAP[(size_t)e*M*8+r] : 0;
   xr[j]=X+(size_t)(FIRST ? route/8 : route)*K;
 }

 for(int g0=warp*U;g0<K/128;g0+=WARPS*U) {
   uint4 qa[U],qb[U],qu[U],qv[U];
   uint32_t sa[U],sb[U],su[U],sv[U];
   #pragma unroll
   for(int u=0;u<U;++u) {
     const uint8_t* p=buf+(g0+u)*1280;
     qa[u]=loadp(p+lane*16);qb[u]=loadp(p+512+lane*16);
     sa[u]=*(const uint32_t*)(p+1024+lane*4);sb[u]=*(const uint32_t*)(p+1152+lane*4);
     if constexpr(FIRST) {
       p+=(size_t)(N/16)*tilebytes;
       qu[u]=loadp(p+lane*16);qv[u]=loadp(p+512+lane*16);
       su[u]=*(const uint32_t*)(p+1024+lane*4);sv[u]=*(const uint32_t*)(p+1152+lane*4);
     }
   }
   #pragma unroll
   for(int u=0;u<U;++u) {
     #pragma unroll
     for(int c=0;c<4;++c) {
       uint32_t a[4],b[4],au[4],bu[4];
       const uint32_t* aq=(const uint32_t*)&qa[u];const uint32_t* bq=(const uint32_t*)&qb[u];
       decode8(aq[c],sa[u]>>(c*8),a);decode8(bq[c],sb[u]>>(c*8),b);
       if constexpr(FIRST) {
         const uint32_t* uq=(const uint32_t*)&qu[u];const uint32_t* vq=(const uint32_t*)&qv[u];
         decode8(uq[c],su[u]>>(c*8),au);decode8(vq[c],sv[u]>>(c*8),bu);
       }
       #pragma unroll
       for(int j=0;j<NT;++j) {
         uint4 v=valid[j] ? loadx(xr[j]+(g0+u)*128+c*32+t*8) : make_uint4(0,0,0,0);
         mma(acc[j],a,b,v.x,v.y,0);mma(acc[j],a,b,v.z,v.w,2);
         if constexpr(FIRST) {mma(accu[j],au,bu,v.x,v.y,0);mma(accu[j],au,bu,v.z,v.w,2);}
       }
     }
   }
 }
 #pragma unroll
 for(int j=0;j<NT;++j) {
   red[warp][g*COLS+j*8+2*t]=acc[j][0];red[warp][g*COLS+j*8+2*t+1]=acc[j][1];
   red[warp][(g+8)*COLS+j*8+2*t]=acc[j][2];red[warp][(g+8)*COLS+j*8+2*t+1]=acc[j][3];
   if constexpr(FIRST) {
     redu[warp][g*COLS+j*8+2*t]=accu[j][0];redu[warp][g*COLS+j*8+2*t+1]=accu[j][1];
     redu[warp][(g+8)*COLS+j*8+2*t]=accu[j][2];redu[warp][(g+8)*COLS+j*8+2*t+1]=accu[j][3];
   }
 }
 __syncthreads();
 float gs=GS[eg]*0x1p115f;
 for(int idx=threadIdx.x;idx<16*COLS;idx+=WARPS*32) {
   const int row=idx&15,col=idx>>4;
   if(localrow+col<cnt) {
     float v=0,u=0;
     #pragma unroll
     for(int q=0;q<WARPS;++q) {v+=red[q][row*COLS+col];if constexpr(FIRST) u+=redu[q][row*COLS+col];}
     v=__bfloat162float(__float2bfloat16_rn(v*gs));
     if constexpr(FIRST) {
       u=__bfloat162float(__float2bfloat16_rn(u*gs));
       v=__bfloat162float(__float2bfloat16_rn(v/(1.f+__expf(-v))))*u;
     }
     int route=MAP[(size_t)e*M*8+localrow+col];
     OUT[(size_t)route*N+slab*16+row]=__float2bfloat16_rn(v);
   }
 }
 __syncthreads();
 }
}

void run(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,
 torch::Tensor map,torch::Tensor count,torch::Tensor ends,torch::Tensor out,
 int64_t M,int64_t E,int64_t start,int64_t N,int64_t K,bool first,int64_t nt,int64_t warps,int64_t U) {
 const int cols=nt*8;
 const int smem=128+(K/128)*1280*(first ? 2 : 1);
 const int blocks=std::min((int)M*8,(int)E+((int)M*8+cols-1)/cols-1);
 dim3 grid(132*4);
 auto stream=at::cuda::getCurrentCUDAStream();
 #define GO(T,W,F,U_) if(nt==T && warps==W && first==F && U==U_) { \
 experts<T,W,F,U_><<<grid,W*32,0,stream>>>( \
 (const __nv_bfloat16*)x.data_ptr(),w.data_ptr<uint8_t>(),(const uint8_t*)sf.data_ptr(),gs.data_ptr<float>(), \
 map.data_ptr<int>(),count.data_ptr<int>(),ends.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),M,E,start,N,K);return;}
 #define GU(T,W,F) GO(T,W,F,1) GO(T,W,F,2) GO(T,W,F,4)
 #define GF(T,W) GU(T,W,true) GU(T,W,false)
 GF(1,4) GF(1,8) GF(2,4) GF(2,8) GF(4,4) GF(4,8)
 TORCH_CHECK(false,"unsupported configuration");
}
