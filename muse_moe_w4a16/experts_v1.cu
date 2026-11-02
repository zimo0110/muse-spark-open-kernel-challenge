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
__device__ __forceinline__ uint16_t bits4(uint32_t c) {
  uint32_t m = c & 7;
  return (uint16_t)((m == 0 ? 0 : m == 1 ? 0x3f00 : (m + 252) << 6) | ((c & 8) << 12));
}
__device__ __forceinline__ uint32_t decodepair(uint32_t c, __nv_bfloat162 scale) {
  union { uint32_t u; __nv_bfloat162 b; } v;
  v.u = (uint32_t)bits4(c & 15) | ((uint32_t)bits4((c >> 4) & 15) << 16);
  v.b = __hmul2(v.b, scale);
  return v.u;
}
__device__ __forceinline__ __nv_bfloat162 scale2(uint8_t b) {
  __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
  __half h = hr;
  return __float2bfloat162_rn(__half2float(h));
}
__device__ __forceinline__ void decode8(uint32_t q, uint8_t s, uint32_t* d) {
  __nv_bfloat162 sc = scale2(s);
  #pragma unroll
  for (int z=0;z<4;++z) d[z] = decodepair(q >> (8*z),sc);
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
 const int block=blockIdx.y;
 if (block >= ENDS[E-1]) return;
 int e=0;
 while(block >= ENDS[e]) ++e;
 const int localrow = (block - (e ? ENDS[e-1] : 0))*COLS;
 const int cnt=COUNT[e];
 const int eg=e+start;
 const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
 const int g=lane>>2,t=lane&3;
 const int slab=blockIdx.x;
 const size_t row0=((size_t)eg*(FIRST ? N*2 : N)+slab*16+g);
 const uint8_t* w0=W+row0*(K/2);
 const uint8_t* w1=w0+(size_t)8*(K/2);
 const uint8_t* s0=SF+row0*(K/16);
 const uint8_t* s1=s0+(size_t)8*(K/16);
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
 for(int c0=warp*U;c0<K/32;c0+=WARPS*U) {
   uint32_t qa[U],qb[U],qu[U],qv[U];
   uint8_t sa[U],sb[U],su[U],sv[U];
   #pragma unroll
   for(int u=0;u<U;++u) {
     int k=(c0+u)*32+t*8;
     qa[u]=loadq(w0+k/2); qb[u]=loadq(w1+k/2);
     sa[u]=s0[k/16]; sb[u]=s1[k/16];
     if constexpr(FIRST) {
       qu[u]=loadq(w0+(size_t)N*(K/2)+k/2); qv[u]=loadq(w1+(size_t)N*(K/2)+k/2);
       su[u]=s0[(size_t)N*(K/16)+k/16]; sv[u]=s1[(size_t)N*(K/16)+k/16];
     }
   }
   #pragma unroll
   for(int u=0;u<U;++u) {
     uint32_t a[4],b[4],au[4],bu[4];
     decode8(qa[u],sa[u],a);decode8(qb[u],sb[u],b);
     if constexpr(FIRST) {decode8(qu[u],su[u],au);decode8(qv[u],sv[u],bu);}
     #pragma unroll
     for(int j=0;j<NT;++j) {
       uint4 v=valid[j] ? loadx(xr[j]+(c0+u)*32+t*8) : make_uint4(0,0,0,0);
       mma(acc[j],a,b,v.x,v.y,0);mma(acc[j],a,b,v.z,v.w,2);
       if constexpr(FIRST) {mma(accu[j],au,bu,v.x,v.y,0);mma(accu[j],au,bu,v.z,v.w,2);}
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
 float gs=GS[eg];
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
}

void run(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,
 torch::Tensor map,torch::Tensor count,torch::Tensor ends,torch::Tensor out,
 int64_t M,int64_t E,int64_t start,int64_t N,int64_t K,bool first,int64_t nt,int64_t warps,int64_t U) {
 const int cols=nt*8;
 const int blocks=std::min((int)M*8,(int)E+((int)M*8+cols-1)/cols-1);
 dim3 grid(N/16,blocks);
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
