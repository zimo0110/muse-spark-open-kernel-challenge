// Independent FP32 diagnostic reference. Never used by the timed kernel.
// NVFP4 weights stay packed; scalar weights are decoded inside CUDA kernels.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace independent_moe {
constexpr int H=4096;
struct P {
 const unsigned char *a,*sa,*b,*sb;
 const float *ga,*gb;
 const __nv_bfloat16 *x,*nw;
 const int *ids;
 const float *tw;
 float *inter,*y;
 __nv_bfloat16 *out;
 int m,lo,nloc;
};
__device__ __forceinline__ float fp4(unsigned v) {
 const unsigned a=v&7;
 float r=a<2?0.5f*a:__uint_as_float(((126u+(a>>1))<<23)|((a&1)<<22));
 return v&8?-r:r;
}
__device__ __forceinline__ float fp8(unsigned v) {
 const unsigned a=v&127;
 float r=a<8?a*(1.f/512.f):a==127?__uint_as_float(0x7fffffffu):__uint_as_float((((a>>3)+120u)<<23)|((a&7)<<20));
 return v&128?-r:r;
}
__device__ __forceinline__ float weight(const unsigned char *w,const unsigned char *sf,size_t row,int k,float gs) {
 unsigned code=__ldg(w+row*2048+(k>>1));code=(code>>((k&1)*4))&15;
 return (fp4(code)*fp8(__ldg(sf+row*256+(k>>4))))*gs;
}
__device__ __forceinline__ bool routed(P p,int e,int t) {
 bool active=false;
#pragma unroll
 for(int s=0;s<8;++s) active|=__ldg(p.ids+t*8+s)==e;
 return active;
}
__global__ __launch_bounds__(256) void first(P p) {
 const int local=blockIdx.z,e=p.lo+local,t=blockIdx.y;
 if(local>=p.nloc || !routed(p,e,t))return;
 const int lane=threadIdx.x&31,col=blockIdx.x*8+(threadIdx.x>>5);
 float a=0.f,b=0.f;const float gs=__ldg(p.ga+e);
 for(int k=lane;k<H;k+=32) {
  const float x=__bfloat162float(__ldg(p.x+(size_t)t*H+k));
  a=fmaf(weight(p.a,p.sa,(size_t)e*8192+col,k,gs),x,a);
  b=fmaf(weight(p.a,p.sa,(size_t)e*8192+4096+col,k,gs),x,b);
 }
#pragma unroll
 for(int off=16;off;off>>=1) {a+=__shfl_down_sync(0xffffffffu,a,off);b+=__shfl_down_sync(0xffffffffu,b,off);}
 if(lane==0)p.inter[((size_t)local*p.m+t)*H+col]=__fdividef(a,1.f+__expf(-a))*b;
}
__global__ __launch_bounds__(256) void second(P p) {
 const int local=blockIdx.z,e=p.lo+local,t=blockIdx.y;
 if(local>=p.nloc || !routed(p,e,t))return;
 const int lane=threadIdx.x&31,col=blockIdx.x*8+(threadIdx.x>>5);
 float y=0.f;const float gs=__ldg(p.gb+e);
 for(int k=lane;k<H;k+=32)
  y=fmaf(weight(p.b,p.sb,(size_t)e*H+col,k,gs),__ldg(p.inter+((size_t)local*p.m+t)*H+k),y);
#pragma unroll
 for(int off=16;off;off>>=1)y+=__shfl_down_sync(0xffffffffu,y,off);
 if(lane==0)p.y[((size_t)local*p.m+t)*H+col]=y;
}
__global__ __launch_bounds__(256) void finish(P p) {
 __shared__ float red[8],scale;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,t=blockIdx.x;
 float acc[16],norm[16];
#pragma unroll
 for(int i=0;i<16;++i){acc[i]=0.f;norm[i]=1.f+__bfloat162float(__ldg(p.nw+tid*16+i));}
#pragma unroll
 for(int s=0;s<8;++s) {
  const int e=__ldg(p.ids+t*8+s)-p.lo;
  if(e>=0 && e<p.nloc) {
   float v[16],sum=0.f;
#pragma unroll
   for(int i=0;i<16;++i) {v[i]=__ldg(p.y+((size_t)e*p.m+t)*H+tid*16+i)*norm[i];sum+=v[i]*v[i];}
#pragma unroll
   for(int off=16;off;off>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,off);
   if(lane==0)red[warp]=sum;
   __syncthreads();
   if(tid==0){float all=0.f;for(int w=0;w<8;++w)all+=red[w];scale=__ldg(p.tw+t*8+s)*rsqrtf(all*(1.f/H)+1e-8f);}
   __syncthreads();
#pragma unroll
   for(int i=0;i<16;++i)acc[i]+=scale*v[i];
   __syncthreads();
  }
 }
#pragma unroll
 for(int i=0;i<16;++i)p.out[(size_t)t*H+tid*16+i]=__float2bfloat16_rn(acc[i]);
}
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void diagnostic_reference(std::vector<torch::Tensor> w,torch::Tensor x,torch::Tensor ids,torch::Tensor tw,torch::Tensor inter,torch::Tensor y,torch::Tensor out,int64_t lo,int64_t cnt) {
 TORCH_CHECK(x.size(0)>0 && x.size(0)<=16 && cnt<=32,"bounded diagnostic reference");
 independent_moe::P p{};
 p.a=(const unsigned char*)w[0].data_ptr();p.sa=(const unsigned char*)w[1].data_ptr();p.ga=(const float*)w[2].data_ptr();
 p.b=(const unsigned char*)w[3].data_ptr();p.sb=(const unsigned char*)w[4].data_ptr();p.gb=(const float*)w[5].data_ptr();p.nw=(const __nv_bfloat16*)w[6].data_ptr();
 p.x=(const __nv_bfloat16*)x.data_ptr();p.ids=(const int*)ids.data_ptr();p.tw=(const float*)tw.data_ptr();p.inter=(float*)inter.data_ptr();p.y=(float*)y.data_ptr();p.out=(__nv_bfloat16*)out.data_ptr();
 p.m=x.size(0);p.lo=lo;p.nloc=cnt;auto s=at::cuda::getCurrentCUDAStream();
 dim3 grid(512,p.m,p.nloc);
 independent_moe::first<<<grid,256,0,s>>>(p);independent_moe::second<<<grid,256,0,s>>>(p);independent_moe::finish<<<p.m,256,0,s>>>(p);
}
