# okc: problem=muse_moe_w4a16
"""Grouped W4A16 experts. Weights stay NVFP4; decode only inside GEMMs."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _routes(IDs, Map, Count, M: tl.constexpr, START: tl.constexpr, R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(IDs + r, r < M * 8, other=-1)
    hit = (ids == e + START) & (r < M * 8)
    rank = tl.cumsum(hit.to(tl.int32)) - 1
    tl.store(Map + e * M * 8 + rank, r, hit)
    tl.store(Count + e, tl.sum(hit.to(tl.int32)))


@tr.jit
def _schedule(Count, Ends, E: tl.constexpr, BE: tl.constexpr, BM: tl.constexpr):
    e = tl.arange(0, BE)
    n = tl.load(Count + e, e < E, other=0)
    end = tl.cumsum(tl.cdiv(n, BM))
    tl.store(Ends + e, end, e < E)


@tr.jit
def _decode(W, S, G, e, n, k, N: tl.constexpr, K: tl.constexpr):
    packed = tl.load(W + e.to(tl.int64) * N * (K // 2) + n[:, None] * (K // 2) + k[None, :] // 2)
    c = (packed.to(tl.uint16) >> ((k[None, :] % 2) * 4)) & 15
    mag = c & 7
    bits = tl.where(mag == 0, 0, tl.where(mag == 1, 0x3f00, (mag + 252) << 6)) | ((c & 8) << 12)
    v = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    scale = tl.load(S + e.to(tl.int64) * N * (K // 16) + n[:, None] * (K // 16) + k[None, :] // 16).to(tl.float32)
    return (v * scale).to(tl.bfloat16)


@tr.jit
def _gemm(X, W, S, G, Map, Count, Ends, Out,
          M: tl.constexpr, E: tl.constexpr, BE: tl.constexpr,
          N: tl.constexpr, K: tl.constexpr, FIRST: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pn = tl.program_id(0)
    block = tl.program_id(1)
    sp = tl.program_id(2)
    es = tl.arange(0, BE)
    ends = tl.load(Ends + es, es < E, other=2147483647)
    total = tl.sum(tl.where(es == E - 1, ends, 0))
    if block < total:
        e = tl.sum((block >= ends).to(tl.int32))
        begin = tl.load(Ends + e - 1, e > 0, other=0)
        rows = (block - begin) * BM + tl.arange(0, BM)
        count = tl.load(Count + e)
        route = tl.load(Map + e * M * 8 + rows, rows < count, other=0)
        tok = route // 8 if FIRST else route
        ns = pn * BN + tl.arange(0, BN)
        kk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(K // BK // SPLIT):
            ks = sp * (K // SPLIT) + ki * BK + kk
            x = tl.load(X + tok[:, None] * K + ks[None, :], rows[:, None] < count, other=0)
            w = _decode(W, S, G, e, ns, ks, N, K)
            acc = tl.dot(x, tl.trans(w), acc)
        gs = tl.load(G + e)
        tl.store(Out + sp * M * 8 * N + route[:, None] * N + ns[None, :], acc * gs, rows[:, None] < count)


@tr.jit
def _act(P, IDs, Out, M: tl.constexpr, I: tl.constexpr, START: tl.constexpr, E: tl.constexpr,
         SPLIT: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    e = tl.load(IDs + r)
    if (e >= START) & (e < START + E):
        gate = tl.full((B,), 0, tl.float32)
        up = tl.full((B,), 0, tl.float32)
        for sp in range(SPLIT):
            gate += tl.load(P + sp * M * 8 * 2 * I + r * 2 * I + n)
            up += tl.load(P + sp * M * 8 * 2 * I + r * 2 * I + I + n)
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        silu = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(Out + r * I + n, (silu * up).to(tl.bfloat16))


@tr.jit
def _reduce_norm(P, IDs, NW, Y, SS, M: tl.constexpr, H: tl.constexpr,
                 START: tl.constexpr, E: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    e = tl.load(IDs + r)
    if (e >= START) & (e < START + E):
        acc = tl.full((B,), 0, tl.float32)
        for sp in range(SPLIT):
            acc += tl.load(P + sp * M * 8 * H + r * H + n)
        w = 1 + tl.load(NW + n).to(tl.float32)
        y = (acc.to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16)
        tl.store(Y + r * H + n, y)
        tl.store(SS + r * (H // B) + tl.program_id(1), tl.sum(y.to(tl.float32) * y.to(tl.float32)))


@tr.jit
def _finish2(Y, SS, IDs, Gates, Out, H: tl.constexpr, START: tl.constexpr,
             E: tl.constexpr, NB: tl.constexpr, B: tl.constexpr):
    t = tl.program_id(0)
    ns = tl.program_id(1) * B + tl.arange(0, B)
    ss = tl.arange(0, 8)
    jj = tl.arange(0, NB)
    ids = tl.load(IDs + t * 8 + ss)
    valid = (ids >= START) & (ids < START + E)
    sums = tl.load(SS + (t * 8 + ss[:, None]) * NB + jj[None, :], valid[:, None], other=0)
    inv = tl.rsqrt(tl.sum(sums, 1) / H + 1e-8)
    g = tl.load(Gates + t * 8 + ss)
    y = tl.load(Y + (t * 8 + ss[:, None]) * H + ns[None, :], valid[:, None], other=0).to(tl.float32)
    out = tl.sum(y * (g * inv)[:, None], 0)
    tl.store(Out + t * H + ns, out.to(tl.bfloat16))


@tr.jit
def _finish(Y, IDs, Gates, NW, Out, H: tl.constexpr, START: tl.constexpr, E: tl.constexpr,
            BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    norm = 1.0 + tl.load(NW + h, h < H, other=0).to(tl.float32)
    result = tl.full((BH,), 0, tl.float32)
    for s in range(8):
        e = tl.load(IDs + t * 8 + s)
        valid = (e >= START) & (e < START + E)
        y = tl.load(Y + (t * 8 + s) * H + h, valid & (h < H), other=0).to(tl.float32)
        y = (y * norm).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(tl.sum(y * y) / H + 1e-8)
        g = tl.load(Gates + t * 8 + s)
        result += y * inv * g
    tl.store(Out + t * H + h, result.to(tl.bfloat16), h < H)


CUDA_SRC = r'''
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
 extern __shared__ __align__(128) uint8_t storage[];
 uint64_t* bar=(uint64_t*)storage;
 uint8_t* buf=storage+128;
 const int tilebytes=(K/128)*1280;
 const uint8_t* p0=W+((size_t)eg*(FIRST ? N*2/16 : N/16)+slab)*tilebytes;
 if(threadIdx.x==0) {initbar(bar);if constexpr(FIRST) initbar(bar+1);}
 __syncthreads();
 if(threadIdx.x==0) {
   copybulk(buf,p0,tilebytes,bar);
   if constexpr(FIRST) copybulk(buf+tilebytes,p0+(size_t)(N/16)*tilebytes,tilebytes,bar+1);
 }
 // Gate and up slabs are separate in the source; first copy is handled below.
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
 waitbar(bar);
 if constexpr(FIRST) waitbar(bar+1);
 for(int g0=warp*U;g0<K/128;g0+=WARPS*U) {
   uint4 qa[U],qb[U],qu[U],qv[U];
   uint32_t sa[U],sb[U],su[U],sv[U];
   #pragma unroll
   for(int u=0;u<U;++u) {
     const uint8_t* p=buf+(g0+u)*1280;
     qa[u]=*(const uint4*)(p+lane*16);qb[u]=*(const uint4*)(p+512+lane*16);
     sa[u]=*(const uint32_t*)(p+1024+lane*4);sb[u]=*(const uint32_t*)(p+1152+lane*4);
     if constexpr(FIRST) {
       p+=tilebytes;
       qu[u]=*(const uint4*)(p+lane*16);qv[u]=*(const uint4*)(p+512+lane*16);
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
}

void run(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,
 torch::Tensor map,torch::Tensor count,torch::Tensor ends,torch::Tensor out,
 int64_t M,int64_t E,int64_t start,int64_t N,int64_t K,bool first,int64_t nt,int64_t warps,int64_t U) {
 const int cols=nt*8;
 const int smem=128+(K/128)*1280*(first ? 2 : 1);
 const int blocks=std::min((int)M*8,(int)E+((int)M*8+cols-1)/cols-1);
 dim3 grid(N/16,blocks);
 auto stream=at::cuda::getCurrentCUDAStream();
 #define GO(T,W,F,U_) if(nt==T && warps==W && first==F && U==U_) { \
 static bool attr=false;if(!attr){cudaFuncSetAttribute(experts<T,W,F,U_>,cudaFuncAttributeMaxDynamicSharedMemorySize,200*1024);attr=true;} \
 experts<T,W,F,U_><<<grid,W*32,smem,stream>>>( \
 (const __nv_bfloat16*)x.data_ptr(),w.data_ptr<uint8_t>(),(const uint8_t*)sf.data_ptr(),gs.data_ptr<float>(), \
 map.data_ptr<int>(),count.data_ptr<int>(),ends.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),M,E,start,N,K);return;}
 #define GU(T,W,F) GO(T,W,F,1) GO(T,W,F,2) GO(T,W,F,4)
 #define GF(T,W) GU(T,W,true) GU(T,W,false)
 GF(1,4) GF(1,8) GF(2,4) GF(2,8) GF(4,4) GF(4,8)
 TORCH_CHECK(false,"unsupported configuration");
}

'''
EXT = None

def prepare(weights):
    global EXT
    import tempfile, time
    from torch.utils.cpp_extension import load_inline
    t0=time.time()
    EXT=load_inline(name="okc_moe_cuda", cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,int64_t,int64_t,int64_t);", cuda_sources=CUDA_SRC, functions=["run"],
        build_directory=tempfile.mkdtemp(), extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a","--use_fast_math"])
    print(f"CUDA build {time.time()-t0:.1f}s")
    result=dict(weights)
    for key in ['w13','w2']:
        w=weights[key];sf=weights[key+'_sf'].view(torch.uint8)
        e,n,k2=w.shape;k=k2*2
        q=w.view(e,n//16,2,8,k//128,4,4,4).permute(0,1,4,2,3,6,5,7).contiguous().view(e,n//16,k//128,1024)
        word=q.view(torch.int32)
        low=(word&15)|((word&0x00000f00)>>4)|((word&0x000f0000)>>8)|((word&0x0f000000)>>12)
        high=((word&0x000000f0)<<12)|((word&0x0000f000)<<8)|((word&0x00f00000)<<4)|(word&(-268435456))
        q=(low|high).view(torch.uint8)
        del word,low,high
        sc=sf.view(e,n//16,2,8,k//128,4,2).repeat_interleave(2,-1).permute(0,1,4,2,3,6,5).contiguous().view(e,n//16,k//128,256)
        result[key+'_packed']=torch.cat([q,sc],dim=-1)
        del q,sc
    print(f"prepare {time.time()-t0:.1f}s")
    return result


def kernel_fn(data, state):
    x=data['hidden_states'];c=data['config'];m,h=x.shape
    i,e,start=c['d_expert'],c['local_count'],c['local_start']
    nt=1 if m<=32 else 4
    route=torch.empty((e,m*8),device=x.device,dtype=torch.int32)
    count=torch.empty((e,),device=x.device,dtype=torch.int32)
    ends=torch.empty_like(count)
    inter=torch.empty((m*8,i),device=x.device,dtype=torch.bfloat16)
    y=torch.empty((m*8,h),device=x.device,dtype=torch.bfloat16)
    out=torch.empty_like(x)
    _routes[(e,)](data['topk_ids'],route,count,m,start,tr.next_power_of_2(m*8))
    _schedule[(1,)](count,ends,e,tr.next_power_of_2(e),nt*8)
    EXT.run(x,state['w13_packed'],state['w13_sf'],state['w13_gs'],route,count,ends,inter,m,e,start,i,h,True,nt,4,1)
    EXT.run(inter,state['w2_packed'],state['w2_sf'],state['w2_gs'],route,count,ends,y,m,e,start,h,i,False,nt,4,1)
    _finish[(m,)](y,data['topk_ids'],data['topk_weights'],state['post_norm_w'],out,h,start,e,tr.next_power_of_2(h),num_warps=4)
    return out
