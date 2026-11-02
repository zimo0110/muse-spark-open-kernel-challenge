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


CUDA_SRC = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n\n__device__ __forceinline__ uint32_t loadq(const void* p) {\n  uint32_t r;\n  asm("ld.global.nc.L1::no_allocate.L2::256B.u32 %0, [%1];" : "=r"(r) : "l"(p));\n  return r;\n}\n__device__ __forceinline__ uint4 loadx(const void* p) {\n  uint4 r;\n  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"l"(p));\n  return r;\n}\n__device__ __forceinline__ uint16_t bits4(uint32_t c) {\n  uint32_t m = c & 7;\n  return (uint16_t)((m == 0 ? 0 : m == 1 ? 0x3f00 : (m + 252) << 6) | ((c & 8) << 12));\n}\n__device__ __forceinline__ uint32_t decodepair(uint32_t c, __nv_bfloat162 scale) {\n  union { uint32_t u; __nv_bfloat162 b; } v;\n  v.u = (uint32_t)bits4(c & 15) | ((uint32_t)bits4((c >> 4) & 15) << 16);\n  v.b = __hmul2(v.b, scale);\n  return v.u;\n}\n__device__ __forceinline__ __nv_bfloat162 scale2(uint8_t b) {\n  __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);\n  __half h = hr;\n  return __float2bfloat162_rn(__half2float(h));\n}\n__device__ __forceinline__ void decode8(uint32_t q, uint8_t s, uint32_t* d) {\n  __nv_bfloat162 sc = scale2(s);\n  #pragma unroll\n  for (int z=0;z<4;++z) d[z] = decodepair(q >> (8*z),sc);\n}\n__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b,\n                                     uint32_t x0,uint32_t x1,int j) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])\n    : "r"(a[j]),"r"(b[j]),"r"(a[j+1]),"r"(b[j+1]),"r"(x0),"r"(x1));\n}\n\ntemplate<int NT, int WARPS, bool FIRST, int U>\n__global__ void __launch_bounds__(WARPS*32) experts(\n const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,\n const uint8_t* __restrict__ SF, const float* __restrict__ GS,\n const int* __restrict__ MAP,const int* __restrict__ COUNT,const int* __restrict__ ENDS,\n __nv_bfloat16* __restrict__ OUT,int M,int E,int start,int N,int K) {\n constexpr int COLS=NT*8;\n __shared__ float red[WARPS][16*COLS];\n __shared__ float redu[WARPS][16*COLS];\n const int block=blockIdx.y;\n if (block >= ENDS[E-1]) return;\n int e=0;\n while(block >= ENDS[e]) ++e;\n const int localrow = (block - (e ? ENDS[e-1] : 0))*COLS;\n const int cnt=COUNT[e];\n const int eg=e+start;\n const int warp=threadIdx.x>>5, lane=threadIdx.x&31;\n const int g=lane>>2,t=lane&3;\n const int slab=blockIdx.x;\n const size_t row0=((size_t)eg*(FIRST ? N*2 : N)+slab*16+g);\n const uint8_t* w0=W+row0*(K/2);\n const uint8_t* w1=w0+(size_t)8*(K/2);\n const uint8_t* s0=SF+row0*(K/16);\n const uint8_t* s1=s0+(size_t)8*(K/16);\n float acc[NT][4], accu[NT][4];\n const __nv_bfloat16* xr[NT];\n bool valid[NT];\n #pragma unroll\n for(int j=0;j<NT;++j) {\n   #pragma unroll\n   for(int z=0;z<4;++z) acc[j][z]=accu[j][z]=0;\n   int r=localrow+j*8+g;\n   valid[j]=r<cnt;\n   int route=valid[j] ? MAP[(size_t)e*M*8+r] : 0;\n   xr[j]=X+(size_t)(FIRST ? route/8 : route)*K;\n }\n for(int c0=warp*U;c0<K/32;c0+=WARPS*U) {\n   uint32_t qa[U],qb[U],qu[U],qv[U];\n   uint8_t sa[U],sb[U],su[U],sv[U];\n   #pragma unroll\n   for(int u=0;u<U;++u) {\n     int k=(c0+u)*32+t*8;\n     qa[u]=loadq(w0+k/2); qb[u]=loadq(w1+k/2);\n     sa[u]=s0[k/16]; sb[u]=s1[k/16];\n     if constexpr(FIRST) {\n       qu[u]=loadq(w0+(size_t)N*(K/2)+k/2); qv[u]=loadq(w1+(size_t)N*(K/2)+k/2);\n       su[u]=s0[(size_t)N*(K/16)+k/16]; sv[u]=s1[(size_t)N*(K/16)+k/16];\n     }\n   }\n   #pragma unroll\n   for(int u=0;u<U;++u) {\n     uint32_t a[4],b[4],au[4],bu[4];\n     decode8(qa[u],sa[u],a);decode8(qb[u],sb[u],b);\n     if constexpr(FIRST) {decode8(qu[u],su[u],au);decode8(qv[u],sv[u],bu);}\n     #pragma unroll\n     for(int j=0;j<NT;++j) {\n       uint4 v=valid[j] ? loadx(xr[j]+(c0+u)*32+t*8) : make_uint4(0,0,0,0);\n       mma(acc[j],a,b,v.x,v.y,0);mma(acc[j],a,b,v.z,v.w,2);\n       if constexpr(FIRST) {mma(accu[j],au,bu,v.x,v.y,0);mma(accu[j],au,bu,v.z,v.w,2);}\n     }\n   }\n }\n #pragma unroll\n for(int j=0;j<NT;++j) {\n   red[warp][g*COLS+j*8+2*t]=acc[j][0];red[warp][g*COLS+j*8+2*t+1]=acc[j][1];\n   red[warp][(g+8)*COLS+j*8+2*t]=acc[j][2];red[warp][(g+8)*COLS+j*8+2*t+1]=acc[j][3];\n   if constexpr(FIRST) {\n     redu[warp][g*COLS+j*8+2*t]=accu[j][0];redu[warp][g*COLS+j*8+2*t+1]=accu[j][1];\n     redu[warp][(g+8)*COLS+j*8+2*t]=accu[j][2];redu[warp][(g+8)*COLS+j*8+2*t+1]=accu[j][3];\n   }\n }\n __syncthreads();\n float gs=GS[eg];\n for(int idx=threadIdx.x;idx<16*COLS;idx+=WARPS*32) {\n   const int row=idx&15,col=idx>>4;\n   if(localrow+col<cnt) {\n     float v=0,u=0;\n     #pragma unroll\n     for(int q=0;q<WARPS;++q) {v+=red[q][row*COLS+col];if constexpr(FIRST) u+=redu[q][row*COLS+col];}\n     v=__bfloat162float(__float2bfloat16_rn(v*gs));\n     if constexpr(FIRST) {\n       u=__bfloat162float(__float2bfloat16_rn(u*gs));\n       v=__bfloat162float(__float2bfloat16_rn(v/(1.f+__expf(-v))))*u;\n     }\n     int route=MAP[(size_t)e*M*8+localrow+col];\n     OUT[(size_t)route*N+slab*16+row]=__float2bfloat16_rn(v);\n   }\n }\n}\n\nvoid run(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,\n torch::Tensor map,torch::Tensor count,torch::Tensor ends,torch::Tensor out,\n int64_t M,int64_t E,int64_t start,int64_t N,int64_t K,bool first,int64_t nt,int64_t warps,int64_t U) {\n const int cols=nt*8;\n const int blocks=std::min((int)M*8,(int)E+((int)M*8+cols-1)/cols-1);\n dim3 grid(N/16,blocks);\n auto stream=at::cuda::getCurrentCUDAStream();\n #define GO(T,W,F,U_) if(nt==T && warps==W && first==F && U==U_) { \\\n experts<T,W,F,U_><<<grid,W*32,0,stream>>>( \\\n (const __nv_bfloat16*)x.data_ptr(),w.data_ptr<uint8_t>(),(const uint8_t*)sf.data_ptr(),gs.data_ptr<float>(), \\\n map.data_ptr<int>(),count.data_ptr<int>(),ends.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),M,E,start,N,K);return;}\n #define GU(T,W,F) GO(T,W,F,1) GO(T,W,F,2) GO(T,W,F,4)\n #define GF(T,W) GU(T,W,true) GU(T,W,false)\n GF(1,4) GF(1,8) GF(2,4) GF(2,8) GF(4,4) GF(4,8)\n TORCH_CHECK(false,"unsupported configuration");\n}\n'
EXT = None

def prepare(weights):
    global EXT
    import tempfile, time
    from torch.utils.cpp_extension import load_inline
    t0=time.time()
    EXT=load_inline(name="okc_moe_cuda", cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,bool,int64_t,int64_t,int64_t);", cuda_sources=CUDA_SRC, functions=["run"],
        build_directory=tempfile.mkdtemp(), extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a","--use_fast_math"])
    print(f"CUDA build {time.time()-t0:.1f}s")
    return weights


def kernel_fn(data, state):
    x=data['hidden_states'];c=data['config'];m,h=x.shape
    i,e,start=c['d_expert'],c['local_count'],c['local_start']
    nt=1 if m<=32 else 2
    route=torch.empty((e,m*8),device=x.device,dtype=torch.int32)
    count=torch.empty((e,),device=x.device,dtype=torch.int32)
    ends=torch.empty_like(count)
    inter=torch.empty((m*8,i),device=x.device,dtype=torch.bfloat16)
    y=torch.empty((m*8,h),device=x.device,dtype=torch.bfloat16)
    out=torch.empty_like(x)
    _routes[(e,)](data['topk_ids'],route,count,m,start,tr.next_power_of_2(m*8))
    _schedule[(1,)](count,ends,e,tr.next_power_of_2(e),nt*8)
    EXT.run(x,state['w13'],state['w13_sf'],state['w13_gs'],route,count,ends,inter,m,e,start,i,h,True,nt,4,2)
    EXT.run(inter,state['w2'],state['w2_sf'],state['w2_gs'],route,count,ends,y,m,e,start,h,i,False,nt,4,2)
    _finish[(m,)](y,data['topk_ids'],data['topk_weights'],state['post_norm_w'],out,h,start,e,tr.next_power_of_2(h),num_warps=4)
    return out
