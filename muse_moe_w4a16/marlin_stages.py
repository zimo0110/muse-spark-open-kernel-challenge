# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
import triton as tr
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks, "original_norm": weights["post_norm_w"],
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


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




@tr.jit
def _alignsmall(IDs, Local, Sorted, Experts, Padded, M:tl.constexpr, E:tl.constexpr,
                START:tl.constexpr, BS:tl.constexpr, R:tl.constexpr, BE:tl.constexpr):
    rr=tl.arange(0,R)
    ee=tl.arange(0,BE)
    ids=tl.load(IDs+rr,rr<M*8,other=-1)-START
    tl.store(Local+rr,tl.where((ids>=0)&(ids<E),ids,-1),rr<M*8)
    hit=(ids[None,:]==ee[:,None])&(rr[None,:]<M*8)&(ee[:,None]<E)
    rank=tl.cumsum(hit.to(tl.int32),1)-1
    count=tl.sum(hit.to(tl.int32),1)
    blocks=tl.cdiv(count,BS)
    begin=tl.cumsum(blocks,0)-blocks
    # Padding and live routes have disjoint output addresses.
    tl.store(Sorted+(begin*BS)[:,None]+rr[None,:],M*8,
        (rr[None,:]>=count[:,None])&(rr[None,:]<(blocks*BS)[:,None]))
    tl.store(Sorted+(begin*BS)[:,None]+rank,tl.broadcast_to(rr[None,:],(BE,R)),hit)
    tl.store(Experts+begin[:,None]+rr[None,:],tl.broadcast_to(ee[:,None],(BE,R)),rr[None,:]<blocks[:,None])
    tl.store(Padded,tl.sum(blocks)*BS)


@tr.jit
def _finishv(Y, IDs, Gates, NW, Out, H:tl.constexpr, START:tl.constexpr, E:tl.constexpr,
             BH:tl.constexpr):
    t=tl.program_id(0)
    h=tl.arange(0,BH)
    s=tl.arange(0,8)
    ids=tl.load(IDs+t*8+s)
    valid=(ids>=START)&(ids<START+E)
    nw=1+tl.load(NW+h,h<H,other=0).to(tl.float32)
    y=tl.load(Y+(t*8+s[:,None])*H+h[None,:],valid[:,None]&(h[None,:]<H),other=0).to(tl.float32)
    y=(y*nw[None,:]).to(tl.bfloat16).to(tl.float32)
    inv=tl.rsqrt(tl.sum(y*y,1)/H+1e-8)
    g=tl.load(Gates+t*8+s)
    z=tl.sum(y*(inv*g)[:,None],0)
    tl.store(Out+t*H+h,z.to(tl.bfloat16),h<H)


@tr.jit
def _route_map(IDs, Local, Map, Count, M:tl.constexpr, START:tl.constexpr, R:tl.constexpr):
    e=tl.program_id(0)
    r=tl.arange(0,R)
    ids=tl.load(IDs+r,r<M*8,other=-1)-START
    hit=(ids==e)&(r<M*8)
    rank=tl.cumsum(hit.to(tl.int32))-1
    tl.store(Map+e*M*8+rank,r,hit)
    tl.store(Count+e,tl.sum(hit.to(tl.int32)))
    if e==0:
        tl.store(Local+r,tl.where((ids>=0)&(ids<32),ids,-1),r<M*8)


@tr.jit
def _route_build(Map,Count,Sorted,Experts,Padded,M:tl.constexpr,E:tl.constexpr,
                 BS:tl.constexpr,R:tl.constexpr,BE:tl.constexpr):
    e=tl.program_id(0)
    es=tl.arange(0,BE)
    counts=tl.load(Count+es,es<E,other=0)
    blocks=tl.cdiv(counts,BS)
    begin=tl.sum(tl.where(es<e,blocks,0))
    n=tl.sum(tl.where(es==e,counts,0))
    nb=tl.cdiv(n,BS)
    r=tl.arange(0,R)
    route=tl.load(Map+e*M*8+r,r<n,other=M*8)
    tl.store(Sorted+begin*BS+r,route,r<nb*BS)
    tl.store(Experts+begin+r,e,r<nb)
    if e==0:
        tl.store(Padded,tl.sum(blocks)*BS)


CUSTOM=None
LAUNCH_SRC=r'''
// Launcher tuning around the Apache-2.0 Marlin kernel supplied by SGLang.
#include "gemm/marlin_moe/moe_wna16_marlin.cuh"

template<int BS,int TN,int TK,int TH,int ST>
void launch(tvm::ffi::TensorView a,tvm::ffi::TensorView c,tvm::ffi::TensorView w,tvm::ffi::TensorView sf,tvm::ffi::TensorView gs,
 tvm::ffi::TensorView ws,tvm::ffi::TensorView sorted,tvm::ffi::TensorView experts,tvm::ffi::TensorView padded,tvm::ffi::TensorView gates,
 tvm::ffi::TensorView tmp,int M,int N,int K,int topk,int occupancy) {
 using namespace sglang;
 using namespace device::marlin_moe;
 constexpr int MB=(BS+15)/16;
 auto kernel=Marlin<__nv_bfloat16,host::kFE2M1f.id(),host::kFE4M3fn.id(),TH,MB,TN/16,TK/16,BS==8,
   ST,1,false,false,false>;
 int dev=0,sms=0,maxsm=0;
 cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
 cudaDeviceGetAttribute(&maxsm,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);
 int cap=maxsm/occupancy-1024;
 thread_config_t cfg{TK,TN,TH};
 int bm=MB*16;
 int bsize=ST*TK*TN/2, rsize=bm*(TN+8)*2, bias=TN*2;
 int tmp_size=std::max(std::max(bsize,rsize),std::min(bsize,rsize)+bias);
 int need=tmp_size+ST*bm*TK*2+ST*(TK/16)*TN*2+bm*4+1024;
 sglang::host::RuntimeCheck(need+512<=cap,"tile doesn't fit shared memory");
 cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,maxsm);
 auto stream=sglang::host::LaunchKernel::resolve_device(a.device());
 kernel<<<sms*occupancy,TH,cap,stream>>>(
   (const int4*)a.data_ptr(),(const int4*)w.data_ptr(),(int4*)c.data_ptr(),(int4*)tmp.data_ptr(),nullptr,
   (const int4*)sf.data_ptr(),(const uint16_t*)gs.data_ptr(),nullptr,nullptr,
   (int*)sorted.data_ptr(),(int*)experts.data_ptr(),(int*)padded.data_ptr(),(float*)gates.data_ptr(),
   topk,false,false,K/16,M,N,K,(int*)ws.data_ptr(),false,false,true,cap);
}

void run(tvm::ffi::TensorView a,tvm::ffi::TensorView c,tvm::ffi::TensorView w,tvm::ffi::TensorView sf,tvm::ffi::TensorView gs,
 tvm::ffi::TensorView ws,tvm::ffi::TensorView sorted,tvm::ffi::TensorView experts,tvm::ffi::TensorView padded,tvm::ffi::TensorView gates,
 tvm::ffi::TensorView tmp,int64_t M,int64_t N,int64_t K,int64_t topk,int64_t bs,int64_t kind,int64_t occupancy,int64_t stages) {
 #define TILE(B,N_,K_,T_,Q,S) if(bs==B && kind==Q && stages==S){launch<B,N_,K_,T_,S>(a,c,w,sf,gs,ws,sorted,experts,padded,gates,tmp,M,N,K,topk,occupancy);return;}
 #define SHAPES(B,S) TILE(B,64,128,128,0,S) TILE(B,128,64,128,1,S) TILE(B,128,128,256,2,S) TILE(B,256,64,256,3,S) TILE(B,64,256,256,4,S)
 SHAPES(8,2) SHAPES(8,3) SHAPES(8,4)
 SHAPES(16,2) SHAPES(16,3) SHAPES(16,4)
 SHAPES(32,2) SHAPES(32,3) SHAPES(32,4)
 SHAPES(64,4)
 sglang::host::RuntimeCheck(false,"unknown launch configuration");
}

'''


def _custom_gemm(a,c,w,unused,sf,gs,zp,gi,perm,ws,sorted,experts,padded,gates,*,
                 top_k,size_m,size_n,size_k,moe_block_size,launch_cfg,reduce_tmp,**kwargs):
    kind,occupancy,stages=launch_cfg
    if kind<0 or CUSTOM is None:
        kwargs["is_ep"]=True
        return moe_wna16_marlin_gemm(a,c,w,unused,sf,gs,zp,gi,perm,ws,sorted,experts,padded,gates,
            top_k=top_k,size_m=size_m,size_n=size_n,size_k=size_k,moe_block_size=moe_block_size,**kwargs)
    CUSTOM.run(a,c,w,sf,gs,ws,sorted,experts,padded,gates,reduce_tmp,size_m,size_n,size_k,top_k,
        moe_block_size,kind,occupancy,stages)
    return c


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    block_size_m = state.get('tiles', {}).get(num_tokens, 16 if num_tokens <= 64 else 32)
    topk_weights=data['topk_weights']
    local_ids=torch.empty_like(data['topk_ids'])
    maxpad=num_tokens*8+bank['count']*(block_size_m-1)
    sorted_ids=torch.empty((maxpad,),dtype=torch.int32,device=x.device)
    block_experts=torch.empty((tr.cdiv(maxpad,block_size_m),),dtype=torch.int32,device=x.device)
    n_padded=torch.empty((1,),dtype=torch.int32,device=x.device)
    if num_tokens<=8:
        _alignsmall[(1,)](data['topk_ids'],local_ids,sorted_ids,block_experts,n_padded,num_tokens,
            bank['count'],cfg['local_start'],block_size_m,tr.next_power_of_2(max(num_tokens*8,block_size_m)),
            tr.next_power_of_2(bank['count']),num_warps=4)
    else:
        mapping=torch.empty((bank['count'],num_tokens*8),dtype=torch.int32,device=x.device)
        counts=torch.empty((bank['count'],),dtype=torch.int32,device=x.device)
        _route_map[(bank['count'],)](data['topk_ids'],local_ids,mapping,counts,num_tokens,cfg['local_start'],tr.next_power_of_2(num_tokens*8))
        _route_build[(bank['count'],)](mapping,counts,sorted_ids,block_experts,n_padded,num_tokens,bank['count'],
            block_size_m,tr.next_power_of_2(num_tokens*8+block_size_m),tr.next_power_of_2(bank['count']))

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(reduce_tmp=state["reduce_tmp"],moe_block_size=block_size_m, mul_topk_weights=False, is_ep=False,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = _custom_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, launch_cfg=state.get("launch1",{}).get(num_tokens,(-1,4,4)), **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = _custom_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, launch_cfg=state.get("launch2",{}).get(num_tokens,(-1,4,4)), **common).view(num_tokens, TOP_K, size_k)
    out=torch.empty_like(x)
    _finishv[(num_tokens,)](y, data['topk_ids'], topk_weights, state['original_norm'], out,
        size_k, cfg['local_start'], cfg['local_count'], tr.next_power_of_2(size_k), num_warps=8)
    return out


def _bench(fn, flush, reps=18):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn();fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for a,b in zip(starts,ends):
        flush.sum();a.record();graph.replay();b.record()
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b)*1000 for a,b in zip(starts,ends))[reps//2]


def prepare(weights):
    global CUSTOM
    import time,math,tempfile
    from pathlib import Path
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    started=time.time()
    state=_prepare(weights)
    device=weights['w13'].device
    sms=torch.cuda.get_device_properties(device).multi_processor_count
    state['reduce_tmp']=torch.empty(sms*4*64*256*2,device=device,dtype=torch.float32)
    try:
        source=Path(tempfile.mkdtemp())/'launcher.cuh'
        source.write_text(LAUNCH_SRC)
        CUSTOM=load_jit('okc_marlin_stage_launch',cuda_files=[str(source)],cuda_wrappers=[('run','run')],
            extra_include_paths=[str(KERNEL_PATH/'csrc')],extra_cuda_cflags=['-O3'])
        print(f'custom Marlin build {time.time()-started:.1f}s')
    except Exception as exc:
        print(f'custom launcher unavailable: {str(exc)[-6000:]}')
    state['tiles']={};state['launch1']={};state['launch2']={}
    flush=torch.ones(64<<20,dtype=torch.float32,device=device)
    h=weights['w13'].shape[-1]*2
    for m in [1,8,16,32,128,256]:
        datasets=[]
        for rep in range(2):
            pop=torch.randn(256,device=device).abs().pow(4)*(0.01 if rep==0 else 0.003)
            scores=torch.rand((m,256),device=device)+pop
            vals,ids=scores.topk(8,dim=-1)
            busiest=torch.bincount((ids//32).flatten(),minlength=8).argmax()
            ids=((ids-32*busiest)%256).to(torch.int32)
            datasets.append({'hidden_states':torch.randn((m,h),device=device,dtype=torch.bfloat16),
                'topk_ids':ids,'topk_weights':(vals/vals.sum(-1,keepdim=True)).float(),
                'config':{'local_start':0,'local_count':32,'d_expert':h,'bs':m}})
        defaultbs=8 if m<=8 else 16 if m<=128 else 32
        state['tiles'][m]=defaultbs
        state['launch1'][m]=(-1,4,4);state['launch2'][m]=(-1,4,4)
        refs=[kernel_fn(d,state).clone() for d in datasets]
        winner=(defaultbs,(-1,4,4),(-1,4,4)); wintime=float('inf')
        for bs in ([8,16] if m<=32 else [16,32,64]):
            state['tiles'][m]=bs
            state['launch1'][m]=(-1,4,4);state['launch2'][m]=(-1,4,4)
            candidates=[(-1,4,4)]
            if CUSTOM is not None:
                candidates += [(kind,occ,stages) for kind in range(5) for occ in [1,2,3,4]
                               for stages in ([2,3,4] if bs<64 else [4])]
            for stage in ['launch1','launch2']:
                best=(-1,4,4);besttime=float('inf')
                for candidate in candidates:
                    if time.time()-started>260: break
                    state[stage][m]=candidate
                    try:
                        errors=[float((kernel_fn(d,state).float()-ref.float()).norm()/ref.float().norm().clamp(min=1e-20)) for d,ref in zip(datasets,refs)]
                        if max(errors)>0.005: continue
                        times=[_bench(lambda:kernel_fn(d,state),flush,reps=6) for d in datasets]
                        score=math.sqrt(times[0]*times[1])
                    except Exception:
                        continue
                    if score<besttime: best,besttime=candidate,score
                state[stage][m]=best
                print(f'STAGE m={m} bs={bs} stage={stage} cfg={best} us={besttime:.2f}')
            try:
                times=[_bench(lambda:kernel_fn(d,state),flush,reps=18) for d in datasets]
                score=math.sqrt(times[0]*times[1])
                if score<wintime:
                    winner=(bs,state['launch1'][m],state['launch2'][m]);wintime=score
            except Exception: pass
        state['tiles'][m],state['launch1'][m],state['launch2'][m]=winner
        print(f'SELECT m={m} cfg={winner} us={wintime:.2f}')
    del flush
    print(f'prepare total {time.time()-started:.1f}s')
    return state
