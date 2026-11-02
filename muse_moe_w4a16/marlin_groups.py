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
                 BS:tl.constexpr,R:tl.constexpr,BE:tl.constexpr,LO:tl.constexpr,HI:tl.constexpr):
    e=tl.program_id(0)
    es=tl.arange(0,BE)
    counts=tl.load(Count+es,es<E,other=0)
    counts=tl.where((counts>LO)&(counts<=HI),counts,0)
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


def kernel_fn(data,state):
    cfg=data['config'];bank=state['banks'][(cfg['local_start'],cfg['local_count'])]
    x=data['hidden_states'];m,h=x.shape;i=bank['size_n'];e=bank['count']
    chosen=state.get('groups',{}).get(m,(16 if m<=128 else 32,0,0))
    small,large,cut=chosen
    groups=[(small,-1,m*8)] if large==0 else [(small,-1,cut),(large,cut,m*8)]
    local=torch.empty_like(data['topk_ids'])
    mapping=torch.empty((e,m*8),dtype=torch.int32,device=x.device)
    counts=torch.empty((e,),dtype=torch.int32,device=x.device)
    _route_map[(e,)](data['topk_ids'],local,mapping,counts,m,cfg['local_start'],tr.next_power_of_2(m*8))
    metas=[]
    for bs,lo,hi in groups:
        maxpad=m*8+e*(bs-1)
        sorted_ids=torch.empty((maxpad,),dtype=torch.int32,device=x.device)
        experts=torch.empty((tr.cdiv(maxpad,bs),),dtype=torch.int32,device=x.device)
        padded=torch.empty((1,),dtype=torch.int32,device=x.device)
        _route_build[(e,)](mapping,counts,sorted_ids,experts,padded,m,e,bs,
            tr.next_power_of_2(m*8+bs),tr.next_power_of_2(e),lo,hi)
        metas.append((bs,sorted_ids,experts,padded))
    cache2=torch.empty((m*8,i),device=x.device,dtype=x.dtype)
    cache13=torch.empty((m*8*max(2*i,h),),device=x.device,dtype=x.dtype)
    cache1=cache13[:m*8*2*i].view(-1,2*i)
    cache3=cache13[:m*8*h].view(-1,h)
    common=dict(mul_topk_weights=False,is_ep=True,b_q_type=scalar_types.float4_e2m1f,
        is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
    for bs,sorted_ids,experts,padded in metas:
        moe_wna16_marlin_gemm(x,cache1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,
            state['workspace'],sorted_ids,experts,padded,data['topk_weights'],top_k=8,
            size_m=m,size_n=2*i,size_k=h,moe_block_size=bs,**common)
    silu_and_mul(cache1,cache2,expert_ids=local.view(-1),expert_step=1)
    for bs,sorted_ids,experts,padded in metas:
        moe_wna16_marlin_gemm(cache2,cache3,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,
            state['workspace'],sorted_ids,experts,padded,data['topk_weights'],top_k=1,
            size_m=m*8,size_n=h,size_k=i,moe_block_size=bs,**common)
    out=torch.empty_like(x)
    _finishv[(m,)](cache3,data['topk_ids'],data['topk_weights'],state['original_norm'],out,
        h,cfg['local_start'],e,tr.next_power_of_2(h),num_warps=8)
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
    import time,math
    started=time.time();state=_prepare(weights);state['groups']={}
    device=weights['w13'].device;h=weights['w13'].shape[-1]*2
    flush=torch.ones(64<<20,dtype=torch.float32,device=device)
    for m in [1,8,16,32,128,256]:
        datasets=[]
        for rep in range(3):
            pop=torch.randn(256,device=device).abs().pow(4)*0.01
            scores=torch.rand((m,256),device=device)+pop
            vals,ids=scores.topk(8,dim=-1)
            busiest=torch.bincount((ids//32).flatten(),minlength=8).argmax()
            ids=((ids-32*busiest)%256).to(torch.int32)
            datasets.append({'hidden_states':torch.randn((m,h),device=device,dtype=torch.bfloat16),
                'topk_ids':ids,'topk_weights':(vals/vals.sum(-1,keepdim=True)).float(),
                'config':{'local_start':0,'local_count':32,'d_expert':h,'bs':m}})
        default=(8 if m<=8 else 16 if m<=128 else 32,0,0)
        state['groups'][m]=default
        refs=[kernel_fn(d,state).clone() for d in datasets]
        candidates=[(8,0,0),(16,0,0),(32,0,0),(64,0,0)]
        if m>8:
            candidates += [(8,32,8),(8,32,16),(8,32,32),(16,64,16),(16,64,32),(8,64,8),(8,64,16)]
        best,besttime=default,float('inf')
        for group in candidates:
            if time.time()-started>250:break
            state['groups'][m]=group
            errors=[float((kernel_fn(d,state).float()-ref.float()).norm()/ref.float().norm().clamp(min=1e-20)) for d,ref in zip(datasets,refs)]
            if max(errors)>0.005:
                print(f'reject m={m} group={group} errors={errors}')
                continue
            times=[_bench(lambda:kernel_fn(d,state),flush) for d in datasets]
            score=math.exp(sum(math.log(t) for t in times)/len(times))
            print(f'group tune m={m} group={group} times={times} us={score:.2f}')
            if score<besttime:best,besttime=group,score
        state['groups'][m]=best
        print(f'SELECT m={m} group={best} us={besttime:.2f}')
    del flush
    print(f'prepare {time.time()-started:.1f}s')
    return state
