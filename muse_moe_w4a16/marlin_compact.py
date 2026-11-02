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
    if num_tokens <= 32:
        local_ids=torch.empty_like(data['topk_ids'])
        maxpad=num_tokens*8+bank['count']*(block_size_m-1)
        sorted_ids=torch.empty((maxpad,),dtype=torch.int32,device=x.device)
        block_experts=torch.empty((tr.cdiv(maxpad,block_size_m),),dtype=torch.int32,device=x.device)
        n_padded=torch.empty((1,),dtype=torch.int32,device=x.device)
        _alignsmall[(1,)](data['topk_ids'],local_ids,sorted_ids,block_experts,n_padded,num_tokens,
            bank['count'],cfg['local_start'],block_size_m,tr.next_power_of_2(max(num_tokens*8,block_size_m)),
            tr.next_power_of_2(bank['count']),num_warps=4)
    else:
        local_ids=expert_map[data['topk_ids']].contiguous()
        sorted_ids,block_experts,n_padded=moe_align_block_size(local_ids,block_size_m,bank['count'])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
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
    import time,math
    start=time.time()
    state=_prepare(weights)
    state['tiles']={}
    device=weights['w13'].device
    flush=torch.ones(64<<20,dtype=torch.float32,device=device)
    h=weights['w13'].shape[-1]*2
    for m in [1,8,16,32,128,256]:
        if time.time()-start>230:
            break
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
        best,besttime=8,float('inf')
        for tile in [8,16,32,48,64]:
            state['tiles'][m]=tile
            times=[_bench(lambda:kernel_fn(d,state),flush) for d in datasets]
            score=math.exp(sum(math.log(t) for t in times)/len(times))
            print(f'm={m} tile={tile} times={times} geom={score:.2f}')
            if score<besttime:
                best,besttime=tile,score
        state['tiles'][m]=best
        print(f'SELECT m={m} tile={best} us={besttime:.2f}')
    del flush
    print(f'prepare {time.time()-start:.1f}s')
    return state
