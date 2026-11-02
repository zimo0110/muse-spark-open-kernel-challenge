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
    gs = tl.load(G + e)
    return (v * (scale * gs)).to(tl.bfloat16)


@tr.jit
def _gemm(X, W, S, G, Map, Count, Ends, Out,
          M: tl.constexpr, E: tl.constexpr, BE: tl.constexpr,
          N: tl.constexpr, K: tl.constexpr, FIRST: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    block = tl.program_id(1)
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
        if FIRST:
            acc2 = tl.zeros((BM, BN), tl.float32)
        for ki in range(K // BK):
            ks = ki * BK + kk
            x = tl.load(X + tok[:, None] * K + ks[None, :], rows[:, None] < count, other=0)
            w = _decode(W, S, G, e, ns, ks, N * 2 if FIRST else N, K)
            acc = tl.dot(x, tl.trans(w), acc)
            if FIRST:
                w2 = _decode(W, S, G, e, ns + N, ks, N * 2, K)
                acc2 = tl.dot(x, tl.trans(w2), acc2)
        if FIRST:
            gate = acc.to(tl.bfloat16).to(tl.float32)
            up = acc2.to(tl.bfloat16).to(tl.float32)
            act = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32) * up
            val = act.to(tl.bfloat16)
        else:
            val = acc.to(tl.bfloat16)
        tl.store(Out + route[:, None] * N + ns[None, :], val, rows[:, None] < count)


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


def prepare(weights):
    return weights


def kernel_fn(data, state):
    x = data['hidden_states']
    c = data['config']
    m, h = x.shape
    i, e = c['d_expert'], c['local_count']
    start = c['local_start']
    bm, bn, bk = 16, 64, 128
    route = torch.empty((e, m * 8), device=x.device, dtype=torch.int32)
    count = torch.empty((e,), device=x.device, dtype=torch.int32)
    ends = torch.empty_like(count)
    inter = torch.empty((m * 8, i), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((m * 8, h), device=x.device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    _routes[(e,)](data['topk_ids'], route, count, m, start, tr.next_power_of_2(m * 8))
    _schedule[(1,)](count, ends, e, tr.next_power_of_2(e), bm)
    blocks = e + tr.cdiv(m * 8, bm)
    for first, inp, dst, n, k, key in [(True, x, inter, i, h, 'w13'), (False, inter, y, h, i, 'w2')]:
        _gemm[(tr.cdiv(n, bn), blocks)](inp, state[key], state[key+'_sf'], state[key+'_gs'],
            route, count, ends, dst, m, e, tr.next_power_of_2(e), n, k, first,
            bm, bn, bk, num_warps=4, num_stages=3)
    _finish[(m,)](y, data['topk_ids'], data['topk_weights'], state['post_norm_w'], out,
                  h, start, e, tr.next_power_of_2(h), num_warps=4)
    return out
