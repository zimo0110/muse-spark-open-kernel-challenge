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


def prepare(weights):
    return weights


def kernel_fn(data, state):
    x = data['hidden_states']
    c = data['config']
    m, h = x.shape
    i, e = c['d_expert'], c['local_count']
    start = c['local_start']
    bm, bn, bk = (16 if m <= 32 else 32), 64, 128
    split = 8 if m <= 32 else 4
    route = torch.empty((e, m * 8), device=x.device, dtype=torch.int32)
    count = torch.empty((e,), device=x.device, dtype=torch.int32)
    ends = torch.empty_like(count)
    inter = torch.empty((m * 8, i), device=x.device, dtype=torch.bfloat16)
    partial1 = torch.empty((split, m * 8, 2 * i), device=x.device, dtype=torch.float32)
    partial2 = torch.empty((split, m * 8, h), device=x.device, dtype=torch.float32)
    y = torch.empty((m * 8, h), device=x.device, dtype=torch.bfloat16)
    ss = torch.empty((m * 8, h // 256), device=x.device, dtype=torch.float32)
    out = torch.empty_like(x)
    _routes[(e,)](data['topk_ids'], route, count, m, start, tr.next_power_of_2(m * 8))
    _schedule[(1,)](count, ends, e, tr.next_power_of_2(e), bm)
    blocks = min(m * 8, e + tr.cdiv(m * 8, bm) - 1)
    _gemm[(tr.cdiv(2*i, bn), blocks, split)](x, state['w13'], state['w13_sf'], state['w13_gs'],
        route, count, ends, partial1, m, e, tr.next_power_of_2(e), 2*i, h, True,
        bm, bn, bk, split, num_warps=4, num_stages=3)
    _act[(m*8, tr.cdiv(i, 256))](partial1, data['topk_ids'], inter, m, i, start, e, split, 256)
    _gemm[(tr.cdiv(h, bn), blocks, split)](inter, state['w2'], state['w2_sf'], state['w2_gs'],
        route, count, ends, partial2, m, e, tr.next_power_of_2(e), h, i, False,
        bm, bn, bk, split, num_warps=4, num_stages=3)
    _reduce_norm[(m*8, h//256)](partial2, data['topk_ids'], state['post_norm_w'], y, ss,
        m, h, start, e, split, 256)
    _finish2[(m, h//256)](y, ss, data['topk_ids'], data['topk_weights'], out, h, start, e, h//256, 256)
    return out
