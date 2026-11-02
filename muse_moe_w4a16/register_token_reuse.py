# okc: problem=muse_moe_w4a16
# Independent fresh-weight correctness confirmation of the selected register kernel.
"""Experimental direct-register NVFP4 MoE, NOT YET GPU-VERIFIED.

Local changes: one-pass lossless GPU packing, gate/up-paired W13 for every
shape, compact per-expert routes, adaptive item sizes and register prefetch.
Warp-MMA fragment layouts, bit decoder and activation shared-layout helpers
are adapted from team bigboateng's MIT-licensed public implementation:
https://github.com/operatorstack/open-kernel-challenge-solutions
commit 6d9b77656b6c295e421be9e1424b0f8f99299b0a
The existing scored submission is maintained separately in submission.py.

MIT License

Copyright (c) 2026 bigboateng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
# Local modifications licensed under Apache-2.0; reused portions retain MIT notice above.
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
_b64SCALE_BIAS = gl.constexpr(2.0 ** 119)
_b64ACC_BIAS = gl.constexpr(2.0 ** 7)
_b64_DECODE_ASM16 = gl.constexpr('\n{\n.reg .b32 t, u;\nand.b32 t, $8, 0x81C081C0;\nmul.rn.bf16x2 $0, t, $24;\nand.b32 t, $10, 0x81C081C0;\nmul.rn.bf16x2 $1, t, $25;\nshf.l.wrap.b32 t, $8, $8, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $2, t, $24;\nshf.l.wrap.b32 t, $10, $10, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $3, t, $25;\nshf.l.wrap.b32 t, $8, $8, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $4, t, $24;\nshf.l.wrap.b32 t, $10, $10, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $5, t, $25;\nshf.l.wrap.b32 t, $8, $8, 6;\nshf.l.wrap.b32 u, $8, $8, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $6, t, $24;\nshf.l.wrap.b32 t, $10, $10, 6;\nshf.l.wrap.b32 u, $10, $10, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $7, t, $25;\n}\n')
_b64MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
_b64A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=_b64MMA_L.value, k_width=2))
_b64B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=_b64MMA_L.value, k_width=2))
_b64L3 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 2]))
_b64LW16 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]], lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]], warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 2, 2, 2, 2]))
_b64LS = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 1]))
_b64LX = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 16]))
_b64LX2 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 16]))
_b64LX32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [8, 0, 0], [16, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 16]))
_b64LX2_32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 16]))
_b64LX8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 16]))
_b64LX2_8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 16]))
_b64LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 2], warps_per_cta=[4, 1], order=[1, 0]))
_b64X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=2, order=[1, 0]))

_b64LX64 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [8, 0, 0], [16, 0, 0], [32, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[64, 4, 16]))
_b64LX2_64 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [32, 0], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[256, 16]))
_b128LX64 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [8, 0, 0], [16, 0, 0], [32, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[64, 4, 32]))
_b128LX2_64 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [32, 0], [64, 0], [128, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[256, 32]))

@gluon.jit
def _b64_idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2), gl.expand_dims(gl.expand_dims(c, 0), 1))

@gluon.jit
def _b64_x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)

@gluon.jit
def _b64_decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, _b64L3: gl.constexpr, _b64A_L: gl.constexpr):
    """Packed words [BN, 4, 2] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * _b64SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 2])
    w16 = gl.convert_layout(_b64_x8(w3), _b64LW16)
    s16 = gl.convert_layout(_b64_x8(s8), _b64LW16)
    t = gl.inline_asm_elementwise(_b64_DECODE_ASM16, '=r,=r,=r,=r,=r,=r,=r,=r,' + ','.join(['r'] * 24), [w16, s16], dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, _b64A_L, assert_trivial=True)

@gluon.jit
def _b64_xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, _b64LXG))
    cc = gl.arange(0, 16, layout=gl.SliceLayout(0, _b64LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + rr % 4 * 16, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return (off, msk)

@gluon.jit
def _b64_x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(_b64LX2_8), [BM, 4, 16])
    elif BM == 32:
        xv = gl.reshape(x_s.load(_b64LX2_32), [BM, 4, 16])
    elif BM == 64:
        xv = gl.reshape(x_s.load(_b64LX2_64), [BM, 4, 16])
    else:
        xv = gl.reshape(x_s.load(_b64LX2), [BM, 4, 16])
    return _b64_x_tile(xv, BM, BK, _b64B_L)

@gluon.jit
def _b64_x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, _b64B_L: gl.constexpr):
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 2, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, _b64B_L, assert_trivial=BM < 32)

@gluon.jit
def _b64_gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))

@triton.jit
def _b64_pack_codes(W, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 64)
    nt = tile // (K // 64)
    j = tl.arange(0, B)
    row = nt * 64 + j // 8
    word = kb * 8 + j % 8
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    q = tl.load(W + e.to(tl.int64) * N * (K // 8) + source_row * (K // 8) + word).to(tl.uint32)
    packed = tl.full((B,), 0, tl.uint32)
    packed |= (q >> 0 & 1) << 6
    packed |= (q >> 1 & 1) << 7
    packed |= (q >> 2 & 1) << 8
    packed |= (q >> 3 & 1) << 15
    packed |= (q >> 4 & 1) << 22
    packed |= (q >> 5 & 1) << 23
    packed |= (q >> 6 & 1) << 24
    packed |= (q >> 7 & 1) << 31
    packed |= (q >> 8 & 1) << 2
    packed |= (q >> 9 & 1) << 3
    packed |= (q >> 10 & 1) << 4
    packed |= (q >> 11 & 1) << 11
    packed |= (q >> 12 & 1) << 18
    packed |= (q >> 13 & 1) << 19
    packed |= (q >> 14 & 1) << 20
    packed |= (q >> 15 & 1) << 27
    packed |= (q >> 16 & 1) << 28
    packed |= (q >> 17 & 1) << 29
    packed |= (q >> 18 & 1) << 30
    packed |= (q >> 19 & 1) << 5
    packed |= (q >> 20 & 1) << 12
    packed |= (q >> 21 & 1) << 13
    packed |= (q >> 22 & 1) << 14
    packed |= (q >> 23 & 1) << 21
    packed |= (q >> 24 & 1) << 0
    packed |= (q >> 25 & 1) << 1
    packed |= (q >> 26 & 1) << 26
    packed |= (q >> 27 & 1) << 9
    packed |= (q >> 28 & 1) << 16
    packed |= (q >> 29 & 1) << 17
    packed |= (q >> 30 & 1) << 10
    packed |= (q >> 31 & 1) << 25
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 64) + tile) * 512 + j, packed)

@triton.jit
def _b64_pack_scales(S, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 64)
    nt = tile // (K // 64)
    j = tl.arange(0, B)
    row = nt * 64 + j // 4
    group = kb * 4 + j % 4
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    value = tl.load(S + e.to(tl.int64) * N * (K // 16) + source_row * (K // 16) + group)
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 64) + tile) * 256 + j, value)

@triton.jit
def _b64_pack_weights(W, S, Out, Scales, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr):
    _b64_pack_codes(W, Out, N, K, GU, 512)
    _b64_pack_scales(S, Scales, N, K, GU, 256)

@triton.jit
def _b64_compact_routes(Ids, Map, Count, P: tl.constexpr, START: tl.constexpr, E: tl.constexpr, R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(Ids + r, r < P, other=-1)
    live = (ids == START + e) & (r < P)
    rank = tl.cumsum(live.to(tl.int32)) - 1
    tl.store(Map + e * P + rank, r, live)
    tl.store(Count + e, tl.sum(live.to(tl.int32)))

@triton.jit
def _b64_work_plan(Count, Work, NWork, E: tl.constexpr, BM: tl.constexpr, WMAX: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    count = tl.load(Count + e, e < E, other=0)
    blocks = tl.cdiv(count, BM)
    end = tl.cumsum(blocks)
    begin = end - blocks
    w = tl.arange(0, WMAX)
    hit = (w[:, None] >= begin[None, :]) & (w[:, None] < end[None, :]) & (e[None, :] < E)
    owner = tl.sum(tl.where(hit, e[None, :], 0), 1)
    offset = (w - tl.sum(tl.where(hit, begin[None, :], 0), 1)) * BM
    cnt = tl.sum(tl.where(hit, count[None, :], 0), 1)
    live = w < tl.sum(blocks)
    tl.store(Work + 3 * w, tl.where(live, owner, 0))
    tl.store(Work + 3 * w + 1, tl.where(live, offset, 0))
    tl.store(Work + 3 * w + 2, tl.where(live, tl.minimum(offset + BM, cnt), 0))
    tl.store(NWork, tl.sum(blocks))

@gluon.jit
def _b64_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K: gl.constexpr, N: gl.constexpr, P, BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 64
    rw, qw, jw = _b64_idx3(BN, 4, 2, _b64L3)
    rs, qs, gs = _b64_idx3(BN, 4, 1, _b64LS)
    blocks: gl.constexpr = K // BK
    wo = (e * (N // BN) + pn) * blocks * 512 + rw * 8 + qw * 2 + jw
    so = (e * (N // BN) + pn) * blocks * 256 + rs * 4 + qs + gs
    xoff, xmask = _b64_xg_offsets(Map + e * P, r0, r1, K, GU, BM)
    xs = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 16], _b64X_SMEM)
    acc = gl.zeros([BN, BM], gl.float32, layout=_b64MMA_L)
    if PREFETCH:
        next_w = gl.load(W + wo)
        next_s = gl.load(S + so)
    for k in range(blocks):
        if PREFETCH:
            w = next_w
            s = next_s
            next_w = gl.load(W + wo + (k + 1) * 512, mask=(rw >= 0) & (k + 1 < blocks), other=0)
            next_s = gl.load(S + so + (k + 1) * 256, mask=(rs >= 0) & (k + 1 < blocks), other=0)
        else:
            w = gl.load(W + wo + k * 512)
            s = gl.load(S + so + k * 256)
        xb = _b64_x_shared(X, xoff, xmask, k * BK, xs, BM, BK)
        a = _b64_decode_tile(w, s, BN, BK, _b64L3, _b64A_L)
        acc = mma_v2(a, xb, acc)
    y = acc * (gl.load(GS + e) * _b64ACC_BIAS)
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, _b64MMA_L))
    m = gl.arange(0, BM, layout=gl.SliceLayout(0, _b64MMA_L))
    row = r0 + m
    pair = gl.load(Map + e * P + row, row < r1, other=0)
    if GU:
        rows2 = gl.expand_dims(n, 1) + gl.expand_dims(m * 0, 0)
        pair2 = gl.expand_dims(pair, 0) + gl.expand_dims(n * 0, 1)
        mask2 = gl.expand_dims(row < r1, 0) & gl.expand_dims(n >= 0, 1)
        gate, up = _b64_gu_split(y, BN, BM)
        nr, _ = _b64_gu_split(rows2, BN, BM)
        pr, _ = _b64_gu_split(pair2, BN, BM)
        mr, _ = _b64_gu_split(mask2.to(gl.int32), BN, BM)
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        silu = (gate / (1 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value = (silu * up).to(gl.bfloat16)
        col = pn * 32 + nr // 16 * 8 + nr % 8
        gl.store(Out + pr * (N // 2) + col, value, mask=mr != 0)
    else:
        gl.store(Out + gl.expand_dims(pair * N, 0) + gl.expand_dims(pn * BN + n, 1), y.to(gl.bfloat16), mask=gl.expand_dims(row < r1, 0))

@gluon.jit
def _b64_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    e = gl.load(Work + 3 * wi)
    r0 = gl.load(Work + 3 * wi + 1)
    r1 = gl.load(Work + 3 * wi + 2)
    if r1 - r0 <= 8:
        _b64_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 8, GU, PREFETCH)
    elif MAX_BM >= 16:
        if r1 - r0 <= 16:
            _b64_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 16, GU, PREFETCH)
        elif MAX_BM >= 32:
            if r1 - r0 <= 32:
                _b64_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 32, GU, PREFETCH)
            elif MAX_BM == 64:
                _b64_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 64, GU, PREFETCH)

@gluon.jit
def _b64_register_dispatch(X, W, S, GS, Map, Work, NWork, Out, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr, PERSISTENT: gl.constexpr):
    if PERSISTENT:
        item = gl.program_id(0)
        total = gl.load(NWork) * (N // 64)
        while item < total:
            _b64_register_work(X, W, S, GS, Map, Work, Out, item % (N // 64), item // (N // 64), K, N, P, MAX_BM, GU, PREFETCH)
            item += gl.num_programs(0)
    else:
        pn = gl.program_id(0)
        wi = gl.program_id(1)
        if wi < gl.load(NWork):
            _b64_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K, N, P, MAX_BM, GU, PREFETCH)

@triton.jit
def _b64_finish_register(Y, Ids, Gates, NW, Out, H: tl.constexpr, START: tl.constexpr, E: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    s = tl.arange(0, 8)
    ids = tl.load(Ids + t * 8 + s)
    valid = (ids >= START) & (ids < START + E)
    nw = 1 + tl.load(NW + h, h < H, other=0).to(tl.float32)
    y = tl.load(Y + (t * 8 + s[:, None]) * H + h[None, :], valid[:, None] & (h[None, :] < H), other=0).to(tl.float32)
    y = (y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, 1) / H + 1e-08)
    gates = tl.load(Gates + t * 8 + s)
    z = tl.sum(y * (inv * gates)[:, None], 0)
    tl.store(Out + t * H + h, z.to(tl.bfloat16), h < H)

def _b64_pack_bank(raw, sf, gate_up):
    e, n, half_k = raw.shape
    k = half_k * 2
    if n % 64 or k % 64:
        raise ValueError('direct-register layout requires rows divisible by64 and K divisible by256')
    codes = torch.empty((e, n // 64, k // 64, 64, 8), device=raw.device, dtype=torch.int32)
    scales = torch.empty((e, n // 64, k // 64, 64, 4), device=raw.device, dtype=torch.uint8)
    grid = (n // 64 * (k // 64), e)
    _b64_pack_weights[grid](raw.contiguous().view(torch.int32), sf.contiguous().view(torch.uint8), codes, scales, n, k, gate_up)
    return (codes, scales)

_b128SCALE_BIAS = gl.constexpr(2.0 ** 119)
_b128ACC_BIAS = gl.constexpr(2.0 ** 7)
_b128_DECODE_ASM16 = gl.constexpr('\n{\n.reg .b32 t, u;\nand.b32 t, $8, 0x81C081C0;\nmul.rn.bf16x2 $0, t, $24;\nand.b32 t, $10, 0x81C081C0;\nmul.rn.bf16x2 $1, t, $25;\nshf.l.wrap.b32 t, $8, $8, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $2, t, $24;\nshf.l.wrap.b32 t, $10, $10, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $3, t, $25;\nshf.l.wrap.b32 t, $8, $8, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $4, t, $24;\nshf.l.wrap.b32 t, $10, $10, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $5, t, $25;\nshf.l.wrap.b32 t, $8, $8, 6;\nshf.l.wrap.b32 u, $8, $8, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $6, t, $24;\nshf.l.wrap.b32 t, $10, $10, 6;\nshf.l.wrap.b32 u, $10, $10, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $7, t, $25;\n}\n')
_b128MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
_b128A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=_b128MMA_L.value, k_width=2))
_b128B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=_b128MMA_L.value, k_width=2))
_b128L3 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
_b128LW16 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0]], lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]], warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 4, 2, 2, 2]))
_b128LS = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 2]))
_b128LX = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 32]))
_b128LX2 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [32, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 32]))
_b128LX32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [8, 0, 0], [16, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 32]))
_b128LX2_32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [32, 0], [64, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 32]))
_b128LX8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 32]))
_b128LX2_8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 32]))
_b128LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[8, 4], warps_per_cta=[4, 1], order=[1, 0]))
_b128X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=4, order=[1, 0]))

@gluon.jit
def _b128_idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2), gl.expand_dims(gl.expand_dims(c, 0), 1))

@gluon.jit
def _b128_x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)

@gluon.jit
def _b128_decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, _b128L3: gl.constexpr, _b128A_L: gl.constexpr):
    """Packed words [BN, 4, 4] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * _b128SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 4])
    w16 = gl.convert_layout(_b128_x8(w3), _b128LW16)
    s16 = gl.convert_layout(_b128_x8(s8), _b128LW16)
    t = gl.inline_asm_elementwise(_b128_DECODE_ASM16, '=r,=r,=r,=r,=r,=r,=r,=r,' + ','.join(['r'] * 24), [w16, s16], dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, _b128A_L, assert_trivial=True)

@gluon.jit
def _b128_xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, _b128LXG))
    cc = gl.arange(0, 32, layout=gl.SliceLayout(0, _b128LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + rr % 4 * 32, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return (off, msk)

@gluon.jit
def _b128_x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(_b128LX2_8), [BM, 4, 32])
    elif BM == 32:
        xv = gl.reshape(x_s.load(_b128LX2_32), [BM, 4, 32])
    elif BM == 64:
        xv = gl.reshape(x_s.load(_b128LX2_64), [BM, 4, 32])
    else:
        xv = gl.reshape(x_s.load(_b128LX2), [BM, 4, 32])
    return _b128_x_tile(xv, BM, BK, _b128B_L)

@gluon.jit
def _b128_x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, _b128B_L: gl.constexpr):
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 4, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, _b128B_L, assert_trivial=BM < 32)

@gluon.jit
def _b128_gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))

@triton.jit
def _b128_pack_codes(W, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 128)
    nt = tile // (K // 128)
    j = tl.arange(0, B)
    row = nt * 64 + j // 16
    word = kb * 16 + j % 16
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    q = tl.load(W + e.to(tl.int64) * N * (K // 8) + source_row * (K // 8) + word).to(tl.uint32)
    packed = tl.full((B,), 0, tl.uint32)
    packed |= (q >> 0 & 1) << 6
    packed |= (q >> 1 & 1) << 7
    packed |= (q >> 2 & 1) << 8
    packed |= (q >> 3 & 1) << 15
    packed |= (q >> 4 & 1) << 22
    packed |= (q >> 5 & 1) << 23
    packed |= (q >> 6 & 1) << 24
    packed |= (q >> 7 & 1) << 31
    packed |= (q >> 8 & 1) << 2
    packed |= (q >> 9 & 1) << 3
    packed |= (q >> 10 & 1) << 4
    packed |= (q >> 11 & 1) << 11
    packed |= (q >> 12 & 1) << 18
    packed |= (q >> 13 & 1) << 19
    packed |= (q >> 14 & 1) << 20
    packed |= (q >> 15 & 1) << 27
    packed |= (q >> 16 & 1) << 28
    packed |= (q >> 17 & 1) << 29
    packed |= (q >> 18 & 1) << 30
    packed |= (q >> 19 & 1) << 5
    packed |= (q >> 20 & 1) << 12
    packed |= (q >> 21 & 1) << 13
    packed |= (q >> 22 & 1) << 14
    packed |= (q >> 23 & 1) << 21
    packed |= (q >> 24 & 1) << 0
    packed |= (q >> 25 & 1) << 1
    packed |= (q >> 26 & 1) << 26
    packed |= (q >> 27 & 1) << 9
    packed |= (q >> 28 & 1) << 16
    packed |= (q >> 29 & 1) << 17
    packed |= (q >> 30 & 1) << 10
    packed |= (q >> 31 & 1) << 25
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 128) + tile) * 1024 + j, packed)

@triton.jit
def _b128_pack_scales(S, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 128)
    nt = tile // (K // 128)
    j = tl.arange(0, B)
    row = nt * 64 + j // 8
    group = kb * 8 + j % 8
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    value = tl.load(S + e.to(tl.int64) * N * (K // 16) + source_row * (K // 16) + group)
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 128) + tile) * 512 + j, value)

@triton.jit
def _b128_pack_weights(W, S, Out, Scales, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr):
    _b128_pack_codes(W, Out, N, K, GU, 1024)
    _b128_pack_scales(S, Scales, N, K, GU, 512)

@triton.jit
def _b128_compact_routes(Ids, Map, Count, P: tl.constexpr, START: tl.constexpr, E: tl.constexpr, R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(Ids + r, r < P, other=-1)
    live = (ids == START + e) & (r < P)
    rank = tl.cumsum(live.to(tl.int32)) - 1
    tl.store(Map + e * P + rank, r, live)
    tl.store(Count + e, tl.sum(live.to(tl.int32)))

@triton.jit
def _b128_work_plan(Count, Work, NWork, E: tl.constexpr, BM: tl.constexpr, WMAX: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    count = tl.load(Count + e, e < E, other=0)
    blocks = tl.cdiv(count, BM)
    end = tl.cumsum(blocks)
    begin = end - blocks
    w = tl.arange(0, WMAX)
    hit = (w[:, None] >= begin[None, :]) & (w[:, None] < end[None, :]) & (e[None, :] < E)
    owner = tl.sum(tl.where(hit, e[None, :], 0), 1)
    offset = (w - tl.sum(tl.where(hit, begin[None, :], 0), 1)) * BM
    cnt = tl.sum(tl.where(hit, count[None, :], 0), 1)
    live = w < tl.sum(blocks)
    tl.store(Work + 3 * w, tl.where(live, owner, 0))
    tl.store(Work + 3 * w + 1, tl.where(live, offset, 0))
    tl.store(Work + 3 * w + 2, tl.where(live, tl.minimum(offset + BM, cnt), 0))
    tl.store(NWork, tl.sum(blocks))

@gluon.jit
def _b128_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K: gl.constexpr, N: gl.constexpr, P, BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 128
    rw, qw, jw = _b128_idx3(BN, 4, 4, _b128L3)
    rs, qs, gs = _b128_idx3(BN, 4, 2, _b128LS)
    blocks: gl.constexpr = K // BK
    wo = (e * (N // BN) + pn) * blocks * 1024 + rw * 16 + qw * 4 + jw
    so = (e * (N // BN) + pn) * blocks * 512 + rs * 8 + qs * 2 + gs
    xoff, xmask = _b128_xg_offsets(Map + e * P, r0, r1, K, GU, BM)
    xs = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 32], _b128X_SMEM)
    acc = gl.zeros([BN, BM], gl.float32, layout=_b128MMA_L)
    if PREFETCH:
        next_w = gl.load(W + wo)
        next_s = gl.load(S + so)
    for k in range(blocks):
        if PREFETCH:
            w = next_w
            s = next_s
            next_w = gl.load(W + wo + (k + 1) * 1024, mask=(rw >= 0) & (k + 1 < blocks), other=0)
            next_s = gl.load(S + so + (k + 1) * 512, mask=(rs >= 0) & (k + 1 < blocks), other=0)
        else:
            w = gl.load(W + wo + k * 1024)
            s = gl.load(S + so + k * 512)
        xb = _b128_x_shared(X, xoff, xmask, k * BK, xs, BM, BK)
        a = _b128_decode_tile(w, s, BN, BK, _b128L3, _b128A_L)
        acc = mma_v2(a, xb, acc)
    y = acc * (gl.load(GS + e) * _b128ACC_BIAS)
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, _b128MMA_L))
    m = gl.arange(0, BM, layout=gl.SliceLayout(0, _b128MMA_L))
    row = r0 + m
    pair = gl.load(Map + e * P + row, row < r1, other=0)
    if GU:
        rows2 = gl.expand_dims(n, 1) + gl.expand_dims(m * 0, 0)
        pair2 = gl.expand_dims(pair, 0) + gl.expand_dims(n * 0, 1)
        mask2 = gl.expand_dims(row < r1, 0) & gl.expand_dims(n >= 0, 1)
        gate, up = _b128_gu_split(y, BN, BM)
        nr, _ = _b128_gu_split(rows2, BN, BM)
        pr, _ = _b128_gu_split(pair2, BN, BM)
        mr, _ = _b128_gu_split(mask2.to(gl.int32), BN, BM)
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        silu = (gate / (1 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value = (silu * up).to(gl.bfloat16)
        col = pn * 32 + nr // 16 * 8 + nr % 8
        gl.store(Out + pr * (N // 2) + col, value, mask=mr != 0)
    else:
        gl.store(Out + gl.expand_dims(pair * N, 0) + gl.expand_dims(pn * BN + n, 1), y.to(gl.bfloat16), mask=gl.expand_dims(row < r1, 0))

@gluon.jit
def _b128_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    e = gl.load(Work + 3 * wi)
    r0 = gl.load(Work + 3 * wi + 1)
    r1 = gl.load(Work + 3 * wi + 2)
    if r1 - r0 <= 8:
        _b128_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 8, GU, PREFETCH)
    elif MAX_BM >= 16:
        if r1 - r0 <= 16:
            _b128_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 16, GU, PREFETCH)
        elif MAX_BM >= 32:
            if r1 - r0 <= 32:
                _b128_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 32, GU, PREFETCH)
            elif MAX_BM == 64:
                _b128_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 64, GU, PREFETCH)

@gluon.jit
def _b128_register_dispatch(X, W, S, GS, Map, Work, NWork, Out, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr, PERSISTENT: gl.constexpr):
    if PERSISTENT:
        item = gl.program_id(0)
        total = gl.load(NWork) * (N // 64)
        while item < total:
            _b128_register_work(X, W, S, GS, Map, Work, Out, item % (N // 64), item // (N // 64), K, N, P, MAX_BM, GU, PREFETCH)
            item += gl.num_programs(0)
    else:
        pn = gl.program_id(0)
        wi = gl.program_id(1)
        if wi < gl.load(NWork):
            _b128_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K, N, P, MAX_BM, GU, PREFETCH)

@triton.jit
def _b128_finish_register(Y, Ids, Gates, NW, Out, H: tl.constexpr, START: tl.constexpr, E: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    s = tl.arange(0, 8)
    ids = tl.load(Ids + t * 8 + s)
    valid = (ids >= START) & (ids < START + E)
    nw = 1 + tl.load(NW + h, h < H, other=0).to(tl.float32)
    y = tl.load(Y + (t * 8 + s[:, None]) * H + h[None, :], valid[:, None] & (h[None, :] < H), other=0).to(tl.float32)
    y = (y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, 1) / H + 1e-08)
    gates = tl.load(Gates + t * 8 + s)
    z = tl.sum(y * (inv * gates)[:, None], 0)
    tl.store(Out + t * H + h, z.to(tl.bfloat16), h < H)

def _b128_pack_bank(raw, sf, gate_up):
    e, n, half_k = raw.shape
    k = half_k * 2
    if n % 64 or k % 128:
        raise ValueError('direct-register layout requires rows divisible by64 and K divisible by256')
    codes = torch.empty((e, n // 64, k // 128, 64, 16), device=raw.device, dtype=torch.int32)
    scales = torch.empty((e, n // 64, k // 128, 64, 8), device=raw.device, dtype=torch.uint8)
    grid = (n // 64 * (k // 128), e)
    _b128_pack_weights[grid](raw.contiguous().view(torch.int32), sf.contiguous().view(torch.uint8), codes, scales, n, k, gate_up)
    return (codes, scales)

_b256SCALE_BIAS = gl.constexpr(2.0 ** 119)
_b256ACC_BIAS = gl.constexpr(2.0 ** 7)
_b256_DECODE_ASM16 = gl.constexpr('\n{\n.reg .b32 t, u;\nand.b32 t, $8, 0x81C081C0;\nmul.rn.bf16x2 $0, t, $24;\nand.b32 t, $10, 0x81C081C0;\nmul.rn.bf16x2 $1, t, $25;\nshf.l.wrap.b32 t, $8, $8, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $2, t, $24;\nshf.l.wrap.b32 t, $10, $10, 4;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $3, t, $25;\nshf.l.wrap.b32 t, $8, $8, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $4, t, $24;\nshf.l.wrap.b32 t, $10, $10, 10;\nand.b32 t, t, 0x81C081C0;\nmul.rn.bf16x2 $5, t, $25;\nshf.l.wrap.b32 t, $8, $8, 6;\nshf.l.wrap.b32 u, $8, $8, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $6, t, $24;\nshf.l.wrap.b32 t, $10, $10, 6;\nshf.l.wrap.b32 u, $10, $10, 14;\nand.b32 t, t, 0x80C080C0;\nand.b32 u, u, 0x01000100;\nor.b32 t, t, u;\nmul.rn.bf16x2 $7, t, $25;\n}\n')
_b256MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
_b256A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=_b256MMA_L.value, k_width=2))
_b256B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=_b256MMA_L.value, k_width=2))
_b256L3 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 8]))
_b256LW16 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]], lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]], warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 8, 2, 2, 2]))
_b256LS = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
_b256LX = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 64]))
_b256LX2 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 64]))
_b256LX32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0], [16, 0, 0]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 64]))
_b256LX2_32 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 64]))
_b256LX8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]], lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 64]))
_b256LX2_8 = gl.constexpr(gl.DistributedLinearLayout(reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 64]))
_b256LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
_b256X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))

@gluon.jit
def _b256_idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2), gl.expand_dims(gl.expand_dims(c, 0), 1))

@gluon.jit
def _b256_x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)

@gluon.jit
def _b256_decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, _b256L3: gl.constexpr, _b256A_L: gl.constexpr):
    """Packed words [BN, 4, 8] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * _b256SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 8])
    w16 = gl.convert_layout(_b256_x8(w3), _b256LW16)
    s16 = gl.convert_layout(_b256_x8(s8), _b256LW16)
    t = gl.inline_asm_elementwise(_b256_DECODE_ASM16, '=r,=r,=r,=r,=r,=r,=r,=r,' + ','.join(['r'] * 24), [w16, s16], dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, _b256A_L, assert_trivial=True)

@gluon.jit
def _b256_xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, _b256LXG))
    cc = gl.arange(0, 64, layout=gl.SliceLayout(0, _b256LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + rr % 4 * 64, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return (off, msk)

@gluon.jit
def _b256_x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(_b256LX2_8), [BM, 4, 64])
    elif BM == 32:
        xv = gl.reshape(x_s.load(_b256LX2_32), [BM, 4, 64])
    else:
        xv = gl.reshape(x_s.load(_b256LX2), [BM, 4, 64])
    return _b256_x_tile(xv, BM, BK, _b256B_L)

@gluon.jit
def _b256_x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, _b256B_L: gl.constexpr):
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 8, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, _b256B_L, assert_trivial=BM != 32)

@gluon.jit
def _b256_gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))

@triton.jit
def _b256_pack_codes(W, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 256)
    nt = tile // (K // 256)
    j = tl.arange(0, B)
    row = nt * 64 + j // 32
    word = kb * 32 + j % 32
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    q = tl.load(W + e.to(tl.int64) * N * (K // 8) + source_row * (K // 8) + word).to(tl.uint32)
    packed = tl.full((B,), 0, tl.uint32)
    packed |= (q >> 0 & 1) << 6
    packed |= (q >> 1 & 1) << 7
    packed |= (q >> 2 & 1) << 8
    packed |= (q >> 3 & 1) << 15
    packed |= (q >> 4 & 1) << 22
    packed |= (q >> 5 & 1) << 23
    packed |= (q >> 6 & 1) << 24
    packed |= (q >> 7 & 1) << 31
    packed |= (q >> 8 & 1) << 2
    packed |= (q >> 9 & 1) << 3
    packed |= (q >> 10 & 1) << 4
    packed |= (q >> 11 & 1) << 11
    packed |= (q >> 12 & 1) << 18
    packed |= (q >> 13 & 1) << 19
    packed |= (q >> 14 & 1) << 20
    packed |= (q >> 15 & 1) << 27
    packed |= (q >> 16 & 1) << 28
    packed |= (q >> 17 & 1) << 29
    packed |= (q >> 18 & 1) << 30
    packed |= (q >> 19 & 1) << 5
    packed |= (q >> 20 & 1) << 12
    packed |= (q >> 21 & 1) << 13
    packed |= (q >> 22 & 1) << 14
    packed |= (q >> 23 & 1) << 21
    packed |= (q >> 24 & 1) << 0
    packed |= (q >> 25 & 1) << 1
    packed |= (q >> 26 & 1) << 26
    packed |= (q >> 27 & 1) << 9
    packed |= (q >> 28 & 1) << 16
    packed |= (q >> 29 & 1) << 17
    packed |= (q >> 30 & 1) << 10
    packed |= (q >> 31 & 1) << 25
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 256) + tile) * 2048 + j, packed)

@triton.jit
def _b256_pack_scales(S, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 256)
    nt = tile // (K // 256)
    j = tl.arange(0, B)
    row = nt * 64 + j // 16
    group = kb * 16 + j % 16
    if GU:
        source_row = row % 16 // 8 * (N // 2) + row // 64 * 32 + row % 64 // 16 * 8 + row % 8
    else:
        source_row = row
    value = tl.load(S + e.to(tl.int64) * N * (K // 16) + source_row * (K // 16) + group)
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 256) + tile) * 1024 + j, value)

@triton.jit
def _b256_pack_weights(W, S, Out, Scales, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr):
    _b256_pack_codes(W, Out, N, K, GU, 2048)
    _b256_pack_scales(S, Scales, N, K, GU, 1024)

@triton.jit
def _b256_compact_routes(Ids, Map, Count, P: tl.constexpr, START: tl.constexpr, E: tl.constexpr, R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(Ids + r, r < P, other=-1)
    live = (ids == START + e) & (r < P)
    rank = tl.cumsum(live.to(tl.int32)) - 1
    tl.store(Map + e * P + rank, r, live)
    tl.store(Count + e, tl.sum(live.to(tl.int32)))

@triton.jit
def _b256_work_plan(Count, Work, NWork, E: tl.constexpr, BM: tl.constexpr, WMAX: tl.constexpr, BE: tl.constexpr):
    e = tl.arange(0, BE)
    count = tl.load(Count + e, e < E, other=0)
    blocks = tl.cdiv(count, BM)
    end = tl.cumsum(blocks)
    begin = end - blocks
    w = tl.arange(0, WMAX)
    hit = (w[:, None] >= begin[None, :]) & (w[:, None] < end[None, :]) & (e[None, :] < E)
    owner = tl.sum(tl.where(hit, e[None, :], 0), 1)
    offset = (w - tl.sum(tl.where(hit, begin[None, :], 0), 1)) * BM
    cnt = tl.sum(tl.where(hit, count[None, :], 0), 1)
    live = w < tl.sum(blocks)
    tl.store(Work + 3 * w, tl.where(live, owner, 0))
    tl.store(Work + 3 * w + 1, tl.where(live, offset, 0))
    tl.store(Work + 3 * w + 2, tl.where(live, tl.minimum(offset + BM, cnt), 0))
    tl.store(NWork, tl.sum(blocks))

@gluon.jit
def _b256_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K: gl.constexpr, N: gl.constexpr, P, BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 256
    rw, qw, jw = _b256_idx3(BN, 4, 8, _b256L3)
    rs, qs, gs = _b256_idx3(BN, 4, 4, _b256LS)
    blocks: gl.constexpr = K // BK
    wo = (e * (N // BN) + pn) * blocks * 2048 + rw * 32 + qw * 8 + jw
    so = (e * (N // BN) + pn) * blocks * 1024 + rs * 16 + qs * 4 + gs
    xoff, xmask = _b256_xg_offsets(Map + e * P, r0, r1, K, GU, BM)
    xs = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], _b256X_SMEM)
    acc = gl.zeros([BN, BM], gl.float32, layout=_b256MMA_L)
    if PREFETCH:
        next_w = gl.load(W + wo)
        next_s = gl.load(S + so)
    for k in range(blocks):
        if PREFETCH:
            w = next_w
            s = next_s
            next_w = gl.load(W + wo + (k + 1) * 2048, mask=(rw >= 0) & (k + 1 < blocks), other=0)
            next_s = gl.load(S + so + (k + 1) * 1024, mask=(rs >= 0) & (k + 1 < blocks), other=0)
        else:
            w = gl.load(W + wo + k * 2048)
            s = gl.load(S + so + k * 1024)
        xb = _b256_x_shared(X, xoff, xmask, k * BK, xs, BM, BK)
        a = _b256_decode_tile(w, s, BN, BK, _b256L3, _b256A_L)
        acc = mma_v2(a, xb, acc)
    y = acc * (gl.load(GS + e) * _b256ACC_BIAS)
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, _b256MMA_L))
    m = gl.arange(0, BM, layout=gl.SliceLayout(0, _b256MMA_L))
    row = r0 + m
    pair = gl.load(Map + e * P + row, row < r1, other=0)
    if GU:
        rows2 = gl.expand_dims(n, 1) + gl.expand_dims(m * 0, 0)
        pair2 = gl.expand_dims(pair, 0) + gl.expand_dims(n * 0, 1)
        mask2 = gl.expand_dims(row < r1, 0) & gl.expand_dims(n >= 0, 1)
        gate, up = _b256_gu_split(y, BN, BM)
        nr, _ = _b256_gu_split(rows2, BN, BM)
        pr, _ = _b256_gu_split(pair2, BN, BM)
        mr, _ = _b256_gu_split(mask2.to(gl.int32), BN, BM)
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        silu = (gate / (1 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value = (silu * up).to(gl.bfloat16)
        col = pn * 32 + nr // 16 * 8 + nr % 8
        gl.store(Out + pr * (N // 2) + col, value, mask=mr != 0)
    else:
        gl.store(Out + gl.expand_dims(pair * N, 0) + gl.expand_dims(pn * BN + n, 1), y.to(gl.bfloat16), mask=gl.expand_dims(row < r1, 0))

@gluon.jit
def _b256_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    e = gl.load(Work + 3 * wi)
    r0 = gl.load(Work + 3 * wi + 1)
    r1 = gl.load(Work + 3 * wi + 2)
    if r1 - r0 <= 8:
        _b256_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 8, GU, PREFETCH)
    elif MAX_BM >= 16:
        if r1 - r0 <= 16:
            _b256_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 16, GU, PREFETCH)
        elif MAX_BM == 32:
            _b256_register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 32, GU, PREFETCH)

@gluon.jit
def _b256_register_dispatch(X, W, S, GS, Map, Work, NWork, Out, K: gl.constexpr, N: gl.constexpr, P, MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr, PERSISTENT: gl.constexpr):
    if PERSISTENT:
        item = gl.program_id(0)
        total = gl.load(NWork) * (N // 64)
        while item < total:
            _b256_register_work(X, W, S, GS, Map, Work, Out, item % (N // 64), item // (N // 64), K, N, P, MAX_BM, GU, PREFETCH)
            item += gl.num_programs(0)
    else:
        pn = gl.program_id(0)
        wi = gl.program_id(1)
        if wi < gl.load(NWork):
            _b256_register_work(X, W, S, GS, Map, Work, Out, pn, wi, K, N, P, MAX_BM, GU, PREFETCH)

@triton.jit
def _b256_finish_register(Y, Ids, Gates, NW, Out, H: tl.constexpr, START: tl.constexpr, E: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    s = tl.arange(0, 8)
    ids = tl.load(Ids + t * 8 + s)
    valid = (ids >= START) & (ids < START + E)
    nw = 1 + tl.load(NW + h, h < H, other=0).to(tl.float32)
    y = tl.load(Y + (t * 8 + s[:, None]) * H + h[None, :], valid[:, None] & (h[None, :] < H), other=0).to(tl.float32)
    y = (y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, 1) / H + 1e-08)
    gates = tl.load(Gates + t * 8 + s)
    z = tl.sum(y * (inv * gates)[:, None], 0)
    tl.store(Out + t * H + h, z.to(tl.bfloat16), h < H)

def _b256_pack_bank(raw, sf, gate_up):
    e, n, half_k = raw.shape
    k = half_k * 2
    if n % 64 or k % 256:
        raise ValueError('direct-register layout requires rows divisible by64 and K divisible by256')
    codes = torch.empty((e, n // 64, k // 256, 64, 32), device=raw.device, dtype=torch.int32)
    scales = torch.empty((e, n // 64, k // 256, 64, 16), device=raw.device, dtype=torch.uint8)
    grid = (n // 64 * (k // 256), e)
    _b256_pack_weights[grid](raw.contiguous().view(torch.int32), sf.contiguous().view(torch.uint8), codes, scales, n, k, gate_up)
    return (codes, scales)

@gluon.jit
def _one_direct_first(X, W, S, GS, Map, Ids, Out, K: gl.constexpr, N: gl.constexpr):
    pn = gl.program_id(0)
    slot = gl.program_id(1)
    e = gl.load(Ids + slot)
    if (e >= 0) & (e < 32):
        _b256_register_item(X, W, S, GS, Map, Out, e, slot, slot + 1, pn, K, N, 8, 8, True, True)

@gluon.jit
def _one_direct_w2_split(a_ptr, w_ptr, sf_ptr, gs_ptr, sorted_ptr, work_ptr, nwork_ptr, out_ptr, K: gl.constexpr, N: gl.constexpr, BK: gl.constexpr, BM: gl.constexpr, KS: gl.constexpr, PF: gl.constexpr, NLOC: gl.constexpr=32):
    BN: gl.constexpr = 64
    pid_n = gl.program_id(0)
    pid_w = gl.program_id(1)
    pid_k = gl.program_id(2)
    e = gl.load(work_ptr + pid_w) - gl.load(nwork_ptr)
    active = (e >= 0) & (e < NLOC)
    e = e * active.to(gl.int32)
    r0 = pid_w
    r1 = pid_w + active.to(gl.int32)
    if active:
        KB: gl.constexpr = K // BK
        KSB: gl.constexpr = KB // KS
        kb0 = pid_k * KSB
        rw, qw, jw = _b256_idx3(BN, 4, 8, _b256L3)
        w_off = (e * (N // BN) + pid_n) * KB * (BN * 32) + rw * 32 + qw * 8 + jw
        rs, qs, gs_ = _b256_idx3(BN, 4, 4, _b256LS)
        s_off = (e * (N // BN) + pid_n) * KB * (BN * 16) + rs * 16 + qs * 4 + gs_
        xoff, xmsk = _b256_xg_offsets(sorted_ptr, r0, r1, K, False, BM)
        x_s = gl.allocate_shared_memory(gl.bfloat16, [4 * BM, 64], _b256X_SMEM)
        wp = w_ptr.to(gl.pointer_type(gl.int32))
        acc = gl.zeros([BN, BM], gl.float32, layout=_b256MMA_L)
        if PF:
            wq_n = gl.load(wp + w_off + kb0 * (BN * 32))
            sq_n = gl.load(sf_ptr + s_off + kb0 * (BN * 16))
        for j in range(0, KSB):
            kb = kb0 + j
            if PF:
                wq = wq_n
                sq = sq_n
                more = j + 1 < KSB
                wq_n = gl.load(wp + w_off + (kb + 1) * (BN * 32), mask=(rw >= 0) & more, other=0)
                sq_n = gl.load(sf_ptr + s_off + (kb + 1) * (BN * 16), mask=(rs >= 0) & more)
            else:
                wq = gl.load(wp + w_off + kb * (BN * 32))
                sq = gl.load(sf_ptr + s_off + kb * (BN * 16))
            xb = _b256_x_shared(a_ptr, xoff, xmsk, kb * BK, x_s, BM, BK)
            acc = mma_v2(_b256_decode_tile(wq, sq, BN, BK, _b256L3, _b256A_L), xb, acc)
        y = acc * (gl.load(gs_ptr + e) * _b256ACC_BIAS)
        n_o = gl.arange(0, BN, layout=gl.SliceLayout(1, _b256MMA_L))
        m_o = gl.arange(0, BM, layout=gl.SliceLayout(0, _b256MMA_L))
        ro = r0 + m_o
        pair_o = gl.load(sorted_ptr + ro, mask=ro < r1, other=0)
        optr = out_ptr + pid_k * (8 * N) + gl.expand_dims(pair_o, 0) * N + gl.expand_dims(pid_n * BN + n_o, 1)
        gl.store(optr, y, mask=gl.expand_dims(ro < r1, 0) & gl.expand_dims(n_o >= 0, 1))

@triton.jit
def _one_finish_direct(P, Ids, TW, NW, Out, H: tl.constexpr, KS: tl.constexpr):
    h = tl.arange(0, H)
    norm = 1 + tl.load(NW + h).to(tl.float32)
    result = tl.full((H,), 0, tl.float32)
    for slot in tl.static_range(8):
        e = tl.load(Ids + slot)
        if (e >= 0) & (e < 32):
            v = tl.full((H,), 0, tl.float32)
            for split in tl.static_range(KS):
                v += tl.load(P + split * 8 * H + slot * H + h)
            v = v.to(tl.bfloat16).to(tl.float32)
            v = (v * norm).to(tl.bfloat16).to(tl.float32)
            inv = tl.rsqrt(tl.sum(v * v) / H + 1e-08)
            result += v * inv * tl.load(TW + slot)
    tl.store(Out + h, result.to(tl.bfloat16))

@gluon.jit
def _one_direct_first_partial(X, W, S, Map, Ids, Out, K: gl.constexpr, N: gl.constexpr, KS: gl.constexpr):
    pn = gl.program_id(0)
    slot = gl.program_id(1)
    sp = gl.program_id(2)
    e = gl.load(Ids + slot)
    if (e >= 0) & (e < 32):
        rw, qw, jw = _b256_idx3(64, 4, 8, _b256L3)
        rs, qs, gs = _b256_idx3(64, 4, 4, _b256LS)
        blocks: gl.constexpr = K // 256
        count: gl.constexpr = blocks // KS
        wo = (e * (N // 64) + pn) * blocks * 2048 + rw * 32 + qw * 8 + jw
        so = (e * (N // 64) + pn) * blocks * 1024 + rs * 16 + qs * 4 + gs
        off, mask = _b256_xg_offsets(Map + e * 8, slot, slot + 1, K, True, 8)
        xs = gl.allocate_shared_memory(gl.bfloat16, [32, 64], _b256X_SMEM)
        acc = gl.zeros([64, 8], gl.float32, layout=_b256MMA_L)
        next_w = gl.load(W + wo + sp * count * 2048)
        next_s = gl.load(S + so + sp * count * 1024)
        for j in range(count):
            k = sp * count + j
            w = next_w
            scale = next_s
            next_w = gl.load(W + wo + (k + 1) * 2048, mask=(rw >= 0) & (j + 1 < count), other=0)
            next_s = gl.load(S + so + (k + 1) * 1024, mask=(rs >= 0) & (j + 1 < count), other=0)
            xb = _b256_x_shared(X, off, mask, k * 256, xs, 8, 256)
            acc = mma_v2(_b256_decode_tile(w, scale, 64, 256, _b256L3, _b256A_L), xb, acc)
        n = gl.arange(0, 64, layout=gl.SliceLayout(1, _b256MMA_L))
        m = gl.arange(0, 8, layout=gl.SliceLayout(0, _b256MMA_L))
        gl.store(Out + sp * 8 * N + slot * N + gl.expand_dims(pn * 64 + n, 1) + gl.expand_dims(m * 0, 0), acc, mask=gl.expand_dims(m == 0, 0))

@triton.jit
def _one_direct_first_reduce(P, Ids, GS, Out, I: tl.constexpr, KS: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0)
    e = tl.load(Ids + slot)
    if (e >= 0) & (e < 32):
        c = tl.program_id(1) * B + tl.arange(0, B)
        row = c // 8 * 16 + c % 8
        gate = tl.full((B,), 0, tl.float32)
        up = tl.full((B,), 0, tl.float32)
        for sp in tl.static_range(KS):
            gate += tl.load(P + sp * 8 * 2 * I + slot * 2 * I + row, c < I, other=0)
            up += tl.load(P + sp * 8 * 2 * I + slot * 2 * I + row + 8, c < I, other=0)
        scale = tl.load(GS + e) * 128.0
        gate = (gate * scale).to(tl.bfloat16).to(tl.float32)
        up = (up * scale).to(tl.bfloat16).to(tl.float32)
        silu = (gate / (1 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(Out + slot * I + c, (silu * up).to(tl.bfloat16), c < I)

_PACK={64:_b64_pack_bank,128:_b128_pack_bank,256:_b256_pack_bank}
_DISPATCH={64:_b64_register_dispatch,128:_b128_register_dispatch,256:_b256_register_dispatch}


def kernel_fn(data,state):
    m,h=data['hidden_states'].shape;i=data['config']['d_expert'];dev=data['hidden_states'].device
    if (data['config']['local_start'],data['config']['local_count'])!=(0,32):raise ValueError('prepared for EP8 experts0..31')
    if m==1:
        bank=state['banks'][256]
        raw=torch.empty((4,8,2*i),device=dev,dtype=torch.float32)
        inter=torch.empty((8,i),device=dev,dtype=torch.bfloat16)
        _one_direct_first_partial[(2*i//64,8,4)](data['hidden_states'],bank['w13'],bank['s13'],state['map1'],data['topk_ids'],raw,h,2*i,4,num_warps=4)
        _one_direct_first_reduce[(8,triton.cdiv(i,256))](raw,data['topk_ids'],state['gs13'],inter,i,4,256)
        partial=torch.empty((4,8,h),device=dev,dtype=torch.float32)
        _one_direct_w2_split[(h//64,8,4)](inter,bank['w2'],bank['s2'],state['gs2'],state['iota'],data['topk_ids'],state['zero'],partial,K=i,N=h,BK=256,BM=8,KS=4,PF=True,NLOC=32,num_warps=4)
        out=torch.empty_like(data['hidden_states'])
        _one_finish_direct[(1,)](partial,data['topk_ids'],data['topk_weights'],state['norm'],out,h,4,num_warps=8)
        return out
    cfg=state['choices'].get(m,state['default']);p=m*8
    bm13=8 if m<=32 else cfg['bm13'];bm2=8 if m<=32 else cfg['bm2']
    mapping=torch.empty((32,p),device=dev,dtype=torch.int32);counts=torch.empty(32,device=dev,dtype=torch.int32)
    _b256_compact_routes[(32,)](data['topk_ids'],mapping,counts,p,0,32,triton.next_power_of_2(p))
    plans={}
    for bm in dict.fromkeys((bm13,bm2)):
        wmax=triton.next_power_of_2(triton.cdiv(p,bm)+32)
        work=torch.empty((wmax,3),device=dev,dtype=torch.int32);nwork=torch.empty(1,device=dev,dtype=torch.int32)
        _b256_work_plan[(1,)](counts,work,nwork,32,bm,wmax,32,num_warps=4);plans[bm]=(wmax,work,nwork)
    inter=torch.empty((p,i),device=dev,dtype=torch.bfloat16);y=torch.empty((p,h),device=dev,dtype=torch.bfloat16)
    a=state['banks'][cfg['bk13']];b=state['banks'][cfg['bk2']]
    wmax,work,nwork=plans[bm13]
    g1=_DISPATCH[cfg['bk13']][(2*i//64,wmax)](data['hidden_states'],a['w13'],a['s13'],state['gs13'],mapping,work,nwork,inter,h,2*i,p,bm13,True,cfg['pf13'],False,num_warps=4,maxnreg=cfg['cap13'])
    wmax,work,nwork=plans[bm2]
    g2=_DISPATCH[cfg['bk2']][(h//64,wmax)](inter,b['w2'],b['s2'],state['gs2'],mapping,work,nwork,y,i,h,p,bm2,False,cfg['pf2'],False,num_warps=4,maxnreg=cfg['cap2'])
    if state.get('report',False):print('RES',m,[(getattr(z,'n_regs',None),getattr(z,'n_spills',None)) for z in (g1,g2)])
    out=torch.empty_like(data['hidden_states'])
    _b256_finish_register[(m,)](y,data['topk_ids'],data['topk_weights'],state['norm'],out,h,0,32,triton.next_power_of_2(h),num_warps=8)
    return out


def _cold(fn,flush):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fn();fn()
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    events=[]
    for rep in range(8):
        flush.sum();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record();g.replay();b.record();events.append((a,b))
    torch.cuda.synchronize()
    return sorted(a.elapsed_time(b)*1000 for a,b in events)[4]


def prepare(weights):
    import math,time
    started=time.monotonic();dev=weights['w13'].device;h=weights['w13'].shape[-1]*2
    state={'banks':{},'gs13':weights['w13_gs'][:32].float().contiguous(),'gs2':weights['w2_gs'][:32].float().contiguous(),'norm':weights['post_norm_w'],'choices':{}}
    for bk,pack in _PACK.items():
        w13,s13=pack(weights['w13'][:32],weights['w13_sf'][:32],True)
        w2,s2=pack(weights['w2'][:32],weights['w2_sf'][:32],False)
        state['banks'][bk]={'w13':w13,'s13':s13,'w2':w2,'s2':s2}
    state['iota']=torch.arange(8,device=dev,dtype=torch.int32);state['map1']=state['iota'].repeat(32,1).contiguous();state['zero']=torch.zeros(1,device=dev,dtype=torch.int32)
    defaults={'bk13':256,'bk2':256,'bm13':16,'bm2':16,'cap13':128,'cap2':128,'pf13':False,'pf2':False};state['default']=defaults
    flush=torch.ones(64<<20,device=dev,dtype=torch.float32)
    for m in (8,16,32,128,256):
        datasets=[]
        for pop in (.02,.05):
            u=torch.rand(256,device=dev).clamp(min=1e-12);heat=torch.randn(256,device=dev)/(-u.log()).sqrt()*pop
            values,ids=(torch.rand((m,256),device=dev)+heat).topk(8,-1)
            busiest=torch.bincount((ids//32).flatten(),minlength=8).argmax()
            datasets.append({'hidden_states':torch.randn((m,h),device=dev,dtype=torch.bfloat16),'topk_ids':((ids-32*busiest)%256).to(torch.int32),'topk_weights':(values/values.sum(-1,keepdim=True)).float(),'config':{'local_start':0,'local_count':32,'d_expert':h,'bs':m}})
        state['choices'][m]=dict(defaults);refs=[kernel_fn(d,state).clone() for d in datasets]
        best=dict(defaults);besttime=float('inf')
        for key,options in [('bk13',[64,128,256]),('bk2',[64,128,256]),('bm13',[8,16,32] if m>32 else[8]),('bm2',[8,16,32] if m>32 else[8]),('cap13',[48,64,80,96,128,192]),('cap2',[48,64,80,96,128,192]),('pf13',[False,True]),('pf2',[False,True]),('bk13',[64,128,256]),('bk2',[64,128,256])]:
            winner=dict(best);win=besttime
            for value in options:
                if time.monotonic()-started>180:break
                trial=dict(best);trial[key]=value;state['choices'][m]=trial
                try:
                    errors=[float((kernel_fn(d,state).float()-r.float()).norm()/r.float().norm().clamp(min=1e-20)) for d,r in zip(datasets,refs)]
                    if any(not math.isfinite(x) or x>.004 for x in errors):print('REJECT',m,key,value,errors);continue
                    ts=[_cold(lambda:kernel_fn(d,state),flush) for d in datasets];score=math.sqrt(ts[0]*ts[1])
                    if score<win:winner,win=trial,score
                except Exception as e:print('SKIP',m,key,value,str(e)[-200:])
            best,besttime=winner,win;state['choices'][m]=best
        state['report']=True;kernel_fn(datasets[0],state);state['report']=False
        print('SELECT',m,best,besttime)
    print('prepare',time.monotonic()-started)
    return state


_reuse_base_prepare=prepare

def prepare(weights):
    import math,time
    started=time.monotonic();state=_reuse_base_prepare(weights)
    dev=weights['w13'].device;h=weights['w13'].shape[-1]*2
    flush=torch.ones(64<<20,device=dev,dtype=torch.float32)
    for m in (128,256):
        datasets=[]
        for pop in (.01,.03,.08):
            u=torch.rand(256,device=dev).clamp(min=1e-12)
            heat=torch.randn(256,device=dev)/(-u.log()).sqrt()*pop
            values,ids=(torch.rand((m,256),device=dev)+heat).topk(8,-1)
            busiest=torch.bincount((ids//32).flatten(),minlength=8).argmax()
            local=((ids-32*busiest)%256).to(torch.int32)
            datasets.append({'hidden_states':torch.randn((m,h),device=dev,dtype=torch.bfloat16),
                'topk_ids':local,'topk_weights':(values/values.sum(-1,keepdim=True)).float(),
                'config':{'local_start':0,'local_count':32,'d_expert':h,'bs':m}})
            counts=torch.bincount(local[local<32].flatten().long(),minlength=32)
            print('REUSE_COUNTS',m,pop,counts.cpu().tolist())
        best=dict(state['choices'].get(m,state['default']))
        refs=[kernel_fn(d,state).clone() for d in datasets]
        besttime=math.prod(_cold(lambda d=d:kernel_fn(d,state),flush) for d in datasets)**(1/len(datasets))
        for stage in ('13','2'):
            winner=dict(best);win=besttime
            for bk,bm,cap in ((64,64,128),(128,64,224),(64,32,128),(128,32,192)):
                if time.monotonic()-started>220:break
                trial=dict(best);trial['bk'+stage]=bk;trial['bm'+stage]=bm;trial['cap'+stage]=cap;trial['pf'+stage]=False
                state['choices'][m]=trial
                try:
                    errors=[float((kernel_fn(d,state).float()-r.float()).norm()/r.float().norm().clamp(min=1e-20)) for d,r in zip(datasets,refs)]
                    if any(not math.isfinite(e) or e>.004 for e in errors):
                        print('REUSE_REJECT',m,stage,trial,errors);continue
                    ts=[_cold(lambda d=d:kernel_fn(d,state),flush) for d in datasets]
                    score=math.prod(ts)**(1/len(ts))
                    print('REUSE_TRIAL',m,stage,bk,bm,cap,score,errors)
                    if score<win*.985:winner,win=trial,score
                except Exception as e:print('REUSE_SKIP',m,stage,bk,bm,str(e)[-500:])
            best,besttime=winner,win;state['choices'][m]=best
        print('REUSE_SELECT',m,best,besttime)
        state['report']=True;kernel_fn(datasets[0],state);state['report']=False
    print('reuse_prepare_seconds',time.monotonic()-started)
    return state
