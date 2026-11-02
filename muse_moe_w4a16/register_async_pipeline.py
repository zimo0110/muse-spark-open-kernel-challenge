# okc: problem=muse_moe_w4a16
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
import triton as triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy as cp

SCALE_BIAS = gl.constexpr(2.0 ** 119)
ACC_BIAS = gl.constexpr(2.0 ** 7)
_DECODE_ASM16 = gl.constexpr("""
{
.reg .b32 t, u;
and.b32 t, $8, 0x81C081C0;
mul.rn.bf16x2 $0, t, $24;
and.b32 t, $10, 0x81C081C0;
mul.rn.bf16x2 $1, t, $25;
shf.l.wrap.b32 t, $8, $8, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $2, t, $24;
shf.l.wrap.b32 t, $10, $10, 4;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $3, t, $25;
shf.l.wrap.b32 t, $8, $8, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $4, t, $24;
shf.l.wrap.b32 t, $10, $10, 10;
and.b32 t, t, 0x81C081C0;
mul.rn.bf16x2 $5, t, $25;
shf.l.wrap.b32 t, $8, $8, 6;
shf.l.wrap.b32 u, $8, $8, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $6, t, $24;
shf.l.wrap.b32 t, $10, $10, 6;
shf.l.wrap.b32 u, $10, $10, 14;
and.b32 t, t, 0x80C080C0;
and.b32 u, u, 0x01000100;
or.b32 t, t, u;
mul.rn.bf16x2 $7, t, $25;
}
""")


# Layouts (CTA of 4 warps, BN = 64 weight rows, BM = 16 token rows, BK = 256).
MMA_L = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8]))
A_L = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=MMA_L.value, k_width=2))
B_L = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=MMA_L.value, k_width=2))
# packed words [BN rows, 4 (q), 8 (j)]: j in registers (contiguous 32 bytes), q in lanes
L3 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 8]))
# Decode input/output [BN rows, 4 (q), 8 (word), 2 (b0), 2 (b2), 2 (b3)]: the 16 elements of one asm call are, in
# register order, (b0, row + 8, b2, b3): the MMA A quads of two k16 steps for rows r and r + 8.
LW16 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1, 0, 0], [8, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
               [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 4, 0, 0, 0]],
    lane_bases=[[0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
    warp_bases=[[16, 0, 0, 0, 0, 0], [32, 0, 0, 0, 0, 0]], block_bases=[], shape=[64, 4, 8, 2, 2, 2]))
# block-scale bytes [BN, 4 (q), 4 (g)]
LS = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[16, 0, 0], [32, 0, 0]], block_bases=[], shape=[64, 4, 4]))
# activations [BM token rows, 4 (q), 64 (8j + i)]
LX = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[16, 4, 64]))
# The same activation tile viewed as [64 rows (4n + q), 64]: per-warp read layout from shared memory ...
LX2 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[64, 64]))
LX32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [8, 0, 0], [16, 0, 0]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[32, 4, 64]))
LX2_32 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [32, 0], [64, 0]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[128, 64]))
LX8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32]],
    lane_bases=[[0, 1, 0], [0, 2, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]],
    warp_bases=[[0, 0, 0], [0, 0, 0]], block_bases=[], shape=[8, 4, 64]))
LX2_8 = gl.constexpr(gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
    lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
    warp_bases=[[0, 0], [0, 0]], block_bases=[], shape=[32, 64]))
# ... cooperative global load layout (each thread 16 contiguous bytes) ...
LXG = gl.constexpr(gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[4, 8], warps_per_cta=[4, 1], order=[1, 0]))
# ... and the shared buffer, 16-byte chunks XOR-swizzled by row so a quarter warp's rows hit distinct banks.
X_SMEM = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))

@gluon.jit
def _idx3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, L: gl.constexpr):
    a = gl.arange(0, D0, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    b = gl.arange(0, D1, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    c = gl.arange(0, D2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    return (gl.expand_dims(gl.expand_dims(a, 1), 2), gl.expand_dims(gl.expand_dims(b, 0), 2),
            gl.expand_dims(gl.expand_dims(c, 0), 1))


@gluon.jit
def _x8(v):
    v2 = gl.join(v, v)
    v4 = gl.join(v2, v2)
    return gl.join(v4, v4)


@gluon.jit
def _decode_tile(w3, sb, BN: gl.constexpr, BK: gl.constexpr, L3: gl.constexpr, A_L: gl.constexpr):
    """Packed words [BN, 4, 8] + scale bytes [BN, 4, 4] -> decoded A operand [BN, BK] (logical K
    k' = 32 w + 16 b3 + 8 b2 + 2 q + b0 <-> physical K 64 q + 8 w + 4 b3 + 2 b2 + b0)."""
    s = (sb.to(gl.float8e4nv, bitcast=True).to(gl.float32) * SCALE_BIAS).to(gl.bfloat16)
    s8 = gl.reshape(gl.join(s, s), [BN, 4, 8])
    w16 = gl.convert_layout(_x8(w3), LW16)
    s16 = gl.convert_layout(_x8(s8), LW16)
    t = gl.inline_asm_elementwise(_DECODE_ASM16, "=r,=r,=r,=r,=r,=r,=r,=r," + ",".join(["r"] * 24), [w16, s16],
                                  dtype=gl.bfloat16, is_pure=True, pack=16)
    t = gl.reshape(gl.permute(t, (0, 2, 5, 4, 1, 3)), [BN, BK])
    return gl.convert_layout(t, A_L, assert_trivial=True)


@gluon.jit
def _xg_offsets(sorted_ptr, r0, r1, K: gl.constexpr, BY_TOKEN: gl.constexpr, BM: gl.constexpr):
    """Offsets/mask [4 BM, 64] in LXG for the cooperative load: row 4n + q, column c -> x[row_n, 64q + c]."""
    rr = gl.arange(0, 4 * BM, layout=gl.SliceLayout(1, LXG))
    cc = gl.arange(0, 64, layout=gl.SliceLayout(0, LXG))
    rv = r0 + rr // 4
    pair = gl.load(sorted_ptr + rv, mask=rv < r1, other=0)
    row = pair // 8 if BY_TOKEN else pair
    off = gl.expand_dims(row * K + (rr % 4) * 64, 1) + gl.expand_dims(cc, 0)
    msk = gl.expand_dims(rv < r1, 1) & gl.expand_dims(cc >= 0, 0)
    return off, msk


@gluon.jit
def _x_shared(x_ptr, off, msk, k0, x_s, BM: gl.constexpr, BK: gl.constexpr):
    xg = gl.load(x_ptr + off + k0, mask=msk, other=0.0)
    gl.barrier()
    x_s.store(xg)
    gl.barrier()
    if BM == 8:
        xv = gl.reshape(x_s.load(LX2_8), [BM, 4, 64])
    elif BM == 32:
        xv = gl.reshape(x_s.load(LX2_32), [BM, 4, 64])
    else:
        xv = gl.reshape(x_s.load(LX2), [BM, 4, 64])
    return _x_tile(xv, BM, BK, B_L)


@gluon.jit
def _x_tile(xv, BM: gl.constexpr, BK: gl.constexpr, B_L: gl.constexpr):
    # [n, q, 64 physical = 8 w + 4 b3 + 2 b2 + b0] -> [k' = 32 w + 16 b3 + 8 b2 + 2 q + b0, n]
    xb = gl.reshape(gl.permute(gl.reshape(xv, [BM, 4, 8, 2, 2, 2]), (2, 3, 4, 1, 5, 0)), [BK, BM])
    return gl.convert_layout(xb, B_L, assert_trivial=BM != 32)


@gluon.jit
def _gu_split(t, BN: gl.constexpr, BM: gl.constexpr):
    """[BN, BM] with rows 16 s + 8 g + i -> (g = 0 part, g = 1 part), each [BN / 16, 8, BM] (row 8 apart: one thread)."""
    return gl.split(gl.permute(gl.reshape(t, [BN // 16, 2, 8, BM]), (0, 2, 3, 1)))



@triton.jit
def _pack_codes(W, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr,
                B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 256)
    nt = tile // (K // 256)
    j = tl.arange(0, B)
    row = nt * 64 + j // 32
    word = kb * 32 + j % 32
    if GU:
        source_row = ((row % 16) // 8) * (N // 2) + (row // 64) * 32 + ((row % 64) // 16) * 8 + row % 8
    else:
        source_row = row
    q = tl.load(W + e.to(tl.int64) * N * (K // 8) + source_row * (K // 8) + word).to(tl.uint32)
    packed = tl.full((B,), 0, tl.uint32)
    # Fully unrolled bit permutation; each source bit has one destination.
    packed |= ((q >> 0) & 1) << 6
    packed |= ((q >> 1) & 1) << 7
    packed |= ((q >> 2) & 1) << 8
    packed |= ((q >> 3) & 1) << 15
    packed |= ((q >> 4) & 1) << 22
    packed |= ((q >> 5) & 1) << 23
    packed |= ((q >> 6) & 1) << 24
    packed |= ((q >> 7) & 1) << 31
    packed |= ((q >> 8) & 1) << 2
    packed |= ((q >> 9) & 1) << 3
    packed |= ((q >> 10) & 1) << 4
    packed |= ((q >> 11) & 1) << 11
    packed |= ((q >> 12) & 1) << 18
    packed |= ((q >> 13) & 1) << 19
    packed |= ((q >> 14) & 1) << 20
    packed |= ((q >> 15) & 1) << 27
    packed |= ((q >> 16) & 1) << 28
    packed |= ((q >> 17) & 1) << 29
    packed |= ((q >> 18) & 1) << 30
    packed |= ((q >> 19) & 1) << 5
    packed |= ((q >> 20) & 1) << 12
    packed |= ((q >> 21) & 1) << 13
    packed |= ((q >> 22) & 1) << 14
    packed |= ((q >> 23) & 1) << 21
    packed |= ((q >> 24) & 1) << 0
    packed |= ((q >> 25) & 1) << 1
    packed |= ((q >> 26) & 1) << 26
    packed |= ((q >> 27) & 1) << 9
    packed |= ((q >> 28) & 1) << 16
    packed |= ((q >> 29) & 1) << 17
    packed |= ((q >> 30) & 1) << 10
    packed |= ((q >> 31) & 1) << 25
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 256) + tile) * 2048 + j, packed)


@triton.jit
def _pack_scales(S, Out, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr,
                 B: tl.constexpr):
    tile = tl.program_id(0)
    e = tl.program_id(1)
    kb = tile % (K // 256)
    nt = tile // (K // 256)
    j = tl.arange(0, B)
    row = nt * 64 + j // 16
    group = kb * 16 + j % 16
    if GU:
        source_row = ((row % 16) // 8) * (N // 2) + (row // 64) * 32 + ((row % 64) // 16) * 8 + row % 8
    else:
        source_row = row
    value = tl.load(S + e.to(tl.int64) * N * (K // 16) + source_row * (K // 16) + group)
    tl.store(Out + (e.to(tl.int64) * (N // 64) * (K // 256) + tile) * 1024 + j, value)


@triton.jit
def _pack_weights(W, S, Out, Scales, N: tl.constexpr, K: tl.constexpr, GU: tl.constexpr):
    # One launch per bank, each code and scale read/written once.
    _pack_codes(W, Out, N, K, GU, 2048)
    _pack_scales(S, Scales, N, K, GU, 1024)


@triton.jit
def _compact_routes(Ids, Map, Count, P: tl.constexpr, START: tl.constexpr,
                    E: tl.constexpr, R: tl.constexpr):
    e = tl.program_id(0)
    r = tl.arange(0, R)
    ids = tl.load(Ids + r, r < P, other=-1)
    live = (ids == START + e) & (r < P)
    rank = tl.cumsum(live.to(tl.int32)) - 1
    tl.store(Map + e * P + rank, r, live)
    tl.store(Count + e, tl.sum(live.to(tl.int32)))


@triton.jit
def _work_plan(Count, Work, NWork, E: tl.constexpr, BM: tl.constexpr,
               WMAX: tl.constexpr, BE: tl.constexpr):
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
    # Initialize even inactive descriptors so speculative compiler loads are safe.
    live = w < tl.sum(blocks)
    tl.store(Work + 3 * w, tl.where(live, owner, 0))
    tl.store(Work + 3 * w + 1, tl.where(live, offset, 0))
    tl.store(Work + 3 * w + 2, tl.where(live, tl.minimum(offset + BM, cnt), 0))
    tl.store(NWork, tl.sum(blocks))


@gluon.jit
def _register_item(X, W, S, GS, Map, Out, e, r0, r1, pn,
                   K: gl.constexpr, N: gl.constexpr, P: gl.constexpr,
                   BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 256
    rw, qw, jw = _idx3(BN, 4, 8, L3)
    rs, qs, gs = _idx3(BN, 4, 4, LS)
    blocks: gl.constexpr = K // BK
    wo = (e * (N // BN) + pn) * blocks * 2048 + rw * 32 + qw * 8 + jw
    so = (e * (N // BN) + pn) * blocks * 1024 + rs * 16 + qs * 4 + gs
    xoff, xmask = _xg_offsets(Map + e * P, r0, r1, K, GU, BM)
    ring:gl.constexpr=gl.SwizzledSharedLayout(vec=8,per_phase=1,max_phase=8,order=[2,1,0])
    xs = gl.allocate_shared_memory(gl.bfloat16, [2,4 * BM,64], ring)
    acc = gl.zeros([BN, BM], gl.float32, layout=MMA_L)
    cp.async_load(xs.index(0),X+xoff,mask=xmask)
    cp.commit_group()
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
        cp.wait_group(0)
        gl.barrier()
        current=xs.index(k%2)
        if BM==8:
            xv=gl.reshape(current.load(LX2_8),[BM,4,64])
        elif BM==32:
            xv=gl.reshape(current.load(LX2_32),[BM,4,64])
        else:
            xv=gl.reshape(current.load(LX2),[BM,4,64])
        xb=_x_tile(xv,BM,BK,B_L)
        cp.async_load(xs.index((k+1)%2),X+xoff+(k+1)*BK,mask=xmask&(k+1<blocks))
        cp.commit_group()
        a = _decode_tile(w, s, BN, BK, L3, A_L)
        acc = mma_v2(a, xb, acc)
    cp.wait_group(0)
    gl.barrier()
    y = acc * (gl.load(GS + e) * ACC_BIAS)
    n = gl.arange(0, BN, layout=gl.SliceLayout(1, MMA_L))
    m = gl.arange(0, BM, layout=gl.SliceLayout(0, MMA_L))
    row = r0 + m
    pair = gl.load(Map + e * P + row, row < r1, other=0)
    if GU:
        # Paired rows: [gate 0..7, up 0..7, gate 8..15, up 8..15, ...].
        rows2 = gl.expand_dims(n, 1) + gl.expand_dims(m * 0, 0)
        pair2 = gl.expand_dims(pair, 0) + gl.expand_dims(n * 0, 1)
        mask2 = gl.expand_dims(row < r1, 0) & gl.expand_dims(n >= 0, 1)
        gate, up = _gu_split(y, BN, BM)
        nr, _ = _gu_split(rows2, BN, BM)
        pr, _ = _gu_split(pair2, BN, BM)
        mr, _ = _gu_split(mask2.to(gl.int32), BN, BM)
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        silu = (gate / (1 + gl.exp(-gate))).to(gl.bfloat16).to(gl.float32)
        value = (silu * up).to(gl.bfloat16)
        col = pn * 32 + nr // 16 * 8 + nr % 8
        gl.store(Out + pr * (N // 2) + col, value, mask=mr != 0)
    else:
        gl.store(Out + gl.expand_dims(pair * N, 0) + gl.expand_dims(pn * BN + n, 1),
                 y.to(gl.bfloat16), mask=gl.expand_dims(row < r1, 0))


@gluon.jit
def _register_work(X, W, S, GS, Map, Work, Out, pn, wi,
                  K: gl.constexpr, N: gl.constexpr, P: gl.constexpr,
                  MAX_BM: gl.constexpr, GU: gl.constexpr, PREFETCH: gl.constexpr):
    e = gl.load(Work + 3 * wi)
    r0 = gl.load(Work + 3 * wi + 1)
    r1 = gl.load(Work + 3 * wi + 2)
    if r1 - r0 <= 8:
        _register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 8, GU, PREFETCH)
    elif MAX_BM >= 16:
        if r1 - r0 <= 16:
            _register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 16, GU, PREFETCH)
        elif MAX_BM == 32:
            _register_item(X, W, S, GS, Map, Out, e, r0, r1, pn, K, N, P, 32, GU, PREFETCH)


@gluon.jit
def _register_dispatch(X, W, S, GS, Map, Work, NWork, Out,
                       K: gl.constexpr, N: gl.constexpr, P: gl.constexpr,
                       MAX_BM: gl.constexpr, GU: gl.constexpr,
                       PREFETCH: gl.constexpr, PERSISTENT: gl.constexpr):
    if PERSISTENT:
        # Visit live work only; avoid thousands of conservatively sized inactive CTAs.
        item = gl.program_id(0)
        total = gl.load(NWork) * (N // 64)
        while item < total:
            _register_work(X, W, S, GS, Map, Work, Out, item % (N // 64), item // (N // 64),
                           K, N, P, MAX_BM, GU, PREFETCH)
            item += gl.num_programs(0)
    else:
        pn = gl.program_id(0)
        wi = gl.program_id(1)
        if wi < gl.load(NWork):
            _register_work(X, W, S, GS, Map, Work, Out, pn, wi,
                           K, N, P, MAX_BM, GU, PREFETCH)


@triton.jit
def _finish_register(Y, Ids, Gates, NW, Out, H: tl.constexpr,
                     START: tl.constexpr, E: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BH)
    s = tl.arange(0, 8)
    ids = tl.load(Ids + t * 8 + s)
    valid = (ids >= START) & (ids < START + E)
    nw = 1 + tl.load(NW + h, h < H, other=0).to(tl.float32)
    y = tl.load(Y + (t * 8 + s[:, None]) * H + h[None, :],
                valid[:, None] & (h[None, :] < H), other=0).to(tl.float32)
    y = (y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(y * y, 1) / H + 1e-8)
    gates = tl.load(Gates + t * 8 + s)
    z = tl.sum(y * (inv * gates)[:, None], 0)
    tl.store(Out + t * H + h, z.to(tl.bfloat16), h < H)


def _pack_bank(raw, sf, gate_up):
    e, n, half_k = raw.shape
    k = half_k * 2
    if n % 64 or k % 256:
        raise ValueError('direct-register layout requires rows divisible by64 and K divisible by256')
    codes = torch.empty((e, n // 64, k // 256, 64, 32), device=raw.device, dtype=torch.int32)
    scales = torch.empty((e, n // 64, k // 256, 64, 16), device=raw.device, dtype=torch.uint8)
    grid = ((n // 64) * (k // 256), e)
    _pack_weights[grid](raw.contiguous().view(torch.int32), sf.contiguous().view(torch.uint8), codes, scales, n, k, gate_up)
    return codes, scales


def prepare(weights):
    # Full precision representations stay unchanged: 4-bit codes + raw E4M3 scale bytes.
    count = 32
    w13, s13 = _pack_bank(weights['w13'][:count], weights['w13_sf'][:count], True)
    w2, s2 = _pack_bank(weights['w2'][:count], weights['w2_sf'][:count], False)
    return {'w13': w13, 's13': s13, 'gs13': weights['w13_gs'][:count].float().contiguous(),
            'w2': w2, 's2': s2, 'gs2': weights['w2_gs'][:count].float().contiguous(),
            'norm': weights['post_norm_w'], 'local': (0, count),
            'prefetch_small': True, 'prefetch_large': False,
            'max_bm13': 32, 'max_bm2': 16, 'register_cap': 128,
            'persistent': False, 'grid_blocks': 3 * torch.cuda.get_device_properties(weights['w13'].device).multi_processor_count}


def kernel_fn(data, state):
    cfg = data['config']
    start, count = cfg['local_start'], cfg['local_count']
    if (start, count) != state['local']:
        raise ValueError('weights prepared for experts0..31 only')
    x = data['hidden_states'].contiguous()
    m, h = x.shape
    i = cfg['d_expert']
    p = m * 8
    # Shape choices, never host branches on input values.
    bm13 = 8 if m <= 32 else state['max_bm13']
    bm2 = 8 if m <= 32 else state['max_bm2']
    pf = state['prefetch_small'] if m <= 32 else state['prefetch_large']
    route_map = torch.empty((count, p), device=x.device, dtype=torch.int32)
    counts = torch.empty((count,), device=x.device, dtype=torch.int32)
    _compact_routes[(count,)](data['topk_ids'], route_map, counts, p, start, count, triton.next_power_of_2(p))
    plans = {}
    for bm in dict.fromkeys((bm13, bm2)):
        wmax = triton.next_power_of_2(triton.cdiv(p, bm) + count)
        work = torch.empty((wmax, 3), device=x.device, dtype=torch.int32)
        nwork = torch.empty((1,), device=x.device, dtype=torch.int32)
        _work_plan[(1,)](counts, work, nwork, count, bm, wmax, triton.next_power_of_2(count), num_warps=4)
        plans[bm] = (wmax, work, nwork)
    inter = torch.empty((p, i), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((p, h), device=x.device, dtype=torch.bfloat16)
    wmax, work, nwork = plans[bm13]
    grid13 = (min(state['grid_blocks'], (2 * i // 64) * wmax),) if state['persistent'] else (2 * i // 64, wmax)
    _register_dispatch[grid13](x, state['w13'], state['s13'], state['gs13'],
        route_map, work, nwork, inter, h, 2 * i, p, bm13, True, pf, state['persistent'],
        num_warps=4, maxnreg=state['register_cap'])
    wmax, work, nwork = plans[bm2]
    grid2 = (min(state['grid_blocks'], (h // 64) * wmax),) if state['persistent'] else (h // 64, wmax)
    _register_dispatch[grid2](inter, state['w2'], state['s2'], state['gs2'],
        route_map, work, nwork, y, i, h, p, bm2, False, pf, state['persistent'],
        num_warps=4, maxnreg=state['register_cap'])
    out = torch.empty_like(x)
    _finish_register[(m,)](y, data['topk_ids'], data['topk_weights'], state['norm'], out,
                          h, start, count, triton.next_power_of_2(h), num_warps=8)
    return out
