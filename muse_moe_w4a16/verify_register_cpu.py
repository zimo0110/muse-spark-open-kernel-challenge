"""CPU semantic checks for the actual packing/routing kernel bodies.

Uses a small NumPy interpreter for the Triton intrinsics exercised here.
This does NOT compile or execute Gluon/MMA and does NOT establish GPU correctness.
Run: python3 muse_moe_w4a16/verify_register_cpu.py
"""
import ast
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Ptr:
    def __init__(self, array, index=0):
        self.array = array.reshape(-1)
        self.index = index

    def __add__(self, index):
        return Ptr(self.array, self.index + index)

    __radd__ = __add__


class CPU:
    constexpr = object
    int32, int64, uint32 = np.int32, np.int64, np.uint32
    pid = (0, 0)

    def program_id(self, axis):
        return tensor(self.pid[axis], np.int32)

    def arange(self, lo, hi):
        return tensor(np.arange(lo, hi), np.int32)

    def full(self, shape, value, dtype):
        return tensor(np.full(shape, value, dtype=dtype))

    def load(self, ptr, mask=True, other=0):
        index = np.asarray(ptr.index)
        mask = np.broadcast_to(np.asarray(mask), index.shape)
        out = np.full(index.shape, other, dtype=ptr.array.dtype)
        out[mask] = ptr.array[index[mask]]
        return tensor(out)

    def store(self, ptr, value, mask=True):
        index, value, mask = np.broadcast_arrays(ptr.index, value, mask)
        index, mask = np.asarray(index), np.asarray(mask, dtype=bool)
        ptr.array[index[mask]] = value[mask]

    def cumsum(self, x, axis=0):
        return tensor(np.cumsum(x, axis=axis, dtype=np.int32))

    def sum(self, x, axis=None):
        return tensor(np.sum(x, axis=axis))

    def cdiv(self, x, y):
        return (x + y - 1) // y

    def where(self, condition, a, b):
        return tensor(np.where(condition, a, b))

    def minimum(self, a, b):
        return tensor(np.minimum(a, b))


def load_cpu_kernels():
    names = {'_pack_codes', '_pack_scales', '_pack_weights', '_compact_routes', '_work_plan'}
    tree = ast.parse((ROOT / 'register_candidate.py').read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
            body.append(node)
    cpu = CPU()
    scope = {'tl': cpu}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<cpu-kernels>', 'exec'), scope)
    return cpu, scope


def rol(x, shift):
    return ((x << np.uint32(shift)) | (x >> np.uint32(32 - shift))).astype(np.uint32)


def decode_words(q):
    # Independently execute the four PTX bit expressions, then reinterpret BF16.
    q = np.asarray(q, np.uint32)
    parts = [q & 0x81C081C0, rol(q, 4) & 0x81C081C0,
             rol(q, 10) & 0x81C081C0,
             (rol(q, 6) & 0x80C080C0) | (rol(q, 14) & 0x01000100)]
    a = np.stack(parts, axis=-1)
    bits = np.stack([a & 65535, a >> 16], axis=-1).reshape(q.shape + (8,))
    return (bits.astype(np.uint32) << 16).view(np.float32)


def scale_values(raw):
    raw = np.asarray(raw, np.uint8)
    exponent, mantissa = (raw >> 3) & 15, raw & 7
    val = np.where(exponent == 0, mantissa.astype(np.float32) * 2 ** -9,
                   (1 + mantissa.astype(np.float32) / 8) * np.exp2(exponent.astype(np.float32) - 7))
    return np.where(raw & 128, -val, val).astype(np.float32)


def raw_weights(raw, scales):
    codes = np.stack([raw & 15, raw >> 4], axis=-1).reshape(raw.shape[:-1] + (raw.shape[-1] * 2,))
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    return lut[codes] * np.repeat(scale_values(scales), 16, axis=-1)


def pack(cpu, funcs, raw, scales, gu):
    e, n, halfk = raw.shape
    k = halfk * 2
    codes = np.empty((e, n // 64, k // 256, 64, 32), np.uint32)
    sf = np.empty((e, n // 64, k // 256, 64, 16), np.uint8)
    for ei in range(e):
        for tile in range(n // 64 * (k // 256)):
            cpu.pid = (tile, ei)
            funcs['_pack_weights'](Ptr(raw.view(np.uint32)), Ptr(scales), Ptr(codes), Ptr(sf), n, k, gu)
    return codes, sf


def packed_weights(codes, sf):
    e, nt, kb, _, _ = codes.shape
    small = decode_words(codes)
    # Each word holds8 values; consecutive pairs of words share a group16 scale.
    sb = np.repeat(scale_values(sf), 2, axis=-1)[..., None]
    decoded = small * (sb * np.float32(2 ** 119)) * np.float32(128)
    tiles = decoded.reshape(e, nt, kb, 64, 256)
    return tiles.transpose(0, 1, 3, 2, 4).reshape(e, nt * 64, kb * 256)


def test_packing(cpu, funcs):
    rng = np.random.default_rng(20261029)
    for n, k in [(64, 256), (128, 512), (256, 768)]:
        raw = rng.integers(0, 256, (2, n, k // 2), dtype=np.uint8)
        sf = rng.integers(0, 127, (2, n, k // 16), dtype=np.uint8)
        reference = raw_weights(raw, sf)
        for gu in (False, True):
            codes, packed_sf = pack(cpu, funcs, raw, sf, gu)
            decoded = packed_weights(codes, packed_sf)
            rows = np.arange(n)
            if gu:
                rows = (rows % 16 // 8) * (n // 2) + rows // 64 * 32 + rows % 64 // 16 * 8 + rows % 8
            np.testing.assert_array_equal(np.sort(rows), np.arange(n))
            np.testing.assert_array_equal(decoded, reference[:, rows])
            # Recover every original nibble, including the sign of zero.
            tiny = decode_words(codes)
            bits = tiny.view(np.uint32) >> 16
            nibble = ((bits >> 6) & 7) | ((bits >> 12) & 8)
            recovered = np.sum(nibble.astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32)), axis=-1, dtype=np.uint32)
            recovered = recovered.transpose(0, 1, 3, 2, 4).reshape(raw.shape[0], n, k // 8)
            np.testing.assert_array_equal(recovered, raw[:, rows].copy().view(np.uint32))
            scale_roundtrip = packed_sf.transpose(0, 1, 3, 2, 4).reshape(raw.shape[0], n, k // 16)
            np.testing.assert_array_equal(scale_roundtrip, sf[:, rows])
            # Stored formats contain exactly the original number of code/scale bits.
            assert codes.nbytes == raw.nbytes and packed_sf.nbytes == sf.nbytes
    # Every input bit, each code at each word position, and all positive finite scales.
    words = np.concatenate([np.array([1 << b for b in range(32)], np.uint32),
                            np.array([c << (4 * p) for p in range(8) for c in range(16)], np.uint32)])
    raw = np.resize(words.view(np.uint8), (1, 64, 128)).copy()
    sf = np.resize(np.arange(127, dtype=np.uint8), (1, 64, 16)).copy()
    codes, out_sf = pack(cpu, funcs, raw, sf, False)
    np.testing.assert_array_equal(packed_weights(codes, out_sf), raw_weights(raw, sf))
    # Exhaust all16 FP4 codes crossed with all127 positive finite E4M3 scales.
    groups = np.arange(128 * 16)
    group_codes = (groups // 127 % 16).astype(np.uint8)
    raw = np.repeat(group_codes * 17, 8).reshape(1, 128, 128)
    sf = (groups % 127).astype(np.uint8).reshape(1, 128, 16)
    codes, out_sf = pack(cpu, funcs, raw, sf, False)
    np.testing.assert_array_equal(packed_weights(codes, out_sf), raw_weights(raw, sf))
    print('PASS: actual packing bodies, exhaustive code/scale cases, random multi-tile banks, no precision expansion')


def test_k_permutation():
    logical = np.arange(256)
    physical = (logical // 2 % 4) * 64 + (logical // 32) * 8 + (logical // 16 % 2) * 4 + (logical // 8 % 2) * 2 + logical % 2
    np.testing.assert_array_equal(np.sort(physical), np.arange(256))
    rng = np.random.default_rng(9)
    a = rng.integers(-4, 5, (64, 256)).astype(np.float32)
    x = rng.integers(-4, 5, (16, 256)).astype(np.float32)
    np.testing.assert_array_equal(a @ x.T, a[:, physical] @ x[:, physical].T)
    print('PASS: logical/physical K bijection and operand permutation preserves dot product')


def test_routes(cpu, funcs):
    rng = np.random.default_rng(2)
    for m in (1, 4, 8, 16, 32, 33, 128, 256, 257):
        p, e, start = m * 8, 32, 32
        for mode in ('random', 'hot', 'empty', 'all_local'):
            ids = rng.integers(-1, 256, p, dtype=np.int32)
            if mode == 'hot': ids[::2] = start + 3
            if mode == 'empty': ids[:] = 255
            if mode == 'all_local': ids %= 32; ids += start
            mapping = np.full((e, p), -999, np.int32)
            counts = np.zeros(e, np.int32)
            for ei in range(e):
                cpu.pid = (ei,)
                funcs['_compact_routes'](Ptr(ids), Ptr(mapping), Ptr(counts), p, start, e, 1 << (p - 1).bit_length())
            for ei in range(e):
                expected = np.flatnonzero(ids == start + ei)
                assert counts[ei] == len(expected)
                np.testing.assert_array_equal(mapping[ei, :counts[ei]], expected)
            for bm in (8, 16, 32):
                wmax = 1 << (((p + bm - 1) // bm + e - 1).bit_length())
                work = np.full((wmax, 3), -999, np.int32)
                nwork = np.zeros(1, np.int32)
                funcs['_work_plan'](Ptr(counts), Ptr(work), Ptr(nwork), e, bm, wmax, 32)
                visited = []
                for ei, lo, hi in work[:nwork[0]]:
                    assert 0 <= lo < hi <= counts[ei] and hi - lo <= bm
                    visited.extend(mapping[ei, lo:hi].tolist())
                np.testing.assert_array_equal(np.sort(visited), np.flatnonzero((ids >= start) & (ids < start + e)))
                assert np.all(work[nwork[0]:] == 0)
    print('PASS: actual compaction/planning bodies, nearby shapes, skew, empty/local routing, exact route coverage')


def test_layouts_and_persistent_schedule():
    tree = ast.parse((ROOT / 'register_candidate.py').read_text())
    found = 0
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(isinstance(t, ast.Name) and t.id in {'L3', 'LS', 'LW16'} for t in node.targets): continue
        call = node.value.args[0]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        bases = kwargs['reg_bases'] + kwargs['lane_bases'] + kwargs['warp_bases']
        coords = []
        for value in range(1 << len(bases)):
            coord = np.zeros(len(kwargs['shape']), dtype=np.int32)
            for bit, basis in enumerate(bases):
                if value >> bit & 1: coord ^= basis
            coords.append(tuple(coord))
        assert len(set(coords)) == int(np.prod(kwargs['shape']))
        assert all(all(0 <= c < size for c, size in zip(coord, kwargs['shape'])) for coord in coords)
        found += 1
    assert found == 3
    for ntiles in (4, 64, 128):
        for nwork in (0, 1, 7, 33, 128):
            for grid in (1, 17, 396):
                visits = [(i % ntiles, i // ntiles) for block in range(grid) for i in range(block, ntiles * nwork, grid)]
                assert len(visits) == len(set(visits)) == ntiles * nwork
                assert set(visits) == {(n, w) for n in range(ntiles) for w in range(nwork)}
    print('PASS: register-layout coverage and persistent scheduling cover every live tile exactly once')


def bf16(x):
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def activation(g, u):
    g, u = bf16(g), bf16(u)
    return bf16(bf16(g / (1 + np.exp(-g))) * u)


def norm(y, nw):
    y = bf16(bf16(y) * (1 + nw))
    return y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + 1e-8)


def test_staged_model(cpu, funcs):
    rng = np.random.default_rng(71)
    e, h, i, m = 3, 256, 256, 5
    raw13 = rng.integers(0, 256, (e, 2 * i, h // 2), dtype=np.uint8)
    raw2 = rng.integers(0, 256, (e, h, i // 2), dtype=np.uint8)
    sf13 = rng.integers(24, 49, (e, 2 * i, h // 16), dtype=np.uint8)
    sf2 = rng.integers(24, 49, (e, h, i // 16), dtype=np.uint8)
    w13, w2 = raw_weights(raw13, sf13), raw_weights(raw2, sf2)
    c13, s13 = pack(cpu, funcs, raw13, sf13, True)
    c2, s2 = pack(cpu, funcs, raw2, sf2, False)
    d13, d2 = packed_weights(c13, s13), packed_weights(c2, s2)
    x = bf16(rng.standard_normal((m, h)).astype(np.float32) * .05)
    nw = bf16(rng.standard_normal(h).astype(np.float32) * .1)
    ids = rng.integers(-1, e + 2, (m, 8))
    gates = rng.random((m, 8)).astype(np.float32)
    gates /= gates.sum(axis=1, keepdims=True)
    expected = np.zeros((m, h), np.float32)
    actual = np.zeros_like(expected)
    for t in range(m):
        for slot in range(8):
            expert = ids[t, slot]
            if not 0 <= expert < e: continue
            gu = w13[expert] @ x[t] / 16
            mid = activation(gu[:i], gu[i:])
            y = bf16(w2[expert] @ mid / 16)
            expected[t] += gates[t, slot] * norm(y, nw)
            # Paired64-row tiles contain gate8/up8 groups, then32 intermediate outputs.
            paired = (d13[expert] @ x[t] / 16).reshape(-1, 4, 2, 8)
            mid2 = activation(paired[:, :, 0], paired[:, :, 1]).reshape(i)
            y2 = bf16(d2[expert] @ mid2 / 16)
            actual[t] += gates[t, slot] * norm(y2, nw)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    print('PASS: staged BF16 model, gate/up-paired fusion, global scales, norm and weighted routes')


if __name__ == '__main__':
    cpu, funcs = load_cpu_kernels()
    test_packing(cpu, funcs)
    test_k_permutation()
    test_routes(cpu, funcs)
    test_staged_model(cpu, funcs)
    test_layouts_and_persistent_schedule()
    print('CPU semantic checks passed. GPU compilation, fragment mapping, replay correctness and timing remain UNVERIFIED.')
