"""CPU validation for the persistent CUDA port, not CUDA execution.

Checks both embedded sources, reachable-template preprocessing, actual Python
weight packers, and the exact new C++ numeric helpers using a CPU BF16 shim.
"""
import ast
import ctypes
import hashlib
import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from build_persistent_port import CONTROL_HASH, ROOT, STRICT_HELPERS, cuda_literals


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._EXTS, "Import must not compile or launch CUDA"
    return module


def test_sources(control, lean, strict):
    assert hashlib.sha256(control.read_bytes()).hexdigest() == CONTROL_HASH
    original = ast.parse(control.read_text())
    for path in (lean, strict):
        tree = ast.parse(path.read_text())
        # Both the main and actually selected alternate MID source are retained.
        sources = cuda_literals(tree)
        assert set(sources) == {"CUDA_SRC", "helper0"}
        for label, node in sources.items():
            assert node.value.count("#if !OKC_TINY") >= 1
            if path == strict:
                assert node.value.count("codex_gu_bf16(") == 6
                assert node.value.count("codex_pb_bf16(") == 13
                assert "SILU_DIV(gt, 1.f + __expf(-gt)) * up" not in node.value
        # Host routing, scratch reset logic and every weight-packing function
        # remain unchanged. _ext changes only its build-name string.
        baseline = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name != "_ext"]
        current = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "_ext"]
        assert [ast.dump(n) for n in baseline] == [ast.dump(n) for n in current]
    print("PASS control hash, both CUDA strings, all epilogue sites, unchanged host/packing/scratch functions")


def test_reachable_launchers(path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("C++ preprocessor required for reachable-template validation")
    for label, node in cuda_literals(ast.parse(path.read_text())).items():
        launcher = node.value[node.value.index("void okc_run("):]
        for tiny in (0, 1):
            result = subprocess.run(
                [compiler, "-E", "-P", "-x", "c++", f"-DOKC_TINY={tiny}", "-"],
                input=launcher, text=True, capture_output=True, check=True)
            assert result.stdout.count("moe_fused<true>") == (0 if tiny else 3)
            assert result.stdout.count("moe_fused<false>") == 3
    print("PASS actual C++ preprocessing: zero unreachable BIG references in TINY; MAIN retains all three")


def test_shape_dispatch(module):
    for defines, maximum in ((module.TINY_DEFINES, module.TINY_MAX_M),
                             (module.TINY2_DEFINES, module.TINY2_MAX_M),
                             (module.MID_DEFINES, module.MID_MAX_M)):
        assert "-DOKC_TINY=1" in defines and maximum < 64
        assert not any(d.startswith("-DOKC_BIGM=") for d in defines)
    assert "@helper0" in module.MID_DEFINES
    assert "-DOKC_TINY=1" not in module.MAIN_DEFINES
    assert module._stored_pos() == module._POS
    assert sorted(module._POS.values()) == list(range(32))
    assert sorted(module._kperm_src().tolist()) == list(range(256))
    normal = torch.arange(8, 127, dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert module._scales_prmt_exact(normal)
    for byte in (0, 1, 7, 127, 128, 255):
        assert not module._scales_prmt_exact(torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn))
    print("PASS four shape variants, selected helper0, scale guard fallback and bit/K bijections")


def test_packers(module):
    gen = torch.Generator().manual_seed(4421)
    for rows_count in (16, 256):
        raw = torch.randint(0, 256, (rows_count, 2048), generator=gen, dtype=torch.uint8)
        sfbytes = torch.randint(0, 256, (rows_count, 256), generator=gen, dtype=torch.uint8)
        sf = sfbytes.view(torch.float8_e4m3fn)
        rows = torch.randperm(rows_count, generator=gen)
        words = module._pack_codes(raw, rows)
        scales = module._pack_scales(sf, rows)
        nb = rows_count // 16
        word = words.to(torch.int64).view(nb, 8, 8, 8, 4, 4)
        values = torch.empty(tuple(word.shape) + (2, 4), dtype=torch.uint8)
        for reg, (h, j0) in enumerate(module._REG_HJ):
            for up in range(2):
                code = torch.zeros_like(word)
                for bit in range(4):
                    code |= ((word >> module._POS[(reg, up, bit)]) & 1) << bit
                values[..., h, j0 + up] = code.to(torch.uint8)
        nib = values.permute(0, 6, 3, 1, 2, 4, 5, 7).reshape(rows_count, 4096)
        reconstructed = nib[:, 0::2] | (nib[:, 1::2] << 4)
        assert torch.equal(reconstructed, raw[rows])
        recovered_sf = scales.view(nb, 8, 8, 4, 8, 2).permute(0, 5, 2, 1, 4, 3).reshape(rows_count, 256)
        assert torch.equal(recovered_sf, sfbytes[rows])
        assert words.numel() * 4 == raw.numel() and scales.numel() == sfbytes.numel()
        unit = module._unit_order(words, scales).view(nb, 36864)
        assert torch.equal(unit[:, :32768].contiguous().view(torch.int32).flatten(), words)
        assert torch.equal(unit[:, 32768:].flatten(), scales)
        if rows_count == 256:
            fat = module._fat_order(words, scales).view(1, 16, 36864)
            fc = fat[:, :, :32768].contiguous().view(torch.int32).view(1, 8, 2, 16, 4, 32, 4)
            assert torch.equal(fc.permute(0, 3, 1, 2, 4, 5, 6).contiguous().flatten(), words)
            fs = fat[:, :, 32768:].contiguous().view(1, 8, 2, 16, 32, 4, 2)
            assert torch.equal(fs.permute(0, 3, 1, 4, 2, 5, 6).contiguous().flatten(), scales)
            wgs = module._wgs_copy(raw, sf, rows).view(1, 16, 4, 9216)
            wc = wgs[:, :, :, :8192].contiguous().view(torch.int32).view(1, 16, 4, 4, 4, 32, 4)
            wn = wc.permute(0, 2, 4, 1, 3, 6, 5).contiguous().view(16, 8, 8, 4, 8, 4)
            wn = wn.permute(0, 1, 2, 4, 5, 3).contiguous().flatten()
            expected = module._pack_codes_nat(module._kperm_codes(raw), rows)
            assert torch.equal(wn, expected)
            ws = wgs[:, :, :, 8192:].contiguous().view(1, 16, 4, 4, 8, 4, 4, 2)
            ws = ws.permute(0, 2, 3, 7, 4, 1, 5, 6).contiguous().view(256, 256)
            assert torch.equal(ws, sfbytes[rows])
    print("PASS actual CPU packers: every code/scale bit, random row orders, unit/fat/WGMMA layouts")


def test_numeric_helpers():
    compiler = shutil.which("clang++") or shutil.which("g++")
    shim = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __expf expf
#define SILU_DIV(a,b) ((a)/(b))
struct __nv_bfloat16 { uint16_t bits; };
inline __nv_bfloat16 __float2bfloat16(float value) {
  uint32_t bits; std::memcpy(&bits,&value,4);
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return {static_cast<uint16_t>(bits >> 16)};
}
inline float __bfloat162float(__nv_bfloat16 value) {
  uint32_t bits=static_cast<uint32_t>(value.bits)<<16; float out;
  std::memcpy(&out,&bits,4);return out;
}
"""
    wrapper = r"""
extern "C" void run(int n,const float*a,const float*b,const float*c,float*r,float*g,float*p) {
  for(int i=0;i<n;++i) {
    r[i]=codex_bf16_round(a[i]);
    g[i]=codex_gu_bf16(a[i],b[i]);
    p[i]=codex_pb_bf16(a[i],b[i],c[i]);
  }
}
"""
    rng = np.random.default_rng(11)
    a = rng.normal(0, 5, 20000).astype(np.float32)
    a[:8] = [0., -0., 1e-40, -1e-40, 1., -1., 65504., -65504.]
    b = rng.normal(0, .2, len(a)).astype(np.float32)
    c = rng.normal(1, .1, len(a)).astype(np.float32)
    outputs = [np.empty_like(a) for _ in range(3)]
    with tempfile.TemporaryDirectory(prefix="moe-bf16-helper-") as tmp:
        src = Path(tmp)/"helpers.cpp";lib = Path(tmp)/"helpers.so"
        src.write_text(shim + STRICT_HELPERS + wrapper)
        compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-shared", "-fPIC", str(src), "-o", str(lib)], text=True, capture_output=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        fn = ctypes.CDLL(str(lib)).run
        pointer = ctypes.POINTER(ctypes.c_float)
        fn.argtypes = [ctypes.c_int] + [pointer]*6
        fn(len(a), *(x.ctypes.data_as(pointer) for x in [a,b,c,*outputs]))
    def bf(x):
        return x.to(torch.bfloat16).float()
    ta,tb,tc = map(torch.from_numpy,(a,b,c))
    np.testing.assert_array_equal(outputs[0].view(np.uint32),bf(ta).numpy().view(np.uint32))
    np.testing.assert_array_equal(outputs[2],bf(bf(ta*tb)*tc).numpy())
    reference_gu = bf(bf(bf(ta)/(1+torch.exp(-bf(ta))))*bf(tb)).numpy()
    actual_gu = bf(torch.from_numpy(outputs[1])).numpy()
    # CPU libm and torch exp can differ at BF16 ties; this checks stage algebra,
    # not equivalence of CPU exp to CUDA __expf or GPU race freedom.
    error = np.linalg.norm(actual_gu-reference_gu)/max(np.linalg.norm(reference_gu),1e-20)
    assert np.isfinite(error) and error < 1e-4
    print("PASS actual C++ BF16 helpers against torch: rounding/PB exact, staged GU within CPU-exp tolerance")


if __name__ == "__main__":
    torch.set_num_threads(1)
    control = ROOT/"cuda_persistent_control.py"
    lean = ROOT/"cuda_persistent_lean.py"
    strict = ROOT/"cuda_persistent_bf16.py"
    test_sources(control,lean,strict)
    test_reachable_launchers(lean)
    test_reachable_launchers(strict)
    module = load(strict)
    test_shape_dispatch(module)
    test_packers(module)
    test_numeric_helpers()
    print("CPU checks passed. CUDA compilation, replay/scratch synchronization and performance remain UNVERIFIED.")
