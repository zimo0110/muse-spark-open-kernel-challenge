"""Reproducible ports of the pinned Apache-2.0 public CUDA MoE implementation.

The control remains byte-for-byte upstream. Local variants preserve both CUDA
source strings, all shape variants, weight packing, routing, and scratch rules.
No performance claim transfers from upstream to these unverified variants.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL_HASH = "b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281"

STRICT_HELPERS = r"""
// Local conservative numeric BF16 boundaries; FP32 accumulators stay FP32.
__device__ __forceinline__ float codex_bf16_round(float value) {
  return __bfloat162float(__float2bfloat16(value));
}
__device__ __forceinline__ float codex_gu_bf16(float gate, float up) {
  const float g = codex_bf16_round(gate), u = codex_bf16_round(up);
  const float activation = codex_bf16_round(SILU_DIV(g, 1.f + __expf(-g)));
  return activation * u;  // caller performs the final BF16 product store
}
__device__ __forceinline__ float codex_pb_bf16(float acc, float gs, float norm) {
  const float projection = codex_bf16_round(acc * gs);
  return codex_bf16_round(projection * norm);
}

"""


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new, 1)


def prune_unreachable_big(source):
    """TINY/TINY2/MID never receive M>=64; MAIN retains both templates."""
    old = """    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;"""
    new = """    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0;
#if !OKC_TINY
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e1;
#endif"""
    source = replace_once(source, old, new)
    old = """    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);"""
    new = """#if OKC_TINY
    cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
#else
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
#endif"""
    source = replace_once(source, old, new)
    old = """  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);"""
    new = """#if OKC_TINY
  okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
#else
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
#endif"""
    return replace_once(source, old, new)


def staged_bf16(source):
    """Patch every GU and PB epilogue, including the selected helper0 source."""
    marker = "__device__ __forceinline__ float probe_q(float v) {"
    source = replace_once(source, marker, STRICT_HELPERS + marker)
    gu = "SILU_DIV(gt, 1.f + __expf(-gt)) * up"
    assert source.count(gu) == 5
    source = source.replace(gu, "codex_gu_bf16(gt, up)")

    replacements = [
        ("r[h] * gsc * w0", "codex_pb_bf16(r[h], gsc, w0)", 1),
        ("r[2 + h] * gsc * w1", "codex_pb_bf16(r[2 + h], gsc, w1)", 1),
        ("s0 * gsc * nwc0", "codex_pb_bf16(s0, gsc, nwc0)", 2),
        ("s1 * gsc * nwc1", "codex_pb_bf16(s1, gsc, nwc1)", 2),
        ("(h ? sm.y : sm.x) * gsc * nw0", "codex_pb_bf16((h ? sm.y : sm.x), gsc, nw0)", 2),
        ("(h ? sm.w : sm.z) * gsc * nw1", "codex_pb_bf16((h ? sm.w : sm.z), gsc, nw1)", 2),
    ]
    for old, new, count in replacements:
        assert source.count(old) == count, (old, source.count(old), count)
        source = source.replace(old, new)

    # WGMMA folds GS into norm, which must be separated to round both boundaries.
    old = "const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;"
    new = "const float w0 = 1.f + __bfloat162float(__ldg(p.normw + n0)), w1 = 1.f + __bfloat162float(__ldg(p.normw + n0 + 8));"
    source = replace_once(source, old, new)
    source = replace_once(source, "d4[h] * w0", "codex_pb_bf16(d4[h], gsc, w0)")
    source = replace_once(source, "d4[2 + h] * w1", "codex_pb_bf16(d4[2 + h], gsc, w1)")
    assert source.count("codex_gu_bf16(") == 6  # five calls plus definition
    assert source.count("codex_pb_bf16(") == 13  # twelve values plus definition
    return source


def cuda_literals(module):
    result = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CUDA_SRC":
                result["CUDA_SRC"] = node.value
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                if target.value.id == "ALT_SRCS" and isinstance(target.slice, ast.Constant) and target.slice.value == "helper0":
                    result["helper0"] = node.value
    assert set(result) == {"CUDA_SRC", "helper0"}
    return result


def emit_variant(original, precision, filename):
    tree = ast.parse(original)
    starts = [0]
    for line in original.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    edits = []
    for label, node in cuda_literals(tree).items():
        cuda = prune_unreachable_big(node.value)
        if precision:
            cuda = staged_bf16(cuda)
        assert "'''" not in cuda
        literal = "r'''" + cuda + "'''"
        begin = starts[node.lineno - 1] + node.col_offset
        end = starts[node.end_lineno - 1] + node.end_col_offset
        assert ast.literal_eval(original[begin:end]) == node.value
        edits.append((begin, end, literal))
    result = original
    for begin, end, value in sorted(edits, reverse=True):
        result = result[:begin] + value + result[end:]
    tag = "bf16stage" if precision else "lean"
    result = replace_once(result, "okc_moe_w4a16_v216x_", f"okc_moe_w4a16_{tag}_")
    lines = result.splitlines(keepends=True)
    lines.insert(1, f"# Local Apache-2.0 port: {tag}; GPU-UNVERIFIED; public score does not transfer.\n")
    result = "".join(lines)
    compile(result, filename, "exec")
    (ROOT / filename).write_text(result)
    return {"file": filename, "sha256": hashlib.sha256(result.encode()).hexdigest(),
            "status": "Python syntax verified; CUDA compilation/correctness/performance UNVERIFIED"}


def main():
    original = (ROOT / "cuda_persistent_control.py").read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == CONTROL_HASH
    variants = [emit_variant(original, False, "cuda_persistent_lean.py"),
                emit_variant(original, True, "cuda_persistent_bf16.py")]
    provenance = {
        "repository": "https://github.com/alygg77/okc-muse-spark-kernels",
        "commit": "6cae49a639ee6d14b929a5fd4a63c627f96d7dc0",
        "upstream_path": "kernels/muse_moe_w4a16.py",
        "upstream_sha256": CONTROL_HASH,
        "upstream_recorded_job": "e1bcad5a55cd4a29",
        "upstream_recorded_speedup": 2.4015,
        "not_ours": "Public historical result; not reproduced by us, not a score for local ports.",
        "lineage": "vjeux/okc-2026 f35a41ee1c13e316dabbf772830d47b23b453393; alygg77 KPERM3",
        "license": "Apache-2.0; LICENSE/NOTICE retained beside this provenance and original file attribution retained",
        "changes": ["Drop unreachable BIG=true template references in all three TINY launch sites",
                    "BF16 variant: all five GU and six PB epilogue sites patched in BOTH CUDA sources",
                    "Distinct extension names avoid control/port build-cache collisions"],
        "unchanged": ["All four shape variants and selected helper0", "Lossless packed weights and scale guards",
                      "Routing, persistent scheduling, streamK fixups and scratch synchronization"],
        "activation_storage_note": "BF16 port rounds stage values; the existing ybuf scratch container remains FP32 holding exactly BF16-representable values.",
        "variants": variants,
    }
    dest = ROOT / "research/alygg77_2_4015/PROVENANCE.json"
    dest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(variants, indent=2))


if __name__ == "__main__":
    main()
