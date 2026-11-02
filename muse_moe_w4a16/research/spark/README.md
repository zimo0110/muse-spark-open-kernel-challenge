# Development validation and compiler evidence

This directory contains structured experiment results, exact source snapshots, and offline Hopper compiler reports. The project-level [experiment record](../../EXPERIMENTS.md) summarizes the outcomes and distinguishes H200 measurements from development checks.

## Development environment

- NVIDIA GB10: 48 SMs, compute capability 12.1, 101,376-byte opt-in shared-memory limit per block.
- NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.11-py3`.
- PyTorch `2.10.0a0+b558c986e8.nv25.11`; CUDA and NVCC 13.0.
- Container digest: `sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21`.

GB10 cannot execute Hopper WGMMA directly. Development runners substitute synchronous warp-level MMA, use smaller TMA rings, and omit unused shared-memory allocations where necessary. These changes support mathematical and layout validation. They do not establish Hopper asynchronous correctness or predict H200 timing.

## Evidence map

| Files or directory | Evidence |
|---|---|
| `phase_probe_h200_result.json`, `phase_probe_h200_summary.json` | H200 phase measurements and probe-overhead comparisons |
| `phase_probe_v1/` | Frozen profiler sources, builders, and development validation |
| `fixed_width_v1/` | Fixed-width consumer prototypes and validation |
| `fixed_width_binary_sizes.json` | ELF executable-section sizes, excluding debug-section inflation |
| `fixed_width_compile_result.json`, `fixed_width_register_compile_result.json` | Hopper register and spill diagnostics |
| `compact_private_fixup_v3/` | Compact partial-result buffers with explicit proxy ordering |
| `compact_private_fixup_h200_result.json` | H200 paired timings and queued-replay consistency results |
| `native_warp_v1/`, `native_warp_v2/` | Archived one-warp prototypes; H200 failures are recorded separately |

The selected MoE entry remains `../../submission.py`. Experimental snapshots include unsuccessful and unverified variants; their presence does not imply promotion to the selected entry.

## Reproducing development checks

From `muse_moe_w4a16`, in a CUDA/PyTorch environment with the appropriate GPU:

```bash
# Offline sm_90a compilation; no H200 execution.
python tools/hopper_crosscompile.py ../research/spark/fixed_width_cases.json

# GB10 functional validation using synchronous WGMMA emulation.
PYTHONPATH="$PWD" python tools/spark_validate_fixed_width.py ../research/spark/fixed_width_cases.json
```

The compiler runner writes artifacts under `tools/hopper_build/`. Validation runners write their result files beside the runner. These results are development evidence; the challenge CLI supplies the authoritative H200 correctness and performance checks.
