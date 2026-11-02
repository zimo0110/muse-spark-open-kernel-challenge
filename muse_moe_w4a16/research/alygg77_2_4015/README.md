# Attributed persistent CUDA baseline

Upstream: [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels/tree/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0), derived from [vjeux/okc-2026](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393).

The pinned source matches its published **2.4015×** judge record. We have not reproduced that result. The local ports have **no GPU performance result**. Original attribution and Apache-2.0 LICENSE/NOTICE are preserved.

| File in the MoE directory | Purpose |
|---|---|
| `cuda_persistent_control.py` | Exact upstream bytes and SHA-256 |
| `cuda_persistent_lean.py` | Prune unreachable BIG=true instantiations in three small-shape builds |
| `cuda_persistent_bf16.py` | Lean build plus conservative numeric BF16 stage boundaries |

Both embedded CUDA sources, including the selected MID `helper0`, and all four shape variants are retained. The strict variant rounds gate/up, SiLU, intermediate products, projection outputs and norm-weighted values. Existing FP32 `ybuf` scratch holds BF16-representable values; FP32 accumulation and statistics remain FP32.

Rebuild and validate locally:

```sh
python3 muse_moe_w4a16/build_persistent_port.py
python3 muse_moe_w4a16/verify_persistent_port_cpu.py
```

When the actual private GPU-check quota permits, first reproduce build/correctness/timing with the lean control, then check the BF16 port:

```sh
okc check muse_moe_w4a16/cuda_persistent_lean.py --no-wait --json
# Inspect and wait for that job before proceeding.
okc check muse_moe_w4a16/cuda_persistent_bf16.py --no-wait --json
```

On a real Hopper development machine, `verify_persistent_gpu.py` checks fresh-input graph replays, changing shapes, zero/all-local workloads and reused scratch. This runner makes no CLI submissions.

Cold compilation may approach the 300-second preparation limit. The build pruning is structurally checked but its speed benefit is unmeasured. GPU race freedom, exact CUDA compilation compatibility and performance must still be established. Do not publish based on the upstream score alone.

Further optimization should start from measured unit-boundary, fixup and phase-transition costs after reproduction. The authors already report negative results for simply deepening rings. The current leader's implementation was not found publicly, so no claim is made about its technique.
