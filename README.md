# Muse Spark Open Kernel Challenge

CUDA kernels, performance experiments, and reproducible measurements for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/).

The main focus is `muse_moe_w4a16`: routed NVFP4 mixture-of-experts inference on NVIDIA H200. The repository also includes earlier BF16 projection experiments.

## Verified results

Snapshot: **November 2, 2026**. Speedups are over the challenge's SGLang baseline.

| Problem | Result | Evidence | Status |
|---|---|---|---|
| Routed experts (`muse_moe_w4a16`) | **2.382249×**, 95% interval **[2.346516, 2.417983]** | Scored job `c913470f59834b10` | Published best; all five correctness tests passed |
| BF16 projections (`muse_bf16_proj`) | **1.198615×**, interval **[1.180636, 1.216594]** | Recorded check `adc980b817cd4e9c` | Historical private-check result |

A later diagnostic run measured 2.355024× using the unchanged MoE timed kernel. Its purpose was phase profiling; the result is a repeat measurement rather than evidence of a new speedup.

The selected MoE file is [muse_moe_w4a16/submission.py](muse_moe_w4a16/submission.py), byte-identical to [cuda_persistent_aligned_safe.py](muse_moe_w4a16/cuda_persistent_aligned_safe.py).

```text
SHA256 3eba8d457b50ff39e98491a5c325f1183f944cd20edd79213f45077cb0f1bbcb
```

## What the MoE kernel computes

For each token, only its routes to this GPU's 32 local experts are evaluated:

```text
gate, up = hidden_state @ dequantize(w13[expert]).T
intermediate = silu(gate) * up
projection = intermediate @ dequantize(w2[expert]).T
output += route_weight * rmsnorm(projection * (1 + post_norm_weight))
```

The model has 256 experts, top-8 routing, and width 4096. The scored batches contain 1, 8, 16, 32, 128, or 256 tokens. Nearby shapes and skewed routing also matter for correctness and review.

Weights remain losslessly packed NVFP4 codes with their FP8 block scales and FP32 global scales. Dequantization happens inside the GPU kernels. The implementation preserves the required BF16 activation boundaries and accumulates reductions in FP32.

## How the selected implementation works

1. **Untimed preparation.** Repack weights into the layouts used by the GPU kernels, allocate reusable scratch space, and compile the supported variants. Repacking permutes codes and scales; it does not requantize weights or cache converted dense weights.
2. **Fresh GPU routing.** Build expert/token lists from the current call's routing fields. No input-derived outputs are cached across calls.
3. **Persistent fused execution.** A resident grid performs routing, the first GEMM and SwiGLU, the second GEMM, and the final normalization/weighted sum within one kernel launch.
4. **Small batches.** Skinny warp-MMA variants and a helper warp serve batches through 16 tokens. The published dispatch uses separate bands through 4, 8, and 16 tokens.
5. **Larger batches.** Hopper register/shared-memory WGMMA consumers use TMA rings for packed weights and activations. Cost-weighted Stream-K partitions work across SMs. Split units exchange FP32 partial results through a synchronized workspace.
6. **Intermediate layout.** The first GEMM writes the SwiGLU result directly into the layout consumed by the second GEMM, avoiding a separate transpose stage.

The selected CUDA primitives and packing build on the credited Apache-2.0 work of **vjeux** and **alygg77**. Project changes include explicit BF16 boundary handling, empty-routing guards, context handling, experimental scheduling/layout variants, paired comparisons, replay stress tests, and the phase profiler described below. Attribution is retained in the source and research notices.

## What the experiments showed

Each result below distinguishes the experimental path from any fallback selected by its preparation-time comparison.

| Experiment | Measured outcome | Decision |
|---|---|---|
| Two consumer warpgroups | Large-batch paired control/trial ratio **0.9191**; about 8.8% more trial latency | Rejected |
| Extend MID through 24 tokens | About **3.3–8.6%** faster at nearby batches 17 and 24 across three seeds | Useful nearby-shape result; published best remains unchanged |
| Retain weights for adjacent same-expert jobs | Paired ratio **1.00235** | No material gain |
| Wider split-unit readback | Paired ratio **0.98922** | Rejected |
| Independent one-warp tiles | Slower on H200 and failed repeated-call consistency despite passing Spark tests | Parked |
| Separate FP32 partial-result banks | Paired ratio **0.99859** | Performance neutral |
| Compact partial banks plus explicit proxy fence | Paired ratio **0.99876**; 10,000 queued replays with zero drift at all 500 checked outputs | Retained as research; no promotion |
| Fixed-width WGMMA consumer | N32 passes 50 reference cases, 400 mixed replays, and 6,000 integration replays on Spark; Hopper build has no spills | Paired H200 check prepared; performance unverified |

The published path also passed an additional **24,000 queued graph replays**, with maximum observed relative-L2 drift of approximately **6.58e-5**. It is not claimed to be globally bitwise deterministic.

Detailed methods, outcomes, and links to the supporting measurements are in [EXPERIMENTS.md](muse_moe_w4a16/EXPERIMENTS.md).

## H200 profiling findings

[cuda_persistent_phase_probe.py](muse_moe_w4a16/cuda_persistent_phase_probe.py) adds preparation-only measurements on generated inputs. Its timed kernel and dispatch remain unchanged. Scored diagnostic job **`b6387d8f931647f0`** passed; all eight seed/shape records were returned.

- Probe/control overhead ranged from approximately **−0.4% to +1.4%**.
- GEMM1 occupied roughly **60% of per-CTA elapsed time** in these samples.
- Final reduction durations were approximately **1–9 microseconds**, depending on batch size.
- Weight-copy producers experienced sampled waits for consumers to release ring slots. Second-GEMM activation-readiness waits reached approximately **6–10 microseconds at the 95th percentile** across CTAs.

These observations direct the next investigation toward consumer execution and work handoffs. They do **not** identify a particular instruction stall or establish that memory bandwidth is the sole bottleneck. Phase windows overlap across CTAs; their reported durations are not additive whole-kernel costs. Logical bytes issued are not hardware DRAM traffic measurements. Empty-ring waits sample one in 16 positions.

Read the [structured H200 summary](muse_moe_w4a16/research/spark/phase_probe_h200_summary.json) and [full result](muse_moe_w4a16/research/spark/phase_probe_h200_result.json). The profiler passed 76 independent-reference cases and 608 graph replays on Spark before its H200 run, plus a complete Python-runner integration test.

### Current experiment: fixed-width consumer

[cuda_persistent_fixed32_paired.py](muse_moe_w4a16/cuda_persistent_fixed32_paired.py) is prepared for a private H200 check. It uses one 32-token-column WGMMA width for batches 25–32, trading work on unused columns for a smaller executable and fewer dispatch variants. Its control is the existing kernel, and the experimental path is selected only after a sufficient paired gain, per-seed checks, and a positive result specifically at the scored batch of 32. Larger scored batches keep the existing implementation. The previously checked MID extension serves nearby batches through 24.

The offline Hopper build uses **96 registers with zero spills**. Executable text for each kernel entry falls from **373,120 to 107,520 bytes**, approximately **71% smaller**. Static WGMMA instruction sites across the two kernel instantiations fall from 1,024 to 128. These are code-size observations, not a measured speedup or a reduction in executed tensor operations.

The implementation initializes shared activation padding on every call, discards invalid token columns, and guards split-result writes so unused columns cannot contaminate a later call. The narrow prototype also handles duplicate expert IDs by splitting its routing jobs at 32 tokens while retaining the control's scratch-buffer ABI.

A wider N64 variant passed functional Spark checks but produced Hopper register spills. Three register-lifetime/grouping alternatives still spilled, so those builds are parked before spending H200 checks. The next eligible candidate is N32. Its validation and frozen sources are in [fixed_width_v1](muse_moe_w4a16/research/spark/fixed_width_v1/). Spark integration timing is not used to predict H200 performance.

## Repository map

| Path | Contents |
|---|---|
| [muse_moe_w4a16/submission.py](muse_moe_w4a16/submission.py) | Selected, verified MoE entry |
| [muse_moe_w4a16/TASK.md](muse_moe_w4a16/TASK.md) | Computation, precision rules, shapes, and judging method |
| [muse_moe_w4a16/tools](muse_moe_w4a16/tools) | Candidate builders, layout checks, compiler diagnostics, and Spark development runners |
| [muse_moe_w4a16/research/spark](muse_moe_w4a16/research/spark) | Structured experiment results and frozen source snapshots |
| [muse_moe_w4a16/profiles](muse_moe_w4a16/profiles) | CLI benchmark reports |
| [muse_bf16_proj](muse_bf16_proj) | BF16 projection kernels and check reports |

Experimental files include failed and unverified variants. The selected `submission.py`, its recorded hash, and its supporting results identify the current verified entry.

## Running checks

Install and sign in with the challenge CLI following the [challenge instructions](https://openkernelchallenge.dev/). Rule acceptance is performed by the account owner. H200 is the target hardware; the CLI supplies the judge environment.

```bash
# Inspect current limits before spending an attempt.
okc problems --json

# Private correctness and timing run.
okc check muse_moe_w4a16/submission.py --no-wait --json
okc status <job-id> --json

# Inspect the live leaderboard.
okc leaderboard muse_moe_w4a16 --json

# Publish a selected entry when intended.
okc submit muse_moe_w4a16/submission.py --no-wait --json
```

The judge alternates CUDA-graph replays against SGLang on fresh inputs, starts each timed replay with cold L2, and reports paired speedups with confidence intervals. Small changes inside the interval may be noise. A paired candidate can select the original kernel as its fallback; its final score then describes that fallback, not the experimental kernel.

Quotas are per problem and reset at **00:00 UTC**. Use the CLI's current limits rather than historical notes in this repository.

### Spark development

DGX Spark/GB10 is used for portable correctness, packing/layout checks, and offline `sm_90a` compilation. It cannot execute Hopper WGMMA directly. Development runners use synchronous warp-MMA emulation and smaller shared-memory configurations where necessary. **Spark timing is not an H200 speed prediction, and Spark correctness does not validate Hopper asynchronous behavior.**

## License and attribution

[Apache-2.0](LICENSE). Third-party snapshots retain their applicable licenses and notices. See the original source headers and [the MoE research attribution](muse_moe_w4a16/research/alygg77_2_4015/NOTICE), including vjeux/alygg77 kernel lineage and the credited context helper. The challenge starter/reference files retain their supplied notices.

Maintained by [Zoir Imomaliev](https://github.com/zimo0110).
