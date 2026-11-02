# Open Kernel Challenge — work summary

**Account:** `zimo0110` · **Problem:** `muse_bf16_proj` (dense BF16 projections + FP32 router) · **Status:** best check **1.199×** over SGLang, nothing submitted to the board yet.

Kernel track closes **Nov 4, 11:59 pm PT**. Limits: 10 scored runs and 20 checks per problem per day (reset 00:00 UTC). Used so far today: 9 checks, 0 scored runs.

## Setup

- Installed `okc` 0.7.2 to `~/.okc/bin/okc`. The installer verifies its SHA256, and it added a PATH line to `~/.zshrc`.
- You ran `okc login` and accepted the rules (version `2026-10-22.1`).
- Legitimacy checked: Paradigm announces the challenge on its own site (paradigm.xyz, "Muse Spark Open Kernel Challenge"), many participants have public GitHub repos, and the site serves the rules and terms PDF. The kernel boards carry no prizes. The $100K is on the serving track (Nov 5–19). Submissions are licensed Apache-2.0.
- Ran `okc init muse_bf16_proj` in `muse_bf16_proj/`.
- DGX Spark: not used, per your call.

## The problem

`out = x @ W.T` for five weights (qkv_gate, o, pooler, unpooler, router) at m = 1, 8, 16, 32, 128 and 256 tokens. The score is the geometric mean over 6 loads of SGLang's time over ours. Each shape runs as a CUDA graph with a cold L2.

Every shape is **memory-bound**: the time is set by how fast the weights can be read from HBM.

## What's in `submission.py`

One file holding several kernels, plus a tuner in `prepare()` that picks the fastest one for each shape:

1. **Router (FP32 weights):** one CUDA kernel computes the allowed BF16 hi+lo split in a single pass over the weights, with both MMAs feeding one FP32 accumulator. **1.5–3.9× faster** than SGLang on every load.
2. **Lossless 12-bit weight compression + TMA streaming (`tproj_kernel`):**
   - BF16 weights carry only about 2.5 bits of information in their 8-bit exponent field. Each weight is stored as its low byte plus a 4-bit code for the high byte (sign + exponent window), so **25% fewer bytes** are read.
   - About 0.01% of weights fall outside the window. They're listed separately and corrected exactly, so the result is **bit-exact** (verified with a CPU emulation of the kernel's decode).
   - Weights are repacked so each CTA's tile is contiguous and streamed with Hopper TMA bulk copies through an mbarrier ring in shared memory. Decoding takes about 1.6 integer ops per weight, straight into the tensor-core fragments.
   - Pooler at m=1: **21.5 → 15.9 µs**. qkv_gate at m=1: **14.1 → 11.8 µs**.
3. **Triton split-K kernel and a plain CUDA kernel:** fallbacks that win on a few mid-size shapes.
4. **cuBLAS (SGLang's own path):** always a candidate. A custom kernel replaces it only when it's at least 2% faster.

The tuner times each candidate with CUDA-graph replays after flushing L2 by reading a buffer, as the judge does. Each candidate must also match the reference on 4 random inputs before it can be picked. It caches no outputs. It only chooses tile sizes.

## Check history

| # | Change | Speedup |
|---|---|---|
| 1 | Triton split-K + hi/lo router | 1.134× |
| 2 | Tuner flushes L2 by reading (judge-like), 2% margin over cuBLAS | 1.177× |
| 3 | + Triton TMA variant | 1.165× (no gain) |
| 4 | + hand-written CUDA MMA kernel | 1.169× (no gain) |
| 5–6 | + 12-bit compression, per-thread loads | 1.169–1.176× (no gain: latency-bound) |
| 7 | Compression **streamed by TMA** | 1.188× (one race found) |
| 8 | Race fixed (`fence.proxy.async`), stricter validation | 1.119× (router tolerance too strict) |
| 9 | Router tolerance fixed | **1.199×** |

## Lessons

- Plain loads, whether from Triton or a hand-written kernel, top out around 3.6 TB/s. Compression only paid off once it was paired with TMA bulk copies.
- Re-arming a TMA buffer that warps just read needs `fence.proxy.async.shared::cta`. Without it, some configurations gave silently wrong results. The multi-input validation catches that kind of bug before a config is used.
- The tuner has to flush L2 by *reading* a buffer, the way the judge does. Flushing by writing gave misleading timings.

## Where it stands, and next steps

- **Board context:** the leader is at 1.64×. 1.20× would be mid-board.
- **Biggest remaining gaps:** o, unpooler, and the m = 16/32/128/256 shapes still run at roughly cuBLAS speed.
- **Next ideas:**
  - Extend the TMA-compressed kernel to more token counts (m = 16/32) and to o and the unpooler.
  - Add a compressed path for m = 128/256.
  - Cut fixed per-kernel latency on the small 8–16 MB layers.
- **Files:**
  - `muse_bf16_proj/submission.py`: the current best (1.199×).
  - Per-run logs and profiles: `muse_bf16_proj/profiles/`.
  - Backup of the best version: `/private/tmp/claude-501/-Users-zoir-dev-muse-challenge/4fa44f68-7a28-44fa-9420-6fc3f55038dd/scratchpad/submission_v9_best.py`.
