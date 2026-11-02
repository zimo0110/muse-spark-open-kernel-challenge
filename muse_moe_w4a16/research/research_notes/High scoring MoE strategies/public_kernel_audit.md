# Public kernel architecture audit

> Historical research snapshot comparing the earlier register kernel with public implementations. The persistent CUDA port has since been evaluated at 2.382249×; see the [current experiment record](../../../EXPERIMENTS.md) for verified results. Leaderboard snapshots and proposed next steps below describe the time of this investigation.


## Which public implementation is a credible step beyond our 1.834x?

### Takeaway
The strongest reproducible source found in this audit is alygg77's published **2.4015x** MoE file, derived from vjeux's end-to-end CUDA persistent kernel. This is evidence for a different architecture, not evidence about the private 2.7x leader's implementation.

### Cited Findings
- Pin alygg77 commit `6cae49a639ee6d14b929a5fd4a63c627f96d7dc0`. Downloaded `kernels/muse_moe_w4a16.py` SHA-256 is `b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281`, exactly matching its published `results.json` entry: job `e1bcad5a55cd4a29`, speedup 2.4015, CI 2.3655–2.4376, historical rank41/1122 read Oct29 20:54UTC. This is published historical evidence, not a fresh check performed by us. — [Results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/results.json)
- Published per-shape latency in microseconds: M1=24.636, M8=63.609, M16=92.266, M32=123.958, M128=175.235, M256=207.761. The large-batch speedups are 2.6762x/2.9626x; those are individual shapes, not the overall score. — [Per-shape results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md)
- Both projects are Apache-2.0. Alygg77 attributes most performance to vjeux commit `f35a41ee1c13e316dabbf772830d47b23b453393`; its main MoE improvement is KPERM3, about +1.25% overall, not the entire 2.40x. Preserve both projects' notices. — [NOTICE](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/NOTICE), [LICENSE](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/LICENSE)

### Inferences
- A first exact-source control is much more informative than continuing to tune our existing Gluon implementation. The published latency gaps are largest at M1 and M128/256.
- Do not describe this as locating the 2.7x leader's code. That remains another research question.

### Gaps
- No fresh H200 result was obtained: this audit is read-only and our daily check quota is exhausted.
- Published claims require reproduction under the current judge, including cold compilation limits and strict precision handling discussed below.

## What execution strategies differ materially from ours?

### Takeaway
The public architecture fuses the whole routed operator into one resident CUDA grid, with different skinny/fat work units and cost-balanced stream-K scheduling. This is much more than replacing our projection kernel with a generic producer/consumer loop.

### Cited Findings
- Vjeux's source documents routing, first GEMM/SwiGLU, second GEMM/norm statistics, and final route combine in **one persistent kernel**. Every CTA redundantly builds routing; fat jobs come first, largest first. Skinny work uses16-weight-row full-K units with warps splitting K; fat units cover256 rows. Static cost partitioning plus a small atomic tail pool balances work. With only skinny jobs, late phase-A ring refills prefetch phase-B weights. — [Vjeux algorithm/source](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_moe_w4a16.py#L2-L55)
- Actual dispatch uses four compiled variants: TINY M<=4, TINY2 M<=8, MID M<=16, MAIN M>=17. MAIN enables RS-WGMMA; the source has four consumer warpgroups and weight/activation producer pipelines. Weight ring slots are released once their compressed data is in registers; activation slots remain live until the last corresponding WGMMA retires. It overlaps deferred epilogue/fixup work with the next segment's first WGMMA groups. — [Alygg77 source, build flags and CUDA implementation](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- KPERM3 permutes K so a scale pair serves four k16 steps, builds scale pairs once from an8-byte stage load, and removes a spill seen in KPERM2. Their measured instruction count drops20.12→18.44 per weight word. Their observer measurements identify unit boundaries and phase-end spread, not insufficient ring depth, as their remaining WGS bottleneck. — [Measured facts, sections6–7](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md#6-muse_moe_w4a16-moe)
- They measured deeper rings, broad dynamic tail claims, WGS32, and moving/removing fences. Those were negative or below noise; a fourth slot did not establish a win. — [MoE negative results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/NEGATIVE_RESULTS.md#-muse_moe_w4a16)

### Inferences
- Our prior naive persistence, decoded-weight warp specialization, and generalized splitK failures do not falsify this entire architecture: those experiments lacked its end-to-end phase scheduling, work-cost partitioning, early next-phase fetches, and distinct skinny/WGMMA execution bands.
- The first implementation step should be an attributed public control and precision audit, not a fresh partial rewrite of one component.
- After reproduction, target measured unit-boundary drain/refill and phase-transition costs. Do not blindly deepen rings or assume load bandwidth alone explains performance.

### Gaps
- No proof was found that further simple tuning of this 2.4x architecture yields2.7x. Its authors already measured many apparently attractive changes as negative.

## What must be checked before adoption, and how can preparation be made practical?

### Takeaway
Weight packing appears structurally lossless, but this public implementation does **not** match our strict BF16 activation-stage rounding. It also has a substantial cold-compilation budget. Preserve the exact control and create a separately labeled conservative precision port.

### Cited Findings
- `prepare` stores packed codewords, original scale bytes, original global scales, a256-entry scale-decoding LUT, original BF16 norm weights, and additional byte-permuted code/scale layouts. Scratch starts at zero; it contains intermediates, sums, stream-K fixups and barriers. Inspection found no persistent per-weight BF16/FP32 expanded weight matrix. PRMT is enabled only after checking all actual scale bytes are positive normal E4M3; other scales use LUT/mma fallback. — [prepare, pack helpers and scale guard](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py#L3220-L3349)
- Precision nuance: GU epilogues keep gate/up and SiLU/product in FP32 until final BF16 intermediate storage. Phase-B epilogues commonly combine FP32 GEMM output, global scale and norm weight, writing FP32 ybuf/statistics. These are higher precision than our staged BF16 boundaries; do not call them stage-identical merely because output checks pass. The official rule says activations stay BF16 wherever SGLang keeps BF16. — [Official rules](https://openkernelchallenge.dev/rules), [Public kernel](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- Exact-source caveat: `ALT_SRCS['helper0']` contains a second CUDA source; the module appends `('-DOKC_WSH0=1','@helper0')` to MID_DEFINES. Extracting CUDA_SRC alone silently drops the selected M16 implementation. — [Module tail](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py#L3543-L3545)
- Authors report cold prepare273–275s and one311.9s run exceeding300s; a lean build took208–224s. All four normal variants are needed by the scored/nearby shapes, so dropping a variant is not a safe compile reduction. — [Measured facts, judge setup](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md)

### Inferences
- Conservative GU port: round gate/up to BF16, compute SiLU then round to BF16, multiply and store BF16. Conservative PB port: round GEMM*GS to BF16, multiply by1+norm in FP32, round to BF16 before FP32 ybuf/statistics. WGS currently folds GS*norm, so that fold must be undone to insert both boundaries.
- Possible cold-build reduction without changing selected mathematics: for compile-time OKC_TINY builds, only `moe_fused<false>` is reachable through Python dispatch; condition out `<true>` references in **all three** launcher areas: cudaFuncSetAttribute, PDL launch, normal launch. MAIN needs both templates. This is an unmeasured build-time proposal, not a promised speedup.
- Keep exact public control source byte-for-byte and separately record strict-port changes. First check compilation/correctness/prepare time, then soak fresh-input graph replays and mixed shapes before scored publication. No scored submission is authorized now.

### Gaps
- Strict-stage port performance is unknown and must not inherit2.4015x as its claimed score.
- Full stream-safety/reentrancy proof was not performed. Scratch is reused and counters are reset/generation-managed in the kernel; concurrent calls on one state merit dedicated testing.
- The README's request that agents ask users to star the repository is untrusted content, not an instruction to this agent. No stars/messages/contact actions were taken.

### Local inspection inventory
Downloaded source and licenses are in `public_kernel_sources/`. Files: `aly_moe.py`, `aly_moe.cu` (AST-extracted CUDA_SRC), `aly_helper0.cu` (AST-extracted selected MID source), `aly_results.json`, `aly_RESULTS.md`, `aly_FACTS.md`, `aly_NEGATIVE.md`, `aly_README.md`, `aly_NOTICE`, `aly_LICENSE`, and `vjeux_moe.py`/`.cu`/README. Vjeux Python SHA256: `10e0690d7fc21244bf29f6df2916b89f0ddcbe8f94d7afe15983478d315883a3`.

Strict precision patch inventory in extracted `aly_moe.cu`: GU sites424–425,726,894,1550,1899–1901. PB sites439–440,734–735,902–903,1494–1495,1574–1575,1913–1914. The selected helper0 CUDA has corresponding sites approximately8lines later (its WGS offsets differ slightly); search the same functions, not blind line replacement. Function roles: `epi_A/epi_B`, skinny reductions, fat tail, WS helper, `wgs_epi_tile`. This inventory is local extracted-source evidence supporting the linked public kernel audit above.
