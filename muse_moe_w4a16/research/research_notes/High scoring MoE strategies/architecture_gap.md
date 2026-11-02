# Architecture gap in Hopper W4A16 MoE

> Historical research snapshot comparing the earlier register kernel with public implementations. The persistent CUDA port has since been evaluated at 2.382249×; see the [current experiment record](../../../EXPERIMENTS.md) for verified results. Leaderboard snapshots and proposed next steps below describe the time of this investigation.


## What is materially different from our 1.834x kernel?

### Takeaway
The actionable public control is alygg77's documented **2.4015x**, not an evidenced reproduction of the private 2.7x leader. Its main difference is a complete execution decomposition: **full-K skinny units split across warps, register-resident activation reuse, persistent cross-projection scheduling, and a separate cost-balanced RS-WGMMA path**. Grafting TMA or warp specialization onto our existing 64-row/256-K tile never tested this architecture.

### Cited Findings
- Author-published job `e1bcad5a55cd4a29` reports 2.4015x, CI 2.3655–2.4376. Published times for M1/8/16/32/128/256 are 24.64/63.61/92.27/123.96/175.24/207.76 us. These are **their records, not our reproduction**. [Pinned results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md)
- Our corresponding scored times are 34.48/83.59/109.85/141.60/231.00/321.71 us, and the two projection dispatches occupy about 93–97% of GPU kernel time at M8–256. Merely eliminating our routing/finish kernels cannot explain the public gap. [Our scored profile](../../../profiles/register_ready.9cf7ef93107b4052.txt)
- **Our decomposition:** 64 output rows, 256-K steps, four warps split across output rows; all warps traverse full K. `_b256_x_shared` gathers activations, writes shared memory, and performs two CTA barriers each K step. At K4096 that is 32 such barriers per CTA's output tile. [Our source, `_b256_register_item` and `_b256_x_shared`](../../../submission.py)
- **Public skinny decomposition:** one unit is 16 output rows × full K4096, with ≤8 tokens in the active skinny builds. Eight warps cooperate on the same unit; each handles a disjoint 512-K slice. The raw codes occupy 32 KiB and original FP8 scales 4 KiB; the entire **36 KiB full-K unit** is contiguous and bulk-copied into a multi-slot shared ring. Functions: `skinny_ring`, `ring_issue`, `load_x_regs`; constants `NCH=64/NW`, `UNIT_CODES=32768`, `UNIT_SC=4096`. [Pinned source](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- Each skinny warp retains its job's activation slice in `uint4 x0[2*NCH]`, loaded only when `j != curj`. At NW8, this is 16 uint4s = 64 scalar 32-bit registers/thread before other registers. Crucially those activations are reused across output-row units, rather than re-gathered and staged for every 256-K step of every output tile. Four independent FP32 MMA accumulator streams expose instruction-level parallelism. Cross-warp partials are combined in shared memory at the unit end, rather than writing every split-K partial to global memory and launching a separate reducer. [Pinned source, `skinny_ring`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- One persistent grid, normally one CTA per SM, redundantly computes routing in each CTA; phases A/B, normalization statistics, and final combination execute within that grid. With no fat jobs, the last phase-A refills prefetch this CTA's phase-B units. [Pinned source, module algorithm documentation, `route`, `run_phase`, `moe_fused`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- **Large-M MAIN is not a generic extra producer warp:** NW16 means 512 compute threads/four consumer warpgroups, plus two producer warps (weights and activations), 576 threads total. Consumer warpgroups decode A weights into registers and issue RS-WGMMA against shared B activations. Prepared weights stay packed, and the shared weight ring also stays compressed; decoded BF16 weights are not passed through shared memory as in our first slow WS candidate. [Pinned source, `NTL`, `wgs_*`, `_wgs_copy`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- MAIN partitions **cost-weighted K-stage ranges**, with cost tables dependent on the job's rounded token count, rather than uniformly splitting every output tile. Only boundary units require Stream-K fixups. Its W2 producer can consume a 256-wide intermediate stage once the eight relevant phase-A producers have published readiness. [Pinned source, `wgs_prefix`, `wgs_stage_at`, `wgs_range`, `wgs_flush`, `doneAs`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)

### Inferences
- The most important replacement is **our unit of work**, not the async-copy primitive. Four skinny units cover the same 64 output rows as our tile but expose K-parallelism inside each CTA and substantially fewer repeated activation staging collectives. This is not equivalent to our unsuccessful global split-K experiments or fixed-grid persistence.
- Public 2.4015x versus our 1.8337x suggests about 24% lower normalized latency; getting from 2.4015x to 2.7x then requires about another 11% lower latency. Those are useful targets, not controlled same-seed measurements or guaranteed achievable gains.

### Gaps
- We have not reproduced the public code on the current judge. No code provenance for the actual 2.7x leader is established by this architectural analysis; the leader-source researcher covers that separately.
- We cannot attribute all public-versus-ours time differences to one feature because the implementations differ in geometry, precision rounding, scheduling and fusion simultaneously.

## What do production kernels and measured negatives say about the next optimization?

### Takeaway
The public architecture already implements techniques that our isolated trials did not combine. Begin from that disclosed higher-performing control. Do not rebuild another producer/consumer pipeline around our current tile, and do not assume a new TMA buffer means useful overlap.

### Cited Findings
- Marlin's original design explicitly combines offline operand layouts, activation reuse in registers, asynchronous weight loading, double-buffered shared reads, carefully interleaved dequantization and MMA, multiple warps computing partials of the same output tile, and striped scheduling. Its README also says the original kernel is not optimized for Hopper. These are mechanisms, not a challenge score claim. [Marlin primary README](https://github.com/IST-DASLab/marlin#techniques)
- Stream-K partitions aggregate inner-loop work among physical processing elements to reduce wave quantization. It is not synonymous with adding a fixed split factor to all tiles. [Original Stream-K paper](https://arxiv.org/abs/2301.03598)
- GemLite reports different algorithm families for GEMV, skinny Split-K GEMM and larger GEMM, and specifically reports packing/unpacking bottlenecks on A100/H100. Its generic INT4 results do not establish NVFP4 challenge performance. [PyTorch/Mobius/SGLang primary article](https://pytorch.org/blog/accelerating-llm-inference/)
- Alygg77 reports its MoE streaming phase at 3.87–4.05 TB/s against a **size-matched** HBM ceiling of roughly 4.35–4.45 TB/s. It identifies substantial unit-end release/epilogue/fixup costs and says extra weight-ring slots beyond three do not reliably help. [Measured facts, section 6](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md)
- Its KPERM3 scale-sharing change removes repeated scale construction: reported decode instruction count drops from 20.12 to 18.44 per weight word. KPERM2 spilled 32 bytes; KPERM3 keeps an eight-byte scale load per stage and avoids the spill. Reported same-cluster board gain is about 1.25%, with effects on M32/128/256. **This is already in the public base, not a fresh optimization for us to claim.** [Measured facts, section 7](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md)
- The author measured integer ALU plus HGMMA on the same SM and found their combined cost close to the sum rather than ideal overlap. Their public negative results caution against adding decompression merely because tensor-core operations are asynchronous. [Measured facts, section 1](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md)

### Inferences
- **First implementation priority:** import and preserve a pinned, attributed public CUDA control with its actual skinny/full-K and MAIN/WGS geometry; reproduce it before changing its tuning. The sibling audit establishes its Apache-2.0 lineage and exact variant/source choice.
- **First further-performance target after reproduction:** the public MAIN phase-B tail and unit-end communication, not another bulk-loader. Its measured read phase is already near the size-specific bandwidth ceiling; facts report 2.7–8.2 us end spread and roughly 0.9 us per release. A targeted change must reduce those without adding a more expensive global reduction or GPU-scope fence.
- Retain the exact KPERM3 weight/scale/activation permutation as a unit. Taking its scale trick alone while retaining our old layout would not replicate its once-per-four-step scale reuse or register allocation.

### Gaps
- No untested feature is credibly guaranteed to yield the remaining ~11% latency reduction. Unit-end optimization needs real counters/stamps and fresh judge checks; target-Hopper measurements were unavailable during this investigation.

## What must be audited before the public control becomes our candidate?

### Takeaway
Precision and compile-budget behavior must be explicit. Preserve an exact attributed control, then make a separately named strict-BF16 version and safe build pruning; do not silently claim these are already GPU-verified.

### Cited Findings
- Public `epi_A`, skinny epilogues, and `wgs_epi` form FP32 gate/up and compute FP32 SiLU×up, rounding only the final intermediate store to BF16 (`probe_q` currently returns its argument unchanged). Our current implementation explicitly rounds gate/up and SiLU as well. Public `epi_B` and WGS phase-B epilogues form FP32 norm-weighted values in `ybuf`, without our intermediate BF16 Y and norm-product rounding. This is **higher internal precision**, not evidence of prohibited lower precision, but it is not exact staged-BF16 reproduction. [Pinned source, `probe_q`, `epi_A`, `epi_B`, `wgs_epi`, skinny epilogues](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- Task wording says activations stay BF16 wherever SGLang keeps BF16. Our task also sets a 300-second prepare budget. [Challenge specification](https://openkernelchallenge.dev/problems/muse_moe_w4a16)
- Public measurements report cold prepare 273–275 seconds in most containers but 311.9 seconds once, and a leaner equivalent build at 208–224 seconds. [Measured facts, section 3](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md)
- The normal positive-scale run compiles four actual define tuples, MAIN/TINY/TINY2/MID. The source supports PRMT and LUT branches but does not eagerly compile all eight combinations for one weight set. `_main_defines` selects fallback if actual scale bytes fail the PRMT exactness condition. [Pinned source, `prepare`, `_scales_prmt_exact`, `_main_defines`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)
- TINY/TINY2/MID have `OKC_TINY=1` and host ranges M≤4/8/16. The current C++ launcher nevertheless sets attributes on both `moe_fused<false>` and `<true>` and retains a runtime M≥64 branch, forcing otherwise-unreachable BIG=true instantiations. MAIN legitimately needs both M32 and M128/256 branches. [Pinned source, define tuples, `kernel_fn`, `okc_run`](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/kernels/muse_moe_w4a16.py)

### Inferences
- **Safe build candidate:** under `#if OKC_TINY`, set attributes and launch only `moe_fused<false>`; retain both for MAIN. This removes three unused BIG=true kernel instantiations without changing reachable device paths. It is a compile-time hypothesis, not a measured reduction yet.
- Removing unused Python diagnostic strings alone is unlikely to materially improve NVCC time. Avoid claiming an eight-to-four variant reduction when actual `prepare` already selects only four.
- For a strict port, add explicit BF16-round-trip helpers at every gate/up/SiLU boundary and before/after norm weighting, in every relevant epilogue (skinny, fat, WGS and any alternate helper source), while retaining FP32 accumulation. Keep the published control untouched for comparison. The performance and correctness of this stricter version require GPU validation; source passing output tolerance alone does not resolve the wording of the activation rule.
- Because grid synchronization relies on all persistent CTAs progressing, preserve its grid/residency invariants. Copying the fused grid barrier into our oversized grid would risk deadlock, not just performance loss.

### Gaps
- Exact effect of strict BF16-rounding insertion and build pruning cannot be measured on this machine. Neither should be presented as checked or faster before the next real judge run.
