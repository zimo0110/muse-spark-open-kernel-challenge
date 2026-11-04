# Routed MoE optimization experiments

This record separates verified improvements, unsuccessful experiments, and hypotheses awaiting H200 measurements. The operator, shapes, and precision requirements are specified in [TASK.md](TASK.md).

## Selected entry and kernel lineage

[submission.py](submission.py) is identical to [cuda_persistent_aligned_safe.py](cuda_persistent_aligned_safe.py). Scored job `c913470f59834b10` passed all five correctness tests at **2.382249325×** over SGLang, with a 95% interval of **[2.346515585, 2.417983065]**.

The implementation builds on the Apache-2.0 vjeux/alygg77 persistent CUDA kernel. Local changes include explicit BF16 activation boundaries, empty-routing handling, scheduling experiments, replay validation, and instrumentation. Source headers and the [upstream notice](research/alygg77_2_4015/NOTICE) identify the kernel lineage.

The earlier register-based implementation reached **1.833725418×** in scored job `9cf7ef93107b4052`. Moving to the persistent Hopper implementation produced the larger improvement; the following experiments investigate its remaining overhead.

## Measurement method

Scored results come from the challenge's H200 evaluator. It alternates CUDA-graph replays against SGLang, starts timed replays with cold L2, and reports paired speedups with confidence intervals.

Experimental comparisons run the trial and selected control on the same GPU, inputs, and routing distribution, alternating their timing order. Shape selection uses several generated seeds. Correctness checks cover ordinary, skewed, all-local, and empty routing; selected experiments also stress duplicate expert IDs and nearby batch sizes.

A preparation-time comparison may reject the trial and retain the control. In that case, the final official score measures the **control**, even if the filename names the experiment. The paired comparisons identify which implementation actually improved.

Spark/GB10 supports portable mathematical checks and offline Hopper compilation. Synchronous warp-MMA emulation does not validate Hopper asynchronous retirement, and its timings are not H200 speed predictions.

## Scheduling and memory experiments

| Experiment | H200 evidence | Interpretation |
|---|---|---|
| Two consumer warpgroups | Job `a8b06a9748b54177`: 2.228408×; paired job `c71dd6b8a13c49f8`: control/trial ratio 0.919132 | Reducing physical consumer groups lost performance. |
| MID through 24 tokens | Job `6baed3dac7b84b8e`: approximately 3.3–8.6% gains at batches 17 and 24 across three seeds | Useful for nearby shapes; the six scored shapes are unchanged. |
| Adjacent-expert cache retention | Job `70c9c2ddb2994e49`: paired ratio 1.002350 | No material gain. |
| Wider partial-result readback | Job `b415be1c26c046ed`: paired ratio 0.989221 | Additional read-level parallelism did not help this implementation. |
| Separate partial-result banks | Job `e706a69b83644876`: paired ratio 0.998594 | Removing FP32 data atomics did not yield a measurable net speedup. |
| Compact partial banks with proxy fence | Job `a60b76ee29af4f54`: paired ratio 0.998756 | Performance neutral; retained as a deterministic-reduction experiment. |

The compact partial-bank implementation uses two buffers per CTA, covering the head and tail of its contiguous Stream-K range. Contributors write separate FP32 partials; the finisher combines them in K order. Extra workspace is about 20 MiB. A 10,000-replay H200 stress test checked outputs every 20 replays: all 500 observed checkpoints matched the synchronized reference bitwise.

The selected implementation separately passed 24,000 queued replays, with maximum observed self-comparison relative-L2 drift of approximately 6.58e-5. Small bitwise differences from FP32 reduction order remain possible.

## Independent one-warp tiles

This experiment replaced larger WGMMA groups with one-warp tiles, TMA rings, separate GEMM launches, and GPU-produced routing. Mathematical and layout checks passed on Spark, but H200 trials were substantially slower and exposed repeated-call inconsistency. Jobs `acffa48d276d4631` and `afc9fe19c82c4eb0` did not produce a valid new scored implementation.

Synchronous emulation can establish index mapping and arithmetic while missing asynchronous hazards on the target GPU. This branch is parked. A possible proxy-ordering explanation remains a hypothesis rather than a confirmed root cause.

## Phase profiling on H200

Diagnostic job `b6387d8f931647f0` retained the selected timed kernel and measured instrumented variants on generated inputs during preparation. All eight records were returned, covering two seeds at batches 16, 32, 128, and 256. Probe/control overhead ranged from approximately -0.4% to +1.4%.

- GEMM1 occupied roughly 60% of per-CTA elapsed time in these samples.
- Final reduction durations ranged from approximately 1 to 9 microseconds.
- Copy producers experienced sampled waits for consumers to release ring slots.
- Second-GEMM activation-readiness waits reached approximately 6–10 microseconds at the 95th percentile across CTAs.

These results favor investigating consumer execution and work handoffs over routing or final-reduction changes alone. They do not isolate a particular instruction stall or establish a hardware DRAM-bandwidth bottleneck. Phase windows overlap, producer byte counts describe logical issues, and empty-ring waits sample one in 16 positions.

Evidence: [summary](research/spark/phase_probe_h200_summary.json), [full evaluator result](research/spark/phase_probe_h200_result.json), and [instrumented source](cuda_persistent_phase_probe.py).

## Fixed-width WGMMA consumer

This experiment replaces eight WGMMA-width dispatch paths with a fixed N32 consumer for batches 25–32. It trades padded tensor work for a smaller executable. Valid columns retain their original arithmetic and BF16 boundaries; unused shared activation columns are initialized, and inactive partial-result writes are guarded.

The N32 build uses 96 registers without spills. Each kernel entry's executable text shrinks from 373,120 to 107,520 bytes. Static WGMMA instruction sites across the two kernel instantiations fall from 1,024 to 128. Code-size reduction is not a measured speedup or a reduction in executed tensor operations.

N32 passed 50 independent-reference cases and 400 mixed graph replays on Spark. The actual calibration runner also passed 6,000 queued integration replays, with zero observed drift at its 300 checkpoints. The wider N64 version passed functional checks but spilled registers in the Hopper build; three register-lifetime/grouping variants still spilled and are parked.

[cuda_persistent_fixed32_paired.py](cuda_persistent_fixed32_paired.py) retains the control unless the trial passes its paired gain thresholds, including a positive result at the scored batch of 32. H200 job `bdd2a30ed0cd4cd3` passed correctness and 6,000 queued cold replays (zero observed drift at 300 checkpoints), but all nine paired comparisons lost. The geometric-mean control/trial ratio was **0.981843737**, approximately 1.85% higher trial latency. The control was retained; the official 2.356605× result measures that fallback. Smaller executable text alone did not improve end-to-end performance. Frozen builders, sources, and validation evidence are in [fixed_width_v1](research/spark/fixed_width_v1/).

## Dedicated epilogue service warps (prototype)

The prototype adds two epilogue warps to four compute warpgroups (640 threads per CTA). Each compute group publishes FP32 accumulator tiles to a single shared-memory mailbox, then starts its next compute segment while a helper performs the original epilogue. Two-slot weight and activation rings free space for the 64 KiB payload area. Atomic last-arrival completion handles split units without helpers waiting for future global-counter progress.

Mailbox publication and acknowledgment use CTA-scoped shared-memory acquire/release operations and 128-thread producer barriers. Helpers fence their global stores before publishing activation readiness; a final 576-thread join includes consumers and epilogue helpers, while the two TMA producer warps have already exited. The ordering primitives follow the [NVIDIA PTX memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model).

Initial offline Hopper compilation succeeds at 96 registers, with 20 bytes of spill stores and 16 bytes of spill loads per kernel entry. This needs further investigation. The prototype passed **55 independent FP32-reference cases and 440 mixed graph replays** on GB10, including duplicate IDs, empty routing, and chunked M385. The development build uses 94 registers and 88,448 bytes of dynamic shared memory. H200 private job `6392db25a39e452e` subsequently passed the trial/control correctness comparisons and 6,000 queued cold replays, with zero observed drift at 300 checkpoints. All nine paired timings lost: geometric-mean control/trial ratio **0.853063352**, approximately **17.2% higher trial latency**. The control was retained, so the official 2.367177× result measures the fallback. This design is rejected. The run does not isolate the relative costs of reduced ring depth, additional shared traffic, and fewer epilogue lanes. Development emulation uses smaller 16-token job tiles and synchronous warp MMA to fit GB10 shared memory; it cannot establish Hopper asynchronous correctness or speed.

## Register redistribution and MMA grouping

A separate Hopper prototype keeps the original three-slot rings and epilogues. It adds two idle warps to complete the producer warpgroup, then uses warpgroup-uniform `setmaxnreg` operations to donate producer registers to compute warpgroups. The 104/64 split fits the 640-thread entry allocation exactly: `512×104 + 128×64 = 640×96 = 61,440` registers.

An initial common-control-flow placement was ignored by ptxas (C7507). Moving the decrease into the producer-only branch and the increase after its exit emits actual register-reallocation instructions. Compiler reports now explicitly retain C75 diagnostic messages, which ptxas labels as informational rather than warnings.

With that branch placement, a 104/64 build commits four MMA steps for token tiles up to N32 and two for larger tiles. It compiles with zero spills and no C75 serialization diagnostics. Uniform four-step builds either spill or trigger C7512; the mixed grouping avoids those failures in this build. The mixed build also passed 75 FP32-reference cases and 600 mixed graph replays under GB10 synchronous warp-MMA emulation. Register reallocation was stripped in that development test; it does not establish Hopper register-allocation correctness or performance. H200 job `eae5127c234742a2` passed the trial comparisons and 6,000 queued cold replays, but the geometric-mean control/trial ratio was **0.988282350**, approximately 1.2% higher trial latency. The control was retained; the official 2.354507× result measures that fallback. The register/grouping change is rejected. The [PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) requires every warp in a warpgroup to execute the same register-budget operation; individual TMA and idle warps therefore share one producer budget.

## Small-batch block residency

The TINY execution path traps if it encounters fat jobs, but its launcher still reserves the maximum of skinny and fat shared-memory requirements. This prototype removes the unreachable fat reservation, uses two weight-ring slots, and compares eight compute warps with four. The four-warp version handles two original 512-K slices per warp; scale addresses preserve the existing eight-slice packing, so weights remain unchanged. Small routing jobs are split at eight tokens, also supporting repeated expert IDs.

A native GB10 test passed **35 independent-reference cases and 420 mixed graph replays** across batches 1–16. That development test uses a single ring slot with one early prefetch, and one block per SM; it does not measure H200 speed. Initial Hopper compiler probes found 217 registers without spills for the four-warp single-block target and 168 registers with 40/20 bytes of spill stores/loads when targeting two blocks.

H200 private job `f4bf7c125d8b48ba` completed. It compared the control, trimmed eight-warp blocks, four-warp blocks, and a two-block grid. CUDA occupancy must report at least two resident blocks before the two-block grid is eligible, because the persistent kernel contains a grid barrier. Calibration covers three seeds at batches 1, 4, 8, 12, and 16, with control retained unless gains survive every measured seed/shape in a dispatch band.

The residency run passed correctness and 9,000 trial replays. Geometric-mean control/trial ratios were 0.972838 (trimmed eight-warps), 0.806041 (four-warps, one block per SM), and 0.941268 (four-warps, two blocks per SM). All dispatch bands retained control. Maximum observed self-comparison relative-L2 drift was approximately 6.09e-5; the FP32 reductions are not globally bitwise deterministic. Its official 2.363633× result measures the fallback.

## Grid sizing and compact TMA rings

Grid experiment `fd6ad1aafa7f4363` keeps the verified device kernel and searches safe grids no larger than the SM count. One generated seed screens candidate grids; two additional seeds validate the best candidates over scored and nearby batch sizes. The aim is to reduce boundary-unit reductions when a grid divides the work more evenly. All seven bands retained the original 132-block grid. The run passed correctness but found no robust gain; its 2.364838× result measures control. Smaller grids generally lost throughput despite reducing some boundary work.

Compact-ring experiment `61f20f5ad10d4afc` removes unused skinny-path shared storage in the WGS build and uses unsigned 16-bit routing counts and offsets. The maximum is 3,072 entries, so this is an exact representation. The global workspace ABI and full 384-token capacity remain unchanged. Dynamic barrier indices keep full and empty slots disjoint for each ring depth. A host C++ layout check gives an 18,944-byte aligned base; 4/2 weight/activation rings fit in 231,936 bytes, below H200's 232,448-byte block limit. CUDA resource and correctness checks still determine actual eligibility. The comparison includes compact 3/3 and 2/4 variants.

These checks use generated preparation inputs and retain only scheduling/variant metadata. Input-derived tensors and results are not cached for timed calls.

## Small-batch scheduling parameters (pending)

The scored tuning runs keep the original eight-warp device implementation. They compare four- and five-slot weight rings and separately test whether three or four final epilogues can be assigned to compute warps. The existing four partial buffers bound the tail count. Each band uses three generated seeds and nearby shapes, and retains control unless gains pass both the aggregate and per-comparison guards. No input/output tensors survive calibration.
