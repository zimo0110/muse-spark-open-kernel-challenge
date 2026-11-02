# MoE overnight work log

User authorized work on muse_moe_w4a16; at most FIVE scored submissions tonight. Do not modify muse_bf16_proj or Claude's summary.md. No scored submissions used yet. CLI signed in as zimo0110. Daily limits: 20 checks, 10 scored per problem, reset 00:00 UTC. User requested at most two usage sessions; account-limit restart cannot be guaranteed by the agent.

## Research / design

- NVIDIA Hopper tuning guide: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- Marlin authors' implementation and design: https://github.com/IST-DASLab/marlin
- Pinned SGLang source: https://github.com/sgl-project/sglang/tree/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
- Triton matrix multiplication tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

Keep weights NVFP4 in prepare; decode only during timed GEMMs. Keep BF16 activation boundaries. Preserve fresh routing and inputs; no caching outputs. Use CUDA graph replay with cold L2 for any tuning. Target rank 1 currently ~2.69x (provisional); do not promise reaching it.

## Checks

1. fc8f748e8ef64ea9: v1_grouped.py, Triton grouped on-the-fly decode, fused SwiGLU. Passed all tests, 0.223x. GEMMs dominate: 400us small m to >2ms large m. Far slower than Marlin.
2. d9d0ae9a12bc4e5f: v2_split.py, split K8 small m/K4 large m, postpone FP32 global weight scale until after MMA. Passed, 0.319x. Still not competitive.
3. 375ae69463584979: marlin_probe.py, baseline with expert alignment count32 instead of256. Passed 0.988x. Profiles show small m has ~30us of torch postnorm overhead; larger m almost entirely Marlin GEMMs.
4. ef93b0af84cf4930: submission.py (first custom CUDA version), running. experts.cu is editable source mirrored as CUDA_SRC in submission.py. Uses per-expert route compaction + vector packed W4 loads + register decode + BF16 mma + fused SwiGLU. Intermediate and output BF16. Global scales applied to FP32 accumulator before BF16 boundary. NT1/2, 4warps, U2 initially. Profiling and correctness pending.
5. f51fe74ad35f4732: marlin_fused.py, baseline Marlin + own Triton postnorm weighted sum instead of baseline's torch small-batch fallback. Running; useful fallback.

## Files

submission.py: current candidate, self-contained (CUDA source embedded). experts.cu: source for editing CUDA; changes must be re-embedded in submission.py before submitting. starter.py: untouched original starter. baseline/reference/nvfp4/checks/TASK/AGENTS from okc init. v1_grouped.py and v2_split.py: correct but slow experiments. Profiles saved by CLI.

## Next

Read CUDA check result; if good, tune NT (8/16/32 token tiles), WARPS(4/8), U(1/2/4) in prepare using independent synthetic routes and cold-L2 CUDA graphs. Avoid spending a check per tile config. Improve scale decode: E4M3 -> BF16 via exponent bit conversion, with correct zero/subnormal handling. Consider Marlin fallback for large m if CUDA loses. Fused postnorm variant should give safe measurable gain on small m even if custom GEMMs fail. Only submit after passing and choosing a defensible best candidate.

## Update, first session around 01:15 PT

14 of 20 checks used; 0 scored submissions used. Reserve some checks for scheduled second session. One-shot same-chat automation `resume-moe-overnight-work-once` created for 05:05 PT today (after account's 5-hour window reset ~05:01 PT). Must disable itself when session two begins, and never schedule a third session. State in overnight_state.json. This is a timed follow-up, not a guaranteed token-reset trigger; app must remain running on powered computer.

Further results:
6. b9c621365e8f4585: v4_tma.py, repacked slab tiles loaded by TMA, 0.447x, correct.
7. baea60495eb64fdb: marlin_tuned.py, independent synthetic-route cold-L2 graph tuner for block sizes8/16/32/48/64. 1.223x; best_checked.py is this backup (older best).
8. 119f971baa7742ab: v5_fastdecode.py, fast FP4 exponent-embedding decode + native BF16 multiply; passes, 0.755x. Repacked codes are permuted losslessly. Shift factor2^115 applied only inside GEMM, not weights stored at lower precision. This follows the general Marlin exponent-shift technique. PTX supports BF16 multiplication subnormals by default.
9. 150d5bb232fc4707: marlin_compact.py. **1.3023x**, CI1.2828–1.3218, all tests/cases passed. Best conservative checked candidate. Custom one-CTA small-batch alignment and vectorized fused postnorm. Improvements all six scored workloads.
10. 3476f2cb6ada4649: current submission.py (persistent custom CUDA with tuner). Passes 1.023x; not best! Do not submit this blindly. NT/W/U autotuning. v3/v4/v5 backups hold previous CUDA variants; experts.cu current persistent direct-load source.
11. 258cfd67fe46400c: triton_raw.py, native BF16 raw FP4 decoding with tuned splitK. Passes 0.293x. Do not pursue unless new evidence.
12. 531cc79de4c74c5a: marlin_atomic.py standard Marlin atomic-add reduction. Passes1.3199x, little difference from1.3023 conservative FP32 reduction; no compelling reason to prefer atomic variant.
13. e98f75f28c6a4de2: marlin_local.py pending. All routing compacted to local experts, then is_ep=False passed to Marlin; no nonlocal metadata, so unnecessary EP scans eliminated. Uses two-kernel route compaction for larger m, one-CTA for <=8. Norm vectorized, FP32 reduction unchanged. Own local-only metadata is critical: do not pass is_ep=False with baseline metadata containing expert -1.
14. 94a76cb73599425a: marlin_custom.py pending. Compiles a tiny torch CUDA launcher around SGLang's supplied Apache-2.0 Marlin template. Existing baseline launcher considers only two tile geometries; this launcher instantiates N64K128T128, N128K64T128, N128K128T256, N256K64T256, N64K256T256, plus BS8/16/32/64 and occupancy1/2/4. Tunes independent synthetic routing and cold-L2 graphs, comparing candidate outputs to standard Marlin. Keeps FP32 reduction. Includes SGLang header via DEFAULT_INCLUDE. Source mirrored in marlin_launcher.cu and embedded in Python. Catch compile failure falls back and prints diagnostics. Prepare budget265s, total rule300s.

Current main `submission.py` is NOT the best. Before scored submission copy the chosen verified candidate there, preserve metadata / module imports, ensure content matches or recheck if edited. All candidates are self-contained Python submissions with embedded CUDA as needed. No modifications outside muse_moe_w4a16 except /tmp temporary read/research files.

Public full kernel rules read from website assets: precision restrictions cover original weight representations and BF16 activations; no separate MoE accumulation precision clause. Still prefer FP32 reduction. Do not quantize activations, expand weights permanently in prepare, cache outputs, or exploit known test seeds. Serving priority top50 overall; current1.30x is far from MoE top1 (2.6888x), around rank830/1033, so do not misrepresent competitiveness.

## Update around 01:15 PT

13. marlin_local.e98f75f28c6a4de2 passed **1.34678x**, CI1.32658–1.36698. Copied marlin_local.py into main submission.py, so main is now this checked conservative best.
14. marlin_custom.94a76cb73599425a passed1.3386x only via fallback: custom build failed because DEFAULT_INCLUDE doesn't contain csrc. Fallback is_ep=False module cold compilation consumed ~200s. Do not interpret this as a measurement of the custom launcher.
15. dd45e9342a464667: marlin_custom_v2.py pending. Corrected to SGLang's intended load_jit interface with a private temporary .cuh and KERNEL_PATH/csrc include. TVM-FFI TensorView launcher wraps Marlin templates. Fallback is_ep=True with compact local metadata, which is correct and uses the judge-warmed module; this avoids spending ~200s compiling all is_ep=False baseline variants just for validation. Custom templates themselves remain kIsEP=False and preserve FP32 reduction.

**Scored submission 1/5 dispatched:** 98990be0cd294294, exact checked marlin_local.py. Pending. Record final result/rank. No more than FOUR additional scored submissions tonight across both sessions. 15 checks used, FIVE left. Intend reserve at least two for second session if no major repairs needed.

## Update around 01:25 PT

First scored run 98990be0cd294294 passed **1.34858x**, rank769/1036 at read; leader2.6888x (needs~1.99x faster than our current result). One of five scored attempts used. Best conservative main submission.py remains identical to marlin_local.py.

15. marlin_custom_v2.dd45e9342a464667 **built and ran successfully** via SGLang load_jit! All tests passed,1.36075x. Check logs truncated to last16k chars, begins at m16 tuning; real custom configurations did run. Shared-memory-incompatible configurations were skipped safely. Small gain overlaps current posted result uncertainty, so not submitted. This is a valid infrastructure for further tile / occupancy / grouping exploration.
16. marlin_groups.9b27442238384f73 passed1.32970x. Separates cold/hot experts by GPU route counts, uses two independent Marlin launches per GEMM with different BS (8/32/64), tunable split threshold. No improvement overall. Inspect shape-level results before abandoning: there may be useful per-shape wins to combine. Uses baseline Marlin EPTrue on compact local metadata to avoid expensive cold module compile.
17. 653db941b58043ac: gemv_hybrid.py pending. Dedicated single-token path: FP32 vector dot of BF16 activations with NVFP4 weights decoded on the fly (native BF16 exponent embedding), four output rows per CTA, no routing-sort launch. Fused gate/up/SwiGLU first stage, second GEMV then vector postnorm. Larger m use checked marlin_local. Retains original NVFP4 raw weights via state['raw'] (no dequantized cache).

**17 checks used, THREE left today.** Four scored attempts remain under user's total5 cap. Prefer reserve at least two checks and most scored attempts for second session at05:05 PT. Do not spend scored attempts on noisy ~1% changes. Main challenge goal remains substantially unmet; final report must be candid.

## First pass handoff, 01:35 PT Oct29

17. gemv_hybrid.653db941b58043ac passed1.19195x, but single-token path regressed to100us vs~49us Marlin-local. Direct vector reduction didn't overcome decode/dispatch costs. Do not use it in main.
18. hot_dequant.d718a8d6b2554ec6 passed1.32081x. Explores dequantizing only heavily used experts into TRANSIENT BF16 buffers inside the TIMED call, then BF16 tensor-core GEMMs; cold experts remain Marlin. Tuner tried thresholds32/64/128 and disabled for best performance (cut0) at m128/256. Decode/write/read bandwidth cost outweighed savings. No BF16 weight expansion was stored in prepare. Not part of main.

Main submission.py byte-identical to marlin_local.py, the conservative checked and scored result. Highest check1.36075x (custom launcher) overlaps1.34858x posted uncertainty and adds compile complexity, so no second scored attempt spent. Keep posted result until a meaningful gain is verified.

All dispatched GPU jobs are now complete. ONE scored attempt used, FOUR remain under user's total5 cap. EIGHTEEN checks used, TWO remain before00:00UTC reset (17:00PT, after this overnight window). Do not consume them carelessly; code-level synthetic autotuning inside prepare is allowed and much more efficient than one check per tile. Independent synthetic inputs only, no known judge seeds or harness inspection.

The second/final scheduled session is05:05PT today via resume-moe-overnight-work-once; record second_session_started=true and disable automation at its start. If account limits or challenge quota prevents further work, report the limitation honestly; no third session requested. Mac/app must remain running. No rate-limit reset credits have been used or purchased.

### Possible second-session directions

- Read actual custom-launcher tuning log and shape profiles. Valid code in marlin_custom_v2.py wraps SGLang Marlin templates, compiles only20 specific tile variants, and preserves FP32 reduction; this is a better foundation than slow custom CUDA prototypes. It can be extended to choose first/second GEMM configurations separately (currently same tile/occupancy/BS for both), add occupancy3, and tune pipeline stages2/3/4/5 for group16 NVFP4. Baseline factory uses only two geometries and fixed pipe_stages. N64/K128 layout is valid and tested by candidate validation.
- Consider per-stage tile tuning rather than full-pipeline one-size. GEMM1 is8192 output columns, GEMM2 is4096 and has route-major activation rows. Different occupancy/cache settings may matter. Tune CUDA graph replay after cold-L2 reads and use multiple random routings. Use baseline reference outputs (EPTrue module already warmed by judge) to reject new configs in prepare; avoid paying~200s to compile the full baseline EPFalse module just for tuning.
- Expert-binning prototype marlin_groups.py was correct but didn't improve aggregate. Avoid repeating same experiment unchanged. New kernels use own compact local-only metadata; this is required for kIsEP=False.
- If no substantive gain is found in the two remaining checks, preserve current correct result and report rank honestly. Rank1 aspiration remains unmet, ~2x away; don't spend all four remaining scored attempts on noise or call the work number1.

Research already read: NVIDIA Hopper tuning guide, PTX BF16 multiplication/subnormal behavior, Marlin authors' techniques, pinned SGLang Marlin launcher/dequant/template sources. Public kernel rules confirm weight/activation formats, fresh inputs and no harness manipulation; all candidates preserve them.

## Second/final session, Oct29 05:05 PT

Recorded session two started and paused the continuation automation. No third session. No pending jobs at start; CLI confirms18 checks and1 scored used. BF16 files untouched.

19. marlin_stages.py, job24aeb28edd0d4817: tests independent configurations for each GEMM stage, adds occupancy3 and pipeline stages2/3/4. Embeds only valid static Marlin variants (BS64 retains4stages). Shared-memory size uses the candidate stage count and conservative BF16 scale storage, with extra1024-byte safety reserve. All kernels retain local-only metadata and FP32 reduction. Each candidate compared with stock SGLang outputs on two independent synthetic router distributions, followed by cold-L2 full-graph timings. Block size remains common to both stages, launch configurations differ. One check remains after this job.

19. marlin_stages.py passed all correctness tests,1.38248x (CI1.36175–1.40322), prepare63.9s. Selected stage4 throughout. Occupancy3 withN128K64T128 wonlargerM; stage configs differ smallM. LargeM paired speedups1.483/1.460. New bestcheck; conservative main remains posted version until final selection.
20. marlin_refined.py job35b65c3567a24ae3: last daily check. Adds N64K64T128, N256K128T256, N128K256T256; extends stage-specific tuner to three independently sampled signed heavy-tailed Student-t routers acrosspopularity0.015/0.03/0.06 (no reference import/seeds). Corrected conservative metadata shared-memory bound to32*BS (actual template layout), plus1024 device margin. Reports rejected/skippedcounts to clarify stage2/3 behavior. Scored ledger still1/5, four available; no pending prior checks.

20. marlin_refined.py FAILED skewedM128 andM1: outputNaN/Inf. The widersearch exposed invalid configurations and a tuner bug: NaN error comparesfalse with `>0.005`, allowing an invalid config to win. Do NOT use that candidate. No timing result. Dailycheckquota now20/20; do not evade.

Scored attempt2/5: c81426f679214422 marlin_final.py, pending. Actualleaderboardrun within explicitfive-run authorization; no furthercheckrunsavailable. Repairs tuner to requireeveryerrorfinite, restricts pipeline depth4 to the previouslychecked depth, and validates selected fullstagepairs on two FRESH holdout routings (normalheavy-tail andstrongskew). If holdoutfails, fallsbacktostockMarlin. Main remains conservative postedcandidate untilnewresultpasses. Checked marlin_stages.py1.38248 remainsavailablefallback. Jobs inspected before submission: exactly1prior MoEscoredjob, nonepending.

## FINAL HANDOFF — second session complete, Oct29 ~05:20 PT

Scored attempt2/5 **c81426f679214422 passed**, **1.46386884x** versus SGLang (95%CI 1.44191–1.48583). All5 tests and all6 timed workloads passed; relative errors~0.0053–0.0055 versus0.015limit. Prepare80.2s below300s. Noflags. **submission.py now byte-identical to marlin_final.py**, the successful scored source. Conservative marlin_local.py remains available.

The finite-error guard correctly rejects the invalid N64K64 tile; valid N256K128T256 geometry(kind6) wins across shapes. This is a real new tile outside the original stock launcher's search space. BS8 selected forM<=32, BS16 for128, BS32 for256 in this run. Pipeline4, FP32reduction, local-only routes, original NVFP4 weights. Fresh holdout normal/skewed synthetic inputs validated selected full pairs before timing; runtime never caches data-derived outputs.

Verified live leaderboard: rank **688 / 1061**, our1.4639x, leader2.6888x, to_first1.8367. **Rank1 not achieved.** Versus prior posted1.34858x this is ~8.55% higher speedup (~7.88% less kernel time). The remaining gap needs a larger architecture change than launch tuning; GEMMs still dominate, so further tiny postnorm/routing optimizations won't close it.

TOTAL across BOTH sessions: **20/20 daily checks used, 2/5 authorized scored submissions used**,3unusedscoredattempts. No pending GPUjobs. No third session. Resume automation verifiedPAUSED. No otherproblem/account used to evadecheckquota. No credits purchased/reset. BF16folder andClaude's summary untouched. Stopovernightoptimization now.

Next human-directed work, afterquota reset: investigate a Hopper-native groupedW4A16tensorcorekernel with fewer decode/gather costs (and compare real cold-L2bandwidth, not eager timings); retain this correctscoredcandidate asreference. Avoid invalid N64K64 tile and unvalidatedpipeline depths. Researchsources andfailedexperiments documented above.

## User-authorized continuation Oct29 morning

User explicitly asked to continue after the twoovernightsessions, overriding the earlier stop. Automation remainspaused; no additional automatic sessions scheduled. CLIcheckquota remains20/20, scored2used beforecontinuation. Live rank715/1095, leader2.6888x, our1.4639x. Still honororiginal5scoredtotalcap.

Scoredattempt3/5 f9dc020ba65d4d0a, marlin_wide.py pending. Extends validMarlinsearch to N512K64T256,N512K128T256,N256K128T128,N256K256T256,N256K64T128. Tunes persistentgrid independently fromsharedmemoryoccupancy via full/half/quarter grids. Builds onlystage4; retainsfiniteerrorguard andfreshnormal/skewedholdoutvalidations. UseslargerFP32scratchsizedforN512. Currentmain staysbyteidenticalto1.46387xwinner. Research: NVIDIA HopperTuningGuide occupancy/sharedmemorylimits revisited (https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html); pinnedSGLangMarlinCUDAtemplate checked forwarpN/Kmapping andminimumtileconstraints. NoBF16changes.

3/5 f9dc020ba65d4d0a passed1.45766x, CI1.43579–1.47952. Noimprovementover1.46387x; preserveexistingmain. Newshapesvalid, butnooverallgain. Prepare102.6s.
4/5 7a64d9073f94465f marlin_pipeline.py pending. TestsEVENdepths2/4onvalidtiles, addsBS8forM128/256, andsizesinitialsharedcacheboundfromactualNVFP4scale-bytewidthandBS8Arows. BS64staysdepth4tohonorstaticreductionconstraint. NoN64K64invalidshape. Usesfinitevalidation+freshholdoutdata. OnlyONEauthorizedscoredsubmissionremains. Checkquotastill0; noscheduledwakeups.

4/5 7a64d9073f94465f passed1.48365x, CI1.46140–1.50591, prepare121.7s. Alltests/timedcallspass. Slightgain(~1.35%)over1.46387xoverlapsnoise; strongestverifiedcandidatebutsmall. Maincopiedbyte-identicallyfrommarlin_pipeline.py. Selecteddepth4throughout; BS8largeMloses, BS64loses. ExistingtileK128N256andK64N512tradeoffremains.

5/5 e3475f5490094d1f marlin_occupancy.py pending. Finalauthorizedscoredattempt. Extendsresidenttargetto5/6/8CTAsperSM, growsworkspaceandFP32scratchto8*SM, andqueriesactualCUDAoccupancy(register/sharedlimits)beforenewhighoccupancylaunches. Keepsallprevious1–4configsevenwhenphysicalresidencybelowgridtarget, preservingvalidpriorsearchspace. RemovesconsistentlyslowBS64fromautotuning; arbitrarynearbyinputshapesstilluse16/32default. Currentmainisverified4/5source. Noadditionalleaderboardrunswithoutfreshuserauthorization; checkquotareset00UTC=17:00PT.

FlashInferprimaryAPIresearchconfirmsCutlassW4A16ConfigonSM90usesMXFP4weights, whereasNVFP4pathquantizesactivationsandtargetsBlackwell; notadirectvalidreplacementforNVFP4W4A16group16. https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/fused_moe/api.py

### Next architecture research (hypothesis, NOT implemented/verified)

NVIDIA CUTLASS example55 now includes E2M1xBF16 Hopper mixed-input GEMM with register-source narrow operand, offline shuffling for coalesced register loads, and FP32 accumulation. Published limitation: scale group size >= tileK (default64), so ourNVFP4group16 is not a drop-in. https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu

Potentialcustomdirection: transpose expertGEMM so FP4weights formregisteroperandA ofWGMMA and routedBF16activations sharedoperandB. Consider64-output-by8-route tiles; losslesslyrepackFP4codesforregisterfragments andtransposeoriginalE4M3scales forcoalescedgroup16loads. Decode toBF16registersinsideeachK16step andretainFP32accumulation. ThiscouldremoveMarlin'swarp-Kpartialreduction andexploitnativeHopperasynchronousMMA. Exactregisterfragmentmapping,sharedmatrixdescriptorlayout,registerprefetchdepth,splitKreduction andnearby-shaperobustnessallrequireverification; do NOT mistakeconceptforreadycandidate. PriorTMAprototypewasordinarywarpMMAandlost; do notrepeatitunchanged. Group16supportandsmallroutedbatchperformancearethehardparts. NeedactualGPUiterationaccessbeforeattemptingpublication.

## Continuation handoff Oct29 ~10:25PT

5/5 e3475f5490094d1f passed **1.48739943x**, CI1.46509–1.51124, prepare99.4s. All5testsand6timedworkloadspass; noflags. Highoccupancytargetsmostlyrejectedbyphysicalregister/sharedmemorylimit; selectedconfigsstilldepth4at<=3targetblocks/SM. Thisconfirmsmorelaunch-tuningisunlikelytoclosethe~1.8xgap. Newbest1.4874xonly~1.6%aboveprevious1.4639x, withoverlappingintervals, soimprovementmodest/uncertain.

**Main submission.py byte-identical to marlin_occupancy.py**, successfulfifthsubmission. Priorwinner marlin_final.py andfourth marlin_pipeline.py preserved. Liveleaderboardverifiedrank703/1097, leader2.6888x,to_first1.8077. Rank1NOTachieved.

AllFIVEoriginallyauthorizedscoredsubmissionsnowused; actualCLIallows5morescoredtodaybutneedfreshhumanbudgetauthorizationbeforeusingthem. Checks20/20until17:00PTtoday. Nojobsremainpending. AutomationPAUSED. BF16andClaude'ssummaryuntouched. Ongoinggoalremainsunmet; waitingfornewsubmissionbudgetorcheckquotareset, NOTclaimingoptimizationgoalcomplete.

## New direct-register architecture implemented, Oct29 (user: "go do that")

User confirmedNOH100/H200developmentmachine. CLIstill20/20freechecksand5scoredused; original5-scoredcapnotlifted. NoGPUjobsdispatchedandnoquotaevasion. Main1.48739943xsourceSHA b4fc46dc80e7ba019ee0a4d27a97caddc4c0cd28f96d8f114c3bb24018f98746 preservedunchanged.

**register_candidate.py** is a separate486-line experimentalarchitecture, NOTGPU-verified. ItlosslesslypacksNVFP4codesandoriginalE4M3scalebytesinto64-output-row/256-KtilesinONEfusedpackinglaunchperbank. Gate/uprowsarepairedevery8columns, enablinga singleMMAaccumulatortileandfusedSwiGLUinsideGEMM1. Weightsloadcoalesceddirectlyintoregistersanddecodeinline; onlyroutedBF16activationsusesharedmemory. Optionalregisterdoublebufferprefetch(smallMdefault), per-item8/16/32tokentiles,FP32MMAaccumulation,compactlocalroutingandfusedpostnorm. Registercap128andpersistentgrid3*SMareINITIALUNMEASUREDchoices, notclaimedoptimal.

Persistentdispatchwalksonlylivetiles, avoidingconservativegridlaunchesforinactivework. NonpersistentgridmodepreservedforA/Btesting. IdenticalBMplansreused. Everycallbuildsfreshrouting/intermediates;stateholdsweightbytes,originalnormandshapeconfigurationonly. Noimportofreference/harness,nohostdata-dependentbranches,networkorsubprocessinsubmission.

Reuseattribution: warp-MMAfragmentlayouts,FP4bitpermutation/inlinePTXdecoder,activationlayouthelpersandgate/upsplitadaptedfrombigboateng'sMITpublicsourceat6d9b77656b6c295e421be9e1424b0f8f99299b0a. FullMITnoticeembeddedincandidate; localmodificationsApache2. Source/license/readmeandhashprovenanceinresearch/. Published1.832xBELONGSTOUPSTREAM,notourderivative; no claimedperformancefornewcandidate.

**verify_register_cpu.py PASSES**. ExecutesACTUALGPUpacking/routingfunctionbodiesusingasmallNumPyintrinsicinterpreter(noTritoncompilerrequiredonMac). Verifiesexactcode-bitroundtripincludingnegativezero,exactscale-byteroundtrip,noformat/sizeexpansion,all16FP4codescrossedwithall127positivefiniteE4M3scales,randommulti-tilebanks,logicalKpermutationdot-productinvariance,registerlayoutbijectivecoverage,persistenttilecoverage,and36routingcases(9Mvalues×4distributions)with108workplans. AlsochecksastagedBF16MoEmodelwithgate/upfusion,globalscales,normandweightedroutes. TheseareCPUsemanticchecks,NOTGPUfragmentexecution/compilation/correctness/timing.

**verify_register_gpu.py** is a developmentrunnerforSM90only. Capturesbothpersistent/gridmodes,updatesfreshnormal/skewedinputsbetweenreplaybatches,checkswiththelocalofficialFP32reference,soaksreplays,andoptionallytimespairedSGLangbaselineswithcoldL2. RunnerdoesnotmakeCLIsubmissionsandreferenceimportisonlyinthisexternaldevrunner. Syntaxand--helpcheckedlocally;GPUrununavailable.

Nextactualactionafter17:00PT(00UTC)freecheckreset:
`okc check muse_moe_w4a16/register_candidate.py --no-wait --json`
Inspectcompilation/correctnessFIRST,thenprofiles/pairedtimes. DoNOTpromoteovermainwithoutpassingGPUchecksandmeaningfulspeedup. Originalscoredcapremains5/5; furtherleaderboardpublishingrequiresnewauthorization. Noautomaticwakeupcreatedforthiscontinuation.

## Oct29 17:00PT freecheckreset / direct-register GPUvalidation

Useraskedgetreadyandvalidateatreset; firstcheckedCLIquotaactual20left. NoH200devmachine. CPUpreflightagainpassed, verifiedmainSHAunchanged. Firstnewdayfreecheck3fa4a39d98ff46f9: register_grid.py (samecandidatebutpersistentFalse). Startwithsimplerschedulingtoisolatemath/packingcompile; thenpersistentA/B. Prior5scoredtotalcapretained; thisroundusesFREEchecks, noleaderboardattempts.

GPUresetroundresultssofar:
1. grid3fa4a39d98ff46f9 passes1.71942xCI1.69362–1.74521,prepare0.85s. Newarchitecturecorrectandmateriallyfasterthanposted1.4874x.
2. upstreamcontrol27e4362141194970 passes1.85268x; externalbigboatengMITsource, NOTourresult. Maincomparisoncontrolpublished1.832replicates.
3. persistentd06602566df04d55 passes1.56449x:regresses. Fixedsmallpersistentgridreducesutilization/loadbalance; doNOTassumepersistentisbetter.
4. tuned24236946b83a445a passes1.78289x,prepare16.32s. RuntimePallowsJITreuseacrossM. Coldgraphsyntheticindependentroutingtunerselectsmostlygrid;largeMtile16/cap96;smallMcap128/192prefetchmixed. Morethan1700x? NO: score1.78289only.
5. direct5385e8f7ed2c45ec pending: one-tokenM1bypassesroutingworkplan, fusedGUfirstGEMM, W2splitK4FP32partialsfusedintonorm. Reusescreditedpublic_w2_kshelperaswell.
6. sf-fast7c87773133cd4652 pending: weight-dependentpreparecheckprovesallscalesnormalE4M3; thenexactintegerbitconstructionofSF*2^119asBF16insteadFP8toFP32multiplyconvert. Fallbackgenericdecoderforanysubnormal/zeroscale; NOassumptionthatweightsfollowjudgedistribution. IndependentlyverifiedALLfinitenormalscalebytesbothsignsagainstfloatdecode. Newoptionnotyetknownfaster.
7. N1284cf2b18dd2924682 pending: extendweighttile64->128outputswith8warps, coherentlinear-layoutbasisextensionandlosslesspackwidth128. Aimreduceactivationloads/sharedstagingandCTAsperoutput. ActualpackingfunctionsCPUinterpretedandpassed3multi-tilebanksbothGU/ordinary. GPUfragmentlayoutsneedverification.
NoadditionalSCOREDsubmissionsthisroundsofar. Postedmainstill1.4874; newbestoursfreecheck1.78289. Totalnewday7/20checksused.

## Evening reset round progress (~18:00 PT), 19/20 free checks dispatched

- 3fa4a39d98ff46f9 register_grid.py: done, passed=True, speedup=1.7194156682240127
- 27e4362141194970 upstream_control.py: done, passed=True, speedup=1.8526755034259363 (credited upstream control)
- d06602566df04d55 register_candidate.py: done, passed=True, speedup=1.564486314536518
- 24236946b83a445a register_tuned.py: done, passed=True, speedup=1.7828917108629951
- 5385e8f7ed2c45ec register_direct.py: done, passed=True, speedup=1.7558738031496877
- 7c87773133cd4652 register_sf_fast.py: done, passed=True, speedup=1.6930519357885447
- 4cf2b18dd2924682 register_n128.py: done, passed=True, speedup=1.659272833149163
- 9ec58d127cd048b1 register_direct_tuned.py: done, passed=True, speedup=1.7837721866046292
- 7a53a359fa974b14 register_async_x.py: done, passed=True, speedup=1.5943161887152442
- da2c5eb8d27d46ff register_dual_tuned.py: done, passed=True, speedup=1.7710114296715438
- 2135ac9e34194be1 register_async_pipeline.py: done, passed=False, speedup=None
- 8890dcd73d1f43d5 upstream_async_wgmma.py: done, passed=True, speedup=1.7026416923533272 (credited upstream control)
- 2dd1475a13c243a6 register_kmajor.py: done, passed=True, speedup=1.7423295638627787
- f333e5aaf43a4cad register_async_pipeline_v2.py: done, passed=True, speedup=1.6799132157407335
- 6dbb61c518744513 register_swizzle.py: done, passed=True, speedup=1.712480295289881
- c73b0400c21d4a37 register_prefetch_late.py: done, passed=True, speedup=1.593953985174534
- d12fc8e9e98648b1 register_bk64.py: done, passed=True, speedup=1.2787397172597228
- 19b3e683addc4186 register_multi_k.py: done, passed=True, speedup=1.8100825153096407
- eeadefc4f09842e0 register_stream_gemv.py: pending, passed=None, speedup=None

The strongest checked local derivative is register_multi_k.py,1.81008x. The public upstream control is1.85268x, credited to bigboateng, not claimed as our original result. We have NOT achieved2.6x. Posted1.4874x source remains unchanged pending final selection; no scored jobs submitted in this evening round yet.

Important negative results: persistent scheduling, wider128-output tiles, integer scale conversion, async shared-X copies, deferred volatile prefetch, native asynchronous WGMMA, XOR placement, and smaller64-K MMA tiles did not produce a meaningful improvement. K-major placement1.7423x overlaps basic1.7194x uncertainty. Double-buffered copy initially failed compilation because the indexed descriptor retained rank3 layout; rank2 shared layout fixes compilation, but still slower. Resource probes show no spills on decode kernels (typically64–87regs), so decode register spilling was not the primary issue. Large dual-GEMM cases show2–12spills but modest effects.

register_multi_k.py merges separately credited/verified64/128/256-K layout kernels in one self-contained module, tunes stages separately on independent cold-L2 graphs inprepare, and uses the checked one-token split-K path. All judge tests/cases pass. Tuner selected256-K everywhere in that run; smaller fragments lose to extra iteration/synchronization overhead. Prepare23s. BK128 packer semantics CPU-verified; GPU variants were validated by the prepare comparison before any choice.

Last novel experiment register_stream_gemv.py: streamed BK64 FP32 dot for M1 and one-route expert work items at M8/16/32. Avoids shared-X/CTA barriers and padded matrix rows, unlike previous full-K huge-register GEMV failure. TensorCore path remains fallback. Prepare compares fresh independent outputs/timings against checked split-K/MMA paths and only enables the alternative if finite/error-bound checked and faster. All weights retain original NVFP4 bits/scales. It is NOT yet GPU-verified until pending job finishes. One free check remains for final confirmation. No other problems/accounts used; no purchased compute/credits.

### Last development result and independent confirmation

19. eeadefc4f09842e0 register_stream_gemv.py passes1.80536x, but its own cold-L2tuner rejectsALLstreamed-GEMVchoices: single-tokenGEMV~90–93usversus~41uscheckedMMA/splitK;one-routecoldexpertvariantsalsofarworse. Returnedstatekeepsverifiedtensor-corefallbacks. The FP32 reductions/extra scalar work cost more than avoided padding/barriers. DoNOTclaimGEMVwin.
20. bb8aeed3065e46e1 register_ready.py pending. Thisisregister_multi_k.py withanindependent-confirmationcommentonly, usedforfreshjudgeweights/inputs/replays. LastFREEcheckoftheUTCday;20/20used. NoadditionalSCOREDjobsintheeveninground. Target2.6xNOTachieved; bestlocalderivative1.81008x, creditedupstreamcontrol1.85268x. Mainpostedfilepreservedexactbytehashb4fc46dc80e7ba019ee0a4d27a97caddc4c0cd28f96d8f114c3bb24018f98746pendingpublicationdecision.

## Final independent confirmation / publication decision

Final freecheck bb8aeed3065e46e1 finished successfully on H200: **1.8468607626592106x**, CI [1.8191578512193225, 1.8745636740990985]. All correctness tests and six timed cases pass; no static flags. prepare19.925s. register_ready.py and register_multi_k.py are AST-identical; separate fresh-weight runs measured1.81008x and1.84686x. Source SHA256 53c2fcb96024dbc4c65a95df8252d117ee000285b7948f426e15ba880b29a7a7. This is a checked result, NOT a scored leaderboard entry. Credited upstreamcontrol1.85268x overlaps this measurement; no claim to surpass upstream or reach2.6x.

Actual CLI quota20/20freechecks used,0scoredtoday; original user lifetime cap5/5already spent. Allrecentjobsdone. Asked user for ONEadditional scored submission beyond originalcap to publish this verified candidate; no extra submission dispatched pendinganswer. Main submission.py stays exactposted1.487399432xhash; BF16 work preserved. Liveboard read: zimo0110rank736/1137,1.4874x; leaderdolanzhao2.6995x. Notfirst.

User explicitly authorized one additional scored submission beyond original five. Submitted exact checked register_ready.py once as job 9cf7ef93107b4052. Cap now6, all6spent; awaiting result. No other scored dispatch authorized.

## Sixth scored submission completed

Job9cf7ef93107b4052 PASSED, speedup1.833725418404807x,95%CI[1.8062195371287348,1.861231299680879]. All correctness tests and timed cases pass; no error. Liveboard zimo0110rank495/1140,1.8337x; leaderdolanzhao2.6995x. This improves posted1.487399x by23.28%. Target2.6x/rank1 remains unmet.

Promoted exact scored register_ready.py bytes to submission.py, SHA256 53c2fcb96024dbc4c65a95df8252d117ee000285b7948f426e15ba880b29a7a7. Previous posted source preserved as marlin_occupancy.py. No BF16 files touched. Sixscoredjobs total, matching newlyexplicit6cap;20/20freechecksused. No pending jobs or automatic continuation. Next optimization needs further compute/checkquota, and any scored submission needs additional authorization.

## User-requested larger architecture investigation after sixth scored result

User said "go do it" to pursue bottleneck-driven architecture work. Actual CLI checked: MoE20/20freechecks used forUTC2026-10-30,1scoredtoday (6total). NoH100/H200 developmentGPU. NO new remotejob dispatched; other problems/accounts not used. Posted submission.py1.833725x remains unchanged, SHA25653c2fcb96024dbc4c65a95df8252d117ee000285b7948f426e15ba880b29a7a7. BF16 folder/root summary preserved.

Profile evidence: register_dispatch occupies92.7–96.6% of GPU kernel time forM8..256. M32weakest1.448x,141.60uspaired time; M1already2.232x. This establishes projection dominance, NOT DRAM saturation or a particular stall category. No hardware counters available from CLI profiles. Do not describe a measured bandwidth/decoder bottleneck without counters.

Code inspection: original activation helper cooperatively gathersX usingLXG, then calls TWO CTAbarriers per256-Kstep aroundsharedstorage, followed bysharedoperandloads. K4096 means32barriers/CTA/projection. Activation gathering and exchange repeat acrossoutput-column CTAs. W2intermediates were route-pairordered, requiringindirectMapreads toreloadthem. Direct layout LX8/LX/LX32 alreadypermitsreplicatedwarp loads intoBoperand form; hardwareperformance tradeoff unknown.

Implemented TWO SELF-CONTAINED experimental files:
- register_activation_pipeline.py: mode1directactivationloadintoexistingfragmentlayout; mode2also writesW13BF16intermediatesincompactexpert order and readsW2rowslinearly, retainingMaponlyforfinalrouteoutput. New Starts=exclusiveprefix ofactualroutecounts; noinputderivedvaluesinpreparedstate. Eliminates BOTH originalactivationbarriers; no transientsharedactivationsinnewpath. AllowsBM8/16/32 includingmediumdecode, wherepriorcodeclampedBM8. Largerhotexpertitems can reuseeachdecodedweighttileacrossmoretokens. Direct loadsreplicateacrossfourwarps: mayincreaseglobalrequestcount, so this is ahypothesis, not a win.
- register_warp_pipeline.py: includesdirect/compactmodes plusmode3. Producer4warpsload/decodeNVFP4tiles;consumer4warpsloadactivationfragmentsandperformBF16MMA withFP32accumulation. Two32KiBshareddecodedweightbuffers, empty/readyphasebarriers;producerandconsumerpartitionbarriersbeforepublishing/reusingbuffers. NVFP4weightsremainlossless4bit+originalscales inprepare; decodedsharedweights existONLYinsidetimedkernel. Producer/consumeroverlap mayhideconversionlatency but adds64KiBsharedstorageanddecodedsharedtraffic, and canreduceoccupancy. NoTMAimplementedinthisfirstpipeline: loading+decode overlapisbeingisolatedbeforeintroducinganothermechanism.

BothkeepAST-identicalbasefunctiondefinitionsandM1path. preparecomparesnewpathsagainstcheckedbaseonTWOindependentfreshsyntheticrouting/activationdatasets, cold-L2fullgraphs. Checksfiniteglobalerror<=.004 androwerror<=.02beforetiming; requires>1.5%measuredimprovementovercurrentwinner; checkedbasefallbackremainsavailable. AllnewpathsareGPU-UNCOMPILED. CaughtcompilerfailureswillbesurfacedasACTIVATION_SKIP; a passedcheckwithallbasefallbacksISNOTanarchitecturalwin. Unseenshapesusethecheckedbasependingnewpathvalidation. Totalpreparebudget220sversus300slimit.

CPU verifier verify_activation_pipeline_cpu.py PASSES: executesACTUALprefix-scanandrow-addresshelperbodiesusingNumPyshim;9Mvalues(1,4,8,16,32,33,128,256,257)x4distributions(normal,hot,empty,local)x3tilewidths(8,16,32), foreachcandidate. Confirmscompactstoragebounds/exactroutecoverage andstagedBF16projectionoutputs; physical/logicalKbijectionandfour-warpfragmentreplication; enumeratesallorderedproducer/consumerstepinterleavingsfor1,2,3,4,16,17,32,64Ksteps, checkingphaseprogress/nooverwrite/abstractnodeadlock. InitialCPUoracleusedBLASwithdifferentbatchgroupingandhitoneBF16roundingboundary; changedtheoracle'sW2reductiontofixed-orderNumPyeinsumtoisolateaddresssemanticsratherthanBLASalgorithmvariance. This is NOTGPUfragment/synchronizationvalidation.

profile_activation_pipeline_gpu.py is anexternalHopperdevelopmentrunner, NEVERpostsCLIjobs. Forcesmode/directcompactwarpchoicesagainstsamecheckedcontrol,CUDAgraphfresh-inputnormal/skew/empty-route replaycorrectness, pairedcold-L2timing, outputsJSON+experttrafficrequestmodel. DocumentsNsightComputefull-countercapture forprojectionkernels. RequiresrealHopper; noGPUavailablehere. Syntaxand--helpverified. Trafficmodelisrequestedbytes,NOTmeasuredDRAMbytes;L2reusemustbemeasured.

Primary technical sources consulted:
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html — occupancy, TMA, producer/consumerwarp specialization; mechanism inspiration, no assumptionitwillwin.
https://pytorch.org/blog/accelerating-moe-model/ — groupedMoE locality/splitK; itsreported4xnumbersDONOTapplytoourchallenge.
https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/08-warp-specialization.py — explicitHopper+support,partitionAPI,mbarrierphaseprotocol.
https://github.com/triton-lang/triton/blob/main/python/triton/experimental/gluon/language/_core.py — sharedmemorydescriptor/warp_specialize APIs.
https://github.com/triton-lang/triton/blob/main/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp — verifiedcurrentcompilerrewritesbarriersinsidepartitionsintocountedpartitionbarriers. Judge'spinnedcompilercompatibilityisSTILLUNVERIFIED.

Next afterrealquotareset: FIRSTcheckregister_activation_pipeline.py,inspectACTIVATION_TRIAL/SELECTandprofiles,thencheckwarpfileiffirstcorrectness/fragmentlayoutworks. Donotspendchecksretuningunrelatedparameters. GPUrace/protocolmustbesoakedonfreshreplaysandnearby/skewedshapes. Nofurtherscoredsubmissionallowedwithoutspecificnewauthorization. Noautomaticwakeupcreated. Target2.8x/rank1unmet; currentbestscoredremains1.833725x.

User explicitly authorized ONE further scored submission (seventhtotal). Submitted direct-activation candidate register_activation_pipeline.py once, job dd92af663df743ce. Warp candidate not dispatched. Main scored1.833725x stays unchanged pending outcome. Cap7fullyspent.

## Seventh scored result: direct activation architecture rejected by measurement

Jobdd92af663df743ce PASSES1.831370341694092x,CI[1.8038997865686808,1.8588408968195034],prepare27.805s. Crucially all30ACTIVATION_TRIALconfigurations compiled/executed and matched checkedcontrolEXACTLY onbothfreshpreparedatasets (global/rowerrors0). Everyshape selectedmode0/control. Thus this is NOT a newarchitectureperformancegain; officialtimedcases/correctnesstestsexecutedretainedbasepath.

Indicativecoldpreparetimes: M8direct94.2us/compact94.8us versuscontrol86.5us; M16bestdirect117.5us versus112.6us; M32bestdirect182.3us versus134.9us; M128bestdirect500.8us versus234.9us; M256bestdirect797.5us versus291.7us. LargerBMdidnotrescueit. Compactexpertorderingaddedlittlebenefit. RemovingactivationCTAbarriersdidNOToffsetcostsofreplicatedwarpglobalgathers. This negatesbarrierremovalasastandalonewin, but DOESNOTmeasurehowmuchtimewasbarrierstallsorproveDRAM/decoderlimits. Needhardwarecounterstoidentifycause.

Main submission.py preservedexactprevioussixthscored1.833725xbytes; no promotionofnoisy/slowerentry. Liveboard {"rank": 508, "team": "zimo0110", "speedup": 1.8337, "first": false, "ahead": 507, "to_first": 1.4722} rankedcount1149. Sevenofsevenuserauthorizedscoredtotalused;thisday2scoredused;nootherjobdispatched.Warp-specializedfileSTILLGPU-unverified.

## Oct30 evening GPU quota reset

User asked continue. Actual CLI20freechecksavailable,10scoredavailable; no pending jobs. First newdayFREEcheck ebc0a420b6ac47e8 for register_warp_pipeline.py. No scored submission dispatched; usercap7spent unchanged. Main1.833725x preserved.

While firstwarpcheckqueued, preparedregister_warp_cooperative.py. ProducercooperativelyloadsactivationswithcheckedLXGlayoutintoa secondtwo-slotsharedringalongside decodedweights; consumerloadscheckedLX2layoutandperformsMMA. Thisremovesfour-warpduplicatedglobalactivationgathersfromfirstwarpvariant, using72/80/96KiBshareddependingBM. Testsmode3onlyagainstcheckedcontrol ratherthanretestingknown-losingdirectvariants. CPUactualhelper/layout/ringmodelchecksallpass; GPUUNVERIFIED. Noadditionaljobdispatchedyet.

FirstwarpFREEcheckebc0a420b6ac47e8passes1.80947x. All15warpconfigurationscompile and matchcontroloutputsEXACTLY onbothpreparedatasets,butalllose: M8~148us,M16~208us,M32~294us,M128~711us,M256~1086us versuscontrolsroughly88/106/139/237/309us. No warp pathselected. M8directmode1selectedinthisrun; othersmode0. Not a warp win. Proceedingcooperativeactivationloadcounterexperiment.

CooperativewarpFREEcheck6125d7c670204921passes1.82458x butall15newpathtrialsloseandselectmode0. Alltrialcontrolerrors0. CooperativeXreduceslargeMwarpoverhead: bestM128424usversuscontrol238us;M256524usversus312us;smallM170/203/330usversus97/114/176us. Thereforedecodedweightsharedstaging+extra warps/synchronizationstilloutweighsoverlap. No warp architecture win; no source promotion. Preparingfocusedload/decode/fullprojectiondiagnosticsinsteadfurtherblindtiling.

Diagnosticregister_projection_probe.py usescheckedkernel_fn unchanged; prepare-onlyprobestreamsallpackedcode/scalevaluesintoachecksum, ordecodesandreducesallvalues, orrunsafullindividualprojection. CollectedonM8/32/256independentroutings. TheseareNOTadditivecostmeasurements:reduction/probecompilerbehavioraddswork, andrequestedbytesarenotDRAMbytesbecauseL2mayserveweightreuse.

Preparedregister_token_reuse.py duringdiagnosticqueue: checkedbaseplusnew64-tokenactivationlayoutsforBK64/128, dynamic8/16/32/64worktailselection. OriginalBK256familyunchanged. Motivation: BM64withBK256replicatedBoperandalonecanrequire~256registers/thread; smallerKreducesBregisterdemandwhileamortizingweightdecodeacrossmoretokens. PreviousexperimentsnevercombinedBM64withsmallK. Newlayoutbasiscoverageand4-warpreplicationCPUverified;FP4packformatunchanged. OnlyM128/256coordinatestage-specificBM64/BK64/128trialsagainstcheckedbase,usingthreefreshpopularitydatasets(.01,.03,.08);requiresfiniteglobalerror<=.004and>1.5%coldgraphgain. GPUUNVERIFIED,notsubmittedyet.

ProjectiondiagnosticFREEcheck976d901d83c34df6passes1.81772x. Fullprojectionmodelsvalid,butload/checksumanddecode/checksumprobesALLcompilefail:rank0reductionsproduceplainfloat32ratherthandistributedtensor;convert_layoutonthereducedscalarinvalid. Fixedbykeepingplainfloat32loopaccumulator. Therefore NO load-only/decode-onlytimingwasobtainedyet. Fullprojectionmeasurements: M32W13requested245.4MB/69.6us=3.53TB/srequestedeffective,W2requested122.7MB/45us=2.72TB/s;M256W13requested717.2MB/199.7us=3.59TB/s,W2requested358.6MB/109.8us=3.27TB/s. M25638token tiles27activeexperts=>1.41xrepeatedweightrequests. TheseareREQUESTbytesnotDRAMthroughput. CombiningcorrecteddiagnosticswithBM64smallKcandidateinonenextcheck.

Fifthfreecheck 640a80a8a24845c4 dispatchedindependentlyforregister_sector_packed.py. Newlosslessintra-tileweightcodewordpacking: oldoffsetr*32+q*8+j -> newoffset(j//4)*1024+r*16+q*4+j%4. SameNVFP4codes,scales,decoderandkernels,withallTHREEBK256wordloadconsumersupdatedincludingM1splitKfirst/W2. Originalscaleformatunchanged;BK64/128banksunchanged. CPUactualpackerdecodedroundtriptestsandactualconsumerASToffsettestspassfor3randommultitilebanks,ordinary/GUrows,bothlocalexperts,exactnoexpansion. Under16-bytevectorloadmodelonewarp2instructionsuses64sectorsold/32new;actualPTXvectorizationandDRAMtrafficunverified. PriorKmajor/XORexperimentschangedtileplacement,notthisintra-tilewordarrangement. Thischeckindependentofpendingtoken-reuse/diagnosticcheck;noSCOREDjobs.

Completedchecks4..10:
4c5048cc1c4846f4 register_reuse_probe.py passes1.82215x; ALL16newBM64/smallKtrialslose,globalerrors~.0002withinbound. Correctedload/decodeprobecompiles. Load+checksumvsfullM32W13~84.6/92.1us,W2~44.2/55.3us;M256W13~129.9/187.2us,W2~74.5/101.5us. Decode+reductionisSLOWERthanfullMMA(~211/119usM32,415/219usM256),becauseconvert_layout+reductionprobeaddswork;cannotextractdecoderlatencybysubtraction. Requestedthroughput>4.8TB/sM256loadprobeconfirmsL2reuse/requestmodel,NOTphysicalDRAMthroughput.
640a80a8a24845c4 sectorpackingpasses1.84373x,CIoverlapscurrentbest;noestablishedendtoendgain.
5b19642fe8274222 warpTMAcompressedpipelinepasses1.81202x;all20TMAtrialscompileandcontrolerrors0,butalllose. Bestcap64: M8~101vs97us;M16~132vs118;M32~165vs144;M128~293vs241;M256~384vs298. cap96worse. Stage4worsethan2. No warpTMAwin.
c4d316db147a4764 packingABFAILShostsetupKeyErrorchoose_packed:basepreparecallsglobalkernel_fnwithrawstatebeforefinalwrapperstateexists. Fixedguardinv2,locallyexecutedactualhostdispatcherwithsentinelscoveringrawstateandbothfinalchoices.
e0ce63e3e3174899 packingABv2passes1.83396x. SameGPU/weights/inputs16alternatingcoldgraphpairspersample,2independentdatasets: onlyM1packingshowsrobust~3.4–3.7%gain. OtherM~0–2%mixed;no2%bothdatasetthreshold,choosescanonical. Not a majoroverallgain.
99d471467eda45c5 single-groupTMApassesfallback1.82067xBUTALLnewtrialsCOMPILER_SKIP;tail-truncatedexceptionhidesrootcause. ThisisNOTaGPUvalidatedsingle-groupTMAimplementation. Needexposefullcompilererror,thenfix.
1880f31e4b86496e four-waypacking/cacheABpasses1.83052x. CanonicalCG(L1bypass)regresses~2–6%;packedCGsmallM1~3–4.6%gain,selected. M16packedCA~3.3–4.1%gainselected;otherM0fallback. Nooverallimprovementvsbest. Cachechangesonly8BK256codewordloads;noSF/Xcachepolicychanges.4implementationsshare2preparedbanks,not4weightcopies. Allcompareerrors0.

Primarysourcememorysectorreference:https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/ . TMAAPIreference:https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/04-tma.py . Sectorpackingisfullylossless;requestmodelconditionaloncompiler16-bytevectorloads,notdirectDRAMcountermeasurement.

## Continued after midnight Oct31;18of20freechecks completed

User explicitly saidcontinue. Actualquota18used,2left;UTC2026-10-31. Last3previouslypendingjobsallDONE. No scoredattemptsusedthisUTCday. Main1.833725xsourcepreserved.

11.066892601bac4305 singleTMAdebugFAILSasintendedtoexposerootcause: staticfor-init/invalidateusedname`slot`,runtimeKloopreused`slot`;Tritonloopcarriedvariablechangedconstexpr/tensor type. Renamedstaticindicesinit_slot/end_slot;checkedASTstatic/runtimebindingsdisjoint.
12.4e0290d2994543a6 scheduledpasses1.83015x;all40largerpersistentgrids4/8/16/32CTAsperSMstagecoordinatesloseoroverlapbaseline;allerrors0. No schedulechoicekept. EmptygridblocksareNOTamajorspeeduptargetinthisimplementation.
13.dff30240292f49c2 singleTMAv2passes1.81347x;newpathNOWcompiles/validates(all30trials),nonekept. BestsmallMcap80~85/105/146usvs84/104/132controls;largeM~318/495vs225/297us. Stages2/3worsethan1. NoTMAwin. M8controlBK128versusnewBK256explainsmodest~.0004roundingdifference;withinchecks.
14.ab287ca0ef3e464d generalizeddecodesplitpasses1.82902x;all18splitK2/4/8trialscompile andoutputerrors~.0001–.0008,butallloseorimprove<1.5%;nonekept. SmallestM8W2KS2~77.8vs78.1uscontrol;notmeaningful. First-stage/w2partialreductionskeptBF16boundariesandFP32partialaccumulators,originalprecision. verify_decode_split_cpu.py actualGUaddressandKcoverage+stagedBF16splitmathchecksPASS. No splitKwinbeyondexistingM1path.
15.eb65d4c3190d44be asyncpackedweightCPpasses1.82412x;all15newpathscompile andmatchcontrolEXACTLY,nonekept. Explicit2Dswizzledsharedrawweightbuffers+same4warpsoutperformdecodedsharedWSbutarestill~7–20%slowerthancontrol. Tailmaskedcopygroupscommittedanddrained;originalNVFP4byteformatpreserved. Not a win.
16.ef7207dbc6f94b20 explicitvectorhintpasses1.84314x. EightBK256codewordloadsreplacedbyaligned16-bytePTXld.global.L1::evict_first.L2::256B.v4.b32;originalcodeword/scaleformatunchangedexceptprevioussectorpacking. Fullcorrectnesspasses,overallwithinnoiseofsector/control. Noestablishedlargegain. Vectorgroupsaligned/contiguousCPUverified.
17.3f483656c8aa492a hybridK64weights/K256activationspasses1.85231xBUTall40hybridtrialslose,nokept. Thus1.85231isthesectorpackedbaseexecuting,NOTahybridarchitecturewin. Smallweightfragmentsreducepotentialregisterlivenessbutdonotpayoffevenwithactivationbarriersamortized256K. Actualmatherrorsfinite~.0002firststage,.00005secondstage,withrowguardsallwithinbound. Newlayoutbasesandfour-warpreplicationCPUcoverageverified;normalK256cooperativegather/reshapeandslice64microdotmathCPUverified.
18.a10a52a3e87f4c53 hybridK128weights/K512activationspasses1.81622x. All40trialscompileandcorrect;M8firstprojectionselected~5.7%preparedatasetgain(85.1->80.2us),M16second~2.1%(100.4->98.4us)butofficialoverallregresses. DoNOTpromoteorextrapolatelocaltunerwin. ActualnewpathresourcesM8GU80regs0spills8KiBshared;M16W264regs4spills8KiBshared. Othersclassicfallback.512gatherjoin/permutation/slice128CPUverified.

Preparedregister_unrolled.py as19thcheck: duplicateonly3BK256Gluonfunctionsforunroll2/4; nestedouterruntimeandinnerstaticloopsvisittheEXACTsameKorder. Commondataformatsectorpacked;M1sectorpathunchanged. Coordinatesstage-specificunrollfactorsonfreshcoldgraphs,checkedbasefallback. AllKloopcoverageCPUassertionspass;GPUcheckpending.

Preparedcleanregister_m1_packed.pyforpotentialfinalconfirmation: allnonM1projectionsuseexistingcanonicalpacking/checkedkernels,onlyM1usesextraexactlosslesssectorpackedcodeswithoriginalscalebanks. ThreeGPUhelperfunctionbodies(pack,firstpartial,W2split)areAST-identicaltopreviouslyGPUcheckedsectorimplementation. Avoidsallunusedexperimentaljit/tuningcode. NotyetGPUcheckedascombinedmodule. Await19resultbeforelastcheckselection.

Researchsourcesforfurtherwork:https://arxiv.org/abs/2301.03598(Stream-K,wavequantization),https://pytorch.org/blog/accelerating-moe-model/(splitK),https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/ld.html(explicitcacheoperators),https://docs.nvidia.com/cuda/parallel-thread-execution/index.html . Neitherpapers'numbersnortheoreticalmodelsguaranteegainsinthischallenge. nvfp4.py wasreadforcompressionfeasibility: originalFP4codesuniformoverall16patterns,noobviouslosslesscodeentropywin;scalescompressiblebutminorweightfraction. No tailoredlossyformat/activationquantizationintroduced.

19.c71325eb74d84e4e unrolledpasses1.858157282x,CI[1.830284923,1.886029641]. All20unrolltrialscompileandmatchcontrolEXACTLY. Unroll2selectedbothprojectionsM8/M16: calibration87.36->80.99us(M8),104.46->97.89us(M16);otherMclassicfallback. Thisisamodestpromisingpercasegain,NOT2.6xandoverallCIoverlapsposted1.8337.
20.d2ca325b52d643a4 register_unrolled_ready.py dispatchedasfinalfreecheck,exactAST-identicalalgorithmwithconfirmationcommentonly. Independentfreshweights/inputstovalidatecorrectnessandstability;allresultswillbereported,notcherry-picked.20/20freechecksnowused;0scoredtoday. Pending.

## Final Oct31 confirmation and handoff

20.d2ca325b52d643a4 register_unrolled_ready.py PASSESall5tests/all6timedcases,1.834284118x,CI[1.806769856,1.861798380]. ExactASTsameas19;independentfreshweight/inputcheck. M16calibration100.91->93.28us,u13=2/u2=4selected;M8notselectedinthisrun,otherMbaseline. OfficialM16case109.97us,1.716xversuspreviouspostedM16109.85us,1.712x(onseparateinputs/runs),sonobroadjudgedatasetgainestablished. Donotclaimcalibration7%asendtoendimprovement. Twofullcheckresults1.85816xand1.83428xoverlappreviousposted1.833725xuncertainty. No publicationmeritedinthisround.

All20newUTCdayfreechecksconsumedandALLjobsdone.18officialpassingruns,2setup/compilerdiagnosticfailures;somepassingrunsreturncheckedfallbackafterexperimentaltrialsloseorfail,asnotedindividually.0SCOREDjobsinthese20checks. Original7/7authorizedscoredtotalspent;actualplatform10scoredleftisNOTnewauthorization. Main submission.py staysbyte-exactpreviousscoredsourceSHA25653c2fcb96024dbc4c65a95df8252d117ee000285b7948f426e15ba880b29a7a7. BF16folder/rootClaude summary untouched.

Candidateforfurtherdevelopment:register_unrolled_ready.py,exactcheckedsha d6833784acfba835a9dccae8ab896f17acca65c8d2be7486aa65f6f22b52f12a. ItisGPU-correctandincludespergeometrycontrolfallback,butnotaconfirmedoverallperformancewin. register_m1_packed.pyiscleanpreparedM1-onlycombination,NOTGPUcheckedasawhole;doNOTpromoteitwithoutcheck. Allbroadwarp/TMA/CP/split/scheduling/hybridpathstestscorrectbutslower;avoidrepeatingthemduringnextsession.

Nextusefulfocus:calibrationgeneralizationanddirecthardwarecounterprofiling. CurrentpreparetunerusesonlytwofixedfreshsyntheticroutingdatasetsperM,whereasjudgechangesroutesbetweenreplays. Localgainsdidnotreliablytransfertojudge. Evaluateconfigurationchoicesonmoreindependentroutingdatasetswithpairedcoldgraphs;excludeunnecessaryspillingconfigurationsonlywithmeasuredsupport. HBM/ALUstallcausehasNOTbeendirectlymeasured;requestedbytesmodelsanddecode+checksumprobesarenotNsightcounters. NoHopperdevelopmentmachineavailable. FurtherGPUchecksrequireactualquotareset00UTC(Oct31at17PDT,nextresetNov1UTC). Nootherproblem/accountusedandnopaidcredits/computeobtained. Noautomaticwakeupcreated;originaltwo-sessionone-shotremainspaused. Target2.8x/rank1notachieved.

Liveboardatfinalread:{"rank": 589, "team": "zimo0110", "speedup": 1.8337, "first": false, "ahead": 588, "to_first": 1.504};leader:{"rank": 1, "team": "dolanzhao", "speedup": 2.7578, "first": true, "ahead": 0};ranked:1225.

## User requested competitor strategy research and architecture replacement

Useddeep-researchskillwiththreeexplicitresearchsubtasksandareportwriter. Researchartifactsunderresearch/research_notes/High scoring MoE strategies/ andfinalreportresearch/reports/High scoring MoE strategies.md . Nootherteamscontacted,stars/posts/messagesnotmade. PublicREADMEinstructionsnotfollowed. Researchallread-onlyexceptnotes/sourcecopies;NOnewGPUjobs.

FoundauditableApache2publicalygg77kernelcommit6cae49a639ee6d14b929a5fd4a63c627f96d7dc0. Downloadedsourcesha256b33070898927b7b92df6c7a2515f9c02dd12723c43ca63682fa164e915dd4281matchesupstreamresults.json,judgejobe1bcad5a55cd4a29,2.4015xCI2.3655–2.4376. ThisisUPSTREAMrecordedresult,notours. Mostarchitecturefromvjeuxf35a41ee1c13e316dabbf772830d47b23b453393,KPERM3~1.25%fromaly. Nolinkablepublicdolanzhao2.7578implementationfound;donotclaimknowleadertechnique.

Criticalarchitecturechange: persistentend-to-endCUDAoperator;16-output-rowfullKskinnyunitswith8warpssplittingK,activationregisterreuseacrossunits,36KiBcompressedweightring;largeMRS-WGMMA4consumerwarpgroups+2producerwarps;costbalancedStreamKboundaryfixupsandcrossphaseprefill. ThisisNOTtheworkdecompositionofourfailedWS/TMAexperiments. AvoiddescribinggenericTMAaloneasmissingstrategy.

Audit: preparedweightsstayrawFP4codewords+originalFP8scales+FP32GS+small256scaleLUT/losslesslayoutcopies;noexpandedweightmatrix. HoweverpublicGUandPBepiloguescarrymoreFP32activationprecisionthanourconservativestagedBF16semantics. Higherprecisionisnotnecessarilyanofficialruleviolation,butnotstageidentical. WecreatedseparatelylabelednumericBF16stageport;doNOTinheritpublic2.4015score.

Filescreated:
- cuda_persistent_control.py exactpublicbytes,originalattributionheaderspreserved.
- cuda_persistent_lean.py removesunreachableBIG=trueinstantiationsfromTINY/TINY2/MIDatall3launcherreferences(attribute,PDLlaunch,normal);MAINretainsboth. All4shapevariantsANDselectedALT_SRCShelper0retained. Mathpreservingbuildpruningnotyetmeasured;authorscoldbuild~273–275ssometimes>300s,lean~208–224s.
- cuda_persistent_bf16.py addsBF16gate/up,SiLUandfinalproductboundary;PBroundsGEMM*GStoBF16thenroundsnormweightedBF16. All5GUand6PBsitespatchedinBOTHCUDA_SRCandhelper0. WGSfoldedGS*normunfolded. ybufscratchstillFP32containerholdingBF16-representablevalues,noteinprovenance. AccumulatorsremainFP32.
- build_persistent_port.py reproducibleassertedpatchgenerator;uniqueextensionnamespreventcontrol/portcachecollision. Source+fullApache2LICENSE/NOTICE+upstreamresultcopy+PROVENANCE.jsoninresearch/alygg77_2_4015/.
- verify_persistent_port_cpu.py PASSES: exactcontrolhash;hostpacking/routing/scratchfunctionsASTunchanged;bothsourceepiloguepatchcoverage;actualC++preprocessingconfirmsTINYnoBIGtrueandMAIN3truereferences;shapevariantselectionboundsandhelper0;scaleguard;actualpackersrandomcode/scalebitroundtripsincludingunit/fat/WGSreorders;actualC++numerichelperscompiledwithCPUbf16shimvsTorch(PB/roundingexact,GUwithinCPUexpdifferencebound).
- verify_persistent_gpu.py externaldeveloperonlyrunner(NEVERsubmits);alternatesmixedshapesandsamepreparedscratchacrossfreshnormal/skew/empty/alllocalinputs,CUDAgraphreplaysandreferencechecks. --helpchecked;GPUexecutionunavailable.

Localtestcorrection: CPUC++shimfirstlineaccidentallycontainedliteralplusbefore#include;fixedtestharnessandreranallchecksPASS. NoGPUportbehaviorchangedforthisfix.

Nextquotaavailablecheckcuda_persistent_lean.pyfirsttocheckcoldcompile,correctness,fullgraphscore;thenstrictnumericBF16port. Needmixed-shapescratch/counterfreshreplayvalidationbeforeanyscoredpublish. Morethanpublic2.4requires~11%furtherlatencyreductionto2.7(about13%tocurrent2.7578);unprovenhypothesisunit-release/fixup/phaseendcosts,notgenericdeeperprefetchrings(upstreamalreadynegative).20/20freechecksremainused,0scoredtoday,7/7authorizedscoredtotalspent. Main1.833725xbytehashpreserved,BF16folder/rootClaudesummaryuntouched. Noautomaticwakeupcreated.

## Nov1 user-requested continuation

ActualCLIquota20checksavailable,0scoredtoday;noactivejobs. Verifiedporthashesandpostedmainhash. DispatchedFIRSTfreecheck 07588fc518f147ff forcuda_persistent_lean.py(public2.4015architecturewithunreachableBIGinstantiationpruningonly). Willinspectcoldcompilebudget/correctnessbeforerunningconservativeBF16port. NoSparkSSHreceived. NoSCOREDauthorizationadded;7/7capspent.

FIRSTpublic-architecturecheck07588fc518f147ff PASSES2.368745263x,CI[2.333214084,2.404276442]. Allofficialtests/timedcasespass,nostaticflags. Thisreproducespublicarchitecturegeneralrangeandmateriallybeatsold1.8337x. Creditvjeux/alygg77;localbuildpruningnotclaimedasruntimeinnovation. NextstrictnumericBF16variant.

StrictnumericBF16check2eb94d8e8c014dd1 PASSES2.364978604x,CI[2.3295039246226508, 2.400453282732985]. All5officialtests+6timedcasespass,staticflagsnone,prepare165.74s. NextphysicalBF16ybufportretainsidenticalBF16numericvalues,FP32fixupaddressesunchanged;prepare-only192mixedshapereplayconsistencychecksadded.

Checks3faed083991ee4a49 physicalBF16scratch+prepare192mixedgraphreplaysand4 35bf911880fe455a pairedphaseAreadyrelease+samestresspending. PairedreleaseonlywholeconsecutivesameCTAunitseven/oddfeedingonesameBstage;preserves128threadnamedbarrierandGPUrelease,increment2insteadoftwo1s;split/nonadjacent/jobboundaryunchanged. CPU80streamcombinationscount/ownership/readinessPASS. PTXprimaryreferencehttps://docs.nvidia.com/cuda/parallel-thread-execution/#release-and-acquire-patterns . No arithmetic/weightchanges,noSCOREDjobs.

Compactcheck3faed083991ee4a49 FAILEDsetupatMIDextensionimport(PyInitexportmissing),beforeanymath/stress/perftest. Causeunproven;v2shorthashedextensionnamesretrycheck6. CUDAstringsunchanged. Check5 4feeba9bd29f4f17 nativegridcalibrationpending:5gridcountsx9shapesx2freshseeds,coldpairedgraphs,3%thresholdplusindependentthirdinputconfirmation2%;192mixedreplays. NoScoredjobs.

Check4batchreleasefailedsameMIDlongnamePyInitimportbeforemath. Check5gridfailedLOCALvalidationbug: CUDAgraphcapturedoesnotexecute;compareduninitializedcapturedcontroloutput. Fixedbyexplicitcontrolreplay+syncbeforeclone;check8 308d3a7410c94120 pending. Check7 44d39ff98d984ff5 SGLangaligned: sourceconfirmedfilteredSiLUusesFP32activationproductwithoutBF16SiLUintermediate;normFP32multiply/reductionwithoutBF16weightedintermediate,originalBF16GEMMboundarieskept;effectiveBF16(1+w)prepareexactlybaseline. ActualinsertedC++helpersCPUtestPASS,bothCUDAstringsallGU/PBsitesguarded. Sources pinnedSGLangcommit a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f activation.cuh and fused_post_expert_norm_weighted_sum_triton.py underpython/sglang/kernels/. No changingrulesprecision: fixesover-roundingfromlocalconservativeport.

CodeauditduringstressfoundupstreamWGSrangesunconditionallyindexesjob0/costevenwhenlocalroutejobcount0. Addedexplicitnj==0allfourrangeszeroearlyreturntothreeSAFEvariants;nonemptybehaviorunchanged. ContentaddressedshortextensionnamesincludeactualCUDAstrings+defines. Check9alignedsafedispatched;earlierstressjobsstillpending,mayexposeemptycaseorotherissue. Needactualresults,donotassumecause.

Check10narrowGS4dispatched:alignedsafe+specializeNTT1onlytoGS4;NTT2..8unchangedGS2becauseupstreamGS4globalserializedthoseatregistercap. Samek16order0..15proven,releasepoint8stepssame,nomathprecisionchanges. Pendingofficialtestsand192replaystress;doNOTclaimgainbeforemeasurement.

Check9a4883618fed74d16 ALIGNED_SAFE PASSES2.370773507xCI[2.3352119041905257, 2.406335109394298];allofficialtests/timedcases+192mixedshapefreshgraphreplayconsistencyPASS. Explicitzerojobguardandshortcontentaddressednameswork. Bestverifiedprivatecandidatecuda_persistent_aligned_safe.py;postedmain1.833725xpreserved. UseraskedasyncpermissionforONEadditionalscoredpublication(8thtotal),PENDING,NOTauthorizedbytimeout. Liveleader2.7642,dolanzhao;postedaccountzimo0110rank6611.8337x.

Check10narrowGS4passesalltests+192replays2.355224984xCI[2.319896609,2.390553359],nomaterialgainvsalignedsafe2.37077;notpromoted. Check11occupancyattemptNS2extraTINYvariant:CUDAactualoccupancyquerygates2CTA/SMgrid,nopersistentoversubscription;3freshinputpairedcoldgraphsrequiremin3%gain;2/3nearbyshapesadded224replaystress. Check12batchreleasesafeand13compactybufsafeaddedwithprovenzerojobguard+contenthashednames. Earlier6/7/8unguardedstressjobsstillrunning,likelyzerojobhang;notclaimdefinitivefailurebeforeactualstatus. Allprivatechecks,0SCOREDtoday;8thpermissionPENDING.

UserexplicitlyapprovedONEadditionalscoredsubmission;cap8total. Verifiedledger7priorjobsandactualokcjobschecks-onlytoday;exactcheckedcandidatehashmatched. Dispatched8thSCOREDjobc913470f59834b10cuda_persistent_aligned_safe.py. 8/8authorizedtotalnowspent;NOmoreSCOREDwithoutnewapproval. Pendingresult.

Checks11occupancyPASSED2.365227474xbutNS2capacity1,149regs,214784shared;ALLnewNS2pathslose3seeds,baselinechosen. Concreteissue:okc_smem_bytesreservesmax(skinny,fat)eveninTINYwherefatunreachable. Check12batchreleasepasses2.357017588x+192replays;13compactBF16scratchpasses2.358727483x+192replays;neitherestablisheswin. Check6unguardedcompactv2eventuallyDONEpassedFalse,noerrorlog(nozerojobguard;guardedversion13passes). Check14 cdcc6752d920472f isCUDA/timed-pathidenticaltosafebestplus24000queuedreplays(sync20mixedcycles)with10freshsets;pending. Check15NS2occupancyv2prunesunreachablefatsharedanduseslaunch_boundsminblocks2forNS<=2only,runtimeoccupancygates2xgrid;pending. Check16v3alsoSHAREDROUTINGarraysbounded8/16tokensinTINYonly;GLOBAL EMAX3072andallstride/resetmathUNCHANGEDforcrossvariantdoublebuffercompatibility. Allvariantskeepbaselineunless3independentfreshinputsall>3%win. Pending.

8thSCORED c913470f59834b10 PASSES2.382249325xCI[2.3465155849992345, 2.417983064745404],sameexact9thprivatechecksources. MAINsubmission.pyupdatedBYTEEXACTtothisscoredcuda_persistent_aligned_safe.py;old1.8337sourcestillregister_ready.py. Liveboard{'rank': 183, 'team': 'zimo0110', 'speedup': 2.3822, 'first': False, 'ahead': 182, 'to_first': 1.1604},leader{'rank': 1, 'team': 'dolanzhao', 'speedup': 2.7642, 'first': True, 'ahead': 0}. All8authorizedscoredspent;nofurtherpublicationallowed. Checks7/8expiredunguardedstress,noscore;8nativegridcalibrationall9Msretained132SM(alllowerGchoicesloseorwithinnoise). Neverpromotenativegridexperiment. Current16freechecksused,4left,0newscoredpermissionremaining.

Check14cdcc6752d920472f SOAKPASSES2.378511185xCI[2.342833517,2.414188852]. SameCUDA/timedpathasscoredsource. 24000queuedgraphreplays,2000pershapeover12shapes,10freshsets,syncevery20mixedcycles;maxrelativeL2drift6.57512e-05.251bitwisedifferingcheckpoints(smallFP32reduction-orderrounding),NOTbitwiseidentical. AllofficialFP32referencetests/timedcasespass. Confirms2.37rangewithfreshdatasetandstrongerreplayvalidation. No newscoredsubmission.

Check15v2NS2capacityNOW2,96regs,110336smembutbothNS2and2xgridlose5–10%on3seeds;baselinechosen;passes2.354375967x. Check16v3sharedrouteboundreducessmem95360,capacity2,96regs,but2xgridstillloses6–9%;baselinechosen;passes2.364213060x. DoNOTclaimoccupancywin. Check17 NW4/NS2experiment: NCH8->16,scale-bankloadgeneralizedbothPRMTandLUTpaths;CPUexact64-K/scalebankcoveragePASS. FirstbuilderassertcaughttwoSFselectorsnotone;fixedbothbeforevalidfile/GPUdispatch;CLIattemptnonexistentfilefailedlocally(noquota/job). Check18M1distributedactivationexperiment: keep8activationregs/threadvs64,warp-lanedistKand8shufflesperchunkreconstructexactBfragments;fixedshapeM1onlycompiledassert,duplicateM1routessamex/intervalid.CPU8warpsx8chunksx32lanesx1..8duplicatesbitidentityPASS. SameBF16tensorboundaries+FP32accumulators+NVFP4weights,noarithmeticapproximation. PendingGPUtrials.18privatechecksused,2left;all8scoredused.

Check17NW4capacity2,168regs,86656smem;strictCONTROLcomparison0.001104>local0.001thresholdrejectedBEFOREofficialmath/timing. ThisdoesNOTestablishofficial1.5%referencefailure;changingFP32Kpartitionreorderssums. Check18M1distributedpasses2.372611269xbutnewNS2and2xpathslose4–7%on3seedsandarenotselected. Fallbackran;noarchitecturalperformancewin. Check19 2b6f99f53e394ee9 isolatesM1distributionwithoriginalNS3/uncappedregisterbudget. Check20 20fc1b8977664fae FORCEnewNW4pathforM<=4throughofficialreference/timing;localCONTROLpretest0.3%(stillstricterthanofficial1.5%,notprecisionchange),specpreservingFP32reduceandallBF16/NVFP4boundaries. Final2privatechecksPENDING.20/20dailyprivatechecksused. No other problems/accounts used;all8SCOREDauthorizedspent. Nextreset00UTCNov2=4pmPSTNov1(DSTended). AskedSparkSSHcommand;configcontainsnorelevantalias,noSSHattemptmade. Postedmainexact2.382249xsourcepreserved.


## November 1 final verified handoff

Published **2.382249325×** (CI 2.346515585–2.417983065), job **c913470f59834b10**.
Account **zimo0110**, live rank **185** at the final read; leader **dolanzhao 2.7642×**.
Main `submission.py` is byte-identical to the scored `cuda_persistent_aligned_safe.py`.
The old 1.833725× source remains unchanged in `register_ready.py`. BF16 work and
root Claude summary were not modified. Source attribution is vjeux/alygg77 Apache-2.0;
our work reproduced the architecture, aligned precision, shortened builds, added
an empty-job guard, and validated correctness. Do not claim the whole kernel is ours.

Check 19, 2b6f99f53e394ee9: uncapped NS3 M1 distributed activations passes2.357980023×,
but all three paired trial ratios fail the3% improvement requirement (0.9743,
0.9844,1.0024); original implementation selected. Resource report96regs,132224shared,
capacity1. Do not attribute the overall score to the rejected new implementation.

Check 20, 20fc1b8977664fae: FORCED NW4 implementation passes all official tests and
timed cases at2.284050881×, CI2.249790118–2.318311644, plus224mixedgraphreplays.
New implementation was actually exercised at M<=4. It is slower: three-input
NS2x2/control ratios about0.90. Thus the earlier0.1104% control deviation was not
an official reference failure; FP32 K-partition changes were within task bounds.
Do not promote this candidate or relax the official1.5% reference check.

All20private jobs are complete (13passing,7setup/validation/timeout failures).
No running MoE jobs remain. All8user-authorized scored attempts are spent; only
one scored job was dispatched today. Platform has9scored slots left today, but
that is NOT user authorization. Free-check quota20/20used. Next reset00UTCNov2
is4p.m.PSTNov1. No other account/problem used; no automation or paid compute added.

The exact published GPU algorithm also passed an independent2.378511185× check
with24000queuedreplays,12shapes,10freshinputsets; maximum self-consistency relative
L2 drift6.57512e-5.251checkpoints differed bitwise, so do not claim bitwise determinism.

Best code, profiles, immutable upstream source/license, replay tests, source
builders, complete ledger and readable CUDA review copies are saved. See
NOV1_HANDOFF.md. Goal2.7×/rank1 is not achieved. Further H200 optimization needs
quota reset or a development H200. User has a DGX Spark but has not supplied an
SSH command; local SSH config contains no Spark alias. Spark can support portable
experiments, but not replace Hopper WGMMA validation or H200 performance checks.


## Spark access and offline Hopper development

User supplied SSH_NAME/SSH_PASS variable names and explicitly required no
credential leaks. Located the existing dotenv file inside the BF16 directory;
read only those connection variables through a redacting helper, never printed
values or copied the dotenv file. Password authentication and per-command sudo
work. Connection arguments/passwords are not embedded in generated source.
No BF16 file was changed. Do not cat credential files in future work.

GB10 host initially had no PyTorch/nvcc and no Docker images. Existing DockerCDI
and NVIDIA toolkit expose GPU access without daemon/group changes. Downloaded
NVIDIA's documented25.11PyTorch container; CUDA13.0/PyTorch2.10 BF16matmul smoke
passed. Development code and LICENSE/NOTICE were uploaded via an explicit file
whitelist to /var/tmp/codex-muse-spark/src. All containers have finished/removed;
compiled caches, source and public-image layers remain for reuse.

Native tiny development port passed24functional cases and192graph replays onGB10.
Crucially, Spark's nvcc can compile sm_90a offline:11existingHopper configurations
compiled, register counts matched realH200 data. Confirmed149regs/no spills for
postedtiny; cappedtiny96regs has112Bstores/48Bloads; distributedM196regs hasno
spills; forcedNW4has72B/72Bspills; narrowGS4main144B/308Bspills. Static resource
reports are not H200 performance measurements. Results and instructions saved in
research/spark/README.md and matchingJSON artifacts.

Prepared a NEW larger-batch architectural candidate cuda_persistent_two_wg.py:
2physical compute warpgroups each handle2logical64-row tiles, retaining4logical
tiles and losslessNVFP4 layouts. Coalesced per-group publication/fixup counters,
double gather/final chunks, larger register budget. Pythonhostcode/precision
boundaries/global workspace layout preserved; mainNW16->8. CPU coverage proofs
pass. Offline sm_90a compilePASS at168regs with8Bpointer spill outsideKloop;
refactored4WGcontrol96regs/no spills; all-GS2variant spills20/16B and isnotpreferred.

To validate math onGB10, implemented separate synchronous warp-MMA emulation,
1slot rings and synchronousxretirement. Initial emulatorCONTROL failed due to
incorrect contiguousBfragment loads; fixed to the standard2t/8+2t pairs. Then
published4WG, refactored4WG and new2WG each PASSED44functional cases and352mixed
graph replays includingM385chunking, skewed/all-local/empty routing. FP32reference
errors approximately0.0045–0.005, safelywithin0.015. This is NOT realHopperWGMMA
validation and NOT a newH200score. Hopperasync correctness/performance remains
UNVERIFIED. Nextfreecheckafterreset:cuda_persistent_two_wg.py.

Published submission.py remains byte-exact2.382249325x scored source. No newokc
jobs or scored submissions in thisSpark continuation.20/20privatechecks and8/8
scored authorizations remainused. No automation/wakeup added. No credentials
appear in these notes, artifacts or generated code.


## Nov2UTC / Nov1after4pmPacific continuation

Useraskedcontinueanduseverificationswisely. ActualCLIresetnow10privatechecks/dayand5scored/day,0usedbeforethischeck;old20/10limitschanged. Nootherpendingjobs. Verifiedpostedhash3eba8d45...andnewtwoWGhash71c2f6d0...matchdurablenotes. DispatchedfirstrealH200twoWGcheck a8b06a9748b54177. ONLYprivatecheck;all8user-authorizedscoredalreadyspent. Preserveposted2.382249xandBF16work. Awaitcorrectness+per-shapetimingbeforenextGPUcheck.

## Additional Spark validation while first H200 check waits

The initial two-warpgroup H200 check remains QUEUED, not running. Used only
1/10 private checks on Nov2 UTC; nine remain. No new scored submission.

Prepared striped final-output stores for two warpgroups and a speculative
one-warpgroup version (four logical tiles per physical group). Each passed
44 independent FP32-reference functional cases and352 mixed graph replays under
the synchronous GB10 emulator. This does not establish Hopper asynchronous
correctness or H200 performance. Offline sm_90a compilation passes both:
two-group168regs,8/8B cold-pointer spills; one-group255regs,64/56B spills.
The one-group version is lower priority because its register pressure is high.

Reports: research/spark/striped_hopper_result.json and
research/spark/striped_emulation_result.json. Builders and candidate sources
saved. Published submission.py remains byte-identical to the scored2.382249x
source. No BF16 source or credential file changed.

Prepared cuda_persistent_two_wg_paired.py (not dispatched) via
tools/build_two_wg_paired.py. It embeds the exact published MAIN CUDA source and
flags as a control, compares with striped2WG on the same GPU, alternates graph
replay order, flushes L2 before each side, and uses three fresh routing seeds at
M64/128/192/256. Additional all-local/empty/skewed eager agreement checks precede
measurements. Require >3% aggregate gain, >2% per-seed gain, and no median shape
regression over3%; otherwise use posted control. Only kernel choice survives in
state, never diagnostic inputs/outputs. This calibration is H200-UNVERIFIED and
adds one compile, so inspect first job's prepare budget before dispatching.
CPU AST checks confirm exact control, flags, lossless packers and workspace.

Initial H200 check a8b06a9748b54177 completed: all5official tests and192mixed graph replays PASS, prepare157.13s, speedup2.228408x CI[2.194982,2.261834]. Slower than published2.382249x; do not promote. One private check used, nine remain. Paired comparison ready but not dispatched.

User requested further optimization. Dispatched second private check c71dd6b8a13c49f8: paired striped2WG versus exact published4WG control, three seeds, M64/128/192/256. Initial twoWG regressions include M32 because MID_MAX_M is16. If trial wins calibration, M32/nearby small batches need explicit paired review before promotion. Two private checks used, eight remain; no scored submission.

Independent third private check6baed3dac7b84b8e tests MID versus posted MAIN
on three fresh seeds at M17/24/32/40/48. No extra device compilation: all four
original CUDA/flag/workspace assignments byte-identical by AST. Only extend a
contiguous MID range when every sampled size improves by median>3% and no seed
regresses>2%. Maximum48 keeps worst-case split job count<=76 within88slots.
Control agreement includes all-local/empty/skewed routes before paired cold-L2
graph timing; final chosen path gets192replay stress and official tests.
Three checks used, seven remain. Both pending jobs are private, no scored job.

Two-WG slowdown hypothesis: halving compute warps doubles per-thread tiles,
raises registers96->168 and makes more work serial per warp; code reduction
does not establish lower latency. Both layouts fit one CTA/SM, so do not claim
a measured drop in resident CTA occupancy. Need same-GPU paired result. NVIDIA
CUTLASS efficient_gemm and WGMMA programming guides confirm the resource/latency
tradeoff in general, not this kernel's particular bottleneck.

Prepared a separate routing-aware weight-cache hypothesis. WGS producer detects
when the next job uses the same expert, retaining that job's weights at the
same evict-last policy already used elsewhere; streaming weights retain original
evict-first. Only cache hint changes, not addresses, math, synchronization,
weight precision, scheduling or number of copies. Policy lookup hoisted to job
changes. NVIDIA PTX documents cache priority as a performance hint only, with
no memory-consistency change. Primary source:
https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints

Offline sm_90a compile PASS96regs/no spills, matching posted MAIN. No H200
execution yet and no speed claim. Saved cuda_persistent_reuse_cache.py, paired
version (same-GPU control, M32/64/128/192/256) and builders in tools/. Report in
research/spark/reuse_cache_hopper_result.json. Do not dispatch until inspecting
the two pending checks; seven daily private checks remain.

Reviewed upstream measured MoE facts/negative-results: unit boundaries cost
1–7us; deeper rings and whole-unit dynamic claims did not help that upstream
base. These are upstream measurements, not our own hardware counters. Prepared
independent readback prefetch prototypes with4/8float4 loads instead of2, to
reduce serial L2 round trips at split-unit ends. Offline sm90a: all-phase4
spills76/180B, all-phase8 spills400/732B; PB-only4spills120/288B; PA-only4
spills24/24B. All remain96regs; no C751 serialization warning. No H200check
spent on these. PA-only4 has just a few local-load/store sites, but is still
unverified; started bounded synchronous Spark emulation to validate mapping/math.
Builder:tools/build_fixup_prefetch.py. Phase-specific prototypes built by the
local probe, saved source files and exact device packages retained on Spark.
Reports: research/spark/fixup_hopper_result.json and fixup_phase_hopper_result.json.

Live board Nov2UTC: zimo0110 rank190,2.3822x; leader dolanzhao2.7642x. Public
leader job cannot be read using our CLI account (no such job); no attempt to
bypass access. No rank-one claim. Both own paired private checks remain queued.

PA-only4 readback emulation PASS44independent-reference cases +352mixedgraph replays onSpark. Does not test Hopper async retirement or performance. Saved research/spark/fixup_pa_emulation_result.json. Generic paired builder tools/build_reuse_cache_paired.py now also builds cuda_persistent_fixup_pa4_paired.py, ready but NOTdispatched. Prefetch builder reproduces exact4offline-compiled sources.

Second H200 private check c71dd6b8a13c49f8 PASSES all correctness and192replays.
Paired striped2WG versus exact posted4WG: all12comparisons (three seeds x four
shapes) favor posted. Control/trial geomean0.919132 implies trial8.80%slower;
individual ratios0.8835..0.9597. Every seed geomean loses. FALLBACK SELECTED.
Official2.359770x is the posted control, NOT a gain from the new path.
Abandon reduced-warpgroup approach; retain4WG for next controlled cache and
readback experiments. Current setup's extra compile+calibration43.48s fits the
300sprepare budget. Three checks used, seven remain before newdispatches.

After paired evidence rejects2WG, dispatched two independent4WG experiments: cache retention70c9c2ddb2994e49 and PA-only readback4 b415be1c26c046ed. Each embeds exact posted control and uses alternating cold-graph comparisons on3freshseeds at M32/64/128/192/256. Setup199.44s for prior paired run leaves adequate margin for one extra compile. Five private checks nowused, five remain; no scored submission. MID-range check6baed3dac7b84b8e stillpending. No more dispatches until these outcomes.

MID-range H200 check6baed3dac7b84b8e completed PASS official5tests+192replays.
Across3seeds, MID improves M17 by5.85–8.62% and M24by3.34–8.63%. M32mixed
(-4.48..+1.43%), M40/48 also inconsistent. Selected MID through24only.
Official2.358729x changes no scored shape and is within normal baseline noise.
Do not claim a new leaderboard gain. Preserve the useful nearby-batch finding
for future final kernel; published main remains byte-exact2.382249x.

More substantive device-only probe: a bounded M<=32 two-CTA-per-SM design.
The failed2WG design had one resident CTA with fewer compute warps. Try two
smaller CTAs (same total four physical compute groups/SM), with MMAX32, FT_T32,
two weight/x slots,320fixup slots for264gridCTAs, a112KiB smem static_assert,
and launch_bounds min2. General source cannot be submitted directly: needs
separate scratch, explicit M<=32 dispatch, actual residency guard and doubled
grid. Builder tools/build_two_cta_small.py; offline compile pending. No new
H200 check used; two private4WG experiments remain queued, fivechecks left.

Bounded two-CTA small32 candidate is ready and dispatched as private check
590fe197f7ab4b41. Original small GS2 compile fits112KiB/block staticassert,96regs
with68/92B spills; GS1 worsens to604/752B, half-code-prefetch probe328/356B,
so those are NOTchecked. Full exact small module TorchC++/CUDA link/import
compiledsm90a onSpark ARM in20.97s; noHopper execution. Synchronous Spark
emulation PASS36FP32reference cases+288mixed graphreplays atM1/4/8/16/17/24/25/31/32
and4routingregimes. This doesnotvalidate two-residentCTA Hopper execution.

Builder tools/build_small32_paired.py adds separate exact bounded scratch and
actual CUDAoccupancy guard>=2 before grid2*SM. CPP host rejects M>32, source
prunes NTT>4 cases;320fixup slots bound doubled grid264. CPU mocked dispatch
checks11sizes and occupancy<2 rejection before launches. H200 calibration uses
three fresh seeds atM17/24/25/31/32, baselineMID through24 and postedMAIN after24,
normal/all-local/empty/skewed correctness, cold paired graphs. It selects a
whole contiguous interval17..32 or25..32 onlyif robust>3%gain, elsecontrol.

Six private checks used, four remain. Cache andPAreadback stillpending. No
further dispatch until outcomes. No scored authorization beyond8spent; published
submission.py hash unchanged. Builders/deviceprobes/emulation/compiler results
are durable in tools/ and research/spark/.

Cache-retention check70c9c2ddb2994e49 completed PASS.15paired comparisons
(3seeds x M32/64/128/192/256) have geomean1.002350, seedmeans
0.999148/1.003534/1.004376. No meaningful gain; baseline selected. Official
2.354182x belongs to the baseline fallback, NOT cache-policy improvement.
All5officialtests+192replays pass. Do not infer that memory is unimportant;
this establishes only that this eviction-priority change didnot materially help.
No more cache-hint tweaks planned without new evidence. Two checks pending:
PAreadback b415be1c26c046ed and bounded2CTA590fe197f7ab4b41.6checksused,4left.

PA-only readback4 check b415be1c26c046ed PASSES officialtests+192replays,
but loses all15paired cases: geomean0.989221, seedmeans0.989539/0.988820/0.989303.
Baseline selected. Official2.341610x is baseline fallback, NOT new-path gain.
Reject wider readback. No further related H200 variants planned.

Started a genuinely different native-warp pilot for M<=16: independent one-warp
16row MMA tiles,4/6stage2304B TMA rings, three graph kernels (GEMM1+SwiGLU,
splitK2 GEMM2, full FP32norm/weighted sum). GPU elected routing producer computes
fresh routing once per call; finalization resets readiness includingemptycalls.
No global cooperative-grid barrier, so arbitrary CTA waves are safe. Goal is
more independent memory streams and avoiding cross-unit fixups, at the cost
of extra kernel launches and replicated activation requests. Performance is
UNVERIFIED; prior direct-gather losses are a relevant risk, not proof eitherway.

Reuses credited Apache2DQ7/KPERM bit layout; new warp re-layout preserves exactly
2359296unique byte positions in actual-repacker CPU roundtrip. No dequantized
weights cached; FP8 scales/NVFP4 codes remain lossless. Natural inter feature
columns restored by inverse row map; all SGLang BF16boundaries retained, norm
FP32. tools/warp_moe_device.cu, warp_moe_port.py, spark_validate_native_warp.py
saved. Spark validation started, no H200check for thispilot yet. Only bounded
small32 H200job590fe197f7ab4b41 remains pending;6checksused,4left.

Native warp pilot corrections/verification: fixed C++unsigned-size narrowing
offline.6stagevariant exposed a true mismatch on duplicate-routing inputs
versus4stagecontrol. A register compiler-witness alone didnot resolve it.
Changed TMAbarriers to initialize once per CTA and carry continuous slot/phase
sequence acrossjobs instead of repeated invalidation/reinitialization. CPU
phase model passes7ring sizes x8/16stagejobs x100jobs. Corrected actual native
code then PASSED210independentFP32reference cases and840mixed CUDAgraph replays
onGB10 across3variants ((8,4),(8,6),(16,4)),9Mvalues1..16,normal/skew/alllocal/
duplicates/empty,2freshsame-addressinputepochs. Maxintervariant error0.0.

24additional comparisons to existing checked-math Spark control PASSED with
maxrelativeL20.000958147 (legalKreduction/BF16rounding differences). Runtime
control agreement in newpaired source uses0.003; same-native eager/graph
consistency stays0.001; officialFP32judge bound unchanged at0.015. All3full
TorchC++/CUDAsources compile/link/importsm90a offline in18–19s each. Device
sourcehash matched remote validatedsource; reports saved research/spark/.

Dispatched seventh private check acffa48d276d4631 for cuda_native_warp_paired.py.
Robust cold-graph paired timing:3freshseeds,M1/2/4/5/7/8/9/12/16; baseline
comparison and validroutingextremes, then bucket selection with>3%gain and
per-seed/per-shapeguards, followed192mixedshape stress and officialtests.
One self-containedfile; no developer imports/files/network in candidate.
Published2.382249xsource and BF16work untouched.7privatechecksused,3remain.
No scored submission; all8user-authorized scored attempts alreadyspent.

Small32diagnostic bug fixed in builder: module now namedsmall_module, avoiding
shadowing by floatoutputtensor. Exactfailed checkedsource archived before
rebuild. FirstH200sample M17 showedtrial17.0%higherlatency; no repeatcheck
spent on thisbranch while nativewarp architecture is beingmeasured.

While nativev1 H200check isqueued, prepared v2local-only read-check election:
CTAs load readiness before attemptingCAS, reducing unnecessary same-address
atomics after routing starts. Also calibrates8versus16workers/SM without any
extra device builds. GPUarithmetic/layout/phasecarry unchanged. Archived v1
source, uploaded file and allpreflight reports in research/spark/native_warp_v1/.
Frozen cuda_native_warp_paired.py stillmatches uploaded SHA. Builder default
nowwrites cuda_native_warp_readcheck_paired.py toavoid overwriting pendingv1.
Spark v2validation running across both gridmultipliers; NO H200v2dispatch.
Seven checks used,threeleft; wait forv1result before considering nextcheck.

Nativev1 H200checkacffa48d276d4631 completed withoutofficialscore. Resources
N8NS4[active22/22,72regs,9344shared],N8NS6[15/15,72regs,13952shared],
N16NS4[20/20,96regs,9344shared]. Firstseed M1/2/4timings alllose: ratios
0.606..0.700. Setupthenstops atM5alllocal,N8NS4: native-controlrelativeL2
0.008647 >debugbound0.003. Needindependentreference beforeclassifying which
sideislessaccurate; donotloosencheckblindly. Noofficialtests/timingcompleted.
ExpandedSpark comparison now includesM5/7/9/12 and comparesbothimplementations
toFP32reference. v2read-checkelection passes420referencecases1680replays with
maxintervariantdifference0 butremainsNOTdispatched.7privateused,3remain.

ExpandedSpark independent comparison54cases M1/2/4/5/7/8/9/12/16 confirms
M5alllocal native/control difference only0.00051–0.00073; bothFP32errors~0.0045.
Maxcontrolagreement0.001233 elsewhere. TheH200M5 discrepancy0.008647 isnot
explained bythatportabletest, so doNOTblindlyraiseagreementthreshold.

Built independent FP32CUDA diagnostic reference from originalNVFP4nibbles,
FP8blockscales,FP32global scales, fullFP32SwiGLU and fullFP32norm. Scalar
weights decodedinsideCUDAkernels; NOdenseFP32/BF16weights stored. Temporary
FP32activations/outputs existonlyforourgeneratedpreparationdiagnostics and
are discarded, never returnedinstate orusedbytimedkernel_fn. Validated against
suppliedFP32reference on12Sparkcases,M1/5/8/16normal/skew/alllocal: PASS with
maxrelativeerror5.385973e-5, farbelow0.001cross-referencebound.

Nativev2 check afc9fe19c82c4eb0 dispatched. Usesread-before-CAS election and
8/16workersperSM, reportsnormalabsolutetimes andemptystartuptimes. Same-native
repeatedcall/replayconsistency stays0.001. Ifcomparisonwitholdcontrol exceeds
0.003, independentFP32CUDAreference decides using actualtaskbounds0.015global
and0.075worstrow; botherrorslogged. Neveracceptmerelybylooseningaccuracy.
Includesallmathematicalstages at requiredprecision; timedpathstaysonlynative
three-kerneloperatororcheckedbaseline.8checksused,2remain;noscoredsubmission.

Nativev2 H200 afc9fe19c82c4eb0 FAILED repeated-call consistency M4/all-local N8NS4 workers16. All recorded cold-graph ratios0.573–0.703, substantially slower than postedcontrol. No officialscore. Spark correctness doesnot establish Hoppercorrectness. Park nativewarp/K512packet branch ratherthan spendingremaining2checks on marginal tweaks of a slow unverifiedarchitecture. Publishedsource unchanged. Fullresult research/spark/native_warp_v2_h200_result.json.

New disjoint FP32 split-K partial banks: every contributor writes exclusivebank indexedbyKstart, packed releasecounter tracks stagecount+bankmask. Finisher sums live banks in Korder; no atomicFP32dataadds or zeroing. Additional~300MiBworkspace, NVFP4/BF16weights unchanged. Informed by NVIDIA CUTLASS sm90_tile_scheduler_stream_k.hpp separate-partialworkspace mechanism. Initial offline2tileburst spilled28/84B; retained1tileburst96regs0spills. Preflight caught/fixed hostdoneAsoldoffsetalias beforeanyH200check. CPUall32768partitions/65536arrivalorderspass; Spark44FP32referencecases+352mixedgraphreplayspass; sm90aactualdevicecompile96regs0spills. Pairedcandidate uses INDEPENDENT scratch because originalflags alias trialpayload. Threefreshseeds xM32/64/128/192/256 coldalternatinggraphs; selectonly>3%GM andperseed/shape guards. Dispatched privatecheck e706a69b83644876;9used1left. Publishedsourceunchanged;noscoredsubmission.

While widerbankcheckqueued, compactedexclusiveworkspace from16banksperboundaryslot to2banksperCTA(head/tail). ContributorsareconsecutiveCTAs; packedmasknowindexesCTA-relativepeer, readsKorderunchanged. Thisrequires+20MiBoveroriginal versus+300MiBforfirstpilot. CPU4800jobschedules/383415splitunitsproveexclusiveaddresses/nonemptyranges/<=16contributors for48/132SM grids,jobcounts1..88,tokenjobs1..64. Actual compactsource passes44independentFP32referencecases+352graphreplays onSpark; sm90adevicecompile96regs0spills. Pairedfile cuda_persistent_compact_private_fixup_paired.py prepared,NOTdispatched. Await e706a69b83644876 beforefinalcheck. Technicalsource https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp describes separatepartialworkspaces; ourpacking/compactmappingisownexperiment.

Compact v3 additionally initializes reductionfrom firstknowncontributor insteadzero+add, and explicitly ordersgenericsharedreads beforeTMAoverwrite: all-lanecompilerwitness+fence.proxy.async.shared::cta+syncwarp beforewarp-leaderemptybarrierarrive. BasedonNVIDIA PTX asyncproxy rules https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-copying . Independentpublicfailure-report https://guild.1nterface.ai/t/42 describes analogousreplayonlyfence andselfzeroingatomic hazards; NOT proof ofourHopperrootcause. Correctedv3actualsource againSpark44FP32cases+352graphreplaysPASS, Hopper96regs0spills; reportsupdated,previousv1/v2archived. Compactpaired adds2000coldqueuedreplays perM32/64/128/192/256 across2freshsame-addressinputepochs (10000total) beforepromotion;1e-3consistencybound andbitwisedifferencesreported. StillNOTH200dispatched.

Widerbank e706a69b83644876 DONE: trialpasses eagercontrol/repeatedcalldiagnostics on3seeds x5shapes x4regimes;15coldpairedratiosGM0.9985938262, no measurablespeedgain, postedfallbackselected. Official5tests/192replays PASS and2.355322x belongtoPOSTEDfallback, not newmath. Finalprivatecheck a60b76ee29af4f54 dispatchedforcompact/fencedv3 withindependentworkspaces+10000queuedcoldreplays.10used0left; no scoredsubmission.

Final compact/fenced H200 a60b76ee29af4f54 DONE. Trial passed generated eager
control-agreement/repeat checks on3seeds x5Mvalues x4routing regimes. All10000
coldqueuedgraphreplays (2000perM32/64/128/192/256;2freshinputepochs perM) were
BITWISESTABLE at every100checkedcheckpoint/shape; maxrelativeL2=0.0.
15pairedcoldgraphcomparisonsGM0.9987561074, perseed0.997125/0.997794/1.001355:
no measurable speedgain. Postedfallbackselected. Official5tests+192mixed
replaysPASS; reported2.352799x is the POSTEDfallback, NOT a newtrialscore.
Do not promote this performance-neutral prototype to publishedsubmission.

Pass handoff: 10/10 dailyprivatechecks spent,0pending,0scoredtoday;8/8total
user-authorizedscoredattempts spent. Nextreset Nov3 00:00UTC / Nov2 16:00PST.
Publishedsubmission.py remains byte-identical cuda_persistent_aligned_safe.py,
SHA3eba8d457b50ff39e98491a5c325f1183f944cd20edd79213f45077cb0f1bbcb,
verifiedpublished2.382249x. BF16work/credentials untouched; no new automation.

Next useful investigation: measure Hopper-specific per-phase time/stalls on
M16/M32 and largeM before another redesign. Repeated tests reject fewer-WG,
small-warp native, generic cache tweaks, wider fixup reads, and disjoint
partials as material speedups. Nativewarp consistency rootcause remains
UNPROVEN; proxyfence hypothesis doesnot establish the fix withoutH200soak.
Newdisjointpartial+proxyfence code offers deterministic reduction with~20MiB
extra workspace but its performance is neutral, so retain it as research.
Sources/builders/preflight evidence frozen research/spark/compact_private_fixup_v3/;
fullH200results research/spark/{private_fixup,compact_private_fixup}_h200_result.json.

Nov2 morning continuation: prepared cuda_persistent_phase_probe.py, a diagnostic
candidate whose prefix/timedkernel/selection match publishedsubmission exactly.
Only prepare() adds own synthetic-input phase measurements. Three probes:
MAINcoarse, MAINproducer-detail, MIDcoarse. Consumer timestamps route/gather/
G1/G2/gridbarrier/final. Producer counters record issuewindows,1in16sampled
emptyringwaits, fullactivation-readiness blocking intervals, logicalstagebytes.
Logicalbyteissue is NOT measuredHBMbandwidth; crossCTAphasewindows overlap.
Integer timestamps are subtracted beforefloatconversion toavoidprecisionloss
atlargeepochvalues. OwnCPUdecoder fixture exercises96ns differencesnear2^60.

5sm90aofflinebuildsPASS,0spills, no WGMMAserialization warnings. MAINcontrol/
coarse/detailed all96regs andsame1024HGMMA/4096HMUL2 staticcounts; MIDcontrol152,
coarse154regs. Registercount equality is NOT proof of identical scheduling.
H200probe runtimeoverhead is UNMEASURED: candidate reports pairedcoldgraph
probe/control overhead and flags deviations beyond3%; do not interpret a
high-overhead probe's times as unperturbed costs. It neverautotunesfromtraces.

Sparkvalidation PASS76independentFP32referencecases+608mixedgraphreplays
(maincoarse32/mainproducer32/MIDcoarse12). All detailedlogicalbyte/stagecounts
matchindependent CPUrouting calculations; timestampmonotonicityverified.
FirstMIDGB10devattempts failed because originalHopperunusedfatallocation
exceeded101KiB andsingle-slotFAST1=2 prefetchwasoutofbounds. Dev-onlyport
usesNS=NSF=FAST1=1 andomitsunusedfatbuffers. H200source/flagsunchanged.
WholeactualcandidatePython diagnosticrunner thenPASS8seed/shape cases onGB10,
mapped todevelopmentcompiledmodules. ThoseGB10times validateintegration only,
NOTH200speed orHopperasyncbehavior. Evidence/manifestfrozen
research/spark/phase_probe_v1/.

No newH200checks/scoredruns thisturn:Nov2private10/10used,0pending;all8user-
authorizedscoredattemptsspent. Askeduserwhether toauthorizeONEadditional
scored DIAGNOSTIC now (cap9) orwait forfreechecks atNov2 16:00PST. Request is
PENDING; doNOTsubmit without explicitapproval. Profileritselfkeepspublished
math/timedpath, so no scoregainexpected fromthatdiagnosticrun. Next authorized
command ifapproved: okc submit muse_moe_w4a16/cuda_persistent_phase_probe.py
--no-wait --json . Otherwise use okc check onlyafteractualquotareset. Readjobs/
limits andupdateledger first. Published2.382249x/BF16work/credentials preserved.
Primarytimerreference https://nvidia.github.io/cccl/unstable/libcudacxx/standard_api/time_library.html .

User explicitly APPROVED one additional scored DIAGNOSTIC run, raisingtotalcap8→9. VerifiedCLIjobs/limits andexactfrozenprofilerSHA369c653aed5ccaa04e89ab5c5844af6c834deb30b70ba9d1c79c72139485ae71, then dispatched b6387d8f931647f0 via okc submit. Ninthauthorizationconsumed;0additional scoredpermission remains. Privatequota10/10stillspent. Timedsource/selectionunchanged; no scoregainexpected fromdiagnosticitself. PendingH200results; publishedsubmission/BF16workpreserved.

Ninthscored DIAGNOSTIC b6387d8f931647f0 DONE/PASS. Unchangedtimedpath scored
2.355024x; thisis a repeat, NOT new algorithmgain. All8syntheticseed/shape
records present; probe/control ratios0.99626..1.01385 (allwithin3%). MAIN
G1 CTAp50 ~79..144us, G2 ~36..69us, route~2..3.4us, finalmax~4.4..9.3us;
MID16 G1p50~53.5/68.5us, G2~28.2/34.8us, finalmax<1us. G2readywaitp95
~6.2..9.8us; sampledproduceremptywaits frequently~1..3us. These are producer
backpressure/phase observations, NOT proof of a specific SASS stall or HBM
counter bottleneck. Logicalbytes not physicalHBM. G1consumesmostelapsedbudget;
routing/finalnormalone cannot provide the~21%aggregate latencycut needed3x.

Nextboundedexperiment: reduce consumer instruction/codefootprint by compiling
one fixed WGMMA Nwidth per shape band instead of8dispatchvariants. This trades
paddedtensorwork for fewer hot-code variants/less dispatch. It is a hypothesis,
not an established bottleneck or speedup. Must initializeunusedsharedBcolumns
and guard splitpartialwrites/epilogues by actualvalidNTT topreventworkspace
contamination acrosscalls. Keep all BF16boundaries andlosslessweightlayout.
NoH200budget remains; prepare/compile/Sparkvalidate whilewaitingfree4pmreset.
DoNOTscoreagain without additional explicitapproval;9/9authorizationused.

Fixed-width consumer prototype prepared afterH200phase profiling. OneN32 orN64
WGMMA width replaces8dispatchpaths. Validtokenoutputs retained, sharedB ring
zeroed oncepercall withproxyfence beforehelpersstart; inactivepartialtilewrites
areguarded topreventcrosscall workspacecontamination. N32routechunks capped32
insideunchanged64tokenscratchABI, so duplicateidsalsofitN32. Newpairedcandidate
cuda_persistent_fixed32_paired.py triesonlyM25/28/32 across3seeds,5routingregimes,
24alternatingcoldpairs,6000queuedcoldreplays,>3%overallpairedgain/perseedguards
andM32>2%medianrequirement; otherwiseoriginalfallback. MIDthrough24 carries the
previouslycheckednearbyshapeextension. Actual6scoredshapes changeonlyM32ifchosen.

OfflineHopperN32 PASS96regs,0spills; executablekerneltext107520B versus373120B
control (~71%smaller), staticHGMMA sites128vs1024 acrossbothkerneltemplates.
Staticcode reduction is NOT runtimeinstruction or speedup evidence. N64 base
64/104Bspills; volatilecounts didnotfix; GS1/volatileGS1still48/72Bspills. Reject
thoseN64builds beforeH200. FunctionalSpark: N32PASS50independentFP32reference
cases+400mixedgraphreplays; N64PASS75+600, includingduplicateids,M385chunking.
ActualN32PythoncalibrationintegrationPASS6000queuedreplays onGB10,zero observed
checkpointdrift. GB10pairedtimes/selection are NOT H200predictors. Frozen
sources/builders/reports research/spark/fixed_width_v1/. No newH200job submitted;
9/9scoredpermissionsspent,private10/10used,0pending;nextfreecheckafterNov2 16PST.

Repository work explicitlyrequested byuser: connectedworkspace toexisting
origin git@github.com:zimo0110/muse-spark-open-kernel-challenge.git/main, preserved
initialApachelicense/history. Firstcheckpoint0724a49 pushed withfullREADME,
verifiedsources/results, BF16snapshotunchanged. LocalGit author/committer
ZoirImomaliev <91550816+zimo0110@users.noreply.github.com>; noCo-authored-bytrailer.
All.env/.env.* recursivelyignored;527indexedfiles passedknownlocalcredential+
token/privatekeyscan. Repo-localprecommithookinstalled. Alternate-index tests
proveforced.env.local anddummyprivatekeyfixturesblockedwithoutchangingrealindex.
Userauthorizesperiodiccommits/pushes;updateREADMEatmeaningfulcheckpoints, keep
secretsuntracked, preserveallrequiredthirdpartylicense/attributionnotices.
