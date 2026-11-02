# Public high score provenance

## Is the 2.7x leader's implementation public?

### Takeaway
The live official CLI read during this research reports **dolanzhao 2.7578x, rank1** on muse_moe_w4a16. I could not find a public challenge implementation or technical writeup attributable to that entry. We must not describe a guessed method as “what the 2.7x contestants did.”

### Cited Findings
- Current leaderboard primary source is the challenge board; the authenticated read-only CLI JSON snapshot is preserved as `leader_evidence/live_board.json` (1242 ranked entries, leader dolanzhao2.7578x). The rendered web page sometimes says unavailable/no scored runs despite live CLI rows. — [Official board](https://openkernelchallenge.dev/leaderboard?problem=muse_moe_w4a16)
- The GitHub profile `dolanzhao` calls itself a performance engineer; GitHub marks profile activity private. — [Profile](https://github.com/dolanzhao)
- GitHub's public repository listing returns16 repositories. None has an obvious Muse/OKC entry, and all listed last-push dates are2025or earlier. This is a bounded discovery result, not proof that no public code exists under another account. Copies of the public JSON are preserved in leader_evidence. — [Public repos API](https://api.github.com/users/dolanzhao/repos?per_page=100)
- Public sources I found with explicit MoE claims are below the leader: ruwwww2.3370x and alygg77~2.40x; the sibling researcher is auditing alygg77. — [ruwwww](https://github.com/ruwwww/open-kernel-challenge), [alygg77](https://github.com/alygg77/okc-muse-spark-kernels)

### Inferences
- The strongest actionable claim is “public native-CUDA implementations around2.34–2.40x exist,” not “we recovered the2.7x leader's trick.”

### Gaps
- Leader code, license, per-case timings and a root-cause explanation for2.7578x remain unavailable from this search. No team was contacted, no account registered, no messages posted.

## What does the public 2.337x ruwwww source actually do?

### Takeaway
The ruwwww repository contains native CUDA persistent **whole-operator scheduling**, shape-specific skinny and RS-WGMMA paths, and lower-instruction-count dequantization. This differs materially from adding a weight-producer warp group to each standalone Gluon GEMM. However, its README overstates intermediate fusion, and no repository license file was found.

### Cited Findings
- Pinned repository commit is `3b15eb8a6a16e0519378e28ea6c9b44d04e0a613`, dated2026-10-29T18:35:16Z. — [Commit](https://github.com/ruwwww/open-kernel-challenge/commit/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613)
- Committed judge profile `submission.3ed21907cbff49be.json` reports **2.336980283653101x**, CI[2.3019255793983047,2.3720349879078975],5/5tests passed. Its per-case speedups are M1=3.1018x,M16=1.9740x,M32=1.6260x,M8=2.2451x,M128=2.5621x,M256=2.8445x. All six profiles show one `moe_fused` kernel. This is committed judge evidence, not a new check by us; exact uploaded-source hash is not included, so do not overstate that latest source bytes are proven identical to that run. — [Pinned profile](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/profiles/submission.3ed21907cbff49be.json)
- The other5committed MoE profile summaries are1.836–1.893x, not2.7x. Thus2.3370x is the strongest directly preserved run in this repository. — [Profiles directory](https://github.com/ruwwww/open-kernel-challenge/tree/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/profiles)
- Source algorithm docs atlines8–40: every CTA builds per-expert route/job lists; experts are split into skinny/fat jobs; one persistent grid runs both phases. Skinny units are16output rows/fullK with warp-level splitK. Fat units use256rows and shared staging. Jobs are cost-partitioned with an atomic tail pool, and phase-A final ring refills prefetch early phase-B work when possible. — [Source algorithm](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L8)
- **Correction to README:** code explicitly writes BF16 SwiGLU activations to global `inter`, writes FP32 `ybuf`/sum-of-squares scratch, and uses grid barriers to make them visible. Do not repeat README's “zero-DRAM intermediate activations” claim. Source docs lines19–27 and scratchallocation lines3303–3317 establish global buffers. — [Actual inter/scratch algorithm](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L19), [Scratch](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L3303)
- Shape dispatch: M<=4TINY, M<=8TINY2, M<=16MID; larger batches use MAIN with `OKC_WGS=1` (RS-WGMMA). Source lines124–156 give exact define tuples, and kernel_fn3320–3346 selects only by shape/config. MAIN's WGS implementation has four consumer warp groups plus a TMA producer and streamK fixups. — [Variants](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L124), [Dispatch](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L3320)
- Dequantization is a specific instruction-pipeline optimization: `DQ7` repacking makes eightFP4values decode with7ALUops and2IMAD.SHL operations, using integer multiply to shift some work away from the saturatedALUpipe. It is not just generic prefetching. Positive-normalFP8scale bytes use PRMT+IMAD integer rebias; prepare checks everyactualscale byte and chooses LUT/mma.syncfallback if unsupported scales appear. — [dq4](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L459), [Scale decoder](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L657), [Prepare guard](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L3289)
- Timing-safe state: prepare repacks bits/bytes, keeps supplied BF16postnorm and FP32global scales, and allocates reusable scratch; each GPU invocation advances/reset counters and flags using launch generation. Timed function creates fresh output and launches the selected CUDA function, without host tensor-value sync. These are code-inspection facts, not a full concurrency proof. — [Prepare](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L3244), [Generation/reset](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L2529)
- **Precision audit flag:** epi_A lines606–613 uses FP32 gate/up/SILU and rounds only final inter; epi_B lines620–628 computes scaledW2 times norm and stores FP32ybuf, without visible BF16rounding at the W2/norm boundaries. This differs from our required BF16stage semantics. Higher precision is not inherently a lower-precision shortcut, but strict task wording still requires an audit; do not adopt these epilogues unchanged or assert full compliance from a passing profile. — [Epilogues](https://github.com/ruwwww/open-kernel-challenge/blob/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613/muse_moe_w4a16/submission.py#L606)
- No LICENSE/NOTICE file appears in the pinned tree or source header, and README has no licensing section. Do not copy this repository as though a license was verified; sibling's Apache2alygg/vjeux sources are safer reuse bases. — [Pinned tree](https://api.github.com/repos/ruwwww/open-kernel-challenge/git/trees/3b15eb8a6a16e0519378e28ea6c9b44d04e0a613?recursive=1)

### Inferences
- Main architectural lesson: fuse scheduling/epilogues across the entire operator, keep skinny weights/activations in register-oriented fragments, and use a genuinely different RS-WGMMA engine for larger batches. Our prior per-GEMM producer/consumer Gluon experiment did not test this design.
- DQ7's explicit ALU/FMA instruction mix is a narrowly testable next change, but no2.7x claim can be attached to it. The highest public per-case gains are M1 and larger batches, not M32; improving only M32 will not reproduce the observed2.337xgeomean.

### Gaps
- No full correctness/replay soak was run here, no GPU jobs were dispatched, and no code was executed. Persistent barriers require specific residency/replay auditing before adaptation.
- The repository lacks a clear reuse license; rule-compliance and permission cannot be concluded from a README claim or a single passing judge log.

## What useful negatives come from infraax?

### Takeaway
Infraax is not a high-score source: its final published MoE is1.486x. It is valuable because its measured warp-specialized consumer bottleneck closely matches our failed direction and points to instruction throughput rather than blindly adding more producers.

### Cited Findings
- Pinned commit is `896b816dc5fd418e1f167578eb35680b116b68f3`, dated2026-10-27T11:45:30Z. README reports MoE1.486x versus then-leader2.62x; archive says it stopped after fourdays/86judge runs. — [Repository](https://github.com/infraax/okc-kernel-challenge/tree/896b816dc5fd418e1f167578eb35680b116b68f3)
- `design/breakdown/c30d-moe-ws-result.md` reports bitwise correctness againstv5, but best WS M256GEMM1~267–285us versusv5~229–235us. Producer waits for empty74%, consumer waits for full10%; consumer-only remains~260–280us, while pure weightstream~114–127us. Authors conclude the consumer dequant+mma chain caps the implementation, not merely memory-copy issue. — [Measured negative](https://github.com/infraax/okc-kernel-challenge/blob/896b816dc5fd418e1f167578eb35680b116b68f3/design/breakdown/c30d-moe-ws-result.md)
- The source repository is Apache2licensed. — [LICENSE](https://github.com/infraax/okc-kernel-challenge/blob/896b816dc5fd418e1f167578eb35680b116b68f3/LICENSE)

### Inferences
- Stop treating “warp specialization” as a sufficient strategy. A correct producer/consumer structure can be slower when consumer issue throughput and splitK epilogues dominate.
- Best next source to test is the sibling-audited Apache2alygg77~2.40xsource, rather than reproducing infraax's failed WS setup or copying unlicensed ruwwww bytes. Use ruwwww's exact distinctions (fulloperator scheduling, lowinstruction dequant, shape-specific RS-WGMMA) as architecture hypotheses.

### Gaps
- These are authors' published measurements on their own implementation, not hardware-counter measurements of ours. The reported74%/10%must not be presented as our measured stall fractions.
