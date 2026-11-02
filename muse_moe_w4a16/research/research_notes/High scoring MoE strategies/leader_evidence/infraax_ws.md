# C30d result: v7 with 2 bulk copies/stage (job d7448ac2375f4bc4, 2026-10-26 10:42Z) — verdict: v7 (mma.sync consumers) is NOT a board candidate
Bitwise vs v5 OK on every bucket. GEMM1 µs (3 seeds), D2S5 = best layout:
| bs | v5 | d2s5 | d1s8 | nocomp2 (producer only) | consonly2 (consumers only, L2-hot src) | wstream (DRAM ceiling) |
|---|---|---|---|---|---|---|
| 256 | 229–235 | 267–285 | 298–316 | 201–216 | 260–280 | 114–127 |
| 128 | 180–212 | 220–250 | 248–279 | 170–189 | 222–248 | 104–116 |
| 32 | 87–111 | 110–141 | 124–157 | 87–107 | 112–139 | 53–69 |
| 16 | 67–84 | 84–106 | 95–122 | 67–83 | 85–107 | 46–58 |
| 1 | 21–29 | 37–39 | 41–42 | 30–34 | 38–39 | 16–22 |
Stamps bs256 d2s5: prod_wait_empty 74 %, cons_wait_full 10 %, fix_epi 23 % → the CONSUMERS bind now (the copy-count fix worked: 2.0× vs C30b).
Readings:
1. Consumer ceiling (consonly2) ≈ 1.9–2.0 TB/s-equivalent: 12 smem-fed mma.sync warps (lds tile → unpack → ldmatrix → mma) are no faster than
   v5's register-fed warps. The dequant+mma chain itself is the cap (Fable est. INT pipe 63 %, smem 48–69 %), not memory.
2. Producer-only (nocomp2) ≈ 2.6 TB/s: the single bulk-copy producer also falls short of the 3.2 target (rule iv).
3. fix_epi 20–23 % of the CTA's time: with 1 CTA/SM the split-K fixup + epilogue is dead time for the stream (v5 overlaps it across 3 CTAs/SM).
Decision (fable-10 D3): best WS 284 µs > 205 kill line, and BOTH sides are below target → v7 as built is killed for the board (2nd strike:
C30b copy-issue, C30d consumer ceiling). Kept on main as the measurement harness: CONSONLY is a clean microbenchmark of the consumer chain.
Open lever (new evidence, not a failed class): the dequant chain ("unpack the unpack", X scout) caps BOTH v5 and v7 at ≈ 2 TB/s-equivalent;
a consumer-side dequant experiment can be measured directly in the CONSONLY twin before any rebuild.
