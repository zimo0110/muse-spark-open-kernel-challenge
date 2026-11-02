// Experimental: each physical warpgroup handles R adjacent logical 64-row tiles.
// R=1 preserves four groups; R=2 uses two groups with a larger register budget.
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(WGS_R) : "memory");
    pd.kind = 0; return;
  }
  // One completion counter covers BOTH logical tiles of this physical group.
  // Its release is after both buffers' writes and the128-thread named barrier.
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg * WGS_R;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
  if constexpr (DF) {
    if (pd.kind == 2) {
      if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
      pd.kind = 0; return;
    }
    if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    pd.kind = 0;
  } else {
    if (wt == 0) {
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
      S.wglast[wg] = old + pd.nst == 16;
    }
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    const bool last = S.wglast[wg]; pd.kind = 0;
    if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  const int ntt = (S.jC[j] + 7) >> 3;
#pragma unroll
  for (int ri = 0; ri < WGS_R; ++ri) {
    const int logical_wg = wg * WGS_R + ri;
    float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + logical_wg) * WGS_FIXW + wt * 4;
    const int r16 = ub * 16 + logical_wg * 4 + wp;
    for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
      float4 v[2];
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if (nt0 + k < ntt) {
          *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
          const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
          wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
        }
    }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(WGS_R) : "memory");
  }
}

template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                      int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_R][WGS_NT][4];
  // Wide N has double accumulators: reduce only its commit-group fragment
  // liveness. The physical block's smaller size raises its register budget.
  constexpr int GS = (WGS_R == 2 && NTT >= 5) ? 1 : OKC_WGSGS;
  uint32_t af[WGS_R][2][GS][4];
  int prev = -1;
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    uint4 cwn[WGS_R]; uint2 scs[WGS_R];
#pragma unroll
    for (int ri = 0; ri < WGS_R; ++ri) {
      const unsigned char* pc = ring + ws * UNIT_BYTES + (wg * WGS_R + ri) * WGS_PIECE;
      cwn[ri] = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
      scs[ri] = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
    }
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      uint4 cwg[WGS_R];
#pragma unroll
      for (int ri = 0; ri < WGS_R; ++ri) {
        cwg[ri] = cwn[ri];
        if (lg < 3) {
          const unsigned char* pc = ring + ws * UNIT_BYTES + (wg * WGS_R + ri) * WGS_PIECE;
          cwn[ri] = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        }
      }
      if (lg == 3 && lane == 0) mbar_arrive1(S.bars + BW_E + ws);
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;
#pragma unroll
        for (int ri = 0; ri < WGS_R; ++ri) {
#pragma unroll
          for (int jj = 0; jj < GS; ++jj) {
            const int js = h * GS + jj;
            const uint4 q4 = cwg[ri]; const uint2 scg = scs[ri];
            const uint32_t q = js == 0 ? q4.x : js == 1 ? q4.y : js == 2 ? q4.z : q4.w;
            uint32_t r0,r1,r2,r3; dq4(q,r0,r1,r2,r3);
            const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
            const uint32_t sgp = __byte_perm(sw,0u,0x4040u|sb|((sb+1u)<<8))*k16c+0x5C005C00u;
            af[ri][ci & 1][jj][0]=bmul2l(r0,sgp); af[ri][ci & 1][jj][1]=bmul2h(r1,sgp);
            af[ri][ci & 1][jj][2]=bmul2l(r2,sgp); af[ri][ci & 1][jj][3]=bmul2h(r3,sgp);
          }
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
#pragma unroll
          for (int ri = 0; ri < WGS_R; ++ri) {
            if (FIRST && ci == 0 && jj == 0)
              WGSMMA<NTT>::mma0(d[ri],af[ri][0][0][0],af[ri][0][0][1],af[ri][0][0][2],af[ri][0][0][3],xd);
            else
              WGSMMA<NTT>::mma(d[ri],af[ri][ci & 1][jj][0],af[ri][ci & 1][jj][1],af[ri][ci & 1][jj][2],af[ri][ci & 1][jj][3],xd+(uint64_t)((lg*4+h*GS+jj)*16));
          }
        }
        wg_commit(); wg_wait<1>();
        if (ci == 0 && prev >= 0) {
          if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
          prev = -1;
        }
        // Match the original eight-k16 deferred-publication point.
        constexpr int FCI = (GS == 1) ? 7 : OKC_WGSFCI;
        if (FIRST && ci == FCI && pd.kind) wgs_flush<PB>(p,S,pd);
      }
    }
    prev = xs; ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {
#pragma unroll
    for (int ri = 0; ri < WGS_R; ++ri) {
      float* fb=wgs_fix(p,PB)+(size_t)(slot*4+wg*WGS_R+ri)*WGS_FIXW+wt*4;
#pragma unroll
      for (int nt=0;nt<NTT;++nt)
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb+nt*512),"f"(d[ri][nt][0]),"f"(d[ri][nt][1]),"f"(d[ri][nt][2]),"f"(d[ri][nt][3]) : "memory");
    }
    pd=WgsPend{(((OKC_WGSDFIN>>(PB?1:0))&1)&&(PB?s0==0:s1==16))?3:2,u,slot,s1-s0};
    return;
  }
  const int j=u>>UBL, ub=u&((1<<UBL)-1);
#pragma unroll
  for (int ri=0;ri<WGS_R;++ri) {
    const int r16=ub*16+(wg*WGS_R+ri)*4+wp;
#pragma unroll
    for (int nt=0;nt<NTT;++nt) wgs_epi_tile<PB>(p,S,j,r16,nt,d[ri][nt]);
  }
  if (!PB) pd=WgsPend{1,u,-1,0};
}
