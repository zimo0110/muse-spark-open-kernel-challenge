"""Offline small-CTA probe with two-word code prefetch, same MMA/K order."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
cases=json.loads((root/'research/spark/two_cta32_device_probe.json').read_text())
c=next(c for c in cases if c['name']=='two_cta32_gs2');s=c['source']
a=s.index('    uint4 cwn[WGS_R];');b=s.index('    prev = xs; ++seq;',a)
body=r'''    uint2 cwn[WGS_R]; uint2 scs[WGS_R];
#pragma unroll
    for (int ri = 0; ri < WGS_R; ++ri) {
      const unsigned char* pc = ring + ws * UNIT_BYTES + (wg * WGS_R + ri) * WGS_PIECE;
      cwn[ri] = *reinterpret_cast<const uint2*>(pc + (wp * 32 + lane) * 16);
      scs[ri] = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
    }
    static_assert(GS == 2, "half-code probe needs GS2");
#pragma unroll
    for (int cg = 0; cg < 8; ++cg) {
      uint2 cwg[WGS_R];
#pragma unroll
      for (int ri = 0; ri < WGS_R; ++ri) {
        cwg[ri] = cwn[ri];
        if (cg < 7) {
          const unsigned char* pc = ring + ws * UNIT_BYTES + (wg * WGS_R + ri) * WGS_PIECE;
          cwn[ri] = *reinterpret_cast<const uint2*>(pc + ((((cg + 1) / 2) * 4 + wp) * 32 + lane) * 16 + ((cg + 1) & 1) * 8);
        }
      }
      if (cg == 7 && lane == 0) mbar_arrive1(S.bars + BW_E + ws);
#pragma unroll
      for (int ri = 0; ri < WGS_R; ++ri) {
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int js = (cg & 1) * 2 + jj;
          const uint32_t q = jj == 0 ? cwg[ri].x : cwg[ri].y;
          uint32_t r0,r1,r2,r3; dq4(q,r0,r1,r2,r3);
          const uint2 scg = scs[ri];
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw,0u,0x4040u|sb|((sb+1u)<<8))*k16c+0x5C005C00u;
          af[ri][cg & 1][jj][0]=bmul2l(r0,sgp); af[ri][cg & 1][jj][1]=bmul2h(r1,sgp);
          af[ri][cg & 1][jj][2]=bmul2l(r2,sgp); af[ri][cg & 1][jj][3]=bmul2h(r3,sgp);
        }
      }
      wg_fence();
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
        for (int ri = 0; ri < WGS_R; ++ri) {
          if (FIRST && cg == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d[ri],af[ri][0][0][0],af[ri][0][0][1],af[ri][0][0][2],af[ri][0][0][3],xd);
          else
            WGSMMA<NTT>::mma(d[ri],af[ri][cg & 1][jj][0],af[ri][cg & 1][jj][1],af[ri][cg & 1][jj][2],af[ri][cg & 1][jj][3],xd+(uint64_t)((cg*2+jj)*16));
        }
      }
      wg_commit(); wg_wait<1>();
      if (cg == 0 && prev >= 0) {
        if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
        prev = -1;
      }
      if (FIRST && cg == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p,S,pd);
    }
'''
s=s[:a]+body+s[b:]
c=dict(c,name='two_cta32_half_codes',source=s)
(root/'research/spark/two_cta32_half_codes_probe.json').write_text(json.dumps([c]))
Path('/tmp/muse_two_cta_half_cases.json').write_text(json.dumps([c]))
# Address and scale pairs match the original16k16 steps for every lane/warp.
for wp in range(4):
 for lane in range(32):
  old=[(((lg*4+wp)*32+lane)*16+js*4,js,lg*4+js) for lg in range(4) for js in range(4)]
  new=[(((cg//2*4+wp)*32+lane)*16+(cg&1)*8+jj*4,(cg&1)*2+jj,cg*2+jj) for cg in range(8) for jj in range(2)]
  assert old==new
print('Half-code probe preserves all code/scale/K addresses; device compile pending; not submittable.')
