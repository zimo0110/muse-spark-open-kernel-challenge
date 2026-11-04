"""Compact unused WGS shared fields to test power-of-two TMA rings.

Global workspace, token capacity, packed weights, and numerical operations
remain unchanged. Count/offset maxima are3072, exactly representable in u16.
Unused skinny-only arrays retain one placeholder element for dead functions.
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
t=ast.parse((R/'cuda_persistent_aligned_safe.py').read_text())
def val(k):return next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id==k for v in n.targets))
s=val('CUDA_SRC')
def replace(a,b):
 global s
 assert s.count(a)==1,a[:80]
 s=s.replace(a,b)
replace('uint32_t lut[256];','uint32_t lut[OKC_WGS ? 1 : 256];')
replace('int fcnt[32];','int fcnt[OKC_WGS ? 1 : 32];')
replace('int cnt[64], off[64];','int cnt[OKC_WGS ? 1 : 64], off[OKC_WGS ? 1 : 64];')
replace('int wc[NW][32];','''#if OKC_WGS
  unsigned short wc[NW][32];
#else
  int wc[NW][32];
#endif''')
replace('int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];','int jE[JMAX], jS[JMAX], jC[JMAX], forder[OKC_WGS ? 1 : JMAX];')
replace('int uslot[8];','int uslot[OKC_WGS ? 1 : 8];')
replace('int dync[JMAX];','int dync[OKC_WGS ? 1 : JMAX];')
replace('    S.cnt[lane] = tot;\n    S.off[lane] = off;','''#if !OKC_WGS
    S.cnt[lane] = tot;
    S.off[lane] = off;
#endif''')
replace('#if SKDYN\n  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;','#if SKDYN && !OKC_WGS\n  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;')
replace('constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;','constexpr int BW_F = 0, BW_E = WGS_NS, BX_F = 2*WGS_NS, BX_E = BX_F + WGS_NX;')
replace('constexpr int WGS_NS = 3;','static_assert(OKC_PRMTSC,"compact WGS requires exact PRMT scales");\nconstexpr int WGS_NS = 3;')
cases=[]
for ns,nx in ((3,3),(4,2),(2,4)):
 code=s.replace('constexpr int WGS_NS = 3;',f'constexpr int WGS_NS = {ns};').replace('#define OKC_WGSNX 3',f'#define OKC_WGSNX {nx}')
 cases.append(dict(name=f'compact_rings_{ns}_{nx}',source=code,defines=val('MAIN_DEFINES'),tiny=False))
(R/'research/spark/compact_rings_cases.json').write_text(json.dumps(cases)+'\n')
# Every row count and prefix written by routing stays within the entry count.
assert 384*8<65536
print('Prepared compact shared routing and 3/3,4/2,2/4 TMA ring probes; no runtime/speed claim.')
