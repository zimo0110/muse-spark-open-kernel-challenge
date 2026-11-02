"""Offline prototype: restore four SM compute groups using two smaller CTAs.

Device-source probe only, not submittable. Requires bounded input M<=32,
separate scratch, a residency check and a doubled grid before any real use.
"""
import ast,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tree=ast.parse((root/'cuda_persistent_two_wg_striped.py').read_text())
src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
defs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAIN_DEFINES' for t in n.targets))
for old,new in [('#define OKC_FT_T 64','#define OKC_FT_T 32'),('constexpr int MMAX = 384','constexpr int MMAX = 32'),('constexpr int WGS_NS = 3','constexpr int WGS_NS = 2'),('#define OKC_WGSNX 3','#define OKC_WGSNX 2'),('constexpr int WGS_SLOTS = 160','constexpr int WGS_SLOTS = 320'),('__launch_bounds__(NTL, 1)','__launch_bounds__(NTL, 2)')]:
 assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
a=src.index('    switch (ntt) {',src.index('__device__ __forceinline__ void wgs_phase('));b=src.index('\n    gs +=',a)
body=src[a:b]
body=re.sub(r'      case [567]:[^\n]+\n','',body)
body=re.sub(r'      default: wgs_seg<PB, 8>[^\n]+','      default: __trap();',body)
src=src[:a]+body+src[b:]
# Bounds are important: the standalone prototype must not accidentally accept
# a larger input or silently run a second block wave through a grid barrier.
marker='template <bool BIG>\n__global__ void __launch_bounds__(NTL, 2)'
assert marker in src
src=src.replace(marker,'''static_assert(((sizeof(Smem)+127)/128)*128 + WGS_NS*UNIT_BYTES + WGS_NX*WGS_XS <= 114688,
              "Two CTA prototype exceeds112KiB/block");
'''+marker)
cases=[]
for label,threshold in [('two_cta32_gs2',5),('two_cta32_gs1',3)]:
 s=src.replace('(WGS_R >= 2 && NTT >= 5)',f'(WGS_R >= 2 && NTT >= {threshold})')
 cases.append({'name':label,'source':s,'defines':defs,'tiny':False})
Path('/tmp/muse_two_cta_cases.json').write_text(json.dumps(cases))
(root/'research/spark/two_cta32_device_probe.json').write_text(json.dumps(cases))
print('Prepared bounded M<=32 device probes, two CTA launch bound, smem compile-time assertion; NO H200 dispatch.')
