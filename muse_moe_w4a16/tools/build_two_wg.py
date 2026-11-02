"""Experimental2warpgroup WGMMA design. H200 execution/performance UNVERIFIED.

Preserves four logical64-row tiles and all raw packed weight/fixup layouts.
Each of two physical warpgroups handles two logical tiles. Fewer per-unit
release/count operations and a larger per-thread register budget are hypotheses,
not established speedups. No leaderboard job is launched by this builder.
"""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
path=root/'submission.py';original=path.read_text()
state=json.loads((root/'overnight_state.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest()==state['posted_best']['sha256']
lines=original.splitlines(keepends=True)
node=next(n for n in ast.parse(original).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
src=ast.literal_eval(node.value)
marker='constexpr int WGS_NT = FT_T / 8;'
assert src.count(marker)==1
src=src.replace(marker,'constexpr int WGS_R = 16 / NW;\nstatic_assert(NW == 8 || NW == 16, "WGS physical warpgroup count");\n'+marker)
start=src.index('template <bool PB>\n__device__ __forceinline__ void wgs_flush(')
end=src.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase(',start)
src=src[:start]+(root/'tools/two_wg_device.inc').read_text()+'\n'+src[end:]
# Keep every old512logical gather lanes, with2chunks/thread whenNT256.
start=src.index('__device__ void wgs_gather(');end=src.index('// Work left from the previous segment',start)
body=src[start:end]
body=body.replace('  int j = 0;\n','',1)
body=body.replace('static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");','static_assert(HD % (NT * 8) == 0, "whole logical gather chunks");')
a=body.index('  for (int qb = q0;');b=body.index('  CTA_SYNC();',a)
loop=body[a:b]
loop=loop.replace('(tid >> 5)','(ltid >> 5)').replace('(tid & 31)','(ltid & 31)').replace('const int G = tid & 31;','const int G = ltid & 31;')
body=body[:a]+'#pragma unroll\n  for (int part = 0; part < HD / (NT * 8); ++part) {\n    const int ltid = tid + part * NT;\n    int j = 0;\n'+loop+'  }\n'+body[b:]
src=src[:start]+body+src[end:]
# WGS final pass covers the same4096columns with16values/thread forNT256.
a=src.index('#if OKC_WGS && OKC_WGSFINSS\n');b=src.index('#elif OKC_FINCOL',a)
body=src[a:b]
old='static_assert(HD == NT * 8, "8 columns per thread");';assert body.count(old)==1
body=body.replace(old,'constexpr int FCOLS = HD / NT;\n    static_assert(FCOLS == 8 || FCOLS == 16, "full final row");')
body=body.replace('+ tid * 2;', '+ tid * (FCOLS / 4);')
old='''const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;'''
assert body.count(old)==1
body=body.replace(old,'''#pragma unroll
          for (int vi=0; vi<FCOLS/4; vi+=2) {
            const float4 a=__ldcg(src+vi), b=__ldcg(src+vi+1);
            ps[sl] += a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w+b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w;
          }''')
body=body.replace('float acc[8];','float acc[FCOLS];').replace('i < 8; ++i) acc[i]', 'i < FCOLS; ++i) acc[i]')
old='''const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;'''
assert body.count(old)==1
body=body.replace(old,'''#pragma unroll
          for(int vi=0;vi<FCOLS/4;++vi) {
            const float4 a=__ldcg(src+vi);
            acc[4*vi]+=r*a.x;acc[4*vi+1]+=r*a.y;acc[4*vi+2]+=r*a.z;acc[4*vi+3]+=r*a.w;
          }''')
body=body.replace('uint4 o;','uint4 o[FCOLS/8];').replace('reinterpret_cast<uint32_t*>(&o)','reinterpret_cast<uint32_t*>(o)').replace('i < 4; ++i) {','i < FCOLS/2; ++i) {')
old='*reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;';assert body.count(old)==1
body=body.replace(old,'''uint4* dst=reinterpret_cast<uint4*>(p.out+(size_t)tk*HD+tid*FCOLS);
#pragma unroll
      for(int vi=0;vi<FCOLS/8;++vi) dst[vi]=o[vi];''')
src=src[:a]+body+src[b:]
start=sum(map(len,lines[:node.value.lineno-1]))+node.value.col_offset
end=sum(map(len,lines[:node.value.end_lineno-1]))+node.value.end_col_offset
text=original[:start]+repr(src)+original[end:]
old='MAIN_DEFINES = ("-DOKC_NW=16",';assert text.count(old)==1
text=text.replace(old,'MAIN_DEFINES = ("-DOKC_NW=8",')
text=text.replace('# Local Apache-2.0 port: bf16stage;', '# EXPERIMENTAL2-WG: H200 execution/correctness/performance UNVERIFIED;')
(root/'cuda_persistent_two_wg.py').write_text(text)
# Structured CPU coverage proofs for both original and new warpgroup counts.
for nw in (8,16):
 r=16//nw;groups=nw//4
 assert groups*r==4
 logical=[wg*r+ri for wg in range(groups) for ri in range(r)]
 assert logical==list(range(4))
 for ub in range(32):
  rows=[ub*16+(wg*r+ri)*4+wp for wg in range(groups) for ri in range(r) for wp in range(4)]
  assert sorted(rows)==list(range(ub*16,(ub+1)*16))
 assert sum(r for _ in range(groups))*2==8 #2Aunits feed each Bstage
 seen=[tid+part*(nw*32) for part in range(4096//(nw*32*8)) for tid in range(nw*32)]
 assert sorted(seen)==list(range(512))
 cols=[tid*(4096//(nw*32))+c for tid in range(nw*32) for c in range(4096//(nw*32))]
 assert cols==list(range(4096))
 # Logical fixup slots remain bit-for-bit identical; only ready counters coalesce.
 for slot in (0,1,131,159):
  assert sorted(slot*4+wg*r+ri for wg in range(groups) for ri in range(r))==list(range(slot*4,(slot+1)*4))
print('Generated2-WG experimental candidate; logical tiles, row coverage, fixup addressing, release counts, gather/final coverage PASS. H200 UNVERIFIED.')
