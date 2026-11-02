"""Prepare a routing-aware L2 hint; no math/layout changes and no dispatch."""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'cuda_persistent_aligned_safe.py';text=p.read_text();nodes=ast.parse(text)
n=next(x for x in nodes.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in x.targets));src=ast.literal_eval(n.value)
a=src.index('__device__ void wgs_wproducer(');b=src.index('// x producer:',a);body=src[a:b]
old='  int ws = 0, n = 0;';assert body.count(old)==1
body=body.replace(old,'''  int ws = 0, n = 0;
  int policy_job = -1;
  uint64_t policy = 0x12F0000000000000ull;''')
old='      uint64_t* bar = S.bars + BW_F + ws;';assert body.count(old)==1
body=body.replace(old,'''      if (j != policy_job) {
        // Preserve weights only when another immediately following job uses
        // this expert. Streaming weights retain the original low priority.
        const bool reuse = j + 1 < S.njF + S.njS && S.jE[j + 1] == S.jE[j];
        policy = reuse ? 0x14F0000000000000ull : 0x12F0000000000000ull;
        policy_job = j;
      }
      uint64_t* bar = S.bars + BW_F + ws;''')
assert body.count('L2H_ARG : "memory");')==1
body=body.replace('L2H_ARG : "memory");','''
#if OKC_L2H
                   , "l"(policy)
#endif
                   : "memory");''')
src=src[:a]+body+src[b:]
lines=text.splitlines(keepends=True);a=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;b=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
out=text[:a]+repr(src)+text[b:];ast.parse(out)
(root/'cuda_persistent_reuse_cache.py').write_text(out)
defs=next(ast.literal_eval(n.value) for n in nodes.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAIN_DEFINES' for t in n.targets))
Path('/tmp/muse_reuse_cache_cases.json').write_text(json.dumps([{'name':'reuse_cache_main','source':src,'defines':defs,'tiny':False}]))
print('Routing-aware cache hint prepared; only WGS weight-copy priority changes. NOT dispatched/H200-UNVERIFIED.')
