"""Compare compact shared-memory rings against the verified control."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
s=s.replace('functions=["okc_run", "okc_set_pdl"],','functions=["okc_run", "okc_set_pdl"] + (["okc_resources"] if any(d.startswith("@compact_") for d in defines) else []),')
s+='\nCPP_SRC += "\\nstd::vector<int64_t> okc_resources();\\n"\n'
query='''std::vector<int64_t> okc_resources() {
 size_t sm=okc_smem_bytes();
 cudaError_t e=cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
 TORCH_CHECK(e==cudaSuccess,"shared-memory attribute: ",cudaGetErrorString(e));
 int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,okc::moe_fused<false>,okc::NTL,sm);
 TORCH_CHECK(e==cudaSuccess,"occupancy: ",cudaGetErrorString(e));
 cudaFuncAttributes at;cudaFuncGetAttributes(&at,okc::moe_fused<false>);
 return{active,at.numRegs,(int64_t)sm};
}
'''
for c in json.loads((R/'research/spark/compact_rings_cases.json').read_text()):
 src=c['source'].replace('void okc_run(std::vector<torch::Tensor> w,',query+'\nvoid okc_run(std::vector<torch::Tensor> w,')
 s+='\nALT_SRCS['+repr(c['name'])+']='+repr(src)+'\n'
a=s.index('def kernel_fn(data, state):');b=s.index('\n\ndef ',a+1);part=s[a:b];marker='    for m0 in range(0, M, CHUNK_TOKENS):'
part=part.replace(marker,'''    if M>16 and sfx:
        for upper,defs in state.get("ring_choices",()):
            if M<=upper:
                ext=_ext(tuple(defs)+sfx)
                break
'''+marker);s=s[:a]+part+s[b:]
base=(R/'cuda_skinny_residency_paired.py').read_text();fn=next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=='_skinny_residency_calibrate');cal=ast.get_source_segment(base,fn)
a=cal.index('    for label,nw,blocks in ');b=cal.index('    print(',a)
cal=cal[:a]+'''    for label in ('compact_rings_3_3','compact_rings_4_2','compact_rings_2_4'):
        defs=tuple(MAIN_DEFINES)+('@'+label,)
        mod=_ext(defs+state['sfx']);res=mod.okc_resources();resources[label]=res
        if res[0]>=1:variants[label]=(mod,base,defs)
'''+cal[b:]
cal=cal.replace('_skinny_residency_calibrate','_compact_rings_calibrate').replace('(1,4,8,12,16)','(32,33,128,256,257)').replace("cd=TINY_DEFINES if m<=4 else TINY2_DEFINES if m<=8 else MID_DEFINES","cd=MAIN_DEFINES").replace('m in (1,8,16)','m in (32,128,256)').replace("state['skinny_choices']={}","state['ring_choices']=[]").replace('((4,(1,4)),(8,(8,)),(16,(12,16)))','((64,(32,33)),(192,(128,)),(384,(256,257)))').replace("state['skinny_choices'][band]=(defs,grid)","state['ring_choices'].append((band,defs))").replace("gain>1.03 and min(v)>1.015","gain>1.025 and min(v)>0.985").replace('[skinny','[compact-rings').replace('skinny correctness','compact ring correctness').replace('skinny empty','compact ring empty').replace('skinny queued','compact ring queued')
# A control choice in a lower band must not accidentally fall into a higher
# band's trial choice when the timed dispatcher walks ascending upper bounds.
cal=cal.replace("else:print('[compact-rings selected]',band,'control',flush=True)","else:\n            state['ring_choices'].append((band,tuple(MAIN_DEFINES)))\n            print('[compact-rings selected]',band,'control',flush=True)")
s+='\n'+cal+'\n\ndef prepare(weights):\n    state=_codex_original_prepare(weights)\n    _compact_rings_calibrate(state,weights["w13"].device)\n    _codex_replay_stress(state,weights["w13"].device)\n    return state\n'
ast.parse(s);(R/'cuda_compact_rings_paired.py').write_text(s)
print('Prepared compact routing/ring comparison; same global workspace and full384-token capacity. No dispatch.')
