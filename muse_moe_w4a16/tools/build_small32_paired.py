"""Build a residency-guarded two-CTA small-batch private-check experiment.

No check is dispatched. Device math/layout source comes from the bounded
compiler probe; separate scratch prevents aliasing its smaller internal layout.
"""
import ast,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_aligned_safe.py').read_text()
assert text.count('elif MID_DEFINES and M <= MID_MAX_M:')==1
text=text.replace('elif MID_DEFINES and M <= MID_MAX_M:','elif MID_DEFINES and M <= state.get("mid_max_m", MID_MAX_M):')
small=next(c for c in json.loads((root/'research/spark/two_cta32_device_probe.json').read_text()) if c['name']=='two_cta32_gs2')
source=small['source']
marker='  okc::Params p;';assert source.count(marker)==1
source=source.replace(marker,'  TORCH_CHECK(x.size(0) <= 32, "bounded small32 kernel requires M<=32");\n'+marker)
# This bounded module only executes BIG=false. Avoid compiling an unused twin.
source=source.replace('okc::moe_fused<true>','okc::moe_fused<false>')
resource=r'''
std::vector<int64_t> okc_resources() {
  const size_t sm=okc_smem_bytes();
  cudaError_t e=cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
  TORCH_CHECK(e==cudaSuccess,"dynamic shared-memory attribute: ",cudaGetErrorString(e));
  int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,okc::moe_fused<false>,okc::NTL,sm);
  TORCH_CHECK(e==cudaSuccess,"occupancy query: ",cudaGetErrorString(e));
  cudaFuncAttributes attr;e=cudaFuncGetAttributes(&attr,okc::moe_fused<false>);
  TORCH_CHECK(e==cudaSuccess,"kernel attributes: ",cudaGetErrorString(e));
  return {(int64_t)active,(int64_t)attr.numRegs,(int64_t)sm};
}
'''
# Every build exports the same bindings; resource lookup does not alter device
# math. Original device-source prefixes remain identical to the published code.
old='functions=["okc_run", "okc_set_pdl"],';assert text.count(old)==1
text=text.replace(old,'functions=["okc_run", "okc_set_pdl", "okc_resources"],')
text+='\nSMALL32_DEFINES = '+repr(tuple(small['defines'])+('@small32',))+'\n'
text+='ALT_SRCS["small32"] = '+repr(source)+'\n'
text+='CPP_SRC += '+repr('\nstd::vector<int64_t> okc_resources();\n')+'\n'
text+='CUDA_SRC += '+repr(resource)+'\n'
text+='for _source_name in tuple(ALT_SRCS):\n    ALT_SRCS[_source_name] += '+repr(resource)+'\n'
text+=r'''

def _small32_scratch(dev):
    entries=32*8
    jobs=32+entries//32+8
    xrows=entries+8*jobs
    fix=2*320*4*(4*512)+2*320*4+136*16
    return [torch.zeros(8+1024+entries,dtype=torch.int32,device=dev),
            torch.zeros(entries+entries*HIDDEN+fix,dtype=torch.float32,device=dev),
            torch.zeros((entries+2*xrows)*HIDDEN,dtype=torch.bfloat16,device=dev)]

_small32_parent_kernel = kernel_fn

def kernel_fn(data, state):
    m=data['hidden_states'].shape[0]
    if state.get('small_min_m',33) <= m <= 32:
        _bind_context(data['hidden_states'])
        cfg=data['config']
        assert cfg['local_start']==state['lo'] and cfg['local_count']==state['cnt']
        x=data['hidden_states'].contiguous()
        ids=data['topk_ids'].to(torch.int32).contiguous()
        if ids.data_ptr() & 15:
            ids=ids.clone()
        tw=data['topk_weights'].to(torch.float32).contiguous()
        out=torch.empty((m,HIDDEN),dtype=torch.bfloat16,device=x.device)
        ext=_ext(tuple(SMALL32_DEFINES)+state['sfx'])
        ext.okc_run(state['w'],state['small_scratch'],x,ids,tw,out,
                    state['lo'],state['cnt'],state['small_grid'])
        return out
    return _small32_parent_kernel(data,state)
'''
paired=(root/'cuda_persistent_two_wg_paired.py').read_text();tree=ast.parse(paired)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_paired_main_choice')
f=ast.get_source_segment(paired,fn).replace('_paired_main_choice','_paired_small32_choice').replace('two_wg_striped','small32').replace('(64, 128, 192, 256)','(17, 24, 25, 31, 32)')
old="    candidates = {'posted': PAIRED_CONTROL_DEFINES, 'small32': MAIN_DEFINES}\n    modules = {name: _ext(tuple(defs) + state['sfx']) for name, defs in candidates.items()}"
assert old in f
f=f.replace(old,"""    small_module=_ext(tuple(SMALL32_DEFINES)+state['sfx'])
    active,regs,smem=small_module.okc_resources()
    print('[small32 resources]',{'active_ctas_per_sm':active,'registers':regs,'shared_bytes':smem},flush=True)
    if active < 2 or state['grid']*2 >= 320:
        print('[small32] residency guard rejected; using original kernels',flush=True)
        return
    state['small_grid']=state['grid']*2
    state['small_scratch']=_small32_scratch(dev)""")
old="        for m in shape_ratios:\n            data = _gen_input(m, seed + m, dev)";assert old in f
f=f.replace(old,"""        for m in shape_ratios:
            control_defs=MID_DEFINES if m <= 24 else MAIN_DEFINES
            modules={'posted':_ext(tuple(control_defs)+state['sfx']),'small32':small_module}
            data = _gen_input(m, seed + m, dev)""")
old="""                modules[name].okc_run(state['w'], state['scratch'], x, ids, tw,
                                     outputs[name], state['lo'], state['cnt'], state['grid'])""";assert old in f
f=f.replace(old,"""                workspace=state['small_scratch'] if name=='small32' else state['scratch']
                grid=state['small_grid'] if name=='small32' else state['grid']
                modules[name].okc_run(state['w'],workspace,x,ids,tw,
                                     outputs[name],state['lo'],state['cnt'],grid)""")
a=f.index('    choose_trial = ')
f=f[:a]+'''    selected=33
    for lower in (17,25):
        points=[row for row in samples if row['m'] >= lower]
        means=[math.exp(sum(math.log(row['control_over_trial']) for row in points if row['seed']==seed)/sum(row['seed']==seed for row in points)) for seed in (47821,73931,95813)]
        meanshape=[statistics.median(v) for m,v in shape_ratios.items() if m >= lower]
        gm=math.exp(sum(math.log(row['control_over_trial']) for row in points)/len(points))
        if gm > 1.03 and min(means) > 1.02 and min(meanshape) > 0.97:
            selected=lower
            break
    state['small_min_m']=selected
    if selected==33:
        del state['small_scratch']
    print('[small32] selected minimum batch',selected,'overall_geomean',gain,
          'seconds',time.perf_counter()-started,flush=True)
'''
f=f.replace('[paired-main]','[small32 paired]')
text+='\n'+f+r'''

_small32_base_prepare = _codex_original_prepare

def prepare(weights):
    state=_small32_base_prepare(weights)
    state['mid_max_m']=24
    _paired_small32_choice(state,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
ast.parse(text);(root/'cuda_persistent_small32_paired.py').write_text(text)
# Workspace sizing matches the device's bounded constants exactly.
assert 2*132 < 320
assert 32*8+8*(32+32*8//32+8)==640
print('Prepared bounded, separate-scratch, residency-guarded2CTA paired experiment. NOT dispatched; H200 UNVERIFIED.')
