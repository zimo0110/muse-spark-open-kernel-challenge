"""Trade one ring slot for possible two-CTA/SM tiny-batch residency.

Use the actual CUDA occupancy API before raising the grid, to keep the
persistent kernel's grid barrier resident and avoid oversubscription deadlock.
Only one additional CUDA variant (tiny NS=2) is compiled within prepare budget.
Fresh, paired cold graph calibration retains the checked NS=3 baseline unless
the new configuration wins on two inputs plus independent confirmation.
"""
import ast
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_aligned_safe.py').read_text();lines=text.splitlines(keepends=True);repls=[]
resources=r'''
std::vector<int64_t> okc_resources() {
  const size_t sm = okc_smem_bytes();
  cudaError_t e = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
  TORCH_CHECK(e == cudaSuccess, "set smem: ", cudaGetErrorString(e));
  int active = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, okc::moe_fused<false>, okc::NTL, sm);
  TORCH_CHECK(e == cudaSuccess, "occupancy: ", cudaGetErrorString(e));
  cudaFuncAttributes attr;
  e = cudaFuncGetAttributes(&attr, okc::moe_fused<false>);
  TORCH_CHECK(e == cudaSuccess, "attributes: ", cudaGetErrorString(e));
  return {(int64_t)active, (int64_t)attr.numRegs, (int64_t)sm};
}
'''
for n in ast.parse(text).body:
 if isinstance(n,ast.Assign) and any((isinstance(t,ast.Name) and t.id in ('CUDA_SRC','CPP_SRC')) or (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ALT_SRCS') for t in n.targets):
  src=ast.literal_eval(n.value)
  if isinstance(n.targets[0],ast.Name) and n.targets[0].id=='CPP_SRC':src+='\nstd::vector<int64_t> okc_resources();\n'
  else:
   marker='void okc_run(std::vector<torch::Tensor> w,'
   assert src.count(marker)==1;src=src.replace(marker,resources+'\n'+marker)
  st=sum(map(len,lines[:n.value.lineno-1]))+n.value.col_offset;en=sum(map(len,lines[:n.value.end_lineno-1]))+n.value.end_col_offset
  repls.append((st,en,repr(src)))
assert len(repls)==3
for st,en,src in reversed(repls):text=text[:st]+src+text[en:]
assert text.count('functions=["okc_run", "okc_set_pdl"]')==1
text=text.replace('functions=["okc_run", "okc_set_pdl"]','functions=["okc_run", "okc_set_pdl", "okc_resources"]')
# Drop old validation wrapper; install the instrumented stress function below.
text=text[:text.index('# Development-only prepare validation.')]
text+=r'''
_codex_occ_base_kernel = kernel_fn
_codex_occ_base_prepare = prepare

def _codex_occ_run(data,state,ext,grid):
    x=data['hidden_states'].contiguous()
    _bind_context(x)
    assert data['config']['local_start']==state['lo'] and data['config']['local_count']==state['cnt']
    ids=data['topk_ids'].to(torch.int32).contiguous()
    tw=data['topk_weights'].to(torch.float32).contiguous()
    out=torch.empty_like(x)
    ext.okc_run(state['w'],state['scratch'],x,ids,tw,out,state['lo'],state['cnt'],grid)
    return out

def kernel_fn(data,state):
    m=data['hidden_states'].shape[0]
    cfg=state.get('tiny_occ',{}).get(1 if m==1 else 4)
    if m<=TINY_MAX_M and cfg:
        return _codex_occ_run(data,state,cfg[0],cfg[1])
    return _codex_occ_base_kernel(data,state)

def _codex_occ_capture(data,state,ext,grid):
    stream=torch.cuda.Stream(device=data['hidden_states'].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _codex_occ_run(data,state,ext,grid)
        _codex_occ_run(data,state,ext,grid)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            out=_codex_occ_run(data,state,ext,grid)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    return graph,out

def prepare(weights):
    import statistics
    state=_codex_occ_base_prepare(weights)
    sfx=state['sfx'];full=state['grid'];dev=weights['w13'].device
    control=_ext(tuple(TINY_DEFINES)+sfx)
    small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2',)+sfx)
    capacity,regs,smem=small.okc_resources()
    print(f'[tiny residency] NS2 capacity={capacity} regs={regs} shared={smem}; SMs={full}',flush=True)
    choices=[('NS3',control,full),('NS2',small,full)]
    if capacity>=2:
        choices.append(('NS2x2',small,full*2))
    flush=torch.zeros(64*1024*1024,dtype=torch.int32,device=dev)
    e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    def sample(g):
        flush.add_(1);e0.record();g.replay();e1.record();e1.synchronize()
        return e0.elapsed_time(e1)
    chosen={}
    for m in (1,4):
        ratios={label:[] for label,_,_ in choices[1:]}
        for seed in (40009,70019,100003):
            data=_gen_input(m,seed+m,dev)
            graphs={label:_codex_occ_capture(data,state,ext,grid) for label,ext,grid in choices}
            want=graphs['NS3'][1].float().clone()
            for label in ratios:
                out=graphs[label][1]
                err=(torch.linalg.vector_norm(out.float()-want)/torch.linalg.vector_norm(want).clamp_min(1e-20)).item()
                if err>0.001 or not torch.isfinite(out).all().item():
                    raise AssertionError(('occupancy output mismatch',m,label,err))
                values=[]
                for turn in range(12):
                    if turn%2:
                        a,b=sample(graphs['NS3'][0]),sample(graphs[label][0])
                    else:
                        b,a=sample(graphs[label][0]),sample(graphs['NS3'][0])
                    values.append(a/b)
                ratios[label].append(statistics.median(values))
        eligible={k:min(v) for k,v in ratios.items() if min(v)>1.03}
        pick=max(eligible,key=eligible.get) if eligible else 'NS3'
        if pick!='NS3':
            _,ext,grid=next(v for v in choices if v[0]==pick)
            chosen[m]=(ext,grid)
        print(f'[tiny residency] M={m} chosen={pick} three-seed-ratios={ratios}',flush=True)
    state['tiny_occ']=chosen
    _codex_replay_stress(state,dev)
    return state
'''
stress=(root/'persistent_replay_stress.inc').read_text().split('_codex_original_prepare = prepare')[0]
stress=stress.replace('(1, 4, 8, 16, 17, 32, 33, 65, 128, 256, 257, 385)','(1, 2, 3, 4, 8, 16, 17, 32, 33, 65, 128, 256, 257, 385)')
stress=stress.replace("print('[codex replay] PASS 192", "print('[codex replay] PASS 224")
text+=stress
(root/'cuda_persistent_occupancy.py').write_text(text)
print('Built occupancy-gated NS2 tiny experiment; no unsafe grid larger than measured residency.')
