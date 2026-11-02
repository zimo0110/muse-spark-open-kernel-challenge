"""Integrate the native-warp pilot with the immutable checked larger-M path.

Build only. Never dispatches a check or changes submission.py.
"""
from pathlib import Path
import ast,sys
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_aligned_safe.py').read_text()
old='elif MID_DEFINES and M <= MID_MAX_M:';assert text.count(old)==1
text=text.replace(old,'elif MID_DEFINES and M <= state.get("mid_max_m", MID_MAX_M):')
port=(root/'tools/warp_moe_port.py').read_text();tree=ast.parse(port)
cpp=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='CPP' for k in n.targets))
repack=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='warp_repack')
text+='\nNATIVE_CUDA = '+repr((root/'tools/warp_moe_device.cu').read_text())+'\nNATIVE_CPP = '+repr(cpp)+'\n'
text+='NATIVE_REF_CUDA = '+repr((root/'tools/warp_moe_reference.cu').read_text())+'\n'
text+='NATIVE_REF_CPP = '+repr('#include <torch/extension.h>\nvoid diagnostic_reference(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n')+'\n'
text+=ast.get_source_segment(port,repack)+'\n'
text+=r'''
_NATIVE_MODULES={}
_NATIVE_REF_MODULE=None
NATIVE_VARIANTS=((8,4,8),(8,6,8),(8,4,16),(8,6,16),(16,4,8),(16,4,16))

def _native_ext(variant):
    key=tuple(variant[:2])
    if key not in _NATIVE_MODULES:
        major,minor=torch.cuda.get_device_capability()
        arch=f'{major}{minor}'+('a' if major==9 else '')
        _NATIVE_MODULES[key]=load_inline(name='native_moe_'+hashlib.sha256((NATIVE_CUDA+repr(key)).encode()).hexdigest()[:20],cpp_sources=NATIVE_CPP,cuda_sources=NATIVE_CUDA,functions=['warp_run','warp_resources'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo',f'-DWARP_NPAD={key[0]}',f'-DWARP_NS={key[1]}'],verbose=False)
    return _NATIVE_MODULES[key]


def _native_run(data,state,variant):
    _bind_context(data['hidden_states'])
    x=data['hidden_states'].contiguous()
    cfg=data['config']
    assert cfg['local_start']==state['lo'] and cfg['local_count']==state['cnt']
    ids=data['topk_ids'].to(torch.int32).contiguous()
    if ids.data_ptr()&15:ids=ids.clone()
    tw=data['topk_weights'].to(torch.float32).contiguous()
    out=torch.empty_like(x)
    _native_ext(variant).warp_run(state['native_w'],state['native_scratch'],x,ids,tw,out,state['lo'],state['cnt'],state['grid']*variant[2])
    return out


_native_control_kernel=kernel_fn

def kernel_fn(data,state):
    m=data['hidden_states'].shape[0]
    bucket=1 if m==1 else 4 if m<=4 else 8 if m<=8 else 16 if m<=16 else 0
    variant=state.get('native_select',{}).get(bucket)
    if variant is not None:
        return _native_run(data,state,variant)
    return _native_control_kernel(data,state)


def _native_setup(state,dev):
    if not state['sfx']:
        return False
    state['native_w']=[warp_repack(state['w'][12],8192),warp_repack(state['w'][13],4096),state['w'][2],state['w'][5],state['w'][7]]
    state['native_scratch']=[torch.zeros(6432,dtype=torch.int32,device=dev),torch.empty((128,4096),dtype=torch.bfloat16,device=dev),torch.empty((2,128,4096),dtype=torch.float32,device=dev)]
    state['native_grid']=state['grid']*16
    for variant in sorted({v[:2] for v in NATIVE_VARIANTS}):
        print('[native resources]',variant,_native_ext(variant).warp_resources(),flush=True)
    return True


def _native_error(out,want):
    if not torch.isfinite(out).all().item():
        raise AssertionError('native nonfinite output')
    return (torch.linalg.vector_norm(out.float()-want.float())/torch.linalg.vector_norm(want.float()).clamp_min(1e-20)).item()


def _native_reference(data,weights):
    # Preparation-only independent FP32 reference on our own synthetic inputs.
    # Packed NVFP4 weights are decoded inside CUDA, never stored as dense weights.
    global _NATIVE_REF_MODULE
    if _NATIVE_REF_MODULE is None:
        major,minor=torch.cuda.get_device_capability()
        arch=f'{major}{minor}'+('a' if major==9 else '')
        _NATIVE_REF_MODULE=load_inline(name='independent_moe_'+hashlib.sha256(NATIVE_REF_CUDA.encode()).hexdigest()[:18]+arch,cpp_sources=NATIVE_REF_CPP,cuda_sources=NATIVE_REF_CUDA,functions=['diagnostic_reference'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False)
    x=data['hidden_states'].contiguous();m=x.shape[0]
    packed=[weights[k].contiguous() for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs','post_norm_w')]
    inter=torch.empty((32,m,4096),dtype=torch.float32,device=x.device)
    y=torch.empty_like(inter);out=torch.empty_like(x)
    cfg=data['config']
    _NATIVE_REF_MODULE.diagnostic_reference(packed,x,data['topk_ids'].to(torch.int32).contiguous(),data['topk_weights'].float().contiguous(),inter,y,out,cfg['local_start'],cfg['local_count'])
    return out


def _native_assert_agreement(data,weights,out,control,context):
    difference=_native_error(out,control)
    if difference<=0.003:return None
    ref=_native_reference(data,weights)
    f=ref.float();delta=out.float()-f
    global_error=(delta.norm()/f.norm().clamp_min(1e-12)).item()
    row_norm=f.norm(dim=-1)
    denominator=torch.maximum(row_norm,(0.1*row_norm.mean()).clamp_min(1e-12))
    worst=(delta.norm(dim=-1)/denominator).max().item()
    old_error=_native_error(control,ref)
    if global_error>0.015 or worst>0.075:
        raise AssertionError(('native independent FP32 reference',context,global_error,worst,'control',old_error))
    return (global_error,old_error,worst)


def _native_calibrate(state,weights,dev):
    import math,statistics
    samples=[]
    empty_times=[]
    reference_stats={}
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    for seed in (47821,73931,95813):
        for m in (1,2,4,5,7,8,9,12,16):
            data=_gen_input(m,seed+m,dev)
            ids=data['topk_ids'];normal_ids=ids.clone()
            variants=[v for v in NATIVE_VARIANTS if v[0]==(8 if m<=8 else 16)]
            for variant in variants:
                # These routing regimes retain unique top8ids, as the task does.
                for regime in ('normal','all_local','empty','skewed'):
                    ids.copy_(normal_ids)
                    if regime=='all_local':ids.copy_(torch.arange(8,device=dev).expand(m,8))
                    elif regime=='empty':ids.fill_(255)
                    elif regime=='skewed':ids[:max(1,m//2)].copy_(torch.arange(8,device=dev).expand(max(1,m//2),8))
                    expected=_native_control_kernel(data,state)
                    candidate_output=_native_run(data,state,variant)
                    repeated_output=_native_run(data,state,variant)
                    torch.cuda.synchronize(dev)
                    if _native_error(candidate_output,repeated_output)>0.001:
                        raise AssertionError(('native repeated-call inconsistency',m,variant,regime))
                    if regime=='empty':
                        if torch.count_nonzero(candidate_output).item():raise AssertionError(('native empty',m,variant))
                    # Different legal K reductions can cross BF16 rounding
                    # boundaries; independent FP32 tests retain the judge bound.
                    else:
                        metrics=_native_assert_agreement(data,weights,candidate_output,expected,(m,variant,regime))
                        if metrics is not None:reference_stats.setdefault((m,variant),[]).append(metrics)
                ids.copy_(normal_ids)
                if seed==47821 and m in (1,8,16):
                    ids.fill_(255)
                    empty_graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(empty_graph):
                        empty_output=_native_run(data,state,variant)
                    empty_samples=[]
                    for _ in range(8):
                        flush.add_(1)
                        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        begin.record();empty_graph.replay();end.record();end.synchronize()
                        empty_samples.append(begin.elapsed_time(end)*1000)
                    if torch.count_nonzero(empty_output).item():raise AssertionError('native empty graph')
                    empty_times.append({'m':m,'variant':variant,'median_us':statistics.median(empty_samples)})
                    del empty_graph
                    ids.copy_(normal_ids)
                eager_native=_native_run(data,state,variant).clone()
                graphs={};outputs={}
                for name in ('control','native'):
                    for _ in range(3):
                        _native_control_kernel(data,state) if name=='control' else _native_run(data,state,variant)
                    graphs[name]=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[name]):
                        outputs[name]=_native_control_kernel(data,state) if name=='control' else _native_run(data,state,variant)
                ratios=[];control_times=[];native_times=[]
                for turn in range(24):
                    order=('control','native') if turn%2==0 else ('native','control')
                    times={}
                    for name in order:
                        flush.add_(1)
                        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        begin.record();graphs[name].replay();end.record();end.synchronize()
                        times[name]=begin.elapsed_time(end)
                    ratios.append(times['control']/times['native'])
                    control_times.append(times['control']*1000);native_times.append(times['native']*1000)
                metrics=_native_assert_agreement(data,weights,outputs['native'],outputs['control'],(m,variant,'captured'))
                if metrics is not None:reference_stats.setdefault((m,variant),[]).append(metrics)
                if _native_error(outputs['native'],eager_native)>0.001:
                    raise AssertionError(('native replay inconsistency',m,variant))
                sample={'seed':seed,'m':m,'variant':variant,'ratio':statistics.median(ratios),'control_us':statistics.median(control_times),'native_us':statistics.median(native_times)}
                samples.append(sample)
                print('[native paired]',seed,m,variant,round(sample['ratio'],6),round(sample['control_us'],2),round(sample['native_us'],2),flush=True)
                del graphs
    print('[native empty startup]',empty_times,flush=True)
    for key,values in reference_stats.items():
        print('[native fp32ref]',key,len(values),tuple(max(v[i] for v in values) for i in range(3)),flush=True)
    selected={}
    for upper,shapes in ((1,(1,)),(4,(2,4)),(8,(5,7,8)),(16,(9,12,16))):
        best_gain=1.03;best=None
        for variant in NATIVE_VARIANTS:
            rows=[r for r in samples if r['variant']==variant and r['m'] in shapes]
            if not rows:continue
            gain=math.exp(sum(math.log(r['ratio']) for r in rows)/len(rows))
            seed_means=[math.exp(sum(math.log(r['ratio']) for r in rows if r['seed']==seed)/len(shapes)) for seed in (47821,73931,95813)]
            per_shape=[statistics.median(r['ratio'] for r in rows if r['m']==m) for m in shapes]
            if gain>best_gain and min(seed_means)>1.02 and min(per_shape)>0.97:
                best_gain,best=gain,variant
        selected[upper]=best
        print('[native selected]',upper,best,best_gain if best is not None else None,flush=True)
    state['native_select']=selected
    if not any(selected.values()):
        del state['native_w'],state['native_scratch']


_native_original_prepare=_codex_original_prepare

def prepare(weights):
    state=_native_original_prepare(weights)
    state['mid_max_m']=24
    if _native_setup(state,weights['w13'].device):
        _native_calibrate(state,weights,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
target=Path(sys.argv[1]) if len(sys.argv)>1 else Path('cuda_native_warp_readcheck_paired.py')
assert target.name==str(target) and target.suffix=='.py'
ast.parse(text);(root/target).write_text(text)
print('Built self-contained native-warp paired candidate; NOT dispatched and H200-UNVERIFIED.')
