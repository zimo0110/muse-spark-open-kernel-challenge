"""Tune safe persistent-grid sizes using generated preparation inputs.

Every candidate uses the existing kernel and grid <= SM count. Search one
seed, validate two independent seeds, and retain control for weak/noisy bands.
No input/output cache survives preparation; only grid-size metadata does.
"""
from pathlib import Path
import ast
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
a=s.index('def kernel_fn(data, state):');b=s.index('\n\ndef ',a+1)
p=s[a:b];marker='    for m0 in range(0, M, CHUNK_TOKENS):'
assert p.count(marker)==1
p=p.replace(marker,'''    launch_grid=state["grid"]
    for upper,grid in state.get("tuned_grids",()):
        if M<=upper:
            launch_grid=grid
            break
'''+marker).replace('state["lo"], state["cnt"], state["grid"])','state["lo"], state["cnt"], launch_grid)')
s=s[:a]+p+s[b:]
s+=r'''

def _grid_calibrate(state,dev):
    import math,statistics,json,time
    start=time.perf_counter();G=state['grid']
    grids=sorted({G,*[max(1,min(G,int(G*q/4+0.5)*4)) for q in (32/33,10/11,28/33,8/11,20/33,16/33)]},reverse=True)
    bands={1:(1,),4:(2,4),8:(5,8),16:(12,16),32:(17,24,32),128:(33,64,96,128),384:(192,256,257)}
    screen={};measurements={};records=[]
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    def module(m):
        d=TINY_DEFINES if m<=TINY_MAX_M else TINY2_DEFINES if m<=TINY2_MAX_M else MID_DEFINES if m<=MID_MAX_M else _main_defines(state['sfx'])
        return _ext(tuple(d)+state['sfx'])
    for si,seed in enumerate((47821,73931,95813)):
        for upper,shapes in bands.items():
            if si==0:test=grids
            else:test=[G,*screen[upper]]
            for m in shapes:
                d=_gen_input(m,seed+m,dev);x,ids,tw=[d[k] for k in ('hidden_states','topk_ids','topk_weights')]
                mod=module(m);outs={g:torch.empty_like(x) for g in test};normal=ids.clone()
                def run(g):mod.okc_run(state['w'],state['scratch'],x,ids,tw,outs[g],state['lo'],state['cnt'],g)
                for regime in ('normal','all_local','empty','skewed'):
                    ids.copy_(normal)
                    if regime=='all_local':ids.copy_(torch.arange(8,device=dev).expand(m,8))
                    elif regime=='empty':ids.fill_(255)
                    elif regime=='skewed':ids[:m//2].copy_(torch.arange(8,device=dev).expand(m//2,8))
                    run(G);want=outs[G].clone()
                    for g in test:
                        if g==G:continue
                        run(g);out=outs[g]
                        delta=(torch.linalg.vector_norm(out.float()-want.float())/torch.linalg.vector_norm(want.float()).clamp_min(1e-20)).item()
                        if delta>0.001 or not torch.isfinite(out).all().item():raise AssertionError(('grid correctness',g,m,regime,delta))
                        if regime=='empty' and torch.count_nonzero(out).item():raise AssertionError(('grid empty',g,m))
                ids.copy_(normal);graphs={}
                for g in test:
                    run(g);run(g);graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):run(g)
                    graphs[g]=graph
                ratios={g:[] for g in test if g!=G}
                for turn in range(12 if si==0 else 20):
                    order=test if turn%2 else list(reversed(test));timing={}
                    for g in order:
                        flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record();graphs[g].replay();b.record();b.synchronize();timing[g]=a.elapsed_time(b)
                    for g in ratios:ratios[g].append(timing[G]/timing[g])
                row={g:statistics.median(v) for g,v in ratios.items()}
                measurements[(seed,m)]=row
                print('[grid]',seed,m,json.dumps(row),flush=True)
            if si==0:
                scores={g:math.exp(sum(math.log(measurements[(seed,m)][g]) for m in shapes)/len(shapes)) for g in grids if g!=G}
                screen[upper]=sorted(scores,key=scores.get,reverse=True)[:2]
    chosen=[]
    for upper,shapes in bands.items():
        eligible=[]
        for g in screen[upper]:
            v=[measurements[(seed,m)][g] for seed in (47821,73931,95813) for m in shapes]
            gain=math.exp(sum(math.log(x) for x in v)/len(v))
            validation=[measurements[(seed,m)][g] for seed in (73931,95813) for m in shapes]
            vg=math.exp(sum(math.log(x) for x in validation)/len(validation))
            if gain>1.025 and vg>1.02 and min(v)>0.985:eligible.append((gain,g))
        gain,g=max(eligible) if eligible else (1.,G)
        chosen.append((upper,g));print('[grid selected]',upper,g,gain,flush=True)
    state['tuned_grids']=chosen
    # Exercise the chosen per-shape grids on fresh data at fixed addresses.
    for m in (1,8,16,32,128,256):
        g=next(g for upper,g in chosen if m<=upper)
        if g==G:continue
        data=_gen_input(m,198337+m,dev);x,ids,tw=[data[k] for k in ('hidden_states','topk_ids','topk_weights')];out=torch.empty_like(x);mod=module(m)
        def run():mod.okc_run(state['w'],state['scratch'],x,ids,tw,out,state['lo'],state['cnt'],g)
        run();torch.cuda.synchronize();want=out.clone();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run()
        largest=0.
        for batch in range(50):
            for replay in range(20):flush.add_(1);graph.replay()
            torch.cuda.synchronize();delta=(torch.linalg.vector_norm(out.float()-want.float())/torch.linalg.vector_norm(want.float()).clamp_min(1e-20)).item()
            if delta>0.001 or not torch.isfinite(out).all().item():raise AssertionError(('grid soak',m,g,delta))
            largest=max(largest,delta)
        print('[grid soak]',m,g,1000,largest,flush=True)
    print('[grid total seconds]',time.perf_counter()-start,flush=True)

def prepare(weights):
    state=_codex_original_prepare(weights)
    _grid_calibrate(state,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
ast.parse(s);(R/'cuda_grid_tuned.py').write_text(s)
print('Prepared three-seed, nearby-shape grid calibration; no dispatch.')
