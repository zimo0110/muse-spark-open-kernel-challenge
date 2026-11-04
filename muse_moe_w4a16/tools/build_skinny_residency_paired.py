"""Guarded small-batch residency comparison; no best-file mutation or dispatch."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
s=s.replace('functions=["okc_run", "okc_set_pdl"],','functions=["okc_run", "okc_set_pdl"] + (["okc_resources"] if any(d.startswith("@skinny_") for d in defines) else []),')
s+='\nCPP_SRC += "\\nstd::vector<int64_t> okc_resources();\\n"\n'
cases=json.loads((R/'research/spark/skinny_residency_cases.json').read_text())
for label in ('skinny_nw8_b1','skinny_nw4_b2'):
 c=next(q for q in cases if q['name']==label)
 s+='\nALT_SRCS['+repr(label)+']='+repr(c['source'])+'\n'
# Local launch choices are immutable prepared metadata, never input results.
a=s.index('def kernel_fn(data, state):');b=s.index('\n\ndef ',a+1)
part=s[a:b]
marker='    for m0 in range(0, M, CHUNK_TOKENS):'
assert part.count(marker)==1
part=part.replace(marker,'''    launch_grid=state["grid"]
    if M<=16 and state.get("skinny_choices"):
        band=4 if M<=4 else 8 if M<=8 else 16
        if band in state["skinny_choices"]:
            defs,launch_grid=state["skinny_choices"][band]
            ext=_ext(tuple(defs)+sfx)
'''+marker)
part=part.replace('state["lo"], state["cnt"], state["grid"])','state["lo"], state["cnt"], launch_grid)')
s=s[:a]+part+s[b:]
s+=r'''

def _skinny_residency_calibrate(state,dev):
    import math,statistics,json,time
    start=time.perf_counter()
    if not state['sfx']:return
    base=state['grid']; variants={};resources={}
    for label,nw,blocks in [('skinny_nw8_b1',8,1),('skinny_nw4_b2',4,2)]:
        defs=tuple(d for d in TINY_DEFINES if not d.startswith('-DOKC_NW='))+(f'-DOKC_NW={nw}','-DOKC_NS=2','-DOKC_NSF=1','-DOKC_SPLIT=1','@'+label)
        mod=_ext(defs+state['sfx']);res=mod.okc_resources();resources[label]=res
        variants[label]=(mod,base,defs)
        if nw==4 and res[0]>=2:variants['skinny_nw4_grid2']=(mod,base*2,defs)
    print('[skinny resources]',json.dumps(resources),flush=True)
    main=_ext(tuple(MAIN_DEFINES)+state['sfx'])
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    samples=[]
    for seed in (47821,73931,95813):
        for m in (1,4,8,12,16):
            cd=TINY_DEFINES if m<=4 else TINY2_DEFINES if m<=8 else MID_DEFINES
            choices={'control':(_ext(tuple(cd)+state['sfx']),base,tuple(cd)),**variants}
            data=_gen_input(m,seed+m,dev);x,ids,tw=[data[k] for k in ('hidden_states','topk_ids','topk_weights')]
            outputs={k:torch.empty_like(x) for k in choices};normal=ids.clone()
            def run(name):
                mod,grid,_=choices[name]
                mod.okc_run(state['w'],state['scratch'],x,ids,tw,outputs[name],state['lo'],state['cnt'],grid)
            for regime in ('normal','all_local','empty','skewed','duplicates'):
                ids.copy_(normal)
                if regime=='all_local':ids.copy_(torch.arange(8,device=dev).expand(m,8))
                elif regime=='duplicates':ids.fill_(0)
                elif regime=='empty':ids.fill_(255)
                elif regime=='skewed':ids[:m//2].copy_(torch.arange(8,device=dev).expand(m//2,8))
                if regime=='duplicates':
                    main.okc_run(state['w'],state['scratch'],x,ids,tw,outputs['control'],state['lo'],state['cnt'],base)
                else:run('control')
                expected=outputs['control'].clone()
                for name in variants:
                    run(name);actual=outputs[name].clone()
                    drift=(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()
                    if drift>0.001 or not torch.isfinite(actual).all().item():raise AssertionError(('skinny correctness',name,m,regime,drift))
                    if regime=='empty' and torch.count_nonzero(actual).item():raise AssertionError(('skinny empty',name,m))
            ids.copy_(normal)
            graphs={}
            for name in choices:
                run(name);run(name);g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):run(name)
                graphs[name]=g
            times={k:[] for k in choices};ratios={k:[] for k in variants}
            for turn in range(24):
                order=list(choices)
                if turn%2:order.reverse()
                obs={}
                for name in order:
                    flush.add_(1);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graphs[name].replay();b.record();b.synchronize();obs[name]=a.elapsed_time(b)*1000;times[name].append(obs[name])
                for name in variants:ratios[name].append(obs['control']/obs[name])
            row={'seed':seed,'M':m,'ratios':{k:statistics.median(v) for k,v in ratios.items()},'us':{k:statistics.median(v) for k,v in times.items()}}
            samples.append(row);print('[skinny paired]',json.dumps(row),flush=True)
            # Target queued-call check includes fresh data at the same addresses.
            if seed==47821 and m in (1,8,16):
                for name in variants:
                    largest=0.;different=0
                    for epoch in range(2):
                        fresh=_gen_input(m,seed+epoch*719+m+91337,dev)
                        x.copy_(fresh['hidden_states']);ids.copy_(fresh['topk_ids']);tw.copy_(fresh['topk_weights'])
                        if epoch:ids.copy_(torch.arange(8,device=dev).expand(m,8))
                        run(name);torch.cuda.synchronize();expected=outputs[name].clone()
                        for batch in range(25):
                            for replay in range(20):flush.add_(1);graphs[name].replay()
                            torch.cuda.synchronize();actual=outputs[name]
                            drift=(torch.linalg.vector_norm(actual.float()-expected.float())/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()
                            if drift>0.001 or not torch.isfinite(actual).all().item():raise AssertionError(('skinny queued',name,m,epoch,batch,drift))
                            largest=max(largest,drift);different+=int(not torch.equal(actual,expected))
                    print('[skinny soak]',name,m,'replays',1000,'max_relative_L2',largest,'different_checkpoints',different,flush=True)
    state['skinny_choices']={}
    for band,shapes in ((4,(1,4)),(8,(8,)),(16,(12,16))):
        eligible=[]
        for name in variants:
            rows=[r for r in samples if r['M'] in shapes];v=[r['ratios'][name] for r in rows]
            gain=math.exp(sum(math.log(q) for q in v)/len(v))
            # Require a gain for every measured seed/shape, not just a mean win.
            if gain>1.03 and min(v)>1.015:eligible.append((gain,name))
        if eligible:
            gain,name=max(eligible);_,grid,defs=variants[name];state['skinny_choices'][band]=(defs,grid)
            print('[skinny selected]',band,name,gain,flush=True)
        else:print('[skinny selected]',band,'control',flush=True)
    print('[skinny done seconds]',time.perf_counter()-start,flush=True)

def prepare(weights):
    state=_codex_original_prepare(weights)
    _skinny_residency_calibrate(state,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
ast.parse(s);(R/'cuda_skinny_residency_paired.py').write_text(s)
print('Prepared occupancy-gated paired residency candidate; no dispatch.')
