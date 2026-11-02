"""Prepare guarded fixed-N32 private check; no dispatch or best-file mutation."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
old='ext = _ext(tuple(_main_defines(sfx)) + sfx)'
assert s.count(old)==1
s=s.replace(old,'ext = _ext(tuple(state.get("fixed32_defines", _main_defines(sfx)) if M <= 32 else _main_defines(sfx)) + sfx)')
assert s.count('MID_MAX_M = 16')==1;s=s.replace('MID_MAX_M = 16','MID_MAX_M = 24')
trial=next(q for q in json.loads((R/'research/spark/fixed_width_cases.json').read_text()) if q['name']=='fixed_n32')
s+='\nALT_SRCS["fixed_n32"] = '+repr(trial['source'])+'\n'
s+=r'''
# Preparation-only trials on generated inputs; retain only a kernel choice.
def _fixed32_calibrate(state,dev):
    import math,statistics,time
    started=time.perf_counter()
    if not state['sfx']:return
    defs={'control':tuple(MAIN_DEFINES),'fixed_n32':tuple(MAIN_DEFINES)+('@fixed_n32',)}
    modules={k:_ext(v+state['sfx']) for k,v in defs.items()}
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    samples=[];byshape={m:[] for m in (25,28,32)};byseed=[]
    for seed in (47821,73931,95813):
        ratios=[]
        for m in byshape:
            data=_gen_input(m,seed+m,dev)
            x,ids,tw=[data[k] for k in ('hidden_states','topk_ids','topk_weights')]
            outputs={k:torch.empty_like(x) for k in modules}
            def run(name):
                modules[name].okc_run(state['w'],state['scratch'],x,ids,tw,outputs[name],state['lo'],state['cnt'],state['grid'])
            normal=ids.clone()
            for regime in ('normal','all_local','duplicates','empty','skewed'):
                ids.copy_(normal)
                if regime=='all_local':ids.copy_(torch.arange(8,device=dev).expand(m,8))
                elif regime=='duplicates':ids.fill_(0)
                elif regime=='empty':ids.fill_(255)
                elif regime=='skewed':ids[:m//2].copy_(torch.arange(8,device=dev).expand(m//2,8))
                run('control');expected=outputs['control'].clone()
                run('fixed_n32')
                actual=outputs['fixed_n32'].clone()
                delta=(torch.linalg.vector_norm(actual.float()-expected.float())/
                       torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()
                if delta>0.001 or not torch.isfinite(actual).all().item():
                    raise AssertionError(('fixed N32 control mismatch',m,regime,delta))
                if regime=='empty' and torch.count_nonzero(actual).item():raise AssertionError(('fixed N32 empty',m))
                run('fixed_n32')
                drift=(torch.linalg.vector_norm(outputs['fixed_n32'].float()-actual.float())/
                       torch.linalg.vector_norm(actual.float()).clamp_min(1e-20)).item()
                if drift>0.001:raise AssertionError(('fixed N32 repeated-call drift',m,regime,drift))
            ids.copy_(normal)
            graphs={}
            for name in modules:
                run(name);run(name)
                graphs[name]=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):run(name)
            observations=[]
            for turn in range(24):
                order=('control','fixed_n32') if turn%2 else ('fixed_n32','control')
                timing={}
                for name in order:
                    flush.add_(1)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graphs[name].replay();b.record();b.synchronize()
                    timing[name]=a.elapsed_time(b)*1000
                observations.append(timing)
            ratio=statistics.median(q['control']/q['fixed_n32'] for q in observations)
            ratios.append(ratio);byshape[m].append(ratio)
            row={'seed':seed,'M':m,'control_over_trial':ratio,
                 'control_us':statistics.median(q['control'] for q in observations),
                 'trial_us':statistics.median(q['fixed_n32'] for q in observations)}
            samples.append(row);print('[fixed-n32] '+json.dumps(row),flush=True)
            if seed==47821:
                largest=0.0;different=0
                for epoch in range(2):
                    fresh=_gen_input(m,seed+917237+epoch*191+m,dev)
                    x.copy_(fresh['hidden_states']);ids.copy_(fresh['topk_ids']);tw.copy_(fresh['topk_weights'])
                    if epoch:ids.copy_(torch.arange(8,device=dev).expand(m,8))
                    run('fixed_n32');torch.cuda.synchronize(dev)
                    expected=outputs['fixed_n32'].clone()
                    for batch in range(50):
                        for replay in range(20):flush.add_(1);graphs['fixed_n32'].replay()
                        torch.cuda.synchronize(dev)
                        actual=outputs['fixed_n32']
                        drift=(torch.linalg.vector_norm(actual.float()-expected.float())/
                               torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()
                        if drift>0.001 or not torch.isfinite(actual).all().item():
                            raise AssertionError(('fixed N32 queued drift',m,epoch,batch,drift))
                        largest=max(largest,drift);different+=int(not torch.equal(actual,expected))
                print('[fixed-n32-soak]',m,'replays',2000,'max_relative_L2',largest,'different_checkpoints',different,flush=True)
            del graphs
        byseed.append(math.exp(sum(math.log(q) for q in ratios)/len(ratios)))
    gain=math.exp(sum(math.log(q['control_over_trial']) for q in samples)/len(samples))
    choose=(gain>1.03 and min(byseed)>1.02 and
            min(statistics.median(q) for q in byshape.values())>0.97 and
            statistics.median(byshape[32])>1.02)
    state['fixed32_defines']=defs['fixed_n32' if choose else 'control']
    print('[fixed-n32] selected','fixed_n32' if choose else 'control','geomean',gain,
          'seed_geomeans',byseed,'seconds',time.perf_counter()-started,flush=True)


def prepare(weights):
    state=_codex_original_prepare(weights)
    _fixed32_calibrate(state,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
# Original module imports json only locally in diagnostics. This own helper uses
# it locally as well, leaving timed code untouched apart from the shape choice.
s=s.replace('    import math,statistics,time\n','    import math,statistics,time,json\n')
ast.parse(s);(R/'cuda_persistent_fixed32_paired.py').write_text(s)
print('Prepared fixed-N32 guarded private candidate; MID through24, newpath only25..32; no H200 dispatch.')
