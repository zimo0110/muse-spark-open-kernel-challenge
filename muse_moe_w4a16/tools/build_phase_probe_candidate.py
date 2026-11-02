"""Prepare one self-contained private diagnostic candidate, never dispatch."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
cases={q['name']:q for q in json.loads((R/'research/spark/phase_probe_cases.json').read_text())}
for name in ('phase_main_coarse','phase_main_detail','phase_mid_coarse'):
 s+='\nALT_SRCS['+repr(name)+'] = '+repr(cases[name]['source'])+'\n'
report=(R/'tools/phase_probe_report.py').read_text().replace('def percentiles(', 'def _phase_percentiles(').replace('percentiles([','_phase_percentiles([').replace('def decode(', 'def _phase_decode(')
s+='\n'+report+'\n'
s+=r'''
# All tracing uses our own synthetic preparation inputs. The timed kernel_fn
# and published source/selection remain unchanged; no traces survive in state.
def _phase_probe_prepare(state,dev):
    import json,statistics,time
    started=time.perf_counter()
    if not state['sfx']:
        print('[phase-probe] skipped: PRMT scales unavailable',flush=True)
        return
    main=tuple(MAIN_DEFINES)
    mid=tuple(d for d in MID_DEFINES if not d.startswith('@'))
    definitions={
        'main_coarse':main+('@phase_main_coarse',),
        'main_detail':main+('@phase_main_detail',),
        'mid_coarse':mid+('@phase_mid_coarse',),
    }
    modules={k:_ext(d+state['sfx']) for k,d in definitions.items()}
    flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
    grid=state['grid']
    for seed in (47821,73931):
        for m in (16,32,128,256):
            wgs=m>16
            control=_ext((tuple(MAIN_DEFINES) if wgs else tuple(MID_DEFINES))+state['sfx'])
            modes={'control':control}
            modes.update({k:modules[k] for k in (('main_coarse','main_detail') if wgs else ('mid_coarse',))})
            data=_gen_input(m,seed+m,dev)
            x,ids,tw=[data[k] for k in ('hidden_states','topk_ids','topk_weights')]
            outputs={k:torch.empty_like(x) for k in modes}
            traces={k:torch.zeros(grid*40,dtype=torch.int64,device=dev) for k in modes if k!='control'}
            work={k:list(state['scratch'])+([traces[k]] if k!='control' else []) for k in modes}
            def run(k):
                modes[k].okc_run(state['w'],work[k],x,ids,tw,outputs[k],state['lo'],state['cnt'],grid)
            normal=ids.clone()
            for regime in ('normal','all_local','empty','skewed'):
                ids.copy_(normal)
                if regime=='all_local':ids.copy_(torch.arange(8,device=dev).expand(m,8))
                elif regime=='empty':ids.fill_(255)
                elif regime=='skewed':ids[:m//2].copy_(torch.arange(8,device=dev).expand(m//2,8))
                run('control')
                expected=outputs['control'].clone()
                for name in traces:
                    run(name)
                    delta=(torch.linalg.vector_norm(outputs[name].float()-expected.float())/
                           torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()
                    if delta>0.001 or not torch.isfinite(outputs[name]).all().item():
                        raise AssertionError(('phase probe output mismatch',name,m,regime,delta))
                    if regime=='empty' and torch.count_nonzero(outputs[name]).item():
                        raise AssertionError(('phase probe empty output',name,m))
            ids.copy_(normal)
            graphs={}
            for name in modes:
                run(name);run(name)
                graphs[name]=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):run(name)
            samples=[]
            names=list(modes)
            for turn in range(18):
                shift=turn%len(names)
                order=names[shift:]+names[:shift]
                if (turn//len(names))%2:order=order[::-1]
                times={}
                for name in order:
                    flush.add_(1)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graphs[name].replay();b.record();b.synchronize()
                    times[name]=a.elapsed_time(b)*1000
                samples.append(times)
            overhead={name:statistics.median(v[name]/v['control'] for v in samples) for name in traces}
            # A slow probe may still show ordering, but its durations cannot be
            # treated as unperturbed. Never auto-tune the timed kernel from it.
            record={'seed':seed,'M':m,
                    'cold_graph_us':{k:round(statistics.median(v[k] for v in samples),3) for k in modes},
                    'probe_over_control':{k:round(v,5) for k,v in overhead.items()},
                    'overhead_within_3pct':{k:abs(v-1)<=0.03 for k,v in overhead.items()}}
            for name in traces:
                traces[name].zero_();flush.add_(1);graphs[name].replay();torch.cuda.synchronize(dev)
                rows=traces[name].view(grid,40).cpu().tolist()
                decoded=_phase_decode(rows,wgs)
                if name.endswith('detail'):
                    record['producer']=decoded['producer']
                else:
                    decoded.pop('producer',None)
                    record['consumer']=decoded
            print('[phase-probe] '+json.dumps(record,separators=(',',':')),flush=True)
            del graphs,work,traces,outputs
    print('[phase-probe] preparation_seconds',round(time.perf_counter()-started,3),flush=True)


def prepare(weights):
    state=_codex_original_prepare(weights)
    _phase_probe_prepare(state,weights['w13'].device)
    _codex_replay_stress(state,weights['w13'].device)
    return state
'''
ast.parse(s)
(R/'cuda_persistent_phase_probe.py').write_text(s)
print('Prepared private phase profiler. Timed kernel/selection unchanged; NOT dispatched.')
