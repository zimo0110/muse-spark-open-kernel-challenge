"""Validate profiling math/counters on GB10; no H200 speed claims."""
import json,secrets,hashlib,time
from pathlib import Path
import torch
import reference
from checks import compare
import spark_wgmma_emulate as emulator
from phase_probe_report import decode
R=Path(__file__).resolve().parent

def expected_traffic(ids):
    local=ids[(ids>=0)&(ids<32)].to(torch.int64)
    count=torch.bincount(local,minlength=32).cpu().tolist()
    jobs=[]
    for n in count:
        if n<=8:
            if n:jobs.append(n)
        else:
            while n:jobs.append(min(n,64));n-=min(n,64)
    return {'weight_A':(len(jobs)*512,len(jobs)*512*36864),
            'weight_B':(len(jobs)*256,len(jobs)*256*36864),
            'activation_A':(len(jobs)*512,sum(((n+7)//8)*8*512*512 for n in jobs)),
            'activation_B':(len(jobs)*256,sum(((n+7)//8)*8*512*256 for n in jobs))}

def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    cases={q['name']:q for q in json.loads((R/'phase_probe_cases.json').read_text())}
    weights=reference.generate_weights(secrets.randbits(62))
    allcases={}
    for m in (4,8,16,32,33,128,256,257):
        for regime in ('normal','all_local','empty','skewed'):
            data=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
            if regime=='empty':data['topk_ids'].fill_(255)
            if regime=='all_local':data['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
            want=torch.zeros_like(data['hidden_states']) if regime=='empty' else reference.ref_kernel(data,weights)
            allcases[m,regime]=(data,want)
    results=[]
    for name in ('phase_main_coarse','phase_main_detail','phase_mid_coarse'):
        case=cases[name];wgs='main' in name
        defines=list(case['defines'])
        if not wgs:
            defines=[d for d in defines if not d.startswith('-DOKC_FAST1=')]
            defines+=['-DOKC_NS=1','-DOKC_NSF=1','-DOKC_FAST1=1']
        core=emulator.load_core(name,case['source'],defines)
        if not wgs:
            # GB10 has a101KiB block limit. A skinny-only build does not use
            # the fat-path activation buffers; omit only that unused allocation.
            marker='  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES'
            assert core.CUDA_SRC.count(marker)==1
            core.CUDA_SRC=core.CUDA_SRC.replace(marker,'#if OKC_TINY\n  return base+(size_t)OKC_NS*okc::UNIT_BYTES;\n#endif\n'+marker)
        started=time.monotonic();state=core.prepare(weights)
        print('PHASE_RESOURCES',name,[ext.okc_resources() for ext in core._EXTS.values()],flush=True)
        debug=torch.zeros(state['grid']*40,dtype=torch.int64,device='cuda')
        state['scratch'].append(debug)
        records=[];graphs=[]
        for (m,regime),(data,want) in allcases.items():
            if not wgs and m>16:continue
            core.kernel_fn(data,state);core.kernel_fn(data,state)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=core.kernel_fn(data,state)
            graphs.append((m,regime,data,want,graph,out))
        for turn in range(8):
            for m,regime,data,want,graph,out in (graphs if turn%2 else reversed(graphs)):graph.replay()
        torch.cuda.synchronize()
        for m,regime,data,want,graph,out in graphs:
            ok,message=compare(out,want,reference.REL_L2_TOL) if regime!='empty' else (torch.count_nonzero(out).item()==0,'empty zero')
            assert ok,(name,m,regime,message)
            debug.zero_();graph.replay();torch.cuda.synchronize()
            rows=debug.view(state['grid'],40).cpu().tolist();decoded=decode(rows,wgs)
            if name.endswith('detail'):
                expected=expected_traffic(data['topk_ids'])
                for off,label in ((8,'weight_A'),(16,'weight_B'),(24,'activation_A'),(32,'activation_B')):
                    stages=sum(r[off+7] for r in rows);b=sum(r[off+6] for r in rows)
                    assert (stages,b)==expected[label],(name,m,regime,label,(stages,b),expected[label])
                    assert all(r[off+4]<=r[off+1]-r[off] for r in rows)
            records.append({'M':m,'regime':regime,'passed':True,'reference':message})
            print('PHASE_VALIDATION',name,m,regime,'PASS',flush=True)
        results.append({'name':name,'passed':True,'source_sha256':hashlib.sha256(case['source'].encode()).hexdigest(),
                        'cases':records,'queued_graph_replays':len(graphs)*8,'counter_totals_verified':name.endswith('detail'),
                        'seconds':round(time.monotonic()-started,3)})
        (R/'phase_probe_validation_partial.json').write_text(json.dumps({'variants':results},indent=2)+'\n')
        del graphs,state,core
        torch.cuda.synchronize()
    report={'hardware':'GB10','Hopper_async_validation':False,'H200_performance':None,'variants':results}
    (R/'phase_probe_validation_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PHASE_PROBE_VALIDATION_COMPLETE',[(v['name'],len(v['cases']),v['queued_graph_replays']) for v in results],flush=True)

if __name__=='__main__':main()
