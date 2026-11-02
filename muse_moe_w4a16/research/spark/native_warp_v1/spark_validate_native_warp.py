"""Independent reference/graph tests of the actual native warp kernel on GB10."""
from pathlib import Path
import json,secrets,time
import torch
import reference
from checks import compare
import warp_moe_port as candidate
ROOT=Path(__file__).resolve().parent

def sample(m,regime):
    d=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
    if regime=='all-local':d['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
    elif regime=='empty':d['topk_ids'].fill_(255)
    elif regime=='duplicates':d['topk_ids'].fill_(0)
    return d

def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    weights=reference.generate_weights(secrets.randbits(62));state=candidate.prepare(weights)
    records=[];replays=0;start=time.monotonic()
    for variant in ((8,4),(8,6),(16,4)):
        cases=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for m in (1,2,4,5,7,8,9,12,16):
                if m>variant[0]:continue
                for regime in ('normal','skewed','all-local','duplicates','empty'):
                    data=sample(m,regime)
                    candidate.run(data,state,variant);candidate.run(data,state,variant)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):out=candidate.run(data,state,variant)
                    cases.append((m,regime,data,graph,out))
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        for epoch in range(2):
            wants=[]
            for m,regime,data,graph,out in cases:
                fresh=sample(m,regime)
                for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
                wants.append(torch.zeros_like(data['hidden_states']) if regime=='empty' else reference.ref_kernel(data,weights))
            for turn in range(4):
                for m,regime,data,graph,out in (cases if turn%2 else reversed(cases)):
                    graph.replay();replays+=1
            torch.cuda.synchronize()
            for (m,regime,data,graph,out),want in zip(cases,wants):
                ok,message=compare(out,want,reference.REL_L2_TOL) if regime!='empty' else (torch.count_nonzero(out).item()==0,'empty exact zero')
                control_variant=(8,4) if m<=8 else (16,4)
                control=candidate.run(data,state,control_variant)
                torch.cuda.synchronize()
                norm=torch.linalg.vector_norm(control.float()).clamp_min(1e-20)
                self_error=(torch.linalg.vector_norm(out.float()-control.float())/norm).item()
                row={'variant':variant,'M':m,'regime':regime,'epoch':epoch,'passed':ok,'message':message}
                row['against_control_error']=self_error
                records.append(row);print('NATIVE_WARP_REFERENCE',row,flush=True)
                if not ok or self_error>0.001:raise AssertionError(row)
        del cases
        torch.cuda.synchronize()
    result={'passed':True,'hardware':'GB10','native_MMA_TMA':True,'H200_speedup':None,'reference_cases':records,'graph_replays':replays,'seconds':time.monotonic()-start}
    (ROOT/'native_warp_validation_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('NATIVE_WARP_VALIDATION_COMPLETE',len(records),replays,flush=True)

if __name__=='__main__':main()
