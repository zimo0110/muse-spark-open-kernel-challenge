"""Independent-reference validation of the GB10 development subset, never submits."""
import json,statistics,time,secrets
from pathlib import Path
import torch
import spark_tiny_port as candidate
import reference

def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.cuda.get_device_properties(0)
    print(json.dumps({'GPU':device.name,'capability':torch.cuda.get_device_capability(),'SMs':device.multi_processor_count,'shared_optin':getattr(device,'shared_memory_per_block_optin',None),'torch':torch.__version__,'CUDA':torch.version.cuda}),flush=True)
    start=time.monotonic()
    weights=reference.generate_weights(secrets.randbits(62))
    state=candidate.prepare(weights)
    torch.cuda.synchronize()
    print('prepare_seconds',round(time.monotonic()-start,3),flush=True)
    for defs,ext in candidate._EXTS.items():
        print('variant_resources',{'MID':any(d.startswith('@') for d in defs),'resources':ext.okc_resources()},flush=True)
    shapes=(1,2,4,8,9,16)
    cases=[]
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for m in shapes:
            data=reference.generate_input(m,'ep8',secrets.randbits(62))
            candidate.kernel_fn(data,state);candidate.kernel_fn(data,state)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                out=candidate.kernel_fn(data,state)
            cases.append((m,data,graph,out))
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    results=[]
    for regime in ('normal','skewed','all-local','empty'):
        for m,data,graph,out in cases:
            fresh=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
            if regime=='empty':fresh['topk_ids'].fill_(255)
            if regime=='all-local':fresh['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
            for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
        for turn in range(8):
            order=cases if turn%2 else reversed(cases)
            for m,data,graph,out in order:graph.replay()
        torch.cuda.synchronize()
        for m,data,graph,out in cases:
            if regime=='empty':
                ok=torch.count_nonzero(out).item()==0;message='empty routes exact zero'
            else:ok,message=reference.check_implementation(data,out,weights)
            print('reference',m,regime,ok,message,flush=True)
            results.append({'M':m,'regime':regime,'passed':ok,'message':message})
            if not ok:raise AssertionError((m,regime,message))
    # Fresh ordinary inputs for development-only timing. These are GB10 times,
    # not a challenge score and not a predictor of Hopper ranking.
    latency={}
    flush=torch.zeros(32*1024*1024,dtype=torch.int32,device='cuda')
    for m,data,graph,out in cases:
        fresh=reference.generate_input(m,'ep8',secrets.randbits(62))
        for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
        samples=[]
        for _ in range(12):
            flush.add_(1)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graph.replay();b.record();b.synchronize()
            samples.append(a.elapsed_time(b)*1000)
        latency[str(m)]=statistics.median(samples)
    result={'passed':True,'GPU':device.name,'reference_cases':results,'graph_replays':len(shapes)*4*8,'GB10_only_median_us':latency,'H200_score':None}
    Path('validation_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GB10_VALIDATION_RESULT='+json.dumps(result),flush=True)

if __name__=='__main__':main()
