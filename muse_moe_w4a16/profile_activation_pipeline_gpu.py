"""Hopper-only development A/B and hardware profiling runner. Never submits jobs.

Example:
 python profile_activation_pipeline_gpu.py --shapes 16 32 256 --modes direct compact warp

For hardware counters on a developmentGPU, run under Nsight Compute:
 ncu --set full --kernel-name 'regex:(_stream_dispatch|_ws_dispatch)' --launch-count 2 \
   python profile_activation_pipeline_gpu.py --shapes 32 --modes compact --tiles 16 --rounds 2 --soak 1

Use counters to distinguish DRAM saturation, decoder/ALU throughput, barrier
stalls, register occupancy and Tensor Core utilization. Kernel ablations alone
are not additive cost measurements. No alternate challenge problem/account is used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import secrets

ROOT=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def capture(torch,fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn();fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    return graph,out


def traffic(data,torch,bm):
    ids=data['topk_ids'].flatten()
    counts=torch.bincount(ids[(ids>=0)&(ids<32)].long(),minlength=32).cpu().numpy()
    h=data['hidden_states'].shape[1];i=data['config']['d_expert']
    active=int((counts>0).sum());tiles=int(((counts+bm-1)//bm).sum())
    # FP4 .5byte/value + FP8 group16scales. Global scales are negligible.
    unique=active*3*h*i*(.5+1/16)
    issued=tiles*3*h*i*(.5+1/16)
    return {'local_routes':int(counts.sum()),'active_experts':active,'token_tiles':tiles,
            'tile_weight_bytes_requested':int(issued),'unique_weight_bytes':int(unique),
            'weight_reload_factor':tiles/max(active,1),
            'note':'request model, not measured DRAM traffic; L2 may serve repeated tiles'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shapes',type=int,nargs='+',default=[4,8,16,32,33,128,256,257])
    parser.add_argument('--modes',nargs='+',choices=['direct','compact','warp'],default=['direct','compact','warp'])
    parser.add_argument('--tiles',type=int,nargs='+',choices=[8,16,32],default=[8,16,32])
    parser.add_argument('--soak',type=int,default=16)
    parser.add_argument('--rounds',type=int,default=20)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0]!=9:
        parser.error('Requires a Hopper CUDA developmentGPU. CPU semantic checks are a separate script.')
    ref=load(ROOT/'reference.py','development_reference')
    candidate=load(ROOT/'register_warp_pipeline.py','development_pipeline')
    weights=ref.generate_weights(secrets.randbits(63))
    state=candidate._checked_prepare(weights)
    state['activation_choices']={};state['activation_default']={'mode':0}
    flush=torch.ones(64<<20,dtype=torch.float32,device='cuda')
    report=[]
    for m in args.shapes:
        data=ref.generate_input(m,'ep8',secrets.randbits(63))
        baseline,baseline_out=capture(torch,lambda:candidate._checked_kernel_fn(data,state))
        for mode in args.modes:
            for bm in args.tiles:
                state['activation_choices'][m]={'mode':{'direct':1,'compact':2,'warp':3}[mode],
                    'bm13':bm,'bm2':bm,'cap':128,'pf':False}
                graph,out=capture(torch,lambda:candidate.kernel_fn(data,state))
                for step in range(max(1,args.soak)):
                    fresh=ref.generate_input(m,'ep8',secrets.randbits(63),dist='skewed' if step%2 else'normal')
                    if step%4==2:fresh['topk_ids'].fill_(255)  # no local work, must output zeros
                    for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
                    graph.replay();baseline.replay();torch.cuda.synchronize()
                    ok,message=ref.check_implementation(data,out,weights)
                    if not ok:raise AssertionError((m,mode,bm,step,message))
                    delta=(out.float()-baseline_out.float()).norm()/baseline_out.float().norm().clamp(min=1e-20)
                    if not torch.isfinite(delta) or float(delta)>.004:
                        raise AssertionError(('checked-path mismatch',m,mode,bm,step,float(delta)))
                # Reset the deliberately empty routes before performance measurement.
                fresh=ref.generate_input(m,'ep8',secrets.randbits(63))
                for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
                events=[]
                for rep in range(args.rounds):
                    order=[('control',baseline),('candidate',graph)]
                    if rep%2:order.reverse()
                    pair={}
                    for label,g in order:
                        flush.sum()
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();g.replay();b.record();pair[label]=(a,b)
                    events.append(pair)
                torch.cuda.synchronize()
                control=[p['control'][0].elapsed_time(p['control'][1])*1000 for p in events]
                trial=[p['candidate'][0].elapsed_time(p['candidate'][1])*1000 for p in events]
                def median(xs):return sorted(xs)[len(xs)//2]
                row={'m':m,'mode':mode,'bm':bm,'control_raw_graph_us':median(control),
                     'candidate_raw_graph_us':median(trial),'median_paired_ratio':median([a/b for a,b in zip(control,trial)]),
                     'traffic':traffic(data,torch,bm)}
                report.append(row);print(json.dumps(row),flush=True)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Development checks complete. These measurements are not official challenge results.')


if __name__=='__main__':main()
