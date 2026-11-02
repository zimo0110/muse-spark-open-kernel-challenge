"""Fresh-input queued replay soak, no change to checked GPU algorithm."""
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_aligned_safe.py').read_text()
text+=r'''

_codex_soak_base_prepare=prepare

def _codex_queued_soak(state,dev):
    shapes=(1,4,8,16,17,32,33,65,128,256,257,385)
    cases=[]
    stream=torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        for m in shapes:
            data=_gen_input(m,110003+m,dev)
            kernel_fn(data,state);kernel_fn(data,state)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                output=kernel_fn(data,state)
            cases.append((m,data,graph,output))
    torch.cuda.current_stream(dev).wait_stream(stream)
    torch.cuda.synchronize(dev)
    worst=0.0; bitdiff=0
    for block in range(100):
        if block%10==0:
            expected=[]
            for m,data,graph,output in cases:
                fresh=_gen_input(m,120011+101*block+m,dev)
                if block%20:
                    fresh['topk_ids'].copy_(torch.arange(8,device=dev).expand(m,8))
                for key in ('hidden_states','topk_ids','topk_weights'):
                    data[key].copy_(fresh[key])
                expected.append(kernel_fn(data,state).clone())
            torch.cuda.synchronize(dev)
        # Twenty complete mixed-shape graph cycles without ANY host sync or
        # intervening device validation/copy kernels. Observe outputs afterward.
        for turn in range(20):
            order=cases if (turn+block)%2 else reversed(cases)
            for m,data,graph,output in order:
                graph.replay()
        torch.cuda.synchronize(dev)
        for (m,data,graph,output),want in zip(cases,expected):
            finite=torch.isfinite(output).all().item()
            err=(torch.linalg.vector_norm(output.float()-want.float())/torch.linalg.vector_norm(want.float()).clamp_min(1e-20)).item()
            worst=max(worst,err)
            bitdiff+=int(not torch.equal(output,want))
            if not finite or err>0.001:
                raise AssertionError(('queued replay instability',m,block,finite,err))
    print(f'[queued soak] PASS 2000 replays per shape,12 shapes,24000 total;sync every20 mixed cycles;10 fresh-input sets;max_rel_L2={worst};bitwise_differing_checkpoints={bitdiff}',flush=True)

def prepare(weights):
    state=_codex_soak_base_prepare(weights)
    _codex_queued_soak(state,weights['w13'].device)
    return state
'''
(root/'cuda_persistent_soak.py').write_text(text)
print('Built 24000-replay soak; CUDA source and timed path unchanged.')
