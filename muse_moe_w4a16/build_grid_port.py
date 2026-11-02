"""Generate a native-persistent grid calibration experiment, no CUDA changes."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
p=root/'cuda_persistent_bf16.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='91766194b46a086f3d772238d25f86e87c4668109384798bb8245f4f66f7d9dd'
text=p.read_text()
text += r'''

_codex_untuned_kernel = kernel_fn
_codex_untuned_prepare = prepare
_CODEX_GRID_BOUNDS = (1, 4, 8, 16, 32, 64, 128, 256, 384)

def kernel_fn(data, state):
    grids = state.get('grids', {})
    m = min(data['hidden_states'].shape[0], CHUNK_TOKENS)
    bound = next((b for b in _CODEX_GRID_BOUNDS if m <= b), _CODEX_GRID_BOUNDS[-1])
    chosen = grids.get(bound, state['grid'])
    if chosen == state['grid']:
        return _codex_untuned_kernel(data, state)
    local_state = dict(state)
    local_state['grid'] = chosen
    return _codex_untuned_kernel(data, local_state)

def _codex_capture_grid(data, state, grid):
    arm = dict(state)
    arm['grid'] = grid
    # Use the original entry point; the tuner measures the actual graph path.
    stream = torch.cuda.Stream(device=data['hidden_states'].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _codex_untuned_kernel(data, arm)
        _codex_untuned_kernel(data, arm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = _codex_untuned_kernel(data, arm)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph, output

def _codex_grid_tune(state, dev):
    import statistics
    full = state['grid']
    choices = sorted(set(max(1, min(full, round(full*r))) for r in (0.73,0.85,0.91,0.97,1.0)))
    flush = torch.zeros(64*1024*1024, dtype=torch.int32, device=dev)
    event0, event1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    def sample(graph):
        flush.add_(1)
        event0.record()
        graph.replay()
        event1.record()
        event1.synchronize()
        return event0.elapsed_time(event1)
    selected = {}
    for m in _CODEX_GRID_BOUNDS:
        ratios = {g:[] for g in choices if g != full}
        for seed in (30013, 60017):
            data = _gen_input(m, seed+m, dev)
            graphs = {g:_codex_capture_grid(data, state, g) for g in choices}
            graphs[full][0].replay()
            torch.cuda.synchronize()
            control = graphs[full][1].float().clone()
            for g,(graph,out) in graphs.items():
                graph.replay()
                torch.cuda.synchronize()
                diff = torch.linalg.vector_norm(out.float()-control)
                err = (diff/torch.linalg.vector_norm(control).clamp_min(1e-20)).item()
                if err > 0.001 or not torch.isfinite(out).all().item():
                    raise AssertionError(('grid output mismatch', m, g, err))
            for g in ratios:
                pairs=[]
                for repeat in range(6):
                    order=(full,g) if repeat%2 else (g,full)
                    times={arm:sample(graphs[arm][0]) for arm in order}
                    pairs.append(times[full]/times[g])
                ratios[g].append(statistics.median(pairs))
        eligible = {g:min(rs) for g,rs in ratios.items() if min(rs)>1.03}
        pick=max(eligible,key=eligible.get) if eligible else full
        # Independent third input confirms any selected grid before retaining it.
        if pick != full:
            fresh=_gen_input(m, 90019+m, dev)
            a,oa=_codex_capture_grid(fresh,state,full)
            b,ob=_codex_capture_grid(fresh,state,pick)
            err=(torch.linalg.vector_norm(oa.float()-ob.float())/torch.linalg.vector_norm(oa.float()).clamp_min(1e-20)).item()
            confirm=[]
            for repeat in range(8):
                if repeat%2:
                    ta,tb=sample(a),sample(b)
                else:
                    tb,ta=sample(b),sample(a)
                confirm.append(ta/tb)
            if err>0.001 or statistics.median(confirm)<1.02:
                pick=full
        selected[m]=pick
        print(f'[native grid] M={m} full={full} selected={pick} two-seed-ratios={ratios}',flush=True)
    state['grids']=selected

def prepare(weights):
    state=_codex_untuned_prepare(weights)
    _codex_grid_tune(state, weights['w13'].device)
    _codex_replay_stress(state, weights['w13'].device)
    return state
'''
# Insert the stress function only (its wrapper would overwrite this prepare).
stress=(root/'persistent_replay_stress.inc').read_text().split('_codex_original_prepare = prepare')[0]
text+=stress
(root/'cuda_persistent_grid_v2.py').write_text(text)
print('Generated graph-timed native grid calibration with independent confirmation and replay stress.')
