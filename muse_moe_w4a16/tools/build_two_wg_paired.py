"""Build a private-check candidate with paired cold-graph validation/timing.

Does not dispatch anything or change the published submission. Requires a real
H200 check; Spark emulation cannot establish this candidate's performance.
"""
import ast
from pathlib import Path
root = Path(__file__).resolve().parents[1]
base = (root / 'cuda_persistent_two_wg_striped.py').read_text()
posted = ast.parse((root / 'cuda_persistent_aligned_safe.py').read_text())
def assignment(name):
    for node in posted.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name)
old = 'ext = _ext(tuple(_main_defines(sfx)) + sfx)'
assert base.count(old) == 1
base = base.replace(old, 'ext = _ext(tuple(state.get("large_defines", _main_defines(sfx))) + sfx)')
base += '\nALT_SRCS["paired_posted"] = ' + repr(assignment('CUDA_SRC')) + '\n'
base += 'PAIRED_CONTROL_DEFINES = ' + repr(assignment('MAIN_DEFINES') + ('@paired_posted',)) + '\n'
base += r'''
# Preparation-only comparison. Retain only a compiled-kernel choice in state;
# diagnostic inputs/outputs never survive in state or enter official timed calls.
def _paired_main_choice(state, dev):
    import math
    import statistics
    import time
    started = time.perf_counter()
    if not state['sfx']:
        return
    candidates = {'posted': PAIRED_CONTROL_DEFINES, 'two_wg_striped': MAIN_DEFINES}
    modules = {name: _ext(tuple(defs) + state['sfx']) for name, defs in candidates.items()}
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    samples = []
    seed_ratios = []
    shape_ratios = {m: [] for m in (64, 128, 192, 256)}
    for seed in (47821, 73931, 95813):
        ratios = []
        for m in shape_ratios:
            data = _gen_input(m, seed + m, dev)
            x, ids, tw = (data[k] for k in ('hidden_states', 'topk_ids', 'topk_weights'))
            outputs = {name: torch.empty_like(x) for name in modules}
            def run(name):
                modules[name].okc_run(state['w'], state['scratch'], x, ids, tw,
                                     outputs[name], state['lo'], state['cnt'], state['grid'])
            # Check normal and extreme routing before comparing speed. Agreement
            # with the checked control supplements the judge's independent tests.
            normal_ids = ids.clone()
            for regime in ('normal', 'all_local', 'empty', 'skewed'):
                ids.copy_(normal_ids)
                if regime == 'all_local':
                    ids.copy_(torch.arange(8, device=dev).expand(m, 8))
                elif regime == 'empty':
                    ids.fill_(255)
                elif regime == 'skewed':
                    ids[:m//2].copy_(torch.arange(8, device=dev).expand(m//2, 8))
                for name in modules:
                    run(name)
                torch.cuda.synchronize(dev)
                control = outputs['posted'].float()
                trial = outputs['two_wg_striped'].float()
                if not torch.isfinite(trial).all().item():
                    raise AssertionError(('paired nonfinite', m, regime))
                if regime == 'empty':
                    if torch.count_nonzero(trial).item():
                        raise AssertionError(('paired empty nonzero', m))
                else:
                    err = (torch.linalg.vector_norm(trial-control) /
                           torch.linalg.vector_norm(control).clamp_min(1e-20)).item()
                    if err > 0.001:
                        raise AssertionError(('paired control disagreement', m, regime, err))
            ids.copy_(normal_ids)
            graphs = {}
            for name in modules:
                for _ in range(3):
                    run(name)
                graphs[name] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[name]):
                    run(name)
            observations = []
            for turn in range(24):
                timings = {}
                order = ('posted', 'two_wg_striped') if turn % 2 == 0 else ('two_wg_striped', 'posted')
                for name in order:
                    flush.add_(1)
                    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    timings[name] = begin.elapsed_time(end) * 1000
                observations.append(timings['posted'] / timings['two_wg_striped'])
            ratio = statistics.median(observations)
            ratios.append(ratio)
            shape_ratios[m].append(ratio)
            samples.append({'seed': seed, 'm': m, 'control_over_trial': ratio})
            print('[paired-main]', samples[-1], flush=True)
            del graphs
        seed_ratios.append(math.exp(sum(math.log(r) for r in ratios) / len(ratios)))
    gain = math.exp(sum(math.log(r['control_over_trial']) for r in samples) / len(samples))
    choose_trial = (gain > 1.03 and min(seed_ratios) > 1.02 and
                    min(statistics.median(v) for v in shape_ratios.values()) > 0.97)
    selected = 'two_wg_striped' if choose_trial else 'posted'
    state['large_defines'] = candidates[selected]
    print('[paired-main] selected', selected, 'geomean', gain,
          'seed_geomeans', seed_ratios, 'seconds', time.perf_counter()-started, flush=True)

_paired_base_prepare = _codex_original_prepare

def prepare(weights):
    state = _paired_base_prepare(weights)
    _paired_main_choice(state, weights['w13'].device)
    _codex_replay_stress(state, weights['w13'].device)
    return state
'''
p = root / 'cuda_persistent_two_wg_paired.py'
ast.parse(base)
p.write_text(base)
print('Prepared paired candidate; NOT dispatched and H200-UNVERIFIED.')
