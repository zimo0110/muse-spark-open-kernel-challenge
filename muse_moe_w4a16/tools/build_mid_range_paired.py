"""Build a no-extra-compile MID/Main cold-graph comparison; never dispatches."""
import ast
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_aligned_safe.py').read_text()
old='elif MID_DEFINES and M <= MID_MAX_M:'
assert text.count(old)==1
text=text.replace(old,'elif MID_DEFINES and M <= state.get("mid_max_m", MID_MAX_M):')
paired=(root/'cuda_persistent_two_wg_paired.py').read_text();tree=ast.parse(paired)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_paired_main_choice')
src=ast.get_source_segment(paired,fn)
src=src.replace('_paired_main_choice','_paired_mid_choice')
src=src.replace("{'posted': PAIRED_CONTROL_DEFINES, 'two_wg_striped': MAIN_DEFINES}","{'posted': MAIN_DEFINES, 'mid': MID_DEFINES}")
src=src.replace('two_wg_striped','mid').replace('(64, 128, 192, 256)','(17, 24, 32, 40, 48)')
a=src.index('    choose_trial = ')
src=src[:a]+'''    # Extend one contiguous batch-size range. Every sampled size in the
    # range must improve by at least3%, with no seed losing more than2%.
    limit = MID_MAX_M
    for m, values in shape_ratios.items():
        if statistics.median(values) <= 1.03 or min(values) <= 0.98:
            break
        limit = m
    state['mid_max_m'] = limit
    print('[paired-mid] selected MID through', limit, 'overall_geomean', gain,
          'seconds', time.perf_counter()-started, flush=True)
'''
src=src.replace('[paired-main]','[paired-mid]')
text+='\n\n'+src+'''
_mid_range_prepare = _codex_original_prepare

def prepare(weights):
    state = _mid_range_prepare(weights)
    _paired_mid_choice(state, weights['w13'].device)
    _codex_replay_stress(state, weights['w13'].device)
    return state
'''
ast.parse(text)
# Each MID job contains at most8rows. With32experts and8routes/token,
# sum ceil(c_e/8) <= M+28; M48 needs at most76 of the88 job slots.
assert 48+28 <= 32+384*8//64+8
(root/'cuda_persistent_mid_range_paired.py').write_text(text)
print('Prepared MID-range paired experiment; existing four device builds unchanged; NOT dispatched.')
