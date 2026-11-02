"""Build guarded Hopper register-donation/commit-group experiment."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
old='ext = _ext(tuple(_main_defines(sfx)) + sfx)'
assert s.count(old)==1
s=s.replace(old,'ext = _ext(tuple(state.get("register_donation_defines", _main_defines(sfx))) + sfx)')
case=next(q for q in json.loads((R/'research/spark/register_donation_cases.json').read_text()) if q['name']=='register_donation_104_64_gsnarrow4')
s+='\nALT_SRCS["register_donation"] = '+repr(case['source'])+'\n'
p=(R/'cuda_persistent_epilogue_paired.py').read_text()
n=next(n for n in ast.parse(p).body if isinstance(n,ast.FunctionDef) and n.name=='_epilogue_calibrate')
cal=ast.get_source_segment(p,n).replace('_epilogue_calibrate','_register_donation_calibrate').replace('epilogue_warps','register_donation').replace('epilogue_defines','register_donation_defines').replace('epilogue-warps','register-donation').replace('epilogue warps','register donation')
s+='\n'+cal+'\n\ndef prepare(weights):\n    state=_codex_original_prepare(weights)\n    _register_donation_calibrate(state,weights["w13"].device)\n    _codex_replay_stress(state,weights["w13"].device)\n    return state\n'
ast.parse(s)
(R/'cuda_persistent_register_donation_paired.py').write_text(s)
print('Prepared guarded register-donation candidate; trial correctness, cold-cache paired timing, and 6000-replay soak; no dispatch.')
