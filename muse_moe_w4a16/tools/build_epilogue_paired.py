"""Build a private paired experiment; retain the selected control by default."""
from pathlib import Path
import ast,json
R=Path(__file__).resolve().parents[1]
base=(R/'cuda_persistent_aligned_safe.py').read_text()
old='ext = _ext(tuple(_main_defines(sfx)) + sfx)'
assert base.count(old)==1
base=base.replace(old,'ext = _ext(tuple(state.get("epilogue_defines", _main_defines(sfx))) + sfx)')
case=json.loads((R/'research/spark/epilogue_warps_cases.json').read_text())[0]
base+='\nALT_SRCS["epilogue_warps"] = '+repr(case['source'])+'\n'
# Reuse the established paired graph/cold-cache/soak method without changing
# timing primitives. Calibrate across every scored MAIN shape, not just N32.
tree=ast.parse((R/'cuda_persistent_fixed32_paired.py').read_text())
cal=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fixed32_calibrate')
text=ast.get_source_segment((R/'cuda_persistent_fixed32_paired.py').read_text(),cal)
text=text.replace('_fixed32_calibrate','_epilogue_calibrate').replace('fixed_n32','epilogue_warps').replace('fixed32_defines','epilogue_defines').replace('(25,28,32)','(32,128,256)').replace('fixed-n32','epilogue-warps').replace('fixed N32','epilogue warps')
# Every scored MAIN shape must show at least a modest gain before adoption.
text=text.replace('min(statistics.median(q) for q in byshape.values())>0.97','min(statistics.median(q) for q in byshape.values())>1.015')
base+='\n'+text+'\n\ndef prepare(weights):\n    state=_codex_original_prepare(weights)\n    _epilogue_calibrate(state,weights["w13"].device)\n    _codex_replay_stress(state,weights["w13"].device)\n    return state\n'
ast.parse(base)
(R/'cuda_persistent_epilogue_paired.py').write_text(base)
print('Built guarded epilogue experiment: 3 seeds x 3 MAIN shapes; all routing regimes; 6000 queued cold replays. No dispatch.')
