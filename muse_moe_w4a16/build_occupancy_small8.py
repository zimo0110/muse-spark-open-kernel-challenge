"""Apply occupancy-gated NS2 experiment to the second tiny variant (5..8 tokens)."""
from pathlib import Path
root=Path(__file__).resolve().parent
text=(root/'cuda_persistent_occupancy_v3.py').read_text()
assert text.count("cfg=state.get('tiny_occ',{}).get(1 if m==1 else 4)")==1
text=text.replace("cfg=state.get('tiny_occ',{}).get(1 if m==1 else 4)","cfg=state.get('tiny_occ',{}).get(5 if m==5 else 8)")
assert text.count('if m<=TINY_MAX_M and cfg:')==1
text=text.replace('if m<=TINY_MAX_M and cfg:', 'if TINY_MAX_M<m<=TINY2_MAX_M and cfg:')
assert text.count('control=_ext(tuple(TINY_DEFINES)+sfx)')==1
text=text.replace('control=_ext(tuple(TINY_DEFINES)+sfx)','control=_ext(tuple(TINY2_DEFINES)+sfx)')
assert text.count("small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2',)+sfx)")==1
text=text.replace("small=_ext(tuple(TINY_DEFINES)+('-DOKC_NS=2',)+sfx)","small=_ext(tuple(TINY2_DEFINES)+('-DOKC_NS=2',)+sfx)")
assert text.count('for m in (1,4):')==1
text=text.replace('for m in (1,4):','for m in (5,8):')
text=text.replace('(1, 2, 3, 4, 8, 16, 17, 32, 33, 65, 128, 256, 257, 385)', '(1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 32, 33, 65, 128, 256, 257, 385)')
text=text.replace('PASS 224 mixed-shape','PASS 272 mixed-shape')
(root/'cuda_persistent_small8.py').write_text(text)
print('Prepared 5..8-token NS2 residency experiment with 5/6/7 nearby replay validation.')
