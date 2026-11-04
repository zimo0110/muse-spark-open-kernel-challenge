"""Per-band ring depth and tail-epilogue tuning, bounded cold compilation."""
import ast
from pathlib import Path
R=Path(__file__).resolve().parents[1]
base=(R/'cuda_small_ring_tuned.py').read_text();tree=ast.parse(base)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_small_ring_calibrate')
cal=ast.get_source_segment(base,fn)
a=cal.index('            for ns in (4,5):');b=cal.index("            choices=",a)
cal=cal[:a]+'''            for name,flag,value in specs:
                defs=tuple(d for d in cd if not d.startswith(flag))+(flag+str(value),)
                variants[name]=(_ext(defs+state['sfx']),base,defs)
'''+cal[b:]
cal=cal.replace("    samples=[]","    specs=(('ns4','-DOKC_NS=',4),('ns5','-DOKC_NS=',5),('tail3','-DOKC_WSBLK=',3),('tail4','-DOKC_WSBLK=',4))\n    samples=[]")
a=cal.index('            gain,name=max(eligible);ns=int(name[2:]);');b=cal.index("            print('[small-ring selected]'",a)
cal=cal[:a]+'''            gain,name=max(eligible);_,flag,value=next(q for q in specs if q[0]==name)
            cd=TINY_DEFINES if band==4 else TINY2_DEFINES if band==8 else MID_DEFINES
            defs=tuple(d for d in cd if not d.startswith(flag))+(flag+str(value),)
            state['small_ring_choices'][band]=defs
'''+cal[b:]
for band,shapes in ((4,(1,2,4)),(8,(5,8)),(16,(12,16))):
 c=cal.replace('for m in (1,4,8,12,16):',f'for m in {shapes!r}:')
 c=c.replace('for band,shapes in ((4,(1,4)),(8,(8,)),(16,(12,16))):',f'for band,shapes in (({band},{shapes!r}),):')
 assert f'for m in {shapes!r}:' in c
 s=base[:fn.col_offset] if False else base[:base.index('def _small_ring_calibrate(')]
 s+=c+'\n\ndef prepare(weights):\n    state=_codex_original_prepare(weights)\n    _small_ring_calibrate(state,weights["w13"].device)\n    _codex_replay_stress(state,weights["w13"].device)\n    return state\n'
 ast.parse(s);(R/f'cuda_small_band{band}_tuned.py').write_text(s)
print('Prepared independent bands4/8/16: NS4,NS5,tail3,tail4; original math and source, control fallback.')
