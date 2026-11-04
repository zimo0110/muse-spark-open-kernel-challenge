"""Compare deeper skinny rings with the original source, geometry and hints."""
import ast
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text()
a=s.index('def kernel_fn(data, state):');b=s.index('\n\ndef ',a+1);p=s[a:b];mark='    for m0 in range(0, M, CHUNK_TOKENS):'
p=p.replace(mark,'''    if M<=16 and state.get("small_ring_choices"):
        band=4 if M<=4 else 8 if M<=8 else 16
        if band in state["small_ring_choices"]:
            ext=_ext(tuple(state["small_ring_choices"][band])+sfx)
'''+mark);s=s[:a]+p+s[b:]
t=(R/'cuda_skinny_residency_paired.py').read_text();n=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='_skinny_residency_calibrate');cal=ast.get_source_segment(t,n)
a=cal.index('    for label,nw,blocks in ');b=cal.index('    main=',a)
cal=cal[:a]+cal[b:]
old="            choices={'control':(_ext(tuple(cd)+state['sfx']),base,tuple(cd)),**variants}"
new='''            variants={}
            for ns in (4,5):
                defs=tuple(d for d in cd if not d.startswith('-DOKC_NS='))+(f'-DOKC_NS={ns}',)
                variants[f'ns{ns}']=(_ext(defs+state['sfx']),base,defs)
            choices={'control':(_ext(tuple(cd)+state['sfx']),base,tuple(cd)),**variants}'''
assert old in cal;cal=cal.replace(old,new)
cal=cal.replace("            for regime in ('normal','all_local','empty','skewed','duplicates'):","            for regime in (('normal','all_local','empty','skewed','duplicates') if m>8 else ('normal','all_local','empty','skewed')):")
cal=cal.replace('_skinny_residency_calibrate','_small_ring_calibrate').replace('skinny_choices','small_ring_choices').replace('[skinny','[small-ring')
old="            gain,name=max(eligible);_,grid,defs=variants[name];state['small_ring_choices'][band]=(defs,grid)"
new="            gain,name=max(eligible);ns=int(name[2:]);cd=TINY_DEFINES if band==4 else TINY2_DEFINES if band==8 else MID_DEFINES\n            defs=tuple(d for d in cd if not d.startswith('-DOKC_NS='))+(f'-DOKC_NS={ns}',)\n            state['small_ring_choices'][band]=defs"
assert old in cal;cal=cal.replace(old,new)
s+='\n'+cal+'\n\ndef prepare(weights):\n    state=_codex_original_prepare(weights)\n    _small_ring_calibrate(state,weights["w13"].device)\n    _codex_replay_stress(state,weights["w13"].device)\n    return state\n'
ast.parse(s);(R/'cuda_small_ring_tuned.py').write_text(s)
print('Prepared NS4/NS5 skinny comparison; original eight-warps and arithmetic, no dispatch.')
