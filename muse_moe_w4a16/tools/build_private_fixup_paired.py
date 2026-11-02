"""Construct guarded cold-graph comparison with independent control scratch."""
import ast,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
filename=sys.argv[1] if len(sys.argv)>1 else 'cuda_persistent_private_fixup.py'
assert filename in ('cuda_persistent_private_fixup.py','cuda_persistent_compact_private_fixup.py')
label=filename.removeprefix('cuda_persistent_').removesuffix('.py')
factor=2 if label.startswith('compact_') else 16
s=(r/'cuda_persistent_two_wg_paired.py').read_text()
t=ast.parse(s);new=ast.parse((r/filename).read_text())
def assign(tree,name):return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
changes=[];lines=s.splitlines(keepends=True)
for name in ('CUDA_SRC','MAIN_DEFINES'):
 a=assign(t,name);b=assign(new,name)
 lo=sum(map(len,lines[:a.value.lineno-1]))+a.value.col_offset;hi=sum(map(len,lines[:a.value.end_lineno-1]))+a.value.end_col_offset
 changes.append((lo,hi,repr(ast.literal_eval(b.value))))
for lo,hi,value in sorted(changes,reverse=True):s=s[:lo]+value+s[hi:]
s=s.replace('WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW +',f'WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW * {factor} +')
s=s.replace('two_wg_striped','private_fixup').replace('(64, 128, 192, 256)','(32, 64, 128, 192, 256)')
marker="    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)"
assert marker in s
prefix,s=s.rsplit("def _paired_main_choice(state, dev):",1)
s="def _paired_main_choice(state, dev):"+s
s=s.replace(marker,"""    # These two kernels use different fixup/counter layouts. They must never
    # share those buffers: a trial payload can alias a control release flag.
    old_fixf = 2*WGS_SLOTS*4*WGS_FIXW + 2*WGS_SLOTS*4 + 136*16
    control_scratch = [torch.zeros_like(state['scratch'][0]),
                       torch.zeros(EMAX+EMAX*HIDDEN+old_fixf,dtype=torch.float32,device=dev),
                       torch.zeros_like(state['scratch'][2])]
    workspaces = {'posted':control_scratch,'private_fixup':state['scratch']}
"""+marker)
s=s.replace("modules[name].okc_run(state['w'], state['scratch'], x, ids, tw,","modules[name].okc_run(state['w'], workspaces[name], x, ids, tw,")
s=s.replace("    state['large_defines'] = candidates[selected]","    state['large_defines'] = candidates[selected]\n    state['scratch'] = workspaces[selected]")
s=s.replace('                for name in modules:\n                    run(name)','''                for name in modules:
                    run(name)
                    stable = outputs[name].clone()
                    run(name)
                    drift=(torch.linalg.vector_norm(outputs[name].float()-stable.float()) /
                           torch.linalg.vector_norm(stable.float()).clamp_min(1e-20)).item()
                    if drift>0.001 or not torch.isfinite(outputs[name]).all().item():
                        raise AssertionError(('split bank replay inconsistency',name,m,regime,drift))''')
s=s.replace('            observations = []','            observations = []\n            absolute = []')
s=s.replace("                observations.append(timings['posted'] / timings['private_fixup'])", "                observations.append(timings['posted'] / timings['private_fixup'])\n                absolute.append(timings)")
s=s.replace("samples.append({'seed': seed, 'm': m, 'control_over_trial': ratio})", "samples.append({'seed': seed, 'm': m, 'control_over_trial': ratio, 'control_us':statistics.median(v['posted'] for v in absolute), 'trial_us':statistics.median(v['private_fixup'] for v in absolute)})")
if label.startswith('compact_'):
    s=s.replace('            del graphs', "            if seed == 47821:\n                max_drift=0.0\n                different=0\n                for epoch in range(2):\n                    fresh=_gen_input(m,seed+771293+epoch*173+m,dev)\n                    x.copy_(fresh['hidden_states']);ids.copy_(fresh['topk_ids']);tw.copy_(fresh['topk_weights'])\n                    run('private_fixup')\n                    torch.cuda.synchronize(dev)\n                    expected=outputs['private_fixup'].clone()\n                    for batch in range(50):\n                        for replay in range(20):\n                            flush.add_(1)\n                            graphs['private_fixup'].replay()\n                        torch.cuda.synchronize(dev)\n                        observed=outputs['private_fixup']\n                        drift=(torch.linalg.vector_norm(observed.float()-expected.float()) /\n                               torch.linalg.vector_norm(expected.float()).clamp_min(1e-20)).item()\n                        if drift>0.001 or not torch.isfinite(observed).all().item():\n                            raise AssertionError(('queued split bank drift',m,epoch,batch,drift))\n                        max_drift=max(max_drift,drift)\n                        different+=int(not torch.equal(observed,expected))\n                print('[private-partial-soak]',m,'replays',2000,'max_relative_L2',max_drift,'different_checkpoints',different,flush=True)\n            del graphs")
s=(prefix+s).replace("private_fixup",label)
ast.parse(s);(r/(filename.removesuffix('.py')+'_paired.py')).write_text(s)
print('Prepared disjoint-FP32-partials comparison, independent workspaces,3 fresh seeds,5 shapes; NOT dispatched.')
