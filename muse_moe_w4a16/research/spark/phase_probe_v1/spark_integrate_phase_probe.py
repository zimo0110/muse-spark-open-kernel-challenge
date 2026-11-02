"""Run actual candidate Python diagnostics against GB10 development modules.

Hopper WGMMA is synchronously emulated. MID uses a1slot/1prefetch small-smem
port. This checks integration/graphs/reporting, not H200 performance or hazards.
"""
from pathlib import Path
import ast,json,secrets,time
import torch
import reference
import spark_wgmma_emulate as emulator
R=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 text=(R/'cuda_persistent_phase_probe.py').read_text();tree=ast.parse(text)
 cases={c['name']:c for c in json.loads((R/'phase_probe_cases.json').read_text())}
 weights=reference.generate_weights(secrets.randbits(62))
 cores={};modules={};state=None
 for name,case in cases.items():
  flags=list(case['defines']);mid='mid' in name
  if mid:flags=[d for d in flags if not d.startswith('-DOKC_FAST1=')]+['-DOKC_NS=1','-DOKC_NSF=1','-DOKC_FAST1=1']
  core=emulator.load_core(name,case['source'],flags)
  if mid:
   marker='  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES'
   core.CUDA_SRC=core.CUDA_SRC.replace(marker,'#if OKC_TINY\n  return base+(size_t)OKC_NS*okc::UNIT_BYTES;\n#endif\n'+marker)
  cores[name]=core
  if state is None:state=core.prepare(weights)
  modules[name]=core._ext(tuple(core.MAIN_DEFINES)+state['sfx'])
 def ext(defines):
  tags=[d[1:] for d in defines if d.startswith('@') and d!='@helper0']
  if tags:return modules[tags[-1]]
  return modules['phase_mid_control' if '-DOKC_SPLIT=1' in defines else 'phase_main_control']
 gen=cores['phase_main_control']._gen_input
 mid=tuple(cases['phase_mid_control']['defines'])+('@helper0',)
 env={'torch':torch,'_ext':ext,'_gen_input':gen,'MAIN_DEFINES':tuple(cases['phase_main_control']['defines']),'MID_DEFINES':mid}
 names={'_phase_percentiles','_phase_decode','_phase_probe_prepare'}
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),'<actual-candidate-python>','exec'),env)
 start=time.monotonic()
 env['_phase_probe_prepare'](state,weights['w13'].device)
 torch.cuda.synchronize()
 report={'passed':True,'hardware':'GB10','purpose':'actual candidate Python diagnostics integration',
         'H200_performance':None,'Hopper_async_validation':False,'seconds':round(time.monotonic()-start,3)}
 (R/'phase_probe_integration_result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PHASE_INTEGRATION_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
