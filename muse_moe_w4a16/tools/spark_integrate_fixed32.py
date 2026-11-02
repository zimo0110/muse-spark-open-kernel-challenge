"""Exercise actual private calibration Python with GB10 emulation modules.

The timing/selection recorded here is NOT an H200 prediction.
"""
from pathlib import Path
import ast,json,secrets,time
import torch
import reference
import spark_wgmma_emulate as emulator
R=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 control=next(q for q in json.loads((R/'phase_probe_cases.json').read_text()) if q['name']=='phase_main_control')
 trial=next(q for q in json.loads((R/'fixed_width_cases.json').read_text()) if q['name']=='fixed_n32')
 weights=reference.generate_weights(secrets.randbits(62))
 c=emulator.load_core('fixed_control',control['source'],control['defines']);state=c.prepare(weights)
 t=emulator.load_core('fixed_n32',trial['source'],trial['defines'])
 baseline=c._ext(tuple(c.MAIN_DEFINES)+state['sfx']);alternative=t._ext(tuple(t.MAIN_DEFINES)+state['sfx'])
 def ext(defs):return alternative if '@fixed_n32' in defs else baseline
 env={'torch':torch,'_ext':ext,'_gen_input':c._gen_input,'MAIN_DEFINES':tuple(control['defines'])}
 tree=ast.parse((R/'cuda_persistent_fixed32_paired.py').read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fixed32_calibrate')
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-fixed32-calibration>','exec'),env)
 start=time.monotonic();env['_fixed32_calibrate'](state,weights['w13'].device);torch.cuda.synchronize()
 report={'passed':True,'hardware':'GB10','H200_performance':None,'Hopper_async_validation':False,
         'actual_calibration_runner':True,'queued_replays':6000,'seconds':round(time.monotonic()-start,3)}
 (R/'fixed32_integration_result.json').write_text(json.dumps(report,indent=2)+'\n')
 print('FIXED32_INTEGRATION_COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':main()
