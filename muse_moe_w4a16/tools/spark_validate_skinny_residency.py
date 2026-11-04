"""Native GB10 math/replay check; NS1 development ring, no H200 speed claim."""
import importlib.util,json,secrets,sys
from pathlib import Path
import torch,reference
from checks import compare
R=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 f=Path(sys.argv[1]) if len(sys.argv)>1 else R.parent/'research/spark/skinny_residency_cases.json'
 if not f.exists():f=R/'skinny_residency_cases.json'
 case=next(q for q in json.loads(f.read_text()) if q['name']=='skinny_nw4_b2')
 spec=importlib.util.spec_from_file_location('skinny_core',R/'spark_tiny_port.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 core.CUDA_SRC=case['source'];core.MAIN_DEFINES=tuple(d for d in case['defines'] if not d.startswith(('-DOKC_NS=','-DOKC_FAST1=')))+('-DOKC_NS=1','-DOKC_FAST1=1')
 core.TINY_DEFINES=core.TINY2_DEFINES=core.MID_DEFINES=();core.kernel_fn=core._spark_native_kernel
 w=reference.generate_weights(secrets.randbits(62));state=core.prepare(w);print('RESOURCES',[m.okc_resources() for m in core._EXTS.values()],flush=True)
 cases=[];records=[];graphs=[]
 for m in (1,2,4,5,8,12,16):
  for regime in ('normal','skewed','all-local','duplicates','empty'):
   data=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
   if regime=='all-local':data['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
   if regime=='duplicates':data['topk_ids'].fill_(0)
   if regime=='empty':data['topk_ids'].fill_(255);want=torch.zeros_like(data['hidden_states'])
   else:want=reference.ref_kernel(data,w)
   out=core.kernel_fn(data,state);torch.cuda.synchronize();ok,msg=compare(out,want,reference.REL_L2_TOL)
   print('SKINNY_CASE',m,regime,ok,msg,flush=True)
   if not ok:raise AssertionError((m,regime,msg))
   records.append(dict(M=m,regime=regime,passed=ok,message=msg));g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=core.kernel_fn(data,state)
   graphs.append((g,out,want,data))
 for turn in range(12):
  for g,out,want,data in (graphs if turn%2 else reversed(graphs)):g.replay()
 torch.cuda.synchronize()
 for g,out,want,data in graphs:
  ok,msg=compare(out,want,reference.REL_L2_TOL)
  if not ok:raise AssertionError(('mixed replay',msg))
 result=dict(hardware='GB10',method='native warp MMA and TMA; NS1 development ring; one grid block per SM',H200_validation=False,cases=records,mixed_replays=12*len(graphs),passed=True)
 (R/'skinny_residency_emulation_result.json').write_text(json.dumps(result,indent=2)+'\n');print('SKINNY_COMPLETE',len(records),result['mixed_replays'],flush=True)
if __name__=='__main__':main()
