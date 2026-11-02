"""GB10 protocol/math test of the epilogue-mailbox prototype.

Development-only FT_T=16 and synchronous warp-MMA emulation fit GB10 shared
memory. These changes do not validate H200 asynchronous WGMMA or performance.
"""
import json,importlib.util,re,secrets,time
from pathlib import Path
import torch,reference
from checks import compare
import spark_wgmma_emulate as em
R=Path(__file__).resolve().parent

def load(case):
 src=case['source'].replace('#define OKC_FT_T 64','#define OKC_FT_T 16')
 # Development job capacity is 16, so only N=8/16 instantiations are reachable.
 a=src.index('    switch (ntt) {',src.index('__device__ __forceinline__ void wgs_phase('));b=src.index('    gs += s1 - s0;',a)
 src=src[:a]+'''    if(ntt==1) wgs_seg<PB,1>(p,S,ring,u,s0,s1,slot,seq,epseq);
    else wgs_seg<PB,2>(p,S,ring,u,s0,s1,slot,seq,epseq);
'''+src[b:]
 spec=importlib.util.spec_from_file_location('ep_core',R/'spark_tiny_port.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 core.CUDA_SRC=em.emulate(src)
 core.MAIN_DEFINES=tuple(d for d in case['defines'] if not d.startswith(('-DOKC_NS=','-DOKC_NSF=')))+('-DOKC_NS=1','-DOKC_NSF=1')
 core.TINY_DEFINES=core.TINY2_DEFINES=core.MID_DEFINES=()
 core.kernel_fn=core._spark_native_kernel
 core.WGS_FIXW=1024
 jmax=32+core.EMAX//16+8
 core.WGS_FIXF=2*core.WGS_SLOTS*4*core.WGS_FIXW+2*core.WGS_SLOTS*4+jmax*16
 core.WGS_XROWS=core.EMAX+8*jmax
 return core

def main():
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
 case=json.loads((R/'epilogue_warps_cases.json').read_text())[0]
 core=load(case)
 weights=reference.generate_weights(secrets.randbits(62));state=core.prepare(weights)
 print('EP_RESOURCES', [ext.okc_resources() for ext in core._EXTS.values()],flush=True)
 records=[];graphs=[]
 for m in (1,4,8,16,17,32,33,128,256,257,385):
  for regime in ('normal','skewed','all-local','duplicates','empty'):
   data=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
   if regime=='all-local':data['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
   if regime=='duplicates':data['topk_ids'].fill_(0)
   if regime=='empty':data['topk_ids'].fill_(255);want=torch.zeros_like(data['hidden_states'])
   else:want=reference.ref_kernel(data,weights)
   out=core.kernel_fn(data,state);torch.cuda.synchronize()
   ok,msg=compare(out,want,reference.REL_L2_TOL)
   print('EP_CASE',m,regime,ok,msg,flush=True)
   records.append(dict(M=m,regime=regime,passed=ok,message=msg))
   if not ok:raise AssertionError(records[-1])
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=core.kernel_fn(data,state)
   graphs.append((graph,out,want,data))
 for turn in range(8):
  for graph,out,want,data in (graphs if turn%2 else reversed(graphs)):graph.replay()
 torch.cuda.synchronize()
 for graph,out,want,data in graphs:
  ok,msg=compare(out,want,reference.REL_L2_TOL)
  if not ok:raise AssertionError(('mixed-replay',msg))
 result=dict(hardware='GB10',method='FT_T16 synchronous warp-MMA emulation; NS/NX1',H200_WGMMA_validation=False,H200_performance=None,resources=[ext.okc_resources() for ext in core._EXTS.values()],cases=records,mixed_graph_replays=len(graphs)*8,passed=True)
 (R/'epilogue_warps_emulation_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('EP_COMPLETE',len(records),len(graphs)*8,flush=True)
if __name__=='__main__':main()
