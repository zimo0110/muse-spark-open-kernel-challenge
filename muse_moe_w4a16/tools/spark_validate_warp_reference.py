"""Cross-check the on-the-fly FP32 CUDA diagnostic against supplied reference."""
from pathlib import Path
import hashlib,json,secrets,torch
from torch.utils.cpp_extension import load_inline
import reference
from checks import compare
root=Path(__file__).resolve().parent
src=(root/'warp_moe_reference.cu').read_text()
cpp='#include <torch/extension.h>\nvoid diagnostic_reference(std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'+('a' if major==9 else '')
mod=load_inline(name='independent_moe_'+hashlib.sha256(src.encode()).hexdigest()[:18],cpp_sources=cpp,cuda_sources=src,functions=['diagnostic_reference'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False)
w=reference.generate_weights(secrets.randbits(62));ww=[w[k].contiguous() for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs','post_norm_w')];records=[]
for m in (1,5,8,16):
 for regime in ('normal','skewed','all-local'):
  d=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
  if regime=='all-local':d['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
  inter=torch.empty((32,m,4096),dtype=torch.float32,device='cuda');y=torch.empty_like(inter);out=torch.empty_like(d['hidden_states'])
  mod.diagnostic_reference(ww,d['hidden_states'],d['topk_ids'].to(torch.int32).contiguous(),d['topk_weights'].float().contiguous(),inter,y,out,0,32)
  want=reference.ref_kernel(d,w);torch.cuda.synchronize();ok,msg=compare(out,want,0.001)
  error=(torch.linalg.vector_norm(out.float()-want.float())/torch.linalg.vector_norm(want.float()).clamp_min(1e-20)).item()
  records.append({'m':m,'regime':regime,'passed':ok,'relative_error':error,'message':msg});print('INDEPENDENT_REFERENCE',records[-1],flush=True)
  assert ok
(root/'independent_warp_reference_result.json').write_text(json.dumps({'passed':True,'records':records},indent=2))
print('INDEPENDENT_REFERENCE_PASS',max(v['relative_error'] for v in records),flush=True)
