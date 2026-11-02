"""DEVELOPMENT ONLY: portable native warp-MMA/TMA MoE, bounded M<=16.

Not a challenge submission: imports local packing helpers and reads CUDA source.
An integrated candidate must embed these and retain the checked larger-M path.
"""
from pathlib import Path
import importlib.util,hashlib
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('warp_pack_core',ROOT/'spark_tiny_port.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
SOURCE=(ROOT/'warp_moe_kpack_device.cu').read_text()
CPP='''#include <torch/extension.h>
void warp_run(std::vector<torch::Tensor>,std::vector<torch::Tensor>,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);
std::vector<int64_t> warp_resources();
'''
_MODULES={}
def module(npad,slots,kpack=1):
    key=(npad,slots,kpack)
    if key not in _MODULES:
        major,minor=torch.cuda.get_device_capability();arch=f'{major}{minor}'+('a' if major==9 else '')
        _MODULES[key]=load_inline(name='native_warp_'+hashlib.sha256((SOURCE+repr(key)).encode()).hexdigest()[:20],cpp_sources=CPP,cuda_sources=SOURCE,functions=['warp_run','warp_resources'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo',f'-DWARP_NPAD={npad}',f'-DWARP_NS={slots}',f'-DWARP_KPACK={kpack}'],verbose=False)
    return _MODULES[key]

def warp_repack(flat,rows,count=32):
    units=rows//256
    z=flat.view(count,units,16,4,9216)
    codes=z[...,:8192].reshape(count,units,16,4,4,4,32,16).permute(0,1,3,5,2,4,6,7).contiguous().reshape(count,rows//16,16,2048)
    scales=z[...,8192:].reshape(count,units,16,4,4,32,8).permute(0,1,3,4,2,5,6).contiguous().reshape(count,rows//16,16,256)
    return torch.cat((codes,scales),dim=3).contiguous()

def prepare(weights,variants=((8,4,1),(8,2,2),(8,3,2),(16,4,1),(16,2,2))):
    dev=weights['w13'].device
    assert core._scales_prmt_exact(weights['w13_sf'][:32],weights['w2_sf'][:32])
    rb=torch.arange(4096//8,device=dev);g=torch.arange(8,device=dev)
    rows_a=torch.stack((8*rb[:,None]+g[None,:],4096+8*rb[:,None]+g[None,:]),1).reshape(-1)
    rows_b=torch.arange(4096,device=dev)
    wa=torch.cat([core._wgs_copy(weights['w13'][e],weights['w13_sf'][e],core._kperm_rows(rows_a)) for e in range(32)])
    wb=torch.cat([core._wgs_copy(weights['w2'][e],weights['w2_sf'][e],rows_b) for e in range(32)])
    state={'w':[warp_repack(wa,8192),warp_repack(wb,4096),weights['w13_gs'][:32].contiguous(),weights['w2_gs'][:32].contiguous(),(weights['post_norm_w'].float()+1).to(torch.bfloat16).contiguous()],
           'scratch':[torch.zeros(6432,dtype=torch.int32,device=dev),torch.empty((128,4096),dtype=torch.bfloat16,device=dev),torch.empty((2,128,4096),dtype=torch.float32,device=dev)],
           'lo':0,'cnt':32,'sm_count':torch.cuda.get_device_properties(dev).multi_processor_count,'grid':torch.cuda.get_device_properties(dev).multi_processor_count*16}
    for npad,slots,kpack in variants:
        ext=module(npad,slots,kpack)
        print('NATIVE_WARP_RESOURCES',npad,slots,kpack,ext.warp_resources(),flush=True)
    return state

def run(data,state,variant=None):
    x=data['hidden_states'].contiguous();m=x.shape[0]
    assert 1<=m<=16
    choice=variant or ((8,4) if m<=8 else (16,4))
    npad,slots=choice[:2]
    multiplier=choice[2] if len(choice)>2 else 16
    kpack=choice[3] if len(choice)>3 else 1
    assert m<=npad
    cfg=data['config'];assert cfg['local_start']==state['lo'] and cfg['local_count']==state['cnt']
    ids=data['topk_ids'].to(torch.int32).contiguous()
    if ids.data_ptr()&15:ids=ids.clone()
    tw=data['topk_weights'].to(torch.float32).contiguous()
    out=torch.empty_like(x)
    module(npad,slots,kpack).warp_run(state['w'],state['scratch'],x,ids,tw,out,state['lo'],state['cnt'],state['sm_count']*multiplier)
    return out
