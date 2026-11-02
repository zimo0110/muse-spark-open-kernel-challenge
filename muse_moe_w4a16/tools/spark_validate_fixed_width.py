"""Development math/protocol test; emulate Hopper WGMMA with warp MMA on GB10.

This validates row/tile/fixup mapping against an independent FP32 reference.
It DOES NOT test Hopper asynchronous WGMMA retirement or H200 performance.
The GB10-only test uses one weight/x slot, and synchronously retires each
x slot at stage end. None of these emulation changes enter a submission.
"""
import importlib.util,json,re,secrets,time,sys
from pathlib import Path
import torch
import reference
from checks import compare
ROOT=Path(__file__).resolve().parent

EMULATOR=r'''
template <int NTT> struct WGSMMA {
  template<bool INIT>
  static __device__ __forceinline__ void run(float (&d)[WGS_NT][4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t desc) {
    const int lane=threadIdx.x&31,g=lane>>2,t=lane&3;
    const uint32_t base=(uint32_t)(desc&0x3fffull)<<4;
#pragma unroll
    for(int nt=0;nt<NTT;++nt) {
      const uint32_t addr=base+nt*4096+g*16+t*4;
      uint32_t b0,b1;
      asm volatile("ld.shared.b32 %0, [%2]; ld.shared.b32 %1, [%3];\n" : "=r"(b0),"=r"(b1) : "r"(addr),"r"(addr+128) : "memory");
      if constexpr(INIT) {
#pragma unroll
        for(int k=0;k<4;++k)d[nt][k]=0.f;
      }
      mma_bf16(d[nt],a0,a1,a2,a3,b0,b1);
    }
  }
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t desc) {run<true>(d,a0,a1,a2,a3,desc);}
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint64_t desc) {run<false>(d,a0,a1,a2,a3,desc);}
};
'''
RESOURCE=r'''
std::vector<int64_t> okc_resources() {
 const size_t sm=okc_smem_bytes();
 cudaError_t e=cudaFuncSetAttribute(okc::moe_fused<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm);
 TORCH_CHECK(e==cudaSuccess,"set smem: ",cudaGetErrorString(e));
 int active=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,okc::moe_fused<false>,okc::NTL,sm);
 TORCH_CHECK(e==cudaSuccess,"occupancy: ",cudaGetErrorString(e));
 cudaFuncAttributes a;e=cudaFuncGetAttributes(&a,okc::moe_fused<false>);
 TORCH_CHECK(e==cudaSuccess,"attributes: ",cudaGetErrorString(e));
 return{(int64_t)active,(int64_t)a.numRegs,(int64_t)sm};
}
'''

def emulate(src):
    src=re.sub(r'constexpr int WGS_NS = [123];','constexpr int WGS_NS = 1;',src)
    src=re.sub(r'#define OKC_WGSNX [123]','#define OKC_WGSNX 1',src)
    a=src.index('template <int NTT> struct WGSMMA;');b=src.index('// two producer warps:',a)
    src=src[:a]+EMULATOR+'\n'+src[b:]
    for old in ('asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");','asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");','asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory");'):
        assert src.count(old)==1;src=src.replace(old,'__syncwarp();')
    # One x slot is safe only because this emulator's warp MMA is synchronous.
    a=src.index('__device__ __forceinline__ void wgs_seg(');b=src.index('template <bool PB>\n__device__ __forceinline__ void wgs_phase(',a)
    body=src[a:b]
    body,n=re.subn(r'prev\s*=\s*xs;\s*\+\+seq;', 'if (lane == 0) mbar_arrive1(S.bars + BX_E + xs); prev = -1; ++seq;',body)
    assert n==1
    body=body.replace('if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);','if (lane == 0 && prev >= 0) mbar_arrive1(S.bars + BX_E + prev);')
    src=src[:a]+body+src[b:]
    marker='void okc_run(std::vector<torch::Tensor> w,';assert src.count(marker)==1
    src=src.replace(marker,RESOURCE+'\n'+marker)
    return src

def load_core(label,src,defines):
    spec=importlib.util.spec_from_file_location('core_'+label,ROOT/'spark_tiny_port.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    core.CUDA_SRC=emulate(src)
    core.MAIN_DEFINES=tuple(defines)
    core.TINY_DEFINES=core.TINY2_DEFINES=core.MID_DEFINES=()
    core.kernel_fn=core._spark_native_kernel
    return core

def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    old=json.loads((ROOT/'hopper_cases.json').read_text());new=json.loads((ROOT/'two_wg_cases.json').read_text())
    variants=[next(c for c in old if c['name']=='posted_main'),next(c for c in new if c['name']=='two_wg_r1_control'),next(c for c in new if c['name']=='two_wg_r2')]
    if len(sys.argv)>1:variants=json.loads((ROOT/sys.argv[1]).read_text())
    result_file=ROOT/'fixed_width_validation_result.json'
    weights=reference.generate_weights(secrets.randbits(62))
    cases=[]
    shapes=(1,4,8,16,17,24,25,31,32) if '--small32' in sys.argv else (1,4,8,16,17,24,25,28,31,32,33,128,256,257,385)
    for m in shapes:
        for regime in ('normal','skewed','all-local','duplicates','empty'):
            data=reference.generate_input(m,'ep8',secrets.randbits(62),dist='skewed' if regime=='skewed' else 'normal')
            if regime=='all-local':data['topk_ids'].copy_(torch.arange(8,device='cuda').expand(m,8))
            if regime=='duplicates':data['topk_ids'].fill_(0)
            if regime=='empty':data['topk_ids'].fill_(255);want=torch.zeros_like(data['hidden_states'])
            else:want=reference.ref_kernel(data,weights)
            cases.append((m,regime,data,want))
    result={'hardware':'GB10','method':'synchronous warp-MMA emulation; single-slot rings','H200_WGMMA_validation':False,'H200_performance':None,'variants':[]}
    for variant in variants:
        label=variant['name'];print('EMULATOR_BEGIN',label,flush=True)
        selected=[c for c in cases if label!='fixed_n32' or c[0]<=32]
        core=load_core(label,variant['source'],variant['defines']);start=time.monotonic();state=core.prepare(weights)
        print('EMULATOR_PREPARE',label,round(time.monotonic()-start,2),[ext.okc_resources() for ext in core._EXTS.values()],flush=True)
        graphs=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for m,regime,data,want in selected:
                core.kernel_fn(data,state);core.kernel_fn(data,state)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):out=core.kernel_fn(data,state)
                graphs.append((m,regime,graph,out,want))
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        for turn in range(8):
            for m,regime,graph,out,want in (graphs if turn%2 else reversed(graphs)):graph.replay()
        torch.cuda.synchronize()
        records=[]
        for m,regime,graph,out,want in graphs:
            if regime=='empty':ok=torch.count_nonzero(out).item()==0;message='empty exact zero'
            else:ok,message=compare(out,want,reference.REL_L2_TOL)
            print('EMULATOR_REFERENCE',label,m,regime,ok,message,flush=True)
            records.append({'M':m,'regime':regime,'passed':ok,'message':message})
            if not ok:raise AssertionError((label,m,regime,message))
        result['variants'].append({'name':label,'passed':True,'reference_cases':records,'graph_replays':len(selected)*8})
        result_file.write_text(json.dumps(result,indent=2)+'\n')
        del graphs,state,core
        torch.cuda.synchronize()
    print('TWO_WG_EMULATION_COMPLETE',json.dumps({'variants':len(result['variants']),'cases_per_variant':len(cases),'H200_validation':False}),flush=True)

if __name__=='__main__':main()
