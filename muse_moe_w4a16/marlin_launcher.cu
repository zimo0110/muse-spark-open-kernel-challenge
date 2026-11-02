// Launcher tuning around the Apache-2.0 Marlin kernel supplied by SGLang.
#include "gemm/marlin_moe/moe_wna16_marlin.cuh"

template<int BS,int TN,int TK,int TH>
void launch(tvm::ffi::TensorView a,tvm::ffi::TensorView c,tvm::ffi::TensorView w,tvm::ffi::TensorView sf,tvm::ffi::TensorView gs,
 tvm::ffi::TensorView ws,tvm::ffi::TensorView sorted,tvm::ffi::TensorView experts,tvm::ffi::TensorView padded,tvm::ffi::TensorView gates,
 tvm::ffi::TensorView tmp,int M,int N,int K,int topk,int occupancy) {
 using namespace sglang;
 using namespace device::marlin_moe;
 constexpr int MB=(BS+15)/16;
 auto kernel=Marlin<__nv_bfloat16,host::kFE2M1f.id(),host::kFE4M3fn.id(),TH,MB,TN/16,TK/16,BS==8,
   device::marlin::pipe_stages,1,false,false,false>;
 int dev=0,sms=0,maxsm=0;
 cudaGetDevice(&dev);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev);
 cudaDeviceGetAttribute(&maxsm,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);
 int cap=maxsm/occupancy-1024;
 thread_config_t cfg{TK,TN,TH};
 int need=get_kernel_cache_size(cfg,BS==8,MB,M,N,K,4,16,false,true,false,false);
 sglang::host::RuntimeCheck(need+512<=cap,"tile doesn't fit shared memory");
 cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,maxsm);
 auto stream=sglang::host::LaunchKernel::resolve_device(a.device());
 kernel<<<sms*occupancy,TH,cap,stream>>>(
   (const int4*)a.data_ptr(),(const int4*)w.data_ptr(),(int4*)c.data_ptr(),(int4*)tmp.data_ptr(),nullptr,
   (const int4*)sf.data_ptr(),(const uint16_t*)gs.data_ptr(),nullptr,nullptr,
   (int*)sorted.data_ptr(),(int*)experts.data_ptr(),(int*)padded.data_ptr(),(float*)gates.data_ptr(),
   topk,false,false,K/16,M,N,K,(int*)ws.data_ptr(),false,false,true,cap);
}

void run(tvm::ffi::TensorView a,tvm::ffi::TensorView c,tvm::ffi::TensorView w,tvm::ffi::TensorView sf,tvm::ffi::TensorView gs,
 tvm::ffi::TensorView ws,tvm::ffi::TensorView sorted,tvm::ffi::TensorView experts,tvm::ffi::TensorView padded,tvm::ffi::TensorView gates,
 tvm::ffi::TensorView tmp,int64_t M,int64_t N,int64_t K,int64_t topk,int64_t bs,int64_t kind,int64_t occupancy) {
 #define TILE(B,N_,K_,T_,Q) if(bs==B && kind==Q){launch<B,N_,K_,T_>(a,c,w,sf,gs,ws,sorted,experts,padded,gates,tmp,M,N,K,topk,occupancy);return;}
 #define SHAPES(B) TILE(B,64,128,128,0) TILE(B,128,64,128,1) TILE(B,128,128,256,2) TILE(B,256,64,256,3) TILE(B,64,256,256,4)
 SHAPES(8) SHAPES(16) SHAPES(32) SHAPES(64)
 sglang::host::RuntimeCheck(false,"unknown launch configuration");
}
