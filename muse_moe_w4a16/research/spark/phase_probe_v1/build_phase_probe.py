"""Build diagnostic sources only; never dispatch or edit published submission.

Coarse timestamps: consumer start/route/gather/join/G1/G2/barrier/final.
Detailed timestamps: producer issue windows, sampled empty-ring waits,
activation readiness waits, logical bytes and stage counts. These are not
hardware DRAM counters. Paired runtime overhead must be checked on H200.
"""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'cuda_persistent_aligned_safe.py').read_text();tree=ast.parse(s)
def get(name):return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id==name for a in n.targets))
main=get('CUDA_SRC')
helper=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Subscript) and isinstance(a.value,ast.Name) and a.value.id=='ALT_SRCS' and ast.literal_eval(a.slice)=='helper0' for a in n.targets))
MID=get('MID_DEFINES')+('-DOKC_WSH0=1',)
# 40 words per CTA. First8 retain the existing consumer boundary numbering.
def coarse(src):
 assert src.count('#define OKC_DEBUG 0')==1
 src=src.replace('#define OKC_DEBUG 0','#define OKC_DEBUG 1')
 assert src.count('p.dbg[blockIdx.x * 8 + (k)]')==1
 return src.replace('p.dbg[blockIdx.x * 8 + (k)]','p.dbg[blockIdx.x * 40 + (k)]')

def detail(src):
 # These hooks run only in the two producer lanes, never in a consumer loop.
 helpers=r'''
// Preparation diagnostics only; units are raw globaltimer ticks (nanoseconds).
__device__ __forceinline__ void probe_row(const Params& p,int row,int phase,
 unsigned long long begin,unsigned long long empty,unsigned long long samples,
 unsigned long long ready,unsigned long long ready_count,unsigned long long bytes,
 unsigned long long stages) {
 if(p.dbg) {
  unsigned long long* dst=p.dbg+(size_t)blockIdx.x*40+row+phase*8;
  dst[0]=begin;dst[1]=gtimer();dst[2]=empty;dst[3]=samples;
  dst[4]=ready;dst[5]=ready_count;dst[6]=bytes;dst[7]=stages;
 }
}
'''
 marker='__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {'
 assert src.count(marker)==1;src=src.replace(marker,helpers+'\n'+marker)
 a=src.index(marker);b=src.index('// x producer:',a);body=src[a:b]
 needle='    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);'
 assert body.count(needle)==1
 body=body.replace(needle,needle+'\n    const auto probe_begin=gtimer();\n    unsigned long long probe_empty=0,probe_samples=0;')
 old='      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);'
 new='''      if (n >= WGS_NS) {
        if ((n & 15)==0) {
          const auto before=gtimer();
          wgs_wait(S.bars + BW_E + ws,wph ^ 1,1000+gs);
          probe_empty+=gtimer()-before;++probe_samples;
        } else wgs_wait(S.bars + BW_E + ws,wph ^ 1,1000+gs);
      }'''
 assert body.count(old)==1;body=body.replace(old,new)
 end='''      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}'''
 assert body.count(end)==1
 body=body.replace(end,'''      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
    probe_row(p,8,pb,probe_begin,probe_empty,probe_samples,0,0,
              (unsigned long long)(g1-g0)*UNIT_BYTES,g1-g0);
  }
}''')
 src=src[:a]+body+src[b:]
 a=src.index('__device__ void wgs_xproducer(');b=src.index('// phase A\'s x gather',a);body=src[a:b]
 assert body.count(needle)==1
 body=body.replace(needle,needle+'\n    const auto probe_begin=gtimer();\n    unsigned long long probe_empty=0,probe_samples=0,probe_ready=0,probe_ready_count=0,probe_bytes=0;')
 old='      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);'
 new='''      if (n >= WGS_NX) {
        if ((n & 15)==0) {
          const auto before=gtimer();
          wgs_wait(S.bars + BX_E + xs,xph ^ 1,2000000+gs);
          probe_empty+=gtimer()-before;++probe_samples;
        } else wgs_wait(S.bars + BX_E + xs,xph ^ 1,2000000+gs);
      }'''
 assert body.count(old)==1;body=body.replace(old,new)
 old="if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }"
 new="""if (!pb && !(OKC_WGSX & 8) && ld_acquire(p.xready+j)<npad) {
          const auto before=gtimer();
          do { __nanosleep(32); } while (ld_acquire(p.xready+j)<npad);
          probe_ready+=gtimer()-before;++probe_ready_count;
        }"""
 assert body.count(old)==1;body=body.replace(old,new)
 old='if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }'
 new='''if (pb && !(OKC_WGSX & 8) && ld_acquire(p.doneAs+j*16+st)<8) {
        const auto before=gtimer();
        do { __nanosleep(32); } while (ld_acquire(p.doneAs+j*16+st)<8);
        probe_ready+=gtimer()-before;++probe_ready_count;
      }'''
 assert body.count(old)==1;body=body.replace(old,new)
 old='      const uint32_t bytes = (uint32_t)npad * 512u;'
 assert body.count(old)==1;body=body.replace(old,old+'\n      probe_bytes+=bytes;')
 end='''      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}'''
 assert body.count(end)==1
 body=body.replace(end,'''      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
    probe_row(p,24,pb,probe_begin,probe_empty,probe_samples,probe_ready,
              probe_ready_count,probe_bytes,g1-g0);
  }
}''')
 return src[:a]+body+src[b:]

cases=[{'name':'phase_main_control','source':main,'defines':get('MAIN_DEFINES'),'tiny':False},
 {'name':'phase_main_coarse','source':coarse(main),'defines':get('MAIN_DEFINES'),'tiny':False},
 {'name':'phase_main_detail','source':detail(coarse(main)),'defines':get('MAIN_DEFINES'),'tiny':False},
 {'name':'phase_mid_control','source':helper,'defines':MID,'tiny':True},
 {'name':'phase_mid_coarse','source':coarse(helper),'defines':MID,'tiny':True}]
for c in cases:
 assert c['source'].count('wgmma.mma_async')==main.count('wgmma.mma_async')
 (R/'research/spark'/f"{c['name']}.cu").write_text(c['source'])
(R/'research/spark/phase_probe_cases.json').write_text(json.dumps(cases)+'\n')
print('Prepared5 offline compiler cases; no H200 check dispatched.')
