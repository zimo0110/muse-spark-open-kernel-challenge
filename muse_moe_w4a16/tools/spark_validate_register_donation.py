"""Portable math/replay test; does NOT validate Hopper register reallocation.

Uses the established full-64-token warp-MMA emulator with single-slot rings.
Register redistribution is stripped for GB10; target execution still needs
H200 checks. Input lifetimes are retained by the existing validation runner.
"""
import json,re,sys
from pathlib import Path
import spark_validate_fixed_width as suite
R=Path(__file__).resolve().parent
old_load=suite.load_core

def load(label,src,defines):
 src=re.sub(r'asm volatile\("setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 \d+;":::"memory"\);','__syncwarp();',src)
 return old_load(label,src,defines)

if __name__=='__main__':
 case_file=Path(sys.argv[1]) if len(sys.argv)>1 else R.parent/'research/spark/register_donation_cases.json'
 if not case_file.exists():case_file=R/'register_donation_cases.json'
 cases=json.loads(case_file.read_text())
 cases=[q for q in cases if q['name']=='register_donation_104_64_gsnarrow4']
 assert len(cases)==1
 (R/'register_donation_validation_cases.json').write_text(json.dumps(cases)+'\n')
 suite.load_core=load
 sys.argv=[sys.argv[0],'register_donation_validation_cases.json']
 suite.main()
 # The shared runner uses this fixed output name. Copy before another run.
 d=json.loads((R/'fixed_width_validation_result.json').read_text())
 d['register_reallocation_validation']=False
 d['development_changes']='setmaxnreg stripped; synchronous warp-MMA; NS/NX1'
 (R/'register_donation_emulation_result.json').write_text(json.dumps(d,indent=2)+'\n')
