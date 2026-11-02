"""Execute the sector-packed candidate's actual packer and all three read offsets.
Does not establish GPU load vectorization, correctness or performance.
"""
import ast
from pathlib import Path
import numpy as np
from verify_register_cpu import CPU,Ptr,pack,packed_weights,raw_weights

ROOT=Path(__file__).resolve().parent
path=ROOT/'register_sector_packed.py'
tree=ast.parse(path.read_text())
names={'_b256_pack_codes','_b256_pack_scales','_b256_pack_weights'}
class Rename(ast.NodeTransformer):
    def visit_Name(self,n):
        if n.id in names:n.id=n.id.replace('_b256','',1)
        return n
cpu=CPU();scope={'tl':cpu};body=[]
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name in names:
        n.name=n.name.replace('_b256','',1);n.decorator_list=[]
        for arg in n.args.args:arg.annotation=None
        body.append(Rename().visit(n))
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),scope)
j=np.arange(2048,dtype=np.int32)
permutation=(j%8//4)*1024+(j//32)*16+(j%32//8)*4+j%4
np.testing.assert_array_equal(np.sort(permutation),j)
rng=np.random.default_rng(20261030)
for n,k in ((64,256),(128,512),(256,768)):
    raw=rng.integers(0,256,(2,n,k//2),dtype=np.uint8)
    sf=rng.integers(0,127,(2,n,k//16),dtype=np.uint8)
    for gu in (False,True):
        codes,packed_sf=pack(cpu,scope,raw,sf,gu)
        canonical=codes.reshape(*codes.shape[:3],2048)[...,permutation].reshape(codes.shape)
        rows=np.arange(n)
        if gu:rows=(rows%16//8)*(n//2)+(rows//64)*32+(rows%64//16)*8+rows%8
        np.testing.assert_array_equal(packed_weights(canonical,packed_sf),raw_weights(raw,sf)[:,rows])
        assert codes.nbytes==raw.nbytes and packed_sf.nbytes==sf.nbytes
# Execute each actual consumer offset AST expression, not a retyped read formula.
source=ast.parse(path.read_text())
for function,target in (('_b256_register_item','wo'),('_one_direct_first_partial','wo'),('_one_direct_w2_split','w_off')):
    f=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==function)
    expr=next(n.value for n in ast.walk(f) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
    rr=np.arange(64)[:,None,None];qq=np.arange(4)[None,:,None];jj=np.arange(8)[None,None,:]
    context={'e':0,'N':64,'BN':64,'pn':0,'pid_n':0,'blocks':1,'KB':1,'rw':rr,'qw':qq,'jw':jj}
    actual=eval(compile(ast.Expression(expr),'<consumer-read-offset>','eval'),context)
    np.testing.assert_array_equal(actual.reshape(-1),permutation)
# Transaction model for one row-group, assuming a4-word vector load per lane.
# This counts unique32-byte sectors for each instruction separately. It is not
# a measurement of DRAM traffic; subsequent instructions can hit the same cache.
old=new=0
for base in (0,4):
    old_bytes=[];new_bytes=[]
    for lane in range(32):
        row=lane//4;q=lane%4
        for word in range(base,base+4):
            old_bytes.append(4*(row*32+q*8+word))
            new_bytes.append(4*((word//4)*1024+row*16+q*4+word%4))
    old+=len({x//32 for x in old_bytes});new+=len({x//32 for x in new_bytes})
assert (old,new)==(64,32)
print('PASS actual lossless packer, all three consumer offsets, random tile banks, gate/up and ordinary rows')
print('PASS conditional vector-load model:32-byte sector requests64->32; actual GPU vectorization/traffic unverified')
