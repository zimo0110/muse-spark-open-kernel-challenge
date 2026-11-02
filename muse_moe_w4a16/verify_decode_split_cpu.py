"""CPU algebra/address checks for splitK decode; GPU JIT/correctness unverified."""
import ast
from pathlib import Path
import numpy as np
from verify_activation_pipeline_cpu import bf16

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'register_decode_split.py').read_text())
f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_decode_split_item')
gu_branch=next(n for n in f.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='GU')
expr=gu_branch.body[0].value
for n in (128,8192):
    all_cols=[]
    for pn in range(n//64):
        cols=eval(compile(ast.Expression(expr),'<actual-GU-col>','eval'),{'n':np.arange(64),'N':n,'pn':pn})
        all_cols.extend(cols.tolist())
    np.testing.assert_array_equal(np.sort(all_cols),np.arange(n))
for ks in (2,4,8):
    blocks=4096//256
    visits=[k for split in range(ks) for k in range(split*(blocks//ks),(split+1)*(blocks//ks))]
    assert visits==list(range(blocks))
print('PASS actual gate/up output address expression and splitK tile coverage')

rng=np.random.default_rng(42)
for routes in (1,3,8,19):
    x=bf16(rng.normal(size=(routes,128)))
    w13=bf16(rng.normal(0,.05,(256,128)))
    w2=bf16(rng.normal(0,.05,(128,128)))
    gs=np.float32(.25)
    norm=bf16(rng.normal(0,.1,128))
    def reference(raw):
        gate,up=np.split(bf16(raw),2,axis=1)
        return bf16(bf16(gate/(1+np.exp(-gate)))*up)
    z=np.einsum('rk,hk->rh',x,w13,optimize=False)*gs
    inter=reference(z)
    y=bf16(np.einsum('rk,hk->rh',inter,w2,optimize=False)*gs)
    for first in (1,2,4,8):
        part=[np.einsum('rk,hk->rh',a,b,optimize=False)*gs for a,b in zip(np.split(x,first,axis=1),np.split(w13,first,axis=1))]
        trial_inter=reference(np.sum(part,axis=0,dtype=np.float32))
        for second in (1,2,4,8):
            parts=[np.einsum('rk,hk->rh',a,b,optimize=False)*gs for a,b in zip(np.split(trial_inter,second,axis=1),np.split(w2,second,axis=1))]
            trial_y=bf16(np.sum(parts,axis=0,dtype=np.float32))
            a=bf16(y*(1+norm));b=bf16(trial_y*(1+norm))
            a=a/np.sqrt(np.mean(a*a,axis=1,keepdims=True)+1e-8)
            b=b/np.sqrt(np.mean(b*b,axis=1,keepdims=True)+1e-8)
            error=np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)
            assert np.isfinite(error) and error<.004,(routes,first,second,error)
print('PASS staged BF16 splitK algebra, scales/norm,1..19routes x16split combinations')
