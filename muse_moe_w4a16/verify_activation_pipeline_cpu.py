"""CPU checks for the experimental activation/warp pipeline.

Exercises actual routing helpers and packed-layout declarations. Verifies
expert-major stage boundaries and all possible ordered producer/consumer
interleavings. Does not establish GPU compilation, race freedom, or performance.
"""
import ast
from pathlib import Path
import numpy as np
from verify_register_cpu import CPU, Ptr, tensor

ROOT = Path(__file__).resolve().parent


def kernels(path):
    tree = ast.parse(path.read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in {
                '_expert_starts','_stream_source_rows','_b256_compact_routes','_b256_work_plan'}:
            node.decorator_list=[]
            for arg in node.args.args: arg.annotation=None
            selected.append(node)
    cpu=CPU();scope={'tl':cpu,'gl':cpu}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),scope)
    return cpu,scope


def bf16(x):
    x=np.asarray(x,np.float32)
    bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def test_routes_and_stage_boundaries(path):
    cpu,functions=kernels(path)
    rng=np.random.default_rng(191)
    # Small dimensions keep the same64/256-tile geometry but permit a CPU oracle.
    h,i=256,64
    w13=rng.normal(0,.04,(32,2*i,h)).astype(np.float32)
    w2=rng.normal(0,.04,(32,h,i)).astype(np.float32)
    for m in (1,4,8,16,32,33,128,256,257):
        p=m*8
        for distribution in ('normal','hot','empty','local'):
            ids=rng.integers(0,256,(m,8),dtype=np.int32)
            if distribution=='hot':ids[:,::2]=3
            elif distribution=='empty':ids[:]=255
            elif distribution=='local':ids%=32
            x=bf16(rng.normal(size=(m,h)))
            mapping=np.full((32,p),-999,np.int32);counts=np.zeros(32,np.int32)
            for e in range(32):
                cpu.pid=(e,)
                functions['_b256_compact_routes'](Ptr(ids),Ptr(mapping),Ptr(counts),p,0,32,1<<(p-1).bit_length())
            starts=np.full(32,-999,np.int32)
            functions['_expert_starts'](Ptr(counts),Ptr(starts),32,32)
            np.testing.assert_array_equal(starts,np.cumsum(counts)-counts)
            assert counts.sum()<=p
            # Each route's BF16 gate/up, SiLU and intermediate, in original order.
            ref=np.zeros((p,h),np.float32)
            compact=np.full((p,i),np.nan,np.float32)
            for e in range(32):
                routes=mapping[e,:counts[e]]
                z=bf16(x[routes//8]@w13[e].T)
                gate,up=np.split(z,2,axis=1)
                silu=bf16(gate/(1+np.exp(-gate)))
                mid=bf16(silu*up)
                ref[routes]=bf16(np.einsum('rk,hk->rh',mid,w2[e],optimize=False))
                compact[starts[e]:starts[e]+counts[e]]=mid
            for bm in (8,16,32):
                wmax=1<<(((p+bm-1)//bm+31).bit_length())
                work=np.zeros((wmax,3),np.int32);nw=np.zeros(1,np.int32)
                functions['_b256_work_plan'](Ptr(counts),Ptr(work),Ptr(nw),32,bm,wmax,32)
                out=np.zeros((p,h),np.float32);seen=[]
                for e,lo,hi in work[:nw[0]]:
                    rows=tensor(np.arange(lo,lo+bm),np.int32)
                    for gu in (True,False):
                        for packed in (False,True):
                            actual=functions['_stream_source_rows'](Ptr(mapping),Ptr(starts),e,rows,hi,p,packed,gu)
                            if packed and not gu:expected=starts[e]+rows
                            else:
                                expected=np.zeros(bm,np.int32)
                                expected[:hi-lo]=mapping[e,lo:hi]//(8 if gu else 1)
                            np.testing.assert_array_equal(actual,expected)
                    routes=mapping[e,lo:hi]
                    mid=compact[starts[e]+lo:starts[e]+hi]
                    assert np.isfinite(mid).all()
                    out[routes]=bf16(np.einsum('rk,hk->rh',mid,w2[e],optimize=False))
                    seen.extend((starts[e]+np.arange(lo,hi)).tolist())
                np.testing.assert_array_equal(out,ref)
                np.testing.assert_array_equal(np.sort(seen),np.arange(counts.sum()))
    print('PASS actual helpers:9shapes x4routing distributions x3tile sizes; BF16 stage boundaries; compact addresses and coverage')


def test_layout_offsets(path):
    tree=ast.parse(path.read_text())
    layouts={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name in {'_b256LX8','_b256LX','_b256LX32'}:
                layouts[name]={kw.arg:ast.literal_eval(kw.value) for kw in n.value.args[0].keywords}
    assert len(layouts)==3
    for spec in layouts.values():
        bases=spec['reg_bases']+spec['lane_bases']+spec['warp_bases']
        unique=set()
        for value in range(1<<len(bases)):
            coord=np.zeros(3,np.int32)
            for bit,basis in enumerate(bases):
                if value>>bit&1:coord^=basis
            unique.add(tuple(coord))
        bm=spec['shape'][0]
        assert unique==set(np.ndindex(bm,4,64))
        # Four warps replicate each input; no shared exchange is needed.
        assert (1<<len(bases))==4*bm*256
        # Apply the exact reshaping/permutation used by the candidate's _x_tile.
        offsets=np.arange(bm*256).reshape(bm,4,64)
        actual=offsets.reshape(bm,4,8,2,2,2).transpose(2,3,4,1,5,0).reshape(256,bm)
        k=np.arange(256)
        physical=(k//2%4)*64+(k//32)*8+(k//16%2)*4+(k//8%2)*2+k%2
        expected=np.arange(bm)[None,:]*256+physical[:,None]
        np.testing.assert_array_equal(actual,expected)
    print('PASS activation layout coverage, four-warp replication, physical/logical K permutation')


def test_ring_protocol():
    # Enumerate every possible ordering of complete producer/consumer steps.
    # Separate GPU memory visibility and compiler behavior still require a GPU.
    for blocks in (1,2,3,4,16,17,32,64):
        todo=[(0,0,(1,1),(1,1),(-1,-1))];seen=set();terminal=False
        while todo:
            state=todo.pop()
            if state in seen:continue
            seen.add(state)
            produced,consumed,empty,ready,contents=state
            assert 0<=produced-consumed<=2
            if produced==consumed==blocks:terminal=True;continue
            successors=[]
            if produced<blocks:
                slot=produced%2;phase=produced//2&1
                if empty[slot]==phase^1:
                    assert contents[slot]==-1
                    rr=list(ready);cc=list(contents)
                    rr[slot]=phase;cc[slot]=produced
                    successors.append((produced+1,consumed,empty,tuple(rr),tuple(cc)))
            if consumed<blocks:
                slot=consumed%2;phase=consumed//2&1
                if ready[slot]==phase:
                    assert contents[slot]==consumed
                    ee=list(empty);cc=list(contents)
                    ee[slot]=phase;cc[slot]=-1
                    successors.append((produced,consumed+1,tuple(ee),ready,tuple(cc)))
            assert successors,'deadlock in abstract ring protocol'
            todo.extend(successors)
        assert terminal
    print('PASS two-buffer phase protocol: all interleavings,1..64K steps, no premature reuse/deadlock in abstract model')


if __name__=='__main__':
    for file in ('register_activation_pipeline.py','register_warp_pipeline.py'):
        path=ROOT/file
        compile(path.read_text(),str(path),'exec')
        test_routes_and_stage_boundaries(path)
        test_layout_offsets(path)
    test_ring_protocol()
    print('CPU checks passed. GPU compilation, synchronization and performance remain UNVERIFIED.')
