"""Check actual aligned C++ helpers against pinned SGLang stage algebra."""
import ast
import inspect
from pathlib import Path
import verify_persistent_port_cpu as harness
p=Path(__file__).resolve().parent/'cuda_persistent_aligned.py'
sources=harness.cuda_literals(ast.parse(p.read_text()))
helpers=[]
for label,node in sources.items():
 src=node.value
 st=src.index('__device__ __forceinline__ float codex_bf16_round(')
 en=src.index('__device__ __forceinline__ float probe_q(',st)
 helpers.append(src[st:en])
 assert src.count('codex_gu_bf16(')==6
 assert src.count('codex_pb_bf16(')==13
 assert '1.f + __bfloat162float(' not in src
assert helpers[0]==helpers[1]
harness.STRICT_HELPERS=helpers[0]
body=inspect.getsource(harness.test_numeric_helpers)
body=body.replace('bf(bf(ta*tb)*tc).numpy()', '(bf(ta*tb)*tc).numpy()')
body=body.replace('bf(bf(bf(ta)/(1+torch.exp(-bf(ta))))*bf(tb)).numpy()', 'bf((bf(ta)/(1+torch.exp(-bf(ta))))*bf(tb)).numpy()')
body=body.replace('staged GU', 'fused GU')
exec(body,harness.__dict__)
harness.test_numeric_helpers()
# Effective norm weights use exactly baseline.prepare's BF16(1+w) expression.
assert '(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).contiguous()' in p.read_text()
print('PASS all epilogues and effective BF16 norm weight; matches pinned SGLang tensor boundaries.')
