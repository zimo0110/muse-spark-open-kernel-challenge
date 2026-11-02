"""Prepare a paired cache-policy experiment, without consuming a check."""
import ast,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
text=(root/'cuda_persistent_two_wg_paired.py').read_text()
candidate=Path(sys.argv[1]) if len(sys.argv)>1 else Path('cuda_persistent_reuse_cache.py')
assert candidate.name==str(candidate) and candidate.suffix=='.py'
label=candidate.stem.removeprefix('cuda_persistent_')
assert label.replace('_','').isalnum()
cache=ast.parse((root/candidate).read_text())
def assignment(tree,name):
 return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id==name for k in n.targets))
replacements=[];lines=text.splitlines(keepends=True)
for name in ('CUDA_SRC','MAIN_DEFINES'):
 old=assignment(ast.parse(text),name);new=assignment(cache,name)
 a=sum(map(len,lines[:old.value.lineno-1]))+old.value.col_offset;b=sum(map(len,lines[:old.value.end_lineno-1]))+old.value.end_col_offset
 replacements.append((a,b,repr(ast.literal_eval(new.value))))
for a,b,s in sorted(replacements,reverse=True):text=text[:a]+s+text[b:]
text=text.replace('two_wg_striped',label).replace('(64, 128, 192, 256)','(32, 64, 128, 192, 256)')
ast.parse(text);(root/(candidate.stem+'_paired.py')).write_text(text)
print('Prepared paired experiment',label,'; NOT dispatched.')
