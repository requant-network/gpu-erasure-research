"""Exploratory transfer-aware refinement with unchanged arithmetic and checks."""
import ast
import hashlib
from pathlib import Path

def transformed_source():
    text=Path('multigpu_study.py').read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    source=''.join(text.splitlines(True)[node.lineno-1:node.end_lineno])
    replacements={
        "default=20261113":"default=20261114",
        "default='results/multigpu-study.json'":"default='results/transfer-balance-study.json'",
        "profiles=[(k,size,missing) for k in (8,16) for size in (16<<20,256<<20) for missing in ([0,1,2,3],[0,1,k,k+1])]+[(8,1<<30,[0,1,2,3])]":"profiles=[(8,size,[0,1,2,3]) for size in (256<<20,1<<30)]",
        "(('gpu0',1),('gpu1',0),('dual50',.5),('dual75',.75),('dual875',.875))":"(('gpu0',1),('dual9375',.9375),('dual96875',.96875))",
        "'multigpu_study.py','build_multigpu.py'":"'transfer_balance_study.py','multigpu_study.py','build_multigpu.py'",
        'dual50/75/87.5percent assigned toGPU0;':'transfer-aware93.75/96.875percent assigned toGPU0;exploratory refinement after observing narrowGPU1PCIe,not a claim of global optimality;'}
    for before,after in replacements.items():
        assert source.count(before)==1,before;source=source.replace(before,after)
    return source

def main():
    import multigpu_study as original
    source=transformed_source();digest=hashlib.sha256(source.encode()).hexdigest()
    assert source.count('record=dict(utc=')==1
    source=source.replace('record=dict(utc=',f"record=dict(transformed_main_sha256='{digest}',utc=")
    namespace=dict(vars(original));namespace['__doc__']=__doc__
    exec(compile(source,'<recorded-transfer-balance-main>','exec'),namespace);namespace['main']()

if __name__=='__main__':main()
