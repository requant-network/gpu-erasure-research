"""Check retained evidence structure; CPU arithmetic is checked by cpu_study.py."""
import json,math,pathlib

total=0
for stem,seed in (('gpu-study',20261109),('gpu-replication',20261110),('transfer-study',20261111),('transfer-replication',20261112)):
    r=json.loads(pathlib.Path(f'results/{stem}.json').read_text());assert r['seed']==seed and r['warmups']==5 and r['samples']==30
    assert len(r['cases'])==(20 if stem.startswith('gpu') else 10)
    if stem.startswith('gpu'):
        assert r['polynomial']=='0x11d' and len(r['validation'])==15 and all(c['all_variants_exact'] for c in r['validation'])
    for c in r['cases']:
        assert c['k'] in (8,16) and c['parity']==4 and len(set(c['missing']))==4
        assert c['payload_bytes'] in (65536,1048576,16777216,134217728,268435456)
        assert c.get('every_output_exact',c.get('every_restored_payload_exact'))
        modes=c.get('samples',c.get('samples_ms'));assert len(modes)==(5 if stem.startswith('gpu') else 8)
        for scopes in modes.values():
            if isinstance(scopes,list):scopes={'pipeline':scopes}
            for values in scopes.values():assert len(values)==30 and all(math.isfinite(v) and v>0 for v in values)
        total+=1
for suffix in ('study','replication'):
    reference=json.loads(pathlib.Path(f'results/transfer-{suffix}.json').read_text())
    direct=json.loads(pathlib.Path(f'results/direct-cpu-{suffix}.json').read_text())
    assert direct['seed']==reference['seed'] and len(direct['cases'])==10
    for a,b in zip(reference['cases'],direct['cases']):
        for key in ('k','payload_bytes','missing','input_sha256'):assert a[key]==b[key]
        assert b['every_payload_exact'] and len(b['samples_ms']['direct_cpu_pipeline'])==30
        assert all(math.isfinite(v) and v>0 for v in b['samples_ms']['direct_cpu_pipeline'])
print(total,'GPU/CPU cases and 20 exact-payload direct baseline cases checked')
