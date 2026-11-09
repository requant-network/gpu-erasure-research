"""Derive per-cohort latency and resource tables from retained measurements."""
import csv,json,math,pathlib,statistics

rows=[]
for name in ('gpu-study','gpu-replication','transfer-study','transfer-replication','direct-cpu-study','direct-cpu-replication'):
    record=json.loads(pathlib.Path(f'results/{name}.json').read_text())
    for case in record['cases']:
        samples=case.get('samples',case.get('samples_ms'))
        for mode,scopes in samples.items():
            if isinstance(scopes,list):scopes={'pipeline_wall_ms':scopes}
            for scope,values in scopes.items():
                ordered=sorted(values);assert len(values)==30 and all(math.isfinite(v) and v>0 for v in values)
                rows.append(dict(campaign=name,seed=record['seed'],k=case['k'],parity=4,payload_bytes=case['payload_bytes'],
                    missing=':'.join(map(str,case['missing'])),mode=mode,scope=scope,samples=len(values),
                    median_ms=f'{statistics.median(values):.6f}',p95_ms=f'{ordered[math.ceil(.95*len(values))-1]:.6f}',
                    observed_pool_bytes=case.get('device_pool_reserved_bytes',case.get('observed_device_pool_bytes',''))))
with pathlib.Path('results/summary.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(rows),'latency/resource summaries, with cohorts kept separate')
