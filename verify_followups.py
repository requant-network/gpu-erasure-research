"""CPU replay of follow-up evidence, provenance and derived timing summaries."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
assert __debug__
PATTERNS = ('recovery-*.json','multigpu-*.json','transfer-balance-*.json')
paths=sorted(set(p for pattern in PATTERNS for p in Path('results').glob(pattern)))
assert len(paths)==9,(len(paths),[p.name for p in paths])
records={p.name:json.loads(p.read_text()) for p in paths}
def binding(name,expected,result=False):
 p=(Path('results')/name) if result else Path(name)
 assert not Path(name).is_absolute() and ':' not in name and '..' not in Path(name).parts
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,(p,expected)
def provenance(r):
 if isinstance(r,dict):
  for key,v in r.items():
   if key in ('source_sha256','artifacts_sha256','public_artifacts_sha256','result_sha256'):
    for name,expected in v.items():binding(name,expected,key=='result_sha256')
   elif isinstance(v,(dict,list)):provenance(v)
 elif isinstance(r,list):
  for v in r:provenance(v)
for r in records.values():provenance(r)
assert len(records['recovery-study.json']['cases'])==len(records['recovery-replication.json']['cases'])==24
links=records['multigpu-interconnect.json']['devices'];assert len(links)==2
assert all(d['current_width']>0 and d['current_generation']>0 for d in links)
for name,r in records.items():
 if 'cases' not in r:continue
 if name.startswith('recovery-'):assert r['invalid_native_calls_rejected']==5
 else:
  expected_seeds={'multigpu-study.json':20261113,'multigpu-replication.json':20261213,'transfer-balance-study.json':20261114,'transfer-balance-replication.json':20261214}
  assert r['seed']==expected_seeds[name] and r['warmups']==5 and r['samples']==30
  assert len(r['cases'])==(2 if name.startswith('transfer-balance') else 9) and r['field_products_checked_per_gpu']==65536
  assert [(d['major'],d['minor']) for d in r['hardware']]==[(12,0),(7,5)]
  if name.startswith('transfer-balance'):
   from transfer_balance_study import transformed_source
   assert hashlib.sha256(transformed_source().encode()).hexdigest()==r['transformed_main_sha256']
 for c in r['cases']:
  assert c['every_restored_payload_exact'] and len(c['missing'])==4
  if name.startswith('recovery-'):assert len(c['samples_ms'])==8 and c['batch'] in (1,8)
  else:
   assert c['output_cleared_before_every_launch'] and len(c['samples'])==(7 if name.startswith('transfer-balance') else 9)
   assert set(c['partition_bytes'])==({'gpu0','dual9375','dual96875'} if name.startswith('transfer-balance') else {'gpu0','gpu1','dual50','dual75','dual875'})
   assert all(sum(v)==c['payload_bytes'] for v in c['partition_bytes'].values())
   assert set(c['samples'])==set(c['partition_bytes'])|{f'cpu_{n}' for n in (1,2,4,6)}
   for mode,scopes in c['samples'].items():
    assert set(scopes)==({'complete_wall_ms'} if mode.startswith('cpu_') else {'resident_wall_ms','staged_wall_ms'})

rows=[]
def timings(value,labels,route=''):
 if isinstance(value,list):
  assert len(value)==30 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in value),(labels,route)
  rows.append(dict(**labels,variant_scope=route,median_ms=median(value),p95_ms=sorted(value)[28],repetitions=len(value)))
 elif isinstance(value,dict):
  for key,v in value.items():timings(v,labels,(route+'/' if route else '')+key)
for name,r in records.items():
 for index,c in enumerate(r.get('cases',[])):
  labels=dict(file=name,case=index,seed=c.get('seed',r.get('seed','')),**{key:c.get(key,'') for key in ('application','n','m','k','p','columns','steps','payload_bytes','batch','variant','pattern')})
  v=c.get('result',c)
  for key in ('samples','samples_ms','resident_graph_device_ms'):
   if isinstance(v.get(key),(dict,list)):timings(v[key],labels,key)
assert len(rows)==676,len(rows)
if rows:
 with Path('results/followup-summary.csv').open('w',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(records),'follow-up records validated;',len(rows),'median/p95 timing rows')
