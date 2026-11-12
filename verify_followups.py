"""CPU replay of follow-up evidence, provenance and derived timing summaries."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
assert __debug__
PATTERNS = ('recovery-*.json',)
paths=sorted(set(p for pattern in PATTERNS for p in Path('results').glob(pattern)))
assert len(paths)==3,(len(paths),[p.name for p in paths])
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
for name,r in records.items():
 if 'cases' not in r:continue
 assert r['invalid_native_calls_rejected']==5
 for c in r['cases']:
  assert c['every_restored_payload_exact'] and len(c['missing'])==4
  assert len(c['samples_ms'])==8 and c['batch'] in (1,8)

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
if rows:
 with Path('results/followup-summary.csv').open('w',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print(len(records),'follow-up records validated;',len(rows),'median/p95 timing rows')
