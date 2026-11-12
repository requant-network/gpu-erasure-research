"""Read-only reported PCIe-link qualification for the two-GPU experiment."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

command=['nvidia-smi','--query-gpu=index,name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max','--format=csv,noheader,nounits']
raw=subprocess.check_output(command,text=True);devices=[]
for fields in csv.reader(raw.splitlines()):
    index,name,current_gen,max_gen,current_width,max_width=[s.strip() for s in fields]
    devices.append(dict(index=int(index),name=name,current_generation=int(current_gen),reported_max_generation=int(max_gen),
                        current_width=int(current_width),reported_max_width=int(max_width)))
assert len(devices)==2
record=dict(utc=datetime.now(timezone.utc).isoformat(),devices=devices,command=command,
            source_sha256={'interconnect_probe.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            scope='Read-only nvidia-smi report during the campaign after both GPUs were freed. Link metadata is not a measured transfer-bandwidth result or a claim about another host. No PCIe/clock/power setting changed.')
Path('results/multigpu-interconnect.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(devices))
