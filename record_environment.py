"""Collect non-identifying compiler, package and hardware research metadata."""
import hashlib,importlib.metadata,json,pathlib,platform,subprocess
from datetime import datetime,timezone
import cupy as cp
properties=cp.cuda.runtime.getDeviceProperties(0)
name=properties['name'].decode() if isinstance(properties['name'],bytes) else properties['name']
columns=['name','driver_version','memory.total','power.limit','pcie.link.gen.max','pcie.link.width.max','pcie.link.gen.current','pcie.link.width.current']
cpu='unknown'
for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines():
    if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
packages={p:importlib.metadata.version(p) for p in ('numpy','cupy-cuda12x','nvidia-cuda-nvrtc-cu12','nvidia-cuda-runtime-cu12')}
record=dict(utc=datetime.now(timezone.utc).isoformat(),os=platform.system(),architecture=platform.machine(),kernel=platform.release(),python=platform.python_version(),cpu=cpu,
    device=dict(name=name,major=properties['major'],minor=properties['minor'],multiprocessors=properties['multiProcessorCount'],total_global_memory=properties['totalGlobalMem']),
    packages=packages,cuda_runtime_api=cp.cuda.runtime.runtimeGetVersion(),cuda_driver_api=cp.cuda.runtime.driverGetVersion(),nvrtc=cp.cuda.nvrtc.getVersion(),
    nvcc=subprocess.check_output(['/usr/local/cuda-13.3/bin/nvcc','--version'],text=True).strip(),gxx=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
    nvidia_smi_columns=columns,nvidia_smi_values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu='+','.join(columns),'--format=csv,noheader'],text=True).strip().split(', '),
    source_sha256={'record_environment.py':hashlib.sha256(pathlib.Path('record_environment.py').read_bytes()).hexdigest()},
    policy='No intentional concurrent GPU research jobs; no clock, voltage or power-limit changes. Dynamic clocks, thermals and OS activity remain possible confounders. Metadata excludes usernames, hostnames, device UUIDs and credentials.')
pathlib.Path('results/environment.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
