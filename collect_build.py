"""Bind measured binaries, original sources, upstream revision and raw records."""
import hashlib,json,pathlib,subprocess
revision='ab01645a2d9ab581477c703103fe3450c4696845'
upstream=pathlib.Path('../../isa-l')
assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==revision
assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip()
names=['gpu-study.json','gpu-replication.json','transfer-study.json','transfer-replication.json','direct-cpu-study.json','direct-cpu-replication.json']
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
binaries={p:digest(p) for p in ('build/erasure.cubin','build/libisal_bridge.so','build/libisal_parallel.so','../../isal-build/libisal.so')}
for name in names[:2]:
    for p,h in json.loads(pathlib.Path('results',name).read_text())['binary_sha256'].items():assert binaries[p]==h
for name in names[2:4]:assert json.loads(pathlib.Path('results',name).read_text())['binary_sha256']==binaries['build/libisal_parallel.so']
record=dict(upstream_revision=revision,upstream_clean=True,gxx=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
    nvcc=subprocess.check_output(['/usr/local/cuda-13.3/bin/nvcc','--version'],text=True).strip(),
    cuda_flags=['-O3','-std=c++17','-arch=sm_120','-cubin'],bridge_flags=['-O3','-shared','-fPIC'],parallel_flags=['-O3','-shared','-fPIC','-fopenmp'],
    isal_configuration=dict(build_type='Release',shared=True,tests=False,perf=False,shim=False,nasm='2.16.01',cmake='4.1.3'),
    binary_sha256={pathlib.Path(p).name:h for p,h in binaries.items()},
    result_sha256={p:digest(pathlib.Path('results',p)) for p in names},
    source_sha256={p:digest(p) for p in ('collect_build.py','cuda/erasure.cu','native/isal_bridge.cpp','native/isal_parallel.cpp')},
    scope='Post-run provenance of the exact measured binaries, not a promise of byte-identical rebuilds. ISA-L upstream remains unmodified. Original wrappers implement full-pipeline orchestration and optional OpenMP chunking.')
pathlib.Path('results/build.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
