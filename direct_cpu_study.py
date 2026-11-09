"""Match the transfer campaign's exact payloads with a direct non-OpenMP baseline."""
import argparse,ctypes,hashlib,json,pathlib,time
from datetime import datetime,timezone
import numpy as np
from experiments.field import generator,recovery

assert __debug__
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',default='results/transfer-study.json')
parser.add_argument('--output',default='results/direct-cpu-study.json');args=parser.parse_args()
path=pathlib.Path(args.reference);reference=json.loads(path.read_text(encoding='utf-8'));rng=np.random.default_rng(reference['seed'])
library=ctypes.CDLL(str(pathlib.Path('build/libisal_parallel.so').resolve()));pointer=ctypes.c_void_p
library.requant_isal_tables.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer];library.requant_isal_tables.restype=None
library.requant_isal_pipeline.argtypes=[ctypes.c_int]*3+[pointer]*8;library.requant_isal_pipeline.restype=None
def tables(coeff):
    result=np.empty(coeff.shape[0]*coeff.shape[1]*32,np.uint8)
    library.requant_isal_tables(coeff.shape[1],coeff.shape[0],coeff.ctypes.data,result.ctypes.data);return result
cases=[]
for fixture in reference['cases']:
    k=fixture['k'];payload=fixture['payload_bytes'];size=payload//k;missing=fixture['missing']
    data=rng.integers(0,256,(k,size),np.uint8)
    assert hashlib.sha256(data.tobytes()).hexdigest()==fixture['input_sha256']
    available,coeff=recovery(k,4,missing);et=tables(np.array(generator(k,4)[k:],np.uint8));rt=tables(np.array(coeff,np.uint8))
    indices=np.array(available,np.int32);lost=np.array(missing,np.int32)
    parity=np.empty((4,size),np.uint8);recovered=np.empty_like(parity);restored=np.empty_like(data);samples=[]
    for iteration in range(35):
        start=time.perf_counter()
        library.requant_isal_pipeline(k,4,size,et.ctypes.data,rt.ctypes.data,indices.ctypes.data,lost.ctypes.data,data.ctypes.data,parity.ctypes.data,recovered.ctypes.data,restored.ctypes.data)
        wall=(time.perf_counter()-start)*1000;assert np.array_equal(restored,data)
        if iteration>=5:samples.append(wall)
    cases.append(dict(k=k,parity=4,payload_bytes=payload,missing=missing,input_sha256=fixture['input_sha256'],every_payload_exact=True,samples_ms={'direct_cpu_pipeline':samples}))
    print(k,payload,float(np.median(samples)),flush=True)
sources=['direct_cpu_study.py','experiments/field.py','native/isal_bridge.cpp']
record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=reference['seed'],warmups=5,samples=30,cases=cases,
    source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},
    result_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
    scope='Direct single-thread ISA-L full encode/recovery/assembly without an OpenMP team. Exact same seeded payloads as the referenced transfer study. Five warm-ups/thirty repeats, preallocated matrices/buffers. Separate execution window, not interleaved with GPU modes; do not infer precise paired speedups from cross-window timings. Retained to prevent OpenMP setup overhead from weakening the small-payload CPU baseline.')
pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
