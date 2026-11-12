"""Recovery-only application batches with exact direct/multicore CPU baselines."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OMP_PROC_BIND'] = 'TRUE'
os.environ['OMP_PLACES'] = 'cores'
import cupy as cp
import numpy as np
from experiments.field import generator, recovery


def main():
    assert __debug__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--output', default='results/recovery-study.json')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    pointer = ctypes.c_void_p
    baseline = ctypes.CDLL(str(Path('build/libisal_bridge.so').resolve()))
    baseline.requant_isal_tables.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer]
    baseline.requant_isal_encode.argtypes = [ctypes.c_int]*3 + [pointer]*3
    library = ctypes.CDLL(str(Path('build/libisal_recovery.so').resolve()))
    recover = library.requant_isal_recover
    recover.argtypes = [ctypes.c_int]*4 + [pointer]*5; recover.restype = ctypes.c_int
    for k, rows, size, threads in ((0,4,16,1),(8,0,16,1),(8,4,0,1),(8,4,16,0),(8,4,16,13)):
        assert recover(k, rows, size, threads, None, None, None, None, None) == -1
    module = cp.RawModule(path=str(Path('build/erasure.cubin').resolve()))
    encode, restore = [module.get_function(name) for name in ('encode_clmad4','restore')]
    def tables(coeff):
        result = np.empty(coeff.shape[0]*coeff.shape[1]*32, np.uint8)
        baseline.requant_isal_tables(coeff.shape[1], coeff.shape[0], coeff.ctypes.data, result.ctypes.data)
        return result
    rng = np.random.default_rng(args.seed); cases = []
    profiles = [(65536,1)] if args.quick else [(65536,1),(1048576,1),(16777216,1),(268435456,1),(1048576,8),(16777216,8)]
    warmups, repetitions = (1,1) if args.quick else (5,30)
    for k in (8,16):
      for missing in ([0,1,2,3],[0,1,k,k+1]):
       for payload, batch in profiles:
        setup = time.perf_counter(); total_payload = payload*batch; size = total_payload//k
        host = rng.integers(0,256,(k,size),np.uint8)
        parity = np.empty((4,size),np.uint8); enc = np.array(generator(k,4)[k:],np.uint8); et = tables(enc)
        baseline.requant_isal_encode(k,4,size,et.ctypes.data,host.ctypes.data,parity.ctypes.data)
        available, coefficients = recovery(k,4,missing); coeff = np.array(coefficients,np.uint8); rt = tables(coeff)
        survivor = np.stack([host[i] if i<k else parity[i-k] for i in available])
        mapping = np.array([available.index(i) if i in available else -missing.index(i)-1 for i in range(k)],np.int32)
        hr = np.empty((4,size),np.uint8); hout = np.empty_like(host)
        def cpu(threads):
            assert recover(k,4,size,threads,rt.ctypes.data,mapping.ctypes.data,
                           survivor.ctypes.data,hr.ctypes.data,hout.ctypes.data) == 0
        cpu(1); assert np.array_equal(hout,host)
        data = cp.asarray(survivor); dc = cp.asarray(coeff); dm = cp.asarray(mapping)
        recovered = cp.empty((4,size),cp.uint8); restored = cp.empty_like(data)
        def pipeline():
            encode(((size+1023)//1024,1),(256,),
                   (data,dc,recovered,np.int32(k),np.int32(4),np.int32(size)))
            restore(((total_payload+255)//256,),(256,),
                    (data,recovered,dm,restored,np.int32(k),np.int32(size)))
        pipeline(); assert np.array_equal(cp.asnumpy(restored),host)
        capture = cp.cuda.Stream(non_blocking=True)
        with capture:
            capture.begin_capture(); pipeline(); graph = capture.end_capture()
        pin_in = cp.cuda.alloc_pinned_memory(total_payload); pin_out = cp.cuda.alloc_pinned_memory(total_payload)
        pi = np.frombuffer(pin_in,np.uint8).reshape(survivor.shape); po = np.frombuffer(pin_out,np.uint8).reshape(host.shape)
        np.copyto(pi,survivor); setup_ms = (time.perf_counter()-setup)*1000
        samples = {name:[] for name in ('cpu_1','cpu_2','cpu_4','cpu_6','gpu_resident','gpu_pageable','gpu_pinned','gpu_staged')}
        for iteration in range(warmups+repetitions):
            names = list(samples)
            if iteration%2: names.reverse()
            for name in names:
                cp.cuda.get_current_stream().synchronize(); start = time.perf_counter()
                if name.startswith('cpu_'): cpu(int(name.split('_')[1]))
                elif name=='gpu_resident': graph.launch(); cp.cuda.get_current_stream().synchronize()
                elif name=='gpu_pageable': data.set(survivor); graph.launch(); restored.get(out=hout)
                else:
                    if name=='gpu_staged': np.copyto(pi,survivor)
                    data.set(pi); graph.launch(); restored.get(out=po,blocking=False); cp.cuda.get_current_stream().synchronize()
                    if name=='gpu_staged': np.copyto(hout,po)
                elapsed = (time.perf_counter()-start)*1000
                actual = hout if name.startswith('cpu_') or name in ('gpu_pageable','gpu_staged') else po if name=='gpu_pinned' else cp.asnumpy(restored)
                assert np.array_equal(actual,host),(name,k,payload,batch,missing)
                if iteration>=warmups: samples[name].append(elapsed)
        cases.append(dict(k=k,parity=4,payload_bytes=payload,batch=batch,total_payload_bytes=total_payload,
                          missing=missing,available=available,input_sha256=hashlib.sha256(survivor.tobytes()).hexdigest(),
                          restored_sha256=hashlib.sha256(host.tobytes()).hexdigest(),every_restored_payload_exact=True,
                          setup_ms=setup_ms,samples_ms=samples,observed_device_pool_bytes=cp.get_default_memory_pool().total_bytes()))
        print(k,payload,batch,missing,{name:round(float(np.median(values)),4) for name,values in samples.items()},flush=True)
        del graph,data,dc,dm,recovered,restored,pi,po,pin_in,pin_out
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    paths = ['recovery_study.py','build_recovery.py','native/isal_recovery.cpp','native/isal_bridge.cpp','experiments/field.py','cuda/erasure.cu']
    build = Path('results/recovery-build.json')
    record = dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=warmups,samples=repetitions,cases=cases,
                  invalid_native_calls_rejected=5,
                  source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                  result_sha256={build.name:hashlib.sha256(build.read_bytes()).hexdigest()},
                  binary_sha256={p:hashlib.sha256((Path('build')/p).read_bytes()).hexdigest() for p in ('libisal_recovery.so','libisal_bridge.so','erasure.cubin')},
                  scope='Starts from already available k surviving shards; original encoding and loss simulation are outside all recovery timers. Recovery includes missing-shard arithmetic and complete original-payload assembly. Batch payloads are prepacked contiguously inside each shard; setup retains packing/encoding/pinning/plan costs. Direct CPU has no OpenMP team; multicore uses 256KiB chunks. GPU resident graph, pageable offload, reusable pinned buffers and caller staging are separate. Every timed payload is byte-exact. No distributed/network/energy claim.')
    Path(args.output).write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__': main()
