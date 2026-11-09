"""Test pinned/staged GPU offload against one through six ISA-L CPU threads."""
import argparse,ctypes,hashlib,json,os,pathlib,time
from datetime import datetime,timezone
os.environ['OMP_WAIT_POLICY']='PASSIVE'
os.environ['OMP_PROC_BIND']='TRUE'
os.environ['OMP_PLACES']='cores'
import cupy as cp
import numpy as np
from experiments.field import generator,recovery

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--output',default='results/transfer-study.json');args=parser.parse_args()
    library=ctypes.CDLL(str(pathlib.Path('build/libisal_parallel.so').resolve()));pointer=ctypes.c_void_p
    library.requant_isal_tables.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer]
    library.requant_isal_parallel.argtypes=[ctypes.c_int]*4+[pointer]*8
    library.requant_isal_tables.restype=None;library.requant_isal_parallel.restype=None
    module=cp.RawModule(path=str(pathlib.Path('build/erasure.cubin').resolve()))
    kernels={name:module.get_function(name) for name in ('encode_clmad4','gather','restore')}
    def tables(coeff):
        out=np.empty(coeff.shape[0]*coeff.shape[1]*32,np.uint8);library.requant_isal_tables(coeff.shape[1],coeff.shape[0],coeff.ctypes.data,out.ctypes.data);return out
    rng=np.random.default_rng(args.seed);cases=[]
    for k in (8,16):
      for payload in (65536,1048576,16777216,134217728,268435456):
        size=payload//k;host=rng.integers(0,256,(k,size),np.uint8);missing=[0,1,2,3]
        available,coeff=recovery(k,4,missing);enc=np.array(generator(k,4)[k:],np.uint8);rec=np.array(coeff,np.uint8)
        et,rt=tables(enc),tables(rec);indices=np.array(available,np.int32);lost=np.array(missing,np.int32)
        hp=np.empty((4,size),np.uint8);hr=np.empty_like(hp);hout=np.empty_like(host)
        def cpu(threads):library.requant_isal_parallel(k,4,size,threads,et.ctypes.data,rt.ctypes.data,indices.ctypes.data,lost.ctypes.data,host.ctypes.data,hp.ctypes.data,hr.ctypes.data,hout.ctypes.data)
        cpu(1);assert np.array_equal(host,hout)
        data=cp.asarray(host);parity=cp.empty((4,size),cp.uint8);survivors=cp.empty_like(data);recovered=cp.empty_like(parity);restored=cp.empty_like(data)
        de,dr,di=cp.asarray(enc),cp.asarray(rec),cp.asarray(indices)
        mapping=np.array([available.index(i) if i in available else -missing.index(i)-1 for i in range(k)],np.int32);dm=cp.asarray(mapping)
        numbers=tuple(np.int32(x) for x in (k,4,size));grid=((size+1023)//1024,1)
        def pipeline():
            kernels['encode_clmad4'](grid,(256,),(data,de,parity,*numbers))
            kernels['gather'](((payload+255)//256,),(256,),(data,parity,di,survivors,np.int32(k),np.int32(size)))
            kernels['encode_clmad4'](grid,(256,),(survivors,dr,recovered,*numbers))
            kernels['restore'](((payload+255)//256,),(256,),(survivors,recovered,dm,restored,np.int32(k),np.int32(size)))
        pipeline();cp.cuda.get_current_stream().synchronize();assert np.array_equal(cp.asnumpy(restored),host)
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:stream.begin_capture();pipeline();graph=stream.end_capture()
        start=time.perf_counter();pin_in=cp.cuda.alloc_pinned_memory(payload);pin_out=cp.cuda.alloc_pinned_memory(payload)
        input_array=np.frombuffer(pin_in,np.uint8).reshape(host.shape);output_array=np.frombuffer(pin_out,np.uint8).reshape(host.shape)
        np.copyto(input_array,host);pinned_preparation_ms=(time.perf_counter()-start)*1000
        samples={name:[] for name in ('cpu_1','cpu_2','cpu_4','cpu_6','gpu_resident','gpu_pageable','gpu_pinned','gpu_staged')}
        for iteration in range(35):
            names=list(samples)
            if iteration%2:names.reverse()
            for name in names:
                cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
                if name.startswith('cpu_'):cpu(int(name.split('_')[1]))
                elif name=='gpu_resident':graph.launch();cp.cuda.get_current_stream().synchronize()
                elif name=='gpu_pageable':data.set(host);graph.launch();restored.get(out=hout)
                else:
                    if name=='gpu_staged':np.copyto(input_array,host)
                    data.set(input_array);graph.launch();restored.get(out=output_array,blocking=False);cp.cuda.get_current_stream().synchronize()
                    if name=='gpu_staged':np.copyto(hout,output_array)
                wall=(time.perf_counter()-start)*1000
                actual=hout if name.startswith('cpu_') or name in ('gpu_pageable','gpu_staged') else output_array if name=='gpu_pinned' else cp.asnumpy(restored)
                assert np.array_equal(actual,host),(name,k,payload)
                if iteration>=5:samples[name].append(wall)
        cases.append(dict(k=k,parity=4,payload_bytes=payload,missing=missing,every_restored_payload_exact=True,
            input_sha256=hashlib.sha256(host.tobytes()).hexdigest(),pinned_allocation_and_initial_copy_ms=pinned_preparation_ms,
            samples_ms=samples,observed_device_pool_bytes=cp.get_default_memory_pool().total_bytes()))
        print(k,payload,{name:round(float(np.median(v)),4) for name,v in samples.items()},flush=True)
        del graph,pipeline,data,parity,survivors,recovered,restored,de,dr,di,dm,input_array,output_array,pin_in,pin_out
        cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    sources=['transfer_study.py','experiments/field.py','cuda/erasure.cu','native/isal_bridge.cpp','native/isal_parallel.cpp']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,cases=cases,
        source_sha256={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},
        binary_sha256=hashlib.sha256(pathlib.Path('build/libisal_parallel.so').read_bytes()).hexdigest(),
        scope='Complete encode/gather/four-data-shard recovery/original-payload assembly. ISA-L threads 1/2/4/6 use 256 KiB per-shard work chunks and passive OpenMP waiting; matrices and buffers preallocated. GPU CLMAD graph uses the unchanged independently validated kernel. Pageable path includes H2D/D2H. Pinned path assumes caller-owned pinned inputs/outputs, excludes initial pinned allocation/copy. Staged path includes ordinary-host to pinned copy and pinned-output to ordinary-host copy. No universal crossover is assumed; patterns other than four missing data shards are covered by the primary campaign.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
