"""Exact GPU encoding/recovery and a pinned optimized ISA-L CPU baseline."""
import argparse,ctypes,hashlib,json,pathlib,subprocess,time
from datetime import datetime,timezone
import cupy as cp
import numpy as np
from experiments.field import encode,generator,multiply,recovery

REVISION='ab01645a2d9ab581477c703103fe3450c4696845'

def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream',type=pathlib.Path);parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--output',default='results/gpu-study.json');args=parser.parse_args()
    assert subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()==REVISION
    library=ctypes.CDLL(str(pathlib.Path('build/libisal_bridge.so').resolve()))
    pointer=ctypes.c_void_p
    library.requant_isal_tables.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer]
    library.requant_isal_encode.argtypes=[ctypes.c_int]*3+[pointer]*3
    library.requant_isal_pipeline.argtypes=[ctypes.c_int]*3+[pointer]*8
    for name in ('requant_isal_tables','requant_isal_encode','requant_isal_pipeline'):getattr(library,name).restype=None
    module=cp.RawModule(path=str(pathlib.Path('build/erasure.cubin').resolve()))
    kernels={name:module.get_function(name) for name in ('validate_product','encode_serial','encode_table','encode_nibble4','encode_clmad4','gather','restore')}
    table=np.array([[multiply(a,b) for b in range(256)] for a in range(256)],np.uint8);dtable=cp.asarray(table)
    products=cp.empty(65536,cp.uint8);kernels['validate_product']((256,),(256,),(products,))
    assert np.array_equal(cp.asnumpy(products),table.reshape(-1))
    def cpu_tables(coeff):
        rows,k=coeff.shape;tables=np.empty(rows*k*32,np.uint8)
        library.requant_isal_tables(k,rows,coeff.ctypes.data,tables.ctypes.data);return tables
    def cpu_encode(data,coeff,tables=None,out=None):
        rows,k=coeff.shape;size=data.shape[1]
        if tables is None:tables=cpu_tables(coeff)
        if out is None:out=np.empty((rows,size),np.uint8)
        library.requant_isal_encode(k,rows,size,tables.ctypes.data,data.ctypes.data,out.ctypes.data);return out
    def plan(mode,data,coeff,out):
        rows,k=coeff.shape;size=data.shape[1];dc=cp.asarray(coeff)
        nibble=np.concatenate([table[coeff][:,:,:16],table[coeff][:,:,np.arange(16)*16]],axis=2)
        dn=cp.asarray(nibble);numbers=tuple(np.int32(x) for x in (k,rows,size))
        if mode=='serial':call=lambda:kernels['encode_serial'](((size+255)//256,rows),(256,),(data,dc,out,*numbers))
        elif mode=='table':call=lambda:kernels['encode_table'](((size+255)//256,rows),(256,),(data,dc,dtable,out,*numbers))
        elif mode=='nibble4':call=lambda:kernels['encode_nibble4'](((size+255)//256,(rows+3)//4),(256,),(data,dn,out,*numbers))
        elif mode=='clmad4':call=lambda:kernels['encode_clmad4'](((size+1023)//1024,(rows+3)//4),(256,),(data,dc,out,*numbers))
        else:raise ValueError(mode)
        call();cp.cuda.get_current_stream().synchronize();return call,(dc,dn)
    def capture(call):
        stream=cp.cuda.Stream(non_blocking=True)
        with stream:stream.begin_capture();call();graph=stream.end_capture()
        return graph
    rng=np.random.default_rng(args.seed);validation=[]
    for size in (1,3,4,17,513):
        for kind in ('zero','maximum','random'):
            host=np.zeros((8,size),np.uint8) if kind=='zero' else np.full((8,size),255,np.uint8) if kind=='maximum' else rng.integers(0,256,(8,size),np.uint8)
            coeff=np.array(generator(8,4)[8:],np.uint8);expected=np.array([list(x) for x in encode(coeff.tolist(),[bytes(x) for x in host])],np.uint8)
            assert np.array_equal(cpu_encode(host,coeff),expected)
            data=cp.asarray(host);out=cp.empty((4,size),cp.uint8)
            for mode in ('serial','table','nibble4','clmad4'):
                call,keep=plan(mode,data,coeff,out);assert np.array_equal(cp.asnumpy(out),expected),(size,kind,mode)
            validation.append(dict(size=size,pattern=kind,all_variants_exact=True))
    cases=[]
    for k in (8,16):
      parity=4;g=generator(k,parity);coeff=np.array(g[k:],np.uint8);encode_tables=cpu_tables(coeff)
      for payload in (65536,1048576,16777216,134217728,268435456):
       size=payload//k;host=rng.integers(0,256,(k,size),np.uint8);expected_parity=cpu_encode(host,coeff,encode_tables)
       for missing in ([0,1,2,3],[0,1,k,k+1]):
        available,recovery_coeff=recovery(k,parity,missing);rc=np.array(recovery_coeff,np.uint8);recovery_tables=cpu_tables(rc)
        available_array=np.array(available,np.int32);missing_array=np.array(missing,np.int32)
        mapping=np.array([available.index(i) if i in available else -missing.index(i)-1 for i in range(k)],np.int32)
        data=cp.asarray(host);encoded=cp.empty((4,size),cp.uint8);survivors=cp.empty((k,size),cp.uint8)
        recovered=cp.empty((4,size),cp.uint8);restored=cp.empty_like(data)
        di,dm=cp.asarray(available_array),cp.asarray(mapping)
        host_parity=np.empty_like(expected_parity);host_recovery=np.empty_like(expected_parity);host_restored=np.empty_like(host)
        def cpu_pipeline():
            library.requant_isal_pipeline(k,4,size,encode_tables.ctypes.data,recovery_tables.ctypes.data,available_array.ctypes.data,missing_array.ctypes.data,host.ctypes.data,host_parity.ctypes.data,host_recovery.ctypes.data,host_restored.ctypes.data)
        cpu_pipeline();assert np.array_equal(host_restored,host)
        plans={};keeps=[]
        for mode in ('serial','table','nibble4','clmad4'):
            enc,keep=plan(mode,data,coeff,encoded);repair,keep2=plan(mode,survivors,rc,recovered);keeps.extend([keep,keep2])
            def pipeline(enc=enc,repair=repair):
                enc();kernels['gather'](((payload+255)//256,),(256,),(data,encoded,di,survivors,np.int32(k),np.int32(size)))
                repair();kernels['restore'](((payload+255)//256,),(256,),(survivors,recovered,dm,restored,np.int32(k),np.int32(size)))
            enc_graph=capture(enc);pipeline_graph=capture(pipeline);pipeline_graph.launch()
            assert np.array_equal(cp.asnumpy(encoded),expected_parity)
            assert np.array_equal(cp.asnumpy(restored),host)
            plans[mode]=(enc_graph,pipeline_graph)
        samples={mode:{scope:[] for scope in ('encode_device_ms','encode_wall_ms','recovery_pipeline_device_ms','recovery_pipeline_wall_ms','transfer_pipeline_wall_ms')} for mode in plans}
        samples['isa_l_cpu']={scope:[] for scope in ('encode_wall_ms','recovery_pipeline_wall_ms')}
        begin,end=cp.cuda.Event(),cp.cuda.Event()
        for iteration in range(35):
            names=list(samples)
            if iteration%2:names.reverse()
            for mode in names:
                if mode=='isa_l_cpu':
                    start=time.perf_counter();cpu_encode(host,coeff,encode_tables,host_parity);a=(time.perf_counter()-start)*1000
                    start=time.perf_counter();cpu_pipeline();b=(time.perf_counter()-start)*1000
                    assert np.array_equal(host_parity,expected_parity) and np.array_equal(host_restored,host)
                    if iteration>=5:samples[mode]['encode_wall_ms'].append(a);samples[mode]['recovery_pipeline_wall_ms'].append(b)
                    continue
                encode_graph,pipeline_graph=plans[mode];values=[]
                for graph in (encode_graph,pipeline_graph):
                    begin.record();graph.launch();end.record();end.synchronize();device=cp.cuda.get_elapsed_time(begin,end)
                    start=time.perf_counter();graph.launch();cp.cuda.get_current_stream().synchronize();wall=(time.perf_counter()-start)*1000
                    values.extend([device,wall])
                assert np.array_equal(cp.asnumpy(encoded),expected_parity) and np.array_equal(cp.asnumpy(restored),host)
                start=time.perf_counter();data.set(host);pipeline_graph.launch();restored.get(out=host_restored);transfer=(time.perf_counter()-start)*1000
                assert np.array_equal(host_restored,host)
                if iteration>=5:
                    for scope,value in zip(samples[mode],values+[transfer]):samples[mode][scope].append(value)
        cases.append(dict(k=k,parity=4,payload_bytes=payload,shard_bytes=size,missing=missing,
            input_sha256=hashlib.sha256(host.tobytes()).hexdigest(),parity_sha256=hashlib.sha256(expected_parity.tobytes()).hexdigest(),
            every_output_exact=True,device_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),samples=samples))
        print(k,payload,missing,{mode:float(np.median(scopes['recovery_pipeline_wall_ms'])) for mode,scopes in samples.items()},flush=True)
        del plans,keeps,enc_graph,pipeline_graph,encode_graph,pipeline,enc,repair,keep,keep2,data,encoded,survivors,recovered,restored,di,dm
        cp.get_default_memory_pool().free_all_blocks()
    properties=cp.cuda.runtime.getDeviceProperties(0)
    sources=['gpu_study.py','experiments/field.py','cuda/erasure.cu','native/isal_bridge.cpp']
    record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,upstream_revision=REVISION,
        source_sha256={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in sources},
        binary_sha256={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in ('build/erasure.cubin','build/libisal_bridge.so')},
        device=properties['name'].decode() if isinstance(properties['name'],bytes) else properties['name'],polynomial='0x11d',validation=validation,cases=cases,
        scope='Exact GF(256) systematic Cauchy code. Recovery pipeline includes encode, gather surviving shards, reconstruct all missing shards and assemble original payload. Resident graphs exclude plan, coefficients, allocation, transfers and verification. Transfer path adds original-payload H2D and restored-payload D2H; CPU path uses preallocated host buffers and optimized ISA-L. Independent reference checks edge fixtures; ISA-L checks every large GPU output. Repetitions reuse each payload. No storage/DA protocol or profitability claim.')
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
