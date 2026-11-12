"""Complete heterogeneous two-GPU recovery, including host packing and assembly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.update(OMP_WAIT_POLICY='PASSIVE',OMP_PROC_BIND='TRUE',OMP_PLACES='cores')
import cupy as cp
import numpy as np
from experiments.field import generator,multiply,recovery


class Plan:
    def __init__(self,device,start,end,survivor,coeff,tables,mapping,module):
        self.device,self.start,self.end=device,start,end;self.size=end-start;self.k=len(survivor)
        with cp.cuda.Device(device):
            self.stream=cp.cuda.Stream(non_blocking=True)
            self.pin_in=cp.cuda.alloc_pinned_memory(self.k*self.size);self.pin_out=cp.cuda.alloc_pinned_memory(self.k*self.size)
            self.pi=np.frombuffer(self.pin_in,np.uint8,count=self.k*self.size).reshape(self.k,self.size)
            self.po=np.frombuffer(self.pin_out,np.uint8,count=self.k*self.size).reshape(self.k,self.size)
            np.copyto(self.pi,survivor[:,start:end]);self.data=cp.asarray(self.pi)
            self.coeff=cp.asarray(coeff if device==0 else tables);self.mapping=cp.asarray(mapping)
            self.recovered=cp.empty((4,self.size),cp.uint8);self.out=cp.empty_like(self.data)
            self.encode=module.get_function('encode_clmad4' if device==0 else 'encode_nibble4');self.restore=module.get_function('restore')
            cp.cuda.get_current_stream().synchronize()
            self.launch(self.stream);self.stream.synchronize()
            self.stream.begin_capture();self.launch(self.stream);self.graph=self.stream.end_capture()
    def launch(self,stream):
        step=1024 if self.device==0 else 256
        self.encode(((self.size+step-1)//step,1),(256,),
                    (self.data,self.coeff,self.recovered,np.int32(self.k),np.int32(4),np.int32(self.size)),stream=stream)
        self.restore(((self.k*self.size+255)//256,),(256,),
                     (self.data,self.recovered,self.mapping,self.out,np.int32(self.k),np.int32(self.size)),stream=stream)
    def clear(self):
        with cp.cuda.Device(self.device):
            cp.cuda.runtime.memsetAsync(self.out.data.ptr,0xa5,self.out.nbytes,self.stream.ptr);self.stream.synchronize()
    def run(self,transfer):
        with cp.cuda.Device(self.device):
            if transfer:self.data.set(self.pi,stream=self.stream)
            self.graph.launch(self.stream)
            if transfer:self.out.get(out=self.po,stream=self.stream,blocking=False)
            self.stream.synchronize()
    def check(self,host,transfer):
        with cp.cuda.Device(self.device):actual=self.po if transfer else self.out.get(stream=self.stream)
        assert np.array_equal(actual,host[:,self.start:self.end])


def main():
    assert __debug__
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=20261113)
    parser.add_argument('--output',default='results/multigpu-study.json');args=parser.parse_args()
    assert cp.cuda.runtime.getDeviceCount()==2
    build_path=Path('results/multigpu-build.json');build=json.loads(build_path.read_text());modules={};hardware=[]
    table=np.array([[multiply(a,b) for b in range(256)] for a in range(256)],np.uint8)
    for device,arch in ((0,120),(1,75)):
        with cp.cuda.Device(device):
            properties=cp.cuda.runtime.getDeviceProperties(device);assert properties['major']*10+properties['minor']==arch
            path=Path(f'build/erasure-sm{arch}.cubin');assert hashlib.sha256(path.read_bytes()).hexdigest()==build['binary_sha256'][path.name]
            module=cp.RawModule(path=str(path.resolve()));modules[device]=module
            out=cp.empty(65536,cp.uint8);module.get_function('validate_product')((256,),(256,),(out,))
            assert np.array_equal(cp.asnumpy(out),table.reshape(-1))
            hardware.append(dict(device=device,name=properties['name'].decode(),major=properties['major'],minor=properties['minor'],total_memory=properties['totalGlobalMem']))
    pointer=ctypes.c_void_p;baseline=ctypes.CDLL(str(Path('build/libisal_bridge.so').resolve()))
    baseline.requant_isal_tables.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer];baseline.requant_isal_tables.restype=None
    baseline.requant_isal_encode.argtypes=[ctypes.c_int]*3+[pointer]*3;baseline.requant_isal_encode.restype=None
    native=ctypes.CDLL(str(Path('build/libisal_recovery.so').resolve())).requant_isal_recover
    native.argtypes=[ctypes.c_int]*4+[pointer]*5;native.restype=ctypes.c_int
    def tables(coeff):
        out=np.empty(coeff.size*32,np.uint8);baseline.requant_isal_tables(coeff.shape[1],coeff.shape[0],coeff.ctypes.data,out.ctypes.data);return out
    profiles=[(k,size,missing) for k in (8,16) for size in (16<<20,256<<20) for missing in ([0,1,2,3],[0,1,k,k+1])]+[(8,1<<30,[0,1,2,3])]
    rng=np.random.default_rng(args.seed);cases=[]
    with ThreadPoolExecutor(max_workers=2) as executor:
      def together(plans,method,*arguments):
        tasks=[executor.submit(getattr(plan,method),*arguments) for plan in plans]
        for task in tasks:task.result()
      for k,payload,missing in profiles:
        live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
        assert all(int(v.strip())==os.getpid() for v in live),'Unrelated GPU process must not share the timing interval'
        begin=time.perf_counter();size=payload//k;host=rng.integers(0,256,(k,size),np.uint8)
        parity=np.empty((4,size),np.uint8);encoding=np.array(generator(k,4)[k:],np.uint8);et=tables(encoding)
        baseline.requant_isal_encode(k,4,size,et.ctypes.data,host.ctypes.data,parity.ctypes.data)
        available,coefficients=recovery(k,4,missing);coeff=np.array(coefficients,np.uint8);rt=tables(coeff)
        survivor=np.stack([host[i] if i<k else parity[i-k] for i in available]);hr=np.empty((4,size),np.uint8);hout=np.empty_like(host)
        mapping=np.array([available.index(i) if i in available else -missing.index(i)-1 for i in range(k)],np.int32)
        policies={}
        for name,fraction in (('gpu0',1),('gpu1',0),('dual50',.5),('dual75',.75),('dual875',.875)):
            split=int(size*fraction);parts=[]
            if split:parts.append(Plan(0,0,split,survivor,coeff,rt,mapping,modules[0]))
            if split<size:parts.append(Plan(1,split,size,survivor,coeff,rt,mapping,modules[1]))
            policies[name]=parts
        setup_ms=(time.perf_counter()-begin)*1000
        samples={f'cpu_{threads}':{'complete_wall_ms':[]} for threads in (1,2,4,6)}
        samples.update({name:{'resident_wall_ms':[],'staged_wall_ms':[]} for name in policies})
        for iteration in range(35):
            names=list(samples)
            if iteration%2:names.reverse()
            for name in names:
                if name.startswith('cpu_'):
                    start=time.perf_counter();status=native(k,4,size,int(name[4:]),rt.ctypes.data,mapping.ctypes.data,survivor.ctypes.data,hr.ctypes.data,hout.ctypes.data)
                    elapsed=(time.perf_counter()-start)*1000;assert status==0 and np.array_equal(hout,host)
                    if iteration>=5:samples[name]['complete_wall_ms'].append(elapsed)
                    continue
                plans=policies[name]
                for transfer,scope in ((False,'resident_wall_ms'),(True,'staged_wall_ms')):
                    together(plans,'clear');start=time.perf_counter()
                    if transfer:
                        for plan in plans:np.copyto(plan.pi,survivor[:,plan.start:plan.end])
                    together(plans,'run',transfer)
                    if transfer:
                        for plan in plans:np.copyto(hout[:,plan.start:plan.end],plan.po)
                    elapsed=(time.perf_counter()-start)*1000
                    if transfer:assert np.array_equal(hout,host)
                    together(plans,'check',host,transfer)
                    if iteration>=5:samples[name][scope].append(elapsed)
        cases.append(dict(k=k,payload_bytes=payload,missing=missing,available=available,setup_ms=setup_ms,
                          input_sha256=hashlib.sha256(survivor.tobytes()).hexdigest(),output_sha256=hashlib.sha256(host.tobytes()).hexdigest(),
                          every_restored_payload_exact=True,output_cleared_before_every_launch=True,samples=samples,
                          partition_bytes={name:[p.k*p.size for p in plans] for name,plans in policies.items()}))
        record=dict(utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,warmups=5,samples=30,hardware=hardware,
                    field_products_checked_per_gpu=65536,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ('multigpu_study.py','build_multigpu.py','cuda/erasure.cu','native/isal_recovery.cpp','native/isal_bridge.cpp','experiments/field.py')},
                    result_sha256={build_path.name:hashlib.sha256(build_path.read_bytes()).hexdigest()},cases=cases,
                    scope='Offline recovery from surviving shards, exact GF256(0x11d). GPU0 nativeCLMAD, GPU1 portable nibble-table backend;not FP4/CLMAD on unsupported hardware. Independent byte-column partitions;dual50/75/87.5percent assigned toGPU0;CPU futures/launch/join overhead timed. Resident excludes uploads/downloads/packing, each partition validated separately after timer. Staged full wall includes packing ordinary buffers into reusable pinned partition buffers, both uploads, graphs, both downloads and assembly into full ordinary payload. No peer copies, no distributed/network scope. Every output cleared outside each timer, all bytes independently checked;CPU ISA-L direct/multicore same inputs. Services using GPU1 temporarily stopped under explicit user authorization;both GPUs exclusively used by this campaign. No two-GPU result is inferred for other algorithms.')
        Path(args.output).write_text(json.dumps(record,indent=2)+'\n')
        print(args.seed,k,payload,missing,{name:round(float(np.median(v.get('resident_wall_ms',v.get('complete_wall_ms')))),3) for name,v in samples.items()},flush=True)
        del policies,plans,parts,plan
        for device in (0,1):
            with cp.cuda.Device(device):cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

if __name__=='__main__':main()
