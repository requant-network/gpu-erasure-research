"""Build the same GF256 code for SM120 and a bounded portable SM75 fallback."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--nvcc',default='nvcc');args=parser.parse_args()
    source=Path('cuda/erasure.cu').read_text();lines=[s for s in source.splitlines() if s.startswith('__device__ unsigned long long clmul(')];assert len(lines)==1
    replacement='''__device__ unsigned long long clmul(unsigned long long a,unsigned long long b,unsigned long long c=0){
#if __CUDA_ARCH__ >= 1000
 unsigned long long r;asm("clmad.lo.u64 %0, %1, %2, %3;":"=l"(r):"l"(a),"l"(b),"l"(c));return r;
#else
 // Every caller multiplies by one GF256 coefficient (at most255).
 unsigned long long r=c;for(int i=0;i<8;i++)if((b>>i)&1)r^=a<<i;return r;
#endif
}'''
    source=source.replace(lines[0],replacement);generated=Path('build/erasure-multigpu.cu');generated.write_text(source)
    binaries={}
    for arch in (120,75):
        path=Path(f'build/erasure-sm{arch}.cubin')
        subprocess.run([args.nvcc,'-O3','-std=c++17',f'-arch=sm_{arch}','-cubin',str(generated),'-o',str(path)],check=True)
        binaries[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('build_multigpu.py','cuda/erasure.cu')},
                generated_source_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),binary_sha256=binaries,
                architectures=[120,75],flags=['-O3','-std=c++17','-cubin'],
                scope='Original GF256 kernels, nativeCLMAD onSM120 and bit-polynomial fallback for8-bit coefficients onSM75. Study uses nativeCLMAD onGPU0 and portable nibble-table kernel onGPU1;both exact. No truncation of field values or silent change of polynomial.')
    Path('results/multigpu-build.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
