# GPU erasure coding: final research report

A research project by [Requant Network](https://requant.network). Research date: 2026-11-09. Raw timestamps use UTC.

## Finding

On an RTX 5060 Ti 16 GB, an original packed CUDA 13.3 carryless-multiply implementation correctly performs systematic Reed-Solomon encoding and complete recovery. For a 16+4 code and a 256 MiB original payload, the primary cohort's resident pipeline median is 9.745 ms, versus 11.231 ms for a packed nibble-table kernel, 12.022 ms for a byte-table kernel and 19.620 ms for bit-serial arithmetic. Every large output is compared byte-for-byte with pinned ISA-L; every restored payload equals the original.

This kernel improvement does **not** make ordinary host-to-GPU offload the fastest full pipeline. After adding an optimized six-thread ISA-L baseline, the 16+4, 256 MiB pipeline takes 48.860/52.310 ms on CPU in the two follow-up cohorts, versus 87.419/87.465 ms including pageable H2D/D2H. Pinned buffers barely change the large-payload result. Adding ordinary-to-pinned staging copies makes it worse. GPU-resident workflows remain promising; a stand-alone host storage pipeline should use the CPU under the tested conditions.

These are storage-code measurements, not a deployed data-availability network, a proof-of-work protocol or a profitability study. No universal offload crossover is claimed.

## Mathematical contract

The field is GF(2^8) with modulus x^8+x^4+x^3+x^2+1 (`0x11d`). Addition is XOR. The generator consists of an identity matrix for the k data shards and four Cauchy parity rows, with entry `1/(i XOR j)` for parity index i in k..k+3 and data index j in 0..k-1. The two index sets are disjoint; the Cauchy construction gives a systematic MDS code. Recovery selects k surviving rows, inverts the survivor matrix in the same field and reconstructs every missing shard.

[experiments/field.py](experiments/field.py) contains independent polynomial multiplication, exponentiation/inversion, Gaussian elimination, encoding and recovery. [cpu_study.py](cpu_study.py) checks all 65,536 multiplication pairs against a separate polynomial oracle, all 255 nonzero inverses, 616 erasure patterns, and rejection of a singular matrix. Small cases are exhaustive; larger erasure spaces use retained deterministic random samples. This is not an exhaustive check of every large-code erasure pattern.

## Implementations and validation

[cuda/erasure.cu](cuda/erasure.cu) implements bit-serial arithmetic, a 256-entry multiplication lookup, a packed four-output nibble lookup, and packed carryless multiplication. The `clmad.lo.u64` variant places four independent byte symbols into 16-bit lanes, XOR-accumulates unreduced products and reduces with the field polynomial. Odd shard lengths use a byte fallback, avoiding assumptions about aligned four-byte tails. This uses integer polynomial arithmetic; tensor cores and floating-point tolerances play no role.

The carryless approach was motivated by NVIDIA's [CUDA 13.3 CLMAD description](https://developer.nvidia.com/blog/?p=119649). Our implementations and all measurements here are independent. A carryless multiply instruction still requires correct packing, reduction, memory movement and full-pipeline verification.

Before each main cohort, all 65,536 GPU multiplication results match the CPU field table. Fifteen zero/maximum/random fixtures at shard lengths 1, 3, 4, 17 and 513 match the independent Python encoder and ISA-L in every kernel mode. The two primary cohorts then cover 40 full cases: 8+4 and 16+4, five payload sizes, and either four missing data shards or two missing data/two missing parity shards. All encoded parity bytes and all restored original bytes are checked at runtime, outside timed regions.

ISA-L is pinned to [revision ab01645a2d9ab581477c703103fe3450c4696845](https://github.com/intel/isa-l/tree/ab01645a2d9ab581477c703103fe3450c4696845). Its source remains unmodified. Original C++ wrappers orchestrate encoding, survivor gathering, reconstruction and payload assembly. A second wrapper divides each shard into 256 KiB chunks using OpenMP with 1, 2, 4 or 6 threads. Passive waiting and core binding are explicit. This is a tested multicore baseline, not a claim that every possible ISA-L integration has been optimized.

## Experimental design

The primary seeds are 20261109 and 20261110. Each cohort covers payloads 64 KiB, 1 MiB, 16 MiB, 128 MiB and 256 MiB; payload means original data bytes, excluding parity. Each case uses five warm-ups and thirty measured repetitions, with alternating implementation order. Repetitions reuse one payload; they are latency observations, not thirty independent data seeds.

The transfer/multicore follow-up uses seeds 20261111 and 20261112 and four missing data shards, with ten cases per cohort. It compares all CPU thread counts and resident/pageable/pinned/staged GPU paths. It retains pinned-buffer preparation time separately. The original two-erasure-pattern comparison remains in the primary campaign.

OpenMP team overhead can weaken the small-payload comparator. We therefore added a direct, non-OpenMP ISA-L follow-up on the **exact same payloads**, regenerated and checked by input SHA-256. It also uses five warm-ups/thirty measurements. It runs in a separate timing window; exact paired speedups across these windows are not claimed. For 16+4, 1 MiB, direct CPU medians are 0.290/0.290 ms, faster than pinned offload at 0.362/0.363 ms. A weak OpenMP-only comparison would obscure this result.

## Timing and memory boundaries

The full pipeline encodes parity, gathers k surviving shards, reconstructs four missing shards and assembles the original payload. Matrices, ISA-L lookup plans and buffers are preallocated. Resident GPU graphs exclude allocation, plan construction, transfers and verification. Primary device-event intervals and synchronized host wall intervals are retained separately; they are not interchangeable.

Pageable offload adds original-payload H2D and restored-payload D2H. The pinned mode assumes caller-owned pinned input/output and excludes their initial allocation/copy. Staged mode additionally includes ordinary-host to pinned-input and pinned-output to ordinary-host copies. Verification runs after timing. Recovery here includes encoding for a complete controlled loss simulation; a recovery-only application may have different costs.

The largest observed CuPy reserved pool is 1,073,893,888 bytes. These are pool observations after a case, not an instrumented global allocation peak and not host/pinned memory usage. Raw records retain the observations per case. No clock or power limits were changed; no other research GPU benchmark was intentionally overlapped. Dynamic clocks and ordinary OS activity remain confounders. No energy or profitability claim is made.

## Results

Full pipeline medians in ms, four missing data shards. Each cell is primary/replication from the transfer follow-up:

| 16+4 original payload | CPU, six threads | GPU resident | Pageable offload | Pinned offload | Staged offload |
| --- | --- | --- | --- | --- | --- |
| 128 MiB | 24.500 / 25.569 | 4.897 / 4.910 | 43.850 / 43.877 | 43.542 / 43.560 | 76.045 / 76.532 |
| 256 MiB | 48.860 / 52.310 | 9.735 / 9.761 | 87.419 / 87.465 | 86.981 / 87.017 | 153.678 / 155.706 |

At 8+4 and 16+4, CPU multicore also wins against large-payload offload at 16 MiB. The small-payload direct baseline prevents claiming a crossover merely because OpenMP setup is expensive. Across the sampled sizes, we find no consistent region where transfer-inclusive GPU coding beats the strongest appropriate tested CPU path. A crossover between sampled points or under a different interconnect/workload is not excluded.

The initial single-thread-only comparison suggested a possible win around 128 MiB. The multicore follow-up changes that decision. Conversely, the GPU's resident advantage survives both payload cohorts. Packing and transfer policy must therefore be evaluated in the application's actual memory-residency model.

[summary.csv](results/summary.csv) contains 1,060 rows of separate cohort/case/mode/scope medians and nearest-rank p95 values. A p95 from thirty repetitions is descriptive, not a tail-latency guarantee. Do not average unlike timing boundaries. [RESULTS.md](RESULTS.md) maps each raw file to its role.

## Reproduction

CPU-only checks use Python 3.12 standard library and run without `-O`:

```sh
python cpu_study.py --output build/ci.json
python verify_records.py
python verify_results.py
python summarize_results.py
git diff --exit-code -- results/summary.csv
```

GPU runs use Linux x86_64, RTX 5060 Ti, CuPy 13.6.0, NumPy 2.2.6, CUDA 12.x runtime/headers and CUDA 13.3.73 nvcc. See [environment.json](results/environment.json) and [build.json](results/build.json). Install NASM, CMake and a C/C++ compiler, then build the exact upstream:

```sh
git clone https://github.com/intel/isa-l.git upstream-isal
git -C upstream-isal checkout ab01645a2d9ab581477c703103fe3450c4696845
cmake -S upstream-isal -B build/isal -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DISAL_BUILD_TESTS=OFF -DISAL_BUILD_PERF_TESTS=OFF -DISAL_BUILD_ISAL_SHIM=OFF -DISAL_BUILD_FUZZ_TESTS=OFF -DISAL_BUILD_IGZIP_CLI=OFF
cmake --build build/isal -j6
g++ -O3 -shared -fPIC native/isal_bridge.cpp -Iupstream-isal/include -Lbuild/isal -Wl,-rpath,$PWD/build/isal -lisal -o build/libisal_bridge.so
g++ -O3 -shared -fPIC -fopenmp native/isal_bridge.cpp native/isal_parallel.cpp -Iupstream-isal/include -Lbuild/isal -Wl,-rpath,$PWD/build/isal -lisal -o build/libisal_parallel.so
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gpu.txt
/path/to/cuda-13.3/bin/nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/erasure.cu -o build/erasure.cubin
export CUDA_PATH=/path/to/cuda-12.8
export PYTHON=python
export CUDA_VISIBLE_DEVICES=0
bash run_gpu.sh upstream-isal
bash run_gpu.sh upstream-isal --seed 20261110 --output results/gpu-replication.json
STUDY=transfer_study.py bash run_gpu.sh
STUDY=transfer_study.py bash run_gpu.sh --seed 20261112 --output results/transfer-replication.json
python direct_cpu_study.py
python direct_cpu_study.py --reference results/transfer-replication.json --output results/direct-cpu-replication.json
python record_environment.py
```

Run timing campaigns sequentially. The post-run provenance collector uses the measurement workspace's sibling ISA-L/build directories; adjust those two paths for another layout. Binary hashes bind our measured builds and do not promise byte-identical rebuilds. Source and raw-record hashes are checked in CPU CI. CI independently reruns field/recovery controls and validates retained records/tables; it does not rerun GPU kernels or reconstruct large outputs from hashes alone. All large byte comparisons are runtime evidence, explicitly distinguished from retained independently recheckable algebraic certificates.

## Decisions and limits

The carryless kernel hypothesis is supported on this device and workload. The unconditional offload-speedup hypothesis is rejected over the sampled grid. Caller-owned pinned memory helps small transfers but does not repair the large transfer bottleneck; staging adds real cost. A GPU-resident application, batching with useful neighboring GPU computation, and recovery-only workloads are worthwhile separate studies. Network coding matrices, adversarial erasure models, distributed traffic, multi-GPU scaling, other field sizes and energy economics remain untested. This bounded campaign is complete without claiming those results.
