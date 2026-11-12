# GPU Erasure Research

Exact Reed-Solomon encoding and recovery on consumer GPUs, with optimized CPU and transfer-inclusive baselines.

A research project by [Requant Network](https://requant.network).

**Research campaign completed: CPU references, replicated GPU/multicore experiments and final conclusions published.**

CUDA 13.3 packed carryless arithmetic improves the tested resident pipeline: at 16+4 shards and 256 MiB of original data, the primary median is **9.745 ms**, versus 11.231 ms for packed nibble tables and 19.620 ms for bit-serial arithmetic. However, pageable PCIe offload takes about **87.4 ms**, while a six-thread ISA-L pipeline takes about **48.9–52.3 ms** across the two follow-up cohorts. Data residency changes the engineering decision.

Read the [final research report](FINAL_REPORT.md) for the mathematical contract, full comparisons, reproduction and limits. The [complete median/p95 table](results/summary.csv) contains 1,060 separate cohort/case/mode/scope rows.

## Research question

Can packed GPU finite-field arithmetic accelerate complete systematic erasure encoding and recovery, and when do CPU throughput, PCIe transfers or host staging remove the kernel advantage?

The workload models storage-code computation relevant to distributed storage and data-availability systems. It does not implement a network protocol, adversarial availability sampling or a mining reward mechanism.

## Mathematical scope and implementation

The field is GF(2^8), using the polynomial x^8+x^4+x^3+x^2+1 (`0x11d`). The generator has k identity rows for original data and four Cauchy parity rows, with coefficient `1/(i XOR j)` for parity index i in k..k+3 and data index j in 0..k-1. Recovery selects k surviving rows, inverts that matrix in the same field and reconstructs the missing shards.

The independent [CPU reference](experiments/field.py) implements polynomial multiplication, inverses, matrix inversion, encoding and recovery. The original [CUDA implementation](cuda/erasure.cu) compares four arithmetic strategies:

| Variant | Implementation |
| --- | --- |
| `serial` | Bit-serial multiplication and reduction. |
| `table` | Byte multiplication lookup and XOR accumulation. |
| `nibble4` | Packed four-output nibble-table encoding. |
| `clmad4` | Four byte symbols packed into 16-bit lanes, `clmad.lo.u64` accumulation and field reduction. |

Odd shard lengths use a byte fallback. Arithmetic is exact integer polynomial arithmetic. The CPU baseline uses unmodified ISA-L at revision `ab01645a2d9ab581477c703103fe3450c4696845`, with original wrappers for the full pipeline and optional OpenMP chunking.

## Completed experiments

The primary campaign covers 8+4 and 16+4 shards, original payloads of 64 KiB, 1 MiB, 16 MiB, 128 MiB and 256 MiB, and two loss patterns: four missing data shards, or two missing data and two missing parity shards. Both seeds, 20261109 and 20261110, are retained: **40 complete cases** with all four GPU kernels and direct ISA-L comparisons.

The transfer/multicore follow-up adds **20 cases** across seeds 20261111 and 20261112. It compares 1/2/4/6 CPU threads with GPU-resident, pageable, caller-owned pinned and staged-pinned paths. A further **20 direct CPU cases** regenerate the exact same payloads and check their SHA-256 hashes, avoiding an unfair small-payload comparison against OpenMP team overhead. This direct follow-up runs in a separate timing window; precise paired speedups across windows are not claimed.

## Measured findings

Complete pipeline medians in ms for 16+4 shards and four missing data shards. Cells show primary / replication from the transfer follow-up:

| Original payload | CPU, six threads | GPU resident | Pageable offload | Pinned offload | Staged offload |
| --- | --- | --- | --- | --- | --- |
| 128 MiB | 24.500 / 25.569 | 4.897 / 4.910 | 43.850 / 43.877 | 43.542 / 43.560 | 76.045 / 76.532 |
| 256 MiB | 48.860 / 52.310 | 9.735 / 9.761 | 87.419 / 87.465 | 86.981 / 87.017 | 153.678 / 155.706 |

The resident GPU advantage survives both cohorts. Large host-to-GPU offload loses against the tested multicore CPU baseline, and extra host staging copies make it slower still. At 16+4 and 1 MiB, the direct CPU follow-up takes about 0.290 ms in both cohorts, versus 0.362/0.363 ms for pinned offload. An OpenMP-only comparator would obscure that small-payload result.

No consistent offload-win region is established over the sampled grid against the strongest appropriate tested CPU path. This is a result for the measured platform and workload; it does not rule out another crossover on a different interconnect, at other sizes or inside an application that already keeps data on GPU.

## Correctness evidence

- All 65,536 CPU field products checked against a separate polynomial oracle, plus all 255 nonzero inverses.
- 616 erasure-pattern checks, exhaustive on small cases and seeded on larger spaces; singular matrices are rejected.
- All 65,536 GPU multiplication results checked against the CPU field table before each primary cohort.
- Fifteen zero/maximum/random edge fixtures at shard lengths 1, 3, 4, 17 and 513 checked against both the independent encoder and ISA-L in every GPU variant.
- In the primary campaign, every large encoded parity output compared byte-for-byte with ISA-L; every restored payload checked against the input across the campaigns, outside timed regions.

Large runtime outputs are retained as input/parity hashes and exact-comparison records rather than complete byte arrays. CPU CI can replay the small arithmetic controls and inspect retained evidence, but cannot reconstruct a past large GPU output from its hash.

## Reproduce the CPU checks

Requires Python 3.12 and its standard library. Run without `-O`, because assertions are correctness gates:

```sh
python cpu_study.py --output build/ci.json
python verify_records.py
python verify_results.py
python summarize_results.py
git diff --exit-code -- results/summary.csv
```

These commands rerun independent field/recovery controls, validate the retained record structure and source/result hashes, and regenerate the published table. CPU CI runs this path without CUDA or a native ISA-L rebuild.

## Build the native CPU baseline

The measured GPU/native platform is Linux x86_64 with Ryzen 5 5500 and RTX 5060 Ti 16 GB (SM 120). Install a C/C++ compiler, NASM and CMake. The recorded native build used g++ 13.3, NASM 2.16.01 and CMake 4.1.3. Build the pinned upstream and both original wrappers:

```sh
git clone https://github.com/intel/isa-l.git upstream-isal
git -C upstream-isal checkout ab01645a2d9ab581477c703103fe3450c4696845
cmake -S upstream-isal -B build/isal -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DISAL_BUILD_TESTS=OFF -DISAL_BUILD_PERF_TESTS=OFF -DISAL_BUILD_ISAL_SHIM=OFF -DISAL_BUILD_FUZZ_TESTS=OFF -DISAL_BUILD_IGZIP_CLI=OFF
cmake --build build/isal -j6
g++ -O3 -shared -fPIC native/isal_bridge.cpp -Iupstream-isal/include -Lbuild/isal -Wl,-rpath,$PWD/build/isal -lisal -o build/libisal_bridge.so
g++ -O3 -shared -fPIC -fopenmp native/isal_bridge.cpp native/isal_parallel.cpp -Iupstream-isal/include -Lbuild/isal -Wl,-rpath,$PWD/build/isal -lisal -o build/libisal_parallel.so
```

## Prepare and reproduce the GPU campaign

The measured stack uses Python 3.12, NumPy 2.2.6, CuPy 13.6.0, CUDA 12.x runtime/headers and CUDA 13.3.73 nvcc for the SM 120 CUBIN. Use a separate Python environment; a newer system compiler does not replace the pinned CuPy runtime/header setup.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-gpu.txt
/path/to/cuda-13.3/bin/nvcc -O3 -std=c++17 -arch=sm_120 -cubin cuda/erasure.cu -o build/erasure.cubin
export CUDA_PATH=/path/to/cuda-12.8
export CUDA_VISIBLE_DEVICES=0
export PYTHON=python
bash run_gpu.sh upstream-isal
bash run_gpu.sh upstream-isal --seed 20261110 --output results/gpu-replication.json
STUDY=transfer_study.py bash run_gpu.sh
STUDY=transfer_study.py bash run_gpu.sh --seed 20261112 --output results/transfer-replication.json
python direct_cpu_study.py
python direct_cpu_study.py --reference results/transfer-replication.json --output results/direct-cpu-replication.json
STUDY=record_environment.py bash run_gpu.sh
python summarize_results.py
```

Run timing campaigns sequentially on a free GPU. Scripts write the corresponding raw records under `results/`. [build.json](results/build.json) binds the measured binaries and upstream revision; the post-run collector assumes the original sibling ISA-L/build layout and compiler location, which must be adapted for another installation. See the [full reproduction notes](FINAL_REPORT.md#reproduction).

## Measurement rules

Each fixed payload/mode uses five warm-ups and thirty measured repetitions, with alternating implementation order. Repetitions reuse one payload; they are not thirty independent input seeds. Median and nearest-rank p95 remain separate by cohort and timing scope.

The complete pipeline encodes parity, gathers survivors, recovers four missing shards and assembles the original payload. Matrices, plans and buffers are preallocated. Resident graph timing excludes transfers, allocation and verification. Pageable mode adds original-payload H2D and restored-payload D2H. Pinned mode assumes caller-owned pinned buffers; staged mode also includes ordinary-host/pinned copies. Pinned preparation time is retained separately.

Reserved device-pool observations are recorded per case; they are not an instrumented global memory peak or a measure of pinned host memory. No intentional overlapping GPU research workload, clock change or power-limit adjustment was used.

## Repository and research record

| Artifact | Purpose |
| --- | --- |
| [RESEARCH.md](RESEARCH.md), [METHODS.md](METHODS.md), [FINAL_REPORT.md](FINAL_REPORT.md) | Required scope, measurement rules and final interpretation. |
| [RESULTS.md](RESULTS.md) | Index of all raw campaigns and their evidence boundaries. |
| [CPU controls](results/cpu-study.json) | Independent arithmetic and erasure-pattern checks. |
| [Primary GPU campaign](results/gpu-study.json), [replication](results/gpu-replication.json) | All kernel variants, loss patterns and resident/offload timings. |
| [Transfer primary](results/transfer-study.json), [replication](results/transfer-replication.json) | Multicore CPU, pinned buffers and staging comparisons. |
| [Direct CPU primary](results/direct-cpu-study.json), [replication](results/direct-cpu-replication.json) | Exact-payload baseline without OpenMP team overhead. |
| [summary.csv](results/summary.csv), [build.json](results/build.json), [environment.json](results/environment.json) | Derived median/p95 table and execution provenance. |

## Limits and next research decisions

The campaign supports the tested carryless kernel and GPU-resident workflow, while rejecting an unconditional offload-speedup claim over this grid. It covers one field polynomial, a systematic Cauchy matrix, two code sizes and controlled erasure patterns. Recovery includes encoding to simulate loss; a recovery-only application has different costs. Distributed traffic, availability security, multi-GPU scaling, energy and profitability were not measured.

Useful follow-ups are coding adjacent to existing GPU computation, application-level batching, recovery-only workloads and other interconnects. A faster resident kernel alone is insufficient evidence for a faster storage service or a secure data-availability protocol.

## License and contributions

Original experiment code is [MIT licensed](LICENSE). ISA-L retains its upstream license. Contributions should include exact field/matrix definitions, reproducible commands, source revisions, environment, byte-for-byte correctness evidence, CPU comparator configuration and timing/transfer boundaries. Negative crossover results are welcome.

## Primary references

- [Pinned ISA-L baseline](https://github.com/intel/isa-l/tree/ab01645a2d9ab581477c703103fe3450c4696845)
- [NVIDIA CUDA 13.3 carryless multiplication description](https://developer.nvidia.com/blog/?p=119649)

## Related Requant Network research

- [GPU Wagner research](https://github.com/requant-network/gpu-wagner-research)
- [GPU algebra search](https://github.com/requant-network/gpu-algebra-search)
- [GPU ZK research](https://github.com/requant-network/gpu-zk-research)
- [GPU FHE research](https://github.com/requant-network/gpu-fhe-research)
- [GPU proof-of-useful-work research](https://github.com/requant-network/gpu-pouw-research)

## Follow-up closure, 12 November 2026

The [follow-up report](FOLLOWUP_REPORT.md) records the disposition of the concrete next steps: new experiments, negative findings and measured resource limits. The earlier completed campaign remains available for historical comparison. Run `python verify_followups.py` for the added evidence checks.
