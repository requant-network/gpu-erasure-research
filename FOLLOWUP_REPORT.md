# Recovery follow-up and final decision

Requant Network, 12 November 2026. This supplement closes the recovery-only and application-batching questions in the [historical report](FINAL_REPORT.md).

The input now consists of the **already available k surviving shards**. Encoding and simulated loss are preparation, outside recovery timers. Recovery computes four missing shards and assembles the entire original payload. Both 8+4 and 16+4 codes use the same GF(256) polynomial, with all-data and mixed data/parity losses. Batches are contiguous independent payloads within each shard; their packing cost is retained in setup, not silently called free.

Two seeded campaigns each cover 24 cases: 64 KiB, 1 MiB, 16 MiB and 256 MiB payloads, plus batches of eight 1 MiB or 16 MiB payloads. Each mode has five warm-ups and thirty measured repetitions; mode order alternates. Every restored byte is compared against the original, and five malformed native calls must reject. Direct ISA-L uses one thread without an OpenMP team; the multicore comparators use 2/4/6 threads and 256 KiB chunks.

Selected medians, milliseconds. `pinned` reuses caller-owned pinned input/output; `staged` includes ordinary-to-pinned and pinned-to-ordinary copies. Resident timing requires the input already on the device. Full median/p95 data: [followup-summary.csv](results/followup-summary.csv).

| Cohort | Code | Payload | Batch | Missing | Best CPU | GPU resident | GPU pinned | GPU staged |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| study | 8+4 | 256 MiB | 1 | 0,1,2,3 | 31.150 | 5.163 | 82.927 | 143.638 |
| study | 8+4 | 1 MiB | 8 | 0,1,2,3 | 1.077 | 0.158 | 2.594 | 4.138 |
| study | 8+4 | 16 MiB | 8 | 0,1,2,3 | 15.605 | 2.593 | 41.502 | 71.585 |
| study | 8+4 | 256 MiB | 1 | 0,1,8,9 | 31.792 | 5.139 | 82.903 | 143.885 |
| study | 8+4 | 1 MiB | 8 | 0,1,8,9 | 0.938 | 0.158 | 2.593 | 4.152 |
| study | 8+4 | 16 MiB | 8 | 0,1,8,9 | 15.387 | 2.587 | 41.490 | 71.397 |
| study | 16+4 | 256 MiB | 1 | 0,1,2,3 | 33.413 | 4.910 | 82.677 | 144.400 |
| study | 16+4 | 1 MiB | 8 | 0,1,2,3 | 1.245 | 0.152 | 2.595 | 4.239 |
| study | 16+4 | 16 MiB | 8 | 0,1,2,3 | 17.866 | 2.473 | 41.382 | 71.687 |
| study | 16+4 | 256 MiB | 1 | 0,1,16,17 | 36.410 | 4.903 | 82.672 | 145.197 |
| study | 16+4 | 1 MiB | 8 | 0,1,16,17 | 1.130 | 0.152 | 2.594 | 4.235 |
| study | 16+4 | 16 MiB | 8 | 0,1,16,17 | 15.942 | 2.470 | 41.377 | 70.789 |
| replication | 8+4 | 256 MiB | 1 | 0,1,2,3 | 31.993 | 5.164 | 82.928 | 144.202 |
| replication | 8+4 | 1 MiB | 8 | 0,1,2,3 | 1.016 | 0.158 | 2.597 | 4.143 |
| replication | 8+4 | 16 MiB | 8 | 0,1,2,3 | 15.930 | 2.594 | 41.502 | 71.395 |
| replication | 8+4 | 256 MiB | 1 | 0,1,8,9 | 30.508 | 5.153 | 82.909 | 144.284 |
| replication | 8+4 | 1 MiB | 8 | 0,1,8,9 | 0.992 | 0.158 | 2.593 | 4.141 |
| replication | 8+4 | 16 MiB | 8 | 0,1,8,9 | 15.526 | 2.596 | 41.502 | 71.344 |
| replication | 16+4 | 256 MiB | 1 | 0,1,2,3 | 31.915 | 4.926 | 82.686 | 143.661 |
| replication | 16+4 | 1 MiB | 8 | 0,1,2,3 | 1.154 | 0.153 | 2.591 | 4.138 |
| replication | 16+4 | 16 MiB | 8 | 0,1,2,3 | 15.898 | 2.481 | 41.389 | 70.487 |
| replication | 16+4 | 256 MiB | 1 | 0,1,16,17 | 32.131 | 4.919 | 82.682 | 143.900 |
| replication | 16+4 | 1 MiB | 8 | 0,1,16,17 | 1.001 | 0.152 | 2.590 | 4.157 |
| replication | 16+4 | 16 MiB | 8 | 0,1,16,17 | 17.937 | 2.477 | 41.384 | 71.331 |

The result supports GPU recovery where shards are already on the GPU and the restored data remains useful there. Host-memory recovery does not meet the complete-path improvement gate: transfers and payload assembly dominate, even with reusable pinned buffers and batching. The CPU baseline remains the practical choice for this host-resident workload. Increasing arithmetic throughput alone does not resolve this result.

The implemented batching is local contiguous batching; network scheduling, distributed repair, storage I/O, heterogeneous multi-GPU operation and energy efficiency have not been measured. They are separate deployments, not prerequisites for this bounded recovery decision.

Reproduce with `python build_recovery.py --help`, then `python recovery_study.py` and `python recovery_study.py --seed 20261212 --output results/recovery-replication.json` in the documented CUDA/CuPy/ISA-L environment. [recovery_study.py](recovery_study.py) and [native/isal_recovery.cpp](native/isal_recovery.cpp) define the timing and error checks; [build provenance](results/recovery-build.json) pins ISA-L and the native source. [CPU evidence replay](verify_followups.py) validates all added source/result bindings and derives the CSV. CI validates retained evidence; it does not rerun GPU timings.

## Server timing boundary

GPU campaigns run serially on the same RTX 5060 Ti. Clocks, power limits and unrelated server services are unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement or heterogeneous multi-GPU throughput is inferred.
