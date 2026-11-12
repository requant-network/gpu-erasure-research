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

The implemented batching is local contiguous batching. Network scheduling, distributed repair, storage I/O and energy efficiency have not been measured. The heterogeneous two-GPU follow-up below tests a specific local recovery deployment.

Reproduce with `python build_recovery.py --help`, then `python recovery_study.py` and `python recovery_study.py --seed 20261212 --output results/recovery-replication.json` in the documented CUDA/CuPy/ISA-L environment. [recovery_study.py](recovery_study.py) and [native/isal_recovery.cpp](native/isal_recovery.cpp) define the timing and error checks; [build provenance](results/recovery-build.json) pins ISA-L and the native source. [CPU evidence replay](verify_followups.py) validates all added source/result bindings and derives the CSV. CI validates retained evidence; it does not rerun GPU timings.

## Server timing boundary

The original single-GPU campaigns run serially on the same RTX 5060 Ti with server services unchanged. The later two-GPU campaigns use the RTX 5060 Ti and CMP 50HX exclusively: four GPU-consuming services were temporarily stopped under the user's explicit authorization. All four were restored after the final cohort, and their running state was independently checked. Clocks and power limits were unchanged. Measurements describe this server and retained workload; small percentage differences are not treated as universal improvements. No energy measurement is inferred.

## Heterogeneous two-GPU recovery

Two independent seeds cover nine workloads each: 8+4 and 16+4 codes, 16/256 MiB payloads, four data or two data/two parity losses, plus an 8+4, 1 GiB, four-data-loss case. Every policy uses five warm-ups and thirty measured observations. Input encoding and loss simulation remain outside recovery. GPU 0 uses native CLMAD on SM120; GPU 1 uses the portable nibble-table backend on SM75. All 65,536 field products are checked on each GPU before timing. The output is sentinel-cleared outside every timer and every restored byte is checked after every launch.

Independent byte-column partitions run concurrently through two host worker threads. Resident wall time includes dispatch and joining both workers; downloads for independent validation are outside that timer. Staged wall time includes copying ordinary input into reusable pinned partitions, both uploads, graph execution, both downloads and assembly into the full ordinary output. Setup/allocation is reported separately. This is local partitioned recovery, with no peer copies or distributed storage.

Selected medians in milliseconds, four missing data shards. `50/50` assigns half the bytes to each GPU. The CPU column is the lowest median among the retained 1/2/4/6-thread ISA-L controls, rather than a fixed weak comparator.

| Cohort | Code | Payload | Best CPU | RTX resident | CMP resident | 50/50 resident | RTX staged | CMP staged | 50/50 staged |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| study | 8+4 | 256 MiB | 31.770 | 5.196 | 5.112 | 2.648 | 143.522 | 1408.361 | 739.829 |
| replication | 8+4 | 256 MiB | 30.434 | 5.213 | 5.119 | 2.654 | 144.251 | 1419.591 | 739.779 |
| study | 16+4 | 256 MiB | 33.949 | 4.961 | 4.950 | 2.564 | 145.882 | 1421.525 | 731.874 |
| replication | 16+4 | 256 MiB | 32.975 | 4.962 | 4.953 | 2.564 | 145.535 | 1419.635 | 731.868 |
| study | 8+4 | 1 GiB | 131.624 | 20.535 | 20.184 | 10.316 | 578.425 | 5682.136 | 2964.864 |
| replication | 8+4 | 1 GiB | 132.075 | 20.545 | 20.185 | 10.321 | 570.023 | 5672.168 | 2952.310 |

At 1 GiB, the resident two-GPU policy is about 1.99 times faster than RTX alone in both cohorts. This is a replicated resident-memory result. It does not carry over to the host-memory path. During this campaign, [read-only link metadata](results/multigpu-interconnect.json) reports Gen3 x8 for the RTX and Gen2 x1 for the CMP. Metadata alone is not a bandwidth measurement; the timed full-path observations establish the deployment cost here. Sending 50%, 25% or even 12.5% of the payload through the CMP path makes staged recovery slower than RTX alone. Resident and host-memory deployments therefore require different partition policies.

Build and timing evidence: [builder](build_multigpu.py), [study](multigpu_study.py), [build record](results/multigpu-build.json), [first cohort](results/multigpu-study.json), [replication](results/multigpu-replication.json). Reproduce with `python build_multigpu.py --nvcc /path/to/nvcc`, `python multigpu_study.py` and `python multigpu_study.py --seed 20261213 --output results/multigpu-replication.json`, with both GPUs available and the original ISA-L recovery build present. CPU CI replays provenance, sample structure, exact-validation flags and derived summaries; it does not rerun CUDA or independently rerun the byte comparisons.

## Transfer-aware refinement and final deployment decision

After observing the narrow CMP link and the poor staged results through 87.5% RTX assignment, an explicitly exploratory refinement tests 93.75% and 96.875% RTX assignment. Two fresh seeds each cover the 8+4, four-data-loss, 256 MiB and 1 GiB workloads. The arithmetic, CPU controls, full byte comparisons, five warm-ups and thirty measured observations are unchanged. The [wrapper](transfer_balance_study.py) applies a guarded source transformation to the frozen main study; its reconstructed source hash is retained and replayed by CPU CI. These additional policies were selected after the first experiment, not preregistered, and do not establish a globally optimal partition.

Staged full-wall medians in milliseconds:

| Cohort | Payload | Best CPU | RTX alone | 93.75% RTX | 96.875% RTX |
| --- | --- | ---: | ---: | ---: | ---: |
| study | 256 MiB | 30.511 | 144.625 | 143.378 | 138.528 |
| replication | 256 MiB | 31.831 | 144.701 | 143.684 | 139.776 |
| study | 1 GiB | 132.829 | 578.197 | 578.064 | 559.803 |
| replication | 1 GiB | 132.122 | 576.192 | 577.730 | 559.786 |

Assigning 96.875% of bytes to RTX reduces full-path median time by 2.8-4.2% relative to RTX alone across these four observations. The 93.75% policy has no consistent 1 GiB advantage. The modest retained gain corrects an overly broad conclusion that a second GPU can never help the host path: a transfer-aware split can help a little. It does **not** overturn the application decision: the best tested staged GPU policy remains more than four times slower than the best CPU control on these workloads. Equal partitioning is useful for resident recovery; for host-resident data use ISA-L on this server. A different interconnect, persistent GPU consumer or deployment needs its own full-path measurement.

Evidence: [refinement cohort](results/transfer-balance-study.json), [independent replication](results/transfer-balance-replication.json). Reproduce with `python transfer_balance_study.py` and `python transfer_balance_study.py --seed 20261214 --output results/transfer-balance-replication.json` after the same two-GPU build. In total the added evidence comprises 48 original recovery-only cases, 18 main two-GPU cases and 4 refinement cases; [CPU replay](verify_followups.py) validates nine records and derives 676 median/p95 rows. Raw samples and setup costs remain in the JSON records. No distributed, storage, energy or other-algorithm multi-GPU claim follows from this result.
