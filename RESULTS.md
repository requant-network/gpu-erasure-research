# Results and evidence

A research project by [Requant Network](https://requant.network).

| Evidence | Role |
| --- | --- |
| [CPU controls](results/cpu-study.json) | Exhaustive field products/inverses, 616 erasure patterns. |
| [Primary](results/gpu-study.json), [replication](results/gpu-replication.json) | 40 cases, four GPU modes, optimized direct ISA-L, two missing-shard patterns, resident and full offload. |
| [Transfer primary](results/transfer-study.json), [replication](results/transfer-replication.json) | 20 cases, ISA-L 1/2/4/6 threads and resident/pageable/pinned/staged GPU paths. |
| [Direct CPU primary](results/direct-cpu-study.json), [replication](results/direct-cpu-replication.json) | Exact matching follow-up payloads, no OpenMP team; separate timing window. |
| [Summary](results/summary.csv) | Regenerable per-cohort median/p95 and reserved-pool observations. |
| [Build](results/build.json), [environment](results/environment.json) | Source/result/binary bindings, compiler flags, upstream and device metadata. |

See the [final report](FINAL_REPORT.md) for numerical findings, crossover decision, reproduction and limits. Resident kernel wins and transfer-inclusive losses are both retained. Large runtime outputs are represented by input/parity hashes and exact-comparison flags, not archived byte arrays; CPU CI validates the retained structure and independent small arithmetic but cannot replay a past GPU byte comparison from a hash.
