# Research protocol

A research project by [Requant Network](https://requant.network).

## Required experiments

1. Pin the GF(256) polynomial and systematic MDS matrix. Implement independent CPU multiplication, matrix inversion, encoding and recovery with exhaustive small erasure-pattern checks.
2. Build a pinned optimized ISA-L CPU baseline and compare byte-for-byte with the independent reference. Record compiler and source provenance.
3. Implement GPU bit-serial, table/XOR and CUDA 13.3 carryless variants with full output checks; test zero, maximum and independent random payloads.
4. Execute encoding and complete erasure recovery for 8+4 and 16+4 shards across payload sizes, including resident and transfer-inclusive paths. Establish CPU/GPU crossover against the optimized baseline.
5. Include peak allocation observations, median/p95, two independent payload seeds and negative outcomes. Separate storage-code results from network-specific data-availability/security claims.

## Shared evidence requirements

Use the released RTX 5060 Ti 16 GB without concurrent GPU experiments. Retain exact executed-source hashes, pinned upstream revisions, environment, raw measurements, independent correctness evidence, timing boundaries and reproduction commands. Fixed-input latency comparisons use five warm-ups and thirty measured repetitions where practical. Stochastic search uses independent seeds and retains unsuccessful/censored trials. Energy requires sampled measurements; no TDP-derived efficiency claims.

## Completion requirements

Publish executable CPU/GPU studies, all required comparisons, independent verification, raw results, derived tables, English final conclusions and limitations. Public repository topics, homepage and attribution must identify Requant Network. CPU/reference and artifact CI must pass at the published commit. A negative result may complete a hypothesis test; missing experiments may not.

## Completion evidence

All five required experiments are covered by `cpu_study.py`, `gpu_study.py`,
`transfer_study.py` and `direct_cpu_study.py`, with both cohorts retained under
`results/`. The [final report](FINAL_REPORT.md) records exact arithmetic, pinned
ISA-L, all GPU variants, full recovery, crossover against multicore/direct CPU,
memory observations, medians/p95, negative outcomes and scope limits. Build and
environment records bind the execution. CPU CI reruns reference arithmetic and
checks source/result hashes, retained-record invariants and derived tables.

This bounded campaign is complete. Resident GPU wins do not imply a universal
transfer-inclusive advantage or a deployed data-availability protocol.

## Follow-up closure, 12 November 2026

The [follow-up report](FOLLOWUP_REPORT.md) records the disposition of the concrete next steps: new experiments, negative findings and measured resource limits. The earlier completed campaign remains available for historical comparison. Run `python verify_followups.py` for the added evidence checks.
