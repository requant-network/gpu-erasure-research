# Methods

A research project by [Requant Network](https://requant.network).

The complete mathematical contract, timing boundaries, pinned revision and runnable commands are in the [final report](FINAL_REPORT.md). Sources and exact measured binaries are bound in [build.json](results/build.json); hardware/software are in [environment.json](results/environment.json).

Independent field arithmetic and recovery checks run in `cpu_study.py`. The main GPU study performs all-pairs multiplication and edge-payload validation before large comparisons. It checks every parity/restored byte against ISA-L/original inputs at runtime. The transfer study adds thread counts and pinned/staged modes; the direct CPU follow-up prevents an OpenMP setup penalty from weakening the comparator.

Five warm-ups and thirty observations apply per fixed payload/mode. Two independent payload cohorts remain separate. Order alternates within each case; cases execute sequentially. p95 uses the nearest-rank convention, index ceil(0.95n)-1. Buffers/plans and correctness checks lie outside timed graph intervals; offload/staging boundaries are explicitly distinct. Reserved-pool observations are not a global memory-peak trace. No energy is estimated from TDP.
