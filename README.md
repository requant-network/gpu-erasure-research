# GPU Erasure Research

Exact Reed-Solomon encoding and recovery on consumer GPUs.

A research project by [Requant Network](https://requant.network).

**Status: completed bounded research campaign, with two independent cohorts per experiment.**

CUDA 13.3 carryless arithmetic improves the tested resident GPU code, but PCIe
offload does not beat the strongest tested CPU baseline over the sampled payloads.
The resident/offload distinction changes the engineering decision.

- [Final report and reproduction](FINAL_REPORT.md)
- [Methods](METHODS.md) and [results](RESULTS.md)
- [Raw measurements](results/) and [derived median/p95 table](results/summary.csv)

CPU verification: `python cpu_study.py --output build/ci.json`, then
`python verify_records.py`, `python verify_results.py` and `python summarize_results.py`.

See [RESEARCH.md](RESEARCH.md) for the committed experimental scope. Original code is MIT licensed; upstream sources retain their own licenses.
