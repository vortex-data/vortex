# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet: **≥2× for both end-to-end projected reads and queries
across Q1/Q5/Q6/Q9/Q10 at SF1/SF10, warm and cold**. The goal is **not met**;
SF100 scaling/profiling is deferred until these matrices are stable.

[Setup](benchmarks/cudf-ndsh/README.md) ·
[Validation and profile safety](benchmarks/cudf-ndsh/VALIDATION.md) ·
[Commit checkpoint and next steps](benchmarks/cudf-ndsh/PROGRESS.md)

## Current state

- Default-OFF integration covers all five queries and projected reads with shared
  full-table fixtures, projections, post-read filters, and generic cuDF operations.
  No hand-fitted query kernels; `q1_fused` is absent. Original benchmarks stay separate.
- Q1 uses sum/count groupby plus averages; Q5/Q9/Q10 read independent tables concurrently
  for both formats. Vortex uses 16M-row blocks, cacheable pinned staging, 8 GiB pool
  retention, and one final owning cuDF materialization. No event optimization is retained.
- Renamed `_local` benchmarks expose warm/cold cache axes and include SF10 by default.
  Cold callbacks verify zero file-page residency after sync/eviction; Vortex data uses
  `O_DIRECT`, metadata is buffered, and Parquet keeps its native reader. Lower-level
  storage caches are not flushed. Pinned-host staging → HtoD → decode, **not GDS**.
- Timing includes complete read/query work, destruction, and device completion;
  excludes fixture writing, checks, and eviction. See README for the full contract.

## Validation gate and next steps

The patch is refreshed and apply-checked; generic performance is committed as `59a7ea66a`.
Cold-cache patch/tests are committed separately as `cd3192a02`. Source/offline checks
pass, but the 1200 s Release build timed out before cuDF benchmark/adapter relinks:
**binaries are stale**.
Only the Vortex Release archive completed. No fresh runtime validation or performance runs.

1. Explicitly choose a longer bounded build window; finish relinks and validate runtime.
   Do not retry automatically. Earlier passing tests do not establish final-source validity.
2. Collect stable SF1/SF10 warm/cold matrices; existing timings predate the event revert
   and are diagnostic only. Capture the still-missing full Q1 query profile with the
   linked profiling safety guard; existing profiles contain sensitive environment metadata.
3. Only then consider SF100 with separate Vortex/RMM memory accounting, publish Vortex
   prerequisites, update the retained pin, and prepare the upstream POC.

Local-file benchmark scope only; no public API, GPU writer, remote I/O, or full RMM
integration. Regenerate fixtures after generator fixes; this is not full TPC-H conformance.
