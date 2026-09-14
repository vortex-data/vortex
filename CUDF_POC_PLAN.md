# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet: **≥2× for both end-to-end projected reads and queries
across Q1/Q5/Q6/Q9/Q10 at SF1/SF10, warm and cold**. The goal is **not met**;
SF100 scaling/profiling is deferred until these matrices are stable.

[Setup](benchmarks/cudf-ndsh/README.md) · [Validation](benchmarks/cudf-ndsh/VALIDATION.md) · [Progress](benchmarks/cudf-ndsh/PROGRESS.md)

## Minimal scope

- `upstream.patch` touches only `cpp/benchmarks/ndsh/` and one include hook in
  `cpp/benchmarks/CMakeLists.txt`; no generator changes or generator-test target.
  Default-OFF comparisons use the **original pinned cuDF generator** and identical
  logical fixtures, scan projections, and post-read predicates across formats.
  Original native Parquet-pushdown benchmarks remain separate.
- Q1 uses original `SUM`/native `MEAN`/`COUNT`; Q5/Q9/Q10 table reads are sequential.
  No hand-fitted kernels or event optimization. Vortex retains 16M-row blocks,
  cacheable pinned staging, an 8 GiB pool, and one final owning materialization.
- Warm/cold controls remain: cold callbacks sync/evict and verify zero resident pages;
  Vortex data uses `O_DIRECT`, metadata is buffered, and Parquet keeps its native reader.
  Lower-level caches are not flushed. Pinned-host staging → HtoD → decode, **not GDS**.
- Timing includes complete reads/queries, destruction, and device completion;
  writing/checks/eviction are untimed. No public API, GPU writer, remote I/O, or full RMM.

Optional `benchmarks/cudf-ndsh/generator-fixes.patch` preserves four independent fixes
and its own test across seven files. It applies independently to pinned cuDF; both
application orders produce identical trees. It is separately reviewable, not bundled
or automatic. Applying it requires regenerating both formats and labeling the dataset.
Neither dataset establishes full TPC-H conformance.

Original data may yield empty/degenerate Q6/Q10 or low-SF supplier joins. Projection/value
and CPU result checks remain, including zero matches and synthetic nonempty cases.
Q6 checks zero-match `SUM` is NULL and reports revenue `"NULL"`, not zero; its migrated
and expanded query tests remain in the main patch. Explicit zero match counts disclose
queries that are **not meaningful full-query performance evidence**.

## Validation and next steps

17 offline tests, five-query clang-format, Ruff, and patch checks passed. Query compile-only
validation passed **10/10**, Vortex ON/OFF, without warnings; the optional generator test
also compiled (not run). The commit breakdown is recorded in the linked progress notes.
**Binaries remain stale:** no CMake regeneration, full build, relinks, GPU, or fresh
minimal-source runtime/memcheck/performance runs. The previous full build timed out at
1200 s / 223 of 635 steps. No broad automatic retry; see linked validation details.

Prior timings used altered generator/performance code and event suppression: **HISTORICAL,
not minimal-patch baselines**. After bounded relinking/runtime validation, regenerate both
formats' fixtures and collect fresh labeled SF1/SF10 warm/cold baselines and a safe Q1
query profile. Defer SF100; publish Vortex prerequisites and update the retained pin.
