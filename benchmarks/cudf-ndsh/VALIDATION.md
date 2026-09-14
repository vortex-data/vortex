# Validation

Checkpoint: 2026-09-14. Pinned cuDF:
`5339497a1a17d799687cbf189fb113411fb015ca`, Release (`-O3 -DNDEBUG`).
[Current state](PROGRESS.md) · [Setup and commands](README.md)

## Validation status

**Binaries are stale: no fresh runtime validation or new performance runs.** All five
Release query targets built before default kernel-event suppression was reverted.
The latest build did not relink the cuDF benchmarks or adapter.

Completed checks:

| Check                                                       | Result    |
| ----------------------------------------------------------- | --------- |
| `cargo nextest run -p vortex-cuda pinned`                   | 5 passed  |
| `cargo +nightly fmt --all`                                  | Passed    |
| `cargo clippy --all-targets --all-features`                 | Passed    |
| clang-format, 8 edited C++ files                            | Passed    |
| `python3 -B benchmarks/cudf-ndsh/test_build_integration.py` | 14 passed |
| `.venv/bin/ruff check` / `format --check`, integration test | Passed    |
| Refreshed patch: forward/cached and reverse apply checks    | Passed    |

Release build attempted:

```sh
cmake --build build/cudf-ndsh-build \
  --target NDSH_Q01_NVBENCH NDSH_Q05_NVBENCH NDSH_Q06_NVBENCH \
    NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH NDSH_VORTEX_IO_TEST -j4
```

**Timed out after 1200 s.** CMake regeneration expanded to 644, then 635 build steps;
progress reached 223/635. The Vortex Release archive built successfully, but the cuDF
benchmarks and adapter were **not relinked**. No automatic retry: explicitly choose a
longer bounded build window before completing relinks and fresh runtime checks.

### Earlier runtime and integration checks

Earlier validation, not a fresh post-revert run:

| Check                                                       | Result                                 |
| ----------------------------------------------------------- | -------------------------------------- |
| Q1/Q5/Q6/Q9/Q10 matched projected read/query, SF0.01        | All 28 states passed                   |
| Pinned Release SF1 / selected SF10 states                   | 28 / 20 passed                         |
| Same-fixture CPU references and independent synthetic cases | Passed                                 |
| Compute Sanitizer, all 28 SF0.01 states                     | 0 errors, no skips                     |
| Pinned adapter / CUDA FFI tests                             | 15 / 20 passed; adapter memcheck clean |
| Generator tests, including order-date/supplier regressions  | 8 passed; memcheck clean               |
| NDS-H / FFI CMake integration                               | 8 / 13 passed                          |
| Original Q10 Parquet-pushdown/write benchmark, SF0.01       | Passed                                 |

SF1 covered all Q9 amount engines; the earlier 20-state SF10 check used binary-op.
SF1/SF10 benchmark memcheck was not rerun after performance changes. Earlier Python
byte-compilation and offline CMake checks passed; Ruff/cmake-format were unavailable.
The earlier integration counts above are separate from the current 14-test offline run.
Supplemental cuDF 26.08 Debug runs established execution, not pinned Release performance.

## Timing contract

Both formats use shared full-table fixtures, matching projections/post-read filters,
and generic cuDF Q1/Q5/Q6/Q9/Q10 execution. Retained implementation details are in
[README.md](README.md).

- Compare CPU wall means. Timed work includes complete reads, import/copies/final
  materialization, query execution where selected, destruction, and device completion.
  Fixture writing, validation, and cache eviction are outside timing.
- Before the timed portion of **every** manual cold callback, each input file gets
  `fdatasync` + `POSIX_FADV_DONTNEED`, followed by a required `mincore` residency == 0.
  Cold Vortex data reads use `O_DIRECT`; metadata remains buffered. Parquet keeps its
  native reader. Lower-level storage caches are **not** flushed: this is OS-page-cache
  coldness, not guaranteed cold media.
- Local files → pinned-host staging → HtoD → GPU decode; **no GPUDirect Storage**.
  RMM peaks exclude Vortex allocations; separate accounting is required before SF100.

## Latest timings — diagnostic/historical only

**Every measurement below predates the revert of default kernel-event suppression.**
They are not final performance evidence for the committed source or refreshed patch.
The suppression (~2% benefit) is no longer retained; fresh post-revert measurements are
required. Filenames containing `rebuilt` do not mean rebuilt after that revert.

All values are CPU wall means in **ms**. Query means include reads, not query-only work.

### SF10 warm

| Query               | Parquet read | Vortex read | Parquet query | Vortex query |
| ------------------- | -----------: | ----------: | ------------: | -----------: |
| Q1                  |       69.955 |      34.139 |       101.245 |       65.529 |
| Q5 (prior warm run) |       53.956 |      22.541 |        58.808 |       27.388 |
| Q6                  |       38.466 |      18.568 |        40.432 |       20.598 |
| Q9 binaryop         |       74.063 |      24.907 |        81.133 |       32.197 |
| Q9 AST              |       73.632 |      25.100 |        81.005 |       32.216 |
| Q9 transform        |       73.807 |      25.079 |        80.927 |       32.238 |
| Q10 (noisy)         |       67.160 |      29.154 |        75.112 |       37.495 |

### SF10 cold

| Query        | Parquet read | Vortex read | Parquet query | Vortex query |
| ------------ | -----------: | ----------: | ------------: | -----------: |
| Q1           |      141.331 |      62.248 |       174.044 |       93.190 |
| Q5           |      113.263 |      40.002 |       120.876 |       45.064 |
| Q6           |       81.915 |      30.455 |        85.390 |       32.310 |
| Q9 binaryop  |      142.833 |      48.408 |       153.286 |       55.584 |
| Q9 AST       |            — |           — |       152.012 |       55.646 |
| Q9 transform |            — |           — |       152.408 |       55.706 |
| Q10          |      121.270 |      49.929 |       131.099 |       58.386 |

Q9 cold has one reported read comparison; the other rows report query engines only.
Artifacts under ignored `build/cudf-ndsh-build/`:

- Q1 warm/cold: `sf10-q1-rebuilt-warm-cold-pinned-release.json`.
- Q5: `sf10-q5-cold-pinned-release.json`.
- Q6: `sf10-q6-cold-pinned-release.json`.
- Q9: `sf10-q9-cold-pinned-release.json`.
- Q10: `sf10-q10-cold-pinned-release.json`.

### SF1 Q1 warm/cold

Artifact: `build/cudf-ndsh-build/sf1-q1-rebuilt-warm-cold-pinned-release.json` (ignored).

| Cache | Parquet read | Vortex read | Parquet query | Vortex query |
| ----- | -----------: | ----------: | ------------: | -----------: |
| Warm  |       14.011 |       6.540 |        18.373 |       10.876 |
| Cold  |       18.771 |       6.561 |        23.383 |       10.926 |

Vortex cold read/query noise is high: 10.7% / 6.7%. This is not a stable final matrix.

The goal of **≥2× for both end-to-end read and query at SF1/SF10, warm/cold, is not
met**. Q1 warm and SF10 cold query remain below target; Q6 warm query is near but below
2×; Q10 is marginal/noisy. No SF100 scaling until the matrices are stable.

## Profile evidence and safety

Existing full-read SQLite evidence records ~36.31 ms timed Vortex read: file reads
extend to 22.6 ms, first decode starts at 22.9 ms, HtoD transfers 1.62 GiB in 7.89 ms,
decode takes 7.25 ms, and final materialization 1.94 ms. This is read evidence, not a
full Q1 query profile; that profile has **not yet been captured**.

**Prefix future profiler launches with `env -u ANTHROPIC_API_KEY`** (before `nsys`)
and otherwise use a sanitized environment. Existing old profiles contain sensitive
environment metadata: **do not inspect or publish that metadata**, or share raw profiles
containing it. No old profile environment metadata was inspected for this checkpoint.

## Rejected experiments

Default kernel-event suppression, a fixed three-split cap, duplicate HtoD event removal,
and earlier StringDict enablement were all rejected/reverted. The cap regressed warm
read from 34.12 to 35.36 ms; no scheduling change from that experiment is retained.
No event optimization from these experiments is current.

## Correctness scope

Projected names/types/values match exactly. Independent CPU and synthetic cases cover
results, boundaries, joins, nulls, and empty results; generated CPU oracles target
non-null schemas. Q1 counts/quantity sums are exact; floating checks use `1e-10`
relative tolerance with an absolute floor of `1e-10`. Q9 follows the benchmark's
unrounded `SUM(amount)`, not the separate streaming SQL's two-decimal rounding.

Failing-before generator regressions cover correlated discount/quantity RNG, correlated
order year/month RNG, unordered price alignment, and truncated fractional supplier
scale factors. Fixes affect **all** NDS-H consumers, including Vortex OFF: regenerate
fixtures and old baselines. Other correlations remain; this is not full TPC-H conformance.

All benchmark JSON, logs, Nsight reports, and SQLite exports remain ignored. Follow-up:
complete the build/relinks with an explicitly chosen time budget, validate runtime, then
stabilize SF1/SF10 warm/cold matrices and capture a full Q1 query profile safely.
