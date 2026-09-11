# Validation

2026-09-11, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                       | Result                                 |
| ----------------------------------------------------------- | -------------------------------------- |
| Q1/Q5/Q6/Q9/Q10: matched projected read/query, SF0.01       | All 28 states passed                   |
| Pinned Release SF1 matched projected read/query             | All 28 states passed                   |
| Pinned Release SF10 matched projected read/query            | All 20 selected states passed          |
| Same-fixture CPU references and independent synthetic cases | Passed                                 |
| Compute Sanitizer: all 28 SF0.01 states                     | 0 errors, no skips                     |
| Pinned Release libcudf and five benchmark executables       | Built and runtime-validated            |
| Generator tests, including order-date/supplier regressions  | 8 passed; memcheck clean               |
| Preserved original Q10 Parquet-pushdown/write benchmark     | Passed at SF0.01                       |
| Pinned adapter / current CUDA FFI tests                     | 15 / 20 passed; adapter memcheck clean |
| NDS-H / FFI CMake integration                               | 8 / 13 tests passed                    |
| Full pinned libcudf Release build                           | Passed with NVCC 13.1 workaround       |

Rust/C++ formatting, workspace Clippy, CUDA FFI tests, Python byte-compilation, offline
CMake tests, and patch reverse checks passed. Ruff and cmake-format were unavailable.

## SF0.01 supplemental runs

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` with matching
headers/libraries and locally compiled helpers carrying the generator fixes.
Current query/reference/I/O logic is used; ignored Q9/Q10 source copies only adapt
26.08's transform and stream accessor APIs. **cuDF is Debug; benchmark/Vortex are
Release**, on a shared GPU—not pinned-upstream performance.

| CPU wall mean                       |   Parquet |    Vortex |
| ----------------------------------- | --------: | --------: |
| Q1 projected read                   |  3.972 ms |  2.405 ms |
| Q1                                  |  6.818 ms |  5.431 ms |
| Q5 six-table projected read         | 12.829 ms |  5.839 ms |
| Q5                                  | 18.177 ms | 11.613 ms |
| Q6 projected read                   |  2.722 ms |  1.506 ms |
| Q6                                  |  3.785 ms |  2.615 ms |
| Q9 six-table projected read, binary | 13.646 ms |  6.417 ms |
| Q9, binary amount engine            | 19.304 ms | 11.673 ms |
| Q10 four-table projected read       | 12.136 ms |  7.264 ms |
| Q10                                 | 18.113 ms | 13.412 ms |

NVBench sampling-limit warnings occurred; these runs establish execution, not
reliable performance ratios. No outer command timed out in these supplemental runs.

- **Q1:** 60,170 input rows → 52,574 matches/four sorted groups; all eight aggregates
  match CPU. Counts/quantity sums are exact; floating metrics use `1e-10` relative
  tolerance with an absolute floor of `1e-10`.
- **Q5:** six full input tables, 76,800 total rows → 35 matches/four countries.
  Both GPU results match CPU country revenues within the same tolerance and sort
  descending. The independent synthetic fixture has five matches: ALPHA 150, ZULU 250;
  both CPU and GPU are checked, with separate null-date/empty-result cases.
- **Q6:** 60,170 input rows → 554 matches, revenue **569,510.47908567684**;
  both GPU results match CPU. Independent boundary/null fixture expects **18**.
- **Q9:** six full tables, 85,295 total rows → 538 matches/119 nation-year groups.
  Binary-op, AST, and transform outputs all match CPU and sort nation ascending/year
  descending. Synthetic groups are ALPHA-1996 90, ALPHA-1994 110, and ZULU-1995 130.
  The oracle intentionally follows the existing benchmark's unrounded `SUM(amount)`;
  the separate streaming SQL's `round(..., 2)` is not claimed here.
- **Q10:** four full tables, 76,695 total rows → 535 matches/70 customers. Both GPU
  results match all CPU customer attributes and revenues, sorted descending. Synthetic
  cases cover date boundaries, null dates/flags, missing joins, and empty results.
- All projected names/types/values match exactly. CPU oracles target non-null generated
  schemas. Warm-cache timing includes cleanup/device completion but excludes writing/
  validation. RMM peaks exclude Vortex allocations.

## Pinned cuDF Release scaling

Pinned cuDF commit `5339497a1a17d799687cbf189fb113411fb015ca` was built and run with
`CMAKE_BUILD_TYPE=Release`, `CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG`, and
`CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG`. NVBench timed complete warm-cache reads/query
execution and cleanup; fixture writing/checks remain outside timing. SF1 benchmark
memcheck was not rerun after the performance changes.

### SF1

All 28 read/query/engine states passed.

| CPU wall mean                       |   Parquet |    Vortex | Vortex speedup |
| ----------------------------------- | --------: | --------: | -------------: |
| Q1 projected read                   | 13.871 ms |  6.807 ms |          2.04× |
| Q1                                  | 19.003 ms | 11.766 ms |          1.62× |
| Q5 six-table projected read         | 17.577 ms | 10.786 ms |          1.63× |
| Q5                                  | 20.402 ms | 13.704 ms |          1.49× |
| Q6 projected read                   |  7.896 ms |  4.775 ms |          1.65× |
| Q6                                  |  8.506 ms |  5.264 ms |          1.62× |
| Q9 six-table projected read, binary | 26.576 ms | 11.281 ms |          2.36× |
| Q9, binary amount engine            | 29.784 ms | 14.359 ms |          2.07× |
| Q10 four-table projected read       | 26.121 ms | 12.579 ms |          2.08× |
| Q10                                 | 30.047 ms | 16.313 ms |          1.84× |

Q9 AST and transform states also passed and differ by less than 0.2 ms from the binary-op
Vortex query. Q6 revenue is **58,942,077.17243248**.

### SF10

All 20 selected states passed; Q9 used the representative binary-op amount engine.

| CPU wall mean                       |    Parquet |    Vortex | Vortex speedup |
| ----------------------------------- | ---------: | --------: | -------------: |
| Q1 projected read                   |  70.544 ms | 45.607 ms |          1.55× |
| Q1                                  | 108.095 ms | 82.630 ms |          1.31× |
| Q5 six-table projected read         |  61.122 ms | 33.460 ms |          1.83× |
| Q5                                  |  66.068 ms | 37.841 ms |          1.75× |
| Q6 projected read                   |  38.971 ms | 21.062 ms |          1.85× |
| Q6                                  |  40.820 ms | 23.110 ms |          1.77× |
| Q9 six-table projected read, binary |  81.313 ms | 45.550 ms |          1.79× |
| Q9, binary amount engine            |  88.411 ms | 53.136 ms |          1.66× |
| Q10 four-table projected read       |  74.819 ms | 49.808 ms |          1.50× |
| Q10                                 |  82.863 ms | 57.635 ms |          1.44× |

| Query | Prior 1M → final Vortex read | Improvement | Prior 1M → final Vortex query | Improvement |
| ----- | ---------------------------: | ----------: | ----------------------------: | ----------: |
| Q1    |          106.752 → 45.607 ms |       2.34× |           143.610 → 82.630 ms |       1.74× |
| Q5    |           53.561 → 33.460 ms |       1.60× |            58.392 → 37.841 ms |       1.54× |
| Q6    |           34.322 → 21.062 ms |       1.63× |            36.750 → 23.110 ms |       1.59× |
| Q9    |           63.443 → 45.550 ms |       1.39× |            69.582 → 53.136 ms |       1.31× |
| Q10   |           90.973 → 49.808 ms |       1.83× |            98.146 → 57.635 ms |       1.70× |

| Query | Fixture rows |    Matches | Result cardinality | Parquet / Vortex size | Max Parquet / Vortex RMM |
| ----- | -----------: | ---------: | -----------------: | --------------------: | -----------------------: |
| Q1    |   59,990,397 | 52,600,853 |           4 groups |     3.720 / 4.138 GiB |        4.801 / 4.788 GiB |
| Q5    |   76,590,427 |     51,468 |        5 countries |     4.910 / 5.413 GiB |        3.127 / 2.312 GiB |
| Q6    |   59,990,397 |    568,275 |          1 revenue |     3.720 / 4.138 GiB |        2.589 / 1.570 GiB |
| Q9    |   85,090,422 |    646,134 |         175 groups |     5.840 / 6.342 GiB |        3.497 / 3.078 GiB |
| Q10   |   76,490,422 |    534,901 |   69,959 customers |     4.899 / 5.402 GiB |        3.280 / 2.251 GiB |

All formats agree with independent CPU references. RMM peaks exclude Vortex allocations;
separate Vortex peak accounting remains required before SF100.

### Profile-guided I/O optimization

The original aligned writer used 1,048,576-row blocks. A focused Q1 SF10 Nsight Systems
trace measured 58 delivered batches, 46,061 CUDA API calls, 1,108 kernels, 1,596 copies,
and only 32% GPU utilization. HtoD, decode, and final materialization ran as serialized
phases, with 89.6 ms GPU idle in a 131.9 ms profiled interval.

Increasing physical/read blocks to 16,777,216 rows reduced Q1 to four batches, 3,472 API
calls, 80 kernels, and 152 copies. GPU idle fell to 24.4 ms. Retaining 2 GiB in CUDA's
default pool then reduced aggregate `cuMemAllocAsync` time from 28.0 to 17.9 ms. Q1's
benchmark median fell monotonically from 106.518 ms (1M) to 62.481 (4M), 54.850 (8M),
51.041 (16M), and 45.967 ms with pool retention.

Q10 exposed a separate fragmentation source: `orders.o_custkey` exceeded the layout
dictionary's 65,535-value limit, splitting one explicit block into 214 dictionary runs.
Disabling outer layout dictionaries only for explicit CUDA row blocks reduced orders to
one batch, event calls from 21,957 to 142, kernels from 860 to three, and its profiled read
from 46.9 to 6.2 ms. Overall Q10 profile time fell 92.9 → 50.4 ms. The other table reads
kept the same batch/kernel/copy counts. A 140K-row/70K-key regression verifies that an
explicit high-cardinality row block stays whole.

`batch_rows` remains a maximum using layout-aware sub-splitting, preserving physical
boundaries. The reader retains Arrow Device arrays and borrowed cuDF views until one final
owning copy/concatenation. Adapter NVTX ranges identify scan open, `get_next`, Arrow Device
import, materialization, consumer synchronization, and release. The 15-test adapter suite
passes normally and under its earlier memcheck run; current CUDA FFI tests pass 20/20.

## Generator fixes and limits

Failing-before regressions established four defects: shared discount/quantity RNG
left Q6 empty; shared order year/month RNG left each year with only 2–3 months and Q10
empty; unordered join output misaligned prices; integer scale-factor arguments produced
zero supplier keys at SF0.01/0.1. Fixes use fixed independent discount and order
month/day streams, row-aligned part-key prices, and `double` supplier scale arguments.
Month/day remain paired to generate valid dates. Tests cover all 84 year/month pairs at
SF0.01, every lineitem price, and both supplier-key formulas/ranges at SF0.01/0.1/1.

These affect **all** NDS-H consumers, even with Vortex OFF: regenerate fixtures and
old baselines. Other correlations remain; this is not full TPC-H generator conformance.

## Commands and local artifacts

Build instructions: patched `cpp/benchmarks/ndsh/VORTEX.md`. `/home/ubuntu/cudf`
remains untouched. Example pinned Release run:

```sh
build/cudf-ndsh-build/benchmarks/NDSH_Q10_NVBENCH \
  --benchmark ndsh_q10_local_warm --axis scale_factor=10 \
  --min-samples 3 --timeout 30 \
  --json build/cudf-ndsh-build/sf10-q10-optimized-pinned-release.json
```

Final pinned JSON:

- SF1: `sf1-q{1,5,6,9,10}-optimized-pinned-release.json`.
- SF10: `sf10-q{1,5,6}-optimized-pinned-release.json`,
  `sf10-q9-binary-optimized-pinned-release.json`, and
  `sf10-q10-optimized-pinned-release.json`.

Focused Nsight reports/SQLite exports are
`sf10-q1-vortex-read-full-profile*`,
`sf10-q1-block16m-vortex-read-full-profile*`,
`sf10-q1-block16m-pool2g-vortex-read-full-profile*`,
`sf10-q10-block16m-pool2g-vortex-read-full-profile*`, and
`sf10-q10-optimized-vortex-read-full-profile*` under `build/cudf-ndsh-build/`.
These artifacts are ignored. Memcheck prepends
`compute-sanitizer --tool memcheck --error-exitcode 99`; its timings are not benchmarks.

The NVCC13.1 workaround passed the full pinned Release build and runtime validation.
Next: **SF100 memory validation and matched-cache profiles**; see
[PROGRESS.md](PROGRESS.md). nvCOMP co-loading probes do not establish decompression
compatibility across versions.
