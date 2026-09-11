# Validation

2026-09-11, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                       | Result                                   |
| ----------------------------------------------------------- | ---------------------------------------- |
| Q1/Q5/Q6/Q9/Q10: matched projected read/query, SF0.01       | All 28 states passed                     |
| Same-fixture CPU references and independent synthetic cases | Passed                                   |
| Compute Sanitizer: all 28 states                            | 0 errors, no skips                       |
| Pinned Q9/Q10 ON/OFF; changed generator/Q6 sources          | Compile-only passed                      |
| Generator tests, including order-date/supplier regressions  | 8 passed; memcheck clean                 |
| Preserved original Q10 Parquet-pushdown/write benchmark     | Passed at SF0.01                         |
| Earlier adapter / CUDA FFI tests                            | 15 / 17 passed; adapter memcheck clean   |
| NDS-H / FFI CMake integration                               | 8 / 13 tests passed                      |
| Full pinned libcudf build                                   | Timed out after 600s; runtime unverified |

C++ formatting, Python byte-compilation, offline CMake tests, and patch reverse checks passed.
Ruff and cmake-format were unavailable; no Rust source changed.

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

Build instructions: patched `cpp/benchmarks/ndsh/VORTEX.md`. Local consumer:
`build/cudf-q6-prebuilt`; `/home/ubuntu/cudf` remains untouched. Example final run:

```sh
build/cudf-q6-prebuilt/build/NDSH_Q10_NVBENCH \
  --benchmark ndsh_q10_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001-q10-date-seed-final.json
```

Final JSON there: `sf001-q{1,5,6,9}-date-seed.json`,
`sf001-q10-date-seed-final.json`, and corresponding `*-memcheck.json`. Memcheck prepends `compute-sanitizer --tool memcheck
--error-exitcode 99` and uses `--min-samples 1 --timeout 1`; its timings are not
benchmarks. The date failing-before run reported only 2–3 months per year; supplier
evidence is `supplier-scale-summary.json`. Pinned Q9/Q10 ON/OFF objects/logs and
current generator/Q6 compile manifests are in `build/cudf-seeded-compile/`.

The local NVCC13.1 workaround passed the original failing translation unit, but
pinned linking/runtime, larger SFs, and Nsight remain unverified. Next: **validate the
pinned runtime, then scale SF1 → SF10 → SF100**; see [PROGRESS.md](PROGRESS.md).
nvCOMP co-loading probes do not establish decompression compatibility across versions.
