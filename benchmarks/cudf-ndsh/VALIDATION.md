# Validation

2026-09-10, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                           | Result                                   |
| --------------------------------------------------------------- | ---------------------------------------- |
| Q1/Q5/Q6: Parquet/Vortex × projected read/query, SF0.01         | All 12 states passed                     |
| Same-fixture CPU references and independent synthetic cases     | Passed                                   |
| Compute Sanitizer: all 12 states                                | 0 errors, no skips                       |
| Pinned Q5 ON/OFF; Q1/Q6 ON; changed utilities/generator sources | Compile-only passed                      |
| Generator tests, including fractional supplier-key regression   | 7 passed; memcheck clean                 |
| Earlier adapter / CUDA FFI tests                                | 15 / 17 passed; adapter memcheck clean   |
| NDS-H / FFI CMake integration                                   | 8 / 13 tests passed                      |
| Full pinned libcudf build                                       | Timed out after 600s; runtime unverified |

C++/CMake/Rust formatting, workspace Clippy, Ruff, and patch forward/reverse checks passed.

## SF0.01 supplemental runs

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` with matching
headers/libraries and locally compiled helpers carrying the generator fixes.
Current query/reference/I/O sources are used unchanged. **cuDF is Debug;
benchmark/Vortex are Release**, on a shared GPU—not pinned-upstream performance.

| CPU wall mean               |   Parquet |    Vortex |
| --------------------------- | --------: | --------: |
| Q1 projected read           |  4.058 ms |  2.553 ms |
| Q1                          |  6.916 ms |  5.486 ms |
| Q5 six-table projected read | 13.037 ms |  5.984 ms |
| Q5                          | 18.090 ms | 11.366 ms |
| Q6 projected read           |  2.656 ms |  1.492 ms |
| Q6                          |  3.677 ms |  2.568 ms |

NVBench sampling-limit warnings occurred; these runs establish execution, not
reliable performance ratios. No outer command timed out in these supplemental runs.

- **Q1:** 60,170 input rows → 44,973 matches/four sorted groups; all eight aggregates
  match CPU. Counts/quantity sums are exact; floating metrics use `1e-10` relative
  tolerance with an absolute floor of `1e-10`.
- **Q5:** six full input tables, 76,800 total rows → 35 matches/four countries.
  Both GPU results match CPU country revenues within the same tolerance and sort
  descending. The independent synthetic fixture has five matches: ALPHA 150, ZULU 250;
  both CPU and GPU are checked, with separate null-date/empty-result cases.
- **Q6:** 60,170 input rows → 563 matches, revenue **592,346.90374810458**;
  both GPU results match CPU. Independent boundary/null fixture expects **18**.
- All projected names/types/values match exactly. CPU oracles target non-null
  generated schemas. Warm-cache timing includes cleanup/device completion but
  excludes writing/validation. RMM peaks exclude Vortex allocations.

## Generator fixes and limits

Failing-before regressions established three defects: shared discount/quantity RNG
left Q6 empty; unordered join output misaligned prices; integer scale-factor arguments
produced zero supplier keys at SF0.01/0.1. Fixes use a separate fixed discount seed,
row-aligned part-key prices, and `double` supplier scale arguments. Tests check every
lineitem price and both supplier-key formulas/ranges at SF0.01/0.1/1.

These affect **all** NDS-H consumers, even with Vortex OFF: regenerate fixtures and
old baselines. Other correlations remain; this is not full TPC-H generator conformance.

## Commands and local artifacts

Build instructions: patched `cpp/benchmarks/ndsh/VORTEX.md`. Local consumer:
`build/cudf-q6-prebuilt`; `/home/ubuntu/cudf` remains untouched. Example final run:

```sh
build/cudf-q6-prebuilt/build/NDSH_Q05_NVBENCH \
  --benchmark ndsh_q5_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001-q5-golden.json
```

Final logs/JSON there: `sf001-q5-golden*`, `sf001-q1-q5-helpers*`, and
`sf001-q6-q5-helpers*`. Memcheck prepends `compute-sanitizer --tool memcheck
--error-exitcode 99` and uses `--min-samples 1 --timeout 1`; its timings are not
benchmarks. Supplier failing-before evidence: `supplier-scale-summary.json`.
Pinned compile commands/hashes/results: `build/cudf-seeded-compile/q5-final-*`
(Q5 ON/OFF) and `q5-*` (shared helpers and Q1/Q6).

The local NVCC13.1 workaround passed the original failing translation unit, but
pinned linking/runtime, larger SFs, and Nsight remain unverified. Next: **Q9/Q10 at
SF0.01 before scaling**; see [PROGRESS.md](PROGRESS.md). nvCOMP co-loading probes
do not establish decompression compatibility across versions.
