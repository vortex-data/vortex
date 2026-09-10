# Validation

2026-09-10, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                         | Result                                               |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| Q1/Q6: Parquet/Vortex × projected read/query, SF0.01          | All 8 states passed                                  |
| Same-fixture CPU references: Q1 groups/aggregates, Q6 revenue | Passed                                               |
| Q1 cutoff/null-date/slices/empty and Q6 boundary/null checks  | Passed                                               |
| Compute Sanitizer: all 8 Q1/Q6 states                         | 0 errors, no skips                                   |
| Pinned cuDF: Q1/Q6, Vortex ON/OFF                             | Compile-only passed                                  |
| Earlier generator / adapter / CUDA FFI tests                  | 6 / 15 / 17 passed; generator/adapter memcheck clean |
| NDS-H / FFI CMake integration                                 | 8 / 13 tests passed                                  |
| Full pinned libcudf build                                     | Timed out after 600s; runtime unverified             |

Formatting, workspace Clippy, Ruff, and patch forward/reverse checks passed.

## SF0.01 supplemental runs

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` with matching
headers/libraries and locally compiled helpers carrying the generator fixes.
Current query/reference/I/O sources are used unchanged. **cuDF is Debug;
benchmark/Vortex are Release**, on a shared GPU—not pinned-upstream performance.

```sh
build/cudf-q6-prebuilt/build/NDSH_Q01_NVBENCH \
  --benchmark ndsh_q1_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001-q1.json
build/cudf-q6-prebuilt/build/NDSH_Q06_NVBENCH \
  --benchmark ndsh_q6_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001-q6-shared-io.json
```

| CPU wall mean     |  Parquet |   Vortex |
| ----------------- | -------: | -------: |
| Q1 projected read | 3.999 ms | 2.566 ms |
| Q1                | 6.915 ms | 5.597 ms |
| Q6 projected read | 2.586 ms | 1.478 ms |
| Q6                | 3.666 ms | 2.594 ms |

Three Q1 states hit the sampling limit; Vortex Q1 had about 13% noise. These
measurements establish execution, not reliable performance ratios.

Each benchmark generates one **60,170-row, 16-column** fixture shared by its two
formats. Q1 projects eight columns; Q6 four. File sizes are recorded in the JSON.

- **Q1:** 44,973 matched rows, four sorted groups; both GPU results match all eight
  CPU aggregates. Counts/quantity sums are exact; floating metrics use `1e-10`
  relative tolerance with an absolute floor of `1e-10`.
- **Q6:** 563 matches, revenue **592,346.90374810458**; both GPU results match CPU.
- Projected names/types/values match exactly. Independent synthetic cases verify
  cutoff inclusion, null-date exclusion, nonzero slices, empty results, and Q6's
  expected revenue **18**. CPU oracles target generated non-null schemas.

Warm-cache timing includes import, copies, concatenation, query work when selected,
destruction, and GPU completion. Writing/validation are excluded. RMM peaks exclude
Vortex. Q6 was rerun after extracting the shared local-file/projection helpers.

## Generator fixes and limits

Regressions failed before both fixes: shared discount/quantity RNG draws left Q6
empty, and unordered join output misaligned prices. Discount now uses a separate
fixed seed; retail prices use the existing part-key formula in lineitem order.
Per-row prices are checked against the referenced part on CPU. Both changes affect
all NDS-H consumers, even with Vortex OFF: regenerate fixtures and old baselines.
Other generator correlations remain; this is not full TPC-H generator conformance.

## Recheck and artifacts

Build/run instructions: patched `cpp/benchmarks/ndsh/VORTEX.md`. Host-specific
consumer: `build/cudf-q6-prebuilt`; `/home/ubuntu/cudf` remains untouched.

That directory contains `sf001-q1*.{json,log}`, `sf001-q6-shared-io*.{json,log}`,
`q1-shared-io-{normal,memcheck}-summary.json`, and exact commands/source hashes in
`q1-shared-io-commands.md` and `q1-shared-io-memcheck-commands.md`. Memcheck uses
`--tool memcheck --error-exitcode 99`, with one-second NVBench sampling limits;
its timings are not benchmarks. Pinned compiler evidence: `build/cudf-seeded-compile/`.

The local NVCC 13.1 dispatcher workaround passed the original failing translation
unit, but the full pinned build stopped at 241/515 on its 600s timeout. Pinned
linking/runtime, larger SFs, Nsight, and deployment remain unverified. nvCOMP
co-loading probes do not establish decompression compatibility across versions.
