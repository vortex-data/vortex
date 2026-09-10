# Validation

2026-09-10, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                             | Result                                       |
| ----------------------------------------------------------------- | -------------------------------------------- |
| Generator: numeric seeds, CPU boundary/slice, price alignment, Q6 | 6 tests passed                               |
| Same-fixture CPU vs GPU Q6/read, Parquet/Vortex, SF0.01           | All 4 states passed                          |
| Compute Sanitizer: generator tests and all 4 Q6/read states       | 0 errors                                     |
| Pinned cuDF: generator C++/CUDA, test, Q6 ON/OFF                  | Compile-only passed                          |
| Earlier adapter / CUDA FFI checks                                 | 15 / 17 tests passed; adapter memcheck clean |
| NDS-H / FFI CMake integration                                     | 8 / 13 tests passed                          |
| Fresh pinned libcudf build                                        | Timed out after 600s; runtime unverified     |

Formatting, workspace Clippy, Ruff, and patch forward/reverse checks passed.

## SF0.01 supplemental run

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` with matching
headers/libraries and locally compiled helpers carrying the same generator fixes. Current
Q6/test/reference sources are used unchanged. **cuDF is Debug; benchmark/Vortex
are Release**, on a shared GPU. This is not pinned-upstream performance evidence.

```sh
build/cudf-q6-prebuilt/build/NDSH_Q06_NVBENCH \
  --benchmark ndsh_q6_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001-price-aligned.json
```

| CPU wall mean  |  Parquet |   Vortex |
| -------------- | -------: | -------: |
| Projected read | 2.628 ms | 1.500 ms |
| Q6             | 3.686 ms | 2.577 ms |

Both files contain the same **60,170 rows** and all 16 columns; reads select four.
Sizes: **3,412,340 bytes Parquet / 4,037,432 bytes Vortex**. Q6 has **563 matches**;
both GPU revenues agree with the same-table CPU reference **592,346.90374810458**
within `1e-10` relative tolerance. Projected schemas/values match exactly.

Warm-cache timing includes import, copies, concatenation, query work when selected,
destruction, and GPU completion; writing and validation are excluded. RMM peaks
exclude Vortex. Memcheck independently passed all four cases on its own fixture.

## Generator regression and limits

Before the discount-seed fix, all four numeric tests passed but generated Q6 failed
with **zero matches**. The old shared draws satisfy `quantity = floor(500*discount)+1`;
CPU RNG replay and SQLite confirmed the defect. A fixed, non-query-tuned seed now
separates discount's draws; default-seeded callers retain their numeric sequences.
All NDS-H lineitem consumers get changed discount/dependent values, even with Vortex
OFF. Regenerate fixtures; do not compare new timings to the old empty-Q6 workload.

The price-alignment regression also failed before its fix. Retail prices now use
the existing part-key formula in lineitem order, avoiding unordered join output.
Every generated extended price is checked against quantity × the referenced part's
retail price on CPU. Q6 counts/revenue repeated exactly across five tested fixtures.
Regenerate fixtures after either fix; old revenues/timings are not comparable.
Other default-seeded columns remain correlated: this is not full TPC-H conformance.
CPU boundary/slice and GPU boundary/null fixtures independently expect revenue **18**.

## Recheck and artifacts

Build/run instructions: patched `cpp/benchmarks/ndsh/VORTEX.md`. Host-specific
consumers: `build/cudf-q6-prebuilt` and `build/cudf-vortex-roundtrip`.
`/home/ubuntu/cudf` remains untouched.

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cudf-q6-prebuilt/build/NDSH_Q06_NVBENCH \
  --benchmark ndsh_q6_local_warm --axis scale_factor=0.01 --min-samples 1 --timeout 1
```

Evidence in `build/cudf-q6-prebuilt`: `sf001-price-aligned*.{json,log}`,
`price-order-before-test.log`, `price-aligned-test-{1,2,memcheck}.log`,
`price-aligned-final-summary.json`, and `price-aligned-commands.md`.
Pinned compiler commands/hashes: `build/cudf-seeded-compile/`.
Sanitizer timings are not benchmarks.

The local NVCC 13.1 dispatcher workaround passed the original failing translation
unit, but the full pinned build stopped at 241/515 on its 600s timeout. Pinned
linking/runtime, Q1, larger SFs, Nsight, and deployment remain unverified. nvCOMP
co-loading probes do not establish decompression compatibility across versions.
