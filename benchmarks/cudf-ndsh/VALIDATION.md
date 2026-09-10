# Validation

2026-09-10, Linux aarch64 / GH200 (SM90), CUDA 13.1.115.

| Check                                                      | Result                                            |
| ---------------------------------------------------------- | ------------------------------------------------- |
| Adapter, matching prebuilt cuDF 26.08, GCC 13.3            | 15 tests passed                                   |
| Actual Q6/read, Parquet/Vortex, SF0.01                     | All 4 states passed                               |
| Compute Sanitizer: adapter and all 4 Q6/read states        | 0 errors                                          |
| Pinned cuDF: Q6 ON/OFF, utilities, adapter, adapter tests  | Compile-only passed                               |
| CUDA FFI                                                   | 17 tests passed; 6 CPU projection cases rechecked |
| Scoped/workspace Clippy, Rust/C++ formatting, FFI doctests | Passed                                            |
| NDS-H / FFI CMake integration                              | 8 / 13 tests passed                               |
| Patch forward/reverse application, CMake formatting, Ruff  | Passed                                            |
| Fresh pinned libcudf build                                 | Timed out after 600s; runtime unverified          |

Adapter coverage includes nulls, decimals, empty/sliced/batched tables, reordered
projection, invalid names, ownership, and stream cleanup. A guarded segment-source
test proves projection does not request unselected column segments. Previous
prerequisite checks passed: decimal/layout/Arrow tests (95/12/221), doctests
(74 passed, 21 ignored), and adapter memcheck.

## SF0.01 supplemental run

Uses cuDF 26.08 at `e4b0646588790a00054e4dfa65a4eaa9ba6aa609` with matching
headers/libraries, the current adapter/Q6 sources, and a locally compiled matching
generator/helpers. **cuDF is Debug; benchmark/Vortex are Release.** The shared GH200
had roughly 15 GiB free of 96 GiB. This is not pinned-upstream performance evidence.

```sh
build/cudf-q6-prebuilt/build/NDSH_Q06_NVBENCH \
  --benchmark ndsh_q6_local_warm --axis scale_factor=0.01 \
  --min-samples 3 --timeout 3 --json build/cudf-q6-prebuilt/sf001.json
```

| CPU wall mean        |  Parquet |   Vortex |
| -------------------- | -------: | -------: |
| Projected read       | 2.746 ms | 1.512 ms |
| Q6, **empty result** | 3.549 ms | 2.441 ms |

Both files contain the same 60,170 rows and all 16 columns; reads select four.
File sizes: 3.254 MiB Parquet / 3.902 MiB Vortex. Warm-cache timing includes import,
copies, concatenation, query work when selected, destruction, and GPU completion;
fixture writing and correctness checks are excluded. RMM peaks exclude Vortex.

Exact projected data matches the generated input. Revenue matches too, but is
**NULL**: shared RNG draws produce `quantity = floor(500 * discount) + 1`, so
Q6's discount/quantity predicates cannot both pass. CPU RNG replay and SQLite
independently confirmed all 60,170 rows; raising SF will not fix this correlation.
Pinned generator source retains the logic, but its runtime is untested. A separate
boundary/null fixture returns the independently expected revenue **18**. No data or
predicates were changed; representative Q6 needs a separately justified generator fix.

## Recheck and artifacts

Build/run instructions are in patched `cpp/benchmarks/ndsh/VORTEX.md`.
Host-specific consumers live in `build/cudf-q6-prebuilt` and
`build/cudf-vortex-roundtrip`; `/home/ubuntu/cudf` remains untouched.

```sh
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build/cudf-q6-prebuilt/build/NDSH_Q06_NVBENCH \
  --benchmark ndsh_q6_local_warm --axis scale_factor=0.01 --min-samples 1 --timeout 1
```

Local evidence: `build/cudf-q6-prebuilt/{sf001.json,final-pinned-compile.log,
runtime-validation-sf001-memcheck.log}`, `build/cudf-q6-data-diagnosis/REPORT.md`,
and `build/validation-projection-final-20260910/`. Sanitizer timings are not benchmarks.

## Remaining limits

A minimal repro isolated an NVCC 13.1 dispatcher miscompilation; a local-only
`--pre-include=cudf/detail/utilities/dispatchers.hpp` workaround passed the failing
translation unit. The full pinned build then reached 241/515 before its 600s timeout;
no libcudf was produced. Pinned linking/runtime, Q1, larger SFs, Nsight, and deployment
remain unverified. Co-loading Vortex nvCOMP 5.1 with prebuilt cuDF's 5.2 passed only
limited API probes, not decompression compatibility; pinned cuDF's 5.3 is untested.
