# Validation

Checkpoint: 2026-09-14. Pinned cuDF:
`5339497a1a17d799687cbf189fb113411fb015ca`, Release (`-O3 -DNDEBUG`).
[Current state](PROGRESS.md) · [Setup and commands](README.md)

## Validation status

The minimal implementation is complete; the commit breakdown is in [PROGRESS.md](PROGRESS.md).
**Binaries remain stale: fresh minimal-source runtime, memcheck,
and performance have not been run.** Current validation was offline/compile-only,
with no CMake regeneration, full build, relinks, or GPU runs.

### Current minimal-source checks

| Check                                                           | Result                       |
| --------------------------------------------------------------- | ---------------------------- |
| Offline integration tests                                       | 17 passed                    |
| Offline Cargo-directory configure / CUDA architecture tests     | 10 / 2 passed                |
| clang-format, all five query files                              | Passed                       |
| Ruff lint / format                                              | Passed                       |
| Main patch: forward/cached and reverse apply checks             | Passed                       |
| Optional generator patch: forward apply to pristine pinned cuDF | Passed                       |
| Both patch application orders                                   | Identical trees              |
| Compile-only Q1/Q5/Q6/Q9/Q10, Vortex ON/OFF                     | 10/10 passed, no warnings    |
| Separate generator regression test compile                      | Passed; no runtime execution |

The main patch touches only `cpp/benchmarks/ndsh/` and one include hook in
`cpp/benchmarks/CMakeLists.txt`, with no generator changes or generator-test target.
Patch checks cover the final export, including documentation and Q9 duplicate-join checks.

Compile artifacts under ignored `build/cudf-ndsh-minimal-compile/`:

- `summary.md`, `results.json`: all ten query variants recompiled after the native-output
  lifetime restoration and Q9 duplicate-join reference change; prior results are retained.
- `ninja-compdb.json`, `commands.txt`, `commands.json`: original and executed commands.
- `generator-test-summary.md`, `generator-test-command.json`: standalone test compile.

Compiles reused the existing toolchain/flags, with scratch outputs and Vortex ON/OFF
variants; source and Ninja input hashes stayed unchanged. Compilation does not validate
runtime behavior. The earlier Rust pinned tests/fmt/Clippy below were not rerun because
there were no Rust changes this turn.

### Historical checks

Before the minimal-scope revision (not new-patch passes):

| Check                                                       | Result    |
| ----------------------------------------------------------- | --------- |
| `cargo nextest run -p vortex-cuda pinned`                   | 5 passed  |
| `cargo +nightly fmt --all`                                  | Passed    |
| `cargo clippy --all-targets --all-features`                 | Passed    |
| clang-format, 8 edited C++ files                            | Passed    |
| `python3 -B benchmarks/cudf-ndsh/test_build_integration.py` | 14 passed |
| `.venv/bin/ruff check` / `format --check`, integration test | Passed    |
| Refreshed patch: forward/cached and reverse apply checks    | Passed    |

Previous full cuDF Release build attempted (historical command, not a retry instruction):

```sh
cmake --build build/cudf-ndsh-build \
  --target NDSH_Q01_NVBENCH NDSH_Q05_NVBENCH NDSH_Q06_NVBENCH \
    NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH NDSH_VORTEX_IO_TEST -j4
```

**Timed out after 1200 s.** CMake regeneration expanded to 644, then 635 build steps;
progress reached 223/635. The Vortex Release archive built successfully, but the cuDF
benchmarks and adapter were **not relinked**. All five Release query targets had built
before the event-suppression revert, not from the minimal source. No broad automatic
retry; fresh runtime claims require relinked binaries and reported results.

### Earlier runtime and integration checks

Historical validation, not a fresh post-revert or minimal-patch run:

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
The earlier integration counts above are separate from the prior 14-test offline run.
Supplemental cuDF 26.08 Debug runs established execution, not pinned Release performance.

## Timing contract

The approved minimal scope uses the **original pinned cuDF generator** and identical
logical fixtures across formats, matching scan projections/post-read predicates, and
generic cuDF execution. Q1 uses original `SUM`/native `MEAN`/`COUNT`; Q5/Q9/Q10 table
reads are sequential. No hand-fitted kernels. Original native Parquet-pushdown
benchmarks remain separate and retain their input/intermediate lifetimes through output. The default-OFF adapter and Vortex-internal cacheable
pinned staging, 16M-row blocks, and 8 GiB pool remain. Details: [README.md](README.md).

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

## Prior timings — HISTORICAL, not minimal-patch baselines

**These measurements used altered generator/performance code and event suppression.**
They are not valid baselines for the new minimal patch. Restoring original query/read
code and the pinned generator requires regenerating both formats' fixtures and fresh
baselines labeled by generator and match counts. The optional generator-fixed dataset
also needs separately labeled fixtures and baselines. Evidence tables are preserved
below, not revalidated. Filenames containing `rebuilt` do not imply current binaries.

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
met**. In these historical runs, Q1 warm and SF10 cold query were below target; Q6 warm
query was near but below 2×; Q10 was marginal/noisy. These observations do not describe
minimal-patch performance. No SF100 scaling until fresh nondegenerate matrices are stable.

## Profile evidence and safety

Historical full-read SQLite evidence records ~36.31 ms timed Vortex read: file reads
extend to 22.6 ms, first decode starts at 22.9 ms, HtoD transfers 1.62 GiB in 7.89 ms,
decode takes 7.25 ms, and final materialization 1.94 ms. This is historical read evidence,
not a minimal-patch baseline or full Q1 query profile; that profile is **still absent**.

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

Exact projected names/types/values and independent CPU result checks remain **even for
zero matches**. The original pinned generator can yield empty/degenerate Q6/Q10 and
low-SF supplier joins; generated results need not be nonempty. Explicit zero match
counts disclose degenerate queries, which are not meaningful full-query performance
evidence. Synthetic **nonempty** cases remain alongside boundary, join, null, and
empty-result tests; generated CPU oracles target non-null schemas.
Q6 checks zero-match `SUM` is NULL and reports revenue as the string `"NULL"`, not zero.
Its CPU reference boundary/sliced/float32 test migrated to `q06.cpp` and was expanded
with no-match/empty GPU cases, so generator separation loses no main-query test.
These cases compiled but have not been run on the minimal source.
Q1 counts/quantity sums are exact; floating checks use `1e-10` relative tolerance with
an absolute floor of `1e-10`. Q9 follows the benchmark's unrounded `SUM(amount)`, not
the separate streaming SQL's two-decimal rounding. Its CPU reference preserves duplicate
`partsupp` join multiplicity instead of assuming composite-key uniqueness. Handwritten
cases cover different costs for matching duplicates, unmatched duplicates, and empty
inputs; the duplicate case expects seven matches and profits of 190/170/130 for
ALPHA-1994/ALPHA-1996/ZULU-1995. These GPU cases compiled but have not run.
Runtime correctness is not newly validated by the compile-only passes.

Optional `benchmarks/cudf-ndsh/generator-fixes.patch` preserves four independent fixes
and its own regression test across seven files: correlated discount/quantity RNG,
correlated order year/month RNG, unordered price alignment, and truncated fractional
supplier scale factors. The patch is separately reviewable and independently applicable
to pinned cuDF, **not bundled or automatically applied** with `upstream.patch`.
Both application orders were checked and produce identical trees.
The main patch has no generator edits/tests or `NDSH_DATA_GENERATOR_TEST` target;
only the optional patch adds that target. Applying it affects all NDS-H consumers,
including Vortex OFF: regenerate both formats' fixtures, label the dataset, and collect
separate baselines. Earlier generator-test results above are historical. Other
correlations remain; neither dataset establishes full TPC-H conformance.

All benchmark JSON, logs, Nsight reports, and SQLite exports remain ignored. Follow-up:
explicitly bounded relinking/runtime/memcheck validation (no broad automatic build retry),
then regenerated fixtures, fresh labeled SF1/SF10 warm/cold baselines, and a safely
captured full Q1 query profile.
