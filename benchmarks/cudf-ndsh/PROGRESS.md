# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-14 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Source and commit checkpoint

The minimal-scope revision is recorded in these commits; earlier history is unchanged:

- `ad318457b`: configurable CMake Cargo target directory, preserved on clean, with
  typed/untyped path validation and offline tests.
- `a56063f1a`: standalone optional generator-fix patch and regression target.
- `8ff127aaa`: query-neutral cuDF integration, original generator by default, native
  query/output lifetimes, Q6/Q9 correctness checks, and offline scope guards.

### Earlier implementation

- Prior signed-off series: through `e81905f44`.
- `9b6df1978`: cacheable pinned CUDA staging.
- `59a7ea66a`: generic performance-only `upstream.patch` changes — shared Q1 sum/count
  groupby with averages afterward, concurrent independent table reads in Q5/Q9/Q10
  for both formats, and 8 GiB Vortex CUDA pool retention.
- `cd3192a02`: cold-cache patch and tests — renamed
  `ndsh_q{1,5,6,9,10}_local`, `cache=warm/cold`, SF10 in the default scale-factor axis,
  per-callback eviction/residency verification, cold Vortex direct data reads, and
  offline integration coverage.
- Before the minimal-scope revision, the cumulative patch was refreshed against cuDF
  `5339497a1a17d799687cbf189fb113411fb015ca`; forward/cached and reverse apply checks
  and all 14 offline tests passed. These are prior results, not new-scope validation.

### Completed minimal scope

- The refreshed main patch touches only `cpp/benchmarks/ndsh/` plus one include hook
  in `cpp/benchmarks/CMakeLists.txt`; no generator changes or generator-test target.
- Original Q1 `SUM`/native `MEAN`/`COUNT` and sequential Q5/Q9/Q10 table reads are
  restored for both formats, superseding those parts of `59a7ea66a`. No hand-fitted query kernels
  or `q1_fused` path. Original input/intermediate lifetimes through native Parquet output
  are preserved. No event optimization is retained.
- Retained: the default-OFF adapter, warm/cold controls, and Vortex-internal improvements:
  cacheable pinned staging, 16M-row blocks, 8 GiB pool retention, and final owning
  materialization. Both formats keep identical scan projections and post-read predicates;
  original native Parquet-pushdown benchmarks stay separate.
- Default comparisons use the **original pinned cuDF generator**, with identical
  logical fixtures across formats. Optional `benchmarks/cudf-ndsh/generator-fixes.patch`
  preserves four independent fixes and its own regression test across seven files.
  It is separately reviewable and independently applicable to the pin, not bundled or
  automatic; both patch orders produce identical trees. Only it adds
  `NDSH_DATA_GENERATOR_TEST`. Applying it requires regenerating both formats' fixtures
  and labeling the dataset.
- Original data may yield empty/degenerate Q6/Q10 and low-SF supplier joins.
  Projection/value and independent CPU checks remain even for zero matches, plus
  synthetic nonempty tests. Explicit zero match counts disclose queries that are
  **not meaningful full-query performance evidence**.
- Q6 checks zero-match `SUM` is NULL and reports revenue `"NULL"`, not zero. Its CPU
  reference boundary/sliced/float32 test moved to `q06.cpp`, with no-match/empty GPU
  cases added; generator separation loses no main-query tests. New GPU cases are unrun.
- Q9's CPU reference preserves duplicate `partsupp` join multiplicity, which the original
  fractional-SF generator can produce. Handwritten cases cover matching duplicates with
  different costs, unmatched duplicates, and empty inputs; GPU execution is still pending.

See [README.md](README.md) for the I/O/timing contract and
[VALIDATION.md](VALIDATION.md) for preserved historical evidence.

## Build and evidence status

**Binaries are stale; no fresh minimal-source runtime, memcheck, or performance runs.**
The previous full Release build targeting all five queries plus `NDSH_VORTEX_IO_TEST`
timed out after 1200 s.
CMake regeneration expanded to 644, then 635 steps; the build reached 223/635.
The Vortex Release archive built successfully, but cuDF benchmarks and the adapter
were **not relinked**. Do not retry automatically.

Current checks passed: **17 offline tests**, clang-format on all five query files,
Ruff lint/format, main-patch forward/cached and reverse checks, optional-patch forward
application to pristine pinned cuDF, and identical trees for both patch application orders.
Compile-only Q1/Q5/Q6/Q9/Q10 with Vortex ON/OFF passed **10/10 without warnings**.
All ten variants were recompiled after the native-output lifetime restoration and Q9
reference changes. The standalone generator regression test compiled too, but was not run.
The Cargo-directory change passed 10 configure and 2 CUDA-architecture offline tests;
its untyped relative-path regression failed before the validation fix.
Artifacts: ignored `build/cudf-ndsh-minimal-compile/`; see [VALIDATION.md](VALIDATION.md).
No CMake regeneration, full build, relinks, or GPU runs this turn. Earlier 5 pinned Rust
tests, nightly fmt, and all-target/all-feature Clippy passed; no Rust changes, so not rerun.

Prior timings used altered generator/performance code and event suppression:
**HISTORICAL, not valid baselines for the minimal patch**. Source/data changes require
regenerated fixtures and fresh baselines; old evidence tables remain in VALIDATION.md.
The ≥2× goal for both end-to-end reads and queries at SF1/SF10, warm/cold, is **not met**.
The existing full-read profile is not a full Q1 query profile; the latter is still absent.

## Next steps

1. Choose an explicit bounded build/relink and runtime/memcheck validation window;
   no broad automatic retry. The final patch is exported and apply-checked.
2. After validation/relinking, regenerate both formats' fixtures using the original
   pinned generator and collect fresh SF1/SF10 warm/cold read/query baselines. Label
   the generator and match counts; exclude degenerate queries from full-query
   performance claims. Optional generator-fixed datasets need separate labels and
   baselines. Run GPU benchmarks sequentially.
3. Capture a full Q1 query profile using the
   [profiling safety guard](VALIDATION.md#profile-evidence-and-safety).
   Defer SF100 until matrices are stable, with separate Vortex/RMM memory accounting.
4. Publish Vortex prerequisites, update the retained pin, and prepare the upstream POC.

## Worktree cautions

- Main deliverable: `benchmarks/cudf-ndsh/upstream.patch`; optional generator fixes
  remain a separate patch. Editable cuDF source: ignored `build/cudf-ndsh-src`.
  Apply to a fresh pinned checkout elsewhere; never modify `/home/ubuntu/cudf`.
- Pinned Release build tree: `build/cudf-ndsh-build`; NVCC 13.1 workaround:
  `build/cudf-ndsh-build/access-repro/nvcc131-cudf-hook.cmake`. Do not mix pinned headers
  with old cuDF libraries.
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports remain **ignored**.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
