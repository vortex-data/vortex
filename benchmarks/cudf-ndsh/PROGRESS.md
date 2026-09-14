# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-14 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Source and commit checkpoint

- Prior signed-off series: through `e81905f44`.
- `9b6df1978`: cacheable pinned CUDA staging.
- `59a7ea66a`: generic performance-only `upstream.patch` changes — shared Q1 sum/count
  groupby with averages afterward, concurrent independent table reads in Q5/Q9/Q10
  for both formats, and 8 GiB Vortex CUDA pool retention.
- `cd3192a02`: cold-cache patch and tests — renamed
  `ndsh_q{1,5,6,9,10}_local`, `cache=warm/cold`, SF10 in the default scale-factor axis,
  per-callback eviction/residency verification, cold Vortex direct data reads, and
  offline integration coverage.
- The cumulative `benchmarks/cudf-ndsh/upstream.patch` is refreshed against cuDF
  `5339497a1a17d799687cbf189fb113411fb015ca`. Forward/cached and reverse apply checks
  pass; all 14 offline tests pass.

All five queries use generic cuDF; no hand-fitted query kernels or `q1_fused` path.
Default kernel-event suppression is reverted; no event optimization is retained.
See [README.md](README.md) for retained I/O/timing behavior and
[VALIDATION.md](VALIDATION.md) for rejected experiments and historical evidence.

## Build and evidence status

**Binaries are stale; no fresh runtime validation or performance runs.** The Release
build targeting all five queries plus `NDSH_VORTEX_IO_TEST` timed out after 1200 s.
CMake regeneration expanded to 644, then 635 steps; the build reached 223/635.
The Vortex Release archive built successfully, but cuDF benchmarks and the adapter
were **not relinked**. Do not retry automatically.

Completed checks: 5 pinned staging tests, nightly formatting, all-target/all-feature
Clippy, clang-format on 8 edited C++ files, Ruff lint/format, and the patch/offline checks above.
Exact commands and earlier runtime-test caveats are in [VALIDATION.md](VALIDATION.md).
All five query targets had built before the event revert; that does not validate the
current source. Latest timings are pre-revert diagnostics, not final-source evidence.

The ≥2× goal for both end-to-end reads and queries at SF1/SF10, warm/cold, is **not met**.
Q1 warm and SF10 cold query lag; Q6 warm is near the threshold; Q10 is marginal/noisy.
The existing full-read profile is not a full Q1 query profile; the latter is still absent.

## Next steps

1. Explicitly choose a longer bounded build window, complete relinking of all five
   benchmarks and the adapter, then run fresh runtime/reference checks. No automatic retry.
2. Collect stable SF1/SF10 warm/cold read/query matrices before drawing final-source
   performance conclusions. Run GPU benchmarks sequentially.
3. Capture a full Q1 query profile using the
   [profiling safety guard](VALIDATION.md#profile-evidence-and-safety).
   Defer SF100 until matrices are stable, with separate Vortex/RMM memory accounting.
4. Publish Vortex prerequisites, update the retained pin, and prepare the upstream POC.

## Worktree cautions

- Tracked deliverable: `benchmarks/cudf-ndsh/upstream.patch`; editable cuDF source:
  ignored `build/cudf-ndsh-src`. Apply to a fresh pinned checkout elsewhere; never
  modify `/home/ubuntu/cudf`.
- Pinned Release build tree: `build/cudf-ndsh-build`; NVCC 13.1 workaround:
  `build/cudf-ndsh-build/access-repro/nvcc131-cudf-hook.cmake`. Do not mix pinned headers
  with old cuDF libraries.
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports remain **ignored**.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
