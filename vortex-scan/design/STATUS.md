<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Scan V2: status and how to resume

Written 2026-09-30, when work was paused. This file is the handover: what the goal is, what
exists, what was measured, what to do next, and what is not done. Delete it when V2 becomes the
default scan.

## Goal

A scan subsystem built only from

1. planners and morsels (`vortex-scan/src/planning`),
2. plan nodes (`vortex-layout/src/plan/plans`),
3. plan exec nodes (`vortex-layout/src/plan/exec`),

reusing nothing from the layout-reader scan ("V1") that does not fit cleanly, and at least 30%
faster than V1 by per-query geometric mean, on hot and on cold runs, measured without cheating.
A 30% speedup is a V2/V1 time ratio of 0.77 (0.70 if "30% less time" is meant).

V2 is selected at run time with `VORTEX_SCAN_V2=1`. V1 stays untouched until cutover because
every measurement compares against it in the same binary.

## Where things are

| What | Where |
|---|---|
| V2 as the owner left it | branch `ji/scan-feature`, worktree `../vortex-scan-feature`, commit `9c6392e5aa`. Do not build or run git there: it is the owner's worktree and automated builds in it were refused. |
| This work | branch `ji/scan-feature-freeze`: `7b196adcd4` (interface freeze) on top of `ji/scan-feature`, then this document. Local only, not pushed. Checked out in the `vortex-2` worktree since 2026-10-01. |
| Measurement home | `../vortex-scan-freeze`, a second worktree left detached at this branch's head as of 2026-10-01. It holds the build cache, the bench binaries, and the bench data below, none of which are tracked. |
| Superseded prototype | branch `ji/scan-traits-design`, crate `vortex-scan-plan`. Holds the design docs (`vortex-scan/design/*.md` on that branch) and nothing V2 runs. |
| Measurement scripts and results | `vortex-scan/design/measurements/` (see its README). |
| Bench binaries | `../vortex-scan-freeze/target/bench-bins/9c6392e5aa/` (baseline) and `.../7b196adcd4/` (freeze). Rebuild with the command under "How to measure" if that `target/` was cleaned. |
| Bench data | `../vortex-scan-freeze/vortex-bench/data/{tpch/1.0,tpch/10.0,clickbench_partitioned}`, as APFS copy-on-write clones (own inodes, so evicting them from the page cache does not disturb other worktrees). |

What `vortex-2` held before the checkout is in a git stash tagged
`scan-plan-prototype-driver-opt`: a driver optimisation and a divan bench for the superseded
`vortex-scan-plan` crate. It is **not** part of this work and can be dropped; the freeze commit
carries the same ideas into the real driver.

## What V2 is today

```text
engine -> ScanBuilder + ScanFile -> v2::prepare   lower the layout once per reader; optimise
                                                  the projection and each filter conjunct
       -> RepeatedScanV2::split_plans             one SplitPlan per filter split
       -> Run::admit(split.root, scope, io.session())
            AnnouncePlanner     announce the split's likely segments (coalescing hint)
            FilterPlanner       zone pruning (when the filter can be falsified from zones)
            FilterPlanner       one ExecGraph per conjunct, adaptively ordered; prefetch
            ProjectionPlanner   one ProjectionMorsel per projection split
            ProjectionMorsel    ExecGraph over the surviving rows -> Progress::Batch
       -> IoSource session      file: FileScanIo over V1's coalescing read driver
                                other: SegmentScanIo over any SegmentSource
```

DuckDB (`vortex-duckdb/src/file_reader.rs`) calls `prepare(..).execute(row_range)` and each
DuckDB thread blocks on one split future. DataFusion (`vortex-datafusion/src/persistent/opener.rs`)
calls `v2::into_stream`, which spawns one task per split and buffers them. Nothing else reaches
V2: `VortexFile::scan`, Python, FFI, JNI, and the multi-file sources are V1 only.

## What the freeze commit changed (`7b196adcd4`)

The seams that separate the planners, the IO service, and the engines, so they can be worked on
independently. Behaviour of `prepare`, `execute`, `execute_batches`, and `into_stream` is
unchanged.

- **Protocol.** `State::NeedsIO(batch)` became `State::Waiting`. Every request is published
  exactly once, from `compute()`; `state()` never lists requests. `Planner` and `Morsel` are
  `Send`. Waiting with no fetch outstanding, and publishing a fetch that is still outstanding,
  are protocol errors.
- **Driver.** `Run::new()` + `admit(root, scope, io)` for any number of roots, each with its
  own IO session; `advance()` returns `Progress::{Batch, RootDone, Waiting, Idle}` as they
  happen; `poll_completion(cx)` awaits every live root's session; `cancel(root)`. A delivery to
  an item that still waits leaves it parked. `Driver::run` remains as the blocking one-root
  convenience.
- **IO.** `IoTarget::Range` carries the alignment. `IoSource` gained `poll_completion`;
  `IoService::session()` in `vortex-io` replaces the layout crate's `ScanIo`/`SplitIo`.
- **Scan.** `RepeatedScanV2::{split_plans, io, map, join}` and `SplitPlan` let an engine admit
  splits to its own run. Announcements moved into `AnnouncePlanner`.
- **Removed.** The polling bridge (`PollingSegmentSource`, `SplitMorsel`, `FilterProject`,
  `plan_scan`) and the tests that ran through it.

This departs from two decisions in the design docs on `ji/scan-traits-design`, which should be
updated when that branch is next touched: TRAITS.md says `state()` lists outstanding requests
and that live planners need not be `Send`.

Verified for that commit: `cargo nextest run -p vortex-io -p vortex-scan -p vortex-layout -p
vortex-file` (772 passed), `cargo clippy ... --all-targets --all-features -- -D warnings` on the
same four crates (clean), `cargo +nightly-2026-09-10 fmt`, and
`cargo build -p duckdb-bench -p datafusion-bench --profile release_debug` (both engines compile
unchanged). **Not verified:** that the freeze costs no performance (the rebuilt binaries have
not been benchmarked), `vortex-duckdb` and `vortex-datafusion` tests, sqllogictests, doctests,
`typos`.

## Baseline measurements (commit `9c6392e5aa`, 2026-09-30)

V2/V1 time ratio, per-query geometric mean of medians. Below 1.0 means V2 is faster. Hot is
steady state with the first iteration of each process dropped; cold evicts every data file from
the page cache before each fresh process. The machine was busy throughout (load average 10 to
77 on 14 cores), so treat ±3% as noise; V1 and V2 were interleaved. Where a cell was measured
twice both values are given.

| Suite | DuckDB hot | DuckDB cold | DataFusion hot | DataFusion cold |
|---|---|---|---|---|
| TPC-H SF1 | 0.89, 0.90 | 0.92 | 1.04, 1.05 | 1.02 |
| TPC-H SF10 | 0.90, 0.88 | 0.84 | 1.02, 1.01 | 1.03 |
| ClickBench | 0.94, 0.88 | 0.92 | 1.03, 0.90 | 1.01 |

The measuring was stopped part-way through a second pass: the warm-first tables
(`measurements/results-warmfirst.md`) cover DuckDB TPC-H only, with DuckDB ClickBench
incomplete and DataFusion not started. Per-query tables and the attribution of twelve
representative queries are in `measurements/REPORT_*.md`.

- Instructions retired are equal (V2/V1 0.98 to 1.02 per process). V2's lead on DuckDB is wall
  time, not less work.
- Cold: V2 physically reads the same number of bytes as V1 (within 1% on every suite), so V2
  does not over-read relative to V1.
- DataFusion ClickBench hot is unstable (1.03 and 0.90 on two runs); it needs a quiet machine.

### Where the time goes (hot, Samply attribution of every query)

Share of each query's wall time spent inside the scan, and what a scan that cost nothing, or
half as much, would do to the geomean. From `measurements/sweeps.log`, summarised by
`sweep_summary.py`.

| Engine and suite | Scan share of wall, median (V1 / V2) | Geomean if scan cost fell 50% (V1 / V2) | If it fell 100% (V1 / V2) |
|---|---|---|---|
| DuckDB ClickBench | 71% / 69% | 0.66 / 0.68 | 0.20 / 0.24 |
| DuckDB TPC-H SF10 | 63% / 54% | 0.68 / 0.73 | 0.32 / 0.41 |
| DuckDB TPC-H SF1 | 50% / 41% | 0.73 / 0.78 | 0.44 / 0.54 |
| DataFusion ClickBench | 29% / 32% | 0.78 / 0.78 | 0.42 / 0.43 |
| DataFusion TPC-H SF1 | 36% / 38% | 0.80 / 0.80 | 0.58 / 0.57 |
| DataFusion TPC-H SF10 | 24% / 26% | 0.83 / 0.83 | 0.60 / 0.59 |

Mean share of busy thread-time per query, by category:

| Engine and suite | Blocked in scan (IO wait, handoff) V1 / V2 | IO read + dispatch V1 / V2 | Decode V1 / V2 | Filter kernels V1 / V2 | Scheduling, layout V1 / V2 |
|---|---|---|---|---|---|
| DuckDB ClickBench | 39% / 29% | 8% / 9% | 11% / 13% | 3% / 4% | 7% / 8% |
| DuckDB TPC-H SF10 | 41% / 23% | 7% / 9% | 10% / 14% | 5% / 6% | 5% / 6% |
| DuckDB TPC-H SF1 | 47% / 29% | 6% / 7% | 8% / 12% | 4% / 6% | 5% / 5% |
| DataFusion ClickBench | 6% / 7% | 13% / 10% | 16% / 17% | 5% / 5% | 10% / 10% |
| DataFusion TPC-H SF1 | 2% / 1% | 19% / 27% | 9% / 8% | 7% / 7% | 8% / 10% |
| DataFusion TPC-H SF10 | 2% / 3% | 19% / 19% | 9% / 8% | 5% / 5% | 5% / 8% |

What this says:

1. **On DuckDB the scan is most of the time, and the largest single cost is threads blocked
   inside the scan waiting for a read to be handed back**, even though every byte is in the
   page cache: 39 to 47% of busy thread-time in V1, 23 to 29% in V2. V2's current lead is
   exactly this number falling. Driving it towards zero is the clearest route to 0.77 on
   DuckDB.
2. **V2 burns more CPU spinning or yielding on DuckDB** (5 to 8% of all CPU against 1 to 2.5%
   in V1) and uses more system time. Not yet traced to a call site.
3. **On DataFusion the scan is a quarter to a third of the time**, so reaching 0.77 from the
   scan alone needs the scan to cost roughly 60% less. The largest scan cost there is the read
   path itself (IO read + dispatch, 10 to 27% of busy thread-time, hot), then decode.
4. Pruning and filter kernels are small everywhere. The planner-policy ideas (read only what
   survives the filter, prune before creating splits, dynamic-filter checkpoints) cannot move
   the hot geomean much; they matter for cold runs and selective queries only.

## Next steps, in order

Each has the measurement that should confirm or kill it before much is built.

1. **Check the freeze did not cost performance.** Run the hot matrix with
   `BIN_DIR=target/bench-bins/7b196adcd4` and compare V2 there with V2 from the baseline
   binaries. Expect no change. Do this first, on a quiet machine.
2. **Remove the read handoff for bytes that are already resident (IO service).** Today every
   read, hot or cold, goes: event channel -> V1 read driver task -> `spawn_blocking` pread ->
   oneshot -> waker -> the waiting split. Build a protocol-native `IoService` for local files
   that coalesces with a pure function over the registered ranges and completes a fetch inline
   on the requesting thread when the bytes are resident (`pread` directly, or `mmap`), falling
   back to service IO threads otherwise. `measurements/iobench.c` compares pread and mmap
   scaling for hot reads. Files: `vortex-file/src/segments/scan_io.rs` (replace), reusing the
   coalescing algorithm in `vortex-file/src/read/driver.rs` but not its event plumbing. Kill
   test: if "blocked in scan" and "IO read + dispatch" do not fall on DuckDB TPC-H SF10 q1/q6
   after the change, stop. This is the largest expected win on both engines.
3. **Let a DuckDB thread work on another split while one waits (engine integration).** Give
   each DuckDB thread its own `Run`, admit the next k `SplitPlan`s, and advance whichever is
   runnable. Files: `vortex-duckdb/src/file_reader.rs`, `table_function.rs`. Only worth it for
   whatever blocked time step 2 leaves (cold runs, object stores).
4. **Find V2's spin.** Profile DuckDB ClickBench q16/q17/q18 (V2 slower than V1 there) and the
   5 to 8% spin/yield CPU; suspects are the no-op-waker `poll()` of `FuturesUnordered` in
   `FileSplitIo` every driver round, and the executor wake-ups per completion.
5. **DataFusion: bound `prepare` to the partition.** Each partition's `prepare` walks the whole
   file for split boundaries and chunk starts and re-optimises every plan
   (`vortex-layout/src/scan/v2/repeated_scan.rs`). Measure `prepare` against query time on
   TPC-H SF1 first; this is probably the 2 to 5% V2 deficit there. Do not cache prepared scans
   across partitions: that was tried and lost.
6. **Decode.** 8 to 17% of busy thread-time. Share decoded or canonical columns between the
   filter and the projection of a split, and across splits that cut the same chunk
   (`DecodeCache` is per split today).
7. **Coverage and cutover.** List layouts (no exec node for `ListPack`; V2 fails on them),
   file-statistics pruning inside V2, multi-file scans (`file_ordinal` is always 0), filter
   with limit, streaming ordered delivery with backpressure, `SplitBy` and caller-supplied
   splits, then the other entry points (`VortexFile::scan`, Python, FFI, JNI). V2 has no
   fallback to V1: unsupported input is an error.
8. **Remove what V2 still borrows from V1**: the read driver (step 2 does this), the global
   registry of lowered plans keyed by layout-reader address (`scan/v2/file.rs`), `ScanBuilder`
   and the V1 reader tree as the entry point, the copied future-per-split stream
   (`scan/v2/stream.rs`), and the engines' own V1 calls (DuckDB file skip and statistics,
   DataFusion byte-range to row mapping).

Honest expectation, not a measurement: DuckDB hot and cold can plausibly reach 0.77 through
steps 2 to 4. DataFusion is unlikely to reach 0.77 from the scan alone; 0.85 to 0.90 is a
realistic aim unless steps 2 and 6 remove most of the read and decode cost.

## Rules that were each bought with a measured loss

- Never cut a filter split inside a chunk of a filter column.
- Evaluate predicates before filtering, and over the whole piece when at least 20% of rows are
  selected.
- Share a dictionary's values per plan as a shared array; share lowered plans per layout
  reader.
- No cache of prepared scans across DataFusion partitions.
- Never block a driver on the runtime's blocking pool.
- Prefetch a split's reads once pruning has settled its rows.
- Morsel hints are leaf-IO only and batches stay dense: no batch bookkeeping, placeholders at
  flat leaves, the filter node applies the mask per chunk.

## How to measure

`ROOT` is the worktree whose `vortex-bench/data` the binaries read (they are run with that as
their working directory); `BIN_DIR` holds the two bench binaries. Do not run two measurements at
once, and do not compile during one.

```bash
# Binaries for a commit, built wherever that commit is checked out
# (about 12 minutes and 8 GB the first time):
cargo build -p duckdb-bench -p datafusion-bench --profile release_debug
FREEZE=/Users/joeisaacs/git/spiraldb/vortex-scan-freeze
mkdir -p $FREEZE/target/bench-bins/$(git rev-parse --short=10 HEAD)
cp target/release_debug/{duckdb-bench,datafusion-bench} $FREEZE/target/bench-bins/$(git rev-parse --short=10 HEAD)/

# V1 against V2, interleaved, hot then cold, all suites and both engines (about 40 minutes):
cd vortex-scan/design/measurements
ROOT=$FREEZE BIN_DIR=$FREEZE/target/bench-bins/<commit> ./run_matrix.sh all

# One cell:
ROOT=$FREEZE BIN_DIR=... python3 bench_ab.py hot duckdb tpch 10.0 3 5
ROOT=$FREEZE BIN_DIR=... python3 bench_ab.py cold duckdb clickbench - 3
```

Without `ROOT`, the scripts use the worktree they are checked out in, which needs its own
`vortex-bench/data` (including TPC-H SF10) and `target/bench-bins/`.

Cold runs need no root: `evict.py` maps each data file and calls `msync(MS_INVALIDATE)`, then
checks with `mincore()` that nothing is resident. Results append to `results.md` next to the
scripts; raw output goes to `raw/`.

To compare two builds of V2 rather than V1 and V2, set `A_ENV="VORTEX_SCAN_V2=1"` and run the
harness once per `BIN_DIR`, or extend `bench_ab.py` to take two binary directories.

## Things that will bite

- Disk: about 22 GB was free. Building this branch in `vortex-2` starts a second build cache
  next to the one in `../vortex-scan-freeze/target` (about 14 GB); delete whichever is not in
  use. Parallel worktrees with their own `target/` do not fit.
- A DuckDB `duckdb.db` file cloned from another worktree holds views pointing at that
  worktree's files. Delete it and let the bench recreate it.
- A separate rewrite of the DataFusion integration's `persistent/` path was being scoped on
  2026-09-30 (notes in `vortex-datafusion/REWRITE.md` in the `vortex-7` worktree, uncommitted).
  Coordinate before changing `vortex-datafusion/src/persistent/opener.rs` for V2.
- `--features unstable_encodings` no longer exists on the bench crates.
- If cargo fails with exactly `sccache: error: Operation not permitted`, rerun with
  `RUSTC_WRAPPER=`.
