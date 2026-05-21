---
name: bench-performance
description: Iterate on Vortex vx-bench query performance with benchmark comparisons, engine-specific benchmark flags, RUST_LOG/tracing/metrics/explain output, and Samply profiles. Use when optimizing or investigating a vx-bench query, benchmark regression, engine/format comparison, or Vortex benchmark hotspot.
---

# Bench Performance

## Overview

Use this skill for Vortex benchmark performance work driven by `vx-bench` or the direct benchmark
binaries. Start with comparable benchmark evidence, then add diagnostics (`RUST_LOG`, explain,
metrics, tracing), then profile with Samply when the slow target is clear.

This skill complements `$samply`: use `$samply` for profile recording, Firefox-profiler JSON
schema, symbolication, and stack summaries. This skill decides when to benchmark, what engine flags
to use, what logs/metrics to collect, and how to iterate without losing the comparison.

## Reporting Cadence

Emit evidence as soon as it exists:

1. When a benchmark run starts, say the suite/query/engine/format, iteration count, and env toggles.
2. When it finishes, immediately show the run ID or output path and the comparison/timing table.
3. When logs, metrics, explain output, or a Samply summary finishes, immediately summarize the top
   signal before reading more code.
4. Only then form a hotspot hypothesis and make one scoped change.

Do not wait for a deep code read before showing benchmark comparisons or first stack summaries.

## Standard Loop

1. Capture branch and dirty state:

   ```bash
   git status --short
   git branch --show-current
   git diff --stat develop...HEAD
   ```

2. Identify the exact benchmark target:

   - suite: `tpch`, `tpcds`, `clickbench`, `fineweb`, `gh-archive`, `polarsignals`,
     `public-bi`, or `statpopgen`;
   - query number(s), for example `-q <query>`;
   - engine/format target(s), for example `datafusion:vortex` versus `datafusion:parquet`;
   - runtime environment toggles, if the branch exposes any.

3. Run a small comparable benchmark through `vx-bench`:

   ```bash
   FEATURE_TOGGLE=1 UV_CACHE_DIR=/private/tmp/vortex-uv-cache \
     uv run --project bench-orchestrator vx-bench run <benchmark> \
     -e <engine> \
     -f <baseline-format>,<candidate-format> \
     -q <query> \
     -i 3 \
     -l <label> \
     --output /private/tmp/<label>.jsonl \
     --verbose
   ```

   Use `--no-build` only after the relevant binary has been rebuilt from current sources.

   Do not run benchmark measurements in parallel with other benchmark measurements. Parallel shell
   work is useful for source inspection, but competing benchmark binaries distort medians and create
   multi-second outliers that look like engine regressions.

4. If the comparison is surprising or too noisy, rerun the same target with more iterations before
   profiling. Keep the query/engine/format/env identical.

5. Use direct benchmark binaries for diagnostics that `vx-bench` does not expose. Build binaries
   with the orchestrator or directly:

   ```bash
   cargo build -p datafusion-bench --profile release_debug --features unstable_encodings
   cargo build -p duckdb-bench --profile release_debug --features unstable_encodings
   cargo build -p lance-bench --profile release_debug
   ```

6. Collect only the diagnostic needed next: `RUST_LOG`, `--explain`, `--show-metrics`, `--tracing`,
   memory, system tools, or Samply. Report the result, then inspect code near the evidence.

7. Make one scoped change, rebuild the narrow binary, rerun the same benchmark command, and compare
   against the previous run/output before adding broader checks.

## Common Environment

- `RUST_LOG` wins over `--verbose`. Without `RUST_LOG`, `--verbose` raises default logging to
  `TRACE`; otherwise the env filter controls output.
- Useful starting filters:

  ```bash
  RUST_LOG=info
  RUST_LOG=vortex_datafusion=debug,vortex_layout=debug,vortex_file=debug,datafusion=warn
  RUST_LOG=vortex_datafusion::persistent::opener=trace,vortex_layout::layouts::zoned=trace,info
  ```

- `--tracing` attaches a Perfetto layer and writes `trace.json` in the current directory.
- Runtime env toggles are useful for fast iteration because they let you compare behavior without
  recompiling. Use the same toggles on the benchmark, diagnostics, and Samply commands for a given
  run. Compare with and without a toggle as separate labeled runs, and discover available toggles
  from source (`rg -n "std::env::var|env::var"` is a good starting point).

## Engine-Specific Flags

`vx-bench` normalizes the common path, but direct binaries differ. Check source or `--help` before
assuming a flag exists.

### DataFusion: `target/release_debug/datafusion-bench`

Source: `benchmarks/datafusion-bench/src/main.rs`.

Supported diagnostics:

- `--formats parquet,vortex,vortex-compact,lance,arrow`;
- `--queries 6`, `--exclude-queries 1,2`, `--iterations N`, `--display-format gh-json`;
- `--hide-progress-bar`, `-o /private/tmp/out.jsonl`, `--gh-json-v3 /private/tmp/out.v3.jsonl`;
- `--verbose`, `--tracing`, `--track-memory`, `--runner NAME`, `--opt key=value`;
- `--explain` prints query plans instead of timing;
- `--show-metrics` prints Vortex execution-plan metrics after a timed run.

Declared but currently not useful unless the source changes: `--threads`, `--emit-plan`, and
`--export-spans` are parsed but not wired into the execution path.

Examples:

```bash
FEATURE_TOGGLE=1 RUST_LOG=vortex_datafusion=debug,vortex_layout=debug,datafusion=warn \
  target/release_debug/datafusion-bench tpch \
  --display-format gh-json --iterations 5 --hide-progress-bar \
  --formats <format> --queries <query> --show-metrics \
  -o /private/tmp/<label>.jsonl
```

```bash
FEATURE_TOGGLE=1 target/release_debug/datafusion-bench tpch \
  --explain --formats <format> --queries <query>
```

### DuckDB: `target/release_debug/duckdb-bench`

Source: `benchmarks/duckdb-bench/src/main.rs`.

Supported diagnostics:

- `--formats parquet,vortex,vortex-compact,duckdb`;
- `--delete-duckdb-database` rebuilds the per-format DuckDB database;
- `--threads N` sets DuckDB's `threads` config;
- `--reuse` keeps one DuckDB connection across iterations, useful with Samply to keep work on the
  same threads;
- common flags: `--queries`, `--exclude-queries`, `--iterations`, `--display-format`,
  `--hide-progress-bar`, `-o`, `--gh-json-v3`, `--track-memory`, `--verbose`, `--tracing`,
  `--runner`, `--opt`, `--explain`.

Example:

```bash
RUST_LOG=duckdb_bench=trace,vortex_duckdb=debug,info \
  target/release_debug/duckdb-bench tpch \
  --display-format gh-json --iterations 5 --hide-progress-bar \
  --formats <baseline-format>,<candidate-format> --queries <query> --threads 8 --reuse \
  -o /private/tmp/<label>.jsonl
```

### Lance: `target/release_debug/lance-bench`

Source: `benchmarks/lance-bench/src/main.rs`.

Important differences:

- no `--formats`: the binary always generates/registers Lance data and reports `datafusion:lance`;
- no `--explain` or `--show-metrics` path today;
- common flags: `--queries`, `--exclude-queries`, `--iterations`, `--display-format`,
  `--hide-progress-bar`, `-o`, `--gh-json-v3`, `--track-memory`, `--verbose`, `--tracing`,
  `--runner`, `--opt`;
- `--threads` is parsed but currently not wired into the Lance/DataFusion session.

Example:

```bash
RUST_LOG=lance_bench=debug,datafusion=warn \
  target/release_debug/lance-bench tpch \
  --display-format gh-json --iterations 5 --hide-progress-bar \
  --queries <query> \
  -o /private/tmp/<label>.jsonl
```

## Comparing Direct Outputs

For direct `gh-json` output, use the bundled helper:

```bash
python3 .agents/skills/bench-performance/scripts/compare_gh_json.py \
  /private/tmp/<label>.jsonl \
  --baseline <engine>:<baseline-format>
```

It ignores non-JSON log lines, groups by benchmark/query target, reports milliseconds, min/median/max,
and ratios against the selected baseline target or the first target in each query.

## Summarizing Mask/Row-Demand Logs

When a run emits Vortex mask-style debug lines, summarize them before reading more code. This
includes mask-debug rows and pruning rows with the same coordinate fields.
These logs are useful for deciding whether a hot stack is expensive per row, called over too many
rows, or repeated over the same coordinates:

```bash
python3 .agents/skills/bench-performance/scripts/summarize_mask_debug.py \
  /private/tmp/<label>.log \
  --message-regex 'filter|conjunct|flat' \
  --duplicates
```

The output reports batch counts, zero-output percentage, total input/output rows, density, batch
size quantiles, elapsed totals when present, the largest batches, and duplicate coordinate masks.
If a low-selectivity filter still shows very large input batches late in the pipeline, compare this
with the Samply timeline: a few huge all-false batches can explain idle workers even when total row
work looks reasonable.

For conjunct scheduling logs, aggregate compute rows per predicate. This handles candidate
conjunct rows and baseline pruning/filter conjunct rows when the logs include comparable fields:

```bash
python3 .agents/skills/bench-performance/scripts/summarize_conjunct_debug.py \
  /private/tmp/<label>.log
```

Use this when checking whether a pushed-down or shared mask is actually evaluated once, or whether
each projected field is driving the same conjunct work again.

When investigating stream scheduling, enable the relevant flow trace and summarize it immediately:

```bash
<FLOW_TRACE_ENV>=1 RUST_LOG=<flow-target>=debug,datafusion=warn \
  target/<profile-dir>/datafusion-bench clickbench \
  --display-format gh-json --iterations 1 --hide-progress-bar \
  --formats vortex --queries <query> \
  -o /private/tmp/<label>.jsonl > /private/tmp/<label>.log 2>&1

python3 .agents/skills/bench-performance/scripts/summarize_flow_tracing.py \
  /private/tmp/<label>.log
```

Read the summary as a scheduling picture:

- `filter pushdown failed` with no `filtered flat` events means the plan is applying a sparse mask
  after full value/projection work.
- `dict/struct/project pushdown failed` shows which row-preserving node blocked mask pushdown.
- `materialised mask read_all done` counts full mask barriers and their true counts.
- `filtered flat mask read_all done` or `filtered flat incremental mask ready` counts leaf mask
  consumption; compare sums to detect repeated mask use across projected fields.
- `aligned producer waits by label` separates backpressure in `filter`, `struct`, and `conjunct`
  zips. High cumulative send wait means producer tasks are ready but the aligned consumer is
  waiting on another child or on downstream demand.

## Count vs Latency

A hot sampled stack does not by itself say whether the operation is intrinsically slow, called too
many times, or waiting on contention. Before changing code, classify it:

- **More work**: operation count or bytes are much higher than the baseline.
- **Slower work**: operation count is similar but total time or per-operation max/median is higher.
- **Contention/waiting**: samples sit in kernel wait, parking, mutex, semaphore, or blocking-pool
  admission with similar work counts.
- **Scheduler shape**: many idle or parked workers can dominate wall-time samples; compare
  CPU-weighted and sample-weighted summaries.

When reading Samply's timeline view, look at the shape of CPU occupancy, not only the hottest
function names:

- A healthy parallel Vortex/DataFusion profile usually keeps worker threads doing CPU work across
  most of the timed region. Large empty spans or many idle workers while one worker runs a hot
  stack point to scheduling skew, dependency ordering, partition imbalance, or a straggler. In
  that case, investigate why the work became serialized before micro-optimizing the leaf function.
- If a leaf such as a string scan appears hot only after other workers have gone idle, the problem
  may be that it was released late or is waiting on upstream work, not that the leaf loop itself is
  too slow. Check stream readiness, partition sizes, conjunct/order decisions, and whether earlier
  operators delay the selective work.
- If allocation frames such as `with_capacity`, `RawVec`, `reserve`, or allocator symbols appear
  throughout the whole trace, treat that as allocation churn and missing buffer reuse. Look for
  per-batch scratch allocation, repeated materialization, unbounded `Vec` creation, and places
  where reusable buffers or capacity-preserving paths would avoid rebuilding the same memory.

For Vortex/DataFusion scan I/O, prefer `--show-metrics` before OS tracing:

```bash
target/release_debug/datafusion-bench <benchmark> \
  --display-format gh-json --iterations 1 --hide-progress-bar \
  --formats <format> --queries <query> --show-metrics \
  -o /private/tmp/<baseline-label>.jsonl \
  > /private/tmp/<baseline-label>.metrics.txt 2>&1

FEATURE_TOGGLE=1 target/release_debug/datafusion-bench <benchmark> \
  --display-format gh-json --iterations 1 --hide-progress-bar \
  --formats <format> --queries <query> --show-metrics \
  -o /private/tmp/<candidate-label>.jsonl \
  > /private/tmp/<candidate-label>.metrics.txt 2>&1

python3 .agents/skills/bench-performance/scripts/compare_metrics.py \
  /private/tmp/<baseline-label>.metrics.txt \
  /private/tmp/<candidate-label>.metrics.txt \
  --metrics vortex.io.read.duration_count,vortex.io.read.total_size,vortex.file.segments.cache.misses,io.requests.individual,io.requests.coalesced,time_elapsed_scanning_total,vortex.io.read.duration_max
```

If the candidate has far more reads, bytes, or cache misses than the baseline, treat the hot I/O
stack as repeated work first. If counts and bytes are similar but duration grows, investigate
per-operation latency and contention.

Use logs when metrics are missing. Add narrow trace points around the suspected operation and log:
call count, requested byte range, coalesced range, segment id, row range, elapsed time, and whether
the call hit/missed a cache. Keep logs behind existing `tracing` levels and run with a focused
`RUST_LOG` filter.

## Batch Coordinate Diagnostics

When comparing two scan designs, aggregate timings can hide whether the same work ran over the same
rows. Add temporary trace/debug fields that make each compute event joinable:

- stable scan label, usually the file/object path or another per-input identifier;
- root row coordinates, not only child-local row coordinates;
- local row range too, when the execution node works in translated child coordinates;
- conjunct or expression identifier and its chosen order;
- input/output true counts, density, first/last surviving row, and a short sample of survivor
  ranges;
- a deterministic hash of the absolute survivor row set for same-window checks;
- partition-independent fingerprints such as wrapping row-id sum and row-id xor so unions can be
  compared when two paths use different batch boundaries.

Be careful with multi-file benchmarks: `row_start=0..N` is only meaningful with a file label. Be
careful with nested layouts too: child plans may log local coordinates unless the diagnostic uses
the scoped demand, split metadata, or another explicit root-offset source. If two paths partition
the same file differently, identical `(file, row_range)` keys may not exist; compare per-conjunct
input/output row counts first, then add a union-level dump only if exact row-set equality is still
unclear.

Prefer diagnostic logs over changing public batch types. Useful log points are final baseline split
projection, candidate mask/filter nodes, and filtered candidate leaf projection nodes. For each
batch-like event, emit the input coordinate window plus the post-mask survivor summary/hash; that
lets you compare exact row sets even when physical batch boundaries differ. Avoid logging every
unfiltered leaf by default: nested layouts such as dictionary values may live in a different row
space and can drown out the scan-coordinate signal.

## Samply

Profile only after the benchmark identifies a slow target. Prefer direct binary commands so the
profile contains only the target engine:

```bash
FEATURE_TOGGLE=1 samply record --save-only --unstable-presymbolicate --rate 1000 \
  --output /private/tmp/<label>.profile.json.gz \
  -- target/release_debug/datafusion-bench <benchmark> \
    --display-format gh-json --iterations 500 --hide-progress-bar \
    --formats <format> --queries <query>
```

Put environment assignments before `samply record`; the profiled command after `--` should be the
locally built benchmark binary, not a system helper. On macOS in Codex, `Unknown(1100)` from
`samply record` usually means sandboxed profiling was blocked, so rerun the same profile command
with escalated permissions. See the `$samply` skill for the detailed failure modes.

DuckDB profiling usually needs `--reuse`:

```bash
samply record --save-only --unstable-presymbolicate --rate 1000 \
  --output /private/tmp/<label>.profile.json.gz \
  -- target/release_debug/duckdb-bench <benchmark> \
    --display-format gh-json --iterations 500 --hide-progress-bar \
    --formats <format> --queries <query> --reuse
```

Immediately summarize with `$samply`'s script:

```bash
python3 .agents/skills/samply/scripts/profile_summary.py \
  /private/tmp/<label>.profile.json.gz \
  --binary target/release_debug/datafusion-bench \
  --symbolicate --weight-mode cpu \
  --top 12 --threads 2 --stacks 4 --stack-depth 10
```

After the first stack summary, ask what would distinguish count from latency. Examples:

- I/O stack: compare read count, total bytes, read-size distribution, max/read duration, and cache
  misses.
- Lock stack: compare lock acquisition count, lock hold time if logged, wait time, and number of
  contending tasks.
- Decompression stack: compare decoded rows, decoded arrays/segments, selected columns, and whether
  identical segments decode repeatedly.
- Predicate stack: compare predicate evaluation rows and whether pushdown changed between plans.

Do not infer “the function is slow” from samples until operation counts have been checked.

## macOS System Tools

Use system tools only after benchmark metrics/logs cannot answer the question. They often require
Terminal/Developer Tools permissions or root privileges.

- `sample <pid-or-name> 10 1 -file /private/tmp/sample.txt`: quick stack sample. Easier than
  Samply, less structured.
- `spindump <pid-or-name> 10 10 -file /private/tmp/spindump.txt` or `spindump ... -json`: system
  call-tree sample including wait states; useful for contention/scheduler questions.
- `fs_usage -w -f filesys -t 5 datafusion-bench`: filesystem syscall stream. Useful for seeing
  repeated opens/reads by process name. Can be noisy and permission-sensitive.
- `iotop -C -P 1 10`: system-wide I/O pressure over time. Good for “is this actually disk-bound?”,
  weak for per-call attribution.
- `dtrace`, `opensnoop`, `execsnoop`: potentially useful on macOS, but SIP/sandbox permissions can
  make them unavailable. If they fail with permission errors, fall back to benchmark metrics and
  explicit `tracing` logs.
- `xctrace`: command-line Instruments runner, but requires full Xcode, not only Command Line Tools.
  If unavailable, note that and use Samply/spindump/logs.

## Interpreting Signals

- Benchmark regression without profile shift: suspect noise, data generation, environment toggles,
  or changed query plan.
- High `object_store::local::LocalFileSystem::get_opts` / blocking-pool stacks: inspect file IO,
  partitioning, segment reads, and cache behavior.
- High `arrow_ord::cmp::apply_op` or DataFusion expressions: inspect pushed predicates and whether
  Vortex pushdown failed.
- High Vortex decompression (`vortex_fastlanes`, bitpacking): inspect encoding choice, projection,
  mask selectivity, and repeated decode.
- High `Mask`, `BitBufferMut`, or materialized-mask stacks: inspect filter pipeline, CSE, mask
  sharing, zone pruning, and whether masks are computed more than once.
- High scheduler/parking stacks with low useful self frames: compare CPU-weighted and sample-weighted
  summaries before blaming idle threads.

## Final Report

Include:

- exact benchmark/profile commands and env vars;
- run IDs and output/profile paths;
- comparison table or direct-output summary;
- first stack/log/metric evidence;
- code changes made and why they match the evidence;
- checks run, and any checks skipped.
