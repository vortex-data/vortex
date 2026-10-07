<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Non-compact DuckDB execution and IO overlap

Investigation dates: 2026-10-06 and 2026-10-07. PR #10339, revision
`afeb2d1a4879ffae48802fe03939ce3ec7a1e0fa`, with the preceding V1 cache-policy
corrections described in [the driver investigation](v2-driver-performance.md).
All measurements here use normal `vortex`, eight DuckDB threads, and CPU affinity
24–31. Compact files are excluded.

The retained 1 MiB projection read-ahead setting improves Q6's paired median by
about 20%, with unchanged read volume and instruction counts within 0.4%. The final
400-iteration profiles show a 16.0% lower median. Q23 and FineWeb do not establish
a general win, so the setting remains disabled by default. These results use V1's
cache limits throughout.

The continuation measured extension-array BETWEEN delegation, now split into
[#10378](https://github.com/vortex-data/vortex/pull/10378). The implementation,
regression tests, and microbenchmark are on `ji/between-remaining-encodings`;
the driver branch no longer includes them. The measurements below describe the
combined prototype before that split. Q6 executes about 21%
fewer instructions: its date range reaches the storage BETWEEN path instead of
two comparisons that can each decompress the same storage. Separate paired runs
with the same read-ahead budget show 6.4–10.2% lower median times. The new rule is
independent of read-ahead and adds no caching. Detailed evidence follows below.

## Retained change

`VORTEX_SCAN_PROJECTION_PREFETCH_BYTES` starts a bounded set of projection segment
reads when a split first requests filter IO. Previously, projection reads began
after every conjunct finished. The budget counts segment lengths per active
filter split; existing coalescing can include adjacent registered ranges, so it is
not a strict process memory cap. Remaining filter prefetches retain their existing
behavior. Projection prefetch skips zone pruning and chunks excluded by the mask.

The default is zero. The experiments find a repeatable Q6 improvement, but no
general improvement across Q23 and FineWeb Q3. There is no basis for enabling it
globally. For the measured Q6 workload, use:

```bash
VORTEX_SCAN_V2=1 VORTEX_DUCKDB_FILE_PREFETCH=0 \
VORTEX_SCAN_PROJECTION_PREFETCH_BYTES=1048576 \
taskset -c 24-31 target/release_debug/duckdb-bench tpch \
  --formats vortex --threads 8 --queries 6 --iterations 14 \
  --hide-progress-bar --opt scale-factor=10.0 \
  --opt remote-data-dir=file:///home/ec2-user/votex-3/vortex-bench/data/tpch/10.0/
```

The benchmark comparison helper accepts separate budgets for each label. The
final three-query comparison is reproducible with:

```bash
taskset -c 24-31 python3 scripts/bench-v2-drivers.py \
  --baseline duckdb=target/release_debug/duckdb-bench \
  --candidate duckdb=target/release_debug/duckdb-bench \
  --baseline-projection-prefetch-bytes 0 \
  --candidate-projection-prefetch-bytes 1048576 \
  --data-root vortex-bench/data --output target/duckdb-overlap/reproduction \
  --workload tpch:6 --workload clickbench:23 --workload fineweb:3 \
  --formats vortex --threads 8 --prefetch 0 \
  --iterations 14 --warmup 2 --rounds 3 --perf-stat
```

No new cache is introduced. Ordinary decoded-segment reuse remains disabled;
read bytes are shared only by pending registered consumers, matching V1's
filter/projection futures. Dictionary, zone, and existing footer allowances are
unchanged. Default DuckDB timing runs reopen the connection each iteration.
Samply runs explicitly reuse the connection only to preserve profiling threads.
All comparisons use the normal operating-system page cache; they are not
cold-storage measurements.

## Timing and perf evidence

Timing uses tracing disabled. Separate perf processes collect counters after two
warmups, around query callbacks. Callbacks include connection reopening while
DuckDB's reported query time excludes it. Compare instruction counts as workload
controls, not as a replacement for query latency. User-only context-switch
counters returned zero and are not used as evidence.

The five-mode sweep ran 480 query executions: three queries, five modes, two
reversed-order rounds, eight iterations, and separate timing/perf processes.
Changes below are the mean of paired round median changes, relative to prefetch
off, using the same immutable binary.

| Projection budget | TPCH Q6 | ClickBench Q23 | FineWeb Q3 |
| --- | ---: | ---: | ---: |
| 4 MiB | -15.12% | +1.70% | +2.76% |
| 16 MiB | -13.52% | +6.02% | +4.35% |
| 64 MiB | -13.48% | +9.14% | +3.69% |
| 16 MiB + one experimental runtime worker | -12.15% | +11.68% | +1.50% |

The Q6 confirmation ran another 252 executions: three modes, three rounds,
fourteen iterations and separate timing/perf processes. The first two iterations
are excluded from each process's summary.

| Setting | Paired median change | Changes in individual rounds | Instructions |
| --- | ---: | --- | ---: |
| 1 MiB | -16.92% | -16.62%, -23.82%, -10.32% | -0.03% |
| 4 MiB | -14.12% | -16.59%, -22.35%, -3.43% | +0.05% |

Other checkouts ran work on this shared host. A sweep that overlapped a build was
discarded. The subsequent other Vortex benchmark used cores 0–7; a separate Python
benchmark also used the host during part of the sweep. CPU affinity does not
isolate memory bandwidth or storage. Reversed-order rounds and the longer
confirmation establish the Q6 direction here, but small effects elsewhere are
inconclusive.

Artifacts: `target/duckdb-overlap/projection-pinned/` and
`target/duckdb-overlap/projection-confirm/`, including commands, raw iterations,
perf outputs, and records. `projection-prefetch-manifest.json` records the exact
binary hash, build ID, source snapshots and patch. The worker experiment is
disabled in every retained Q6 comparison.

## Queue depth and compute overlap

Opt-in `vortex_io::read_timing` events measure time waiting for a blocking worker,
buffer allocation time, and positional read duration. These are application
requests, not the physical device queue: reads can hit the operating-system page
cache. `vortex_scan::compute_timing` records the wall duration of planner and
morsel compute steps. It excludes DuckDB SQL operators and conversion outside
those steps, and is not a measure of pure CPU time.

`scripts/summarize-scan-overlap.py` clips intervals to query callbacks after
warmup, reconstructs queued/running/pread concurrency, and reports time-weighted
mean and peak depth. Running IO includes allocation; pread excludes it. Compute
overlap is the fraction of compute-step thread-time concurrent with at least one
pread. Logging can change scheduling, so trace durations are excluded from speed
comparisons.

```bash
RUST_LOG=vortex_io::read_timing=debug,vortex_scan::compute_timing=debug,vortex_bench::query_timing=debug,warn \
VORTEX_SCAN_V2=1 VORTEX_SCAN_PROJECTION_PREFETCH_BYTES=1048576 \
target/release_debug/duckdb-bench tpch --formats vortex --threads 8 \
  --queries 6 --iterations 6 --hide-progress-bar --opt scale-factor=10.0 \
  --opt remote-data-dir=file:///home/ec2-user/votex-3/vortex-bench/data/tpch/10.0/ \
  > target/duckdb-overlap/q6.log 2>&1
python3 scripts/summarize-scan-overlap.py target/duckdb-overlap/q6.log \
  --warmup 2 --output target/duckdb-overlap/q6-overlap.json
```

The initial instrumented baseline, medians of four post-warmup executions:

| Workload | Read count | Bytes | Mean queued / peak | Mean running / peak | Compute-step overlap with pread |
| --- | ---: | ---: | ---: | ---: | ---: |
| TPCH Q6 | 171 | 397,851,508 | 0.04 / 3 | 1.01 / 4 | 56.60% |
| ClickBench Q23 | 464.5 | 738,828,764 | 1.45 / 32 | 1.66 / 12 | 48.61% |
| FineWeb Q3 | 199 | 1,539,373,776 | 0.38 / 8 | 3.76 / 8 | 96.82% |

FineWeb already overlaps almost all measured compute with pread. Increasing its
projection read-ahead does not address the limiting resource. Q23's queue can
burst while active read concurrency remains substantially lower; adding runtime
workers alone was not a reliable improvement.

A same-binary Q6 comparison with 1 MiB prefetch records the following medians of
four post-warmup traces:

| Metric | Prefetch off | 1 MiB prefetch |
| --- | ---: | ---: |
| Reads / bytes | 171 / 397,851,508 | 171 / 397,851,508 |
| Mean queued depth / peak | 0.048 / 3 | 0.126 / 5 |
| Mean running depth / peak | 0.949 / 4 | 1.203 / 5 |
| Peak compute-step depth | 8 | 8 |
| Compute-step thread-time with pread | 54.47% | 48.87% |
| Queue p95 | 91.3 us | 225.8 us |

Read concurrency increases without increasing bytes. The compute-overlap
percentage decreases and queue waiting increases: these traces do not establish
a general increase in compute/IO overlap. More projection IO is admitted earlier,
and ordinary timing shows a repeatable latency improvement. Heavy compute-step
logging alters scheduling; the traced callback medians of 66.4/62.3 ms are not the
performance comparison. Artifacts: `q6-current-{off,prefetch1}-depth-*`.

## Before/after profiles

Both Samply runs use the exact `duckdb-projection-prefetch` binary, normal Vortex,
eight threads, CPU affinity 24–31, 400 Q6 iterations, connection reuse explicitly
for profiling, 500 Hz sampling, and tracing off. Only the projection budget
changes, from zero to 1 MiB. Profiling medians are 65.412/51.964 ms, a 20.56%
reduction. Use the ordinary timing confirmation above to assess speed; profiling
adds overhead.

The seven DuckDB worker sample counts span 10,642–10,985 before and 8,958–9,476
after, with all workers participating. The dominant compute stacks persist:
12-bit FoR unpacking, primitive comparisons, rank-mask intersection, and
fixed-width filtering. On the hottest worker their self-sample shares are
15.95/10.76/10.00/6.57% before and 17.69/10.25/10.57/6.62% after. Together with
nearly identical instruction counts, this supports a scheduling/IO change rather
than removal of decode or predicate work.

The summaries symbolicate 11,835/11,836 baseline addresses and 10,891/10,891
candidate addresses in the binary. Samply warns about duplicate module start
addresses; shared-library frames remain partly unresolved. Sample weighting is
used; inflated CPU-delta values on blocking threads are not treated as CPU
utilization evidence. Command manifests record successful benchmark and recorder
exits.

```bash
samply load target/duckdb-overlap/q6-prefetch-off.profile.json.gz
samply load target/duckdb-overlap/q6-prefetch1.profile.json.gz
```

Profiles, command manifests, raw timing, and symbolicated summaries use the two
stems `target/duckdb-overlap/q6-prefetch-off` and
`target/duckdb-overlap/q6-prefetch1`. The frozen binary manifest and source snapshot
include the disabled runtime-worker experiment, which was subsequently removed.

## Rejected execution changes

A nine-mode sweep ran 864 executions comparing zero/one/two/four additional
executor workers and blocking-worker caps of four/eight/default 500. Extra
executor workers improved Q23 only about 1–2% and did not consistently improve Q6
or FineWeb. Their production code was removed. A four-worker blocking cap slowed
FineWeb by 24–31% across rounds; the existing blocking capacity is preserved.
The cross-file preparation queue stays disabled by default.

Artifacts: `target/duckdb-overlap/modes/`, `baseline-depth-*`, and
`worker1-depth-*`. The baseline Q23 Samply profile finds balanced DuckDB workers
and substantial OnPair decoding, string search, reference counting, allocation,
and bit-unpacking costs. It does not justify adding a broad decoded cache.

## Verification

Targeted benchmark builds and all completed measurement processes succeeded.
Regression cases were added for disabled prefetch, byte-budget selection, skipping
segments larger than the remaining budget, skipping a completely excluded chunk
despite more than 64 selection runs, and excluding zone-pruning reads.
Unit tests, formatting, and linting were not run, following the repository's
instruction to leave those checks to the user.

## Final-code comparison

The final implementation removes the runtime-worker experiment and checks
projection chunks directly against the current mask, including fragmented masks
with more than 64 runs. A fresh targeted build passed. The frozen final binary is
`target/duckdb-overlap/duckdb-final-projection`, SHA256
`45642735e1c30cf7cd2654c69003267356c43c4e45df6675506d1e4b002597c0`, build ID
`02159bc31f8d321737168de6c347888138846d9c`.

The final comparison uses that same binary with prefetch zero/1 MiB, three
alternating rounds, fourteen iterations per query per process, and separate perf
processes: 504 query executions. Two warmups are excluded. Unlike the preceding
tables' mean of round changes, the helper's paired result is the median of round
changes; pooled times are medians of all post-warmup iterations.

| Query | Pooled baseline / 1 MiB (ms) | Paired change | Individual round changes | Mean instruction change |
| --- | ---: | ---: | --- | ---: |
| TPCH Q6 | 57.753 / 47.114 | -19.12% | -14.49%, -26.35%, -19.12% | +0.32% |
| ClickBench Q23 | 103.690 / 106.018 | +0.31% | -1.96%, +4.42%, +0.31% | +0.18% |
| FineWeb Q3 | 81.216 / 81.804 | +0.62% | +0.62%, +0.41%, +1.28% | -0.33% |

Q6 improves in every round. Q23's pooled and paired estimates differ because
timing varies between rounds, so it does not establish a gain. FineWeb changes
are small but consistently positive. Prefetch remains opt-in.

Artifacts: `target/duckdb-overlap/final-comparison/` contains every command and
environment, raw timing, perf measurements, and `summary.json`.
`final-projection-manifest.json`, `final-projection.patch`, and
`final-projection-source/` capture the binary and its source snapshot. Subsequent
report edits do not change production code.

The final binary's two 400-iteration Q6 Samply runs show 65.818 ms with prefetch
off and 55.302 ms with 1 MiB, a 15.98% reduction. These runs have the same profile
configuration as above. All 11,799 baseline and 11,231 candidate binary addresses
symbolicate. The seven worker sample counts span 10,803–11,235 before and
9,644–9,855 after; decode, comparison, mask, and filtering stacks remain dominant.
Both benchmark and recorder exits are successful. The duplicate-module warnings
and CPU-delta limitations above still apply.

```bash
samply load target/duckdb-overlap/final-q6-off.profile.json.gz
samply load target/duckdb-overlap/final-q6-prefetch1.profile.json.gz
```

`final-q6-{off,prefetch1}.summary.txt` contains symbolicated stack summaries;
matching command manifests and raw iterations accompany each profile.

The last diagnostic refinement captures elapsed compute time before the event's
wall-clock timestamp. The previous ordering could reconstruct overlapping steps
on one thread: eight intervals in the final candidate trace exceeded eight
concurrent steps, totaling 7,336 ns and lasting at most 2,491 ns each. This did not
affect the tracing-disabled timing or profile runs. The refined diagnostic build
passed and is frozen as `duckdb-final-timing` (SHA256
`047b036cb7a03d357fdebcf6223a8614d9ef3f6490f3dc6b488a58644477fb94`, build ID
`a347b25d620931ab92debe545bc85dda0a413c28`). The scan execution and prefetch policy
are unchanged.

The corrected Q6 traces, medians of four post-warmup executions:

| Metric | Prefetch off | 1 MiB prefetch |
| --- | ---: | ---: |
| Reads / bytes | 171 / 397,851,508 | 171 / 397,851,508 |
| Mean outstanding depth / peak | 1.003 / 4 | 1.326 / 5 |
| Mean queued depth / peak | 0.046 / 3 | 0.119 / 5 |
| Mean running depth / peak | 0.962 / 4 | 1.207 / 5 |
| Peak compute-step depth | 8 | 8 |
| Compute-step thread-time with pread | 52.85% | 47.43% |
| Queue p95 | 91.3 us | 289.5 us |

`outstanding_io` includes queued and active requests; its peak is measured from
the combined timeline, not by summing separate peak values. Corrected trace
artifacts are `target/duckdb-overlap/precise-{off,prefetch1}-depth-*`. This final
diagnostic change has its own manifest and source snapshot; the profiles above
remain tied to their exact earlier binary.

After the diagnostic refinement, another three-round Q6 confirmation runs the
exact `duckdb-final-timing` binary with tracing disabled and budgets zero/1 MiB.
Fourteen iterations per process and separate perf processes total 168 query
executions, excluding two warmups for each summary. The pooled medians are
60.082/48.019 ms, a 20.08% reduction. The paired median change is -20.09%, with
individual round changes -12.08%, -21.06%, and -20.09%. Instructions change
-0.14%, cycles -0.30%, and task-clock -2.18%, using medians across perf processes.
This independently confirms the speedup after the final source change. Artifacts:
`target/duckdb-overlap/final-timing-confirm/`.

## Extension BETWEEN continuation: 2026-10-07

### Source mechanism

DuckDB's `Filter::new` binds and recursively optimizes its filter expression.
The expression optimizer combines compatible literal bounds such as Q6's
`l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'` into BETWEEN.
Extension arrays previously delegated binary comparisons to storage, but did not
delegate BETWEEN. The canonical BETWEEN fallback rebuilt two comparisons and a
Kleene AND. An extension array is canonical even when its storage remains
compressed, so the comparisons can independently decompress that storage.

The new `BetweenReduce for Extension` unwraps compatible constant or extension
bounds and retains one BETWEEN over storage. It checks both bounds' extension
metadata before unwrapping. This is a metadata-only rule: it reads no buffers,
does not force storage decoding, and allows existing storage kernels to dispatch.
For Q6's FoR storage, the storage BETWEEN fallback canonicalizes once, then runs
the existing primitive range kernel. Global BitPacked storage can instead use
its existing streaming BETWEEN kernel. Other bound encodings keep their fallback.
No executor, scheduler, cache, or IO-capacity change is included in this candidate.

Regression source covers the four strictness combinations, null values, nullable
bounds on non-nullable values, row-wise timestamp bounds, one or two null
constant bounds, and mismatched timestamp units/timezones on either bound.
These cases are added but not run. The focused `between_extension` benchmark
compares BETWEEN with explicit comparisons and AND over identical primitive date
storage at 8,192 and 262,144 rows, with and without nulls.

### Binaries and comparison method

Both binaries were built with
`cargo build --locked --profile release_debug -p duckdb-bench -j 16`.
The only production difference is extension BETWEEN delegation and its static
parent-rule registration; benchmark and regression source are not linked into
DuckDB. Frozen binaries and source manifests:

| Role | Binary | SHA-256 | Build ID |
| --- | --- | --- | --- |
| Before | `target/duckdb-overlap/duckdb-final-timing` | `047b036cb7a03d357fdebcf6223a8614d9ef3f6490f3dc6b488a58644477fb94` | `a347b25d620931ab92debe545bc85dda0a413c28` |
| After | `target/duckdb-overlap/duckdb-extension-between` | `a692b57c7484b2925b764dd50d3a074f05b8752252e283847766c4968f612236` | `abd10aa96c7d06415c19bb2b1ca401f2a401edd0` |

All comparisons use normal Vortex files, TPCH SF10, eight DuckDB threads pinned
to CPUs 24–31, cross-file preparation zero, fresh connections, two excluded
warmups, alternating label order, tracing disabled, and separate perf processes.
Other workspaces intermittently used CPU and memory bandwidth. Paired timings
and instruction counts are more useful here than pooled timings alone; the host
was not fully isolated.

A fresh same-binary control used two rounds, ten iterations, and budgets 1 MiB
for both labels. Paired timing differences were Q6 -0.95%, Q19 -0.88%, Q23 -1.00%,
and FineWeb Q3 -0.02%. Q19's pooled medians misleadingly differed by -10.49%.
Artifacts: `target/duckdb-overlap/continue-same-binary/`.

### Incremental results with 1 MiB read-ahead

The first comparison used three rounds and fourteen iterations; the repeat used
four rounds and twenty iterations. Counters are medians across separate perf
processes. Each comparison holds read-ahead fixed between the two labels.

| Query / comparison | Paired time change | Instructions | Cycles | Task-clock |
| --- | ---: | ---: | ---: | ---: |
| Q6, first | -10.23% | -21.12% | -16.47% | -11.70% |
| Q6, repeat | -6.37% | -21.09% | -19.15% | -15.94% |
| Q19, first | -0.44% | +0.27% | +0.55% | +0.18% |
| Q23, first | -1.15% | +3.63% | +2.42% | +1.72% |
| FineWeb Q3, first | +9.12% | +0.08% | +1.20% | +1.59% |
| FineWeb Q3, repeat | +2.36% | -0.18% | +4.15% | +6.57% |
| FineWeb Q3, final isolated workload | -0.06% | +0.18% | +0.06% | -0.19% |

Q6's first three round changes were -12.22%, -10.23%, and -8.23%. Its repeat
rounds were +7.32%, -3.47%, -10.33%, and -9.28%. The instruction reduction is
stable despite timing drift. FineWeb's repeat rounds were +12.51%, +4.38%,
+0.34%, and -0.67%. FineWeb Q3 has string predicates and does not exercise
extension BETWEEN. A final FineWeb-only four-round comparison with twenty
iterations produced round changes +1.11%, -6.13%, -1.22%, and +6.80%; the paired
median was -0.06%, with nearly unchanged counters. The earlier slowdown did not
reproduce consistently; there is no established FineWeb gain or regression.
Q19 and Q23 establish no additional win.

Artifacts: `target/duckdb-overlap/extension-between-compare/` and
`target/duckdb-overlap/extension-between-repeat/`.
The final FineWeb comparison is in
`target/duckdb-overlap/extension-between-fineweb-final/`.

### Default read-ahead disabled

Two paired rounds with eight iterations per process, and both budgets zero:

| TPCH query | Before / after pooled median, ms | Paired time change | Instructions | Task-clock |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 145.770 / 143.738 | -4.49% | -0.73% | -1.03% |
| Q6 | 42.513 / 39.253 | -7.15% | -21.02% | -10.09% |
| Q7 | 146.461 / 141.791 | -2.00% | -4.46% | -2.87% |
| Q10 | 237.966 / 237.848 | -0.19% | -0.42% | -0.20% |
| Q12 | 85.196 / 82.915 | -2.60% | -7.42% | -4.34% |
| Q14 | 64.565 / 60.388 | -6.25% | -16.20% | -6.35% |
| Q20 | 144.501 / 143.215 | -1.30% | -0.03% | +0.15% |

Q6, Q12, and Q14 improved in both rounds with corresponding instruction savings.
Q4 and Q20's timing changes have much smaller instruction changes and should
not be treated as established compute gains. This sweep measures DuckDB;
no new DataFusion performance claim is made for the shared array rule.
Artifacts: `target/duckdb-overlap/extension-between-default-dates-fixed/`.

The initial multi-query invocation aborted after baseline timing because commas
in its FIFO filename confused perf's `--control=fifo:path,ack` syntax. Those
artifacts in `extension-between-default-dates/` are excluded from comparisons.
The helper now replaces commas in artifact names and splits explicit query
lists into individual cases when perf is enabled, collecting per-query counters.

### Matching profiles and IO depth

Both frozen binaries were recorded at 500 Hz for 400 Q6 iterations with eight
threads and 1 MiB read-ahead. Connection reuse is enabled only in these profiles
to preserve worker threads. Profile medians after two warmups were
45.814 / 41.798 ms (-8.77%). Each profile is symbolicated against its own binary;
10,524 / 10,524 before and 10,142 / 10,142 after binary addresses resolved.

The stacks show the expected dispatch change: before, FoR decompression sits
under binary comparison; after, it sits under storage BETWEEN. On a representative
heavy worker, 12-bit FoR unpacking falls from 16.69% to 11.94% self samples.
The old i32 constant comparison occupied 10.68%; the new i32 BETWEEN kernel
occupies 4.85%. These are individual-worker sample fractions, not global cycle
attribution. Separate perf counters above establish the CPU/instruction savings.
The seven background DuckDB workers remain balanced: 8,109–8,543 samples before
and 7,495–7,872 after. Remaining prominent costs are FoR unpacking, portable rank
scattering, fixed-width filtering, other bit unpacking, and allocation.

Artifacts: `extension-before.profile.json.gz`, `extension-after.profile.json.gz`,
their `.profile.command.json` files, and their `.summary.txt` files in
`target/duckdb-overlap/`.

Separate diagnostic runs use six iterations, with four post-warmup query windows,
the same budgets and exact binaries, and all three timing targets enabled. Values
below are medians across those four windows; peaks are maxima.

| Q6 diagnostic metric | Before | After |
| --- | ---: | ---: |
| Positional reads per query | 171 | 171 |
| Read bytes per query | 397,851,508 | 397,851,508 |
| Outstanding IO, mean / peak | 0.988 / 5 | 1.002 / 5 |
| Queued IO, mean / peak | 0.144 / 5 | 0.110 / 5 |
| Running IO, mean / peak | 0.847 / 5 | 0.898 / 5 |
| Compute steps, mean / peak | 2.466 / 8 | 2.063 / 8 |
| Compute-step thread-time, ms | 123.910 | 96.454 |
| Running-read thread-time, ms | 42.338 | 41.448 |
| Compute thread-time overlapping pread | 45.06% | 45.58% |
| Queue duration p95, microseconds | 207.7 | 167.3 |

The change removes compute work while preserving read volume and observed peak
parallelism. It does not establish a larger overlap percentage or deeper queue.
Compute-step thread-time measures wall time inside planner/morsel compute calls;
it excludes DuckDB execution and export outside those calls and is not pure CPU
time. These are application IO queue depths over normal OS-page-cached reads,
not physical device queue depths. Diagnostic timing is excluded from the paired
latency comparisons.

Artifacts: `extension-{before,after}-depth-tpch-6.{log,overlap.json,command.json}`
and `extension-between-depth-summary.json` in `target/duckdb-overlap/`.

Reproduce the default-off comparison with the retained frozen binaries:

```bash
taskset -c 24-31 python3 scripts/bench-v2-drivers.py \
  --baseline duckdb=target/duckdb-overlap/duckdb-final-timing \
  --candidate duckdb=target/duckdb-overlap/duckdb-extension-between \
  --baseline-projection-prefetch-bytes 0 \
  --candidate-projection-prefetch-bytes 0 \
  --data-root vortex-bench/data --output target/duckdb-overlap/extension-reproduction \
  --workload tpch:4,6,7,10,12,14,20 --formats vortex --threads 8 \
  --prefetch 0 --iterations 8 --warmup 2 --rounds 2 --perf-stat
```

### Focused primitive-storage benchmark

The final benchmark source is tagged with `cpu_features`, matching the existing
primitive benchmark convention. It prepares each lazy predicate before timing,
creates fresh execution contexts, and measures execution on CPU 24. Both paths
clone an array handle for each execution. Nulls occur every eleventh row in the
nullable cases; both cases use the same date values and bounds. Results are
native wall-time medians, with a one-second minimum per case:

| Rows | Nullable | Two comparisons + AND, microseconds | Delegated BETWEEN, microseconds | Speedup |
| --- | --- | ---: | ---: | ---: |
| 8,192 | No | 5.383 | 1.368 | 3.93x |
| 8,192 | Yes | 6.132 | 1.442 | 4.25x |
| 262,144 | No | 80.540 | 27.670 | 2.91x |
| 262,144 | Yes | 92.130 | 27.850 | 3.31x |

This benchmark uses primitive storage; it measures fewer predicate passes,
allocations, and dispatches, rather than FoR decompression or IO. The SQL
profiles provide the compressed-storage evidence. Build and reproduce the
microbenchmark on `ji/between-remaining-encodings` with:

```bash
taskset -c 24 cargo bench --locked --profile release_debug \
  -p vortex-array --bench between_extension -- \
  --color never --sample-count 2000 --sample-size 1 --min-time 1 --max-time 2
```

The directly executed binary requires `--bench`; without it this harness only
executes each benchmark once and emits no timing statistics. Final artifacts:
`between-extension-micro-final`, its `.rs` source snapshot,
`extension-between-micro-final-command.json`, and
`extension-between-micro-final.txt` in `target/duckdb-overlap/`.

### Checks and remaining targets

The DuckDB release build and focused benchmark build succeeded. Paired SQL
timing/perf comparisons, matching Samply recordings, diagnostic IO traces, and
the primitive-storage benchmark completed. The first microbenchmark build had
an array-handle ownership error; cloning the handle fixed it before measurement.
The five completed SQL timing/perf comparisons, including the same-binary
control, cover 2,400 query executions in 184 processes, with 2,032 post-warmup
executions. The profiles add 800 query executions and the IO traces add twelve;
these diagnostic executions are excluded from paired latency comparisons.
Unit tests, formatting, and lint checks were not requested and were not run.
This report accompanies the optimization checkpoint on PR #10339.

The retained rule is intentionally small. The next source targets supported by
the new Q6 profile are FoR range execution without full primitive allocation,
portable mask scattering, and fixed-width row filtering. A FoR range kernel
must handle signed wrapping addition, per-block references, patched values, and
sliced offsets. Existing BitPacked streaming predicates and FoR decompression
offer useful implementation components, but a reference-subtraction shortcut
would not preserve all those semantics. No unmeasured FoR kernel or scheduler
policy change is included here.
