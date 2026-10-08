<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Scan V2 IO driver investigation

## Direct Fetch futures: measured improvement

The new direct-storage Fetch future improves warm DataFusion ClickBench Q23 by 2.83%, from
223.395 to 217.072 ms, winning seven of eight paired round medians. TPC-H Q19 is effectively
unchanged at 27.523 versus 27.510 ms, with four of eight round wins. This path is now the source
default; `VORTEX_SCAN_IO_INLINE_FETCH=0` restores the boxed control. Early projection hints
remain enabled because delaying them makes Q23 8.27% slower in this larger comparison.

The comparison uses one release-debug binary with SHA-256
`eb1b2dabf031d37a02d0ff074b2b136c6a5e0d5f584cea04ce23cf2c1776f2f8`. Each configuration has
56 retained timing samples per query: eight balanced rounds, nine iterations per process, and
two warmups discarded. All 16 comparison rounds complete without a detected competing job.
Event logging is disabled during timing; scan-counter snapshots follow the query timer. All
timings finish before the separate diagnostic processes. Quiet gating and repeated whole rounds
use the same criteria as the earlier experiment below.

| Configuration | Q19 median ms | Q19 min–max ms | Q23 median ms | Q23 min–max ms |
|---|---:|---:|---:|---:|
| Early hints, boxed Fetch | 27.523 | 26.064–131.503 | 223.395 | 196.265–250.330 |
| Delayed hints, boxed Fetch | 27.798 | 26.280–36.374 | 241.879 | 201.372–348.619 |
| Early hints, direct Fetch | 27.510 | 25.933–133.873 | 217.072 | 197.764–243.659 |

Individual outliers remain included. This supports a modest Q23 improvement on these workloads,
not a general speedup for every query. All measured iterations return one Q19 result row or ten
Q23 rows; cardinality is not a value checksum. The targeted scan tests compare actual arrays.

| Configuration | Q19 median completed reads | Q19 median completed MB | Q23 median completed reads | Q23 median completed MB |
|---|---:|---:|---:|---:|
| Early hints, boxed Fetch | 182 | 507.162 | 417 | 5081.728 |
| Delayed hints, boxed Fetch | 351.5 | 550.116 | 3854 | 5736.626 |
| Early hints, direct Fetch | 182 | 507.162 | 414 | 5016.773 |

Q19's physical work is identical for the storage comparison. Q23's completed-byte median also
falls by 1.28%, so the entire observed improvement cannot be assigned to allocator CPU cost.
Its LIMIT 10 cancellation schedule can change when work completes sooner. These are after-query
counter snapshots, not device traffic. Delayed hints win two of eight Q19 round medians and
none of eight Q23 rounds, confirming the first experiment's increased IO work.

Every logical Fetch previously allocated a boxed mapping future in addition to the task that
`FuturesUnordered` allocates. `InlineFetch` stores the shared bytes, owner, and request directly
in that task. Generic source types keep the boxed and direct-storage task layouts separate,
preserving the control's original task size. Both use the same shared read, error conversion,
completion protocol, wake forwarding, root cleanup, and registration lifetimes. The
`inline_fetches` scope counter measures wrapper allocations avoided in diagnostics; cancelled
Fetches count too. No ready queue or additional wake protocol is introduced.

The separate direct-storage diagnostics count 4880 wrapper allocations avoided in Q19 and
129318 in Q23. All six boxed/direct/delayed diagnostic reports have no analyzer warnings and
no unmatched session poll events. The direct Q23 capture starts and completes 414 physical
requests covering 4282.530 MB. Its last-reference drops occur after physical completion for
25273 registrations, during a completed read for 1313, before read start for 24, and without
a started read for 11689. The 3084 whole-root rejections reach actual root IO clear in a median
0.0942 ms, p95 0.3392 ms, and maximum 0.9275 ms. This supports keeping the shared-interest
lifetime contract: root retirement already releases interests promptly, while partial branches
can have live siblings and other sessions can share a registration. Earlier segment withdrawal
needs a precise lease contract and evidence of work that it can still prevent.

The delayed-hint Q23 diagnostic starts 3623 physical requests and records about 1700.58 MB of
repeated local ranges, compared with 166.23 MB for direct Fetch storage with early hints.
This explains the failed hint-delay experiment through fragmented and repeated IO. Diagnostic
logging changes scheduling and cancellation; these byte inventories and waits are observations
of each capture, not the untraced speed comparison above.

Reproduce with:

```bash
cargo build -p datafusion-bench --profile release_debug
python3.11 scripts/bench-io.py --data-root vortex-bench/data \
  --output target/scan-io-results/inline-fetch-confirm \
  --workloads tpch-q19 clickbench-q23 \
  --variants v2-project-early v2-project-late v2-inline-fetch --modes direct \
  --iterations 9 --rounds 8 --diagnostic-iterations 1 --timing-metrics \
  --driver-trace --driver-trace-mode json --lifetime-trace \
  --wait-for-quiet --repeat-contended-rounds
```

The harness explicitly sets `VORTEX_SCAN_IO_INLINE_FETCH=0` for both hint controls and `=1`
for the new path. All three use Scan V2, unlimited local admission, and deferred physical
selection. All six captures completed before rebuilding the executable with the new default.
The final binary and unset-environment validation are recorded separately below.

## Projection-hint experiment and actual IO lifetimes

Delaying projection-only announcements until after filtering does not improve these scans.
The first quiet comparison retains 20 timing samples per configuration and query: four balanced
rounds, seven iterations per process, and two warmups discarded. Driver logging is disabled for
timings; `--timing-metrics` snapshots scan counters after each query timer stops. No comparison
round overlapped a detected competing job. Binary SHA-256 is
`d446825c91d819e013d29de06b9465d4c3f299a5e46164926dedc3ac4d76a66d`.

| Projection hints | Q19 median ms | Q19 median completed reads | Q19 median completed MB | Q23 median ms | Q23 median completed reads | Q23 median completed MB |
|---|---:|---:|---:|---:|---:|---:|
| Early, default | 27.778 | 182 | 507.162 | 218.287 | 418.5 | 5008.880 |
| After filtering | 28.289 | 388.5 | 550.137 | 234.340 | 3948 | 5824.902 |

The delayed variant is 1.84% slower on Q19 and 7.35% slower on Q23. It wins three of four Q19
round medians but none of four Q23 rounds. Q19's individual samples range from 26.53–66.31 ms
for early hints and 26.01–31.51 ms for delayed hints; Q23 ranges are 190.79–235.67 ms and
214.50–252.25 ms. Every measured iteration returns one Q19 result row or ten Q23 rows. Counters
are completion snapshots; cancellation can affect their values. These samples use counter output
and a new binary, so they are a separate comparison from the older factorial results below.

The first gate waited from approximately 23:24 UTC on October 6 to 00:14 UTC on October 7,
2026. At admission, one-minute load was 0.24, CPU busy fraction was 0.0406%, and IO full
pressure averaged 0%. The harness requires six consecutive quiet ten-second observations and
repeats entire rounds when competing builds or benchmarks are detected. All timing processes
finish before the separate state-machine diagnostics.

The Q19 diagnostics show the physical mechanism: early hints use 182 bound physical reads and
507.162 MB; delayed hints use 399 and 558.287 MB. Including metadata reads, the local read log
reports 51.13 MB of repeated ranges for delayed hints, versus zero for early hints. The number
of fresh registrations is 2101 in both captures. This demonstrates lost coalescing without
requiring a larger registration inventory. Early announcements also hold shared registrations
across interested scopes, so dropping every projection hint is not a safe performance shortcut.
The early policy remains the default; `VORTEX_SCAN_PROJECT_ANNOUNCE=0` retains the experiment.

The early Q23 diagnostic has 768620 actual scope interests and 37255 last registration drops.
660659 interests are never fetched by that scope, summing to 91.826 GB of **logical** ranges.
Shared and overlapping interests make that sum much larger than physical traffic; it is not
memory usage or avoidable IO. Drop phases distinguish 12344 registrations absent from started
reads, 122 released before a physical read starts, 1320 released during a completed read, and
23469 released afterward. Completed registration retention p95 is 743.79 ms. Forwarded
IO-wake-to-next-session-poll p95 is 386.79 ms; this includes redundant wakes and trace overhead.

2818 of 3638 observed filter decisions reject every remaining row. Each covers its whole
admitted root, and each has an observed IO clear: rejection-to-clear median is 0.0935 ms,
p95 0.3346 ms, and maximum 0.8128 ms. Existing root cleanup therefore withdraws its interests
promptly in this capture. Of the never-fetched registrations whose scopes all completed, 4383
entered physical reads, covering 670.22 MB exclusive of other members. 4280 started before
the last interested scope cleared. Earlier per-segment pruning still needs a precise lease
contract, and most of this captured work was already started by the time root cleanup ran.

This trace spans 3029.34 ms and includes 124176 Fetch publications. That allocation inventory
motivated the direct-storage change measured above, which preserves the completion/wake
protocol while removing the separate boxed Fetch wrapper.

Artifacts: `target/scan-io-results/projection-hints-pass1/summary.json`, its raw timing and
diagnostic files, and `quiet-samples.jsonl`. All eight timing rounds completed; the final delayed
Q23 diagnostic remained queued behind competing host work and was subsequently captured in
the completed larger comparison. The original executable is retained at
`target/scan-io-results/binaries/datafusion-projection-first`. Commands and new lifetime/selection definitions are
in the [measurement README](README.md).

## Validation

The final release-debug benchmark build passes. There are 49 distinct passing targeted Rust
cases: nine layout-planning cases, 32 scan-planning cases, and eight file IO cases. The file
cases also pass with the direct-storage candidate explicitly enabled and after making it the
unset-environment default; boxed/direct future cases verify pending, wake, completion, and error
forwarding. All 33 Python analyzer/harness tests, Ruff, Python type checks, and pinned Rust
formatting pass. Strict scoped File and Scan Clippy pass. Strict Layout Clippy encounters
pre-existing lint failures outside modified code; a check with those six lint classes temporarily
allowed passes. No repository lint allowances were added. Logs and binary validation metadata
are under `target/scan-io-results/`.

The final default binary has SHA-256
`3a918a5389ee072818013a43f49c12ce653efc036f468eb0d12d524f210ca343`. With only Scan V2
enabled and every optimization control unset, five Q19 iterations return one row each and five
Q23 iterations return ten rows each. A separate Q19 lifetime capture counts 4933 directly stored
Fetches, confirming the unset default actually selects the new implementation. The final quiet
smoke gate was interrupted after repeated competing builds; its host observations remain saved.
These final runs validate behavior and their timings are excluded from performance evidence.
The earlier 56-sample comparisons remain the speed measurement. Final metadata is in
`target/scan-io-results/inline-fetch-validation.json` and
`target/scan-io-results/inline-fetch-default-functional/summary.json`.

## Earlier physical-selection and admission comparisons

The quiet-window warm rerun does not show an end-to-end speedup from either scheduling change.
Deferred/unlimited has a median 0.94% higher on Q19 and 2.43% higher on Q23 than eager/unlimited;
it wins two of eight paired Q19 round medians and three of eight Q23 round medians. The
32-read admission limit also provides no warm-query improvement in this rerun. The controlled
late-registration regression still demonstrates fewer physical requests with deferred selection,
but these query measurements do not establish a throughput gain.

The rerun waited from approximately 19:49 to 21:09 UTC on 2026-10-06. It started after six
consecutive 10-second intervals with CPU usage below 10%, IO wait below 2%, full IO pressure
below 1%, one-minute load below 4, and no detected competing builds or benchmarks. At the
gate, load was 1.49, CPU usage was 0.069%, IO wait was 0.009%, and IO full pressure averaged
0.08% over 10 seconds. The same final binary was used for all configurations, with SHA-256
`b1015beff416475554384f62ff4597b259abd4163de6e98ca3d6ba5aa1ac2c7b`.

Each warm configuration ran seven iterations per process in eight balanced rounds, discarding
the first two iterations per process: 40 retained samples per configuration and query. All
timing runs precede diagnostics. Two Q23 attempts overlapped newly started benchmark/profiler
jobs; the entire comparison rounds were retained as rejected attempts and rerun after the host
settled. The tables include only accepted rounds. Detection samples process names and benchmark
Python command paths every 250 ms; it is a contention check, not exclusive ownership of the host.
Host CPU and IO counters include the benchmark's own work, and one-minute load rises during
the measurement even when no competing process is detected.

| Configuration | Q19 median ms | Q19 min–max ms | Q23 median ms | Q23 min–max ms |
|---|---:|---:|---:|---:|
| Eager / unlimited | 27.073 | 26.055–136.260 | 217.435 | 195.824–240.446 |
| Deferred / unlimited | 27.328 | 25.456–104.748 | 222.725 | 204.903–251.562 |
| Eager / 32 | 27.217 | 25.445–123.998 | 221.669 | 201.068–253.320 |
| Deferred / 32 | 27.173 | 25.602–95.814 | 221.845 | 182.953–248.030 |

Q19 round medians range from 26.44 to 28.40 ms across configurations, but occasional slow
samples remain. Its retained-sample p95 is 105.07 ms for eager/unlimited and 97.85 ms for
deferred/unlimited. Q23 p95 is 236.47 and 247.68 ms respectively. No individual timing outlier
was removed. Small median differences should be interpreted alongside those tails and the
paired round results, rather than as a stable speedup or universal regression estimate.

The quiet cold Q19 comparison uses eight balanced rounds, one iteration per process, and
per-file `posix_fadvise(DONTNEED)` before each process. Two rounds overlapping a build or
file copy were retained as rejected attempts and repeated. Eviction remains advisory.

| Configuration | Median ms | Min–max ms | Diagnostic local reads | Application bytes MB | Repeated bytes MB |
|---|---:|---:|---:|---:|---:|
| Eager / unlimited | 4399.504 | 3430.433–4403.780 | 185 | 507.76 | 0 |
| Deferred / 32 | 4399.437 | 4392.092–4409.154 | 185 | 507.76 | 0 |

The medians differ by just 0.068 ms. Deferred/32 wins four of eight paired rounds. The
3430 ms baseline sample remains included. Separate cold diagnostics show median `pread`
falling from 1503.79 to 672.73 ms with the cap, while median admission rises from effectively
zero to 624.73 ms. Blocking-pool queue medians are 4.80 and 6.46 microseconds. The cap moves
wait before allocation and blocking submission; it does not reduce end-to-end time here.
Both diagnostics make 182 bound physical scan reads plus three local metadata reads, with
no repeated ranges. The initial noisy capture's duplicate-read reduction is absent in this
rerun. This cold comparison combines deferred selection and admission; it does not isolate
deferred/unlimited's cold behavior.

Separate driver traces were collected after all timing runs. Their logging changes elapsed
times and the Top-K cancellation schedule, so their wall times and byte inventories are not
untraced throughput comparisons. Within each captured query, the shared clock distinguishes
physical IO from runnable work and dependency waits. For deferred/unlimited:

| Query | Traced window ms | Compute union ms | Ready or active union ms | Physical read union ms | IO/compute overlap ms |
|---|---:|---:|---:|---:|---:|
| Q19 | 164.25 | 69.95 | 103.73 | 133.92 | 63.97 |
| Q23 | 2497.50 | 1002.31 | 2303.57 | 2471.20 | 1002.30 |

The Q19 trace has local `pread` p95 of 1.26 ms, GET preparation p95 of 47.07 ms, blocking-pool
queue p95 of 0.13 ms, Fetch p95 of 74.34 ms, dispatch wait p95 of 35.19 ms, and delivery delay
p95 of 32.39 ms. Those distributions describe different intervals and must not be added.
FilterPlanner sums to 175.22 ms of compute across workers, while ProjectionMorsel sums to
10.81 ms. FilterPlanner's ready-to-compute p95 is 0.191 ms; ProjectionPlanner's is 0.629 ms.
All four Q19 traces request the same 182 physical scan reads, 507.16 MB of scan payload, and
three unbound local metadata reads. Every physical member is fetched; no physical read or
logical Fetch remains unfinished at the boundary.

In Q23, deferred/unlimited's local `pread` p95 is 2.33 ms and blocking-pool queue p95 is
0.14 ms, while GET preparation p95 is 129.88 ms, Fetch p95 is 338.39 ms, dispatch wait p95
is 65.60 ms, and delivery delay p95 is 189.30 ms. Runnable or active work occupies 92% of the
captured query window; this is not an interval in which every scan stage is simply waiting
for disk. ProjectionPlanner's ready-to-compute p95 is 14.12 ms, versus 0.238 ms for
FilterPlanner. FilterPlanner's fetch p95 is 467.26 ms and ProjectionMorsel's is 328.58 ms.
Their summed compute is 652.41 and 827.45 ms respectively. This separates waiting for an IO
dependency from an available stage waiting to be visited. It identifies scheduling and
preparation in this instrumented run, rather than establishing their cost without tracing.

Q23's LIMIT 10 cancellation produces different speculative work in each isolated trace:

| Configuration | Started physical reads | Started request MB | Unfinished physical reads | Physical members not fetched in window |
|---|---:|---:|---:|---:|
| Eager / unlimited | 373 | 4064.87 | 16 | 6700 |
| Deferred / unlimited | 372 | 3852.74 | 5 | 6349 |
| Eager / 32 | 355 | 4005.41 | 4 | 6857 |
| Deferred / 32 | 443 | 4525.17 | 4 | 7476 |

Started request bytes include unfinished work and are distinct from completed local read
bytes. Deferred/unlimited has fewer unfinished requests than baseline in these captures,
but deferred/32 starts more physical reads and bytes. There is no monotonic reduction in
speculative work across policies. Query cancellation and tracing affect these inventories;
they do not demonstrate a general byte reduction. All logical Fetch requests bind to physical
registrations. There are 181–185 unbound local completions, which remain separately counted.
Unfinished physical reads, undelivered Fetch requests, and parked intervals are censored at
cancellation or the observed query boundary, rather than treated as successful completions.

The quiet warm artifacts are in `target/scan-io-results/quiet-factorial/summary.json`.
Every accepted and rejected sample, process-overlap record, round median, command, runtime
control, and host snapshot is retained. The gate samples and automatic rerun procedure are in
`target/scan-io-results/quiet-rerun-setup/`; the cold results are in
`target/scan-io-results/quiet-cold-q19/summary.json`.
All eight fresh driver reports have no consistency warnings and report the expected result
row counts (one for Q19 and ten for Q23). Artifact checks verify eight accepted rounds per
configuration, 40 retained warm samples or eight cold samples, recomputed medians, zero
unbound logical Fetch requests, and saved driver reports and timelines. These row counts
are not value checksums. The final binary's SHA-256 was verified again after the rerun.
No Rust source changed during this rerun; Rust build, lint, and test checks were not repeated.

The saved quiet-rerun logs were subsequently reprocessed offline to add queue depth, stage
startup, reconstructed DFS depth, byte coverage, and batch shape. This did not execute queries
again or change the accepted timing samples. The expanded reports and compact queue timelines
are in `target/scan-io-results/driver-depth/`; `summary.json` identifies the original binary,
analyzer hash, and source logs. All eight reports retain the original physical read/byte counts,
compute totals, query windows, IO/compute overlap, result cardinalities, and logical Fetch
binding coverage. They have no consistency warnings or unknown DFS owners/compute steps.

For the production scheduling policy, deferred/unlimited, startup from query begin is:

| Milestone | Q19 ms | Q23 ms |
|---|---:|---:|
| First scan compute | 6.586 | 25.626 |
| First logical Fetch request | 6.837 | 25.646 |
| First physical range submitted | 7.255 | 26.299 |
| First bound local pread starts | 7.975 | 28.001 |
| First bound local pread finishes | 8.134 | 28.162 |
| First physical completion observed by the file driver | 8.151 | 178.012 |
| First Fetch delivered | 8.161 | 178.038 |
| First projection planner compute | 9.207 | 180.029 |
| First projection morsel compute | 9.234 | 193.689 |
| First morsel batch emitted | 9.976 | 258.616 |
| First batch returned by a scan driver | 9.979 | 258.821 |

The first bound local pread in Q23 finishes about 150 ms before the file driver observes any
physical completion. `read_end` is emitted when `ReadDriver` polls a result, after async read
work and delivery through the result stream. It is not the syscall's completion timestamp.
This is evidence to investigate async resume, result-channel delivery, and driver polling;
the trace does not attribute that entire interval to one of them. Bound local startup excludes
metadata completions without physical bindings. First returned scan batch also differs from
the query's first externally visible result, especially with aggregation or sorting.

Queue depth below is weighted by elapsed time over the whole captured query, including zero.
Runnable work excludes time spent inside `compute()` and counts only `NeedsCompute` owners.
Parked and undelivered work is retained up to cancellation or the observed boundary.

| Queue | Q19 mean / p95 / peak | Q23 mean / p95 / peak |
|---|---:|---:|
| Physical ranges in flight | 27.01 / 78 / 124 | 16.10 / 34 / 43 |
| Selected physical ranges waiting to start | 11.11 / 96 / 121 | 0.72 / 4 / 15 |
| Runnable planners | 5.71 / 16 / 31 | 4.89 / 12 / 22 |
| Runnable morsels | 1.60 / 11 / 22 | 9.84 / 28 / 42 |
| Parked planners | 151.79 / 449 / 473 | 331.93 / 980 / 1029 |
| Parked morsels | 168.06 / 449 / 473 | 42.64 / 126 / 145 |
| Logical Fetch requests outstanding | 707.77 / 1209 / 1268 | 5095.10 / 15411 / 17733 |
| Physically complete Fetch requests awaiting driver delivery | 231.71 / 883 / 1263 | 2441.19 / 11532 / 15562 |

Physical-range depth includes preparation, admission, and async/result delivery. It is not
device queue depth. Q23's completed, bound local pread depth has mean 0.18 and peak 16,
whereas its GET-preparation depth has mean 4.92 and peak 28. Those local counters omit reads
without a completion-phase log. The physical request byte depth peaks at 318.75 MB for Q19
and 562.70 MB for Q23. Complete logical Fetch bytes awaiting delivery peak at 91.03 MB and
1422.48 MB respectively; overlapping logical ranges can reuse the same physical buffer, so
these are request-byte measures, not allocator or resident-memory high-water marks.

The highest per-source physical depth in these two deferred/unlimited traces is 124 for Q19
and 17 for Q23. The object-store reader's per-source request limit is 192. These captures do
not exhaust that limit, so they do not exercise the occupied-slot condition behind the
controlled three-to-two coalescing regression. This explains why that regression is a real
behavioral improvement while these benchmarks do not establish a query-time win.

DFS ancestry is reconstructed from synchronous child emission and spawning within each
single-threaded run. Root depth is zero. Q19 reaches depth four: AnnouncePlanner at zero,
FilterPlanner at one or two, ProjectionPlanner at two or three, and ProjectionMorsel at
three or four. Q23 reaches depth three, with one level for each of those stages. This is
driver work ancestry, not decoder or layout recursion depth.

| Stage | Q19 first compute ms | Q19 root / owner startup p95 ms | Q23 first compute ms | Q23 root / owner startup p95 ms |
|---|---:|---:|---:|---:|
| AnnouncePlanner | 6.586 | 0.337 / 0.258 | 25.626 | 0.395 / 0.297 |
| FilterPlanner | 6.747 | 0.474 / 0.190 | 25.634 | 0.651 / 0.270 |
| ProjectionPlanner | 9.207 | 76.856 / 0.189 | 180.029 | 510.469 / 0.260 |
| ProjectionMorsel | 9.234 | 77.244 / 0.185 | 193.689 | 523.122 / 0.267 |

Root startup measures admission to the first execution of that stage; owner startup measures
spawn to first execution. Quantiles include only roots/owners that reach the milestone,
with missing milestones reported separately. Projection starts quickly once its owner exists,
but is released much later than the filter stage. Q23's previously reported ProjectionPlanner
ready-to-compute p95 of 14.12 ms covers subsequent visits too, including parent continuations
waiting behind child subtrees; it is not its spawn-to-first-execution delay.

Q19 has no consecutive same-owner visits with other compute work ready. Q23 has 141 such
visits at its maximum; its longest consecutive span is 25.36 ms. Across Q23 policies these
maxima are 141–143 visits and 21.99–25.36 ms. Other ready work includes an ancestor's
continuation, which DFS intentionally visits after its child. These counters identify long
runs to inspect; they do not prove sibling starvation or justify changing ordering alone.

Q19 has runnable morsels with no observed scan compute for 14.70 ms, runnable planners for
22.35 ms, and physically complete Fetch requests for 21.13 ms. Q23 has those conditions for
1209.30, 1180.09, and 1275.28 ms respectively. These overlapping gaps are not additive and
are not CPU-idle measurements: downstream DataFusion work, caller demand, async scheduling,
and logging can execute during them. Q23 has pending Fetch work without physical reads or
scan compute for only 0.65 ms in this window.

ProjectionMorsel emits 504 batches in Q19, with median 2798 rows and p95 2888. In Q23 it
emits 923 batches with median two rows and p95 eight; all four Q23 traces have a two-row
median. Q23's gap between emitted batches has p95 8.20 ms and maximum 160.77 ms. Its
`SELECT * ... ORDER BY EventTime LIMIT 10` combines sparse filtering, wide projection,
and Top-K. The small batches motivate measuring per-batch projection, conversion, and
downstream overhead before considering grouping selected rows or delaying wide projection.

Q19's 507.16 MB of started physical requests contains only 0.075 MB of coalescing gaps;
every registered physical member is fetched and delivered. Q23 starts 3852.74 MB, including
113.07 MB of coalescing gaps. It has 997.73 MB with no Fetch in the captured window, of which
884.66 MB belongs to registered members rather than gaps. In total 1106.21 MB is not delivered
to a Fetch by the boundary, and 33.65 MB belongs to unfinished physical requests. These
inventories overlap and must not be summed. Speculation, cancellation, and future reuse are
possible explanations; missing Fetch bytes are not automatically wasted. They motivate
evaluating demand priority and a byte budget for speculation before simply reducing read count.

The next optimization experiments should measure both throughput and startup/tail effects:

| Measure to add or refine | What it can distinguish | Optimization to evaluate |
|---|---|---|
| Low-overhead counters and sampled causal spans | Capture overhead versus production waits | Replace per-event text logging during measurements; retain detailed logs for short diagnostics |
| Wake-to-poll and result-channel send/receive timing | Syscall completion versus data available to the driver | Earlier completion draining or less work per async poll |
| Queue age by stage, depth, and file; downstream demand | Required DFS order versus avoidable scheduling delay | A visit/time budget that preserves demanded-row ordering |
| Predicate survivor density, rows decoded/projected, and per-batch allocation/conversion | Necessary sparse work versus repeated fixed cost | Group selective projection work or postpone wide projection until needed |
| Fetch/Announce priority, cancellation time, and bytes still running after cancellation | Useful lookahead versus speculative amplification | Limit speculative bytes across files and prioritize demanded ranges |
| Buffer lifetime, allocated bytes, RSS, and cache reuse | Request-byte backlog versus retained memory or duplicate decode | Reuse buffers and bound completed-data retention |
| Per-file readiness and completion tail, active worker count | File/partition skew versus broad resource pressure | Rebalance morsels or prioritize a required straggler |
| Device read latency/depth and host CPU, IO, memory pressure | Application pipeline waits versus storage/host contention | Tune concurrency separately for warm cache, local storage, and remote IO |

These are proposed experiments, not measured production wins. The Q23 diagnostic contains
1,569,021 events and runs about eleven times longer than its untraced median; Q19 contains
67,679 events and runs about six times longer. A smaller `--queue-trace` artifact does not
reduce that capture overhead. Lower-overhead capture is needed before assigning production
costs to these scheduling hypotheses. No additional Rust behavior change was made from this
offline analysis. Compact queue timelines retain extrema and endpoints within one-millisecond
intervals, with exact stage-start markers; full-resolution JSON statistics remain unchanged.
That analysis passed 20 Python analyzer/harness tests, Ruff, ty, and whitespace checks. Rust
checks were not repeated.

The same eight logs were reprocessed again to measure the completion that releases each parked
owner, driver state during its delivery wait, peer versus ancestor scheduling competition, output
progress, and inferred registration lifetimes. These reports are in
`target/scan-io-results/driver-causality/`; their summary records source paths, analyzer and binary
hashes, and invariant checks. They preserve the original compute/read counts, bytes, query windows,
occupancy, overlap, and result cardinalities. This is further analysis of the perturbed diagnostic
runs, not another throughput comparison.

For deferred/unlimited, the exact mutually exclusive state partition is:

| Observed query state | Q19 ms | Q23 ms |
|---|---:|---:|
| At least one scan compute running | 69.945 | 1002.305 |
| Runnable scan work, no scan compute | 33.787 | 1301.262 |
| Physically complete Fetch awaiting delivery, no runnable work or compute | 0.157 | 3.989 |
| Physical IO, no ready delivery, runnable work, or compute | 42.638 | 163.743 |
| Pending Fetch without physical IO or ready work | 0.011 | 0.653 |
| Driver advancing without the preceding states | 0.050 | 0.005 |
| No observed scan demand or advancement | 17.664 | 25.544 |

Concurrent compute, IO, and runnable states overlap in the full bitset histogram. The table gives
compute precedence, then runnable work, delivery, IO, pending Fetch, and advancement, and accounts
for the entire query window before rounding. It cannot measure CPU utilization or establish that
all runnable work was being requested downstream. In particular, Q23's 1301 ms with runnable work
and no observed scan compute is a scheduling/demand observation, not proven wasted CPU time.

The driver logs a completion, delivers it, and immediately unparks an owner whose state leaves
`Waiting`. This identifies the gating request rather than taking an arbitrary slow Fetch:

| Gating completion measure | Q19 | Q23 |
|---|---:|---:|
| Matched unpark episodes | 4200 | 9116 |
| Gating Fetch latency p95 ms | 77.219 | 442.316 |
| File read completion to gating Fetch delivery p95 ms | 33.727 | 273.022 |
| Local pread finish to gating Fetch delivery p95 ms | 35.531 | 329.825 |
| Completion delivery to unpark p95 ms | 0.179 | 0.211 |
| Unpark to first subsequent compute p95 ms | 0.239 | 5.408 |
| File-ready interval inside the same run's advancement p95 ms | 0.397 | 12.956 |
| File-ready interval outside the same run's advancement p95 ms | 33.692 | 273.022 |
| Fraction of summed gate delivery wait outside the same run's advancement | 98.647% | 98.356% |
| Bound local async resume to file-driver completion p95 ms | 3.278 | 96.998 |
| Runnable releases per physical read p95 / maximum | 65 / 834 | 78 / 163 |
| Undelivered Fetches at the observed boundary | 0 | 2335 |
| Undelivered Fetch age p95 ms, censored | absent | 737.710 |

Every observed unpark required one delivery in these captures and made work `NeedsCompute`.
Every first subsequent compute returned `continue`; that is a protocol-graph step and does not
mean the visit was unnecessary. Quantiles describe separate populations and are not additive.
The fraction uses sums over overlapping gate episodes, not query wall time. Outside `advance()`
includes completion polling/wakeup, caller work, downstream demand, and logging. The capture does
not distinguish their causes. Both this interval and the local async-to-file-driver handoff should
be instrumented before changing the driver's 64-visit completion polling budget. Q23 has no
observed local phase ordering conflicts. Its censored 738 ms Fetch tail must stay separate from
the completed Fetch p95 of 338 ms.

Q23 has 19,313 reconstructed pairwise DFS priority comparisons and zero selections bypassing
an earlier runnable priority. ProjectionPlanner's 411 competing visits are ancestor continuation
waits, with 13.848 ms of summed compute exposure. ProjectionMorsel has 18,902 peer-branch competing
visits, with 120.106 ms of summed exposure and a maximum observed peer-ready age of 37.661 ms.
Exposure sums can count multiple waiting owners for the same compute. Q19 has no visit with
another owner in `NeedsCompute` within the same run. A fairness experiment should target peer
branches and preserve continuation ordering; a long parent wait by itself is insufficient evidence.

Progress also shows what first-batch latency hides:

| Time from query begin | Q19 ms | Q23 ms |
|---|---:|---:|
| First emitted batch | 9.976 | 258.616 |
| 50% of observed completed physical bytes | 113.850 | 1494.046 |
| 90% of observed completed physical bytes | 158.618 | 2122.872 |
| 50% of observed emitted batches | 157.730 | 2064.429 |
| 90% of observed emitted batches | 162.673 | 2409.217 |
| 50% of observed emitted rows | 158.356 | 2162.807 |

The denominators are completed/emitted work in the diagnostic window, including any cancellation
effects. Q23 emits 923 batches and 3069 scan rows before the final query returns ten rows. Its
17.5 GB of delivered logical Fetch bytes reuse about 3.85 GB of started physical requests.
Neither logical bytes nor the observed progress fraction should be interpreted as total required
future work.

Segment ownership needs finer pruning information. The file service already shares live
registrations through `Arc<Read>` with weak entries in the file-wide range map. Each split retains
one strong reference per range until `clear()`, and the last read reference sends `Dropped` if
the shared result has not completed. `FileSplitIo::release(owner)` is currently empty. Whole-root
pruning eventually retires the last owner and clears its session; partial pruning has no operation
to withdraw ranges whose remaining row demand disappeared.

| Registration interest measure | Q19 | Q23 |
|---|---:|---:|
| Distinct registrations observed | 2101 | 37139 |
| Inferred split-scope interests | 5261 | 769142 |
| Registrations shared by overlapping scopes | 328 | 34916 |
| Maximum overlapping split references to one registration | 473 | 202 |
| Never-fetched optional interests in completed scopes | 456 | 375219 |
| Registrations never fetched in any scope, with all observed scopes completed | 0 | 5638 |
| Those registrations included in started physical reads | 0 | 4583 |
| Their selected member bytes excluding overlap with other members, MB | 0 | 650.902 |
| Interests whose scope admission/end cannot be joined | 0 | 242688 |

The completed-scope never-fetched interest ranges sum to 52.697 GB for Q23 because many scopes
reference the same registration. That is not resident memory or physical IO waste. Q19's 456
unused scope interests all refer to registrations fetched by other scopes, so withdrawing one
scope must preserve their other consumers. Unknown and boundary-limited scopes remain excluded
from the all-completed candidate inventory. A completed scope with no Fetch does not prove the
precise pruning cause or when the range became unnecessary.

The 650.902 MB is 16.895% of Q23's started request bytes and 650.901 MB completed in the window.
Another 1055 all-completed never-fetched registrations are absent from started physical reads.
Of the 4583 selected candidates, 4522 started before the last interested scope retired, 3560
completed before it retired, and 61 started after its observed retirement. Ten members belong
to unfinished physical reads. These are counts of registrations/members, not counts of physical
reads. Retirement approximates the subsequent `clear()`; the asynchronous read driver can observe
its withdrawal event later. This inventory supports measuring revocation, but cannot tell how
many bytes earlier pruning-aware withdrawal would save. In particular, whole-root `clear()`
already withdraws its last split interest, and most selected candidate reads started before
that point.

All four Q23 diagnostic inventories contain this candidate coverage:

| Configuration | Never-fetched registrations, all scopes completed | Included in started reads | Exclusive member MB |
|---|---:|---:|---:|
| Eager / unlimited | 6179 | 4583 | 668.800 |
| Deferred / unlimited | 5638 | 4583 | 650.902 |
| Eager / 32 | 6024 | 4833 | 698.131 |
| Deferred / 32 | 6935 | 4887 | 731.872 |

These are separate traced cancellation schedules. They establish an inventory to investigate,
not a measured saving from revocation or a policy throughput comparison. Q19 has no such
all-completed never-fetched registration in any configuration. All eight expanded reports pass
the preserved-inventory, state-partition, release-join, DFS ancestry/priority, and byte-bound checks.

A safe pruning implementation should own optional interests by live row scope and remaining
plan demand, then update them when pruning/filtering narrows that demand. Child handoff must
transfer the interest: simply releasing AnnouncePlanner's hints on retirement removes lookahead
that its descendants still need. Shared registrations remain live while any other scope holds
an optional interest or a required Fetch is outstanding. Outstanding Fetch futures must pin
`Arc<Read>` before a split map entry can be withdrawn; today they only clone `SharedRead`, relying
on that map entry to prevent `Dropped`. Revocation before coalescing selection can prevent
speculative membership; after physical read submission it does not promise to abort a blocking
pread. A demanded neighbor can still cover the same bytes as a coalescing gap.

The next capture needs acquire/revoke/handoff events with cause, selected-row and remaining-plan
segment sets, optional reference counts, required Fetch pins, last-interest time, and coalescing
selection/read-start/read-finish timestamps. Those distinguish bytes prevented before dispatch,
already running bytes, retained completed buffers, and ranges still needed by other scopes.
This offline analysis changes no Rust ownership or scheduling behavior. The expanded analysis
suite has 27 passing tests; Ruff, ty, formatting, and whitespace checks cover the changed Python
files. Rust checks were not repeated.

The initial noisy measurements below were taken on 2026-10-06 from PR #10339 head
`afeb2d1a4879ffae48802fe03939ce3ec7a1e0fa`, with the instrumentation and scheduling changes
in this working tree. Benchmark data was copied from `/home/ec2-user/votex-2/vortex-bench/data`
into this checkout. The host has 32 available CPUs and about 61 GiB RAM. Other benchmark and
compilation processes were active. One-minute load varied roughly from 42 to 126 during the warm
comparisons, and cold-run host IO wait varied from 3% to 81%. These are shared-machine results,
not stable estimates of a production speedup.

The default change delays coalescing physical ranges until a file reader has free slots. This
keeps later registrations and cancellations visible to the coalescer. A controlled regression
test occupies the only slot, then registers a neighboring range: eager selection makes three
physical reads; deferred selection makes two, delivers all three requested buffers, and coalesces the late
neighbor. Existing refill, straggler, progress, and panic-propagation tests pass.

Local object-store file payloads also have optional process-wide admission before buffer
allocation and blocking `pread`. The permit remains with submitted blocking work if its async
caller is cancelled. Positive `VORTEX_LOCAL_READ_CONCURRENCY` values enable it; the production
default remains unlimited because the timing evidence does not justify a universal limit.
Network stream payloads do not use local permits.

The initial comparison used one binary and explicit runtime controls. Its SHA-256 is
`bc75409cb26d4b4afdea8269836b996a7d25bd361cbf7b8f8511b4362a29c393`.
The final build changes only the admission default from 32 to opt-in; every measured configuration
explicitly sets either 0 or 32, so that default is not used by these comparisons. The baseline
shares tracing and allocation-after-GET code with the candidates; it isolates the scheduling
policies rather than reproducing the PR binary byte for byte.

| Configuration | Physical range selection | Aggregate local read admission |
|---|---|---|
| `v2-baseline` | Eager (`VORTEX_SCAN_IO_LOOKAHEAD=1`) | Unlimited (`=0`) |
| `v2-unlimited` | Deferred | Unlimited (`=0`); final production scheduling |
| `v2-lookahead` | Eager | 32 |
| `v2` | Deferred | 32 |

Timing processes have driver and read-timing logging disabled. In the initial comparison, each
configuration ran five query
iterations in four processes, discarding the first two iterations of each process. Configuration
order rotates, giving each configuration every position once. Each cell below uses 12 retained
samples. DataFusion scans Vortex files directly; TPC-H uses SF10 and ClickBench uses the
partitioned dataset. ClickBench query numbers are zero-based.

| Configuration | Q19 median ms | Q19 min–max ms | Q23 median ms | Q23 min–max ms |
|---|---:|---:|---:|---:|
| Eager / unlimited | 69.193 | 40.132–111.259 | 423.503 | 348.801–4026.078 |
| Deferred / unlimited | 61.665 | 27.637–175.978 | 360.253 | 235.321–498.651 |
| Eager / 32 | 77.714 | 35.909–164.795 | 308.334 | 247.194–392.311 |
| Deferred / 32 | 77.324 | 32.367–142.619 | 380.436 | 216.511–565.500 |

Round medians make the variation clearer:

| Configuration | Q19 round medians ms | Q23 round medians ms |
|---|---|---|
| Eager / unlimited | 70.3, 68.1, 89.1, 59.1 | 3234.1, 406.8, 352.6, 434.5 |
| Deferred / unlimited | 63.4, 31.7, 54.5, 133.0 | 362.7, 241.9, 439.3, 357.8 |
| Eager / 32 | 62.4, 114.4, 52.6, 128.3 | 287.5, 326.4, 331.1, 299.5 |
| Deferred / 32 | 95.8, 43.2, 102.9, 60.8 | 231.6, 431.6, 382.9, 432.3 |

Deferred/unlimited has lower pooled medians on both queries, but it wins only three of four Q19
rounds and two of four Q23 rounds. The 32-read cap alone beats baseline in all four Q23 rounds,
but Q19 is mixed. Combining the changes does not combine their apparent gains. The deterministic
coalescing regression supports deferred selection; the noisy query results provide a useful
comparison, not a general speedup claim.

An additional Q19 comparison requests Linux page-cache eviction for this checkout's Vortex files
before every process using `posix_fadvise(DONTNEED)`. It uses one iteration per process and four
balanced rounds, with no global cache flush. Eviction is advisory; storage caches and host work
still affect the run.

| Configuration | Median ms | Min–max ms | Diagnostic application bytes | Repeated range bytes |
|---|---:|---:|---:|---:|
| Eager / unlimited | 4467.251 | 4405.788–4572.006 | 511.43 MB | 3.67 MB |
| Deferred / 32 | 4424.461 | 4415.453–4487.506 | 507.76 MB | 0 MB |

These cold query durations differ by about 1%. The separate local-read diagnostics explain why:
median `pread` falls from 1474.8 ms to 676.0 ms, while median admission becomes 609.8 ms. Much of
the wait moves ahead of allocation and the blocking pool. Read concurrency control reduces
outstanding blocking work, but this experiment remains dominated by storage throughput.

`scripts/scan-io.py` joins stable run, root, owner, session, registration, and physical-read IDs
on a shared monotonic clock. It retains file and row scopes, concrete planner/morsel types,
state transitions, compute outputs and rows, Announce/Fetch/Prefetch intents, coalesced members,
request publication, dispatch, delivery, and subsequent compute. It reports:

- Query-window read and compute occupancy, overlap, peak and mean concurrent work.
- Per-stage compute, parked, ready-to-compute, and fetch distributions, including p95/p99.
- All-work IO blocking, wake delay, caller gaps, and time inside driver advancement.
- GET, admission, allocation, blocking-pool queue, `pread`, and async-resume phases.
- Unbound, unfetched, undelivered, cancelled, and unfinished work, including censored spans in
  the timeline up to cancellation or the observed boundary.

The one-iteration traces were captured separately before the warm timing processes. Their cache
state and logging overhead differ, and later configurations benefit from earlier reads. Their
elapsed times are therefore unsuitable as before/after performance comparisons. They still
distinguish the waiting mechanisms within each captured scan.

For deferred/unlimited, the Q19 trace spans 240.2 ms. Scan compute occupies 105.1 ms of the
window, and ready or active scan work occupies 172.6 ms. Local `pread` p95 is 1.25 ms, while fetch
p95 is 83.77 ms, dispatch wait p95 is 49.85 ms, and delivery delay p95 is 32.81 ms. Once data is
warm, fetch latency includes publication, driver scheduling, GET preparation, and delivery;
it cannot be explained by read bytes or the syscall alone. FilterPlanner accounts for 219.8 ms
of summed compute and ProjectionMorsel for 17.9 ms. Those sums overlap across workers.

For the same configuration, Q23 spans 16.0 seconds. Scan compute occupies 1.63 seconds of the
window, and ready or active work occupies 3.39 seconds. Local `pread` p95 is 10.43 seconds,
while blocking-pool queue p95 is 1.26 ms. The long tail is inside blocking reads in this capture,
rather than admission to the pool. FilterPlanner's fetch p95 is 12.36 seconds; ProjectionMorsel's
is 6.12 seconds. ProjectionPlanner's ready-to-compute p95 is 16.18 ms, versus 0.18 ms for the
filter. These distinguish an IO dependency from a runnable stage waiting to be visited, and do
not support treating every parked interval as wasted compute capacity.

Q23 selects all columns where URL matches `%google%`, ordered by EventTime with LIMIT 10. Top-K cancellation
matters: the eager/unlimited trace has 110 physical requests unfinished at the query boundary and
9245 physical members not fetched within that window. Deferred/unlimited has 34 and 6085 in its
separate, warmer capture. These differences are observations of different schedules/cache states,
not proof of an optimization reducing speculative reads. Q19 fetched every physical member in all
four traces. All logical Fetch requests bind to physical registrations in these direct scans.
There are 3 unbound local completions in Q19 and 181 in Q23; local reads outside the file-driver
binding inventory are counted explicitly, rather than assigned to a scan owner.

Compute measures wall time inside `compute()`, not CPU time. Read occupancy includes preparation
and admission; observed `pread` occupancy uses only completed, bound local reads. Concurrent
split waits can sum to thousands of seconds within a short query. Use the query unions and the
timeline for wall-time interpretation. Driver overhead in a traced run includes logging costs.
Bytes count application requests, not device traffic. Result row counts are not value checksums.
Experimental batch IO and segment-cache adapters emit logical driver events but do not yet have
the direct file service's physical binding coverage.

Local artifacts are in `target/scan-io-results/factorial/` and `target/scan-io-results/cold-q19/`:
`summary.json` retains every timing sample, round median, environment, binary hash, and driver
summary. Each diagnostic has `.log`, `.driver.json`, `.summary.txt`, and `.trace.json` files.
Open a trace in Perfetto to inspect individual scopes and read-to-fetch delivery flows. The
[measurement README](README.md) contains reproducible commands and control descriptions.

The final binary was also exercised with `VORTEX_SCAN_V2=1`, both new control variables unset,
five Q19 iterations, and `--show-metrics`. Its last three samples were 29.78, 33.71, and 29.43 ms;
the final scan reports 182 physical reads totaling about 507.2 MB. This is a final-default smoke
check under another host/cache state, not an additional baseline comparison. Artifacts are
`target/scan-io-results/final-default-q19.{jsonl,log}`.

Initial validation: the release-debug benchmark build passes; 8 file-driver tests, 30 scan-driver protocol
tests, 3 object-store IO tests, and 11 Python analysis/harness tests pass. Pinned nightly formatting,
Ruff, ty, scoped Clippy with all targets/features and `-D warnings`, and patch whitespace checks
pass. Clippy uses `--no-deps`: the dependency-inclusive invocation encounters an existing
`manual_map` lint in unchanged `vortex-buffer/src/trusted_len.rs:33`. Workspace-wide tests were
not run.
