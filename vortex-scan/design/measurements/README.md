# Scan V2 measurements

Scripts and results behind [../STATUS.md](../STATUS.md). Everything writes next to itself, so
run the scripts from this directory. Raw output (`raw/`, `raw-warmfirst/`, `prof/`) is ignored
by git; the tables are kept.

| File | What it does |
|---|---|
| `bench_ab.py` | Interleaved V1 (`VORTEX_SCAN_V2` unset) against V2 (`=1`) runs of `duckdb-bench` or `datafusion-bench`, hot or cold. Appends tables to `results.md`. Usage is in its docstring. `ROOT` defaults to the worktree the script is in and `BIN_DIR` to `$ROOT/target/bench-bins/9c6392e5aa`; STATUS.md says where the data and binaries actually are. |
| `evict.py` | Evicts files from the macOS page cache without root (`msync(MS_INVALIDATE)`) and reports residency with `mincore()`. Used by the cold mode. |
| `run_matrix.sh` | The whole matrix: six hot and six cold steps over TPC-H SF1, SF10 and ClickBench on both engines. |
| `run_hot_rerun.sh` | The hot steps again plus the warm-first measurement, for a quieter machine. |
| `bench_first.py` | Warm-first variant: the first iteration of a process with the data already cached. Appends to `results-warmfirst.md`. |
| `run_sweeps.sh`, `prof_sweep.py` | Samply profile of every query, V1 and V2, hot. Writes `prof/` and `sweeps.log`. |
| `prof_attr.py`, `proflib.py` | Attribute a Samply profile to scan and engine categories (IO, decode, filter kernels, pruning, scheduling, conversion, blocked). |
| `prof_cold.py` | Samply profiles of cold runs. |
| `sweep_summary.py` | Summarises `sweeps.log` per engine and suite; its output is `sweep-summary.md`. |
| `final_tables.py` | Builds consolidated per-query tables from the results files and profiles. |
| `iobench.c` | Hot page-cache read scaling: N threads reading 1 MiB chunks with `pread` or `mmap`. Build with `cc -O2 -o iobench iobench.c`. |
| `randread.py` | Random 4 KiB read latency probe, to check that eviction worked. |
| `attr_table.py`, `sweep_dist.py` | Build the attribution and scan-share distribution tables in the reports below. |
| `results.md` | Hot and cold tables for the baseline commit `9c6392e5aa`, measured 2026-09-30. Sections marked SUPERSEDED were rerun further down. |
| `results-warmfirst.md` | Warm-first tables for the same commit. Incomplete: DuckDB TPC-H only. |
| `sweeps.log`, `sweep-summary.md` | The per-query scan-share sweep and its summary. |
| `REPORT_TABLES.md` | Consolidated per-query hot and cold tables. |
| `REPORT_SCAN_SHARE.md` | Scan share of wall time per engine and suite: distribution, and the geomean if the scan cost fell by 30, 50, or 100%. |
| `REPORT_SWEEP_PER_QUERY.md` | Scan share and blocked time for every query. |
| `REPORT_ATTRIBUTION.md` | Where the time goes, V1 against V2, for twelve representative queries. |

Samply must be installed for the profiling scripts, and the bench binaries must exist under
`target/bench-bins/<commit>/` (see STATUS.md for the build command).

## Driver and IO diagnostics

[REPORT_IO_INTEGRATION.md](REPORT_IO_INTEGRATION.md) records integration with the updated
PR #10339 branch, including preservation of its pending-consumer read lifetimes and
per-consumer announcements. The timings below precede that integration.

[REPORT_IO_ONLY.md](REPORT_IO_ONLY.md) records the measured IO-only source: compressed LIKE,
projection grouping and split-task scheduling experiments have been removed and preserved
separately. Fresh cold SSD runs lower geometric-mean latency 22.95% across all 65 queries,
with 59 faster medians and six regressions no larger than 3.07%. Eight-round confirmation
improves Q23 by 19.84%, TPC-H Q19 by 54.14%, Q24 by 63.03% and ClickBench Q19 by 5.94%, each
winning 8/8 pairs. The coalescer alone improves Q23 by 18.16% at fixed handle reuse/native32.
The compute implementation and projection batch boundaries remain fixed across these IO
comparisons. Settings remain opt-in; earlier reports retain their original compute context.

[REPORT_IO_COST_MODEL.md](REPORT_IO_COST_MODEL.md) records a subsequent cost-based speculative
coalescing prototype: built and checked, then rejected after 432 cold SSD timings and 52 separate
diagnostics. Its apparent Q23 median gain did not hold as a paired geometric-mean improvement
against the previous executable. The prototype is archived; the earlier IO changes remain.

[REPORT_IO_COST_FOLLOWUP.md](REPORT_IO_COST_FOLLOWUP.md) investigates that result with identical
controls, 864 cold read-probe batches and exact V2 acceptance/Fetch captures. Half the accepted
members are later fetched, representing 38% of charged extra bytes. The report separates backend
throughput from first-needed data latency and records corrected benchmark extrema and host-wait
counters. The cost policy remains archived.

[REPORT_IO_COLD_LOOP.md](REPORT_IO_COLD_LOOP.md) contains the earlier fixed-compute follow-up: 993 accepted
cold SSD timings, including all 65 TPC-H/ClickBench queries, plus 295 separate diagnostics.
The new demanded-range coalescer improves Q23 by 20.67% at fixed handle reuse/native32, with
8/8 paired wins. The combined setting lowers geometric-mean latency 22.98% across all 65 queries
(56 faster medians; nine regressions below 10%). Eight-round confirmation against current V2
improves Q23 by 19.09%, TPC-H Q19 by 55.03%, Q24 by 71.36% and ClickBench Q19 by 5.27%, each
with 8/8 paired wins. Settings remain opt-in. The report includes the native admission sweep,
traffic/CPU tradeoffs, traced startup, native versus driver queue depth, compute and delivery
waits, shared-reference withdrawals, and the tracing effect that reverses Q23 timings.

[REPORT_IO_SSD.md](REPORT_IO_SSD.md) contains the first cold instance-SSD rerun: 232 screening
samples across 29 IO configurations and 128 confirmation samples. Across eight paired rounds,
disabling announcements with native concurrency 32 improves Q23 by 17.07%. Local handle reuse
improves TPC-H Q19 by 50.56%, Q24 by 65.77% and ClickBench Q19 by 37.04%, while regressing Q23
by 2.78%. Settings remain opt-in. The report includes timing ranges, traffic, native read phases,
driver startup, queue depth, compute overlap and release waits. Earlier reports below retain
their original storage and measurement conditions.

[REPORT_IO_PIPELINE.md](REPORT_IO_PIPELINE.md) records the larger sparse-projection candidate:
the final six-round matrix measures 3.89% lower Q23 medians versus current ungrouped and 3.50%
versus the saved boxed-Fetch executable, with physical bytes essentially unchanged. Q19 and Q24
regress by 0.97% and 2.26% against current ungrouped. Earlier pilots of 9–12% and a factorial
combination result of 7.08% remain recorded as separate measurements. That historical candidate
applies grouping and interest withdrawal automatically only to wide projections; a broad scan
win has not been established. The six-round backend IO matrix shows a
modest 3.33% Q23 advantage for local handle reuse, with Q19 effectively tied; a 4 MiB request cap
loses 3.92%. Exact withdrawal/reference metrics explain why partial pruning alone saves little
IO in the captured schedule. Earlier file pruning adds only 2.46% on Q23 (4/6 paired wins) and remains opt-in. The report
separates driver work reduction from physical-byte savings.

[REPORT_IO_OVERHEADS.md](REPORT_IO_OVERHEADS.md) follows registration, allocation, atomic,
coalescer, and Fetch-delivery overheads through source changes and paired CPU profiles. Its
42-sample comparison shows a modest 0.83% Q23 median gain, with Q19 tied. The ready-completion
queue loses 2.21% on Q23 and remains disabled. Raw outliers and rejected rounds are retained.

[REPORT_IO_DRIVER.md](REPORT_IO_DRIVER.md) records a 2.83% warm Q23 improvement from storing
Fetch futures directly, with Q19 unchanged, using 56 timing samples per configuration and query.
It also records actual registration drops, pruning-to-release and wake-to-poll waits, failed
projection-hint experiments, the earlier quiet-window rerun and initial noisy-host comparisons,
stage and wait interpretation, cold Q19 experiments, and scheduling choices.
The earlier rerun retains 40 warm samples per configuration and eight cold samples, repeats rounds
overlapping detected competing work, and captures driver traces after all timing runs. Expanded
offline analysis of those same logs is in `target/scan-io-results/driver-depth/`; the report
includes startup and queue-depth tables, DFS ancestry, byte coverage, and optimization leads.
Further reprocessing in `target/scan-io-results/driver-causality/` adds the completion that releases
each parked owner, driver advancement during delivery waits, joint query states, peer versus
ancestor DFS competition, output progress, and inferred registration interests. Its summary
records the analyzer and original binary hashes. Both analyses use the existing logs; they do
not rerun queries or change the accepted timing samples.

`scripts/scan-io.py` joins a V2 planning run's roots, row scopes, owners, requests, parked work,
and completions to the default file service's physical reads. It produces a per-run JSON report
and a Perfetto timeline with named compute stages, IO blocking, wake delay, caller gaps, and
read-to-fetch flow links. It measures the query window, physical concurrency, IO/compute overlap,
stage outputs, ready-to-compute delay, and latency p50/p95/p99. Local reads break down into GET,
admission, allocation, blocking-pool queue, pread, and async resume. It also reconstructs the
driver's DFS paths and measures stage startup, runnable and parked planner/morsel queues,
outstanding Fetch requests, completed reads awaiting Fetch delivery, request byte depth, batch
sizes and gaps, and repeated visits while other work is runnable. Optional segment-cache adapters
and the experimental batch IO service still report logical
driver events; fetches without physical bindings are counted explicitly. Partial and cancelled
runs, unfinished reads, and events outside the diagnostic iteration are not treated as completed
work. Compute is wall time inside `compute()`, and concurrent split times overlap.

Build the benchmark from the current sources, then run from the repository root:

```bash
cargo build -p datafusion-bench --profile release_debug
python3.11 scripts/bench-io.py \
  --data-root vortex-bench/data --output target/scan-io-results/comparison \
  --workloads tpch-q19 clickbench-q23 --variants v2-baseline v2 --modes direct \
  --iterations 7 --rounds 4 --diagnostic-iterations 1 --driver-trace \
  --note "Shared machine with competing CPU and IO work"
```

Driver tracing is enabled only in the diagnostic processes. Timing processes discard their first
two iterations and rotate configuration order between rounds. The summary retains every sample,
round medians, min/max, before/after load averages, host CPU busy/steal fractions, and the binary
hash. A noisy machine can change coalescing and cancellation as well as wall time; compare byte
counts and scheduling evidence alongside the timing spread.

For the projection-hint experiment, use `--variants v2-project-early v2-project-late`. Both use
deferred physical selection and unlimited local reads. `VORTEX_SCAN_PROJECT_ANNOUNCE=0` leaves
filter announcements in place and defers projection-only interest to the existing selected-row
projection prefetch. Unfiltered scans still announce their projection. This isolates early
projection speculation without changing predicate or projection evaluation.

`--variants v2-project-early v2-project-late v2-inline-fetch` also tests the Fetch future's
storage. The first two explicitly select boxed futures with `VORTEX_SCAN_IO_INLINE_FETCH=0`;
`v2-inline-fetch` preserves early projection hints and uses `=1` to store the concrete future
inside `FuturesUnordered`'s task allocation, the new default. A scope's `inline_fetches` counts
wrapper allocations avoided, including Fetches cancelled before delivery. It does not count all
allocator operations.

For changes requiring separate builds, pass `--baseline-binary <saved-executable>` and use
`--variants v2-previous v2-unlimited v2-ready-fetch`. The first two use identical runtime
settings with ready-completion delivery disabled; `v2-previous` runs the saved executable.
The third enables `VORTEX_SCAN_IO_READY_FETCH=1` in the candidate. This experimental path
bypasses a Fetch task when its shared result is already available. Lifetime counters report
`ready_fetches` and `future_queues`, and summaries retain both executable hashes.

```bash
python3.11 scripts/bench-io.py --data-root vortex-bench/data \
  --output target/scan-io-results/projection-hints \
  --workloads tpch-q19 clickbench-q23 --variants v2-project-early v2-project-late \
  --modes direct --iterations 7 --rounds 4 --diagnostic-iterations 1 \
  --timing-metrics --wait-for-quiet --repeat-contended-rounds \
  --driver-trace --driver-trace-mode json --lifetime-trace
```

`--timing-metrics` prints scan counters after each query timer stops, with driver/read event
logging disabled. The report retains counters and cardinality for every iteration, including
warmups. Counters are aggregated after the query returns; cancellation and background completion
can affect the snapshot. This is a separate measurement mode from earlier timing runs without
counter output. Direct and adapter diagnostics now follow all timing runs; cached-mode diagnostics
still run first to verify preload completeness. `--driver-trace-mode` chooses a full timeline,
compact queues, or JSON only; its default is compact queues.

`--wait-for-quiet` requires six consecutive samples spanning `--quiet-seconds` (default 60), with CPU usage below
10%, IO wait below 2%, full IO pressure below 1%, and no detected competing benchmarks/builds.
Warm runs also require load below 4. Cold runs instead require the dataset's block device to
have zero current IO in flight when its counter is available. Historical load remains recorded;
blocked threads from a finished cold query can keep it high while the CPU and device are idle.
Every gate observation is saved. Competing process names are sampled during timing rounds.
`--repeat-contended-rounds` preserves rejected rounds in `attempts` and repeats the whole round,
keeping order comparisons intact. Detection is a shared-host contention check, not exclusive
ownership. The counters include the benchmark's own activity.

Lifetime tracing uses `RUST_LOG=vortex_file::scan_lifetime=debug`, separately from driver tracing.
It records each actual scope clear/drop with request/reuse counts, peak interests, never-fetched
interest bytes, and completion polling counts. Last `Read` releases record the registration ID
and whether the shared result was observed. The analyzer joins these to read selection/start/end
to distinguish queued withdrawal, in-flight work, and completed registration retention. Actual
scope clears resolve many lifecycle endpoints that root admission alone cannot identify; repeated
use of a session is split into separate interest episodes. Byte-time is logical member retention,
not allocated memory or complete buffer lifetime.

Forwarded IO wake notifications and session poll begin/end events measure wake-to-next-poll,
poll duration, outcomes, and notifications without a later poll. Multiple wakes in a burst and
wakes during polling remain explicit. A trace waker forwards the original notification; this
extra tracing affects diagnostic scheduling. Driver selection events report each completed
predicate/pruning decision's input and surviving row counts, including unchanged and all-false
decisions. Pruning and ordinary filtering use separate stage names.

For all-false decisions with nonempty input, `selection_release` reports the delay to the next
exact root IO clear. Decisions covering the entire admitted file/row scope are separated from
partial branches, whose siblings may still need IO. Missing clears remain censored. This exposes
the retain-until-root-clear policy without claiming a rejected branch exclusively owns a segment.

Open a generated `.trace.json` in <https://ui.perfetto.dev>. To inspect an existing diagnostic log:

```bash
python3.11 scripts/scan-io.py target/scan-io-results/comparison/tpch-q19-v2-direct-diagnostic.log \
  --output /tmp/scan-driver.json --trace /tmp/scan-driver.trace.json
```

For a smaller timeline containing queue counters and first execution of each stage, replace
`--trace` with `--queue-trace /tmp/scan-queues.trace.json`. This changes offline output size,
not diagnostic logging overhead. Compact counter timelines keep each one-millisecond interval's
extrema and endpoints at their original timestamps. Shorter fluctuations between those points
are omitted; JSON measurements and first-stage timestamps remain exact. Use
`--queue-resolution-us 100` for finer detail or `0` for every counter change. Omit both trace
flags for JSON-only analysis. The parser streams the log rather than reading the whole file
into a string.

The default timeline omits compute steps shorter than 10 microseconds while including them in
compute totals; pass `--min-compute-us 0` to retain every step. `--iteration` selects a diagnostic
iteration by zero-based index, defaulting to the last complete one.

Queue depth quantiles are weighted by elapsed time, including time at zero. Runnable queues
include only `NeedsCompute` work and exclude time inside `compute()`; parked queues include
unfinished waits up to cancellation or the observed boundary. Completed-read delivery depth
counts logical Fetch requests whose physical read has finished but whose completion the driver
has not delivered. One physical read can answer many Fetch requests. Request byte depth measures
outstanding ranges, not allocated or resident memory; logical ranges can overlap or reuse data.
Bound local phase counters cover completed reads only, so they are not complete at cancellation.
The JSON report also groups physical request count and byte depth by IO source, with its URI,
read inventory, and unfinished count. Per-source quantiles use the whole query window too.

Startup is reported from query begin, owner spawn, and root admission. Missing milestones stay
absent. Driver DFS depth starts at zero for a root and follows emitted planner/morsel children;
it is not layout or decoder recursion depth. Reconstruction depends on the synchronous child
emission/spawn order within each run; missing ancestry is reported as unknown. Per-depth queue
counters and per-stage first-compute markers make depth and startup visible in Perfetto.
Other runnable work includes ancestor continuations, which DFS intentionally visits after a
child's subtree. Consecutive-visit counters alone do not establish starvation or incorrect order.

Byte coverage merges overlapping Fetch/member ranges within each physical read. It separates
coalescing gaps from bytes with no Fetch request or delivery in the captured window. The latter
can reflect speculation, cancellation, or later reuse; they are not automatically wasted bytes.
Reads absent from the physical binding stream remain explicitly unbound.

Unpark episodes identify the last completion delivered while the owner was parked: the driver
logs a completion, delivers it, and immediately unparks an owner whose state changed. These
episodes measure the gating Fetch latency, pread/file completion to delivery, delivery to unpark,
and unpark to the first subsequent compute. Cancellation before compute remains censored.
The first compute's output is also reported; `continue` can be necessary graph bookkeeping.
Physical completions can release many owners, so completion fanout and episode counts accompany
latency quantiles. A read that releases no parked owner can still answer a later Fetch.

For each gating Fetch, the analyzer intersects its file-ready-to-delivery interval with the
same run's `advance()` and compute intervals. This distinguishes work inside advancement from
time outside it. Outside advancement can include waiting for wakeup/completion polling, caller
work, downstream demand, or logging. It does not prove backpressure. Exposure sums overlap across
owners. Bound local reads also report async-resume-to-file-completion delay; local phase timestamps
are reconstructed from logged durations on the shared clock.

`query.state_time` partitions the query exactly by the simultaneous presence of compute,
runnable work, deliverable Fetches, physical IO, pending Fetches, driver advancement, and caller
gaps. Its mutually exclusive categories give compute precedence, then runnable work, delivery,
IO, pending Fetch, advancement, and no observed demand. This describes observed state, not CPU
idleness or a causal bottleneck. The full bitset histogram remains available for other groupings.
`query.progress` measures time to 10/50/90/99/100% of completed physical bytes, delivered logical
Fetch bytes, emitted batches, and emitted rows. Denominators are the work observed in this window;
logical bytes can reuse physical data and emitted rows differ from final query results.

DFS competition is measured for each ready owner while another owner computes, separately for
ancestor continuations, descendants, peer branches, and unknown ancestry. A parent waiting for
its own child's subtree follows the driver's contract. Peer exposure measures a different source
of queue delay. Reconstructed priorities are compared with the chosen owner's priority on each
visit; zero earlier-priority bypasses demonstrates that observed choices follow DFS, not fairness.

Registration lifetimes infer one interest per `(source, registration, session)` from the first
binding until root completion/cancellation/boundary under the current file service's policy.
Multiple owners in one session share its interest; separate sessions can keep the same read
alive. These counts are not `Arc::strong_count()`. Byte depth sums logical interests and may
count a range many times. It is neither RSS nor physical traffic. Never-fetched optional interests
are separated by completed, cancelled, censored, and unknown scope lifetimes. Unknown admission
remains unknown. For registrations with no Fetch in any observed scope and only completed scopes,
the report measures their member coverage in started physical reads, subtracting overlap with
other members. This is a candidate inventory, not proven avoidable IO: the inventory alone cannot
identify which selection made a segment unnecessary, and a coalesced read could still cover
those bytes after hint withdrawal.

Local file payloads can share a process-wide admission limit before allocation and blocking reads.
`VORTEX_LOCAL_READ_CONCURRENCY=0` selects unlimited admission (`v2-unlimited`), the production
default; positive values choose a limit. The benchmark's `v2` and `v2-lookahead` configurations
explicitly use `--local-read-concurrency`, defaulting to 32. The permit stays with
submitted blocking work when its async caller is cancelled. Remote stream payloads do not take
local permits. The file driver also delays choosing physical ranges until read slots are available,
so later announcements and cancellations remain visible to the coalescer.

`VORTEX_SCAN_IO_READ_CONCURRENCY=N` separately caps physical reads per file before choosing
ranges. It clamps the reader's advertised concurrency, never raises it, and ignores zero or
invalid values. The default has no override. `v2-io-read-slots-1` and `v2-io-read-slots-4` select
experimental limits while retaining unlimited native admission and the current scan policy.
This cap differs from `VORTEX_SCAN_SPLIT_CONCURRENCY`, which limits live split graphs.

For a factorial comparison, use `--variants v2-baseline v2-unlimited v2-lookahead v2`:
baseline has eager range selection and unlimited local reads, unlimited delays range selection,
lookahead limits local reads, and v2 applies both. `VORTEX_SCAN_IO_LOOKAHEAD=1` restores the eager
physical queue. The baseline shares the diagnostic instrumentation and allocation-after-GET
behavior with the candidates; it isolates scheduling changes rather than reproducing an older
binary byte for byte. Repeat with `--local-read-concurrency N` to evaluate admission limits.

Ready-to-compute delay includes driver queueing and downstream demand. A parked owner can overlap
many other owners, so stage parked totals are not query duration. The compute and read unions and
their overlap describe wall occupancy, not CPU utilization or device service time. Query rows are
result cardinality, not a checksum; the benchmark's normal result checks still apply. Cancelled
reads may finish after the query marker and remain explicitly unfinished in that window.

For cold Linux data-page comparisons, use `--evict-local-files --iterations 1
--discard-iterations 0 --diagnostic-iterations 1 --modes direct`. Every sample is one execution
in a fresh process. Before each process, the harness inspects file-page residency using `mincore`,
issues per-file `posix_fadvise(DONTNEED)` for the dataset's Vortex files, and inspects residency
again without faulting in data. It refuses to run if more than 0.1% of pages remain resident.
Before/after counts are retained in each command record and timing sample. Dataset metadata and
storage-device caches are not flushed.

`process_storage_read_bytes` records Linux child storage-read accounting for the whole process,
including setup, readahead and work completing after the query timer. It is separate from
`vortex.io.read.total_size`, which counts completed application ranges at the query snapshot.
This Linux field derives from `ru_inblock * 512`; the kernel defines those blocks in 512-byte
units. See [Linux task IO accounting](https://github.com/torvalds/linux/blob/master/include/linux/task_io_accounting_ops.h)
and [getrusage aggregation](https://github.com/torvalds/linux/blob/master/kernel/sys.c).
Cold samples also retain the dataset's block-device counter snapshots and their deltas:
read/write bytes and operations, read MB/s, mean read latency and time-weighted queue depth.
The device is resolved from the dataset filesystem; partition snapshots use the parent device
so partition and disk activity are not double-counted. These are whole-device measurements
including other host activity. They cover process initialization and shutdown as well as the
query, and they do not measure the driver or blocking-pool queues. Counter resets and unavailable
devices produce unavailable results rather than zero. Definitions follow
[Linux block IO statistics](https://www.kernel.org/doc/html/latest/admin-guide/iostats.html).
A cold query's wall time, process storage bytes, application counters and separate read-phase
trace must be assessed together. Quiet gating and rejected-round retention also apply to cold
runs. Warm measurements above cannot establish cold-IO gains.

[REPORT_IO_PIPELINE.md](REPORT_IO_PIPELINE.md) records cold matrices and driver
interpretation. On the earlier EBS dataset, projection deferral plus `VORTEX_SCAN_IO_READ_CONCURRENCY=1` improves Q23's
median by 12.60% across six pairs but regresses Q24 by 27.61%, so both settings remain opt-in.
The harness variant is `v2-io-project-late-read-slots-1`; compare it with `v2-unlimited` using
the cold flags above. The report retains first-scan-batch, queue-depth, physical byte-depth,
storage-byte and compute-wait evidence alongside timings and validation limits.

All subsequent local tests use the instance SSD mounted at `/mnt/vortex-ssd`, device `nvme0n1`.
The complete benchmark dataset is copied to `/mnt/vortex-ssd/votex-4/data`; executables,
temporary files and results use that same SSD. Set `TMPDIR=/mnt/vortex-ssd/votex-4/tmp`
for tests. For benchmark runs, also pass `--require-storage-device nvme0n1`: the harness
refuses to run if data, executable, output or temporary storage resolves to another device.
It records all four paths, device names and models in `summary.json`. This distinguishes the
instance SSD from EBS, which is also exposed through an NVMe device.

```bash
TMPDIR=/mnt/vortex-ssd/votex-4/tmp python3.11 scripts/bench-io.py \
  --binary /mnt/vortex-ssd/votex-4/bin/datafusion-io \
  --data-root /mnt/vortex-ssd/votex-4/data \
  --output /mnt/vortex-ssd/votex-4/results/confirm-new \
  --require-storage-device nvme0n1 \
  --workloads clickbench-q23 tpch-q19 clickbench-q24 clickbench-q19 \
  --variants v2-unlimited v2-io-file-handle late v2-io-project-late-read-slots-1 \
  --modes direct --scale-factor 10.0 \
  --iterations 1 --discard-iterations 0 --rounds 8 --diagnostic-iterations 1 \
  --timing-metrics --evict-local-files --wait-for-quiet --quiet-seconds 10 \
  --repeat-contended-rounds
```
