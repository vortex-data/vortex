<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Scan pipeline work reduction

## Local SSD measurements

The current [IO-only measurements](REPORT_IO_ONLY.md) remove the compute experiments recorded
in this historical report. With the original compute path and projection batch boundaries,
the combined IO setting lowers full-suite geometric-mean latency 22.95% (59/65 faster), and
the coalescer alone improves Q23 by 18.16% at fixed handle reuse/native32. Archived source and
measurements below retain their original scope.

The completed [cold IO follow-up](REPORT_IO_COLD_LOOP.md) adds a demanded-range coalescer and
993 accepted cold timings, including all 65 TPC-H/ClickBench queries. At fixed handle reuse
and native32, the coalescer improves Q23 by 20.67% (8/8 paired wins). The combined setting
lowers geometric-mean latency 22.98% across the full suite; settings remain opt-in. Its report
separates this code change from handle reuse, includes admission sweeps and compute/IO waits,
and records smaller regressions and the effect of detailed tracing.

The completed [cold SSD rerun](REPORT_IO_SSD.md) supersedes the timing comparison below:
360 accepted cold samples cover 29 IO configurations and an eight-round confirmation.
Q23 improves 17.07% with announcements disabled and native concurrency 32. Local handle reuse
improves TPC-H Q19 by 50.56%, Q24 by 65.77% and ClickBench Q19 by 37.04%; it regresses Q23.
These are different settings and remain opt-in. Four cold driver captures and all 267 targeted
tests complete on SSD. The EBS comparisons below remain historical evidence.

Subsequent tests use the 1.7 TiB local instance SSD, `nvme0n1`, identified as Amazon EC2
NVMe Instance Storage. It is mounted as XFS with `noatime` at `/mnt/vortex-ssd`; readahead
remains 128 KiB. All 350 benchmark files were copied: 43,574,143,340 bytes, with matching
sizes and modification times and no changes in a subsequent rsync dry run. Every copied
entry was checked to reside on the SSD. `vortex-bench/data` now points to that copy; the
original EBS data is preserved in `target/scan-io-results/ebs-data-backup`.

The dataset, benchmark and Rust test executables, temporary files and results use
`/mnt/vortex-ssd/votex-4/{data,bin,tmp,results}`. The benchmark's new
`--require-storage-device nvme0n1` guard checks the data root, executable, output and
`TMPDIR` before running and records their device names and models in `summary.json`.
An actual EBS dataset was rejected by the guard. The storage-backed eviction test now
honors `TMPDIR` instead of forcing the repository filesystem.

The SSD comparison reuses the frozen executable, SHA-256
`3af02184c6cd7b825873addcb9ca9fbb6562c27aea347a387c9f9d8ec5f5d82e`, so changing storage
does not also change the engine. The harness snapshot is SHA-256
`a1c5c2dc3761d46ff948d3bb4acb1a225681ff5f2bfe54a937336a9af862f7ea`.
Raw data and source snapshots are in `/mnt/vortex-ssd/votex-4/results/`, also linked from
`target/scan-io-results/ssd-io/`. Storage setup and copy verification are saved as
`storage-setup.json` and `dataset-copy.json`.

With SSD temporary storage, all 46 Python analyzer/harness tests, 14 scan IO tests with
projection deferral and one read slot, and 10 default source tests pass. The Rust test
binary also executes from the SSD. Targeted Ruff and ty checks pass. No Rust code changed
for the storage switch; the executable and prior source patch are retained.

## Earlier EBS comparison

These results precede the switch to the local instance SSD. The EBS six-pair Q23 comparison confirms a
12.60% median reduction from announcing projection segments after filtering and limiting physical
reads to one slot per file: 38.376 to 33.541 seconds, winning all six pairs. Mean time improves
12.66% and median process storage traffic falls 12.39%. The candidate samples span
33.363–33.898 seconds, versus 37.778–39.229 seconds for the control. This is a directly measured
combined gain; no earlier warm or cold percentages are added.

The policy remains opt-in: driver captures show a significant first-scan-batch latency tradeoff,
the combined setting regresses cold ClickBench Q24 by 27.61%, and TPC-H Q19 is tied. Projection
deferral alone previously regressed in warm runs. Q23 sustains about 134 MB/s at
the device, making avoided storage work the main target. Earlier cold split-count, pruning and
coalescing experiments show no substantial gain.

The earlier six-round warm comparison confirms a modest Q23 improvement: the automatic wide-projection
policy measures 231.925 ms versus 241.323 ms for the current ungrouped path (3.89% lower), winning
all six paired rounds. Against the saved pre-audit executable with boxed Fetches it is 3.50%
lower. This is the directly measured combined gain; earlier percentages must not be added.
Physical bytes are essentially unchanged. Q19 and Q24 medians regress against the current
ungrouped path, so this does not establish a broad scan speedup. Earlier larger pilots and their
limitations are preserved below. The subsequent six-round earlier-file-pruning comparison adds only a 2.46% median reduction
against the automatic policy and remains opt-in.

## EBS cold IO comparison

The investigation prioritizes cold Linux data pages. The warm results below are historical
comparisons, not evidence of cold-storage improvement. For these EBS runs, the data resided on the root XFS volume,
`nvme1n1`, identified by the machine as Amazon Elastic Block Store. Its readahead setting is
128 KiB; no device settings have been changed.

The harness now verifies `posix_fadvise(DONTNEED)` using non-faulting `mincore` checks before and
after advice. It refuses a sample if more than 0.1% of file pages remain resident. Initial
checks of all 16 TPC-H Vortex files (4.831 GB) and 175 ClickBench Vortex files (16.884 GB) show
zero resident pages afterward. Those totals include the prepared format directories; the
query uses the `vortex-file-compressed` format directory. Every timed sample starts a fresh process and executes only
one query, discarding no iteration. Only file contents are evicted; metadata and device caches
can remain warm.

Linux child resource accounting adds whole-process `process_storage_read_bytes`, including
setup, readahead and reads finishing after query completion. It is derived from `ru_inblock`
in 512-byte units as implemented by
[Linux task IO accounting](https://github.com/torvalds/linux/blob/master/include/linux/task_io_accounting_ops.h).
Application bytes and counts remain separate query-snapshot metrics. Whole-process storage
bytes can establish actual storage work even when cancelled/background reads are absent from
the query's counters. They are not automatically attributable only to Vortex data ranges.

The cold harness also records the dataset's parent block-device counters around each process,
with read/write bytes, completed operations, mean device read latency, throughput and
time-weighted mean queue depth. Raw snapshots remain available. These metrics include all
activity on that device; they are not per-query counters and do not include user-space IO
queues. The average depth derives from weighted IO milliseconds divided by the observed
window, following [Linux block IO statistics](https://www.kernel.org/doc/html/latest/admin-guide/iostats.html).
Counter resets are rejected. Tests distinguish the current IO gauge, which can decrease,
from monotonic cumulative counters and verify the device-sector units.

A four-round Q23 pilot completed with the immutable automatic-policy executable above, comparing
current V2 with eight live splits, earlier file pruning and a 64 KiB coalescing gap. Each of
four variants visits each order position once. All four rounds were accepted, with no rejected
rounds or removed timing outliers. Each configuration retains four cold samples and every
post-eviction residency check reports zero cached file pages. Every query returns ten rows.
This is an initial comparison, not enough on its own to establish a production gain. Raw output is in
`target/scan-io-results/cold-io/pilot/`, with the exact harness snapshot in `cold-io/bench-io.py`.
The initial wait was restarted at its noise gate, before any benchmark process started, to
include device metrics. Its zero-sample observations remain in `cold-io/pilot-wait-only/`.
A second zero-sample gate restart adds detection of native Vortex binaries such as `to_arrow`,
whose names do not contain `bench`; observations remain in `cold-io/pilot-wait-device-metrics/`.
The final harness detects those native processes and retains the existing direct-child exclusion.

| Cold Q23 configuration | Median seconds | Change versus control | Paired wins | Median process storage GB | Median application GB | Median completed application reads |
|---|---:|---:|---:|---:|---:|---:|
| Current V2 | 37.679 | — | — | 5.049 | 4.559 | 436 |
| Eight live splits | 37.049 | 1.67% lower | 3/4 | 4.953 | 4.830 | 444.5 |
| Earlier file pruning | 38.228 | 1.46% slower | 1/4 | 5.107 | 4.518 | 439.5 |
| 64 KiB coalescing gap | 38.963 | 3.41% slower | 0/4 | 5.201 | 4.787 | 1217.5 |

These populations differ: application counters stop at query return, whereas process storage
accounting also includes setup and background completion. The difference cannot establish
how much traffic comes from kernel readahead or cancelled requests. Lower application bytes
alone do not establish a cold-storage improvement; earlier pruning demonstrates that directly.

The control's first sample sustains 134 MB/s at the block device with mean read latency around
227 ms and time-weighted mean device depth around 120. Separate cold read-phase captures show
median `pread` durations of 3.328 seconds for the control and 1.696 seconds with eight live splits.
Control median native GET time is 48 microseconds, blocking dispatch wait 5 microseconds, and
async completion resumption 11 microseconds. Concurrent request phases overlap and must not
be summed into query wall time. On this host, actual storage service dominates these dispatch
costs; none of the first four policies substantially reduces the total storage traffic.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/auto-wide/datafusion-io \
  --data-root vortex-bench/data --output target/scan-io-results/cold-io/pilot \
  --workloads clickbench-q23 \
  --variants v2-unlimited v2-splits-8 v2-file-pruning v2-io-tight-gap --modes direct \
  --iterations 1 --discard-iterations 0 --rounds 4 --diagnostic-iterations 1 \
  --timing-metrics --evict-local-files --wait-for-quiet --repeat-contended-rounds
```

The eleven harness tests, targeted Ruff formatting/lint and targeted ty checks pass. New coverage
verifies that residency inspection does not reload evicted pages and that ineffective eviction
is rejected. The storage-backed integration test initially used the repository filesystem because this
machine's `/tmp` is tmpfs, where disk eviction has no meaning. It now honors `TMPDIR`, allowing
the test to run on the SSD. The first pilot reuses the
existing frozen executable, SHA-256
`6163c62b8b3ead42d02988a803c5f38ba3674e678d27095bdbbc328ea766f3d8`.

### Cold admission and prefetch follow-up

The next exploratory comparison retains the current inline Fetch and automatic wide-projection
policy while independently disabling projection announcements, disabling all early
announcements, reducing maximum coalesced span to 4 MiB, or allowing one physical read slot per
file. Two rotated rounds across five configurations are a screening experiment, not a fully
position-balanced confirmation. The executable is immutable at
`target/scan-io-results/cold-io/admission/datafusion-io`, SHA-256
`3af02184c6cd7b825873addcb9ca9fbb6562c27aea347a387c9f9d8ec5f5d82e`;
its source patch, extra Rust files and harness snapshot are saved alongside it. Results are
written to `cold-io/admission/pilot/`.

Both timing rounds completed without detected contention; no timing samples were removed.
Post-eviction residency is zero and all queries return ten rows. The projection-only change
is the sole promising candidate; a six-round paired comparison across Q23, Q19 and Q24 is
needed before interpreting its preliminary gain as repeatable.

| Cold Q23 screening configuration | Median seconds | Change versus control | Paired wins | Median process storage GB | Median completed application reads |
|---|---:|---:|---:|---:|---:|
| Current V2 | 38.445 | — | — | 5.150 | 456.5 |
| Announce projection after filtering | 35.503 | 7.65% lower | 2/2 | 4.750 | 5220 |
| Disable all early announcements | 46.056 | 19.80% slower | 0/2 | 6.148 | 787.5 |
| 4 MiB maximum coalesced span | 38.983 | 1.40% slower | 1/2 | 5.218 | 1515 |
| One physical read slot per file | 37.641 | 2.09% lower | 1/2 | 5.038 | 479 |

Projection-only deferral lowers process storage traffic by 7.77%, despite increasing application
read bytes from 4.746 to 6.285 GB and making many more application requests. Within-query cached
reads and differing cancellation snapshots prevent equating application traffic with storage
traffic. Disabling all announcements loses this benefit: filter announcements still matter.
The one-slot cap lowers mean device depth from about 120 to 52 without materially changing
throughput or total storage traffic; reducing depth alone does not establish a large speedup.

`VORTEX_SCAN_IO_READ_CONCURRENCY` is a new opt-in cap on physical reads per file. It never raises
the reader's advertised capacity; zero and invalid values are ignored. The default remains
unchanged. Fewer slots retain more pending interests in the coalescer, where pruning can
withdraw them before selecting physical ranges. This differs from limiting live split graphs
or limiting native reads after ranges have already been selected.

The default ten source tests and strict `vortex-file` clippy pass. Running those same tests
with a global one-slot override passes seven and fails three: the batching fixture expects
one four-range submission, and two refill/straggler fixtures wait for exactly four active reads.
Those assertions are incompatible with a one-slot cap. Their original expectations remain
unchanged; this run does not validate every test under the experimental override.

The follow-up quiet gate continues to require six consecutive ten-second samples without
competing builds or benchmarks, CPU usage below 10%, IO wait below 2%, and full IO pressure below
1%. For cold runs it also requires the dataset's block device to have zero current IO in flight
when that counter is available. Historical Linux load is retained but no longer used to reject
a cold sample: many blocked threads from a finished cold query can keep load elevated while
the CPU and device are idle. Warm runs retain the load-below-four gate. Device metrics still
cover shared-host activity, and contention detection cannot provide exclusive ownership.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/cold-io/admission/datafusion-io \
  --data-root vortex-bench/data --output target/scan-io-results/cold-io/admission/pilot \
  --workloads clickbench-q23 \
  --variants v2-unlimited v2-io-project-late v2-io-no-announcements v2-io-4m v2-io-read-slots-1 \
  --modes direct --iterations 1 --discard-iterations 0 --rounds 2 \
  --diagnostic-iterations 1 --timing-metrics --evict-local-files \
  --wait-for-quiet --repeat-contended-rounds
```

The paired cold confirmation uses the same executable and six rounds for each query. There
is one timed query per fresh process, with data-page eviction before every process. Filter
announcements remain enabled in both configurations. The candidate alone sets
`VORTEX_SCAN_PROJECT_ANNOUNCE=0`. Detailed driver and scope lifetime diagnostics follow all
timings. Projection deferral remains opt-in because the earlier warm comparison regressed;
these experiments do not establish a single best policy for every storage/cache state.

The six Q23 timing pairs completed with no detected competing processes or rejected rounds.
All twelve fresh processes report zero data-page residency after eviction and return ten rows.

| Confirmed cold Q23 measure | Current V2 | Projection after filtering |
|---|---:|---:|
| Median query seconds | 38.003 | 35.581 |
| Mean query seconds | 37.949 | 36.536 |
| Query range seconds | 37.180–38.409 | 34.254–41.934 |
| Median process storage GB | 5.102 | 4.760 |
| Median completed application GB | 4.683 | 6.243 |
| Median completed application reads | 449 | 5252.5 |
| Median process CPU seconds | 6.795 | 7.754 |
| Median device read MB/s | 133.843 | 133.363 |
| Median device mean queue depth | 120.691 | 120.831 |

The candidate wins five of six paired rounds; its 41.934-second sample reads 5.612 GB from
storage at 133.356 MB/s without detected competing jobs. It remains in every summary.
These measurements support a median benefit from reduced storage traffic but not a tail-latency
improvement or lower CPU/request overhead. They are a new comparison against the same current
binary, not a percentage to add to previous warm results. Derived data is in
`cold-io/admission/confirm/analysis.json`.

The same six-round ClickBench comparison includes two narrower queries. Q19 here is the UserID
lookup, distinct from the TPC-H Q19 SF10 workload used in earlier sections.

| Cold ClickBench query | Current median ms | Projection-after-filter median ms | Median reduction | Paired wins | Current / candidate median process storage MB |
|---|---:|---:|---:|---:|---:|
| Q19 | 1845.639 | 1822.789 | 1.24% | 4/6 | 320.291 / 316.963 |
| Q23 | 38003.327 | 35581.396 | 6.37% | 5/6 | 5102.377 / 4759.585 |
| Q24 | 1958.365 | 1818.410 | 7.15% | 5/6 | 341.817 / 319.185 |

All eighteen paired rounds were accepted with no detected competing jobs; zero data-page
residency was verified for all thirty-six timed processes. No sample was removed. Q19 returns
four rows and Q23/Q24 ten rows per query. Q19 is essentially tied. Detailed driver diagnostics
follow all timings; their event-log costs mean they cannot replace the untraced comparison.

A separate six-pair TPC-H Q19 SF10 comparison uses the same binary and policies. Its control
median is 4219.067 ms versus 4160.268 ms for projection deferral (1.39% lower, five of six pairs).
Median process storage bytes are 579.662 versus 571.081 MB. Both configurations emit 504 scan
batches and 1,289,098 scan rows, and every query returns one aggregate row. All six rounds are
accepted with zero post-eviction residency and no removed samples. This is essentially a tie,
not evidence of a meaningful TPC-H speedup. Raw results are in
`cold-io/admission/tpch-confirm-sf10/`. An initial attempt used the directory name `10` instead
of the copied dataset's `10.0`; eviction rejected it before any query ran. Its zero-sample
observations and error remain in `cold-io/admission/tpch-confirm/`.

### Cold driver interpretation

Separate Q23 driver/lifetime captures expose a significant first-output tradeoff. The event
log is large and affects execution, so these are one-capture observations, not paired
production latency estimates. In particular, they do not replace the six untraced timing pairs.

| Q23 driver observation | Current V2 | Projection after filtering |
|---|---:|---:|
| Traced query window seconds | 45.525 | 37.039 |
| Scan compute union seconds | 0.938 | 1.347 |
| IO pending, with no ready delivery or compute, seconds | 42.894 | 33.673 |
| First projection compute ms | 193.060 | 59.162 |
| First returned scan batch ms | 299.701 | 15687.101 |
| Per-root projection startup p95 ms | 21552.867 | 15330.075 |
| Mean physical reads in flight | 195.67 | 715.64 |
| Peak physical reads in flight | 301 | 1295 |
| Mean physical bytes in flight MB | 2582.40 | 1928.02 |
| Ready morsel queue peak | 32 | 32 |
| Maximum reconstructed DFS depth | 3 | 3 |
| File completion to delivery p95 ms | 22.059 | 5.509 |
| Median native pread ms | 3339.150 | 1083.694 |
| Started member bytes never fetched by completed scopes MB | 809.652 | 0 |

The query-state durations are mutually exclusive observations, not CPU utilization or causal
proof. IO-without-ready-work dominates both captures; ready/active scan compute occupies only
2.555 and 3.224 seconds respectively. The candidate reaches its first projection compute
earlier but its first scan batch much later. More overlapping small reads and higher request
counts are a real tradeoff despite lower total storage traffic and median query time.

The 809.652 MB inventory is exclusive of overlapping member ranges and restricted to registrations
whose observed scopes completed without fetching them. It is still not proven avoidable storage
traffic: coalescing gaps, aliases, cancelled work and cache residency affect actual device work.
Optional withdrawal in the control is often too late: 1083 of 1836 withdrawal events follow
physical completion. The candidate has no comparable never-fetched completed-scope member
inventory, but it reads larger coalescing gaps and repeats cached ranges. These observations
motivate testing projection deferral together with the per-file physical read cap, rather than
promoting deferral on time-to-first-compute alone.

The combined-policy Q23 screen runs three rounds of current V2, projection deferral, and
projection deferral with one physical read slot per file. Every configuration visits each
order position once. `v2-io-project-late-read-slots-1` sets only the two experimental overrides
on top of the current policy. The executable remains unchanged; the exact harness snapshot is
`cold-io/admission/bench-combined.py`. Untraced timings precede driver/lifetime captures and
results are saved under `cold-io/admission/combined/`.

All three timing rounds completed without detected contention or removed samples; all nine
queries return ten rows and report zero data-page residency after eviction. The combined
policy is a stronger candidate, but three rounds remain a screen. The control is slower than
in the preceding six-round comparison, so the 16.21% reduction must not replace that earlier
measurement or be treated as a confirmed gain.

| Cold Q23 combined-policy screen | Median seconds | Change versus control | Paired wins | Median process storage GB | Median completed application reads | Mean device depth, median |
|---|---:|---:|---:|---:|---:|---:|
| Current V2 | 39.915 | — | — | 5.328 | 464 | 120.91 |
| Projection after filtering | 34.509 | 13.54% lower | 3/3 | 4.619 | 5192 | 120.68 |
| Projection after filtering, one read slot | 33.444 | 16.21% lower | 3/3 | 4.492 | 1403 | 48.10 |

The combined policy's samples span 33.282–33.529 seconds. Relative to projection deferral
alone, its median is 3.08% lower while completed application request count is about 73% lower.
Fewer available slots allow more pending targets to accumulate before coalescing and selection;
the measured storage bytes also decrease. This motivates a separate six-round confirmation
against current V2, including the narrower workloads. The source and binary are unchanged
between these matrices; only the saved environment overrides differ.

The three screening driver captures also separate physical pressure from logical backlog.
These are one traced capture per policy; they are not production latency confirmations.

| Cold Q23 screening driver measure | Current V2 | Projection after filtering | Deferral plus one read slot |
|---|---:|---:|---:|
| Traced query seconds | 45.049 | 35.363 | 34.036 |
| First returned scan batch seconds | 0.302 | 15.141 | 8.471 |
| Mean physical reads in flight | 212.22 | 692.29 | 40.27 |
| Peak physical reads in flight | 317 | 1345 | 56 |
| Mean physical bytes in flight MB | 2661.46 | 1728.60 | 272.54 |
| Peak physical bytes in flight MB | 3608.58 | 2416.60 | 381.55 |
| IO pending with no ready delivery or compute, seconds | 42.376 | 32.089 | 30.914 |

The cap reduces submitted IO pressure substantially and improves first output relative to
deferral alone, while still regressing first output relative to the early-announcement control.
Its average logical fetch backlog remains 44,552 requests; these are graph references, not
physical requests or resident byte buffers. Mean physical bytes in flight is likewise a range
inventory, not a resident-memory measurement. No policy is promoted based on these traces alone.

The subsequent six-pair confirmation compares current V2 directly with the combined policy
for Q23, TPC-H Q19 SF10 and ClickBench Q24. It retains the same executable and starts every
sample with verified data-page eviction. Lightweight read diagnostics follow all timings;
large driver traces are not repeated. Results are saved in `cold-io/admission/combined-confirm/`.

The six Q23 pairs completed without detected competing jobs or rejected rounds. All twelve
timed processes report zero post-eviction data-page residency and ten result rows. No samples
were removed. The following values supersede the combined-policy screen's 16.21% estimate.

| Confirmed cold Q23 measure | Current V2 | Deferral plus one read slot |
|---|---:|---:|
| Median query seconds | 38.376 | 33.541 |
| Mean query seconds | 38.432 | 33.565 |
| Query range seconds | 37.778–39.229 | 33.363–33.898 |
| Median process storage GB | 5.126 | 4.491 |
| Median completed application GB | 4.818 | 4.904 |
| Median completed application reads | 454.5 | 1413 |
| Median process CPU seconds | 6.966 | 7.183 |
| Median device read MB/s | 133.340 | 133.599 |
| Median device mean queue depth | 120.961 | 48.270 |

Application bytes rise slightly while actual process storage bytes fall 12.39%. The gain comes
from reduced storage work, not lower overall application request count or CPU time relative
to the original control. Compared with projection deferral alone, the physical read cap also
reduces application request overhead and physical pressure, as the screening comparison shows.
The first scan batch remains a separate tradeoff measured only in traced captures. These two
policy overrides remain opt-in; neither the six pairs nor the traces establish a general
best policy for every cache state, backend or query.

All eighteen paired rounds of the final matrix completed without detected contention or
rejected rounds. All thirty-six fresh processes report zero data-page residency after eviction.
No sample was removed; Q23/Q24 return ten rows and TPC-H Q19 one row per query.

| Final cold comparison | Current median ms | Combined-policy median ms | Change versus current | Paired wins | Current / candidate median process storage MB |
|---|---:|---:|---:|---:|---:|
| ClickBench Q23 | 38375.831 | 33540.906 | 12.60% lower | 6/6 | 5126.267 / 4491.203 |
| TPC-H Q19 SF10 | 4229.437 | 4231.256 | 0.04% slower | 3/6 | 579.662 / 579.662 |
| ClickBench Q24 | 1939.828 | 2475.378 | 27.61% slower | 0/6 | 339.444 / 403.573 |

The narrow Q24 scan reads 18.89% more storage bytes and its median device throughput drops
from 170.581 to 162.052 MB/s under the combined policy. Limiting request concurrency is not
a universal way to reduce IO: it also changes coalescing, completion order and pruning feedback.
Projection deferral alone improved Q24 in its separate six-pair matrix; that is a distinct
measurement, not a percentage to combine with this table. TPC-H Q19 likewise demonstrates
that lower device depth does not guarantee lower elapsed time: mean device depth falls from
99.48 to 1.90 while throughput and total storage bytes are effectively unchanged.

### Cold-loop validation

- The release-debug DataFusion benchmark builds and strict `vortex-file` clippy passes.
- All ten default source tests pass. With a global one-slot override, seven pass and three
  four-slot capacity/batching fixtures fail, as detailed above; their expectations were not changed.
- All fourteen IO scope tests pass with projection deferral and one read slot. They include
  array equality between V2 and the default scan for unfiltered, filtered and empty results,
  last-registration cancellation, preserving an active Fetch during Forget, and excluding
  forgotten targets from later coalescing.
- All eleven harness tests pass after the final variant addition. Targeted Ruff formatting/lint,
  targeted ty checks and `git diff --check` pass.
- No workspace-wide test or lint run was performed. The earlier broader V2-suite failures
  documented below were not rerun or hidden by changing expected inventories.

The combined cold policy remains an experimental opt-in. The evidence establishes a Q23
throughput improvement with first-scan-batch and narrow-query tradeoffs; it does not justify
changing the global defaults. The source change is the per-file admission cap, applied before
coalescing selects ranges. Benchmark updates verify cache state and distinguish actual storage
work from application counters, while the driver analyzer identifies where IO and compute wait.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/cold-io/admission/datafusion-io \
  --data-root vortex-bench/data --output target/scan-io-results/cold-io/admission/combined-confirm \
  --workloads clickbench-q23 tpch-q19 clickbench-q24 --scale-factor 10.0 \
  --variants v2-unlimited v2-io-project-late-read-slots-1 --modes direct \
  --iterations 1 --discard-iterations 0 --rounds 6 --diagnostic-iterations 1 \
  --timing-metrics --evict-local-files --wait-for-quiet --repeat-contended-rounds
```

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/cold-io/admission/datafusion-io \
  --data-root vortex-bench/data --output target/scan-io-results/cold-io/admission/confirm \
  --workloads clickbench-q23 clickbench-q19 clickbench-q24 \
  --variants v2-unlimited v2-io-project-late --modes direct \
  --iterations 1 --discard-iterations 0 --rounds 6 --diagnostic-iterations 1 \
  --timing-metrics --evict-local-files --wait-for-quiet --repeat-contended-rounds \
  --driver-trace --driver-trace-mode queues --lifetime-trace
```

## Sparse projection pilot

Grouping sparse projection across column chunk boundaries lowers ClickBench Q23's median from
246.473 to 217.586 ms in a two-round pilot, a preliminary 11.72% reduction with both paired round
medians lower. This is a candidate, not a confirmed production speedup. Each configuration retains
ten samples, from seven iterations per process with two warmups discarded. Configuration order
rotates; two rounds do not balance all three positions. The control includes a 1970.597 ms outlier.
No timing sample was removed. Four comparison rounds were accepted and none rejected.

| Configuration | Q23 median ms | Q23 round medians ms | Q19 median ms | Q19 round medians ms |
|---|---:|---|---:|---|
| Current path | 246.473 | 257.335, 229.728 | 29.759 | 29.789, 28.627 |
| Group sparse projection | 217.586 | 218.744, 214.582 | 29.468 | 30.748, 28.973 |
| Existing file pruning | 231.560 | 233.754, 226.705 | 28.849 | 29.227, 27.830 |

Q19 is a tie for grouping: its pooled median is 0.98% lower but neither paired round median is
lower. File pruning is an existing opt-in path assessed independently, not a new implementation.

| Q23 retained-sample counter or process measure | Current | Grouped |
|---|---:|---:|
| Median emitted batches | 936.5 | 779.5 |
| Median scan output rows | 3213.5 | 3058 |
| Median completed physical reads | 410.5 | 393.5 |
| Median completed application bytes MB | 5014.777 | 4866.636 |
| Median CPU seconds per seven-iteration process | 50.905 | 45.792 |

Process CPU includes startup, warmups and shutdown; it is not per-query compute time. Q23 returns
ten final rows in every retained iteration, while dynamic Top-K bounds change the scan work and
cancellation schedule. The byte reduction is about 2.95%, process CPU reduction about 10.04%, and
batch-count reduction about 16.76%. These percentages describe different populations and cannot
be added. Q19's scan work stays at 504 batches, 1,289,098 rows, 182 reads and 507.162 MB.

The candidate retains the original nonempty projection cuts when discovering prefetches, then
joins their row span into one morsel when at most 8192 rows survive and the input scope spans at
most 262,144 rows. The original mask keeps empty intervening chunks excluded. The execution graph
already joins fields with different chunk boundaries. Grouping avoids rebuilding one complete
wide projection graph and converting one output batch for every surviving boundary.

`VORTEX_SCAN_GROUP_SPARSE_PROJECTION=1` enables the experiment. The pilot binary is immutable at
`target/scan-io-results/pipeline-pass1/datafusion-sparse`, SHA-256
`2d58425285107e43553992b908f9a64f29f5cacbfdbd67f84df750b7e10fdfce`. Timing, diagnostic and
host observations are in `pipeline-pass1/quiet/`; `pipeline-pass1/analysis.json` records derived
comparisons. Detailed event logging is disabled for timings, and diagnostics follow them.

## Further candidates

A second same-binary Q23 pilot retains ten samples per configuration across two rotated rounds.
It again favors grouping, but two rounds cannot balance all six configuration positions.

| Configuration | Median ms | Lower versus control | Paired round wins |
|---|---:|---:|---:|
| Current path | 239.390 | — | — |
| Group sparse projection | 217.665 | 9.08% | 2/2 |
| Compressed LIKE | 233.945 | 2.27% | 1/2 |
| Group plus compressed LIKE | 223.898 | 6.47% | 2/2 |
| Split concurrency 8 | 234.423 | 2.07% | 1/2 |
| Split concurrency 32 | 229.871 | 3.98% | 1/2 |

Grouping's median batch count falls from 939 to 785.5, completed physical reads from 413 to
401.5, and application bytes from 5017.156 to 4849.044 MB. Whole-process CPU falls from 49.705
to 45.973 seconds. These are counter snapshots and process measures, with the same cancellation
and CPU-population limitations as the first pilot. All retained iterations return ten final rows.
No samples are removed; the combined candidate retains a 507.290 ms maximum. The two rounds
were accepted without detected competing processes.

The second pilot uses `target/scan-io-results/pipeline-pass2/datafusion-pipeline`, SHA-256
`f8ffee408c8fb62f90a2428023273e83313bf37e703030f136fdc0439e5c5df9`.
Its raw summary is `pipeline-pass2/quiet/summary.json` and its derived comparisons are in
`pipeline-pass2/analysis.json`. An eight-round confirmation was queued with that same immutable
binary, grouping alone, and Q23, Q19 and Q24. It was stopped at the noise gate with zero accepted
rounds to prioritize the user's clarified IO focus. Its gate observations are retained in
`pipeline-confirm/`. Grouping was opt-in for those binaries; the two pilots remain preliminary. The later automatic
policy and its completed comparison are described below.

The existing profile identifies OnPair decompression and canonical LIKE scanning as substantial
CPU work. Q23's SQL is `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10`.
An earlier driver report incorrectly described its predicate as `SearchPhrase <> ''`; that text
has been corrected. The actual benchmark query and dataset are unchanged.

`VORTEX_ONPAIR_COMPRESSED_LIKE=1` registers a kernel using OnPair's token-level substring search
for constant case-sensitive `%literal%` patterns, including negation. Unsupported wildcards,
escapes, case-insensitive matching, nonconstant/null patterns and literals longer than 255 bytes
fall back to existing execution. This avoids materializing strings and scanning decoded bytes.
It remains experimental until query measurements establish a benefit.

`VORTEX_SCAN_SPLIT_CONCURRENCY` experimentally overrides the number of live split tasks per file
stream, rather than limiting physical reads. Values 8 and 32 assess whether fewer active graphs
improve first output and Top-K feedback while avoiding unnecessary speculative work. The default
continues to derive concurrency from the builder and available workers.

## IO-only follow-up

The object-store adapter uses the object-storage coalescing preset even for local file payloads:
1 MiB gaps and 16 MiB total spans. The driver now accepts experimental
`VORTEX_SCAN_IO_COALESCE_MAX_BYTES` and `VORTEX_SCAN_IO_COALESCE_DISTANCE_BYTES` overrides.
They apply when opening the default file segment source, before its alignment adjustment;
defaults are unchanged. A 4 MiB span, 64 KiB gap, and their combination test whether reducing
speculative bytes and range service time compensates for additional requests. This changes
physical request formation, independently of the projection grouping and compute kernel.
The benchmark harness clears ambient overrides and provides isolated variants for each setting.
The first IO-only comparison was queued for four balanced rounds and nine iterations per process.
It remained at the noise gate with no accepted rounds and was replaced before timing by a
combined coalescing and local-handle-reuse comparison. Its observations remain in
`io-policy/coalescing-wait-only/`. The coalescing diagnostic binary is
`target/scan-io-results/io-policy/datafusion-io`, SHA-256
`f146614e55832b706e5b0c321ebe2b442fcb42c07563ecbfa0b99737e0dad554`.

While the host remained busy, a separate diagnostic pass ran three iterations per configuration
and inspected the last iteration. The timing harness was stopped during these diagnostics and
then resumed at its quiet gate. Configuration order is sequential rather than balanced and the
host/cache state changes between captures. These values describe request formation and observed
handoffs; they cannot establish a production latency improvement.

| Q23 IO diagnostic | Completed local reads | Requested MB | Repeated MB | GET p95 ms | Blocking queue p95 ms | pread p95 ms | Resume p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current 16 MiB / 1 MiB gaps | 460 | 5434.025 | 63.245 | 31.908 | 9.760 | 15.319 | 19.461 |
| 4 MiB / 1 MiB gaps | 1477 | 5106.965 | 39.827 | 31.927 | 4.825 | 3.399 | 20.382 |
| 16 MiB / 64 KiB gaps | 742 | 4594.191 | 15.692 | 22.527 | 4.545 | 9.492 | 16.047 |
| 4 MiB / 64 KiB gaps | 2218 | 4849.904 | 35.135 | 35.629 | 2.355 | 2.335 | 21.971 |

Tighter gaps reduce requested bytes by 15.45% in these captures while increasing request count
by 61.30%. The smaller span alone triples request count without proportionate byte savings.
GET preparation and async resume remain substantial: their medians increase from 3.904 and
0.092 ms to 8.081 and 1.959 ms for the 4 MiB cap. GET includes the backend's open/metadata work
and its scheduling; this instrumentation does not isolate the syscall portion. Phase intervals
overlap across reads and must not be summed into query wait time. Every capture returns ten final
rows. Raw logs, parsed counters, p50/p95/p99 and phase sums are retained in `io-policy/busy-diagnostics/`.

## Local file handle reuse

The GET preparation distribution motivates `VORTEX_LOCAL_FILE_HANDLE_REUSE=1`, another opt-in
IO experiment. Once an object-store GET returns a local file payload, the reader retains its
descriptor. Later positional reads skip the object's repeated open/metadata/GET handoff and go
directly to buffer allocation and blocking pread. The original admission permit still belongs
to the blocking operation until it finishes, including after cancellation. Each streamed range
still completes independently; there is no all-ranges completion barrier.

This reader pins the opened inode. Reads and size queries continue to reference it if the path
is replaced, which is explicitly tested against the disabled behavior. Stream payloads retain
their ordinary object-store lookup behavior. The experiment therefore stays opt-in until its
performance and desired file-version behavior are established. The disabled path retains its
owned file payload and adds no per-read Arc allocation. Diagnostic read records include
`file_handle_reused`, and summaries report observed and reused-read counts.

The initial opportunistic implementation, binary SHA-256
`c92ee7f9d9e838ad9a6f9b42bee3054a05539fdf73539db20a61edac1c57011d`, was probed separately on
the busy host using three Q23 diagnostic iterations. In the last iteration, control records
444 local reads and median GET preparation 4.960 ms. Reuse records 403 reads, 274 with the
descriptor reused, and median GET preparation 0.000364 ms. Reuse with the tighter gap records
822 reads, 680 reused, and median GET preparation 0.000143 ms. Raw outputs are in
`io-handle/busy-diagnostics/`. These confirm skipped GET work; they do not establish a query
speedup. Different request counts, cancellation and host conditions prevent a throughput claim.

The remaining first-use misses motivated shared asynchronous initialization. One successful
initial GET now supplies the descriptor to concurrent readers; a failed or cancelled initializer
leaves initialization retryable. A stream initializer keeps its original payload, and subsequent
stream reads make their ordinary GETs. Concurrent first stream reads may wait for that initial
backend classification, another reason this local-file experiment is not a global default.
`file_handle_reused` includes callers waiting for the shared initializer; their GET preparation
time still includes that wait.

The completed comparison runs Q23 and Q19 with six configurations: control, 4 MiB span, 64 KiB
gap, both coalescing overrides, handle reuse, and handle reuse with a 64 KiB gap. Six balanced
rounds and seven iterations per process retain 30 timings per configuration and query after
two warmups. Projection grouping, compressed LIKE, ready-Fetch delivery and local admission
limits remain disabled, independently of these IO settings.
The opportunistic-cache matrix stayed at its noise gate with zero accepted rounds and was
replaced before timing. Its observations remain in `io-handle/quiet/`.
The frozen shared-initialization matrix binary is
`target/scan-io-results/io-shared-handle/datafusion-io`, SHA-256
`0252e2ea19ce2bc3c3c868c9011c4f78029bc6bd9998f2ca07ab5c75f0752085`.
The tracked source patch is saved beside it; raw timing and diagnostic outputs are in
`io-shared-handle/quiet/`, and `io-shared-handle/run.log` records the gate and round progress.

All twelve workload rounds were accepted, with none rejected. Configuration order covers each
of the six positions once per workload. Timings are untraced, with scan counters printed after
the query timer; separate one-iteration read-phase diagnostics follow all timing runs.

| Configuration | Q23 median ms | Q23 lower versus control | Q23 paired wins | Q19 median ms | Q19 lower versus control | Q19 paired wins |
|---|---:|---:|---:|---:|---:|---:|
| Current 16 MiB / 1 MiB gaps | 236.543 | — | — | 28.962 | — | — |
| 4 MiB span | 245.814 | -3.92% | 0/6 | 29.595 | -2.19% | 2/6 |
| 64 KiB gap | 231.800 | 2.01% | 3/6 | 29.129 | -0.58% | 2/6 |
| 4 MiB span / 64 KiB gap | 243.199 | -2.81% | 0/6 | 29.427 | -1.61% | 2/6 |
| Shared local handle | 228.676 | 3.33% | 4/6 | 28.723 | 0.82% | 4/6 |
| Shared handle / 64 KiB gap | 234.188 | 1.00% | 3/6 | 28.819 | 0.49% | 3/6 |

The descriptor candidate has a modest Q23 advantage and Q19 is effectively tied. These results
do not establish a large IO speedup; all experiments remain opt-in. The 4 MiB cap loses every
Q23 paired round. Tight gaps win only half the paired rounds and do not improve Q19. The busy-host
diagnostic byte savings above do not predict the query result.

| Q23 retained-sample measure | Control | 4 MiB span | 64 KiB gap | Both | Shared handle | Handle / tight gap |
|---|---:|---:|---:|---:|---:|---:|
| Median completed application GB | 4.972 | 4.968 | 4.886 | 4.788 | 4.992 | 4.907 |
| Median physical reads | 406 | 1453.5 | 787 | 2116.5 | 408 | 807.5 |
| Median CPU seconds per seven-iteration process | 48.497 | 51.598 | 48.682 | 50.715 | 46.765 | 47.766 |

The smaller span adds roughly 3.6 times as many requests with practically unchanged bytes.
Handle reuse reduces process CPU while its completed-byte snapshot is slightly larger; its
advantage is consistent with avoiding GET/open handoffs, not reducing read volume. Process CPU
includes startup and warmups, and is not compute-only query time. The separate Q23 diagnostic
records 388 reused handles among 536 local reads; Q19 records 181 among 185. These are different
populations from the untraced timing counters. Every retained Q23 iteration returns ten final
rows, and Q19 returns one. Dynamic Top-K work and cancellation still vary between configurations.

No timing outliers were removed: control Q19 retains a 132.673 ms maximum, and the 4 MiB variant
retains 123.688 ms. Q23 control spans 212.821–264.536 ms and handle reuse 209.209–275.901 ms.
The noise gate and competing-process sampler do not guarantee exclusive ownership of this host.
Derived comparisons, per-round medians, counters and diagnostic reuse counts are in
`io-shared-handle/analysis.json`; all raw samples and observations remain in its `quiet/summary.json`.

## Withdrawing pruned projection interests

`VORTEX_SCAN_IO_FORGET_PRUNED_PROJECTION=1` tests precise withdrawal after the final filter mask
is known. Initial projection announcements remain early. Before selected projection prefetches
become eligible, the planner forgets ranges belonging only to empty projection cuts. Byte targets
are compared rather than segment IDs, so a selected alias retains its range. Dense scopes avoid
the extra segment walk.

`IoIntent::Forget` is optional. The default file source releases only this scope's optional
reference; a required Fetch pins the interest until clear, and another scope's Arc keeps a
shared registration live. The existing last-reference drop event removes the stale registration
from subsequent coalescing. Sources without withdrawal support decline it. Withdrawal cannot
undo a physical read already started or prevent a retained neighboring span covering a gap.

Lifetime diagnostics record each successful withdrawal, its logical bytes, observed reference
count, wanted state and phase relative to physical IO. The analyzer closes only that registration's
scope episode, including re-registration, and distinguishes these logical withdrawals from saved
physical bytes. The benchmark provides isolated pruning and pruning-plus-grouping variants.
The timing matrix compares control, withdrawal, grouping, and both together on Q23, Q24 and
TPC-H Q19. Four rotated rounds of nine iterations discard two warmups and retain 28 timings
per configuration and query. Detailed driver/lifetime tracing is disabled for these timings;
one-iteration read-phase diagnostics follow all timed workloads. All other experimental settings
are disabled. The frozen binary is `target/scan-io-results/pruned-interests/datafusion-io`, SHA-256
`476278c261a4704f74379d0d52dc8e334caa067b3c559ae2f674795f87489627`.
The tracked patch and additional Rust sources are saved beside it. Raw observations and accepted
or rejected rounds are retained in `pruned-interests/quiet/`. All three queries completed four
accepted rounds each: twelve accepted rounds total, none rejected. All raw outliers remain.

| Configuration | Q23 median ms | Q23 lower versus control | Q23 paired wins | Q19 median ms | Q19 lower versus control | Q19 paired wins |
|---|---:|---:|---:|---:|---:|---:|
| Current IO | 238.562 | — | — | 29.530 | — | — |
| Withdraw pruned interests | 240.471 | -0.80% | 1/4 | 29.764 | -0.79% | 2/4 |
| Group sparse projection | 227.397 | 4.68% | 4/4 | 29.458 | 0.24% | 2/4 |
| Withdraw and group | 221.661 | 7.08% | 4/4 | 29.886 | -1.21% | 3/4 |

Grouping repeats its Q23 advantage, smaller than the earlier two-round pilots. Withdrawal alone
does not improve either query. The combined Q23 candidate is lower in all four paired rounds,
but the non-additive interaction needs care: grouping alone uses less whole-process CPU than
the combination. Q19 differences are small and paired wins disagree with the pooled ordering.
Both switches defaulted off in this frozen matrix. This is not an original-PR-to-final
cumulative comparison.

| Configuration | Q24 median ms | Lower versus control | Paired wins |
|---|---:|---:|---:|
| Current IO | 17.792 | — | — |
| Withdraw pruned interests | 17.563 | 1.29% | 2/4 |
| Group sparse projection | 17.863 | -0.40% | 2/4 |
| Withdraw and group | 18.227 | -2.44% | 1/4 |

Q24 exposes a regression for the forced combined path. Its control retains a 109.207 ms
maximum, grouping 138.038 ms and the combination 115.386 ms; no outliers were removed.
Every retained Q24 iteration returns ten final rows. Narrow projection behavior should remain
unchanged instead of enabling this combination globally.

| Q23 retained-sample measure | Control | Withdrawal | Grouping | Both |
|---|---:|---:|---:|---:|
| Median emitted batches | 937 | 931 | 793.5 | 792 |
| Median completed application GB | 4.988 | 5.026 | 5.022 | 4.941 |
| Median completed physical reads | 413.5 | 409 | 415 | 403 |
| Median process CPU seconds, nine iterations | 62.905 | 62.964 | 59.884 | 60.399 |

The larger advantage comes with fewer driver/projection graphs and output batches; physical
bytes barely change. Whole-process CPU includes startup and warmups. Dynamic Top-K feedback and
cancelled/background reads still alter the work snapshots. Exact derived results, raw ranges,
per-round medians, output cardinality and counters are in `pruned-interests/analysis.json`.
`pruned-interests/derive.py` regenerates that comparison from the raw summary.

### Exact withdrawal capture

A separate sequential Q23 capture with full driver and lifetime logging ran on the busy host.
The timing harness was paused throughout it and resumed at its gate afterward. This strongly
perturbs the scan and cannot establish production phase percentages, latency or saved traffic.
Both captures return ten final rows. Control records 553 local reads and 3746.792 MB; withdrawal
records 611 and 4307.712 MB. These changed inventories are not an apples-to-apples byte comparison.

The withdrawal capture has 1139 requests, 1138 successful withdrawals and one required-Fetch pin.
Successful withdrawals sum to 161.097 MB of logical interests, including shared/repeated ranges.
Reference-count snapshots have median 3, p95 16 and maximum 49.

| Physical phase at withdrawal | Interests | Last reference observed | Other references observed | Logical MB |
|---|---:|---:|---:|---:|
| After completion | 802 | 170 | 632 | 134.092 |
| In flight | 177 | 2 | 175 | 16.923 |
| Before a later observed physical start | 139 | 0 | 139 | 10.023 |
| No started read observed in query window | 20 | 17 | 3 | 0.059 |

Every withdrawal preceding a later observed start still has another reference. Only 17 combine
a last-reference snapshot with no started read observed; their logical ranges total 58,044 bytes.
Even those bytes are not proven traffic savings: a coalescing gap can cover them, and the query
boundary censors later work. About 70% of withdrawals occur after physical completion, while
83% still have another scope holding the registration. This capture supports preserving shared
registrations and argues against expecting a large byte reduction from late partial pruning.

Full captures, per-run reports and summary are in `pruned-interests/withdrawal-diagnostics/`.
`withdrawal-metrics.json` preserves the joint phase/reference counts and the analyzer hash.
Its filtered event extract omits bindings and wake/poll events; its other lifetime fields are
not a replacement for the full driver report. The actual byte benefit is assessed by untraced
query counters and timing, not by summing logical withdrawals.

### Automatic wide-projection policy

The current candidate automatically groups small selections for projections with at least
eight output fields and withdraws their pruned optional interests. The existing limits remain:
at most 8192 selected rows, at most 262,144 input rows and more than one nonempty cut for grouping.
Narrow projections retain the original per-cut morsels and interest retention. This targets
the wide execution graphs behind Q23's measured combined advantage while keeping Q19's narrow
table projections and Q24's two-field projection on the original path.

`VORTEX_SCAN_GROUP_SPARSE_PROJECTION=0` disables the automatic policy; `=1` forces grouping
eligibility for narrow projections too. `VORTEX_SCAN_IO_FORGET_PRUNED_PROJECTION=0` disables
withdrawal independently, and `=1` forces it. Unset variables use the width-based policy.
The group-only and withdrawal-only benchmark variants explicitly disable the other component.
`v2-ungrouped` explicitly disables both. General default comparisons now run only `v1` and `v2`;
experimental variants must be selected.

The completed fresh matrix compares this automatic default, the current ungrouped path and
`io-overhead-audit/datafusion-before`, SHA-256
`3a918a5389ee072818013a43f49c12ce653efc036f468eb0d12d524f210ca343`.
`v2-previous-boxed` forces `VORTEX_SCAN_IO_INLINE_FETCH=0` in that saved executable. This measures
the combined allocation and pipeline work directly against that saved implementation, not a
build of the original PR commit. The candidate is `auto-wide/datafusion-io`, SHA-256
`6163c62b8b3ead42d02988a803c5f38ba3674e678d27095bdbbc328ea766f3d8`.

Each query has six balanced rounds and 42 retained samples per configuration: nine iterations
per process with two initial iterations discarded. Each configuration occupies every order
position twice. All 18 rounds are accepted and none rejected by the contention detector.
Detailed tracing is disabled during timings; separate diagnostics follow all timings. No
outliers are removed, including a 536.950 ms boxed Q23 sample and 128.876 ms ungrouped Q24 sample.

| Configuration | Q23 median ms | Q19 median ms | Q24 median ms |
|---|---:|---:|---:|
| Saved pre-audit, boxed Fetch | 240.343 | 29.261 | 18.168 |
| Current, ungrouped | 241.323 | 29.603 | 17.652 |
| Automatic wide policy | 231.925 | 29.889 | 18.052 |

Against current ungrouped, the automatic policy reduces Q23's median by 3.89% and wins 6/6 paired
rounds. Q19 is 0.97% slower and Q24 2.26% slower; each wins only 1/6 paired rounds. The width guard
preserves narrow planning behavior, but the measured regressions remain part of the result.
Against the saved boxed executable, Q23 is 3.50% lower (6/6 wins), Q19 2.15% slower (0/6), and Q24
0.64% lower (2/6). The earlier individual gains cannot be summed into a larger cumulative claim.

Q23 median output batches fall from 936.5 to 798 (14.79%) against current ungrouped, while median
completed application bytes change from 5009.806 to 5007.294 MB (0.05% lower) and physical reads
from 415 to 416. Whole nine-iteration process CPU falls from 63.509 to 61.438 seconds (3.26%).
This supports a reduction in execution/delivery overhead rather than a substantial traffic win.
Q19 retains exactly 504 batches, 182 completed reads and 507.162 MB; Q24 retains 391 batches and
359 reads with bytes varying slightly under dynamic bounds. All final query cardinalities match.
Counter snapshots can include background completion/cancellation; process CPU includes setup,
warmups and shutdown. Raw samples and gates are in `auto-wide/quiet/summary.json`, with derived
statistics and round medians in `auto-wide/analysis.json` and `derive.py`.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/auto-wide/datafusion-io \
  --baseline-binary target/scan-io-results/io-overhead-audit/datafusion-before \
  --data-root vortex-bench/data --output target/scan-io-results/auto-wide/quiet \
  --workloads clickbench-q23 tpch-q19 clickbench-q24 \
  --variants v2-previous-boxed v2-ungrouped v2-unlimited --modes direct \
  --iterations 9 --discard-iterations 2 --rounds 6 --diagnostic-iterations 1 \
  --timing-metrics --wait-for-quiet --repeat-contended-rounds
```

The earlier perturbed Q23 capture also bounds another allocation hypothesis: 120,608 Fetch
bindings refer to 18,131 distinct file registrations but 104,522 distinct
`(session, source, registration)` keys. Per-session fanout has median 1, p95 2 and maximum 3.
Only 16,086 Fetches repeat a registration within the same session, about 13.34% of bindings;
this is an upper bound for eliminating duplicate per-session future tasks, not a CPU-time saving.
Cross-session sharing already avoids duplicate physical registration and reading during
overlapping lifetimes. Reducing that larger delivery fanout would require changing completion
and wake ownership across sessions. This finding is recorded in `io-policy/fetch-fanout.json`,
derived from `quiet-factorial/clickbench-q23-v2-unlimited-direct-diagnostic.log`, including
Fetches cancelled before delivery. It is not a new production timing measurement.

## Earlier file pruning on the automatic policy

The existing `VORTEX_SCAN_FILE_PRUNING=1` path loads zone proofs before constructing and
announcing data splits. A further same-binary Q23 comparison tests that path on top of the
automatic wide policy. Six balanced rounds retain 42 samples per configuration, from nine
iterations with two initial iterations discarded. All six rounds are accepted and none
rejected. The same immutable `auto-wide/datafusion-io` executable is used.

| Configuration | Median ms | Paired round wins | Median completed reads | Median completed MB | Whole-process CPU s |
|---|---:|---:|---:|---:|---:|
| Automatic wide policy | 232.258 | — | 411 | 4979.376 | 62.070 |
| Plus earlier file pruning | 226.539 | 4/6 | 399 | 4831.825 | 59.189 |

Earlier pruning lowers the median by 2.46%, completed application bytes by 2.96% and whole
nine-iteration process CPU by 4.64%. The paired evidence is weaker than Q23's grouping result;
this is another modest effect, not a large IO speedup. All retained iterations return ten rows.
All outliers remain, including a 291.249 ms control sample. No narrow-query confirmation was
performed for this combined path, so the existing file-pruning flag remains opt-in.

Separate one-iteration diagnostics record 590 versus 613 local reads and 4930.794 versus
5012.964 MB, including 74.93 versus 93.53 MB of repeated coverage. They have different logging,
scheduling and cancellation from timing processes and cannot replace the timing counter
population or establish a latency win. Raw accepted samples, host gates and diagnostics are in
`auto-wide/early-pruning/summary.json`; `analysis.json` and `derive.py` preserve derived counters
and every round median. This experiment assesses an existing implementation, not newly added
file-pruning code.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/auto-wide/datafusion-io \
  --data-root vortex-bench/data --output target/scan-io-results/auto-wide/early-pruning \
  --workloads clickbench-q23 --variants v2-unlimited v2-file-pruning --modes direct \
  --iterations 9 --discard-iterations 2 --rounds 6 --diagnostic-iterations 1 \
  --timing-metrics --wait-for-quiet --repeat-contended-rounds
```

## Validation

The sparse-projection prefetch regression passes and checks that grouping preserves the mask's
empty middle chunk. The V2 scan suite has 33 passing cases with grouping both disabled and enabled,
including comparisons of actual output arrays with the reference path. The same six cases fail
in both configurations: four dictionary fixtures fail before execution because their generated
layout is not dictionary encoded, and two existing zone-pruning read-inventory assertions fail.
Their expected inventories were not changed. The struct-stream test checks the intentionally
different batch count and independently verifies the selected values against reference arrays.

The compressed LIKE kernel has 46 passing targeted tests covering canonical equivalence,
negation, null values, slices, filtered arrays, Unicode, empty/missing substrings and fallback
patterns including the oversized-pattern boundary. The release-debug benchmark build passes.
OnPair strict Clippy, five benchmark harness tests, Ruff and ty pass. The V2 scan failures above
remain unresolved and are not represented as passing validation.
The IO coalescing follow-up rebuild passes, as do all 17 coalescer tests, strict all-targets and
all-features Clippy for `vortex-file`, the five harness tests, Ruff, ty and `git diff --check`.
All six targeted `vortex-io` object-store reader tests pass with handle reuse enabled, covering
aligned reads, opened-file replacement and size, mixed success/EOF ranges, independent streamed
results, non-file lookup behavior and cancellation admission lifetime. Strict all-targets and
all-features Clippy for `vortex-io` passes.
The shared-initialization implementation also passes the same six tests and strict Clippy.
The replacement test now starts with a failing initial GET and concurrent valid/invalid ranges,
independently verifying the returned bytes before and after replacement. Its release-debug
benchmark build passes.

Pruned-interest withdrawal passes all 14 file scan-IO tests with the experiment enabled,
including shared-root fetch protection, pending Fetch pins, exact later coalescing membership,
and comparisons of scanned arrays. Both projection-planning cases pass, including selected
segments aliasing an empty chunk's byte range. Three driver optional-publication cases pass,
as do three source-decline cases. The combined pruning and grouping V2 suite has 33 passing
cases and the same six fixture/inventory failures described above. Strict all-targets and
all-features Clippy for `vortex-file` passes. The 34 analyzer tests and five harness tests
pass with Python 3.11, including exact withdrawal, re-registration episodes, surviving shared
references and physical-phase classification. Ruff and ty pass for the four scripts.

The automatic policy passes six projection-planning cases covering scalar, two-field and
eight-field projections with and without byte-range aliasing. Two struct integration cases
independently compare actual output arrays with the reference executor and verify the stream
batch count for narrow and wide projections. All 14 file IO cases pass with automatic behavior,
as does strict all-targets/all-features File Clippy. The default V2 suite now has 34 passing
cases and the same six fixture/inventory failures. No expected failing inventory was changed.

A further wide-projection regression passes with the automatic policy and with file pruning
explicitly enabled. It writes eight columns with different chunk sizes, filters selected rows
across large empty intervals, and independently compares the complete streamed arrays with the
reference executor. This test was added after the benchmark executable was frozen and changes
no production code in that executable.
