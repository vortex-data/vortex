<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# V2 DuckDB and DataFusion driver investigation

Investigation date: 2026-10-06. Baseline: PR
[#10339](https://github.com/vortex-data/vortex/pull/10339), revision
`afeb2d1a4879ffae48802fe03939ce3ec7a1e0fa`.

The [2026-10-08 DataFusion integration report](datafusion-v2-integration-performance.md)
validates the later V2 feature head with the standalone BETWEEN PR kept separate. It
records 462 targeted Rust tests, the native bitmap kernel gain, the full five-round
non-compact SQL comparison, and default/optional IO overlap measurements.


The current candidate preserves pruning masks during dense predicate evaluation,
reduces scheduler and preparation work, accelerates portable dense-mask rank
expansion, and limits caching to what V1 allows. The optional DuckDB preparation
queue remains disabled by default.

The subsequent [non-compact DuckDB IO-overlap investigation](duckdb-overlap-performance.md)
measures application IO queue depth, compares runtime capacity and projection
read-ahead, and records Q6 before/after profiles under the same cache limits.
Its 2026-10-07 continuation measured extension BETWEEN delegation, now split
into [#10378](https://github.com/vortex-data/vortex/pull/10378). Those historical
measurements include the extracted rule: non-compact
DuckDB Q6 executes about 21% fewer instructions by decoding date storage once,
with additional instruction savings on Q7, Q12, and Q14.

Measurements are labeled by cache-policy stage. The oldest measurements used a
per-split decoded-segment cache and forced DuckDB connection reuse. The first
cache restriction removed those features but still retained completed byte reads
inside IO sessions. An experiment that released bytes after the first fetch was
too strict: V1's pending projection futures share bytes with filters. The current
version bounds byte sharing by registered consumers. Only its explicitly labeled
comparisons cover the current policy. Older results establish historical behavior and coverage;
they do not establish performance for the final policy.

## Changes

| Area | Change | Purpose |
| --- | --- | --- |
| Shared file preparation | Lower outside the global registry lock; retain only weak references in the registry | Allow concurrent preparation without retaining data after active scans end |
| Zone plans | Initialize statistics plans only when preparing a filter | Avoid statistics-plan construction for scans without predicates |
| Preparation traversal | Borrow plan children; allocate child vectors only after a rewrite; skip unsharing when no shared plan exists | Reduce reference-count traffic and allocations without retaining prepared results |
| Portable mask scatter | Use six-stage bit expansion for dense software deposits; retain sparse paths and x86 BMI2 dispatch | Reduce dense ARM mask work without adding caches |
| Cache policy | Rebuild expression plans, natural boundaries, and fallback IO adapters per preparation; disable ordinary decoded-segment reuse | Keep V2 caching no broader than V1 |
| Byte reads | Register filter/projection consumers; consume one registration per fetch; release after the last consumer | Match V1's pending-future byte sharing without retaining bytes for later unregistered fetches |
| Predicate evaluation | Keep the initial mask as a care hint for supported dense predicates; pad skipped chunks with false results | Avoid reading excluded chunks while retaining encoded predicate execution and bitmap intersection |
| Stream execution | Poll singleton or serial split tasks directly; flatten batch iterators without another vector | Remove scheduler handoffs and allocations that expose no additional parallelism |
| DuckDB preparation queue | Cancel queued and in-flight preparation when a skipped file's result receiver is dropped | Avoid work for files the query no longer needs |
| DuckDB default path | Do not initialize the extra file-list scan with preparation disabled | Keep the optional queue disabled by default (`VORTEX_DUCKDB_FILE_PREFETCH=0`) |
| DataFusion benchmark | Apply `--threads` to Tokio workers and execution partitions | Make engine comparisons use the requested concurrency |
| DuckDB benchmark | Require prepared data, matching DataFusion | Benchmark copied Vortex files without downloading or regenerating Parquet |

## Cache policy

V1's flat reader creates a fresh segment/decode future for each predicate or
projection evaluation. Its dictionary reader retains canonical dictionary values
and evaluations on those values, and its zoned reader retains zone statistics and
pruning results, for the layout reader's lifetime. DataFusion's existing session
footer cache and per-source natural split cache also apply to V1.

V1 builds projection futures in `scan/tasks.rs::split_exec` before polling the
filter. `FlatReader::array_future` registers their segment requests immediately.
`SharedSegmentSource` holds a weak reference to shared read futures, so a pending
projection keeps a filter's bytes available without caching across executions.
Fresh evaluation futures therefore do not imply fresh physical reads.

The current V2 path disables ordinary segment caching, including reuse between
filter and projection within a split. Dictionary values and zone statistics remain
cacheable, with dynamic bounds rechecked. V2's global registry holds only weak
references; prepared scans and their outstanding task futures own the shared state.
That state therefore expires by the end of active scans, potentially earlier than
V1's reader caches. Expression plans, natural boundaries, and fallback IO adapters
are rebuilt for each preparation. These limits also apply to file pruning and the
experimental pipeline executor.

The final audit also found completed-byte retention in all three IO adapters:
`FileScanIo`, the generic `SegmentIoSource`, and the optional `FileIoService`.
They now register potential filter and projection consumers separately. Fetches
consume registrations, and the last consumed registration releases the bytes.
An unregistered later fetch creates a fresh read, unless another pending consumer
still owns the shared read, as in V1. Projection morsels each receive a
registration for their row range. The generic adapter holds actual segment
futures for each consumer, allowing the underlying source's V1 sharing behavior
to decide reuse. Clearing drops all pending/prefetched registrations. Explicit
caller-provided segment caches remain available to both V1 and V2.

The earlier implementation already reset its decode cache for every split; it did
not reuse ordinary decoded segments across executions. The broader cache was
within a split. The additional preparation caches and strong registry ownership
have now also been removed.

The comparison helper now uses DuckDB's default fresh connection per iteration.
It clears segment-cache and preload environment variables for both engines and
records these settings in its manifest. DataFusion retains its existing session
footer cache, as V1 does. OS page-cache state is not flushed, so these are warm-file
measurements, not cold-storage measurements. No query-result cache is added.

## Verification of the cache policy

The new ordinary-segment regression compares actual awaited read multiplicities
against V1 on two executions of the same prepared scan, with projection alone and
filter plus projection. Both cases pass with the default executor and the
experimental pipeline. Another regression proves that the global registry does
not retain an inactive shared file even if its reader is still alive.

After the cache-policy revision, 135 layout scan tests, 5 DuckDB scan tests,
75 supported V2 pipeline cases, and 28 DuckDB SQL/array integration cases pass
(243 test executions). The existing five unsupported pipeline list cases remain
excluded, as in the earlier diagnostic. Current logs:
`target/v2-driver-perf/test-cache-parity-{scan,pipeline,duckdb-v2}.log`.
Pinned Rust formatting, Ruff, and patch whitespace checks passed at that stage. The perf and
preparation-cancellation modules are unchanged by this cache-policy revision;
prior results for those modules remain recorded below.

Those 243 executions precede the new borrowed traversal, parallel mask expansion,
and registered-consumer byte sharing. The new mask regression and extended IO regressions
have not been executed. They cover repeated fetches, shared pending fetches,
cancellation, and stricter alignment. The final policy is exercised by rebuilt
benchmark binaries and the separately labeled query runs below; no additional
unit-test or lint run is implied by the earlier results.

## Current registered-consumer policy

The current binaries are `*-candidate-ocean-consumers`, built together in
2m 25s. Their exact SHA256 hashes, ELF build IDs, patch, and changed/untracked
source copies are in `target/v2-driver-perf/ocean-consumers-manifest.json`,
`ocean-consumers.patch`, and `ocean-consumers-source/`.

Each filter plan and potential projection morsel registers its segment consumers.
Announcements are deduplicated within one consumer, not across consumers. The
default file IO service tracks how many consumers remain, the generic adapter
holds the actual segment futures, and the optional batch service keeps its range
claim until registered consumers and pending fetches are delivered. Each fetch
consumes a registration. Once those consumers finish, a later unregistered fetch
creates a fresh read. Dictionary values, zone statistics, and existing caller
caches retain the V1 allowances above. Ordinary decoded reuse stays disabled.

### Read-volume control against V1

Separate normal FineWeb Q3 IO diagnostics use four iterations per mode and the
median of the last two. All six diagnostic processes finish (24 executions).
The current V2 read volume matches V1; the first-fetch-release experiment is
retained only to explain its rejection.

| Mode | Read bytes/query | Read count/query |
| --- | ---: | ---: |
| V1 with the current binary | 1,539,373,770 | 130.0 |
| Earlier V2 retaining bytes until root clear | 1,539,373,812 | 130.0 |
| Rejected first-fetch release | 3,017,158,284 | 293.0 |
| Current registered-consumer V2 | 1,539,373,812 | 130.0 |

Raw commands, per-iteration metrics, and mode comparisons:
`target/v2-driver-perf/ocean-consumer-io/`.

The extended IO regressions distinguish pending-future sharing from arbitrary
reuse: two registered consumers may share one physical read, but a third
unregistered fetch must read again. The generic source also preserves independent
requests when its underlying segment source does not share them. These regressions
are added but have not been run. Existing optimizer regression cases cover
nested rewrites and unchanged plan identity; they have not been rerun for this change.

### Five-suite execution and timing screen

The reference is `*-candidate-ocean-mask`, with decoded reuse already disabled
but completed raw reads retained until the IO root clears. The current candidate
adds borrowed optimizer traversal and bounded registered-consumer byte sharing.
Normal Vortex, V2, eight threads, queue 0, fresh DuckDB connections, three
executions per query, one excluded warmup, one round, and serial runs. All five
suites finish: 368 query results per version, 736 total, and 2,208 query
executions. Expected row counts are checked where supplied; complete result
contents are not compared.

| Suite | Queries per engine | DataFusion median query change | DuckDB median query change |
| --- | ---: | ---: | ---: |
| TPC-H SF10 | 22 | -5.09% | -12.78% |
| ClickBench | 43 | -4.45% | +0.20% |
| TPC-DS SF1 | 99 | -5.87% | +5.51% |
| FineWeb | 9 | -0.79% | -1.22% |
| StatPopGen | 11 | -0.36% | -1.16% |

These single-round medians are screening evidence, not confirmed speedups.
DataFusion ClickBench Q19 and Q42 rise 61.5% and 73.0%; Q29 falls 30.4%.
The repeated comparison includes those queries and the previous large outliers.
Raw commands and iterations:
`target/v2-driver-perf/ocean-consumer-broad-{tpch,clickbench,tpcds,fineweb,statpopgen}/`.
Combined summary: `ocean-consumer-screening.json`.

### FineWeb Q3 against V1

The same current binary runs V1 and V2 for both engines and both Vortex formats.
Eight threads, queue 0, eight iterations, two excluded warmups, two reversed-order
rounds, and separate query-callback perf processes. All 32 processes finish
(256 query executions). The OS page cache remains warm; benchmark segment caches
and ordinary decoded-array reuse remain disabled.

| Engine | Format | V1 median ms | V2 median ms | Paired latency change | Instructions/query change | V1 task-clock ms | V2 task-clock ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DataFusion | normal | 68.78 | 70.86 | +2.45% | -0.54% | 519.19 | 507.12 |
| DataFusion | compact | 1,311.91 | 1,177.76 | -10.29% | -12.46% | 10,264.60 | 9,127.39 |
| DuckDB | normal | 77.61 | 70.30 | -8.59% | -0.77% | 513.12 | 514.55 |
| DuckDB | compact | 1,487.30 | 1,329.36 | -10.47% | -12.73% | 10,310.31 | 9,029.26 |

The rejected policy's 35–55% normal-format regression disappears. Normal
DataFusion's round changes are -0.89%/+5.80%, with lower instructions and
task-clock, so the small latency increase is not evidence of increased CPU work.
Normal DuckDB's two rounds improve 9.30%/7.88%, with almost unchanged CPU work.
Compact gains are consistent in both rounds and supported by lower instructions
and task-clock. This is a V1/V2 comparison of the current binary, not an isolated
measurement of the last optimizer or IO change.

The compact slowdown relative to normal remains in both drivers. Its recorded
profile attributes 87–88% of busy-worker samples to Zstd sequence decoding;
decoded-array caching is not proposed as a remedy.
Raw commands, iterations, counters, and summary:
`target/v2-driver-perf/ocean-consumer-fineweb-v1-v2/`.

### Repeated outlier and instruction checks

Normal Vortex, eight threads, queue 0, eight iterations, two excluded warmups,
and two reversed-order rounds. Each timing process has a separate query-callback
perf process. The 15-query panel and six additional outlier queries finish:
42 comparisons, 336 processes, and 2,688 executions. These compare the current
registered-consumer build with the decoded-cache-disabled, byte-retaining
`*-candidate-ocean-mask` reference; they do not compare V2 with V1.

| Query | DataFusion paired latency | DataFusion instructions | DuckDB paired latency | DuckDB instructions |
| --- | ---: | ---: | ---: | ---: |
| clickbench_q01 | +2.15% | -0.06% | +1.04% | +4.06% |
| clickbench_q19 | -0.34% | -0.43% | +0.63% | +0.72% |
| clickbench_q23 | +2.16% | +4.16% | +2.60% | -0.98% |
| clickbench_q27 | -1.81% | +0.13% | -0.81% | -0.35% |
| clickbench_q29 | -3.95% | +0.60% | -0.99% | +3.56% |
| clickbench_q36 | -0.02% | +0.34% | -3.21% | -1.25% |
| clickbench_q40 | -0.91% | -0.22% | -0.66% | +1.46% |
| clickbench_q42 | +0.86% | +0.08% | +5.05% | +0.31% |
| fineweb_q00 | -5.60% | -0.02% | +1.01% | -5.30% |
| fineweb_q03 | +2.33% | +4.45% | +0.43% | +0.78% |
| tpcds_q28 | +2.65% | +0.35% | +3.74% | +2.70% |
| tpcds_q30 | +0.34% | +1.40% | -0.75% | -2.72% |
| tpcds_q32 | +1.23% | -0.22% | +8.09% | -5.58% |
| tpcds_q33 | -2.78% | -0.36% | +0.28% | -0.57% |
| tpcds_q63 | -3.65% | +0.38% | +1.94% | -0.67% |
| tpch_q06 | -5.71% | -1.31% | +0.08% | +0.83% |
| tpch_q14 | -0.01% | -0.85% | +0.57% | +2.07% |
| tpch_q16 | -0.69% | -0.42% | -5.27% | +0.11% |
| tpch_q19 | +1.30% | +1.00% | +2.85% | -0.23% |
| tpch_q20 | -0.02% | -0.00% | +7.86% | -0.63% |
| tpch_q22 | +1.40% | +0.50% | +0.17% | +1.21% |

The large DataFusion ClickBench Q19/Q42 increases and DuckDB TPC-DS Q63/Q32
increases from the single-round screen do not repeat at their screening sizes.
TPC-DS Q32 has opposite-sign DuckDB rounds (+18.0%/-1.8%) and 5.6% fewer
instructions, so its pooled paired increase is not a stable CPU regression.
DataFusion ClickBench Q23 and normal FineWeb Q3 show smaller, repeatable
latency increases (about 2%) and about 4% more instructions. Q23 task-clock
rises 9.7%; FineWeb Q3 task-clock rises 1.2%. DuckDB ClickBench Q1 also uses
4.1% more instructions, with lower task-clock. These costs remain recorded;
fair byte-sharing bounds are retained. No broad SQL speedup is established.

Commands, both round deltas, raw iterations, and all counters:
`target/v2-driver-perf/ocean-consumer-focused/` and
`target/v2-driver-perf/ocean-consumer-tpcds-outliers/`.

### Optional IO controls with registered consumers

Q6 and normal FineWeb Q3 run in both engines with default IO,
`VORTEX_SCAN_BATCH_IO=1`, and `VORTEX_SCAN_EARLY_ANNOUNCE=0`.
Six iterations, two excluded warmups, two reversed-order rounds, and separate
perf processes. All 48 processes finish (288 executions), exercising the
current batch service.

| Engine | Query | Mode versus default | Paired latency | Instructions/query | Cycles/query |
| --- | --- | --- | ---: | ---: | ---: |
| datafusion | tpch_q06 | batch | -1.30% | -0.54% | -2.21% |
| datafusion | tpch_q06 | late | +4.23% | +0.37% | +0.89% |
| datafusion | fineweb_q03 | batch | +12.45% | +0.06% | +6.60% |
| datafusion | fineweb_q03 | late | +12.75% | -0.05% | +2.17% |
| duckdb | tpch_q06 | batch | +1.97% | -0.54% | -2.65% |
| duckdb | tpch_q06 | late | -2.15% | +0.76% | +2.45% |
| duckdb | fineweb_q03 | batch | +4.74% | +3.93% | +4.90% |
| duckdb | fineweb_q03 | late | +9.90% | +1.79% | +7.83% |

Batching is mixed; the DataFusion FineWeb batch rounds disagree (-5.1%/+30.0%).
Late announcements increase FineWeb latency in both rounds of both engines.
Neither experiment justifies a default change. Early announcements stay enabled,
batch IO stays opt-in, and the DuckDB cross-file queue remains 0. Raw records
and counters: `target/v2-driver-perf/ocean-consumer-batch-io/`.

### Rejected entry-based IO bookkeeping

An additional experiment replaced remove/reinsert operations in the default
file adapter with occupied/vacant hash-map entries. It preserved the registered
consumer counts and byte lifetime. Both binaries built in 1m 56s, then ran Q6,
ClickBench Q1/Q23, and normal FineWeb Q3 against the registered-consumer build:
eight iterations, two excluded warmups, two reversed rounds, separate perf
processes, eight threads, and queue 0. All 64 processes finish (512 executions).

| Engine | Query | Paired latency | Instructions/query | Cycles/query | Task-clock/query |
| --- | --- | ---: | ---: | ---: | ---: |
| datafusion | clickbench_q01 | -0.81% | -0.50% | +0.33% | +2.95% |
| duckdb | clickbench_q01 | +7.04% | -0.78% | +1.34% | +1.38% |
| datafusion | clickbench_q23 | +4.98% | -0.01% | -1.97% | -2.16% |
| duckdb | clickbench_q23 | -6.14% | -1.06% | +0.61% | +0.25% |
| datafusion | fineweb_q03 | -1.17% | +0.56% | +8.22% | +0.84% |
| duckdb | fineweb_q03 | -4.70% | -0.12% | -1.60% | -0.68% |
| datafusion | tpch_q06 | +2.39% | +1.21% | +0.20% | +2.60% |
| duckdb | tpch_q06 | +1.98% | +0.58% | -4.14% | -7.44% |

The change is rejected: instruction savings are inconsistent, and DataFusion
Q23 latency increases about 5% in both rounds despite slightly lower CPU work.
The evidence does not establish why latency rises, but it does not justify
keeping the experiment. The original registered-consumer source and its matching
benchmark binaries are restored. The broader sweep and focused comparisons above
cover the retained production version. No unit-test or lint run was added.

The rejected binaries, hashes, source patch, and source copies remain in
`target/v2-driver-perf/ocean-entry-manifest.json`, `ocean-entry.patch`, and
`ocean-entry-source/`. Raw iterations and counters: `ocean-entry-sql/`.

## Rejected first-fetch byte-release experiment

This intermediate policy was rejected after IO diagnostics showed it was
stricter than V1. The code now tracks registered consumers instead. These
measurements explain the rejection and are not final performance claims.

The rejected experiment's binaries are `*-candidate-ocean-byte-parity`, built together with
`cargo build --locked --profile release_debug -p datafusion-bench -p duckdb-bench -j16`.
Its rebuild passed in 1m 51s. ELF build IDs, SHA256 hashes, tracked patch,
and copies of changed/untracked sources are in
`target/v2-driver-perf/ocean-byte-parity-manifest.json`, `ocean-byte-parity.patch`,
and `ocean-byte-parity-source/`. This snapshot was superseded by the registered-consumer build.
The rejected version includes borrowed optimizer traversal and consumed byte reads,
so the comparison does not isolate their individual costs.

### Five-suite execution and timing screen

The reference is `*-candidate-ocean-mask`, which still retains completed byte
reads within IO sessions. The rejected candidate releases a registration after the first fetch.
Normal Vortex, V2, eight threads, queue 0, fresh DuckDB connections, three
executions per query, one excluded warmup, one round, and serial engine/version
runs. All invocations succeed: 368 query results per version, 736 total, and
2,208 query executions. The harness checks expected row counts where supplied;
it does not compare complete result contents.

| Suite | Queries per engine | DataFusion median query change | DuckDB median query change |
| --- | ---: | ---: | ---: |
| TPC-H SF10 | 22 | -3.48% | +3.72% |
| ClickBench | 43 | -1.36% | +1.17% |
| TPC-DS SF1 | 99 | +0.95% | +1.82% |
| FineWeb | 9 | +4.57% | +21.86% |
| StatPopGen | 11 | +1.27% | -0.35% |

These are screening deltas, not confirmed performance changes. DataFusion
TPC-H Q19 rises 51%; DuckDB ClickBench Q27 rises 25% and Q29 falls 53%.
FineWeb Q3 rises 33–39% against the byte-retaining reference. The focused
comparisons below recheck these outliers. Raw commands and iterations:
`target/v2-driver-perf/ocean-final-broad-{tpch,clickbench,tpcds,fineweb,statpopgen}/`;
combined summary: `ocean-final-screening.json`.

### FineWeb Q3 V1/V2 with consumed byte reads

The rejected binaries run both drivers and both Vortex formats, eight threads,
queue 0, eight iterations, two excluded warmups, two reversed-order rounds,
and separate query-callback perf processes. All 32 processes finish
(256 query executions). OS page cache is warm; decoded-segment and benchmark
segment caches are disabled. The remaining V1 footer/dictionary/zone behavior
is retained as described above.

| Engine | Format | V1 median ms | V2 median ms | Paired latency change | Instructions/query change | V1 task-clock ms | V2 task-clock ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DataFusion | normal | 70.01 | 106.05 | +55.41% | +0.40% | 551.97 | 804.67 |
| DataFusion | compact | 1,322.00 | 1,238.37 | -6.51% | -12.45% | 10,432.55 | 9,536.94 |
| DuckDB | normal | 79.82 | 108.35 | +35.77% | -0.52% | 521.93 | 809.88 |
| DuckDB | compact | 1,554.28 | 1,407.11 | -8.30% | -12.60% | 10,471.12 | 9,572.31 |

Releasing bytes before registered projection consumers finish exposes a normal-format V2 regression in this
control. Instructions are almost unchanged, user cycles change -4.5%/+2.5%,
but task-clock rises considerably. Those measurements motivate IO diagnostics;
they do not alone distinguish waiting, kernel work, or memory transfer.
Compact V2 remains faster than V1, and the compact slowdown exists in both.
Raw iterations and counters: `target/v2-driver-perf/ocean-final-fineweb-v1-v2/`.

### IO service and announcement controls

The rejected binaries run Q6 and normal FineWeb Q3 in both engines, default IO,
`VORTEX_SCAN_BATCH_IO=1`, and `VORTEX_SCAN_EARLY_ANNOUNCE=0`. Six iterations,
two excluded warmups, two reversed-order rounds, and separate perf processes.
All 48 processes finish (288 query executions), including the optional batch
service with consumed registrations.

| Engine | Query | Mode versus default | Paired latency | Instructions/query | Cycles/query |
| --- | --- | --- | ---: | ---: | ---: |
| DataFusion | Q6 | batch | -1.92% | -0.75% | -1.21% |
| DataFusion | Q6 | late | +5.02% | +0.45% | +2.39% |
| DataFusion | FineWeb Q3 | batch | -3.59% | +1.48% | +9.96% |
| DataFusion | FineWeb Q3 | late | -4.05% | +0.02% | +0.23% |
| DuckDB | Q6 | batch | +2.95% | -1.50% | -1.72% |
| DuckDB | Q6 | late | -6.00% | +0.46% | +2.03% |
| DuckDB | FineWeb Q3 | batch | -0.57% | -1.02% | -1.59% |
| DuckDB | FineWeb Q3 | late | +1.72% | +0.52% | +0.92% |

Neither mode removes the FineWeb regression. The effects are mixed and smaller
than the V1/V2 gap. Both defaults remain unchanged; the cross-file preparation
queue also remains 0. Raw mode commands and counters:
`target/v2-driver-perf/ocean-final-batch-io/`.

### Decisive read-byte diagnostics

Separate DataFusion normal FineWeb Q3 runs, four iterations, eight threads,
with the last two IO measurements summarized. Each mode uses identical input
files and cleared benchmark caches. All five processes finish (20 executions).

| Mode | Read bytes/query | Read count/query |
| --- | ---: | ---: |
| V1 | 1,539,373,790 | 130.0 |
| V2 holding bytes until root clear | 1,539,373,812 | 130.0 |
| V2 releasing after first fetch | 3,013,154,730 | 298.5 |
| First-fetch release with batch IO | 3,018,101,520 | 323.0 |
| First-fetch release with late announcements | 3,021,727,596 | 286.0 |

The extra reads, not a kernel speed change, explain the failed cache restriction.
Source inspection confirms that V1's pending projection futures retain shared
bytes while the filter runs. Consumer-counted registrations reproduce this
lifetime without serving arbitrary later fetches from a completed-byte cache.
Raw logs and commands: `target/v2-driver-perf/ocean-final-io/`.

## Measurements after decoded-cache removal

This section predates the registered-consumer byte-sharing policy. Its references
to the first V1 cache limit mean decoded/prepared-state restrictions; ordinary
bytes could still be retained within IO sessions. The studies remain useful for
isolating the preparation and mask changes against that intermediate policy.

### Borrowed preparation traversal

A new Q6 DataFusion profile under the V1 cache limit shows substantial
reference-count traffic while traversing plan children. The experiment borrows
children for split/read visitors, skips unsharing when no shared plan exists,
and allocates rewritten child vectors only after a child changes. It adds no
cache. Profile: `target/v2-driver-perf/ocean-datafusion-q6.profile.json.gz`;
sample-weighted atomic call tree: `ocean-datafusion-q6-atomics.txt`. Use sample
weights for this recording: its reported per-thread CPU deltas also accumulate
on idle threads and cannot support CPU-time attribution.

The paired comparison uses the cache-limited candidate as its baseline, normal
Vortex files, eight threads, V2, queue window 0, fresh DuckDB connections, eight
iterations, two excluded warmups, and two alternating rounds. Perf counters run
in separate processes. All 64 processes complete (512 query executions).

| Engine | Query | Paired latency change | Instructions/query change |
| --- | --- | ---: | ---: |
| DataFusion | TPC-H Q6 | -0.64% | -1.20% |
| DuckDB | TPC-H Q6 | -0.35% | -0.19% |
| DataFusion | ClickBench Q0 | +1.28% | -0.10% |
| DuckDB | ClickBench Q0 | -1.91% | +0.02% |
| DataFusion | ClickBench Q1 | -2.74% | -0.12% |
| DuckDB | ClickBench Q1 | +1.64% | -0.53% |
| DataFusion | ClickBench Q23 | -1.14% | -0.85% |
| DuckDB | ClickBench Q23 | -1.75% | +0.10% |

The instruction reductions support a small preparation improvement. Timing
changes are small and mixed; this does not establish a general SQL speedup.
Raw results and commands: `target/v2-driver-perf/ocean-preparation/`.
Linux profile helpers now handle the recording's `samples.time` timestamps,
LLVM symbolication, and sample-weighted thread sorting.

### Fresh five-suite sweep

Both versions in this sweep use the V1 cache limit. The baseline is
`*-candidate-cache-parity` and the candidate is
`*-candidate-ocean-preparation`, recorded with its patch and binary hashes in
`ocean-preparation-manifest.json`. Normal Vortex, eight threads, V2, queue
window 0, fresh DuckDB connections, three iterations, one excluded warmup, and
one round. All processes complete: 736 query results, 2,208 query executions.
Expected row counts are checked where the suite supplies them; full result
contents are not compared.

| Suite | Queries per engine | DataFusion median query change | DuckDB median query change |
| --- | ---: | ---: | ---: |
| TPC-H SF10 | 22 | -1.97% | -1.43% |
| ClickBench | 43 | -0.31% | +0.86% |
| TPC-DS SF1 | 99 | -0.21% | +1.08% |
| FineWeb | 9 | -0.90% | -0.91% |
| StatPopGen | 11 | -1.21% | -1.34% |

These are screening results, not confirmed latency improvements. Outliers
include DataFusion TPC-H Q14/Q16/Q20/Q22, ClickBench Q40/Q42, and DuckDB
FineWeb Q0. Raw commands, source snapshots, iterations, and suite summaries:
`target/v2-driver-perf/ocean-broad-{tpch,clickbench,tpcds,fineweb,statpopgen}/`;
combined summary: `ocean-screening.json`.

### Portable mask scatter

The Q6 profile also points to `intersect_by_rank` on ARM. A bitwise rewrite of
the per-bit conditional was rejected: the compiler already emits a branchless
selection and the rotating-mask timings are effectively unchanged. Disassembly
is saved as `ocean-mask-{baseline,branchless}.asm`.

The second experiment computes six parallel compaction stages for each 64-bit
mask word and applies those moves in reverse to scatter rank bits. It uses
48 bytes of stack scratch per word, adds no cache, retains the sparse-source
and small-mask paths, and leaves x86 BMI2 dispatch unchanged. The algorithm
is a 64-bit adaptation of the parallel expansion described in
[Hacker's Delight's accompanying source](https://raw.githubusercontent.com/hcs0/Hackers-Delight/master/expand.c.txt).
A deterministic random-word oracle regression was added; it has not been run.

`vortex-mask/benches/intersect_by_rank.rs` covers 30 cases, including eight
rotating random pairs of 262,144 rows each. Two rounds reverse baseline/candidate
order on CPU 24, with 200 requested samples, 0.2 seconds minimum, and 0.6 seconds
maximum per case. Divan extends sampling to satisfy the minimum time.

| Rotating case | Baseline median (µs), rounds 0/1 | Parallel median (µs), rounds 0/1 |
| --- | ---: | ---: |
| self 50%, rank 50% | 119.0 / 118.5 | 47.33 / 47.30 |
| self 90%, rank 90% | 179.9 / 180.2 | 47.58 / 47.53 |
| self 2%, rank 50% | 28.68 / 29.35 | 29.25 / 29.60 |
| self 50%, rank 2% | 60.35 / 60.04 | 60.64 / 59.43 |

The dense rotating cases improve by 60% and 74%; the sparse rotating controls
range from -1% to +2%. These are kernel results on this ARM host, not SQL claims.
Raw logs: `target/v2-driver-perf/ocean-mask-{baseline,parallel}-r{0,1}.log`;
all-case comparisons: `ocean-mask-comparison-r{0,1}.json`.

```bash
cargo bench --locked --profile release_debug -p vortex-mask \
  --bench intersect_by_rank --no-run -j 16
taskset -c 24 target/v2-driver-perf/mask-parallel --bench \
  --sample-count 200 --min-time 0.2 --max-time 0.6
```

The comparison helper now supports `--perf-stat --perf-branches`, collecting
query-callback branch counts and misses in separate diagnostic processes. This
lets SQL follow-ups distinguish reduced instruction work from branch behavior.

### SQL after parallel mask expansion

The baseline is `*-candidate-ocean-preparation`; the candidate is
`*-candidate-ocean-mask`, with ELF build IDs and a patch snapshot in
`ocean-mask-manifest.json`. Both keep the V1 cache limit. Four queries, both
engines and Vortex formats, eight threads, queue 0, fresh DuckDB connections,
eight iterations, two excluded warmups, two alternating rounds, and separate
query-callback perf processes with branch counters. All 128 processes finish
(1,024 query executions).

| Engine | Query | Format | Paired latency | Instructions/query | Cycles/query |
| --- | --- | --- | ---: | ---: | ---: |
| datafusion | clickbench_q01 | compact | +0.05% | +1.13% | +0.25% |
| datafusion | clickbench_q01 | normal | +1.79% | -0.42% | -1.10% |
| duckdb | clickbench_q01 | compact | -1.12% | -3.14% | -1.83% |
| duckdb | clickbench_q01 | normal | -1.66% | -1.92% | -4.29% |
| datafusion | clickbench_q23 | compact | +0.03% | +0.32% | +0.30% |
| datafusion | clickbench_q23 | normal | +0.58% | +1.01% | +2.72% |
| duckdb | clickbench_q23 | compact | +2.34% | -0.21% | -0.56% |
| duckdb | clickbench_q23 | normal | +0.44% | -0.58% | -0.80% |
| datafusion | fineweb_q03 | compact | +2.20% | +0.03% | -0.17% |
| datafusion | fineweb_q03 | normal | +1.31% | +2.16% | -0.74% |
| duckdb | fineweb_q03 | compact | -0.50% | +0.15% | +0.03% |
| duckdb | fineweb_q03 | normal | -0.57% | -1.90% | -1.08% |
| datafusion | tpch_q06 | compact | +0.09% | -1.07% | -0.12% |
| datafusion | tpch_q06 | normal | +0.17% | -0.19% | -1.85% |
| duckdb | tpch_q06 | compact | -0.70% | -0.48% | -0.70% |
| duckdb | tpch_q06 | normal | -1.06% | -0.68% | -2.54% |

The SQL effects are small and mixed. Dense-mask microbenchmarks do not establish
an equivalent SQL improvement: the actual queries also use sparse masks, bitmap
AND, decompression, string predicates, and projection. Dynamic filtering can
change Q23/FineWeb read demand, so their instruction changes are not solely
kernel costs. Q6 branch misses decrease 0.7–5.1% across the four configurations.
Raw iterations, branch counters, perf commands, and summaries:
`target/v2-driver-perf/ocean-mask-sql/`.

Compact FineWeb Q3 takes about 1.2 seconds in DataFusion and 1.3 seconds in
DuckDB; normal Vortex takes about 62 milliseconds in both. This format gap is
much larger than the mask optimization.

### Compact FineWeb Q3 profile

Forty executions of DataFusion FineWeb Q3, compact Vortex, V2, eight threads,
queue 0, and the V1 cache limit were recorded at 500 Hz. The exact executable is
`datafusion-candidate-ocean-mask`. The three busiest workers have about 22,000
samples each: 87.3–88.4% self samples are in
`ZSTD_decompressSequences_default.constprop.0`, another 4.9–5.3% are in Huffman
decode, and 2.1–2.3% are in `vortex_zstd::array::walk_views`. The hot stacks run
through `ZstdData::decompress_slice`. Decompression dominates this recording.

The compact compressor enables Zstd for string columns, whereas the normal
preset excludes it (`vortex-btrblocks/src/builder.rs`). FineWeb Q3 combines
`url LIKE '%google%'` and `text LIKE '%Google%'` and projects every column.
FSST supports these simple containment predicates directly on its encoding;
Zstd takes the decompression path. This explains a codec-specific cost, but the
profile alone does not prove whether frame decoding is repeated unnecessarily.
The V1/V2 control below isolates the driver on identical files.

Use sample weights: the Linux profile's CPU deltas overstate idle-worker time.
Samply emitted duplicate-module-start-address warnings, but all 3,307 binary
addresses in this summary symbolicated and the hot decode stacks agree across
workers. No lost-event message was reported. Raw commands, logs, exact binary
identity, and sample-weighted summary are under `target/v2-driver-perf/` with
the `ocean-datafusion-fineweb-q3-compact` prefix.

```bash
samply load target/v2-driver-perf/ocean-datafusion-fineweb-q3-compact.profile.json.gz
```

### FineWeb Q3 V1/V2 control

The same `*-candidate-ocean-mask` binaries run FineWeb Q3 in V1 and V2, normal
and compact Vortex, eight threads, queue 0, fresh DuckDB connections, eight
iterations, two excluded warmups, and two rounds with reversed scan order.
Separate perf processes measure query callbacks. All 32 processes complete
(256 query executions). This control still predates consumable byte reads.

| Engine | Format | V1 median ms | V2 median ms | Paired latency change | Instructions/query change |
| --- | --- | ---: | ---: | ---: | ---: |
| DataFusion | normal | 60.30 | 59.92 | -0.87% | -0.11% |
| DataFusion | compact | 1,301.27 | 1,171.42 | -9.71% | -12.46% |
| DuckDB | normal | 66.90 | 62.07 | -7.32% | -0.79% |
| DuckDB | compact | 1,476.42 | 1,312.66 | -11.06% | -12.70% |

The compact format gap also exists in V1. The profile and control identify Zstd
decompression as a stronger target than preparation for this query. They do not
establish that redundant frame decoding is absent or prove a codec change's
benefit. Raw commands, iterations, and counters:
`target/v2-driver-perf/ocean-fineweb-v1-v2/`.

### Comparison against the original PR

The cache-policy binaries before borrowed traversal are `*-candidate-cache-parity`; see
`target/v2-driver-perf/cache-parity-manifest.json`, `cache-parity.patch`, and
`build-cache-parity.log`. The rebuilt production benchmark binaries use this policy.

Fresh-connection controls cover Q6 and ClickBench Q23, both engines, both Vortex
formats, V1/V2, eight threads, and queue window 0. Each process runs four iterations
with one excluded warmup; two paired rounds alternate order. All 64 runs complete
(256 query executions). The reference is the instrumented original PR; its V2
driver still has the broader caches. Both versions use fresh DuckDB connections
and cleared benchmark segment-cache/preload settings. This comparison measures
the resulting current driver, rather than isolating one cache removal.

| Engine | Query | Vortex format | PR baseline (ms) | Current (ms) | Paired change |
| --- | --- | --- | ---: | ---: | ---: |
| datafusion | ClickBench Q23 | compact | 1397.997 | 1389.057 | -0.6% |
| datafusion | ClickBench Q23 | normal | 474.029 | 459.927 | -3.0% |
| datafusion | Q6 | compact | 81.888 | 83.444 | +1.5% |
| datafusion | Q6 | normal | 37.637 | 39.738 | +5.0% |
| duckdb | ClickBench Q23 | compact | 163.680 | 159.548 | -3.2% |
| duckdb | ClickBench Q23 | normal | 98.811 | 90.822 | -9.3% |
| duckdb | Q6 | compact | 90.590 | 93.695 | +0.8% |
| duckdb | Q6 | normal | 45.556 | 48.504 | +8.0% |

Normal-format Q6 shows a 5–8% latency regression against the original PR in this
study. Q23 timings improve, but unchanged V1 controls also drift: normal DuckDB
Q23 shows -9.4% against -9.3% for V2, and normal DataFusion Q6 shows -4.3%.
These controls qualify any latency gain. Raw iterations and commands are in
`target/v2-driver-perf/cache-parity-controls/`.

```bash
taskset -c 24-31 python3 scripts/bench-v2-drivers.py \
  --baseline duckdb=target/v2-driver-perf/duckdb-baseline-final \
  --candidate duckdb=target/v2-driver-perf/duckdb-candidate-cache-parity \
  --baseline datafusion=target/v2-driver-perf/datafusion-baseline-final \
  --candidate datafusion=target/v2-driver-perf/datafusion-candidate-cache-parity \
  --data-root vortex-bench/data --output target/v2-driver-perf/cache-parity-controls \
  --workload tpch:6 --workload clickbench:23 --formats vortex vortex-compact \
  --threads 8 --scan v1 v2 --prefetch 0 --iterations 4 --warmup 1 --rounds 2
```

The separate counter study uses normal Vortex, V2, six iterations with two
excluded warmups, two paired rounds, eight threads, and queue window 0. It completes
16 timing processes and 16 diagnostic processes (192 additional query executions).
Perf counts cover query callbacks; DuckDB connection reopening is included.

| Engine | Query | Instructions | Task-clock | Paired latency |
| --- | --- | ---: | ---: | ---: |
| datafusion | ClickBench Q23 | -0.92% | -1.84% | -3.68% |
| datafusion | Q6 | +7.93% | +7.55% | -0.91% |
| duckdb | ClickBench Q23 | +0.34% | -0.80% | -4.00% |
| duckdb | Q6 | +0.93% | +0.75% | -1.07% |

DataFusion Q6 performs approximately 8% more instructions under the current
driver and cache limits than the original PR. DuckDB Q6 increases by about 1%;
Q23 is within 1% for both engines. Q6 latency regressions from the first study
do not repeat in the counter study, so latency remains qualified by host noise.
The policy removes extra reuse as requested; it does not establish a CPU gain
for Q6. Raw counters and commands are in
`target/v2-driver-perf/cache-parity-counters/`.

Reproduce the counter study with the command above, replacing the output path
with a new directory and using `--formats vortex --scan v2 --iterations 6
--warmup 2 --rounds 2 --perf-stat`. The full five-suite sweep has not been repeated
after this revision; its earlier results remain historical. No workspace-wide
linting or testing was run.

## Inputs and tooling

Benchmark data was copied from `/home/ec2-user/votex-2/vortex-bench/data` into
`vortex-bench/data`. It includes TPC-H SF10, partitioned ClickBench (100 files),
TPC-DS SF1, FineWeb, and StatPopGen (1000 rows), with normal and compact Vortex
formats. TPC-H Parquet was generated with ZSTD level 3. Existing ClickBench
Parquet was preserved. Parquet controls are restricted to TPC-H and ClickBench;
TPC-DS, FineWeb, and StatPopGen have copied Vortex inputs but no prepared Parquet.
Benchmark data and raw measurements are git-ignored.

Installed tooling: Linux perf, numactl, the local `vx-bench` orchestrator, and the
pinned `nightly-2026-09-10` formatter. Samply and LLVM symbolication are available.
Measurements use ARM64, Rust 1.98.0, the `release_debug` profile, and an eight-core
affinity mask (`24-31`) for the focused studies. The thread scaling controls
use sixteen CPUs (`16-31`) so 16 workers are not oversubscribed. Both engines are built together so
Cargo feature unification matches between baseline and candidate:

```bash
cargo build --locked --profile release_debug -p datafusion-bench -p duckdb-bench -j16
```

The instrumented baseline restores the PR's original driver files and retains
the candidate's benchmark controls. Immutable executable copies, build logs,
and the exact restored paths are in `target/v2-driver-perf/`; see
`final-baseline-manifest.json`.

The historical wide and focused comparisons use the preserved `*-candidate-final` binaries.
The subsequent pipeline compatibility fix is in `*-candidate-pipeline-compatible`
and the refreshed `target/release_debug/*-bench` binaries; see
`pipeline-compatible-manifest.json`. The final default-path check compares those
two candidate versions directly, so it isolates the compatibility guard.
Partially masked pipeline predicates use selected evaluation. These executable
copies predate the cache-policy revision; current artifacts are recorded separately.

## What the profile showed

The Q6 DuckDB recording is
`target/v2-driver-perf/duckdb-q6-attached.profile.json.gz`, symbolicated against
the preserved `duckdb-preflight` executable. On a representative worker, self
samples were approximately 16.7% bit-packed decompression, 12.0% mask
intersection, 10.7% integer comparison, and 7.4% fixed-width filtering. These
samples identify execution work; they do not establish that preparation is the
dominant cost. The recording reported lost events, so the percentages are
approximate.

```bash
samply load target/v2-driver-perf/duckdb-q6-attached.profile.json.gz
python3 .agents/skills/samply/scripts/profile_summary.py \
  target/v2-driver-perf/duckdb-q6-attached.profile.json.gz \
  --binary target/v2-driver-perf/duckdb-preflight --symbolicate \
  --weight-mode cpu --top 10 --threads 2 --stacks 2 --stack-depth 9
```

A numeric-predicate experiment intended to reduce mask scattering was removed:
four paired rounds showed DataFusion Q6 about 1.5% slower and no useful gain on
Q19 or ClickBench Q23. Evidence remains in
`target/v2-driver-perf/numeric-isolation/` and `numeric-experiment.patch`.

## Comparison workflow

`scripts/bench-v2-drivers.py` runs engines serially, alternates baseline/candidate
order between rounds, excludes warmup, and records raw iterations, commands,
environment, CPU affinity, and source changes. It supports all five copied
suites, V1/V2, both Vortex formats, Parquet controls, thread counts, and optional
DuckDB preparation windows.

`--perf-stat` launches a separate diagnostic process. Acknowledged perf control
FIFOs enable counters around query callbacks after warmup. Initial setup, data
registration, result reporting, and warmup are excluded. With fresh DuckDB
connections, the callback includes reopening the connection; its instruction and
CPU counts therefore include reopening, while DuckDB's reported query time does not. Both executable copies
must contain the benchmark runner's perf-control support. Counts are normalized
by measured query executions. For query-level attribution, use one query per
`--workload`; a multiple-query invocation produces aggregate counts instead.

```bash
taskset -c 24-31 python3 scripts/bench-v2-drivers.py \
  --baseline duckdb=target/v2-driver-perf/duckdb-baseline-final \
  --candidate duckdb=target/v2-driver-perf/duckdb-candidate-final \
  --baseline datafusion=target/v2-driver-perf/datafusion-baseline-final \
  --candidate datafusion=target/v2-driver-perf/datafusion-candidate-final \
  --data-root vortex-bench/data --output target/v2-driver-perf/query-counters-dense \
  --workload tpch:6 --workload tpch:19 --workload clickbench:0 \
  --workload clickbench:1 --workload clickbench:23 --formats vortex --threads 8 --prefetch 0 4 \
  --iterations 12 --warmup 2 --rounds 3 --perf-stat
```

Inspect `summary.json` for paired-round changes and counter comparisons;
`*.perf.csv` and `*.perf.metrics.json` retain the underlying hardware counts.
Separate timing and counter processes avoid mixing instrumentation overhead
into the reported latency comparison.

For format and thread controls, keep the same binary arguments and data root,
use `--workload tpch:6 --workload clickbench:23 --formats vortex vortex-compact parquet
--threads 1 8 16 --prefetch 0 --iterations 5 --warmup 2 --rounds 2`, and pin the
process to CPUs `16-31`. Use a new output directory for every study.

For scan controls, use `--scan v1 v2 --formats vortex vortex-compact --threads 8`.
For queue controls, use DuckDB only with `--workload clickbench:0,1,23
--formats vortex vortex-compact --threads 8 --prefetch 0 1 4 8`.

The host also runs unrelated benchmarks and builds. Earlier timing studies
had large drift; `focused-candidate3` is unsuitable for speedup claims.
Instruction counts can establish reductions in CPU work, but cannot establish
latency improvements on a contended host. A quiet-host rerun is required for
release-level latency claims.

## Final focused comparison

Three alternating rounds, 12 iterations per invocation, two warmups, eight threads,
V2, normal Vortex, and separate query-only perf processes. Negative changes are
improvements. The latency change is the median of paired round changes;
instruction changes compare the median query-normalized counts. Each DuckDB
comparison uses the same preparation window in baseline and candidate.

| Query | Engine | Window | Baseline ms | Candidate ms | Paired latency | Instructions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| clickbench_q00 | datafusion | 0 | 0.742 | 0.717 | +0.97% | -0.03% |
| clickbench_q00 | duckdb | 0 | 0.902 | 0.908 | -0.42% | -0.09% |
| clickbench_q00 | duckdb | 4 | 0.892 | 0.963 | +3.78% | -0.01% |
| clickbench_q01 | datafusion | 0 | 8.287 | 8.255 | +1.70% | +0.06% |
| clickbench_q01 | duckdb | 0 | 10.553 | 10.481 | -2.14% | +3.54% |
| clickbench_q01 | duckdb | 4 | 10.265 | 10.474 | +3.36% | -0.06% |
| clickbench_q23 | datafusion | 0 | 457.961 | 450.524 | -1.08% | +1.09% |
| clickbench_q23 | duckdb | 0 | 92.479 | 100.570 | +3.98% | +2.15% |
| clickbench_q23 | duckdb | 4 | 104.391 | 102.888 | -3.00% | +0.10% |
| tpch_q06 | datafusion | 0 | 36.113 | 36.528 | +2.03% | -0.75% |
| tpch_q06 | duckdb | 0 | 41.766 | 41.871 | +1.92% | +0.04% |
| tpch_q06 | duckdb | 4 | 42.405 | 42.188 | -1.78% | +0.03% |
| tpch_q19 | datafusion | 0 | 52.460 | 51.881 | -0.03% | -0.35% |
| tpch_q19 | duckdb | 0 | 86.056 | 83.947 | -1.26% | +0.49% |
| tpch_q19 | duckdb | 4 | 87.358 | 87.150 | +3.30% | +0.51% |

Most default-window results are close to baseline. DuckDB ClickBench Q23 at
window 4 improved 2.4–3.0% in all three rounds, with essentially unchanged
instructions. This compares implementations at window 4; it does not establish
that enabling the queue beats window 0. Default DuckDB Q1 and Q23 used 3.5% and
2.1% more instructions, respectively. These are material remaining costs,
despite variable latency changes, and should be revisited with a focused profile.
DataFusion Q6 used 0.75% fewer instructions, but its latency did not improve.
There is no evidence here for a broad SQL speedup.

Raw results, per-round deltas, perf CSV files, and manifests:
`target/v2-driver-perf/query-counters-dense/`. A same-binary control study in
`perf-control-final-smoke/` showed roughly 1–5.5% latency variation while
DataFusion instruction counts varied by only 0.008%.

## Five-suite execution sweep

V2, normal Vortex, eight threads, queue disabled, three executions per query,
one warmup, and one comparison round. Every invocation succeeded. The benchmark
runner checks expected row counts where the suite supplies them; this sweep
does not compare complete result contents.

| Suite | Queries per engine and implementation | Candidate query results | Baseline query results |
| --- | ---: | ---: | ---: |
| tpch | 22 | 44 | 44 |
| clickbench | 43 | 86 | 86 |
| tpcds | 99 | 198 | 198 |
| fineweb | 9 | 18 | 18 |
| statpopgen | 11 | 22 | 22 |

Across both engines: 368 candidate results and 368 baseline results, totaling
2,208 query executions. Raw results and logs are in
`target/v2-driver-perf/broad-{tpch,clickbench,tpcds,fineweb,statpopgen}/`.
The manifest for each suite records the exact command and source snapshot.

Large single-round timing changes, including TPC-H Q20 and Q21, are unsuitable
for performance claims on this host. The sweep establishes successful execution
coverage; the focused multi-round study is the stronger timing/counter evidence.

## SQL IO diagnostics

Separate DataFusion runs used `--io-diagnostics --show-metrics`, three iterations,
and normal Vortex at eight threads. The table uses the median of the last two
iterations. These are aggregated Vortex read metrics; they do not distinguish
disk reads from operating-system page-cache hits.

| Query | Baseline read MiB | Candidate read MiB | Baseline read count | Candidate read count |
| --- | ---: | ---: | ---: | ---: |
| tpch-6 | 379.421 | 379.421 | 154.0 | 154.0 |
| clickbench-1 | 2.047 | 2.049 | 200.5 | 199.0 |
| clickbench-23 | 3236.526 | 3257.041 | 249.5 | 252.0 |

No material SQL read-byte reduction appears in these three cases. Q23 read
demand varies with dynamic filtering and scheduling. The synthetic zone-pruning
tests are the direct evidence for avoiding excluded data segments. Logs, exact
commands, and per-iteration metrics are in
`target/v2-driver-perf/io-diagnostics/`.

## Format and thread controls

TPC-H Q6 and ClickBench Q23, both engines, three formats, 1/8/16 threads,
V2, window 0, five iterations, two warmups, and two alternating rounds. All
36 configurations completed: 72 candidate and 72 baseline query results.
Affinity was `16-31` throughout this study. Values below are pooled candidate
medians in milliseconds; the change column is the median paired round change
at eight threads.

| Query | Engine | Format | 1 thread ms | 8 threads ms | 16 threads ms | Paired change at 8 threads |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tpch_q06 | duckdb | vortex-file-compressed | 194.851 | 45.739 | 30.988 | +8.69% |
| tpch_q06 | duckdb | vortex-compact | 550.279 | 82.446 | 49.531 | +0.02% |
| tpch_q06 | duckdb | parquet | 751.814 | 105.946 | 54.607 | -0.46% |
| tpch_q06 | datafusion | vortex-file-compressed | 143.493 | 31.540 | 27.391 | -2.60% |
| tpch_q06 | datafusion | vortex-compact | 536.898 | 79.627 | 45.871 | +0.39% |
| tpch_q06 | datafusion | parquet | 1191.580 | 174.055 | 108.414 | -0.09% |
| clickbench_q23 | duckdb | vortex-file-compressed | 355.954 | 86.760 | 63.607 | +0.00% |
| clickbench_q23 | duckdb | vortex-compact | 959.789 | 157.171 | 99.022 | +0.20% |
| clickbench_q23 | duckdb | parquet | 2165.728 | 294.746 | 172.474 | +0.44% |
| clickbench_q23 | datafusion | vortex-file-compressed | 2978.933 | 438.029 | 367.438 | -0.51% |
| clickbench_q23 | datafusion | vortex-compact | 10664.989 | 1398.172 | 805.787 | +0.87% |
| clickbench_q23 | datafusion | parquet | 20070.740 | 4963.616 | 2839.246 | +7.20% |

Thread scaling and encoding costs dominate the differences between columns.
Parquet Q23 at eight threads changes +7.2% for DataFusion despite using the
same Parquet execution path; this reinforces the host-noise limitation. The
default-window DataFusion normal-Vortex Q23 pair at sixteen threads changes
+9.2%, so the study does not establish a uniform improvement at higher
concurrency. The larger compact-format costs deserve an execution/encoding
profile, rather than assuming file preparation explains them. Raw study:
`target/v2-driver-perf/format-thread-controls/`.

## V1 and V2 controls

The same two queries and engines ran normal and compact Vortex with eight
threads, windows disabled, five iterations, two warmups, and two rounds. All
16 configurations completed, producing 32 candidate and 32 baseline results.
These V1/V2 differences include the V2 implementation already present in the PR;
they are not improvements attributable solely to this candidate.

| Query | Engine | Format | Candidate V1 ms | Candidate V2 ms | V2 versus V1 |
| --- | --- | --- | ---: | ---: | ---: |
| tpch_q06 | duckdb | vortex-file-compressed | 71.792 | 55.258 | -23.03% |
| tpch_q06 | duckdb | vortex-compact | 113.345 | 91.489 | -19.28% |
| tpch_q06 | datafusion | vortex-file-compressed | 45.905 | 44.898 | -2.19% |
| tpch_q06 | datafusion | vortex-compact | 87.629 | 84.423 | -3.66% |
| clickbench_q23 | duckdb | vortex-file-compressed | 102.073 | 105.633 | +3.49% |
| clickbench_q23 | duckdb | vortex-compact | 274.785 | 164.739 | -40.05% |
| clickbench_q23 | datafusion | vortex-file-compressed | 537.669 | 501.295 | -6.77% |
| clickbench_q23 | datafusion | vortex-compact | 3469.396 | 1532.959 | -55.81% |

DataFusion normal-Vortex Q6 also changes +9.4% for candidate versus baseline
on V1, and +8.1% on V2 in this study. The similar V1 movement is a further
warning against assigning those wall-time changes to V2-specific preparation.
Use the query-only counters in the focused study to distinguish CPU work from
this drift. Raw study: `target/v2-driver-perf/v1-v2-controls/`.

## Queue-window controls

DuckDB ClickBench Q0/Q1/Q23 ran normal and compact Vortex, eight threads,
V2, windows 0/1/4/8, five iterations, two warmups, and two alternating rounds.
All 24 configurations completed: 48 candidate and 48 baseline query results.
The table compares candidate medians across windows in milliseconds.

| Query | Format | Window 0 ms | Window 1 ms | Window 4 ms | Window 8 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| clickbench_q00 | vortex-file-compressed | 0.953 | 1.026 | 1.040 | 1.016 |
| clickbench_q00 | vortex-compact | 0.948 | 0.931 | 0.934 | 0.997 |
| clickbench_q01 | vortex-file-compressed | 10.169 | 19.556 | 10.413 | 10.658 |
| clickbench_q01 | vortex-compact | 16.112 | 23.748 | 16.309 | 17.418 |
| clickbench_q23 | vortex-file-compressed | 101.308 | 101.947 | 104.079 | 112.329 |
| clickbench_q23 | vortex-compact | 161.489 | 171.146 | 169.520 | 186.400 |

There is no consistent local-file benefit from enabling preparation across
these cases. Preserve `VORTEX_DUCKDB_FILE_PREFETCH=0` as the default. A nonzero
window is an opt-in for workloads where preparation can overlap useful work;
test it against window 0 using the same data, engine, cache state, and thread
count. Raw study: `target/v2-driver-perf/queue-controls/`.

## Large-outlier recheck

TPC-H Q20/Q21, both engines, normal Vortex, eight threads, V2, window 0,
seven iterations, two warmups, three alternating rounds, and separate perf
diagnostic processes. The broad sweep's large Q20/Q21 timing changes did not
persist in this isolated comparison.

| Query | Engine | Baseline ms | Candidate ms | Paired latency | Instructions |
| --- | --- | ---: | ---: | ---: | ---: |
| tpch_q20 | datafusion | 172.342 | 175.520 | -0.84% | -0.13% |
| tpch_q20 | duckdb | 153.462 | 159.365 | +0.18% | +0.03% |
| tpch_q21 | datafusion | 700.849 | 691.514 | -0.73% | -0.07% |
| tpch_q21 | duckdb | 665.080 | 685.334 | +1.07% | -0.16% |

Instruction changes are within 0.17%, and median paired latency changes are
within 1.1%. This removes the specific concern raised by those broad-sweep
outliers; it does not eliminate the host-contamination limit for other timing
claims. Evidence: `target/v2-driver-perf/outlier-check/`.

## Final pipeline-guard check

This check compares the previously measured candidate with the final candidate,
isolating the pipeline compatibility guard. It uses the normal executor,
normal Vortex, eight threads, window 0, eight iterations, two warmups, and
two alternating rounds with separate perf processes. All six configurations
completed.

| Query | Engine | Paired latency | Instructions | Task-clock |
| --- | --- | ---: | ---: | ---: |
| clickbench_q01 | datafusion | -0.90% | -0.12% | -1.08% |
| clickbench_q01 | duckdb | -0.77% | -0.33% | +0.96% |
| clickbench_q23 | datafusion | -3.37% | -1.32% | -3.11% |
| clickbench_q23 | duckdb | -1.83% | +0.54% | +0.74% |
| tpch_q06 | datafusion | -0.26% | +0.02% | +1.39% |
| tpch_q06 | duckdb | +7.45% | -0.07% | +5.22% |

Instruction counts stay close to the measured candidate. DuckDB Q6 latency
changes +7.4% despite nearly unchanged instructions, with task-clock +5.2%;
this check is not evidence for a latency improvement. Preserve the original
focused results and their noise qualification. Raw study:
`target/v2-driver-perf/pipeline-guard-default-check/`.

## Validation before the cache-policy revision

The original PR drivers reproduce six failing V2 tests: four dictionary fixtures
depend on a compressor choosing dictionary encoding, and two pruning tests read
segments outside the zone-pruned mask. The final candidate uses an explicit
dictionary probe in those fixtures and prevents broad predicate evaluation from
overriding the initial scan/pruning selection. On the normal executor, supported
primitive and dictionary predicate plans use the initial mask as a care hint, returning dense pieces and
false values for skipped chunks. Other plans retain selected evaluation. The
experimental pipeline compacts source rows under a partial mask, so it retains
selected evaluation in that case rather than using the dense care-hint variant.
Expected result arrays and expected skipped-segment sets are unchanged.

| Zone-pruning regression case | Original data segments read | Final data segments read |
| --- | ---: | ---: |
| Static comparison excluding the first of four zones | 4 | 3 |
| Dynamic bound tightened to exclude the first two zones | 4 | 2 |

These are synthetic regression cases, establishing 25% and 50% reductions in
data-segment reads. They do not predict an equivalent SQL latency reduction.

Targeted checks use the same three-package feature set throughout:

```bash
cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib scan:: -j16 -- --test-threads=8
cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib perf::tests:: -j16
cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib file_prefetch::tests:: -j16
VORTEX_SCAN_V2=1 VORTEX_DUCKDB_FILE_PREFETCH=4 cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib e2e_test::vortex_scan_test:: -j16
cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib compressed_run_ends_preserve_offsets_and_nulls -j16
```

Passing default-executor cases: 132 layout scan tests, 5 DuckDB scan unit tests,
3 perf-control tests, 3 preparation-cancellation tests, 28 DuckDB SQL/array
integration tests with preparation enabled, and 8 compressed run-end
offset/validity cases (179 test executions). Final logs are
`target/v2-driver-perf/test-final-{scan,perf,prefetch,duckdb-v2,run-ends}.log`.
The existing run-end test module needed a missing trait import to compile.

A first implementation disabled dense evaluation whenever the initial mask was
partial. Three paired rounds found ClickBench Q1 9–11% slower, with about 19%
more instructions. It was replaced by dense predicate plans that respect the
read hint. The focused recheck's median paired Q1 changes were +0.3% for each
engine at the default preparation window, and -0.5% for DuckDB at window 4.
DataFusion instructions were unchanged; DuckDB counters were more variable.
See `query-counters/` for the rejected experiment and `q1-dense-check/` for
the corrected comparison.

The experimental pipeline executor passes all 72 non-list V2 cases, including
the new sparse dictionary selections. Its source stages compact masks rather
than treating them as dense read hints; the predicate planner now explicitly
uses selected evaluation for partial masks on that executor. This fixes three
mask-length failures exposed by the new fixtures. The five list cases remain
unsupported because `list_pack` has no pipeline implementation; the normal
executor passes them. The failing diagnostic and successful recheck are
`test-v2-pipeline-supported-final.log` and `test-v2-pipeline-supported-fixed.log`.
The existing pipeline list limitation was not expanded into a separate
implementation task. Including the 72 supported pipeline cases, 251 targeted
test executions pass. The pipeline command is:

```bash
VORTEX_SCAN_EXEC=pipeline cargo test --locked --profile release_debug \
  -p vortex-layout -p vortex-duckdb -p vortex-bench --lib scan::v2:: -j16 \
  -- --test-threads=8 --skip scan::v2::tests::lists::
```

CLI checks also pass: DataFusion rejects `--threads 0`, and DuckDB reports
missing prepared data without generating any files in an empty input directory.
Evidence: `target/v2-driver-perf/cli-checks.json`.

Rust formatting uses `nightly-2026-09-10`, scoped to changed files. Ruff checks
the comparison script and Linux profile-summary changes. Patch whitespace is
checked with `git diff --check`. Workspace-wide tests and linting are not run.

## Search findings and remaining targets

The following audit covers both driver entry points, preparation, expression
rewrites, split selection, scheduling, IO lifetimes, mask operations, dictionary
execution, temporal predicates, string codecs, and benchmark instrumentation.
Source findings are candidates for measurement, not established speedups.

| Area | Evidence | Status or next focused experiment |
| --- | --- | --- |
| Ordinary decoded arrays | V1 flat readers create fresh evaluation futures; the initial V2 decode cache retained ordinary arrays | Disabled in V2; dictionary/zone behavior remains available |
| Completed byte reads | V1 eagerly registers projection futures, which share pending raw reads with filters; V2 previously retained completed bytes until root clear | Track registered consumers, release after their last fetch, and match V1's FineWeb Q3 read volume; new repeated-read and concurrency regressions are unrun |
| Preparation reference counts | Q6 sample-weighted profile has about 12% combined atomic self samples; child walks include redundant clones | Borrowed visitors and lazy rewrite vectors implemented; original preparation study shows small instruction savings |
| Optimizer traversal | `optimize` cloned children twice and allocated a vector for every unchanged parent; parent-rule matching cloned before checking a rule | Borrowed child matching and lazy child vectors implemented; final comparison also includes the byte-read restriction, so its SQL delta cannot isolate this change |
| Remaining plan walks | `dense_predicate`, repeated-scan `reads_only_zones`, and pruning `zone_boundaries` still use owning child iteration; dense predicate rewriting allocates a child vector before knowing whether a child changes | Source candidates for the same borrowed/lazy pattern; measure them separately rather than attribute all remaining atomic samples to preparation |
| Dense rank scatter | ARM software deposit is a Q6 profile hotspot; dense rotating microbenchmarks improve 60–74% | Parallel expansion retained; sparse paths and x86 BMI2 dispatch retained; sampled SQL changes are small and mixed |
| Per-bit branch rewrite | Generated ARM assembly already uses branchless bit selection; paired microbenchmarks show no gain | Rejected |
| Numeric unpack/compare | Q6 has substantial `unfor_pack` and i32 comparison work; compressed FastLanes compare/between kernels already stream unpacking | Earlier generic numeric experiment rejected; target a specific bit width and encoding before changing the kernel |
| Compact string scans | FineWeb Q3 profile has 87–88% Zstd sequence-decoder self samples on busy workers; compact slowdown occurs in V1 too | Strong remaining target: measure frame/byte decode multiplicity and encoded predicate alternatives; no result or decode cache proposed |
| Selective Zstd frame work | `decompress_slice` decodes every frame overlapping a contiguous slice; Zstd has no specialized filter/LIKE kernel. FineWeb Q3 combines a URL predicate with a text predicate and projects all fields | Source candidate: frame-aware selection after the URL predicate, coalescing selected row ranges per frame and preserving null/order semantics; count decoded frames before claiming a gain |
| Temporal BETWEEN | Matching extension bounds delegate a single BETWEEN to storage through a metadata-only reduce rule, now reviewed in #10378 | Historical non-compact continuation: DuckDB Q6 instructions -21%, Q14 -16%, Q12 -7%. The extension regression cases pass in the standalone PR. DataFusion remains unmeasured for this rule |
| Singleton output assembly | `PackNode::assemble`, `ProjectionMorsel::finish`, and filter narrowing collect array vectors even when one piece is returned | Source candidate: bypass the second allocation for singleton pieces while retaining row-coverage and empty-array behavior |
| DataFusion partition preparation | V2 rebuilds expression plans and walks file chunk boundaries for each preparation; it accepts but ignores V1 natural splits | Source candidate: range-aware/lazy preparation. V1 natural boundaries can include artificial intra-chunk cuts, so reusing them as V2 chunk starts may add decoding |
| Driver IO waits | `poll_completion` builds a vector of IO roots for the common one-root case; queue priorities clone row-path vectors | Source candidates; require a scheduler-heavy profile before changing fairness or data structures |
| Filter prefetch | Remaining conjuncts are prefetched before the first predicate's selectivity is known; projection waits for surviving rows | Measure wasted prefetch bytes against IO latency and cancellation; local timings do not justify a blanket prefetch removal |
| V2 scan metrics | `ScanBuilder` accepts a metrics registry but V2 preparation/execution does not record scan metrics; external instrumented readers still report IO | Add preparation duration, split counts, input/output rows, requested bytes, and decode/frame counts in a separate measured instrumentation change |
| Cross-file preparation queue | Optional bounded queue, default 0; receiver-drop cancellation and disabled-path file-list work corrected | Keep disabled by default; measure preparation overlap and unused work on a high-latency object store before recommending a window |
| Profiler evidence quality | Linux recordings contain absolute sample timestamps and inflated idle-thread CPU deltas | Helpers handle absolute timestamps, use sample-weight ranking, and resolve Linux addresses; use separate query-only perf task-clock for CPU work |

## What to measure next

The driver changes target repeated preparation, scheduling, and skipped IO. The
Q6 profile shows that most steady-state work is elsewhere: decompression,
comparison, and filtering. Removing preparation overhead alone cannot promise
a large Q6 speedup. The removed numeric experiment also shows why an attractive
stack hypothesis must survive an isolated before/after comparison.

Use the comparison script's paired rounds together with query-only instruction
counts and task-clock. Record a same-binary control before interpreting a small
latency change. Separate cold registration, first-query cache fill, and warm
execution; the original DuckDB benchmark's automatic preparation could add tens
of seconds before a query. Measure IO requests and bytes independently of CPU
work, and use Parquet to check whether a thread-count change is engine-wide.

For the current non-compact DuckDB focus, measure FoR range evaluation without a
full primitive allocation, selective Q6 mask scattering, and row filtering. FoR
range handling must preserve signed wrapping, per-block references, patches,
and sliced offsets; simply subtracting a reference from a bound is insufficient.
Other source candidates are DataFusion Q23's registered-consumer overhead and
the remaining owning plan walks. Measure each proposed kernel change with its
encoding microbenchmark as well as the SQL query. Profile queue wait and preparation time
on a high-latency object store before recommending any nonzero queue default;
these local-file measurements do not establish a remote-store benefit.
