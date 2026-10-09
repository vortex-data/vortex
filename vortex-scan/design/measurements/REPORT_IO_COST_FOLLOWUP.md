<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Why the SSD coalescing cost prototype did not establish a win

The follow-up adds 91 cold query comparisons, two cold diagnostic-validation timings,
nine separate native diagnostics, 864 controlled read-probe batches, and three valid focused
V2 acceptance/Fetch captures. The cost policy still does not establish a latency improvement.
Small merges can help the SSD backend; that benefit does not automatically shorten this scan.
The previous IO-only implementation remains unchanged.

## Comparable baseline and variance

The primary control is the existing optimized V2 configuration: descriptor reuse, native
concurrency 32, demanded-only extent expansion, announcements enabled, and ready-Fetch disabled.
It uses the saved IO-only executable `f1d851df83356b7d38ef86d5f30e489b89adb03cd841dd574d57d4657091bc54`,
not untouched PR code. The archived cost executable is
`cee27145aea66c8cdb7d8ec2214de59611ac50b914d0378709a526777f6fa542`.

The first experiment uses two identical controls, the archived executable with its policy
disabled, and its 64 KiB/10% policy. Sixteen balanced rounds produce 64 accepted timings.
Every round starts after a fresh quiet window. All accepted outliers remain.

| Comparison | Baseline median ms | Candidate median ms | Candidate paired wins | Paired geometric-mean latency change |
|---|---:|---:|---:|---:|
| Identical control A → identical control B | 425.194 | 433.646 | 7/16 | +0.46% |
| Same archived executable, disabled → cost policy | 418.147 | 428.086 | 5/16 | +1.92% |

Identical controls differ by 1.99% in their configuration medians and by 27% in their first pair.
Their descriptive 95% paired-bootstrap interval spans -7.33% to +5.90% latency reduction.
The cost comparison's interval spans -6.64% to +3.78%. The previous small apparent gain is
within the variation measured without any implementation change.

One 549.630 ms control run used 3.439 system CPU-seconds versus a typical 1.127, and coincided
with 81.5 ms of host memory pressure. This is consistent with allocation-related stalls; these
host counters alone do not establish their cause. Other slow samples show no memory pressure.
The quiet gate detects competing work but cannot make the scan's IO, cancellation, and kernel
behavior deterministic.

## Testing the extra coalescer pass

Nine balanced rounds compare the same archived executable with its policy disabled, enabled
with a zero-use/zero-byte budget, and enabled at 10%: 27 accepted timings. The zero setting
rejects every speculative extension while retaining the additional coalescer pass and counters.

| Setting | Median ms | Paired wins vs disabled | Paired geometric-mean latency change |
|---|---:|---:|---:|
| Disabled | 436.009 | — | — |
| Zero speculation | 426.298 | 6/9 | -2.35% |
| 64 KiB / 10% | 435.656 | 5/9 | +1.91% |

This does not support blaming a consistent slowdown on the additional pass. It also does not
establish a zero-setting optimization. All 27 timing windows record zero selected reclaim and
compaction events; the cost setting still has a 530.389 ms outlier with zero memory-pressure
delta. Memory stalls are not a complete explanation of the spread.

## Service cost, throughput, and first-needed data

A C probe removes Python dispatch from the earlier service-fit experiment. It uses a persistent
native worker pool, fresh descriptors for each batch, positional reads with allocation, and 64
independent nonoverlapping pairs. File pages are evicted and verified completely nonresident
before every batch. Metadata/device caches remain warm. The probe uses glibc allocation and
does not reproduce Vortex's allocator, driver, or future demand.

The symmetric sweep produces 648 batches: three payload sizes, four gaps, reader concurrency
1/8/32, merged/serial/parallel modes, and six balanced repetitions. At 32 readers, merging has
a faster whole-batch median in 6/12 cases, but a later median first-needed payload in all twelve.

| Each payload / gap / readers | Merged batch ms | Parallel batch ms | Merged first payload ms | Parallel first payload ms | Merged / parallel storage MB |
|---|---:|---:|---:|---:|---:|
| 64 KiB / 16 KiB / 32 | 2.263 | 1.015 | 1.428 | 0.568 | 22.3 / 8.4 |
| 1 MiB / 16 KiB / 32 | 17.410 | 19.389 | 12.920 | 11.291 | 144.1 / 143.8 |

The first case is slower and transfers much more physical data after enlarging a read. The
second has better batch throughput but delays the first required payload. A single fixed
overhead/bandwidth fit does not predict both outcomes. Queue occupancy and request-size-dependent
kernel traffic must enter a latency model.

An asymmetric sweep uses a 16 KiB second payload, first payloads of 16/64/1024 KiB, gaps of
0/4/16/32 KiB, and 32 readers: another 216 batches. The existing 10% predicate accepts six
of these cases. Five deliver the first payload earlier when merged, and four complete the batch
earlier. For two adjacent 16 KiB payloads, batch medians are 0.435 ms merged and 0.517 ms
parallel. Thus selective small coalescing has a measured backend benefit. These probes require
both payloads; an announced payload in a scan may never be needed.

## Measuring actual speculative use

The initial rich trace could not identify exactly which members the cost rule accepted.
Undemanded members include free sharing, alignment prefixes, and reads whose demand is not
visible in planner Fetch bindings. Those inferred cohorts are archived but are not used as
cost-policy reuse estimates.

Temporary instrumentation therefore records each exact accepted member, charged bytes, the
first V2 `wanted` transition, and V2 Fetch submissions. A first attempt instrumented the legacy
read future instead; its zero-use result was invalid because V2 uses `FileScanIo`. Those three
captures remain archived and are excluded. The corrected recorder requires nonzero coverage
of the active path and verifies acceptance counts and bytes against the independent counters.

Three valid focused captures record 2,727 accepted extensions. Their counts and charged byte
totals match exactly in every capture. They observe 23,779–26,700 unique fetched registrations.
They are diagnostics, with query times 525–560 ms, rather than performance samples.

| Capture | Accepted members | Later fetched | Charged extra MB associated with later Fetch | Charged extra MB without a Fetch by query end |
|---|---:|---:|---:|---:|
| 0 | 992 | 500 | 5.286 | 8.709 |
| 1 | 830 | 418 | 4.986 | 8.546 |
| 2 | 905 | 459 | 5.579 | 8.525 |

Across these schedules, 50.50% of accepted members are later fetched, accounting for only
38.07% of the charged extra bytes. The remaining 61.93% is associated with members not fetched
by query end. Charged extent is a coalescer decision, not a guarantee that every byte reaches
storage: cancellation and unfinished reads prevent that interpretation.

Physical completion ownership is known for 2,160 members. For 489, the read completed before
the first Fetch; 507 are fetched while the read is incomplete, 21 are already fetched when
accepted, and 1,143 are not fetched by query end. The remaining 567 have unmatched physical
identities: the acceptance event uses the seed request ID, whereas physical events use the first
member after spatial sorting. They are marked unknown rather than counted as cancelled reads.
The 489 early completions are a lower bound, not proof of time saved on the critical path.
There are also 121 acceptances after the underlying registration is already marked wanted;
the coalescer has not yet applied its `Polled` event. Its optional classification can lag demand.

Heavy tracing changes scheduling. These are exact observations of the captured schedules,
not estimates of untraced production probabilities or proof that an early completion advances
the next compute stage.

## Interpretation and retained changes

The policy is small: earlier untraced validation charges about 14 MB against roughly 3.8 GB
of application reads, without a consistent reduction in physical request counts. The focused
captures show that much of that speculative budget is not consumed. The backend probe shows
that read setup savings can improve throughput while delaying the first needed data. Together
with identical-control variance, these findings explain why a small median difference did not
establish a worthwhile scan improvement. They do not disprove selective coalescing.

A useful rule needs conditional future demand, native queue state, the marginal physical cost
of enlarging the current extent, and whether the future read would actually block compute.
Its objective should compare expected future wait avoided with current demanded-data delay
and added IO work. The original fixed-cost equation is insufficient for that objective.

The production IO source was restored byte-for-byte. Retained changes are benchmark diagnostics:
separate user/system CPU, faults, context switches, reclaim/compaction/THP counters, and cumulative
PSI deltas taken around the process after eviction. These are host/process windows, not query-only
or per-read measurements. Missing/reset counters remain unavailable rather than reported as zero.

The benchmark also fixes a measurement bug: DataFusion's default aggregation summed per-file
histogram minima, maxima, and percentiles. IO diagnostics now combine extrema with min/max and
sum counts. They omit per-file p95/p99 values from scan-wide output; native sample logs provide
actual combined distributions. A cold validation reports a maximum read duration of 172.191 ms
for the corrected executable, versus a summed 6458.043 ms value in the saved control. One timing
pair validates the output path and is not a performance claim.

## Artifacts and checks

All commands, raw samples, immutable executables, sources, rejected inference attempts, and
analyses are under `/mnt/vortex-ssd/votex-4/results/cost-explain-20261008T144238Z/`.
`null/paired.json`, `shadow/paired.json`, `parallel.json`, `asymmetric.json`, and
`marked-v2-analysis.json` contain the comparisons above. All query/probe data, native executables,
temporary files, and outputs use instance SSD `nvme0n1`. Cold page residency is zero; compute
kernels and projection/split scheduling remain unchanged.

The restored release build, strict benchmark/file clippy, pinned formatting, 48 Python tests,
the Rust aggregation regression test, Ruff, ty, and patch whitespace checks passed. Runtime tests
ran from SSD with SSD temporary files and native test executables. An intermediate V2 trace-only
variant exceeded the existing submit function's clippy complexity limit by one branch; it is
archived and was removed from the production source. The policy was not promoted or rerun across
all 65 queries. Result row counts match; query result values are not checksummed.
