<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Cold SSD IO follow-up

The source in this report predates the user's IO-only scope correction. Its frozen executable
still contains automatic sparse projection grouping; all configurations below share that same
compute policy, so their within-run IO comparisons remain valid. Grouping, compressed LIKE and
split-task scheduling experiments have since been moved out of the working patch. Fresh
measurements of the resulting IO-only source are recorded in [REPORT_IO_ONLY.md](REPORT_IO_ONLY.md).

The new Rust coalescer change lowers cold Q23 medians **20.67% at fixed handle reuse and
native concurrency 32**, winning all eight paired rounds. Combining it with handle reuse and
native32 lowers geometric-mean latency **22.98% across all 65 queries**, with 56 faster medians.
Eight-round confirmation versus current V2 improves Q23 by 19.09%, TPC-H Q19 by 55.03%, Q24 by
71.36% and ClickBench Q19 by 5.27%; each wins 8/8 pairs. These are distinct comparisons: most of
the TPC-H Q19 gain comes from handle reuse, not the new coalescing change.

This continues the [first SSD rerun](REPORT_IO_SSD.md). Optional registered segments can share
a demanded read, but cannot enlarge its physical extent when
`VORTEX_SCAN_IO_COALESCE_OPTIONAL=0`. Alignment-prefix bytes count as already covered. Unread
interests remain registered for later demand or cancellation. Four new cases cover alignment,
covered optional members and later demand; shared scopes are not unregistered just because one
branch no longer needs them. The policy remains opt-in, together with local handle reuse.

All runs use the same local instance SSD (`nvme0n1`), with executable, data, working directory,
temporary files and output on that device. Every timing uses one query in a fresh process,
and page residency after eviction is zero. Metadata and device caches remain warm. Quiet
gating, whole-round rejection and retained raw samples follow the first SSD report.

## Five-round coalescing screen

Five configurations visit every order position once over five rounds per query: 100 accepted
cold timing samples. The native limit is process-wide and only applies to file payloads.
The optimized setting combines handle reuse, native concurrency 32 and demanded-range
coalescing; early announcements remain enabled. All changes remain opt-in during screening.

| Query | Current median ms | Optimized median ms | Faster | Paired wins | App MB current → optimized | Whole-process storage MB current → optimized | Reads current → optimized |
|---|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 | 533.254 | 431.180 | 19.14% | 4/5 | 4239.220 → 3838.288 | 4876.100 → 4033.556 | 364 → 3132 |
| tpch-q19 | 160.185 | 68.842 | 57.02% | 5/5 | 507.162 → 527.494 | 543.879 → 527.004 | 182 → 554 |
| clickbench-q24 | 79.846 | 35.118 | 56.02% | 5/5 | 218.513 → 192.029 | 285.938 → 264.131 | 364 → 527 |
| clickbench-q19 | 35.972 | 34.058 | 5.32% | 5/5 | 259.165 → 246.216 | 300.474 → 274.936 | 313 → 435 |

Q23 whole-process CPU work rises from 6.079 to 7.102 seconds while query wall time falls
19.14%. Request count rises 8.6 times as ranges become smaller; caching the local descriptor
prevents this from multiplying GET/open/setup work. Mean device queue depth over the entire
process window falls from 156.20 to 32.03. These are kernel block-device counters, not driver
queue depth. The CPU and device windows include initialization and background completion.

Demanded-range coalescing without handle reuse regresses every target: Q23 loses 40.25%,
TPC-H Q19 54.55%, Q24 40.24% and ClickBench Q19 168.69%. Avoiding speculative extent is not
sufficient when each extra small request repeats GET/setup. The mode must be evaluated with
the backend, admission limit, byte coverage and read count together.

Application and storage bytes describe different populations. Application counters stop at
the query snapshot; child storage accounting includes the whole process. Their difference
cannot identify readahead or cancellation bytes. Result cardinality matches across all
accepted configurations and rounds; this is not a result-value checksum.

### Separate native read phases

Per-read medians below come from one separate cold diagnostic process per setting/query.
Times are microseconds. Admission waits overlap across requests and cannot be added to query
wall time; pread distributions change with request size, concurrency and logging.

| Query | GET current → optimized µs | Native admission optimized µs | pread current → optimized µs |
|---|---:|---:|---:|
| clickbench-q23 | 65.370 → 0.089 | 82050.550 | 33153.026 → 2078.804 |
| tpch-q19 | 2368.645 → 0.130 | 9850.940 | 11243.144 → 1691.907 |
| clickbench-q24 | 78.163 → 0.133 | 1939.292 | 975.484 → 905.810 |
| clickbench-q19 | 17.971 → 0.143 | 884.633 | 700.201 → 539.326 |

## Handle reuse, announcement and readahead screen

A preceding ten-configuration, two-round screen contains 80 accepted cold timings and 40
separate light diagnostics. It tests reuse with and without native admission, announcement
deferral, a four-slot per-file cap and a Linux per-descriptor random-access hint. The hint
uses [POSIX_FADV_RANDOM](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html), which
disables readahead on that open descriptor without changing the device-wide setting.

The hint loses 34.75% on TPC-H Q19 versus handle reuse alone (71.201 → 95.943 ms), and does
not establish a Q23 improvement (557.119 → 572.019 ms). It is excluded from the full-suite
candidate. This is a rejected performance experiment, not a recommendation.

## Full 65-query cold screen

All 22 TPC-H queries at SF10 and all 43 zero-indexed ClickBench queries complete three cold
rounds each: 585 accepted timings. Current V2, handle reuse alone and
reuse/native32/demanded coalescing each visit every order position once. One entire contended
round is rejected and retained; its replacement contributes the accepted results.

The combined setting has **22.98% lower geometric-mean query latency**
(1.298× speedup), improving 56/65 medians. The sum of per-query
medians falls from 14.750 to 11.724 seconds;
this is a derived equal-workload total, not a measured continuous execution. Handle reuse alone
improves 47/65 medians with 19.57% lower geometric-mean latency. Adding demanded coalescing and
native32 together improves the geometric mean a further
4.24% versus unlimited handle reuse.
That comparison changes admission and coalescing together, so it does not isolate either effect.

TPC-H geometric-mean latency falls 28.04% (17/22 faster), and ClickBench falls 20.26% (39/43 faster).
Nine medians regress; all are below 10%, with the largest absolute loss 5.43 ms. TPC-H Q2 is
52.579 → 55.236 ms (5.05% slower), and ClickBench Q2 is 21.839 → 23.328 ms (6.82% slower).
Three rounds are a broad screen, not enough to establish each small per-query effect.
Every accepted configuration has matching result row counts; result values are not checksummed.
The complete table is below.

## Native admission and matched confirmation

The native8/16/32/64/128 sweep holds handle reuse and demanded coalescing fixed, with five
rounds per query and every setting visiting every order position once: 100 accepted timings
and 20 separate diagnostics, with no rejected rounds.

| Query | native8 ms | native16 ms | native32 ms | native64 ms | native128 ms |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 | 518.087 | 426.808 | 454.076 | 432.271 | 463.054 |
| tpch-q19 | 82.558 | 68.055 | 68.883 | 69.875 | 70.283 |
| clickbench-q24 | 42.296 | 35.921 | 36.404 | 34.881 | 35.693 |
| clickbench-q19 | 40.320 | 33.858 | 34.211 | 34.216 | 34.641 |

Native8 loses 14.10–19.85% versus native32 across these four targets. Native16 improves the
four-query geometric mean by 2.41%, including 6.01% on Q23, but wins only 3/5 Q23 pairs.
Native64 improves Q24 by 4.18% (5/5 pairs) while slightly regressing TPC-H Q19 and ClickBench
Q19. This sweep does not establish one best limit for every query. Native32 remains the
setting tested across all 65 queries; native16 is an optional tuning lead.

On Q23, mean kernel block-device queue depth grows from 7.86 at native8 to 14.77, 32.49,
67.24 and 141.75 at native16/32/64/128. These whole-process device counts include split kernel
requests and are not the number of live native permits. Native8 saves some application bytes
but starves throughput; native128 expands traffic and does not improve its median.

### Eight-round confirmation

Four configurations visit each order position twice: 128 accepted timings and 16 separate
native-phase diagnostics, with no rejected rounds. “Reuse/native32” retains optional extent
growth, so its comparison with the combined setting isolates the new coalescing policy.
“Announcements off” is the prior Q23 candidate, without handle reuse and with native32.

| Query | Current ms | Reuse/native32 ms | Combined ms | Announcements off ms | Combined faster vs current | Wins vs current | Coalescer gain at fixed reuse/native32 | Coalescer paired wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 | 519.087 | 529.426 | 419.983 | 449.432 | 19.09% | 8/8 | 20.67% | 8/8 |
| tpch-q19 | 154.781 | 70.372 | 69.599 | 134.423 | 55.03% | 8/8 | 1.10% | 4/8 |
| clickbench-q24 | 121.702 | 38.666 | 34.851 | 113.195 | 71.36% | 8/8 | 9.87% | 8/8 |
| clickbench-q19 | 36.055 | 35.469 | 34.155 | 36.636 | 5.27% | 8/8 | 3.70% | 8/8 |

The coalescer gain is established on Q23 and Q24 in this run. TPC-H Q19 improves only 1.10%
with 4/8 paired wins; this does not establish an additional gain there. ClickBench Q19 saves
1.31 ms at fixed reuse/native32, despite its consistent pairs. Against the earlier
announcements-off Q23 setting, the combined setting improves the median by 6.55% with 5/8
paired wins. It keeps early announcements enabled.

All accepted samples, including outliers, remain in the medians. Timing ranges:

| Query | Current min–max ms | Reuse/native32 min–max ms | Combined min–max ms | Announcements off min–max ms |
|---|---:|---:|---:|---:|
| clickbench-q23 | 504.021–683.326 | 513.959–583.325 | 385.696–521.014 | 422.460–569.465 |
| tpch-q19 | 127.259–163.264 | 68.956–82.459 | 68.041–88.726 | 83.958–195.064 |
| clickbench-q24 | 96.749–153.682 | 36.239–39.312 | 34.222–35.491 | 71.474–126.986 |
| clickbench-q19 | 35.849–36.835 | 35.140–36.159 | 33.858–34.570 | 35.795–136.059 |

Detailed driver captures follow these timings and are reported separately below.

### Run the measured combined setting

```bash
VORTEX_SCAN_V2=1 \
VORTEX_LOCAL_FILE_HANDLE_REUSE=1 \
VORTEX_LOCAL_READ_CONCURRENCY=32 \
VORTEX_SCAN_IO_COALESCE_OPTIONAL=0 \
VORTEX_SCAN_IO_READY_FETCH=0 \
/mnt/vortex-ssd/votex-4/bin/datafusion-demanded <benchmark arguments>
```

The harness variant is `v2-io-file-handle-demanded-native-32`. A fresh process is required when
changing these settings because they initialize once per process. This recommendation covers
immutable local SSD data in the measured DataFusion/Vortex workloads; remote-store latency,
warm page-cache scans and other execution engines were not tested by this follow-up. No
production defaults changed.

### Traffic in the confirmed timing samples

These counters come from the eight untraced timing processes, not the later diagnostic runs.
Reads and bytes are medians at the query snapshot; storage and CPU cover the whole process.
A half-integer read count is the median of eight integer counts.

| Query | Reuse/native32 app MB | Combined app MB | Reuse/native32 storage MB | Combined storage MB | Reuse/native32 reads | Combined reads | Reuse/native32 CPU s | Combined CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 | 4963.979 | 3687.157 | 4958.050 | 3825.187 | 437 | 3009 | 7.300 | 6.710 |
| tpch-q19 | 507.162 | 526.300 | 523.616 | 525.441 | 182 | 540 | 0.634 | 0.643 |
| clickbench-q24 | 223.675 | 190.725 | 278.647 | 262.273 | 371 | 534.5 | 0.705 | 0.692 |
| clickbench-q19 | 257.337 | 245.711 | 289.892 | 275.087 | 315 | 439.5 | 0.284 | 0.294 |

At fixed reuse/native32, Q23 application bytes fall 25.72%, whole-process storage bytes fall
22.85%, and CPU work falls 8.08%, despite 6.89 times as many smaller reads. Compared instead
with current unlimited V2, combined Q23 CPU work rises 13.19%. These comparisons should not be
mixed. TPC-H Q19 reads more bytes in the combined setting and has no established additional
wall-time gain over reuse/native32.

## Separate driver and compute-wait captures

Four fresh cold processes capture Q23 and TPC-H Q19 with reuse/native32 held fixed, optional
extent growth on and off. Exact scope/drop tracing is enabled; no competing process is observed
in these captures. Detailed tracing changes the schedule: Q23 wall time rises from 2.671 to
2.950 seconds under the new policy, reversing the untraced 20.67% improvement. Startup and
queue comparisons below describe the traced schedule and are not estimates of production
critical-path savings. Native-phase distributions also use separate diagnostic processes.

“Optional” permits registered segments to expand a read; “demanded” only permits demanded
requests to expand it. First compute includes planner compute. First morsel and first batch
are distinct milestones.

| Query / policy | Traced wall ms | First compute ms | First morsel ms | First returned batch ms | First matched file completion ms | DFS max depth |
|---|---:|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 2670.577 | 33.429 | 168.376 | 256.991 | 145.449 | 3 |
| clickbench-q23 / demanded | 2949.580 | 28.473 | 220.466 | 399.609 | 31.553 | 3 |
| tpch-q19 / optional | 190.724 | 7.219 | 11.710 | 13.130 | 9.432 | 4 |
| tpch-q19 / demanded | 192.681 | 7.219 | 11.035 | 12.358 | 9.208 | 4 |

| Query / policy | Driver pending physical reads mean / peak | Native pread mean / peak | Native admission mean / peak | Async resume mean / peak | Ready morsels mean | Parked morsels mean |
|---|---:|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 20.73 / 63 | 1.11 / 16 | 0.00 / 1 | 8.99 / 32 | 10.37 | 56.03 |
| clickbench-q23 / demanded | 112.59 / 269 | 1.38 / 32 | 11.43 / 138 | 42.75 / 140 | 10.36 | 86.55 |
| tpch-q19 / optional | 23.08 / 124 | 7.66 / 32 | 7.74 / 92 | 0.07 / 5 | 1.12 | 145.64 |
| tpch-q19 / demanded | 92.22 / 192 | 5.28 / 32 | 10.47 / 156 | 3.39 / 34 | 0.83 | 121.02 |

Driver “physical in flight” includes requests waiting before and after native pread. Completed
local phases only cover reads with bound completion logs; unfinished reads are censored.
Native permits remain held until blocking work finishes, including after async cancellation.
Neither driver depth nor block-device depth is the number of native permits.

| Query / policy | IO with no ready delivery or compute ms | Runnable without compute ms | File completion → Fetch delivery p95 ms | Wake → session poll p95 ms | Unpark → compute p95 ms |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 97.600 | 1530.359 | 453.961 | 445.659 | 13.085 |
| clickbench-q23 / demanded | 19.389 | 1643.452 | 319.147 | 292.533 | 14.993 |
| tpch-q19 / optional | 17.832 | 68.276 | 36.910 | 35.090 | 0.361 |
| tpch-q19 / demanded | 3.323 | 75.055 | 30.929 | 28.853 | 0.364 |

The query-state categories partition one query wall window. The separate gap metrics overlap,
and summed owner park times overlap across splits, so neither can be added to query latency.
Both Q23 traces spend most time with runnable work outside observed `compute()` calls. That
state includes trace emission, driver bookkeeping and scheduling; it does not identify one CPU
hotspot. First-stage times, stage compute distributions, ready/parked queues, DFS ancestry,
physical byte depth, local phases, output progress and release waits are retained in the full
JSON reports; a small selection is in `drivers/compact.json`.

The coverage counters expose a tradeoff. In the Q23 trace, bytes started without a matched
Fetch in the window fall from 1077.35 to 546.57 MB, while summed coalescing gap bytes rise from
125.91 to 351.04 MB. TPC-H Q19 gap bytes rise from 0.075 to 48.087 MB, with 48.01 MB of repeated
range coverage. These are application-range diagnostics, not kernel storage bytes or a proof
of saved prefetch bytes. The new policy can lose early sharing and reread covered gaps later.
Q23's demanded trace ends with 25 unfinished reads (72.18 MB); these remain explicit.

Q23 optional-interest withdrawals are mostly shared: 925/1134 in the optional policy and
1078/1244 in the demanded policy still have other references at withdrawal. Last-reference
withdrawals without a matched started read grow from 0.035 to 16.412 logical MB, but other
ranges can still cover those bytes. This supports retaining reference-aware interests and
avoiding optional extent growth; deleting a segment for one pruned scope would be incorrect.

## Measurement audit

This follow-up completes **993 accepted cold timings**, 291 separate native-phase diagnostics
and four detailed driver captures. One full-suite round (three timings) is rejected for
observed competing work, retained and repeated. All other completed runs have zero rejected
rounds. The initial stopped `combine-screen` produces no timing samples and is excluded.
No accepted outliers are removed. Every timed process verifies zero file-page residency after
eviction and reads data on `nvme0n1`; executable, temporary files, working directory and
results also reside on that instance SSD. Compilation artifacts stay in the existing Cargo
target directory and are not benchmark data. No build runs alongside these measurements.

## Source and checks

Frozen demanded-range executable SHA-256:
`4ae75e3c0417955f5ba80a7484e215c64b075e9e32c8e8ed1f684afeba730650`.
The earlier hint experiment executable is
`81faa236561815c8a9c94a760ea7a69997687daed2009a55c924ba3b4c3a2830`.
Both execute from SSD. Exact commands, harness/source snapshots, phase logs, all samples,
rejected rounds, noise observations and the final source snapshot/patch are saved under
`/mnt/vortex-ssd/votex-4/results/cold-loop-20261008/`. Derived selections are in each run's
`analysis.json`; `analyze.py` verifies SSD residency and cardinality and retains all accepted
outliers. `audit.json` checks every accepted timing and separate diagnostic, and `COMPLETED.json`
records the completed runs. The native phase summary keys end in `_ns`, but their values in `read_phases_median_us`
are microseconds.

All 197 IO tests pass with the readahead hint both off and on. All 214 file tests pass after
the coalescing change; all 14 scan IO tests also pass with optional extent growth disabled.
All 46 Python analyzer/harness tests pass and pass again from the final source snapshot on
SSD; the 12 harness tests also pass after adding the new controls. These cover 457 distinct
targeted Rust/Python cases. Final Ruff, ty and patch whitespace checks pass, as does
pinned-nightly formatting. Strict IO clippy passes.
Strict file clippy is blocked by six previously recorded layout dependency errors; the changed
file crate passes strict clippy with `--no-deps`. No workspace-wide check was run.

## Per-query cold medians

Three rounds per setting; all values are milliseconds. “Reuse” uses unlimited native reads.
“Combined” uses reuse, native32 and demanded-range coalescing. Negative percentages are regressions.

| Query | Current | Reuse | Combined | Combined faster | Paired wins |
|---|---:|---:|---:|---:|---:|
| tpch-q1 | 244.976 | 234.975 | 205.504 | 16.11% | 3/3 |
| tpch-q2 | 52.579 | 53.811 | 55.236 | -5.05% | 0/3 |
| tpch-q3 | 205.446 | 118.705 | 101.030 | 50.82% | 3/3 |
| tpch-q4 | 63.916 | 66.221 | 59.319 | 7.19% | 3/3 |
| tpch-q5 | 271.513 | 196.456 | 183.238 | 32.51% | 3/3 |
| tpch-q6 | 141.545 | 52.079 | 50.329 | 64.44% | 3/3 |
| tpch-q7 | 357.136 | 268.174 | 247.853 | 30.60% | 3/3 |
| tpch-q8 | 284.468 | 210.066 | 187.326 | 34.15% | 3/3 |
| tpch-q9 | 392.196 | 351.726 | 326.953 | 16.64% | 3/3 |
| tpch-q10 | 240.044 | 171.450 | 147.838 | 38.41% | 3/3 |
| tpch-q11 | 40.051 | 40.995 | 41.261 | -3.02% | 1/3 |
| tpch-q12 | 68.709 | 70.675 | 70.249 | -2.24% | 1/3 |
| tpch-q13 | 94.888 | 98.347 | 95.365 | -0.50% | 1/3 |
| tpch-q14 | 121.876 | 65.616 | 65.403 | 46.34% | 3/3 |
| tpch-q15 | 192.507 | 84.201 | 83.144 | 56.81% | 3/3 |
| tpch-q16 | 42.335 | 42.249 | 42.412 | -0.18% | 1/3 |
| tpch-q17 | 369.606 | 310.787 | 292.037 | 20.99% | 3/3 |
| tpch-q18 | 488.366 | 421.249 | 421.102 | 13.77% | 3/3 |
| tpch-q19 | 184.965 | 72.568 | 69.475 | 62.44% | 3/3 |
| tpch-q20 | 207.351 | 123.188 | 110.947 | 46.49% | 3/3 |
| tpch-q21 | 310.585 | 315.184 | 292.330 | 5.88% | 3/3 |
| tpch-q22 | 30.115 | 30.724 | 30.108 | 0.03% | 1/3 |
| clickbench-q0 | 6.986 | 6.767 | 6.551 | 6.24% | 3/3 |
| clickbench-q1 | 11.425 | 11.466 | 11.321 | 0.91% | 2/3 |
| clickbench-q2 | 21.839 | 22.691 | 23.328 | -6.82% | 0/3 |
| clickbench-q3 | 99.570 | 43.250 | 42.842 | 56.97% | 3/3 |
| clickbench-q4 | 149.957 | 137.216 | 131.243 | 12.48% | 3/3 |
| clickbench-q5 | 165.511 | 148.659 | 141.540 | 14.48% | 3/3 |
| clickbench-q6 | 6.854 | 6.881 | 6.795 | 0.86% | 2/3 |
| clickbench-q7 | 13.224 | 12.860 | 12.621 | 4.56% | 3/3 |
| clickbench-q8 | 183.124 | 178.396 | 172.612 | 5.74% | 3/3 |
| clickbench-q9 | 256.175 | 228.911 | 231.421 | 9.66% | 3/3 |
| clickbench-q10 | 122.997 | 50.337 | 49.878 | 59.45% | 3/3 |
| clickbench-q11 | 53.103 | 53.840 | 51.715 | 2.61% | 2/3 |
| clickbench-q12 | 161.462 | 139.157 | 133.527 | 17.30% | 3/3 |
| clickbench-q13 | 228.706 | 220.243 | 225.700 | 1.31% | 2/3 |
| clickbench-q14 | 182.317 | 141.250 | 138.106 | 24.25% | 3/3 |
| clickbench-q15 | 167.413 | 157.425 | 156.479 | 6.53% | 3/3 |
| clickbench-q16 | 371.238 | 339.878 | 335.926 | 9.51% | 3/3 |
| clickbench-q17 | 370.646 | 331.484 | 333.720 | 9.96% | 3/3 |
| clickbench-q18 | 718.592 | 721.740 | 690.502 | 3.91% | 3/3 |
| clickbench-q19 | 36.164 | 35.393 | 34.150 | 5.57% | 3/3 |
| clickbench-q20 | 566.979 | 325.705 | 297.536 | 47.52% | 3/3 |
| clickbench-q21 | 463.124 | 366.884 | 312.841 | 32.45% | 3/3 |
| clickbench-q22 | 634.367 | 539.698 | 474.887 | 25.14% | 3/3 |
| clickbench-q23 | 515.184 | 539.430 | 415.176 | 19.41% | 3/3 |
| clickbench-q24 | 132.063 | 36.682 | 35.651 | 73.00% | 3/3 |
| clickbench-q25 | 109.108 | 48.213 | 46.007 | 57.83% | 3/3 |
| clickbench-q26 | 107.680 | 38.191 | 36.484 | 66.12% | 3/3 |
| clickbench-q27 | 425.520 | 321.600 | 304.369 | 28.47% | 3/3 |
| clickbench-q28 | 957.769 | 938.470 | 932.654 | 2.62% | 3/3 |
| clickbench-q29 | 34.564 | 34.511 | 35.154 | -1.71% | 0/3 |
| clickbench-q30 | 204.701 | 140.385 | 127.439 | 37.74% | 3/3 |
| clickbench-q31 | 309.930 | 190.920 | 175.074 | 43.51% | 3/3 |
| clickbench-q32 | 613.261 | 643.492 | 618.691 | -0.89% | 2/3 |
| clickbench-q33 | 821.483 | 748.398 | 740.744 | 9.83% | 3/3 |
| clickbench-q34 | 783.552 | 742.692 | 739.886 | 5.57% | 3/3 |
| clickbench-q35 | 146.296 | 135.930 | 134.187 | 8.28% | 3/3 |
| clickbench-q36 | 38.020 | 38.362 | 37.987 | 0.09% | 3/3 |
| clickbench-q37 | 22.418 | 22.467 | 22.943 | -2.34% | 1/3 |
| clickbench-q38 | 16.555 | 16.336 | 15.191 | 8.24% | 2/3 |
| clickbench-q39 | 73.179 | 73.204 | 72.782 | 0.54% | 1/3 |
| clickbench-q40 | 13.744 | 13.537 | 13.174 | 4.15% | 3/3 |
| clickbench-q41 | 13.256 | 13.227 | 12.524 | 5.52% | 3/3 |
| clickbench-q42 | 14.431 | 14.562 | 14.362 | 0.47% | 2/3 |
