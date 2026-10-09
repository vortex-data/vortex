<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# IO-only cold scan measurements

These timings use the frozen executable and source recorded below, before integration with
PR #10339's newer commits at `e645cd9a97ccf55f6b9fc1ef2013b53ff8780389`. The integration retains
the upstream consumer-release and announcement-count changes, which can change IO behavior.
The figures are historical measurements, not a performance result for the merged source.
See [the integration report](REPORT_IO_INTEGRATION.md) for subsequent validation.

The combined IO policy lowers geometric-mean latency **22.95% across all 65 queries**, with
59 faster medians and six regressions no larger than 3.07%. Eight-round cold confirmation
improves Q23 **19.84%**, TPC-H Q19 **54.14%**, Q24 **63.03%** and ClickBench Q19 **5.94%**, each
winning all eight paired rounds. At fixed handle reuse/native32, the coalescer alone improves
Q23 **18.16%**, also with 8/8 paired wins. All comparisons keep compute unchanged.

The working patch changes IO handling and its diagnostics. Compressed LIKE, sparse projection
grouping and split-task scheduling experiments are removed and saved separately in
`/mnt/vortex-ssd/votex-4/results/io-only-20261008/compute-experiments.patch`. Projection morsels
retain the original cut and batch boundaries; encoding kernels and split scheduling use HEAD.
Driver tracing measures computation only to distinguish IO starvation from delivery and
scheduling waits. Reference-aware IO withdrawal remains independent of projection grouping.

The earlier [cold loop](REPORT_IO_COLD_LOOP.md) used a different fixed compute policy. Its
within-run IO comparisons remain historical evidence; fresh timings below use the IO-only
source and do not mix binaries or measurement windows.

All data, benchmark/test executables, working directories, temporary files and results use
instance SSD `nvme0n1`. Cold means file pages evicted and verified nonresident before one
iteration in a fresh process. Metadata and device caches remain warm. Six quiet observations
span ten seconds; entire rounds with observed competing benchmarks/builds are retained and
repeated. Timing processes disable detailed event logging; diagnostics follow them.

## Eight-round cold confirmation

Eight rounds balance four configurations twice: 128 accepted timing samples and 16 separate
native-phase diagnostics, with no rejected rounds. Current V2 uses unlimited native reads and
no descriptor reuse. “Reuse/native32” retains optional extent growth. “Combined” uses reuse,
native32 and demanded-range coalescing; comparison with reuse/native32 isolates coalescing.
“Announcements off” retains no reuse and uses native32. All settings leave compute unchanged.

| Query | Current ms | Reuse/native32 ms | Combined ms | Announcements off ms | Combined faster vs current | Paired wins | Coalescer alone faster | Coalescer paired wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 | 542.873 | 531.759 | 435.186 | 468.880 | 19.84% | 8/8 | 18.16% | 8/8 |
| tpch-q19 | 150.655 | 71.193 | 69.085 | 143.436 | 54.14% | 8/8 | 2.96% | 6/8 |
| clickbench-q24 | 97.779 | 37.683 | 36.145 | 113.352 | 63.03% | 8/8 | 4.08% | 6/8 |
| clickbench-q19 | 36.518 | 35.710 | 34.349 | 35.888 | 5.94% | 8/8 | 3.81% | 8/8 |

### Attribution of the confirmed reductions

These runs isolate two sequential steps: enabling handle reuse together with native32, then
enabling demanded-range coalescing. They do not separate handle reuse from the native-read
limit. The table below subtracts the configuration medians; percentages use the original
control median throughout, so the two savings columns add to the total reduction. This is
an ordered comparison, not a unique decomposition of interacting changes.

| Query | Handle reuse + native32 saved | Coalescing additionally saved | Total reduction | Reuse/native32 paired wins | Coalescing paired wins |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 | 11.114 ms (2.05%) | 96.573 ms (17.79%) | 19.84% | 7/8 | 8/8 |
| tpch-q19 | 79.462 ms (52.74%) | 2.108 ms (1.40%) | 54.14% | 8/8 | 6/8 |
| clickbench-q24 | 60.096 ms (61.46%) | 1.538 ms (1.57%) | 63.03% | 8/8 | 6/8 |
| clickbench-q19 | 0.808 ms (2.21%) | 1.361 ms (3.73%) | 5.94% | 7/8 | 8/8 |

The earlier table's 18.16%, 2.96%, 4.08% and 3.81% coalescer reductions use the intermediate
reuse/native32 median as their denominator. They cannot be added directly to reductions
relative to the original control. Both tables describe the same measurements.

Reference-aware withdrawal, inline Fetch, streaming read completions, coalescer bookkeeping
and disabled-trace guards are present in every configuration. Their individual contributions
are not measured by this confirmation. Consequently these numbers do not measure all local
changes against the untouched PR. Earlier handle-only runs support the interpretation that
reuse supplies most of the large TPC-H Q19/Q24 gains, but do not supply a separate numeric
attribution for this executable and these rounds.

The full-suite result has the same limitation: reuse/native32 accounts for a 21.32% reduction
relative to the control, and adding coalescing contributes a further 1.63 percentage points
on that original denominator, giving 22.95%. The conditional coalescer improvement is 2.08%.

The coalescer's strongest isolated effect is Q23: 18.16% lower median, with 8/8 paired wins.
Q23 also improves 7.19% over the earlier announcements-off setting, with 8/8 paired wins.
The smaller coalescer effects on TPC-H Q19 and Q24 win 6/8 pairs and do not justify broad claims
about those individual effects. Most of their combined gains come from descriptor reuse.
ClickBench Q19's coalescer gain is 1.36 ms. No accepted outliers are removed.

| Query | Current min–max ms | Reuse/native32 min–max ms | Combined min–max ms | Announcements off min–max ms |
|---|---:|---:|---:|---:|
| clickbench-q23 | 515.805–654.722 | 524.701–611.903 | 400.192–452.248 | 453.981–581.678 |
| tpch-q19 | 120.909–173.523 | 69.265–89.677 | 66.929–87.552 | 99.334–187.001 |
| clickbench-q24 | 67.561–148.254 | 36.345–40.866 | 34.021–37.391 | 81.936–147.527 |
| clickbench-q19 | 35.537–90.104 | 35.332–36.213 | 34.077–34.863 | 35.678–36.272 |

### Traffic and native phases

Traffic below comes from untraced timing processes. Query-snapshot application bytes/reads
and whole-process storage/CPU cover different windows; their differences cannot identify
readahead or cancellation traffic.

| Query | Current app MB → combined | Current storage MB → combined | Current reads → combined | Current CPU s → combined | Current device queue mean → combined |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 | 4316.357 → 3884.449 | 4908.937 → 4074.996 | 381.5 → 3099.5 | 6.330 → 7.265 | 148.81 → 32.12 |
| tpch-q19 | 507.162 → 529.596 | 543.879 → 526.242 | 182 → 537 | 0.602 → 0.635 | 30.15 → 23.80 |
| clickbench-q24 | 220.377 → 190.718 | 289.352 → 263.764 | 364 → 525.5 | 0.703 → 0.663 | 11.97 → 19.59 |
| clickbench-q19 | 257.645 → 244.911 | 298.828 → 275.098 | 318.5 → 438.5 | 0.292 → 0.289 | 38.12 → 25.95 |

Per-read medians below come from one separate cold diagnostic per setting/query; values are
microseconds. Admission waits overlap across requests and cannot be added to query latency.
Request size and concurrency change the pread population. Trace/scanned-counter discrepancies
are retained in the raw summaries, including metadata and cancelled background reads.

| Query | GET current → combined µs | Combined admission µs | Blocking queue current → combined µs | pread current → combined µs | Resume current → combined µs |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 | 52.524 → 0.088 | 83689.248 | 7.329 → 5.992 | 31235.249 → 1985.345 | 10.072 → 8.820 |
| tpch-q19 | 2886.337 → 0.129 | 10619.139 | 5.632 → 4.702 | 15994.290 → 1645.314 | 7.740 → 8.293 |
| clickbench-q24 | 68.999 → 0.137 | 2302.892 | 8.899 → 7.078 | 1008.122 → 903.989 | 9.022 → 9.413 |
| clickbench-q19 | 18.564 → 0.146 | 803.461 | 4.696 → 4.208 | 699.983 → 518.536 | 6.261 → 6.870 |

## Full-suite cold screen

All 22 TPC-H SF10 and 43 zero-indexed ClickBench queries complete three cold rounds: 585
accepted timings, with no rejected rounds. Each of the three settings visits every order
position once. The combined setting lowers geometric-mean latency **22.95%**
(1.298× speedup), improving **59/65 medians**. Reuse/native32 alone lowers
geometric-mean latency 21.32% and improves 57/65 medians. Adding demanded coalescing
at fixed reuse/native32 improves the suite geometric mean a further
2.08%; its strongest confirmed individual effect is Q23's 18.16% improvement.

The sum of per-query medians falls from 14.777 to
11.797 seconds. This is a derived equal-workload sum, not a measured continuous run.
TPC-H geometric-mean latency falls 27.37% (20/22 faster). ClickBench geometric-mean latency
falls 20.59% (39/43 faster). Six medians regress; the largest relative loss is tpch-q2: 52.652 →
54.267 ms (3.07% slower).
Three rounds provide a broad screen, not confirmation of every small per-query effect.

Every accepted configuration/round has matching final result row counts. These counts are not
value checksums; targeted projection tests independently compare actual output arrays with the
reference executor. No accepted outliers are removed. The per-query table is below.

Four rich driver captures complete after all timings and are excluded from performance samples.

## Driver waits and IO startup

Four separate cold processes capture Q23 and TPC-H Q19 at fixed reuse/native32, with optional
extent growth on and off. All captures finish without observed competing work. “Optional”
allows registered segments to enlarge a read; “demanded” only allows demanded requests to
expand it. These captures log driver, request and scope events and take much longer than
untraced queries. Their schedule, startup and wait distributions do not estimate production
critical-path savings. They remain excluded from all timing medians.

| Query / policy | Traced wall ms | First compute ms | First native pread ms | First matched file completion ms | First morsel ms | First returned batch ms | DFS max depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 3186.470 | 38.679 | 45.649 | 47.352 | 153.893 | 317.202 | 3 |
| clickbench-q23 / demanded | 3020.249 | 29.707 | 34.363 | 34.597 | 109.393 | 265.868 | 3 |
| tpch-q19 / optional | 188.712 | 7.691 | 8.605 | 9.325 | 11.570 | 13.003 | 4 |
| tpch-q19 / demanded | 186.122 | 7.030 | 7.448 | 8.678 | 10.706 | 12.230 | 4 |

| Query / policy | Driver pending physical reads mean / peak | Native pread mean / peak | Native admission mean / peak | Async resume mean / peak | Ready morsels mean | Parked morsels mean |
|---|---:|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 21.52 / 52 | 1.05 / 17 | 0.00 / 1 | 5.44 / 30 | 11.94 | 56.22 |
| clickbench-q23 / demanded | 93.54 / 260 | 1.21 / 32 | 0.99 / 25 | 53.29 / 167 | 12.54 | 87.45 |
| tpch-q19 / optional | 23.89 / 124 | 7.77 / 32 | 7.81 / 92 | 0.19 / 4 | 1.04 | 143.99 |
| tpch-q19 / demanded | 94.88 / 192 | 4.85 / 32 | 4.82 / 145 | 1.02 / 30 | 0.85 | 36.66 |

Driver pending reads include waits before and after native pread; native depth covers observed
completed local phases. Kernel block-device depth counts a different population, including
split requests. A native limit of 32 therefore does not imply that driver or device depth
stays at 32. Native permits remain held until blocking work finishes, including after async
cancellation. Unfinished native phases are not fabricated from missing completion events.

| Query / policy | IO without ready delivery or compute ms | Runnable without compute ms | File completion → Fetch delivery p95 ms | Wake → session poll p95 ms | Unpark → compute p95 ms |
|---|---:|---:|---:|---:|---:|
| clickbench-q23 / optional | 32.882 | 1824.738 | 318.715 | 305.138 | 13.956 |
| clickbench-q23 / demanded | 59.973 | 1794.956 | 389.621 | 369.352 | 17.671 |
| tpch-q19 / optional | 14.136 | 68.865 | 34.012 | 33.595 | 0.344 |
| tpch-q19 / demanded | 2.290 | 66.062 | 24.651 | 17.405 | 0.321 |

Query state categories partition one wall window. Owner park times and separate gap counters
overlap across owners and cannot be summed into query latency. Both Q23 captures spend most
time with runnable work outside observed `compute()` calls; this includes logging, driver
bookkeeping and scheduling. The demanded trace has more IO-only waiting despite a lower
traced wall time, illustrating why one wait metric is not a causal speedup explanation.

Q23 started bytes without a matched Fetch in the diagnostic window fall from 1352.02 to
430.05 MB; summed coalescing gaps rise from 161.56 to 266.80 MB. The optional capture ends with
11 unfinished reads (103.45 MB); the demanded capture has none. These are application-range
coverage diagnostics, not measured kernel savings or proof of unused prefetch. TPC-H Q19 gap
bytes rise from 0.075 to 44.351 MB, with 44.28 MB of repeated range coverage: smaller demanded
reads can lose sharing and reread gaps. Its small additional coalescer benefit should be
weighed against that IO cost.

Full reports retain all stage startup times, pending logical Fetches, completed reads awaiting
Fetch delivery, byte depth, scope reference counts, withdrawals, partial/cancelled work and
release paths. `drivers/compact.json` selects startup, wait, native/driver depth, coverage and
lifetime metrics. These explain IO state; no compute algorithms are optimized in this patch.

## Use the measured IO policy

```bash
VORTEX_SCAN_V2=1 \
VORTEX_LOCAL_FILE_HANDLE_REUSE=1 \
VORTEX_LOCAL_READ_CONCURRENCY=32 \
VORTEX_SCAN_IO_COALESCE_OPTIONAL=0 \
VORTEX_SCAN_IO_READY_FETCH=0 \
/mnt/vortex-ssd/votex-4/bin/datafusion-io-only <benchmark arguments>
```

The harness variant is `v2-io-file-handle-demanded-native-32`. The coalescer lets optional
segments share already-covered physical bytes, including alignment prefixes, while retaining
unread interests for later demand or cancellation. Changing the controls requires a fresh
process. Settings remain opt-in; production defaults are unchanged. This recommendation is
measured for immutable local SSD data with DataFusion/Vortex, with cold file pages. Remote
stores, warm scans and other engines were not tested in this scope-correction rerun.

## Completed measurement audit

All **713 accepted cold timings**, **211 separate native-phase diagnostics** and **four driver
captures** finish. No rounds or driver captures are rejected in this rerun, and no accepted
outliers are discarded. `audit.json` verifies all file-page residency snapshots are zero, all
measurements use `nvme0n1`, all query row counts match, and every run uses the frozen executable.
`io-scope-audit.json` confirms the restored kernel/scheduling sources match HEAD and compute
benchmark variants are absent. The earlier mixed-scope source is preserved separately.

## Source and validation


Frozen IO-only executable SHA-256:
`f1d851df83356b7d38ef86d5f30e489b89adb03cd841dd574d57d4657091bc54`.
Commands, source snapshots, samples and observations are under
`/mnt/vortex-ssd/votex-4/results/io-only-20261008/`.

The release-debug benchmark rebuild passes. Twelve targeted projection/announcement cases pass
with withdrawal at its default, disabled and enabled, including actual output-array comparisons
for narrow/wide projections and mismatched chunk boundaries. All 46 Python harness/analyzer
cases pass on SSD. Layout clippy reports six existing library errors and eight existing test
errors; every failing file is unchanged, recorded in `checks/clippy-unchanged-files.json`.
Ruff, formatting, ty and patch whitespace checks pass. The separate compute restoration patch
passes `git apply --check`. No workspace-wide test or lint run was performed.

## Per-query cold medians

Three rounds per setting; milliseconds. Negative percentages are regressions.

| Query | Current | Reuse/native32 | Combined | Combined faster | Paired wins |
|---|---:|---:|---:|---:|---:|
| tpch-q1 | 295.894 | 242.837 | 203.497 | 31.23% | 3/3 |
| tpch-q2 | 52.652 | 53.612 | 54.267 | -3.07% | 0/3 |
| tpch-q3 | 181.968 | 108.321 | 101.735 | 44.09% | 3/3 |
| tpch-q4 | 68.681 | 64.756 | 57.358 | 16.49% | 3/3 |
| tpch-q5 | 239.506 | 183.067 | 185.755 | 22.44% | 3/3 |
| tpch-q6 | 127.925 | 49.876 | 50.304 | 60.68% | 3/3 |
| tpch-q7 | 306.830 | 253.695 | 250.884 | 18.23% | 3/3 |
| tpch-q8 | 317.889 | 191.081 | 187.569 | 41.00% | 3/3 |
| tpch-q9 | 422.789 | 327.431 | 342.159 | 19.07% | 3/3 |
| tpch-q10 | 222.343 | 153.728 | 147.858 | 33.50% | 3/3 |
| tpch-q11 | 40.789 | 40.991 | 39.704 | 2.66% | 2/3 |
| tpch-q12 | 68.663 | 72.092 | 70.246 | -2.31% | 1/3 |
| tpch-q13 | 97.944 | 94.905 | 93.753 | 4.28% | 3/3 |
| tpch-q14 | 141.957 | 65.505 | 65.942 | 53.55% | 3/3 |
| tpch-q15 | 159.956 | 81.057 | 84.756 | 47.01% | 3/3 |
| tpch-q16 | 42.513 | 42.589 | 42.161 | 0.83% | 2/3 |
| tpch-q17 | 392.341 | 296.009 | 291.197 | 25.78% | 3/3 |
| tpch-q18 | 441.752 | 411.535 | 414.874 | 6.08% | 3/3 |
| tpch-q19 | 149.282 | 69.839 | 68.511 | 54.11% | 3/3 |
| tpch-q20 | 217.155 | 114.329 | 114.157 | 47.43% | 3/3 |
| tpch-q21 | 334.386 | 296.290 | 291.609 | 12.79% | 3/3 |
| tpch-q22 | 29.911 | 31.095 | 29.405 | 1.69% | 3/3 |
| clickbench-q0 | 6.744 | 6.747 | 6.589 | 2.29% | 2/3 |
| clickbench-q1 | 11.557 | 11.033 | 11.758 | -1.73% | 1/3 |
| clickbench-q2 | 22.890 | 21.772 | 21.811 | 4.72% | 2/3 |
| clickbench-q3 | 82.866 | 43.244 | 42.704 | 48.47% | 3/3 |
| clickbench-q4 | 140.395 | 132.770 | 132.170 | 5.86% | 2/3 |
| clickbench-q5 | 171.133 | 141.516 | 141.058 | 17.57% | 3/3 |
| clickbench-q6 | 6.978 | 6.629 | 6.756 | 3.18% | 2/3 |
| clickbench-q7 | 12.850 | 12.782 | 12.552 | 2.31% | 3/3 |
| clickbench-q8 | 187.259 | 176.240 | 173.821 | 7.18% | 3/3 |
| clickbench-q9 | 255.276 | 224.387 | 223.695 | 12.37% | 3/3 |
| clickbench-q10 | 122.792 | 50.257 | 50.236 | 59.09% | 3/3 |
| clickbench-q11 | 85.705 | 54.200 | 51.411 | 40.01% | 3/3 |
| clickbench-q12 | 152.370 | 131.920 | 135.357 | 11.17% | 3/3 |
| clickbench-q13 | 229.128 | 222.551 | 219.420 | 4.24% | 3/3 |
| clickbench-q14 | 167.537 | 137.961 | 138.990 | 17.04% | 3/3 |
| clickbench-q15 | 157.919 | 155.339 | 156.334 | 1.00% | 1/3 |
| clickbench-q16 | 373.044 | 333.323 | 334.540 | 10.32% | 3/3 |
| clickbench-q17 | 366.325 | 331.702 | 331.117 | 9.61% | 3/3 |
| clickbench-q18 | 788.290 | 701.465 | 725.398 | 7.98% | 3/3 |
| clickbench-q19 | 36.176 | 35.505 | 34.382 | 4.96% | 3/3 |
| clickbench-q20 | 444.729 | 315.218 | 298.263 | 32.93% | 3/3 |
| clickbench-q21 | 484.503 | 351.035 | 314.609 | 35.07% | 3/3 |
| clickbench-q22 | 592.271 | 522.275 | 476.333 | 19.58% | 3/3 |
| clickbench-q23 | 625.695 | 550.840 | 435.236 | 30.44% | 3/3 |
| clickbench-q24 | 102.204 | 37.068 | 34.704 | 66.04% | 3/3 |
| clickbench-q25 | 142.150 | 45.795 | 45.067 | 68.30% | 3/3 |
| clickbench-q26 | 96.751 | 39.384 | 35.355 | 63.46% | 3/3 |
| clickbench-q27 | 435.164 | 330.934 | 310.350 | 28.68% | 3/3 |
| clickbench-q28 | 980.062 | 931.807 | 955.042 | 2.55% | 3/3 |
| clickbench-q29 | 35.764 | 34.862 | 34.288 | 4.13% | 2/3 |
| clickbench-q30 | 199.415 | 127.524 | 123.658 | 37.99% | 3/3 |
| clickbench-q31 | 281.101 | 167.487 | 171.189 | 39.10% | 3/3 |
| clickbench-q32 | 647.303 | 638.779 | 619.555 | 4.29% | 3/3 |
| clickbench-q33 | 827.480 | 742.899 | 746.677 | 9.76% | 3/3 |
| clickbench-q34 | 812.961 | 744.181 | 738.023 | 9.22% | 3/3 |
| clickbench-q35 | 147.591 | 134.855 | 130.349 | 11.68% | 3/3 |
| clickbench-q36 | 38.244 | 37.908 | 38.629 | -1.01% | 2/3 |
| clickbench-q37 | 22.210 | 22.217 | 22.420 | -0.94% | 0/3 |
| clickbench-q38 | 16.935 | 16.728 | 15.092 | 10.89% | 3/3 |
| clickbench-q39 | 72.570 | 72.612 | 73.822 | -1.73% | 0/3 |
| clickbench-q40 | 13.709 | 13.302 | 13.545 | 1.20% | 3/3 |
| clickbench-q41 | 13.658 | 12.969 | 12.640 | 7.45% | 3/3 |
| clickbench-q42 | 14.473 | 14.109 | 14.329 | 0.99% | 2/3 |
