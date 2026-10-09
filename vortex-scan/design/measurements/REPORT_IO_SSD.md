<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Cold IO on the local SSD

All tests and measurements below use the local instance SSD, `nvme0n1`, mounted as XFS at
`/mnt/vortex-ssd`. Data, executable, temporary directory, result files and working directory
are on that SSD. The storage guard verifies the device. The full dataset contains 350 files
and 43,574,143,340 bytes. The EBS dataset and earlier results remain preserved.

The full screen repeats 29 distinct IO configurations across four queries, with two fresh
processes per configuration/query: 232 accepted cold timing samples. Compute-kernel experiments
and the duplicate current-default inline-Fetch configuration are excluded. Screening order
is rotated but is not fully balanced across 29 configurations; screening results select
candidates rather than establish gains. Three contended rounds were retained and retried.

The follow-up uses four configurations and eight rotated/reversed rounds per query:
128 accepted timing samples. Every configuration visits every order position twice. One
contended round was retained and retried. All post-eviction residency checks report zero
cached Vortex data pages; each process executes exactly one query with no discarded iteration.
Only file contents are evicted; metadata and device caches can stay warm.

## Eight-round confirmation

| Query | Configuration | Median ms | Change vs current | Paired wins | Min–max ms | Process storage MB | Application MB | Application reads |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clickbench-q23 | Current V2 | 533.331 | — | — | 516.057–633.915 | 4913.582 | 4304.895 | 376 |
| clickbench-q23 | Reuse local file handle | 548.157 | 2.78% slower | 1/8 | 533.739–571.890 | 5103.399 | 4794.435 | 412 |
| clickbench-q23 | Disable announcements + native limit 32 | 442.275 | 17.07% faster | 8/8 | 430.680–462.891 | 4324.334 | 4259.720 | 592.5 |
| clickbench-q23 | Defer projection + one physical slot/file | 469.036 | 12.06% faster | 8/8 | 461.219–484.693 | 4437.142 | 4783.048 | 1351.5 |
| tpch-q19 | Current V2 | 156.656 | — | — | 126.148–188.462 | 543.879 | 507.162 | 182 |
| tpch-q19 | Reuse local file handle | 77.448 | 50.56% faster | 8/8 | 72.357–78.601 | 524.667 | 507.162 | 182 |
| tpch-q19 | Disable announcements + native limit 32 | 160.189 | 2.26% slower | 3/8 | 122.449–224.133 | 543.879 | 507.162 | 182 |
| tpch-q19 | Defer projection + one physical slot/file | 232.236 | 48.25% slower | 0/8 | 231.302–233.166 | 543.879 | 509.491 | 202.5 |
| clickbench-q24 | Current V2 | 109.972 | — | — | 55.016–144.150 | 290.177 | 222.450 | 368.5 |
| clickbench-q24 | Reuse local file handle | 37.639 | 65.77% faster | 8/8 | 36.016–43.309 | 277.629 | 224.003 | 370.5 |
| clickbench-q24 | Disable announcements + native limit 32 | 98.942 | 10.03% faster | 5/8 | 54.271–140.161 | 281.020 | 214.735 | 358 |
| clickbench-q24 | Defer projection + one physical slot/file | 42.603 | 61.26% faster | 8/8 | 40.470–44.855 | 330.111 | 250.655 | 453.5 |
| clickbench-q19 | Current V2 | 56.958 | — | — | 35.585–149.346 | 299.315 | 257.985 | 314.5 |
| clickbench-q19 | Reuse local file handle | 35.863 | 37.04% faster | 7/8 | 35.494–36.584 | 291.279 | 258.595 | 316 |
| clickbench-q19 | Disable announcements + native limit 32 | 36.203 | 36.44% faster | 7/8 | 35.310–37.086 | 297.796 | 256.670 | 319.5 |
| clickbench-q19 | Defer projection + one physical slot/file | 35.444 | 37.77% faster | 8/8 | 35.114–36.499 | 296.118 | 254.737 | 309.5 |

Handle reuse improves the three smaller queries by 37–66%, and loses 2.78% on Q23.
Disabling announcements with a native-read limit of 32 improves Q23 by 17.07%, winning every
pair and lowering median whole-process storage reads from 4.914 to 4.324 GB. Deferral plus
one physical slot improves Q23 by 12.06% but regresses TPC-H Q19 by 48.25%. These are different
settings; their percentages are not added and no one setting establishes a broad default.

The control is variable even in accepted rounds: ClickBench Q19 spans 35.585–149.346 ms.
All accepted samples, including the 633.915 ms Q23 control, remain in the medians and ranges.
No timing outlier was removed. Query outputs contain ten rows for Q23/Q24, one for TPC-H Q19,
and four for ClickBench Q19. Cardinality is not a result checksum.

## IO and compute interpretation

Application read bytes describe completed application ranges at query return. Process storage
bytes use Linux child `ru_inblock * 512` and include setup and background reads after query
return. Device counters describe the whole device over the process window, not a query-only
window. Differences between these populations cannot identify readahead or cancellation bytes
by subtraction. Diagnostic logging and driver captures are separate from timing samples.

The handle-reuse implementation retains one opened local descriptor per reader and avoids
repeating object-store GET/setup for every range. It deliberately pins the opened inode if
the path is replaced; remote stream payloads keep issuing dynamic GETs. Both behaviors are
covered by the rerun IO tests. This mode remains opt-in, `VORTEX_LOCAL_FILE_HANDLE_REUSE=1`,
rather than changing path-replacement semantics globally. The Q23 policy uses
`VORTEX_SCAN_EARLY_ANNOUNCE=0 VORTEX_LOCAL_READ_CONCURRENCY=32`; the projection/physical-slot
policy uses `VORTEX_SCAN_PROJECT_ANNOUNCE=0 VORTEX_SCAN_IO_READ_CONCURRENCY=1`.

### Native read phases

These are per-read medians in microseconds from one separate cold diagnostic process per
configuration/query after confirmation, not medians across the eight timing processes.
Phases overlap across reads and cannot be added to reconstruct query wall time.

| Query | GET/setup current → reuse µs | Blocking queue current → reuse µs | pread current → reuse µs | Async resume current → reuse µs |
|---|---:|---:|---:|---:|
| clickbench-q23 | 72.402 → 17.050 | 6.961 → 8.618 | 19868.152 → 48141.683 | 11.569 → 8.145 |
| tpch-q19 | 2695.551 → 0.158 | 7.958 → 97.955 | 13512.893 → 34219.395 | 8.161 → 6.061 |
| clickbench-q24 | 46.089 → 0.165 | 8.362 → 26.221 | 899.612 → 1565.395 | 8.507 → 7.987 |
| clickbench-q19 | 17.114 → 0.175 | 4.637 → 4.557 | 648.671 → 598.539 | 6.139 → 5.341 |

TPC-H Q19's application traffic and read count are unchanged across the confirmation
(507.162 MB and 182 reads), while repeated GET/setup nearly disappears in its diagnostic.
The longer per-read pread median does not establish slower device service: concurrency,
request mix and logging change the distribution. The end-to-end timing establishes the win.

### Driver state and progress

Four separate cold driver/lifetime captures record query boundaries, first execution of each
stage, runnable/parked queues, request and byte depth, compute/IO overlap, DFS ancestry,
completion-to-delivery and wake-to-poll waits, output progress and interest withdrawal.
The compact selection is saved in `drivers/analysis.json`; full reports retain stage and
per-source distributions. Every time below is a traced wall measurement in milliseconds.

| Metric | Q23 current | Q23 announcements off/native 32 | TPC-H Q19 current | TPC-H Q19 handle reuse |
|---|---:|---:|---:|---:|
| Traced query window ms | 3000.481 | 4103.165 | 272.241 | 210.512 |
| Compute union ms | 1097.883 | 1264.704 | 82.815 | 82.578 |
| IO/compute overlap ms | 1096.704 | 1264.047 | 29.454 | 30.041 |
| IO present, no runnable work/delivery/compute ms | 33.359 | 3.624 | 98.363 | 34.837 |
| Runnable work without observed compute ms | 1767.002 | 2819.697 | 68.252 | 70.283 |
| First planner compute ms | 27.313 | 15.149 | 7.042 | 6.687 |
| First morsel compute ms | 156.309 | 421.243 | 11.391 | 11.292 |
| First emitted scan batch ms | 248.227 | 1000.661 | 12.910 | 12.880 |
| First physical completion ms | 29.292 | 405.924 | 9.188 | 8.845 |
| Physical requests mean / peak | 20.18 / 46 | 112.05 / 240 | 35.10 / 124 | 29.96 / 124 |
| Physical request bytes mean / peak MB | 244.63 / 562.91 | 795.47 / 1738.20 | 94.65 / 317.33 | 86.16 / 317.33 |
| Ready morsels mean / peak | 9.77 / 32 | 7.04 / 31 | 0.75 / 11 | 0.99 / 13 |
| Reconstructed maximum DFS depth | 3 | 3 | 4 | 4 |
| File completion → gating Fetch delivery p95 ms | 366.785 | 677.691 | 34.997 | 35.733 |
| Unpark → next compute p95 ms | 11.099 | 9.912 | 0.348 | 0.354 |
| IO wake → next session poll p95 ms | 351.804 | 644.122 | 34.521 | 35.546 |
| Completed registration retention p95 ms | 721.465 | 1123.467 | 176.717 | 118.766 |
| Unfinished physical reads at query boundary | 3 | 99 | 0 | 0 |
| Undelivered logical Fetches at query boundary | 4800 | 13440 | 0 | 0 |

TPC-H Q19's traced reduction is concentrated in the IO-only observed state (98.363 →
34.837 ms), with nearly identical compute union, first batch, DFS depth and peak request
depth. Completion-to-delivery p95 remains around 35 ms, so handle reuse does not establish
a driver-delivery improvement. It removes backend setup and shortens the traced IO tail.

Q23's heavy instrumentation expands wall time by several times and reverses the untraced
ordering: 3000/4103 ms here versus confirmed medians 533/442 ms. Its queue depths, startup and
waits describe this instrumented schedule; they cannot prove the cause or production latency
tradeoff of the 17.07% gain. Native concurrency 32 limits local read admission; file-service
in-flight depth includes waiting requests and can exceed 32. Request byte depth is not RSS.
State partitions describe observed availability, not CPU idleness or causal bottleneck shares.

The Q23 control withdraws 1,194 optional interests totaling 165.302 logical MB, but 831
withdrawals occur after physical completion. All 232 withdrawals before physical start
still have another observed reference; none is the last reference. Fully rejected root scopes
clear IO at p95 0.348 ms. This capture supports retaining shared references and releasing
unneeded scope interests, while providing no evidence that those 232 withdrawals save
physical reads. Shared logical bytes and the observed 686.159 MB of exclusively never-fetched
members in completed scopes are not proven avoidable storage traffic.

## Noise handling and reproducibility

The first Q23 screen used six ten-second quiet checks. Other workspaces kept starting builds,
benchmarks and profiles, so subsequent screens and confirmation use `--quiet-seconds 10`:
six checks spanning ten seconds. CPU, IO wait, IO pressure and current SSD activity are checked;
any detected benchmark/build overlapping a timing round rejects the entire round. Raw rejected
samples and wait observations remain saved. The new option leaves the default at 60 seconds.

The Rust source matches the frozen executable: 25 tracked Rust patches and three extra source
files were compared. No production Rust code was changed in this rerun. Main executable SHA-256:
`3af02184c6cd7b825873addcb9ca9fbb6562c27aea347a387c9f9d8ec5f5d82e`. Saved pre-audit executable:
`3a918a5389ee072818013a43f49c12ce653efc036f468eb0d12d524f210ca343`. The saved executable
is a historical control containing earlier branch work, not the original PR binary.
The harness used after the Q23 pilot is SHA-256
`b4344656206db5a46058d25c4fdf3dccb8f7f7f2d8fbb336c8cc0da97ee8e00a`;
the pilot's default-window snapshot is saved separately.

All source/harness snapshots, exact commands, model/device records, timings and phases are at
`/mnt/vortex-ssd/votex-4/results/rerun-20261007/`, also reachable through
`target/scan-io-results/ssd-io/rerun-20261007/`. Key files:

- `q23-quiet60-pilot/summary.json`: initial Q23 screening and separate diagnostics.
- `screen-remaining/summary.json`: the other three screens, retained rejected attempts and phases.
- `confirm/summary.json` and `confirm/analysis.json`: eight-round confirmation and derived metrics.
- `screen-command.json`, `q23-quiet60-pilot-command.json`, `confirm-command.json`: exact commands and toggles.
- `source/`, `hashes.txt`, `harness-quiet10-hash.txt`: source snapshots and executable/harness hashes.
- `drivers/summary.json`, `drivers/analysis.json` and `drivers/*.driver.json`: separate cold
  state-machine/lifetime captures and selected measurements.

## Checks

The 46 Python analyzer/harness tests, 14 scan IO tests with projection deferral/one slot,
10 default file-source tests and all 197 IO crate tests pass with SSD temporary storage.
The Rust test binaries execute from SSD. Targeted Ruff lint/format and ty checks pass.
The quiet-window change passes the repeated 46 Python tests, including the selected-window
and contention-rejection checks. Rust compilation for the IO test executable succeeds.
No workspace-wide Rust or binding checks were run. Earlier unrelated layout/V2 fixture
failures are documented in the historical reports and were not edited in this rerun.

## Full screening matrix

These are medians of two accepted fresh-process cold timings in milliseconds. This exploratory
screen includes V1 and historical executables as controls. The eight-round table above is the
confirmation; two-round differences alone do not establish a gain. Variant toggles are recorded
in the exact command artifacts and the frozen harness.

| Configuration | ClickBench Q23 ms | TPC-H Q19 ms | ClickBench Q24 ms | ClickBench Q19 ms |
|---|---:|---:|---:|---:|
| v2-unlimited | 528.023 | 135.603 | 105.642 | 82.155 |
| v1 | 500.052 | 139.555 | 148.729 | 36.520 |
| v2 | 492.741 | 144.010 | 113.645 | 36.254 |
| v2-previous | 536.172 | 143.256 | 123.308 | 35.924 |
| v2-previous-boxed | 600.584 | 156.635 | 114.631 | 36.286 |
| v2-ungrouped | 555.684 | 150.251 | 128.644 | 35.896 |
| v2-ready-fetch | 529.614 | 152.937 | 123.269 | 82.381 |
| v2-group-sparse | 621.268 | 149.873 | 116.039 | 82.820 |
| v2-io-prune-interests | 528.973 | 157.194 | 99.569 | 35.968 |
| v2-io-prune-group | 522.536 | 198.476 | 136.216 | 35.990 |
| v2-io-4m | 708.658 | 128.690 | 105.958 | 35.986 |
| v2-io-read-slots-1 | 526.733 | 232.241 | 39.072 | 36.053 |
| v2-io-read-slots-4 | 567.616 | 95.579 | 129.420 | 35.952 |
| v2-io-file-handle | 539.884 | 83.765 | 36.000 | 35.438 |
| v2-io-file-handle-tight-gap | 533.215 | 81.709 | 35.485 | 34.837 |
| v2-io-tight-gap | 584.722 | 217.748 | 133.296 | 130.412 |
| v2-io-4m-tight-gap | 671.756 | 239.722 | 85.316 | 121.110 |
| v2-file-pruning | 526.240 | 160.953 | 78.872 | 34.911 |
| v2-splits-8 | 615.436 | 131.190 | 109.512 | 36.495 |
| v2-splits-32 | 530.316 | 177.884 | 106.080 | 36.014 |
| v2-project-early | 515.623 | 140.601 | 109.286 | 75.297 |
| v2-project-late | 926.047 | 189.314 | 112.953 | 36.342 |
| v2-io-project-late | 863.204 | 231.752 | 113.840 | 36.303 |
| v2-io-project-late-read-slots-1 | 467.768 | 231.094 | 42.652 | 35.449 |
| v2-io-no-announcements | 549.913 | 156.061 | 123.610 | 70.766 |
| v2-io-project-late-tight-gap | 780.423 | 226.302 | 82.411 | 81.877 |
| v2-baseline | 533.966 | 146.236 | 105.585 | 36.069 |
| v2-lookahead | 481.592 | 156.478 | 117.386 | 35.652 |
| late | 434.328 | 164.491 | 108.598 | 35.743 |
