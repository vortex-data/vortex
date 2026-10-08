<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DataFusion V2 integration without the standalone BETWEEN changes

The final SQL comparison starts at the feature branch's
`5ea2e32af8f0f747aab5ee5d9c01c933f04c8f78`, including its new local IO optimization and
scan diagnostics. The branch advanced during validation, so the older-head SQL panel was
stopped and excluded. It integrates the
streaming predicate bitmap kernel and DataFusion concurrency/IO diagnostics. The standalone
BETWEEN changes from [#10378](https://github.com/vortex-data/vortex/pull/10378) and the
experimental filter read-ahead controller are excluded.

The earlier Q6 improvements measured with the separate BETWEEN implementation present do
not describe this integration. The measurements below use binaries that both exclude it.
The validated kernel gain is 39–66% for the existing native bitpacked BETWEEN benchmark;
it must not be presented as a default DataFusion SQL improvement.

## Implementation and scope

- `b5175798a`: use the existing lane predicate kernel for complete 64-bit bitmap words.
  Scalar prefixes and tails preserve bits written by adjacent chunks. The 1,024-value
  scratch buffer and sorted patch cursor are unchanged. The existing native bitpacked
  BETWEEN kernel uses this implementation; no new expression delegation is added.
  Ordinary comparisons already use the feature branch's fused FastLanes compare kernel,
  including patched arrays. Only empty/zero-width comparisons fall back to this helper.
- `72310f5f0`: expose existing scan concurrency through the DataFusion benchmark, add an
  optional dedicated IO executor, route the file read driver through `spawn_io`, and add
  opt-in physical IO/query/projection timing plus the overlap summarizer. Unsupported
  filter read-ahead CLI controls were removed when integrating this commit.
- `4fb0f8e46`: fix seven pre-existing Clippy violations in the feature branch's new
  reader-cache tests by spelling shared ownership as `Arc::clone`. This is a test-only change,
  also present in the newer remote IO commit.
- `ee2312065`: merge the integration onto the new remote tip. Its first parent is
  `5ea2e32af` and its second parent is `4fb0f8e46`. Resolutions preserve local admission
  permits during cancelled blocking reads, file-handle reuse behavior, projection alias
  handling, and the new monotonic trace fields. Physical completion timestamps for the
  overlap summarizer remain in a separate Unix clock domain, matching compute/query events.

Both new benchmark options default to unset. Ordinary Tokio executors already route
`spawn_io` to `spawn`, so the routing fix preserves their scheduling. No extra decoded
array, raw segment, filter-mask, or query-result cache is introduced. The comparison helper
clears segment-cache/preload, scan-policy and local IO overrides and records its exact
per-process environment. Experimental local file-handle reuse remains disabled.

## Method

The Rust 1.98 ARM64 `release_debug` builds are frozen before any subsequent build.
The final SQL baseline comes from clean `5ea2e32af`; the candidate comes from clean
`ee2312065`. The native encoding comparison used clean `e645cd9a9` and `72310f5f0`.
The incoming IO commit changes none of `encodings/fastlanes`, `vortex-array`, `vortex-compute`,
`vortex-buffer`, `vortex-session` or `benchmarks/bench-support`; their source trees are identical
to the measured native candidate after the merge. Native timings characterize that kernel,
not the new IO implementation.

Artifacts are under `/home/ec2-user/votex-3/target/datafusion-v2-integration`.
Each binary has a source-commit and SHA-256 manifest. Runs record commands, environment,
affinity, raw iteration timings and query-scoped perf counters. CPU counters are
reported from all 25 completed diagnostic pairs (three or four per query); the final
ten pairs repeat elapsed-time measurements only. The counters run in separate processes, outside the timed runs.

DataFusion uses the listing-table scan path, eight partitions and eight Tokio workers on
CPUs 24–31. Files are non-compact Vortex, with file pruning enabled and projection read-ahead
disabled. Scan concurrency and the separate IO runtime are left at their defaults in the
before/after comparison. TPC-H uses the existing scale-factor-10 fixture; FineWeb uses the
existing local fixture. OS page-cache and the pre-existing session metadata reuse are allowed
equally in both versions. These are repeated-query measurements, not cold-device or cloud IO
measurements.

There are five alternating process rounds and two excluded warmups. The first nine
accepted pairs use 30 iterations per process; the remaining 26 use 10 iterations to fit
quiet windows between external jobs. Both labels use the same iteration count within each pair.
The reported paired change is the median of the five per-round percentage changes;
pooled times are the median of 80 measured iterations per version for Q1/Q6 and
60 for each other query. Shorter processes were selected because of host contention,
not their measured performance. Negative changes
mean faster. DEBUG tracing is disabled during timing. A process monitor samples known
competing builds/benchmarks every 100 ms and requires a quiet window before starting a measurement: three seconds for the
initial processes and one second for the shortened remaining processes.
The first three pairs were accepted under a whole-pair quiet rule.
The remaining pairs guard each timed or perf process separately: a contaminated process is
archived and retried before proceeding. No accepted process observed a known competitor.
Acceptance depends on observed contention, never on the measured speedup. Earlier panels
against the old feature head and contaminated attempts are excluded; their records remain
in the artifacts.

## Default DataFusion SQL results

All 35 paired comparisons completed against the latest feature head. The CPU column uses
all 25 completed diagnostic pairs, with three or four pairs per query. Each diagnostic
process scopes its counters to query callbacks; their durations are not the SQL timings.

| Query | Baseline (ms) | Candidate (ms) | Median paired change | Five-round range | Instructions change |
| --- | ---: | ---: | ---: | ---: | ---: |
| fineweb_q03 | 67.510 | 67.672 | +2.26% | -4.12% to +4.36% | +0.028% |
| tpch_q01 | 371.131 | 360.012 | -2.82% | -4.46% to +3.74% | +0.004% |
| tpch_q06 | 35.674 | 35.852 | +1.04% | -1.65% to +2.08% | -0.016% |
| tpch_q12 | 75.527 | 74.621 | -1.46% | -6.54% to +2.09% | +0.007% |
| tpch_q14 | 45.539 | 46.366 | +0.44% | -1.10% to +4.98% | -0.005% |
| tpch_q15 | 87.034 | 85.879 | -1.93% | -2.34% to +2.26% | +0.005% |
| tpch_q19 | 50.085 | 50.094 | +2.15% | -5.34% to +6.34% | +0.017% |

Every query has both faster and slower rounds. Instruction counts differ by less than
0.03%, and the paired elapsed-time results range from -2.82% to +2.26%. This panel does
not establish a broad default DataFusion throughput improvement. The native kernel gain
below is the reproducible performance benefit; the IO controls and diagnostics support
future scheduling experiments while retaining current defaults.

## Native comparison kernel

The existing `compute_between` benchmark exercises the native bitpacked kernel directly,
without the new extension/FoR BETWEEN delegation from the separate PR. All nine cases
improve in every one of five accepted rounds, with zero competing-process samples.
The table shows medians of the five process medians; paired changes use per-round ratios.

| Type | Values | Baseline (µs) | Candidate (µs) | Paired change |
| --- | ---: | ---: | ---: | ---: |
| i16 | 2,048 | 2.448 | 1.392 | -43.28% |
| i16 | 16,384 | 13.430 | 4.870 | -63.78% |
| i16 | 32,768 | 26.090 | 8.880 | -65.98% |
| i32 | 2,048 | 2.441 | 1.448 | -40.35% |
| i32 | 16,384 | 13.140 | 5.335 | -59.32% |
| i32 | 32,768 | 25.420 | 9.902 | -61.04% |
| i64 | 2,048 | 2.710 | 1.643 | -39.26% |
| i64 | 16,384 | 15.500 | 6.972 | -55.03% |
| i64 | 32,768 | 30.220 | 13.180 | -56.36% |

The unpatched `bitpack_compare_sweep` is a control: all 62 i32/u32 bit-width cases
change by 0.00% to +0.33%, with no speedup established. The feature branch already routes
ordinary and patched comparisons through the fused `unpack_cmp` implementation.
Only empty/zero-width comparisons use `stream_predicate`; the native BETWEEN kernel
always uses it. This dispatch explains why the native range benchmark improves while
the unpatched compare sweep does not.

Native runs request 100 samples of 32 iterations with a 25 ms minimum per case.
Divan may collect extra samples to meet that minimum. Each label uses identical settings,
and the execution context/session setup follows the existing benchmark unchanged.

## IO diagnostic validation

The final candidate was traced with default scheduling and with `--io-threads 1`.
The summarizer successfully joins physical read, Vortex compute and query callback windows
for four measured iterations per configuration. No competing process was observed.

| Query/configuration | Reads | Bytes | Mean outstanding depth | Empty read-phase time |
| --- | ---: | ---: | ---: | ---: |
| tpch-6-default | 154.0 | 397,851,512 | 2.56 | 50.05% |
| tpch-6-io1 | 156.5 | 397,851,510 | 1.95 | 28.15% |
| fineweb-3-default | 129.0 | 1,539,373,824 | 9.08 | 0.00% |
| fineweb-3-io1 | 129.0 | 1,539,373,828 | 11.10 | 0.21% |

Q6 still has empty IO intervals with the default policy, while FineWeb Q3 already keeps
reads outstanding through essentially its entire read phase. A separate IO worker changes
Q6 scheduling and reduces the empty fraction, with essentially unchanged byte volume.
Read counts and tiny padding differences vary with coalescing. These observations establish
functional IO overlap instrumentation, not a throughput gain from the optional runtime.
The SQL timing comparison uses default scheduling and has DEBUG disabled.

## Optional IO-runtime probe

This exploratory probe compares the same frozen candidate binary with default scheduling
and `--io-threads 1`. Scan concurrency remains unset. DEBUG tracing is disabled, and each
process has 20 iterations with two excluded warmups. One clean pair completed per query.
The remaining four planned pairs were deferred because shared-host work repeatedly
prevented clean measurements. An unmatched later process is excluded.

| Query | Default (ms) | IO worker 1 (ms) | Single-pair change |
| --- | ---: | ---: | ---: |
| fineweb_q03 | 66.795 | 68.175 | +2.07% |
| tpch_q06 | 35.762 | 36.502 | +2.07% |

These initial pairs do not establish a reproducible benefit from a dedicated IO runtime.
The DEBUG queue-depth observations above measure scheduling behavior; they do not show a
throughput win. The runtime remains optional and defaults to unset.

## Correctness and checks

All 462 targeted Rust tests pass on the final merged tree: 299 bitpacking, 91 V2 scan,
10 DataFusion, 11 file-driver, 6 object-store/admission, 21 coalescing, 15 scan-IO,
8 projection and 1 benchmark diagnostic test. The IO-routing regression was also checked
against the old `spawn` call, where it fails with zero IO spawns instead of one, then
restored and passed with `spawn_io`. These checks cover the merge resolutions and incoming
cancellation, coalescing and projection behavior. Five Python overlap summarizer regressions pass.

The post-merge custom equality query returns `[{"matched_rows": 24995}]` identically for the feature
baseline, the integrated candidate, the candidate with `--io-threads 1 --scan-concurrency 1`,
and Parquet in all three configurations. This independent format comparison validates the
result; it does not establish a performance benefit for the optional runtime settings.

Post-merge Clippy passes for all targets and features of the seven affected crates, with warnings denied.
Pinned `nightly-2026-09-10` formatting, Ruff checks/formatting and patch whitespace checks pass.
The first Clippy attempt found the seven pre-existing test-style failures fixed in the separate
commit above. Workspace-wide tests, doctests, DuckDB and cold/cloud storage runs were not run.

## Reproduction

For reproduction, use feature baseline `5ea2e32af` and published runtime merge
`7fff43c56`, whose source tree matches the frozen local candidate `ee2312065`.
Build each version in a clean worktree and copy the executable to a distinct immutable path:

```bash
cargo build --locked --profile release_debug -p datafusion-bench -j8
cargo bench --locked --profile release_debug -p vortex-fastlanes \
  --bench bitpack_compare_sweep --bench compute_between \
  --features _test-harness --no-run -j8
```

A uniform 30-iteration reproduction uses the following command. The accepted local panel
retains nine initial 30-iteration pairs and uses 10 iterations for the remaining pairs,
as described above:

```bash
taskset -c 24-31 python3 scripts/bench-v2-drivers.py \
  --baseline datafusion="$BASELINE" --candidate datafusion="$CANDIDATE" \
  --file-pruning --data-root "$DATA_ROOT" --output "$OUTPUT" \
  --workload tpch:1,6,12,14,15,19 --workload fineweb:3 \
  --formats vortex --threads 8 --prefetch 0 \
  --iterations 30 --warmup 2 --rounds 5 --perf-stat
```

For the custom equality query, restrict the workload to `tpch:6` and set
`VORTEX_BENCH_QUERY_OVERRIDE_DIR` to a directory containing `6.sql`:

```sql
SELECT COUNT(*) AS matched_rows
FROM lineitem
WHERE l_shipdate = DATE '1994-06-15';
```

This is a custom predicate query, not ordinary TPC-H Q6. Do not enable result-file writing or
DEBUG logs during performance comparisons. The native sweep uses each frozen executable with
`--bench 'i32|u32' --color never --timer os --sample-count 100 --sample-size 32 --min-time 0.025 --max-time 0.1`,
also in five alternating rounds. The `compute_between` executable uses the `bitpack` filter
in place of `i32|u32`, with the same sampling settings. Run comparisons serially on a quiet machine.

For IO diagnostics enable `vortex_io::read_timing`, `vortex_scan::compute_timing` and
`vortex_bench::query_timing` at DEBUG, then run:

```bash
python3 scripts/summarize-scan-overlap.py "$LOG" \
  --warmup 2 --ignore-backlog --output "$SUMMARY"
```

DataFusion has independent concurrent partition streams, so a last-reported per-stream
backlog cannot be treated as their global sum. The IO depth is queued/running application
`pread` requests, including reads served from the OS page cache; it is not hardware queue
depth. Explicit timestamps exclude delayed async resumption from physical read duration.
DEBUG traces perturb scheduling and their elapsed times are not performance results.

## Publication and history

The command-line checkout has no Git write credentials. The authenticated GitHub connection
creates commits with the same complete Git trees and ordered parents, then advances the feature
ref with force disabled and an expected-head check. Published feature commits retain their original
hashes. Only the new, previously unpublished local commits receive different hashes from their
publication timestamps; their sign-offs match the authenticated committer.

| Local validation commit | Published commit | Identical source tree |
| --- | --- | --- |
| `b5175798a` | `703d90862` | `1af4d4fbe490abd65d866ddae2e712f58005e113` |
| `72310f5f0` | `b256acc12` | `b84151274fa84b97e85c902ffbdfacc2e5d3042e` |
| `4fb0f8e46` | `8d2d45f3f` | `a2ecb4cc35f6bb0dc007a544da5bb7f15e95b556` |
| `ee2312065` | `7fff43c56` | `51903c142616e06e86dcafd4159edc94991c081e` |

The published runtime merge has `5ea2e32af` as its first parent and `8d2d45f3f` as its second.
The performance report is a subsequent documentation-only commit.

## Frozen binary hashes

| Binary | Source commit | SHA-256 |
| --- | --- | --- |
| `latest-baseline-datafusion-bench` | `5ea2e32af` | `d13caa239eb52600f57f23d122e14a375c70baba7e009d72e13fda93f73ee357` |
| `latest-candidate-datafusion-bench` | `ee2312065` | `9e6327d19de6e0cda6abd684bcd89a1f284c98f28f3a3c9bc2af6f49eb3063cd` |
| `baseline-compute_between` | `e645cd9a9` | `0523fd67e0613ca2b8a0981eaf86f1635ebef29d9325064519a6ccbf21335a94` |
| `candidate-compute_between` | `72310f5f0` | `be3ce519d7ec32e1904794208267c11fdc39529cce9c5a3b3472ea4eb6724622` |
| `baseline-bitpack_compare_sweep` | `e645cd9a9` | `54cb2e008f7a2206744d8539fd54b43922f4cfd870737354f0ddbbac6ca128b3` |
| `candidate-bitpack_compare_sweep` | `72310f5f0` | `e2718004e290da6b8fa38b587b19b45380903a90696a2c7a5654d03336421ed4` |
