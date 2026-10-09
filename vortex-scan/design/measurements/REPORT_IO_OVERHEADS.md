<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Scan IO overhead audit

This follow-up audits registration, shared Fetch delivery, coalescing, physical request
dispatch, result matching, cancellation, allocation, and the local object-store read path.
The previous direct-Fetch improvement remains documented in
[REPORT_IO_DRIVER.md](REPORT_IO_DRIVER.md). In this separate comparison, seven default changes
lower Q23's median from 225.650 to 223.780 ms (0.83%), winning five of six paired rounds. Q19
is essentially tied. The ready-completion prototype skips many allocations but makes Q23
2.21% slower than the final default, so it remains disabled. The gain is modest, and the
shared host still produces an unexplained 1.73-second sample retained in the comparison.

## Changes

| Overhead | Change | Behavioral constraint |
|---|---|---|
| Removing and reinserting a registration for every intent | Update its occupied HashMap entry in place | Announce, Prefetch, and Fetch retain the same shared interest |
| Atomic swap on every already-wanted range | Check with a relaxed load before claiming with the original swap | Concurrent first claim still sends Polled once |
| Fetch-queue allocation for sources without Fetches | Construct FuturesUnordered on the first pending Fetch | Polling an empty source retains the existing result/error contract |
| Fresh queue allocation during clear and Drop | Drop the optional queue without constructing its replacement | Clear cancels Fetches and releases interests |
| Boxing the one-shot receiver before Shared owns it | Store the concrete ReadBytes future directly in Shared | Preserve receiver notifications, backend errors, and channel cancellation |
| Output-vector allocation on an empty coalescer poll | Return before allocation when no requests are eligible | A closed source still ends once eligible requests are exhausted |
| Per-read HashSet and scratch-vector allocation during coalescing | Reuse scratch storage and retire selected/cancelled offsets after each expansion pass | Preserve candidate order, size/alignment constraints, and final offset order |
| Future task for a Fetch whose shared result is already observed | Experimental ready-completion queue | Correct owner/request, errors, clear, and wake forwarding are tested |

The ready-completion experiment uses `VORTEX_SCAN_IO_READY_FETCH=1`; its default remains off
because it regresses Q23. Submission publishes and takes the consumer waker under the same
mutex as polling, then wakes outside that mutex. Other Fetches continue through the normal
future queue. This can change completion order, which the IO protocol leaves unordered.

`future_queues` counts queue constructions and `ready_fetches` counts task allocations bypassed.
`inline_fetches` counts mapping-wrapper allocations avoided for Fetches still using the future
queue. They are different costs. The analyzer reports which scope counters were actually
present: a counter missing from an older trace is not a measured zero.
The existing `new_registrations` counter also counts the receiver-wrapper allocations removed
by concrete read futures. Earlier diagnostic captures create 2101–2161 Q19 registrations
and 39090–39216 Q23 registrations; these counts do not claim identical work between runs.

## CPU evidence

Two 40-iteration DataFusion ClickBench Q23 profiles use immutable release-debug executables,
Scan V2, early projection hints, and unlimited local admission. The candidate profile also
enables ready-completion delivery. This candidate profile precedes the final concrete-receiver
change. Symbols are resolved against each saved executable with
`llvm-symbolizer`, using ELF-relative addresses and CPU-weighted samples.

| Sampled stack/function | Before CPU weight % | Candidate CPU weight % |
|---|---:|---:|
| Atomic leaf directly beneath FileSplitIo submit | 0.613 | 0.219 |
| FileSplitIo submit, inclusive | 2.526 | 2.501 |
| FuturesUnordered Fetch polling, inclusive | 1.568 | 1.309 |
| FileSplitIo poll, inclusive | 1.518 | 1.267 |
| Scope clear, inclusive | 0.725 | 0.728 |
| Coalescer State next, inclusive | 0.194 | 0.149 |
| Local read_exact_at, inclusive | 17.485 | 14.853 |

Inclusive rows overlap and cannot be added. These are hotspot signals, not speedup estimates:
the shared host was contended, Q23 has LIMIT 10, and physical work before cancellation differs.
Samply reported 3575 lost events before and 19038 afterward, as well as duplicate mmap warnings.
The IO classification excludes generic runtime frames that also execute array computation;
its 24.137% and 21.163% totals cover stacks containing the file read/registration modules,
not every backend operation. Kernel reads remain the largest measured IO stack.

Artifacts are under `target/scan-io-results/io-overhead-audit/`: the saved executables,
`binary-hashes.json`, profile command JSON, both `.profile.json.gz` files, and Linux symbol and
summary JSON. Open either recording with `samply load <profile path>`.

## Timing and diagnostics

`scripts/bench-io.py --baseline-binary` adds a `v2-previous` control using a saved executable.
It uses the same runtime environment as `v2-unlimited`, with ready-completion delivery disabled.
`v2-ready-fetch` changes that flag in the candidate executable. Summaries retain both hashes.
Event tracing stays in separate diagnostic processes; timing counters follow the query timer.

The comparison waits for six consecutive quiet ten-second observations and rejects whole
rounds that overlap detected builds or benchmarks. The registration-only attempt in
`target/scan-io-results/io-overhead-registration-quiet/` never admitted a timing round. Its gate
observations are preserved. After the final rebuild it was replaced by the complete comparison
in `target/scan-io-results/io-overhead-final-quiet/`, with six balanced rounds and seven retained
iterations per configuration/round. Both queries finish with 42 retained samples per
configuration: 12 accepted rounds total, plus one rejected Q23 round retained and repeated.
All timings finish before the six separate read-timing diagnostics.

| Configuration | Q19 median ms | Q19 min–max ms | Q23 median ms | Q23 min–max ms |
|---|---:|---:|---:|---:|
| Saved pre-audit default | 28.093 | 26.687–140.282 | 225.650 | 203.995–253.209 |
| Final fixes, ready delivery off | 27.991 | 26.254–144.500 | 223.780 | 203.338–1725.244 |
| Final fixes, ready delivery on | 27.929 | 26.056–146.011 | 228.733 | 206.751–274.978 |

The final default wins two of six Q19 round medians and five of six Q23 rounds. Ready delivery
wins three of six Q19 rounds and two of six Q23 rounds against the final default. Treat Q19's
0.36% and 0.22% median differences as ties. The allocation bypass alone does not establish a
win: completion order and extra ready-queue/waker work also matter.

One final-default Q23 iteration takes 1725.244 ms despite no detected competing process in its
round; host IO wait is 10.04% across that process. It is not silently removed. The separate
rejected round overlaps two detected Python jobs. Quiet admission and process detection do
not provide exclusive ownership or prove that every retained iteration is uncontended.

| Configuration | Q19 median completed reads / MB | Q23 median completed reads / MB | Q23 median process CPU s |
|---|---:|---:|---:|
| Saved pre-audit default | 182 / 507.162 | 407 / 5023.749 | 59.671 |
| Final fixes, ready delivery off | 182 / 507.162 | 415 / 4960.363 | 59.095 |
| Final fixes, ready delivery on | 182 / 507.162 | 412.5 / 5023.719 | 59.517 |

Q19's measured physical work is identical. Q23's final-default completed bytes also fall
1.26%; its LIMIT 10 cancellation prevents assigning the entire median gain to CPU overhead
removal. Read snapshots follow query return and are application counters, not disk traffic.
Process CPU includes setup, all nine iterations, warmups, and shutdown; it is not per-query
compute CPU. Every retained query returns one Q19 row or ten Q23 rows. Cardinality is not a
value checksum; targeted source tests compare actual arrays.

The complete comparison is in `io-overhead-final-quiet/summary.json`; a compact aggregation is
in `io-overhead-audit/final-comparison-analysis.json`. Raw samples, gate observations, rejected
rounds, commands, counter snapshots, read-timing diagnostics, and hashes remain available.
The source corresponding to the final executable is captured in `final-source-state.json`,
`final-source-tracked.patch`, and `final-source-untracked.json` under the audit directory.

The final candidate SHA-256 is
`3e9f24eca6656b43ca931744a28c4cb63379b4c6bf5b347e23162ae87c6362a5`;
the saved pre-audit executable is
`3a918a5389ee072818013a43f49c12ce653efc036f468eb0d12d524f210ca343`.
The candidate used for the earlier CPU profile is
`63fdafbe75e9a071b4d62f3f875707d986fb13bde64087501bacee87f9733d8b`.

```bash
python3.11 scripts/bench-io.py \
  --binary target/scan-io-results/io-overhead-audit/datafusion-final \
  --baseline-binary target/scan-io-results/io-overhead-audit/datafusion-before \
  --data-root vortex-bench/data --output target/scan-io-results/io-overhead-final-quiet \
  --workloads tpch-q19 clickbench-q23 \
  --variants v2-previous v2-unlimited v2-ready-fetch --modes direct \
  --iterations 9 --rounds 6 --diagnostic-iterations 1 --timing-metrics \
  --wait-for-quiet --repeat-contended-rounds
```

Separate diagnostic captures in `target/scan-io-results/io-overhead-diagnostics/` retain the
driver queues, stage startup, IO phases, lifecycle counters, and cancellation inventory. They
are not timing samples. All four captures have no analyzer warnings and return the expected
result cardinality (one Q19 row, ten Q23 rows).

| Ready-completion diagnostic | Fetches | Bypassed future tasks | Direct-storage future tasks | Constructed future queues |
|---|---:|---:|---:|---:|
| Q19 | 4436 | 411 (9.3%) | 4025 | 504 |
| Q23 | 126880 | 45016 (35.5%) | 81864 | 4362 |

The disabled controls count 4944 Q19 Fetches and 125488 Q23 Fetches. Shared reuse, coalescing,
and cancellation differ between captures; these are actual path counts rather than matched
work inventories. Different cache warmth and host contention make these captures unsuitable
for an end-to-end comparison.

The six final read-timing diagnostics provide another breakdown. For the final default, Q23's
median GET is 1.425 ms, blocking-pool queue 0.047 ms, pread 2.260 ms, and async resume 0.052 ms.
Q19's corresponding medians are 2.717, 0.006, 0.573, and 0.825 ms. These are concurrent local
read-phase durations from separate traced runs, not parts to add into query time or a speedup
comparison. Backend GET/reads dominate the per-read phases more than blocking-pool admission.

Older traces may lack GET, admission, or allocation fields. The analyzer now omits unknown
physical phase values and intervals, and distributions count only actual measurements. A
present zero still counts as measured; a missing phase prints `unavailable`. A regression test
reproduces the previous `KeyError: get_ns` before the fix and passes afterward.

## Remaining costs and decisions

* Shared registration references are necessary when overlapping splits still need a range.
  Whole-root retirement already withdraws its interests; partial pruning needs precise leases
  before it can release a sibling's shared segment safely.
* Local object-store GET opens/checks the file before the separate blocking pread dispatch.
  The native file reader already holds its file handle. Caching generic object-store file
  payloads would change backend replacement/version semantics, so this audit preserves them.
* Result matching scans at most the reader's bounded batch. Profiles do not identify it as a
  material hotspot; adding another allocation/index is not justified by this evidence.
* Shared-read reference counts, pending-Fetch wake bookkeeping, and aligned physical buffers
  remain real costs. Most sampled allocator/atomic work also comes from array execution and
  must not all be attributed to IO.
* Local admission limits and delayed projection hints retain their previously measured defaults.
  Removing overhead does not justify accepting their known coalescing/latency regressions.

## Verification

The file library's 207 tests passed after the coalescer change. After lazy queue construction,
30 segment-source tests passed; the following 11 scan-IO tests passed with the added queue
assertions and ready-completion enabled. After concrete receiver storage, all 36 segment tests
passed, including both Fetch implementations with backend failure and channel cancellation.
Strict File Clippy with all targets/features passed after the final concrete-receiver change,
and the final release-debug benchmark build succeeded. Its executable is preserved as
`datafusion-final` before timing.

All 36 Python analyzer/harness tests, targeted Ruff checks, and targeted ty checks passed.
Pinned nightly formatting covers the changed Rust crates. No workspace-wide checks were run.
