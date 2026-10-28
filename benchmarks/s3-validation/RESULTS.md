<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Version 4: full SF10 validation

**Keep real S3 as the default.** The learned model passes **173/242** query/cache medians at the
requested ±10% target. Overall Spearman is **0.9947 cold / 0.9947 warm**, and every iteration
matches requests, ranges, bytes and result rows exactly. This supports opt-in simulation while
retaining a real S3 reference; it does not establish accuracy for native adaptive SQL scans.

[Run 40613168535](https://github.com/vortex-data/vortex/actions/runs/40613168535) completed all
22 TPC-H and 99 TPC-DS SF10 queries, five iterations each except three for TPC-DS q72. Both
backends ran sequentially on one on-demand `c8gd.16xlarge` in `us-east-1`. Cold and immediate warm
results are separate. All **2,412 samples** were independently recomputed from the raw traces.

The [per-query comparison](results/comparison-40613168535.csv) includes median/p90, counts, bytes
and concurrency. The [summary](results/summary-40613168535.json) retains all 69 ±10% flags.

| Suite/cache | Within ±10% | Model/S3 median range | Spearman |
| --- | ---: | ---: | ---: |
| tpch/cold | 21/22 | 0.897–1.097 | 0.9966 |
| tpch/warm | 20/22 | 0.867–1.106 | 0.9887 |
| tpcds/cold | 73/99 | 0.837–1.263 | 0.9914 |
| tpcds/warm | 59/99 | 0.919–1.347 | 0.9913 |
| all/cold | 94/121 | 0.837–1.263 | 0.9947 |
| all/warm | 79/121 | 0.867–1.347 | 0.9947 |

Median absolute ratio error is **6.25%**. Under the separate historical ±30% criterion,
241/242 medians pass. TPC-DS q30 warm is the sole flag: **1.3467×**, with S3/model medians
0.494/0.665 seconds. Eleven p90 comparisons fall outside ±30%. Five samples (three for q72)
limit tail estimates. q72 passes ±10% at 1.048 cold / 1.030 warm.

Each backend made **1,368,393 measured GETs** and consumed **380,744,124,525 bytes**, with no
query/store errors. All 32 object sizes/ETags match the earlier full validation. Counts describe
object-store API calls; internal HTTP retries and recovered 503s are not separately instrumented.

The frozen [version 4 profile](profiles/us-east-1-c8gd-vortex-v4.json) learns **2,420,879 eligible
real GETs** from CI runs 40573067681 and 40591491612, excluding this validation run. It samples
header/body residual pairs jointly within request-size classes, preserving sparse-tail probability.
Scalar HEAD/LIST TTFB is **25.504 ms**, per-request transfer cost **147.414 MiB/s**, concurrency
cap unlimited, seed **20261027**. Query runtimes are excluded from fitting. Profile SHA-256:
`e365685a374cba7cb7939042067f0830d46ea64f0278c417f9c796b569d50fc0`.

TPC-H header means are 28.01 ms model / 28.18 ms S3, with body means 3.81/3.43 ms.
TPC-DS header means are 29.63/30.21 ms, with body means 3.92/3.41 ms. Phase allocation improved,
but matching marginal request timings does not guarantee each query’s critical path. The earlier
version 2 run passed 136/242 medians at ±10%; different CI windows and warmup protocols prevent
treating that comparison as a controlled estimate of improvement.

Both S3 variance and model error remain plausible for individual flags. The same-host
[four-window study](https://github.com/vortex-data/vortex/actions/runs/40661847573) pairs two S3
controls with frozen version 4 at 09:18, 12:18, 15:18 and 18:18 UTC on 2026-10-28. It remains
in progress and will generate a new request-derived candidate after the fourth window. Native
adaptive scans, shared bandwidth, temporal correlation and retry behavior still need validation.

The controlled protocol disables opportunistic coalescing, repartitioning and predicate/dynamic
filter pushdown, retains parallel operators and scans, and resets database caches for cold samples.
A recorded 20-second S3-only warmup occurs outside query timers. End-to-end query medians are
proxies for scans, not isolated scan-operator timings. Neither the ±10% target nor the historical
all-query gate passes, so the simulator remains opt-in and version 2 remains the default *profile*.

Raw [artifact 12807015480](https://github.com/vortex-data/vortex/actions/runs/40613168535/artifacts/12807015480),
SHA-256 `7ec3904649c695575d39937342c73a2edf7249e287b77a2161788b8d5529f851`.
CI cleaned up its temporary S3 objects. Checks: 20 Rust tests, 33 Python tests, scoped Clippy,
pinned formatting, Ruff, strict YAML and the complete same-host AWS comparison.

---

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Version 2: historical full SF10 validation

**Recommendation: keep real S3 as the default reference.** The learned RAM model closely
predicts median runtimes in the controlled benchmark: **241 of 242 comparisons** fall within
0.7–1.3×, with overall Spearman correlations of **0.992 cold / 0.993 warm**. It misses the strict
all-query gate by one comparison, and query tails remain less accurate. Use it for repeatable
median regression experiments in this configuration; it has not validated normal adaptive SQL scans.

[CI run 40477576062](https://github.com/vortex-data/vortex/actions/runs/40477576062) completed
all **22 TPC-H and 99 TPC-DS queries at SF10**, commit
`7bac614d58bb6e7cd48eb2d2bac44ca6f42d9fa3`. Both backends ran on the same on-demand
`c8gd.16xlarge` instance, `i-0be42c9d4d298a70e`, in `us-east-1`, with identical Vortex objects.
Five paired iterations per query, with three for TPC-DS q72, produced **2,412 samples**
with separate cold/immediate warm results.
The TPC-H/TPC-DS snapshots were 2.57/2.70 GiB, measured separately with 123.53 GiB host RAM
and about 119 GiB initially available.

The [per-query comparison table](results/comparison-40477576062.csv) records median/p90 runtimes,
ratios, GET/PUT/LIST/HEAD counts, bytes, peak/average concurrency and request equality for every
query/cache pair. The [summary](results/summary-40477576062.json) records provenance. Local recomputation
from raw traces matched the CI report exactly.

| Suite/cache | Median ratio range (model/S3) | Spearman | Median ratios outside 0.7–1.3 |
| --- | ---: | ---: | ---: |
| TPC-H cold | 0.958–1.154 | 0.982 | 0 / 22 |
| TPC-H warm | 0.984–1.168 | 0.990 | 0 / 22 |
| TPC-DS cold | 0.726–1.267 | 0.987 | 0 / 99 |
| TPC-DS warm | 0.749–1.302 | 0.988 | 1 / 99 |
| All cold | 0.726–1.267 | 0.992 | 0 / 121 |
| All warm | 0.749–1.302 | 0.993 | 1 / 121 |

The median of the 242 ratios is **1.078**. Every paired iteration matched request counts,
range/byte traces and result row counts, including registration counters. Each backend made
**1,368,393 measured GETs** and consumed **380,744,124,525 bytes**, with zero query/store errors.
These count underlying object-store API calls after the identical per-sample footer cache;
HTTP retries inside the S3 SDK are not counted separately.

The sole median flag is **TPC-DS q61 warm: 1.302459×**, real/model medians 2.409/3.138 s.
Real request p90s were 35.4–37.5 ms versus 42.5–44.1 ms for the model: its tails were heavier
than this S3 window. The threshold was not relaxed. The report has `adequate=false` and
`default_ci_recommended=false`.

TPC-DS q72 completed all twelve executions with ratios **0.970 cold / 0.990 warm** and exact
requests. Its 25–28 minute executions required sequential batches and a twelve-hour job budget.
Every original query ran at SF10 without SQL changes.

The [frozen profile](profiles/us-east-1-c8gd-vortex.json) learns **1,205,642 eligible real GETs**
from three earlier CI runs:
[40418787500](https://github.com/vortex-data/vortex/actions/runs/40418787500),
[40423433159](https://github.com/vortex-data/vortex/actions/runs/40423433159), and
[40428937016](https://github.com/vortex-data/vortex/actions/runs/40428937016).
The validation run is excluded from training. Query runtimes are never fitting targets;
there are no per-query runtime multipliers.

| Final parameter | Value |
| --- | ---: |
| Scalar TTFB, used for HEAD/LIST | 26.036 ms |
| Transfer rate per request | 131.728 MiB/s |
| GET delay distributions | 1,000 quantiles per request-size class |
| Size classes | ≤4 KiB, ≤64 KiB, ≤256 KiB, ≤1 MiB; larger requests use global fallback |
| Concurrency cap | Unlimited; observations do not establish a service-side cap |
| Random seed | 20261027 |

GET distributions subtract transfer cost and measured timer overhead. Four samples with
median timer overshoot above 5 ms and one unmatched sample were excluded. Tokio sleeps overlap;
seeded draws use dataset-relative request keys. Input hashes and coverage are recorded.
Its SHA-256 is `2aa62e3fe8486e3183d6b92de095a5429f22e3262242202891adb7b4400469fe`;
CI's parameter file matches it byte-for-byte.

The constant 25 ms / 100 MiB/s baseline was too fast: completed TPC-H q1–11 in run 40423433159
had ratios 0.529–0.693. Learning empirical latency tails corrected that underestimate. The current
profile adds request-size classes to the [first learned profile](profiles/us-east-1-c8gd-vortex-v1.json).

Query p90s miss the band in **17 of 242 comparisons**: 13 cold and four warm. TPC-DS q31 cold
has real/model p90s of 20.303/8.011 s despite a median ratio of 1.093. Inspection of that trace
found one GET taking 23.149 seconds in the last real cold iteration, with 23.145 seconds until
first bytes. The equally weighted quantile table truncates such rare events. Other runs also
show clusters of slow requests during source startup. These end-to-end SDK timings do not
identify whether the delay originates in S3, the network or the client. Three to five iterations
also limit tail estimates; rare stalls and correlated latency require additional held-out runs.

The controlled profile disables predicate/dynamic-filter pushdown, repartitioning and
opportunistic coalescing while retaining parallel query operators and intra-file scans.
Both backends have the same per-sample footer cache. Native adaptive scans can issue different
requests even between repeated real runs; this validation does not establish their equivalence.
Shared bandwidth limits, connection startup, throttling/503s and SDK retries are also not
explicitly modelled. Keep periodic real S3 measurements for those behaviors and for tail validation.

Raw traces and reports are in
[artifact 12756048150](https://github.com/vortex-data/vortex/actions/runs/40477576062/artifacts/12756048150),
SHA-256 `76d9daae89092653ea8d88874aa37fe8965cd0fdcf4c7a95fb4906ba2c4fb22c`.
CI refreshed credentials throughout the long run and successfully removed its temporary S3 objects.

Validation: eight Rust store tests, sixteen Python tests, scoped Clippy, pinned Rust formatting,
Ruff, strict YAML lint, local chunk/CLI smoke tests, and the complete AWS CI comparison.
