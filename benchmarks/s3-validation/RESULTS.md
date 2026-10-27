<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Learned S3 model: validation evidence

**Provisional:** the first learned profile passed TPC-H and most completed TPC-DS comparisons,
but two ratios missed the band and the job timed out during q72. Full-suite validation has not
passed. Keep real S3 as the reference until the revised profile completes validation.

The first [profile](profiles/us-east-1-c8gd-vortex-v1.json) was learned from **519,095 real GETs**
in two separate AWS CI runs:
[40418787500](https://github.com/vortex-data/vortex/actions/runs/40418787500) and
[40423433159](https://github.com/vortex-data/vortex/actions/runs/40423433159).
These are partial TPC-H SF10 runs with five iterations per completed query. Their real request
measurements remain usable for learning despite cancellation of the overall workflows.
The profile contains input hashes, query coverage and source metadata. Query runtimes are
excluded from fitting.

| Parameter | Learned value |
| --- | ---: |
| Scalar TTFB | 26.473 ms |
| Transfer rate per request | 126.042 MiB/s |
| Concurrency cap | Unlimited |
| Empirical delay distribution | 1,000 quantiles |
| Seed | 20261027 |

The constant 25 ms / 100 MiB/s baseline was too fast. In run 40423433159, the completed TPC-H
queries 1–11 had model/S3 median ratios of **0.529–0.693**. All 22 completed query/cache groups
matched physical request counts, ranges, bytes and result row counts. The learned profile
preserves real latency tails that the constant model omitted.

An independent request-level check used **62,077 GETs** from
[40427104780](https://github.com/vortex-data/vortex/actions/runs/40427104780), which was excluded
from training:

| Request duration | Observed S3 | Profile prediction |
| --- | ---: | ---: |
| Median | 28.268 ms | 28.727 ms |
| p90 | 40.523 ms | 44.071 ms |
| p99 | 100.928 ms | 92.302 ms |
| Mean | 31.427 ms | 32.074 ms |

Predictions apply the frozen distribution and transfer cost to that run's request sizes, using
the same seeded sampler as the RAM store and the learned timer overhead. They are analytic
request-duration predictions, rather than measured RAM-backed query runtimes.

The attempted full query-level validation was
[40428937016](https://github.com/vortex-data/vortex/actions/runs/40428937016), commit
`5dcb2b7feeb40424ec828e01d4dee16aceca7a01`. It uses the frozen profile without refitting,
paired real/model measurements on one `c8gd.16xlarge` host in `us-east-1`, all 22 TPC-H and
99 TPC-DS queries at SF10, and three iterations with separate cold/warm results.
The complete [TPC-H per-query comparison](results/tpch-40428937016.csv) contains medians, p90s,
ratios, request counts, bytes and concurrency metrics for all 44 query/cache groups. Every
paired iteration matched physical request counts, ranges, bytes and result row counts.

| Suite/cache | Median ratio range (model/S3) | Spearman | Ratios outside 0.7–1.3 |
| --- | ---: | ---: | ---: |
| TPC-H cold | 0.961–1.279 | 0.973 | 0 / 22 |
| TPC-H warm | 0.994–1.235 | 0.993 | 0 / 22 |
| TPC-DS cold, q1–71 | 0.591–1.232 | 0.983 | 1 / 71 |
| TPC-DS warm, q1–71 | 0.781–1.327 | 0.982 | 1 / 71 |
| All completed cold | — | 0.991 | 1 / 93 |
| All completed warm | — | 0.991 | 1 / 93 |

The median of all 44 runtime ratios is 1.126: the learned model is about 13% slower. It passes
the TPC-H median-runtime gate without fitting to this validation run.

The [partial TPC-DS table](results/tpcds-partial-40428937016.csv) covers completed queries 1–71.
Every paired sample matched physical requests, ranges, bytes and result row counts. Its median
ratio is 1.069. The outliers are q13 warm (1.327) and q41 cold (0.591). Query 41 is short:
real/model medians were 229.3/135.6 ms. Its real cold samples were 265.4, 229.3 and 130.3 ms,
illustrating the uncertainty in a three-sample reference.

Query 72 took 1,462–1,625 seconds per execution on both backends. The job reached its configured
six-hour limit after nine of twelve required samples, so q72–99 are not validated. The first
real cold sample took 1,524 s with 345 s of summed request lifetimes; at least 1,179 s occurred
outside object-store requests. Local SF1 runs with zero latency took 12–14 s with either fixed
or adaptive scans. Query execution dominates. The next run preserves the SQL and SF10 workload,
increases the job budget and measures q72 in a separate sequential step on the same host.

Tail prediction remains imperfect. Query 1 cold has a real p90 of 10.476 s versus 6.544 s in
the model. Its first real iteration took 11.741 s, versus 5.417 s and 5.361 s subsequently, with
identical GET counts. Mean request duration was 48.86 ms in that first iteration, versus
30.31/30.42 ms subsequently. Independent per-request samples do not reproduce such coherent
latency changes. Three iterations also provide limited information about a query's tail.

The [revised profile](profiles/us-east-1-c8gd-vortex.json) learns 1,205,642 eligible GETs from
three runs, including these retained measurements. It uses 26.036 ms scalar TTFB, 131.728 MiB/s,
and separate 1,000-quantile distributions for request-size classes. Four samples with stalled
model timers and one unmatched sample were excluded from service-latency training. Dataset-relative
paths keep seeded delays independent of unique CI prefixes. The revised profile requires a fresh
CI run excluded from training; the preceding results do not validate it.

The controlled validation profile disables adaptive coalescing, repartitioning and predicate
pushdown, while retaining parallel operators and intra-file scans. Both backends use identical
per-sample footer caching. A successful fit under these settings would support this controlled
benchmark configuration; it would not establish equivalence for normal adaptive SQL scans.
