<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# S3 variance and model error

**Both contribute.** CI controls show substantial variation in the real storage path, while
the model has small directional biases and truncates very rare request stalls. These observations
do not identify whether a real delay originates in S3, the network or the SDK.

[Run 40573067681](https://github.com/vortex-data/vortex/actions/runs/40573067681) measured 24 rounds
of 12 selected SF10 queries, with two independent real S3 controls and the unchanged version 2
model on one `c8gd.16xlarge` host. All six backend orders and rotating query positions spread
measurements over time. The 1,728 cold/warm samples matched requests, ranges, bytes and rows
exactly; each backend made 605,832 GETs. Independent recomputation matched every CI report field.

The [per-query table](results/repeatability-40573067681.csv) and
[summary](results/repeatability-summary-40573067681.json) contain the results and provenance.
Across 24 query/cache groups, model/real median ratios were **0.874–1.127** and p90 ratios
**0.785–1.118**. Real-control median ratios were **0.906–1.087**. No group's paired block-bootstrap
95% interval established excess model absolute log error above real-control error. Some median
bias intervals exclude one; this is not proof of equivalence, and the intervals are exploratory
without a multiple-comparison adjustment. These selected queries supplement the
[full 121-query validation](RESULTS.md).

Real performance also changes between CI runs. Object sizes and ETags match the earlier full run
40477576062, and the model parameters are byte-identical:

| Query/cache | Real median, new/earlier CI | Model median, new/earlier CI |
| --- | ---: | ---: |
| TPC-H 15 cold | 1.155 | 1.002 |
| TPC-H 15 warm | 1.167 | 1.005 |
| TPC-DS 61 warm | 1.116 | 0.966 |

The earlier q61 warm flag fell from 1.302× to 1.127× with no model changes. Conversely, a real
q31 GET in the earlier run took 23.149 seconds; version 2's programmed delays stop around
270 ms. First-source scans also show clusters of slow requests: the first real TPC-H cold scan
took 14.5 seconds, including 268 responses whose headers took over 100 ms.

The opt-in [version 3 candidate](profiles/us-east-1-c8gd-vortex-v3.json) learns 2,549,116 eligible
real GETs from both CI runs. It retains each observation in the upper 0.1% at its observed
probability, including the rare stall, rather than truncating the tail to uniform quantiles.
Parameters are 25.769 ms scalar TTFB, 142.297 MiB/s per-request transfer rate, no concurrency cap,
and seed 20261027. Request-size distributions remain separate. Query runtimes are excluded from
learning. Four initial real measurements are excluded by process order, independent of query
label or runtime; legacy inputs identify only their initial process. Six samples with excessive
model timer overshoot are also excluded.

The candidate experiment adds a recorded 20-second S3-only warmup outside query timers, while
retaining fresh database caches for cold samples. Both frozen models and both real controls run
on the same fresh CI host in all 24 backend orders. Its run must be excluded from training.
Version 2 remains the default profile pending held-out candidate and full-suite validation.

Baseline raw artifact: [12772979778](https://github.com/vortex-data/vortex/actions/runs/40573067681/artifacts/12772979778),
SHA-256 `fc80fd8503f5fcaae167d0b75be9f4fd6e88e6bad69f582e27a8709148448329`.
Candidate SHA-256: `7ecb584d4e7c92ccb0b4156ca325f8735be85d086193446697acf2965bcb79d5`.

Checks: 17 Rust tests, 26 Python tests, scoped Clippy, pinned formatting, Ruff, strict YAML lint,
and local CLI tests of both profiles, warmup counters and exact probability serialization.
Local filesystem tests verify the harness and do not count as S3 performance evidence.
