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

The [fresh candidate comparison](https://github.com/vortex-data/vortex/actions/runs/40591491612)
adds a recorded 20-second S3-only warmup outside query timers, retaining fresh database caches
for cold samples. Both frozen models and both real controls ran in all 24 backend orders.
This run is excluded from version 3 training. All 2,304 samples match requests, ranges, bytes
and rows; each backend made 605,832 GETs and read 165,408,534,792 bytes.

Using the requested ±10% target against pooled S3 medians, version 3 passes **18/24 groups**
versus **15/24 for version 2** on that host. Paired log-error improvement intervals favor version 3
in one group and version 2 in two, with the others inconclusive. The
[per-query table](results/model-comparison-40591491612.csv) and
[summary](results/model-comparison-summary-40591491612.json) retain both the point estimates and
uncertainty. These are end-to-end query medians, serving as proxies for read-heavy scans.

Version 3 reproduced 23-second and 10-second stalls in TPC-H, preserving observed tail mass
but increasing paired error when S3 did not stall in those rounds. A persistent structural
problem remains: TPC-H model headers average 29.9 ms versus 27.0 ms for S3, while model body
consumption averages 1.6 ms versus 2.9 ms. In TPC-DS the corresponding figures are 30.0/29.0 ms
for headers and 2.2/4.0 ms for bodies. With warmup, the first real TPC-H cold scans took about
6 seconds; the large initial burst seen in the earlier run was absent.

The frozen [version 4 profile](profiles/us-east-1-c8gd-vortex-v4.json) learns header/body residual
pairs jointly from **2,420,879 eligible real GETs** in runs 40573067681 and 40591491612. Those
are the two runs with measured response-header times; older traces cannot establish the phases.
Joint sampling preserves phase relationships and rare observations, with separate request-size
classes. Measured RAM timer and body-consumer overhead is subtracted. Two initial unprimed
real measurements are excluded by process order, and no samples are excluded for excessive
model timer overshoot. Query runtimes remain excluded from learning.

Version 4 uses **25.504 ms scalar TTFB**, **147.414 MiB/s per-request transfer cost**, no
concurrency cap, and seed 20261027. GETs use the joint distributions; the scalar TTFB applies
to HEAD/LIST. Version 2 remains the default profile pending fresh full-suite validation of
version 4. Its validation run must be excluded from these training IDs.

Baseline raw artifact: [12772979778](https://github.com/vortex-data/vortex/actions/runs/40573067681/artifacts/12772979778),
SHA-256 `fc80fd8503f5fcaae167d0b75be9f4fd6e88e6bad69f582e27a8709148448329`.
Candidate SHA-256: `7ecb584d4e7c92ccb0b4156ca325f8735be85d086193446697acf2965bcb79d5`.
Candidate comparison artifact: [12782523852](https://github.com/vortex-data/vortex/actions/runs/40591491612/artifacts/12782523852),
SHA-256 `be4e805a5d788e303becd08da2e69d7dbcd0531b8c39330fb6b75175f2f43668`.
Version 4 SHA-256: `e365685a374cba7cb7939042067f0830d46ea64f0278c417f9c796b569d50fc0`.

Checks: 20 Rust tests, 29 Python tests, scoped Clippy, pinned formatting, Ruff, strict YAML lint,
and local CLI tests of both profiles, warmup counters and exact probability serialization.
Local filesystem tests verify the harness and do not count as S3 performance evidence.
