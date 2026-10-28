<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Validate modelled S3 against AWS S3

The `S3 Model Validation` workflow runs DataFusion over all 22 TPC-H and 99 TPC-DS queries at
SF10. Real S3 and the RAM-backed model run sequentially on the same AWS machine in `us-east-1`,
with alternating backend order for each query/iteration. The bucket region and runner region
are checked before measurement. The workflow supports Vortex (default) and Parquet.

The default workflow validates a **frozen learned profile** against fresh real S3 measurements.
The long comparison job explicitly uses on-demand capacity. Measurement artifacts include the
GitHub run attempt so an infrastructure retry preserves earlier evidence without name conflicts.

The current profile, `profiles/us-east-1-c8gd-vortex.json`, was learned from 1,205,642 eligible
real GETs in three earlier CI runs. Its JSON records the source run IDs, input SHA-256 hashes,
hardware, region, coverage and calibration method. The first profile is preserved as
`profiles/us-east-1-c8gd-vortex-v1.json`; [measured results](RESULTS.md) distinguish the versions.

**[Latest full SF10 CI results](RESULTS.md):** version 4 passes 173/242 medians at the ±10%
target, with overall Spearman 0.9947 and exact requests/ranges/bytes/rows across all 121 queries.
It remains opt-in; real S3 stays the default reference.

The [24-round repeatability investigation](REPEATABILITY.md) separates real-S3 variation from
model error and describes the opt-in candidate. The default profile remains version 2 until
fresh CI validation establishes whether the candidate improves the fit.

Training includes partial TPC-H/TPC-DS runs; it establishes request
latency parameters, rather than full benchmark accuracy. Validation must use a different CI run,
and verifies that the measured model exactly matches the frozen parameters. No refitting happens
during this validation. Use the `recalibrate` workflow mode to collect initial/tuned pairs instead.

To learn another reusable profile from downloaded CI traces:

```bash
python3 benchmarks/s3-validation/learn.py \
  /tmp/ci-run-one/initial-tpch.jsonl /tmp/ci-run-two/initial-tpch.jsonl \
  --output benchmarks/s3-validation/profiles/my-profile.json
```

Learning requires at least two distinct real-S3 CI runs with matching region, format, CPU count
and memory within 1%. Query runtimes are excluded from fitting. Longer scans contribute more GET
observations; per-source sample counts are recorded so that weighting remains visible. The learner
rejects duplicate files/samples, local data and failed/incomplete GET measurements. To validate a profile:

```bash
python3 benchmarks/s3-validation/run.py --source-root "$REMOTE_ROOT" --phase validate \
  --profile benchmarks/s3-validation/profiles/my-profile.json --iterations 3
```

Data is generated once, uploaded to a unique run prefix, and preloaded from that prefix into
an immutable RAM snapshot. Both backends use the same URLs, object metadata, byte ranges,
file sizes and partitioning. The snapshot stays resident for both backends. Preloading checks
available RAM and reserves 60% for working memory. Only one suite's snapshot is resident at a time.

Validation uses fixed file partitions and disables opportunistic read coalescing, dynamic filter
pushdown, and Vortex predicate pushdown for both stores. DataFusion evaluates the filters after
reading the required columns. Query operators and intra-file scans retain parallel execution.
The normal scan path can issue different physical requests even between repeated real-backend
executions because speculative reads may be cancelled. `--adaptive-io` restores normal scan
behavior for diagnostics. A successful fixed-read fit does not establish equivalent behavior for
adaptive scans; that limits any recommendation to change the default CI storage backend.

Both backends also use the same per-sample cache for ranges in the final 4 MiB of each file,
including large footers. Concurrent identical reads share one request; completed reads remain
available until the sample ends. The cache resets before every sample, including warm samples.
This removes duplicate footer fetches caused by races in the database metadata cache. GET and
byte counters measure underlying operations after this cache; `get_calls` and `tail_cache_hits`
expose total logical GET calls and the calls served by the cache, respectively. The real backend still fetches every cache miss
from its source object store.

The workflow uses five iterations per query, with three for the execution-heavy TPC-DS query 72.
The binary accepts `--iteration-overrides 72=3`; the driver accepts a JSON map by suite, such as
`--iteration-overrides '{"tpcds":{"72":3}}'`. Reports record each query's declared sample count
and require at least three samples. Each iteration creates fresh DataFusion and Vortex sessions, registers tables, runs the
query cold, then immediately runs it warm. Query timers include planning and execution; table
registration has separate timings and request counters. The file catalog is prepared during registration.
Footer metadata and file-statistics caches are cleared after registration so schema inference does not
warm those caches for the cold sample. Prepared file listings stay in the catalog; this avoids
timing-dependent duplicate LIST calls from concurrent scans.
The S3 HTTP client stays alive across
samples within a batch. Cold therefore means fresh database caches, rather than cold TLS or S3 service caches.

TPC-DS runs as three sequential batches (1–71, 72, 73–99) on the same machine, against the same
uploaded objects. `run.py --queries` merges completed batches atomically and rejects different
host IDs, object ETags/sizes, profiles or configurations. Every query still runs at SF10. Logs
preserve each batch's command. The HTTP client and RAM snapshot are recreated between batches.
Checkpoints are uploaded before and after query 72, which takes about 25 minutes per execution
in the measured engine. The job budget is 12 hours for frozen-profile validation and 20 hours
for the optional two-phase recalibration run. These budgets avoid truncating the unchanged suite.

In recalibration mode, the initial deterministic model uses 25 ms per request, 100 MiB/s per request and no concurrency
cap. Sleeps use Tokio timers and response streams retain their semaphore permit until completed
or cancelled. Transfer waits use absolute deadlines to avoid accumulating a timer tick per chunk.
Calibration estimates TTFB from small S3 requests and throughput from large S3 requests, preferring
low concurrency. It also learns 1,000 equally weighted latency quantiles from real GET durations
after subtracting the transfer cost. Separate distributions cover requests up to 4 KiB, 64 KiB,
256 KiB and 1 MiB. Sparse classes fall back to the overall distribution. Calibration excludes
paired samples whose model timers overshot by more than 5 ms at the median, so execution stalls
do not become service latency. Partial paired runs also exclude unmatched samples. This preserves
the long tail that parallel scans wait for.
The extreme 0.05% at each end is represented by the nearest quantile. Programmed delays account
for measured model timer overhead, which is reported separately. Draws use seed 20261027 and a
stable hash of the operation, dataset-relative object path, range, measurement number and per-key request ordinal,
so concurrent task polling order and run-specific S3 prefixes do not choose the delays. HEAD/LIST use the scalar TTFB.
`--model-parameters parameters.json` loads the learned distribution. Traces include each
programmed request delay. Calibration uses request measurements without adjusting query results
or applying per-query runtime factors. A second complete paired run tests these parameters
against fresh S3 samples on the same runner.

The Python driver refreshes a private GitHub OIDC token file every two minutes. The object_store
web-identity provider renews AWS credentials from that file, so a suite can exceed one STS session.
This uses the workflow's existing benchmark role; tokens are kept outside the result artifacts.

Artifacts include raw JSONL with per-request traces, query samples, initial/tuned reports, and
`parameters.json`. The workflow uploads a checkpoint after each suite and around query 72 so its evidence is available
while the next suite runs. Each sample records GET/PUT/LIST/HEAD counts, consumed bytes, errors, peak and
time-averaged in-flight requests, plus registration counts. These are object-store API metrics;
HTTP retries inside the S3 client are not separately instrumented. The current snapshot is flat
and smaller than 1,000 objects, so each LIST is a single S3 page.

The report requires all samples, exact request/range/byte/row-count agreement, every median
model/S3 ratio within 0.7–1.3, and Spearman ≥0.95 for both cache states in both suites. A workflow
can successfully collect evidence while recommending against making the model the CI default.
Query failures and request mismatches are invalid comparisons, not latency calibration targets.

```bash
gh workflow run s3-model-validation.yml --ref ji/ci-s3-model-validation -f iterations=5 -f format=vortex -f mode=learned
```

To separate reference variance from model error, use the focused repeatability experiment:

```bash
gh workflow run s3-model-validation.yml --ref ji/ci-s3-model-validation -f iterations=24 -f mode=repeatability
```

This measures TPC-H 1,2,6,15 and TPC-DS 1,20,31,41,61,80,83,92 at SF10. Each query/round has
independent real S3 controls plus the frozen model, in all six backend orders. Repetitions visit
all selected queries before starting the next round, rotating query positions and spreading samples over time. Cold/warm
semantics, objects, hardware and request checks are unchanged. Request start times and query
positions, response-header times and body-consumption times are recorded. These focused diagnostics supplement full-suite validation; they do not
replace its 121-query coverage.

The repeatability report retains the fixed ratio flags and compares model/real absolute log
errors with real/real errors. Paired circular blocks of consecutive rounds provide exploratory
95% confidence intervals for median bias and excess error over the S3 control. An interval
including one is inconclusive, rather than proof of equivalence. Query runtimes are diagnostic
outcomes and remain excluded from profile learning.

The scan accuracy target is ±10% against the pooled real-control median. Reports distinguish
the point estimate from whether its full bootstrap interval fits inside that band. The suites
measure end-to-end query medians, which serve as proxies for read-heavy scans. Full-suite
reports also flag this target against their real-S3 query medians. Historical ±30% flags remain
separate, and request/range/byte/row equality remains exact.

A candidate can be compared directly with a frozen baseline using the dispatch input
`baseline_profile` (or `run.py --baseline-profile`). Both models then run beside the two real
controls on the same host; 24 rounds visit all 24 backend orders. The paired bootstrap also
reports the change in absolute log error from the old model to the candidate. Both frozen
profiles are verified against their measured configurations. The comparison run must be excluded
from both profiles' training data.

`learn.py --preserve-tail` produces an opt-in version 3 profile with cumulative probabilities.
The bulk uses up to 1,000 buckets, while each observation in the upper 0.1% (rounded up) retains
its observed probability. This avoids the old quantile table's truncation of very rare stalls.
It uses the same request-size classes and request-derived transfer cost, without query runtime
factors. Empty probability arrays retain the legacy sampler and serialized profile shape.
The default checked-in version 2 profile remains unchanged until a candidate is validated in CI.

`learn.py --request-phases` produces a version 4 candidate from traces that also record
`response_ms`. It samples header and body residual delays jointly, preserving their observed
relationship and sparse-tail probabilities. Header arrival and body transfer use separate
deadlines. Calibration subtracts measured RAM timer and body-consumer overhead, and estimates
transfer cost from the measured body phase. Every source must contain response-header timings;
older traces cannot establish these phases. This remains an opt-in model requiring fresh CI
validation before promotion.

The optional dispatch input `source_warmup_seconds` (or driver flag `--source-warmup-seconds`)
primes S3 directly with 32 concurrent 64 KiB range reads before database sessions or query timers.
It records actual warmup time, bytes and requests separately. Database cold/warm cache semantics
are unchanged. The default remains zero for compatibility; the candidate experiment uses an
explicit warmup to separate initial source startup from steady performance. Request metrics for
queries exclude this common setup phase.

## Four measurements over time

The separate `S3 Model Over Time` workflow retains one on-demand `c8gd.16xlarge` host and the
same immutable SF10 objects for four windows at **0, 3, 6 and 9 hours** after preparation. It
stops after the fourth window. `over_time.py` anchors starts to the initial time, rather than
waiting three hours after each measurement finishes. A window more than five minutes late is
reported as failed; measurements are never silently compressed into a catch-up burst.

Each window runs TPC-H 1,2,6,13,15,22 and TPC-DS 1,4,13,20,30,31,41,61,63,80,83,92 at SF10,
including the earlier diagnostic set and observed full-suite outliers. Twelve rounds balance
all six orders of two independent S3 controls and the frozen version 4 RAM model. Cold/warm
samples remain separate, with a recorded 20-second source-only warmup per suite. This focused
study supplements the full 121-query validation, whose runtime exceeds the three-hour cadence.

The workflow checks unchanged host identity, binary/profile hashes, object metadata and exact
request/range/byte/row fingerprints across windows. Checkpoint artifacts preserve each window.
The final CSV/JSON reports include per-query median/p90, bootstrap intervals, S3 and model drift
relative to the first successful window, request header/body metrics, correlation and ±10% flags.
These end-to-end query measurements remain proxies for scan performance.

After four successful windows, the workflow combines their request traces with an earlier
independent CI run to generate `learned-candidate.json`. Window identifiers prevent sample-ID
collisions without treating windows as distinct CI runs. Provenance records actual run IDs,
window times, file hashes and observation weighting; query runtimes remain excluded from fitting.
The candidate needs fresh held-out CI validation and does not replace the frozen profile during
the time study. Existing two-independent-run and held-out-validation requirements remain in place.

```bash
gh workflow run s3-model-over-time.yml --ref ji/ci-s3-over-time
```

This workflow has a separate concurrency group and does not cancel the full-suite validation.

`learn.py --exclude-startup` excludes the first real timed sample from each identifiable unprimed
process, regardless of query label or runtime, and records the exclusions. Legacy inputs without
process timestamps can identify only their initial process. Primed inputs retain every sample.
This targets a steady-service profile while preserving rare stalls elsewhere in the traces.

For local smoke testing, generate a small dataset and explicitly allow a filesystem reference.
This tests the harness and never qualifies as S3 validation:

```bash
cargo build -p datafusion-bench --bin s3-validate -p vortex-bench --bin data-gen --profile release_debug
target/release_debug/data-gen tpch --formats vortex --opt scale-factor=0.01
target/release_debug/s3-validate tpch --allow-local \
  --source "file://$PWD/vortex-bench/data/tpch/0.01/" --scale-factor 0.01 \
  --queries 1,6,14 --iterations 1 --output /tmp/s3-validation-smoke.jsonl
python3 -m unittest discover -s benchmarks/s3-validation -p 'test_*.py'
```

Review high-concurrency scans, small-request workloads, throttling and latency tails before
generalizing a successful fit to another engine, format, scale or runner. The simulator does not
currently model shared bandwidth, retry/503 behavior or connection startup.
