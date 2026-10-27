<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Validate modelled S3 against AWS S3

The `S3 Model Validation` workflow runs DataFusion over all 22 TPC-H and 99 TPC-DS queries at
SF10. Real S3 and the RAM-backed model run sequentially on the same AWS machine in `us-east-1`,
with alternating backend order for each query/iteration. The bucket region and runner region
are checked before measurement. The workflow supports Vortex (default) and Parquet.

The default workflow validates a **frozen learned profile** against fresh real S3 measurements.
The current profile, `profiles/us-east-1-c8gd-vortex.json`, was learned from 1,205,642 eligible
real GETs in three earlier CI runs. Its JSON records the source run IDs, input SHA-256 hashes,
hardware, region, coverage and calibration method. The first profile is preserved as
`profiles/us-east-1-c8gd-vortex-v1.json`; [measured results](RESULTS.md) distinguish the versions.
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
