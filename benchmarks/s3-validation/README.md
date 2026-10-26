<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Validate modelled S3 against AWS S3

The `S3 Model Validation` workflow runs DataFusion over all 22 TPC-H and 99 TPC-DS queries at
SF10. Real S3 and the RAM-backed model run sequentially on the same AWS machine in `us-east-1`,
with alternating backend order for each query/iteration. The bucket region and runner region
are checked before measurement. The workflow supports Vortex (default) and Parquet.

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

Each of five iterations creates fresh DataFusion and Vortex sessions, registers tables, runs the
query cold, then immediately runs it warm. Query timers include planning and execution; table
registration has separate timings and request counters. The file catalog is prepared during registration.
Footer metadata and file-statistics caches are cleared after registration so schema inference does not
warm those caches for the cold sample. Prepared file listings stay in the catalog; this avoids
timing-dependent duplicate LIST calls from concurrent scans.
The S3 HTTP client stays alive across
samples. Cold therefore means fresh database caches, rather than cold TLS or S3 service caches.

The initial deterministic model uses 25 ms per request, 100 MiB/s per request and no concurrency
cap. Sleeps use Tokio timers and response streams retain their semaphore permit until completed
or cancelled. Transfer waits use absolute deadlines to avoid accumulating a timer tick per chunk.
No randomness is used. Calibration estimates TTFB from small S3 requests and throughput from
large S3 requests, preferring low concurrency. The programmed delay accounts for the measured model timer overhead, which is reported separately. It uses request traces, without adjusting query
results or applying per-query runtime factors. A second complete paired run tests these parameters
against fresh S3 samples on the same runner.

Artifacts include raw JSONL with per-request traces, query samples, initial/tuned reports, and
`parameters.json`. Each sample records GET/PUT/LIST/HEAD counts, consumed bytes, errors, peak and
time-averaged in-flight requests, plus registration counts. These are object-store API metrics;
HTTP retries inside the S3 client are not separately instrumented. The current snapshot is flat
and smaller than 1,000 objects, so each LIST is a single S3 page.

The report requires all samples, exact request/range/byte/row-count agreement, every median
model/S3 ratio within 0.7–1.3, and Spearman ≥0.95 for both cache states in both suites. A workflow
can successfully collect evidence while recommending against making the model the CI default.
Query failures and request mismatches are invalid comparisons, not latency calibration targets.

```bash
gh workflow run s3-model-validation.yml --ref ji/ci-s3-model-validation -f iterations=5 -f format=vortex
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
