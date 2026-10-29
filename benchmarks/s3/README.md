<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# S3 measurements

`s3-measure` collects real S3 request observations independently of query engines and simulators.
The read-only probe sweeps 4 KiB, 64 KiB, 256 KiB, 1 MiB and 8 MiB ranges at concurrency 1, 4, 16
and 64. Three rounds use 64 requests per size/concurrency cell, with shuffled cell order and
seeded object/range selection. Each concurrency uses identical ranges so load is the changed input.

```bash
cargo build -p vortex-bench --bin s3-measure --profile release_debug
python3 benchmarks/s3/measure.py --source s3://bucket/prefix/ --count 4 --interval-seconds 10800
```

Objects must already exist; the collector never writes to S3. At least one must be 8 MiB or larger
for the default sweep. The wrapper anchors four windows to 0, 3, 6 and 9 hours, then stops. It
retains a SHA-256 for each trace and the binary, checks identical runner/object/probe identities,
and preserves partial evidence on failure. Each request has a deadline; a window has a 30-minute
budget. Starts over five minutes late are failures, without silently shifting the remaining schedule.

The manual `S3 Measurements` workflow keeps one on-demand `c8gd.16xlarge` in `us-east-1` for the
whole study. It creates four seeded 16 MiB fixtures in a unique prefix, verifies bucket/runner
region, collects one to four windows, uploads evidence and removes only its fixtures. Branch pushes
run one window to exercise the collector. Authentication uses the same pinned
`aws-actions/configure-aws-credentials` action as the S3 benchmarks. CI waits for each fixed start,
then obtains fresh credentials before collecting that window. Python inherits the configured
credentials; it does not fetch tokens, assume roles or write credential files. Standalone multi-window
runs use the caller's standard AWS credential provider, which must remain valid for the study.

Each JSONL starts with a **schema version 1 environment**: actual CI run/attempt and commit,
measurement window ID, source, hardware/region/availability zone, CPU/memory, seed, probe configuration,
and the complete object size/ETag/version manifest. It also identifies local smoke tests and custom
endpoints so a learner can reject them as real-S3 training inputs. Credentials are excluded.

Request records contain operation, object/range, observed identity, monotonic and wall-clock start,
configured/actual in-flight concurrency, consumed bytes, response-header time, first-body-byte time,
completion time, status and errors. A 20-second source warmup is explicitly labelled and excluded
from the measured sweep. GET headers and body completion are separate observations; a learner can
estimate size-dependent transfer cost, latency tails and concurrency effects without query runtimes.
Partial/failed/timed-out reads remain visible and must not become latency samples.

Cell summaries retain duration, bytes, errors, peak/time-averaged concurrency and achieved aggregate
throughput. `measurements.csv` adds per-round median/p90 header, first-byte, body and total timings.
Every successful window has a terminal completion record and exact declared cell/request coverage.
Object replacement is an error, guarded by ETag preconditions and response metadata checks.

Counts and timings cover **SDK operations**. The SDK may retry internally; those delays appear in
the operation duration, while individual HTTP attempts and recovered 503s are not separately counted.
The catalog LIST record covers the listing stream, including pagination. HEAD/LIST catalog setup is
labelled separately. The measured workload issues no PUTs. These files provide request-level model
inputs; they do not establish accuracy for SQL, random-access or decompression benchmarks.

Local harness exercise, explicitly ineligible as S3 training data:

```bash
target/release_debug/s3-measure --source "file://$PWD/fixtures/" --allow-local \
  --output /tmp/local-s3-measure.jsonl --measurement-id local \
  --sizes 4096,65536 --concurrency 1,4 --requests 8 --rounds 1 --warmup-seconds 0
```

This is the first part of a stack: measurement, then learned RAM modelling, then opt-in SQL,
random-access and compression/decompression benchmark integration. Real S3 remains the reference.

## Learn and serve a RAM model

```bash
python3 benchmarks/s3/learn.py results/window-01.jsonl results/window-02.jsonl --output /tmp/s3-profile.json
cargo build -p vortex-bench --bin s3-simulate --profile release_debug
target/release_debug/s3-simulate --root vortex-bench/data --profile /tmp/s3-profile.json \
  --ready-file /tmp/s3-ready.json
```

The learner requires at least two real CI measurement windows with matching region, CPU count and
memory. It rejects local/custom endpoints, incomplete sweeps, failed reads, duplicate inputs and
missing provenance. It also accepts request observations from the earlier SQL calibration harness;
its unprimed process startup samples are excluded. Each source records its exact file hash and actual
CI run, attempt (unknown for older traces), measurement ID and machine identity.

The profile stores **joint** observed header, first-byte gap and remaining-transfer phases, grouped
by request size and actual in-flight load. Integer weights preserve probability mass; bulk quantile
representatives retain real phase tuples and each observation in the sparse upper tail retains its
original weight. Per-byte transfer cost scales by the original request's byte count. No query names,
query runtime targets or per-query multipliers enter fitting. The seed plus object/range and its
request ordinal makes sampling reproducible; scheduling can still change the observed load band.
HEAD/LIST currently use the median small-request header latency; writes are preparation operations
without learned delays. No hard request cap is inferred from these observations.

`s3-simulate` loads regular files into `object_store::memory::InMemory`, preserving their relative
keys, and exposes a path-style S3 HTTP endpoint on loopback. Normal S3 clients can GET bounded,
offset or suffix ranges, HEAD and LIST with delimiter/pagination, and PUT/delete preparation data.
ETag preconditions and range metadata are preserved. The default resident-data limit is 40% of
available memory; override `--max-resident-bytes` only after reserving query working memory and
client buffers. Oversized snapshots/writes fail. Snapshots reject symlinks and non-regular files.

Read delays use asynchronous absolute deadlines: headers, first body bytes, then 16 KiB body frames.
Concurrent reads overlap; deadlines avoid adding one timer tick per frame. `/_vortex/metrics` reports
GET/HEAD/LIST/PUT/delete HTTP counts, served bytes, peak/average in-flight requests and resident bytes.
A completed body or cancellation releases its in-flight slot. Start a fresh process per independent
run; metrics are cumulative and include preparation. Readiness JSON identifies the endpoint,
profile and resident budget. Only loopback is accepted because this endpoint has no authentication.

Use `--zero-delay` instead of `--profile` to measure the endpoint/client overhead. It is an explicit
baseline, not a learned profile. Loopback copies, HTTP and timer overhead add to fitted SDK durations;
held-out workloads on the same CI hardware must measure this effect before a profile is promoted.
Conditional service cost approximates shared bandwidth but omits temporal correlation, network
routing, throttling dynamics and retry feedback. Profiles are opt-in benchmark models, not claims of
S3 equivalence. The third branch wires this endpoint into benchmark workloads.

## Run existing benchmarks through the model

Prepare data with the normal generators first. The launcher mirrors `--data-root` directly at the
bucket root; for SQL this is the suite's prepared directory (including the scale factor), whereas
random access uses the full `vortex-bench/data` layout. Each command starts a fresh endpoint, reserves
query RAM, supplies dummy child credentials and writes a separate provenance/counter manifest.

```bash
cargo build --profile release_debug --bin s3-simulate --bin data-gen --bin datafusion-bench \
  --bin duckdb-bench --bin random-access-bench --bin compress-bench
target/release_debug/data-gen tpch --formats parquet,vortex --opt scale-factor=10.0

python3 benchmarks/s3/run.py --profile /tmp/s3-profile.json --kind sql \
  --data-root vortex-bench/data/tpch/10.0 --output /tmp/tpch-model-manifest.json -- \
  target/release_debug/datafusion-bench tpch --formats parquet,vortex --iterations 5 --opt scale-factor=10.0

# The same SQL path works with duckdb-bench or lance-bench and every BenchmarkArg suite.
# Prepare Lance locally through lance-bench before taking the snapshot.
# CPU-native DuckDB databases and LitData have no remote object-read variant.

target/release_debug/random-access-bench --prepare-data --datasets nested-lists --formats parquet,vortex
python3 benchmarks/s3/run.py --profile /tmp/s3-profile.json --kind random-access \
  --data-root vortex-bench/data --output /tmp/random-model-manifest.json -- \
  target/release_debug/random-access-bench --datasets nested-lists --formats parquet,vortex --open-mode both

python3 benchmarks/s3/run.py --profile /tmp/s3-profile.json --kind compress \
  --output /tmp/decode-model-manifest.json -- target/release_debug/compress-bench \
  --datasets '^wide table cols=100 chunks=1 rows=1000$' --formats parquet,vortex,arrow-ipc \
  --ops decompress --iterations 5
```

SQL forwards `remote-data-dir` without changing plans, pushdown, coalescing or caching. This includes
StatPopGen, PolarSignals, Public BI and custom Vortex queries, whose factory paths now honor this
option. DuckDB's native Parquet reader receives a loopback S3 secret; its Vortex reader and the other
clients use the same endpoint environment. Unsupported local-only formats fail explicitly.

Random access uses the existing Parquet/Vortex range readers and optional Lance reader, including
cached-handle and reopen measurements. Arrow IPC random access remains local-only. Build with
`--features random-access-bench/lance,compress-bench/lance` for optional Lance variants.

Compression keeps its CPU-only metric. With `--remote-data-dir`, decompression uploads the exact
compressed fixture outside timing, then every timed iteration reads and decodes it. Parquet, Vortex
and Arrow IPC use a full object GET followed by their CPU decoder; these results measure full-file
read+decode, not selective range-scan performance. Lance uploads its directory and uses its native
remote scanner. Remote read/decode and random-access results have distinct identities from local
and real-S3 records. The launcher does not publish benchmark results to the ordinary CI baseline.

The `S3 Model Benchmarks` workflow learns on two real request sweeps, freezes the profile, and runs
an independent held-out real sweep, the learned endpoint and a zero-delay loopback baseline. Its
probe table requires exactly matching GET/byte coverage and separates header, first-byte, body and
total median/p90. All clients run sequentially on one on-demand c8gd.16xlarge in the bucket's region.
It then runs all 22 TPC-H and 99 TPC-DS queries through DataFusion, DuckDB and Lance, plus synthetic
random access and read/decode, against real/model/second-real controls using identical files.
Each authenticated task receives fresh credentials from the standard pinned AWS workflow action;
Python inherits them without token retrieval, role assumption or credential files. Task timeouts
fit within the two-hour credential session, and all tasks remain on the same runner.

Branch pushes exercise **all queries at SF0.01, three iterations**. Manual dispatch defaults to
**SF10, five iterations**. `comparison.csv` recomputes median/p90 from raw iteration times,
reports model/pooled-real and real-repeat/real ratios, Spearman and ±10%/±30% flags. Native SQL
results preserve the ordinary cache sequence; they do not replace the separate controlled cold/warm
validation study. Endpoint counters cover whole commands, including registration/warmups; real
per-query request parity is not measured by this native-client smoke workflow. These limitations
remain explicit in the report, and real S3 stays the default reference.

The earlier frozen V4 controlled SF10 study (research branch `ji/ci-s3-over-time`) covered all 121
queries and matched all 1,368,393 GETs and 380,744,124,525 bytes per backend. It reached Spearman
0.9947, with 173/242 query/cache medians within ±10%. Its first three-hour window reached 23/36
within ±10%, with five groups exceeding measured S3 repeatability. That evidence supports an opt-in
simulator and more calibration; it does not establish the accuracy of this new native HTTP model.

For paired remote decoding, pass the same `--fixture-cache /tmp/run-fixtures` to both backend runs.
Use a fresh directory for each revision/input. The first run atomically saves its encoded file or
Lance directory; subsequent runs upload those exact objects, preserving generated data filenames and
manifests. CI uses one such cache for all three controls. This avoids comparing independently
re-encoded fixtures while keeping file creation and upload outside the read/decode timer.
