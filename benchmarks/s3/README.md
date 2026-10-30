<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# S3 measurements

`s3-measure` collects real S3 request observations independently of query engines and simulators.
The CI workflow sweeps 4 KiB, 64 KiB, 256 KiB, 1 MiB and 8 MiB ranges at concurrency 1, 4, 16
and 64. Three rounds use 1,700 requests per size/concurrency cell (about 102k per window), with
shuffled cell order and seeded object/range selection. Each concurrency uses identical ranges so load
is the changed input. These hot reads hit four fresh 16 MiB objects that the sweep reads repeatedly.

A separate cold sweep measures data nobody has read for a while. `--cold-source` names a persistent
prefix in the same bucket with objects under `<range bytes>/` directories; the CI keeps 2,000 per
size in `s3-measurements/cold-v1/`, adds any missing ones with `create_cold.py`, and never rewrites
or deletes them. After the hot sweep, each window reads every cold object exactly once at
`--cold-concurrency`, with no HEAD or warmup, so a read is the first touch of that object since the
previous window or run. Cold requests are labelled `phase: "cold"` and carry the object's
last-modified time.

```bash
cargo build -p vortex-bench --bin s3-measure --release
python3 benchmarks/s3/measure.py --source s3://bucket/prefix/ --count 4 --interval-seconds 10800 \
  --seed 20261028 -- --sizes 4096,65536,262144,1048576,8388608 --concurrency 1,4,16,64 \
  --requests 1700 --rounds 3 --warmup-seconds 20 --timeout-seconds 60 \
  --cold-source s3://bucket/cold/ --cold-concurrency 4
```

Objects must already exist; the collector never writes to S3. At least one must be 8 MiB or larger
for this sweep. The collector requires explicit sizes, concurrency, request count, rounds, seed,
warmup and deadline; callers choose these for their hardware. For a 192-core runner, for example,
extend the concurrency list to `1,4,16,64,192` and use at least `--requests 192` so the probe can
reach that load. The wrapper forwards arguments after `--` and supplies `--seed` separately.
The wrapper anchors four windows to 0, 3, 6 and 9 hours, then stops. It
retains a SHA-256 for each trace and the binary, checks identical runner/object/probe identities,
and preserves partial evidence on failure. Each request has a deadline; a window has a 30-minute
budget by default, which CI raises to an hour for the larger sweep. Starts over five minutes late are failures, without silently shifting the remaining schedule.

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
throughput. `measurements.csv` adds per-round median/p90/p99/p99.9 header, first-byte, body and
total timings for hot cells and one row per cold size. `tails.csv` pools all windows and rounds per
phase, size and concurrency (and across concurrency) so p99.9 rests on enough requests; the final
summary also writes that pooled table to the job summary.
Every successful window has a terminal completion record and exact declared cell/request coverage.
Object replacement is an error, guarded by ETag preconditions and response metadata checks.

Counts and timings cover **SDK operations**. The SDK may retry internally; those delays appear in
the operation duration, while individual HTTP attempts and recovered 503s are not separately counted.
The catalog LIST record covers the listing stream, including pagination. HEAD/LIST catalog setup is
labelled separately. The measured workload issues no PUTs. These files provide request-level model
inputs; they do not establish accuracy for SQL, random-access or decompression benchmarks.

Local harness exercise, explicitly ineligible as S3 training data:

```bash
target/release/s3-measure --source "file://$PWD/fixtures/" --allow-local \
  --output /tmp/local-s3-measure.jsonl --measurement-id local \
  --sizes 4096,65536 --concurrency 1,4 --requests 8 --rounds 1 --seed 20261028 \
  --warmup-seconds 0 --timeout-seconds 60
```

`RESULTS.md` records measured hot and cold p50/p90/p99/p99.9 latencies from a CI run.

This is the first part of a stack: measurement, then learned RAM modelling, then opt-in SQL,
random-access and compression/decompression benchmark integration. Real S3 remains the reference.
