<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 07 - Ongoing ingestion pipeline

The DuckDB database is **owned by the running benchmarks-website server**. It
lives on that server's local disk (an EBS volume on the existing EC2). CI
workers produce `results.json` and **POST it to the server**, which validates
and INSERTs. There is no S3 coordination layer for writes.

This is a deliberate simplification of earlier drafts (which had CI jobs
compare-and-swap a DuckDB file on S3). At our write rate - less than 100
measurements per day, all gated by a merge to `develop` - a single writer
trivially keeps up. No CAS needed.

## Shape

```text
+--------------+  POST /api/ingest         +-----------------------+
| CI runner    | -----------------------> | axum server         |
|              |     results.json        |                       |
|              |     + auth token        |   ...axum routes...   |
+--------------+                         |                       |
                                         |   DuckDB (RW handle)  |
                                         +----------+------------+
                                                    |
                                                    v
                                            /var/lib/bench.duckdb
                                               (EBS volume)
                                                    |
                                                    v (nightly)
                                         +----------+------------+
                                         | S3 backup snapshot    |
                                         | s3://.../backups/...  |
                                         +-----------------------+
```

## The ingest endpoint

A single authenticated HTTP endpoint:

```text
POST /api/ingest
Authorization: Bearer <token>
Content-Type: application/json

{
  "run_meta": {
    "commit_sha":   "<40-hex>",
    "benchmark_id": "random-access-bench",
    "schema_version": 1,
    "started_at":   "2026-04-21T12:34:56Z",
    "hardware_class": "bench-dedicated"
  },
  "records": [
    { ...one raw vortex-bench JSON record... },
    ...
  ]
}
```

Server responsibilities:

1. Validate auth (see below).
2. Upsert the commit row from the full metadata in `run_meta.commit` into
   `commits`. The server **does not** reach out to the GitHub API - CI
   already has this info via `scripts/commit-json.sh` and ships it in the
   payload. Keeps the server free of GitHub API dependencies / tokens /
   rate limits.
3. For each record: parse it with serde into a `ClassifiedMeasurement`
   struct. **No classification / name-parsing step.** `vortex-bench`
   emits v3-shape records directly (see
   [`10-emitter-changes.md`](./10-emitter-changes.md)), so the server
   just validates the shape.
4. Compute `measurement_id` (see [`05-schema.md`](./05-schema.md)).
5. `INSERT ... ON CONFLICT (measurement_id) DO UPDATE`. Duplicate POSTs are
   a no-op.
6. Records that fail serde parsing (malformed JSON, missing required
   fields) go into the `unclassified_records` sidecar table for
   inspection. Response body includes their count so CI logs surface
   them. In steady state this count should always be zero; non-zero
   means an emitter bug or version skew.
7. Return `{inserted: N, updated: M, unclassified: K}` plus any warnings.

The server has **no classifier module**. If the record shape evolves, we
update `vortex-bench`'s emitter + the server's `ClassifiedMeasurement`
struct + `schema_meta.current_version` in lockstep. The one-shot historical
migrator (see [`06-migration.md`](./06-migration.md)) is the only piece
of code that ever parses v2-shape `name` strings, and it's deleted
post-cutover.

### Commit-only POSTs

Not every push to `develop` produces benchmark measurements immediately,
but we want **every commit on `develop`** represented in the `commits`
table so the website can enumerate history even for commits whose bench
runs haven't completed (or never will, e.g. docs-only commits).

Solution: the `commit-metadata` job in `.github/workflows/bench.yml` (which
already runs unconditionally on every push to `develop` to append to
`commits.json` today) gets a second step that POSTs a commit-only payload
(`records: []`) to `/api/ingest`. Cheap. Triggers the same upsert path.

### File-size measurements

Today the SQL-benchmarks workflow writes `file-sizes-<id>.json.gz` files
to S3 separately from the main `data.json.gz`. In v3 these fold into the
same POST: each CI job's `results.json` includes its file-size records
alongside its timing records, and the classifier buckets them into
`metric_kind = 'compression_size'` (same as historical file-sizes data).
No separate sidecar upload.

## Ingesting data

During the dual-write cutover window, each CI bench job emits both v2-shape
and v3-shape JSONL (see [`10-emitter-changes.md`](./10-emitter-changes.md))
and runs both paths in parallel. The legacy `bash scripts/cat-s3.sh`
remains active until cutover. The new step is:

```bash
# Produce v3-shape JSONL alongside the existing v2-shape output.
bash scripts/bench-taskset.sh target/release_debug/${{ matrix.benchmark.id }} \
    --formats ${{ matrix.benchmark.formats }} \
    -d gh-json-v3 -o results.v3.json

# POST the v3-shape JSONL. Bearer token from a GitHub Actions secret.
python3 scripts/post-ingest.py \
    --server  https://bench.vortex.dev \
    --commit-sha  "$GITHUB_SHA" \
    --benchmark-id "${{ matrix.benchmark.id }}" \
    --results results.v3.json \
    --token   "$INGEST_BEARER_TOKEN" \
    --spool   "s3://vortex-ci-benchmark-results/outbox/"
```

Post-cutover the `-d gh-json` + `cat-s3.sh` pair is deleted; only the
`-d gh-json-v3` + POST path remains.

`scripts/post-ingest.py` is ~80 lines: read v3-shape JSONL, wrap in the
payload, POST with retry, print `{inserted, updated, unclassified}`. On
unrecoverable failure (server unreachable after all retries), dump the
payload to the spool S3 prefix - see next section.

## Write buffering (launch requirement)

Single-EC2 deploy means if the server is down when CI tries to POST, we'd
lose CI's best shot at delivering data. Mitigation (adopted as a launch
requirement):

**Spool-to-S3 outbox + scheduled drain.**

1. `post-ingest.py` retries up to 4 times with exponential backoff. On
   final failure, it uploads the payload as-is to
   `s3://vortex-ci-benchmark-results/outbox/<github_run_id>/<benchmark_id>/payload.json`.
   Return code 0 either way - the CI job succeeds even if the server was
   unreachable, because the data is durably in the outbox.
2. A scheduled workflow
   (`.github/workflows/drain-ingest-outbox.yml`, cron `*/10 * * * *`)
   lists `outbox/`, re-POSTs each payload to `/api/ingest`, and deletes
   the S3 object on success. Concurrency-gated (`concurrency: { group:
   ingest-drain }`) so two cron runs can't stomp each other.
3. The drain workflow alerts (via the existing incident.io integration)
   if the outbox has items older than 1 hour - that's the signal the
   server has been down long enough to need a human look.

At-least-once delivery, no new writer class (still just the server
process), minimal extra infra (one cron workflow + an S3 prefix). Total
code cost ≈ 80 LOC Python + ~40 LOC yaml.

**The drain workflow is the one new moving part we accept for launch.**
It's the cheap insurance that lets us treat the single-EC2 server as a
reasonable write path.

## Authentication

**Launch configuration: shared bearer token.** The token is generated once,
stored in a GitHub Actions secret (alongside the existing `GitHubBenchmarkRole`
credentials), and validated by the server against an `INGEST_BEARER_TOKEN`
env var on every POST. Constant-time comparison. Token is rotated manually
if compromised.

This matches v2's existing security complexity: v2 keeps writes off the
website entirely by routing them through AWS IAM on S3, using a single
shared OIDC role stored in GitHub Actions. v3 moves writes onto the website
and gates them with a single shared bearer token in the same place. Same
blast radius, comparable failure modes.

**Follow-up upgrade paths (post-launch, not blocking):**

- **AWS ALB + OIDC**: put the container behind an ALB with a listener rule
  that OIDC-authenticates `/api/ingest` via Cognito + GitHub. TLS via ACM.
  No server-side auth code.
- **Cloudflare Tunnel + Access**: publish through `cloudflared`, attach a
  Cloudflare Access policy to `/api/ingest` that validates GitHub OIDC
  tokens. TLS via Cloudflare. Adds Cloudflare as a dependency.
- **Server-side OIDC validation**: validate `id_token` from GitHub directly
  in the axum handler using a JWKS client. No new infra, more code.

Any of the three is a clean follow-up once v3 is live and we want to retire
the shared token. See [`09-open-questions.md`](./09-open-questions.md).

## DuckDB concurrency model

DuckDB allows one read-write process at a time per database file. Since the
axum server is that one process, this is fine:

- The **writer path** is the `/api/ingest` route handler. It takes a mutex or
  uses DuckDB's built-in transaction isolation (either works - our write rate
  is tiny).
- The **reader paths** are every other route. They share the same DuckDB
  handle (it supports concurrent read-only transactions from the same process).

We do not need a separate reader process or replica. If reads and writes ever
contend (they won't), we can move to DuckDB's built-in snapshot isolation or
to a simple `RwLock<Connection>` pattern.

## Durable storage + backup

- **Primary storage**: DuckDB file on an EBS volume attached to the EC2
  instance. EBS is durable enough for our purposes (annual failure rate <0.2%
  per AWS's own numbers; the data is reproducible from `data.json.gz` anyway
  for the backfilled portion, and from CI re-runs for anything new).
- **Backup**: cron job inside the container runs `duckdb bench.duckdb -c
  ".backup '/tmp/snapshot.duckdb'"` + `aws s3 cp` to
  `s3://vortex-ci-benchmark-results/backups/bench-<date>.duckdb`. Nightly is
  fine.
- **Restore**: download the most recent snapshot, swap it in, restart the
  container.
- **Rollback beyond that**: re-run the full migrator from `data.json.gz` +
  `commits.json`. Always available because those files never get deleted.

## Observability

The ingest endpoint logs per-call:

- `{commit_sha, benchmark_id, records_in, inserted, updated, warnings}`.
- Warnings include any records the classifier had to drop (unknown name
  pattern, etc.). These get aggregated into a Prometheus-style counter so we
  can notice when a benchmark starts emitting records we don't understand.

The server also exposes `/health`:

```jsonc
{
  "db_path":       "/var/lib/bench.duckdb",
  "db_size_bytes": 41943040,
  "row_count":     1823412,
  "commit_count":  5284,
  "latest_commit": {
    "sha":       "...",
    "timestamp": "2026-04-21T16:01:24Z"
  },
  "last_backup_at": "2026-04-21T02:30:00Z"
}
```

## Failure modes

- **Server down when CI tries to POST.** CI retries with backoff (4
  attempts, matching the existing `git push` retry policy). After that the
  job fails. The results aren't lost - they're in `results.json` as a CI
  artifact and can be replayed by hand: `scripts/post-ingest.py --results
  <downloaded_artifact>`. See also the "Write buffering / HA gap" note in
  [`09-open-questions.md`](./09-open-questions.md) - this is the one
  scenario where the current single-EC2 design is visibly fragile, and we
  may want a spool-to-S3 fallback path for launch.
- **Bad record in a payload.** Classifier puts the record into
  `unclassified_records`; the rest of the payload is accepted. The
  response's `{unclassified: K}` field lets CI log the count so the
  operator notices.
- **EBS volume fails.** Restore from the latest S3 backup (nightly
  snapshots). If that's stale, re-run the historical migrator.
- **Ingest endpoint abused.** Bearer-token required; unauthorized POSTs
  return 401 with a constant-time comparison so tokens can't be brute-
  guessed by timing. Launch: no rate limit beyond that. Add one if abuse
  ever materializes.

## Schema version on boot

The server reads `schema_meta.current_version` from the DB on startup and
refuses to serve if the DB's version is newer than the binary expects (see
[`05-schema.md`](./05-schema.md)). This prevents the "deployed an old
container against a migrated-forward DB" footgun. Migrations forward are
applied automatically; there is no downgrade path.

## What replaces `cat-s3.sh` / `commit-json.sh`?

- `cat-s3.sh data.json.gz`: goes away entirely. Replaced by the POST call
  above.
- `commit-json.sh`: stays, but its output is sent in the POST payload
  (`run_meta`) rather than appended to `commits.json` on S3. The server
  upserts into the `commits` table.

Optional: during the cutover window we can run *both* pipelines in parallel
(CI writes to the old JSONL **and** POSTs to the new server). Belt and
braces. See [`06-migration.md`](./06-migration.md).

## What the migrator does vs. what the ingester does

Same hash function, same DuckDB shape. Different JSON input shapes, and
only one of them has a classifier:

- **`/api/ingest`** - receives v3-shape JSON emitted directly by
  `vortex-bench` (see [`10-emitter-changes.md`](./10-emitter-changes.md)).
  Serde-parse → hash → INSERT ON CONFLICT. No classifier.
- **Migrator binary** - reads historical v2-shape `data.json.gz` +
  `commits.json` + `file-sizes-*.json.gz`. Runs its own one-shot
  classifier that parses v2's `name` strings into dimensions, then either
  (a) POSTs v3-shape records to the preview server's `/api/ingest` or
  (b) writes directly to a local DuckDB file. Deleted post-cutover along
  with its classifier. See [`06-migration.md`](./06-migration.md).
