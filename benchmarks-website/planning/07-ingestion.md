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
| CI runner    | -----------------------> | Leptos server         |
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
2. Ensure the commit exists in `commits` (upsert if new - server can fetch
   commit metadata from GitHub's API given the SHA, or accept it in the
   payload).
3. For each record: run the **classifier** (the same Rust logic the migrator
   uses - it's a shared library) to produce a structured row.
4. Compute `measurement_id` (see [`05-schema.md`](./05-schema.md)).
5. `INSERT ... ON CONFLICT (measurement_id) DO UPDATE`. Duplicate POSTs are a
   no-op.
6. Return `{inserted: N, updated: M}` plus any classifier warnings.

The classifier runs server-side, not in CI, so the moment we fix a
classification bug the fix applies retroactively to re-POSTed data without
touching CI.

## Ingesting data

Replace each `bash scripts/cat-s3.sh vortex-ci-benchmark-results data.json.gz
results.json` step in `.github/workflows/bench.yml` and `sql-benchmarks.yml`
with something like:

```bash
# Get a short-lived token (e.g. via GitHub OIDC → a small exchange endpoint
# on the server, or a pre-shared GitHub secret).
TOKEN=$(scripts/get-ingest-token.sh)

# POST the JSONL.
python3 scripts/post-ingest.py \
    --server https://bench.vortex.dev \
    --commit-sha "$GITHUB_SHA" \
    --benchmark-id "${{ matrix.benchmark.id }}" \
    --results results.json \
    --token "$TOKEN"
```

`scripts/post-ingest.py` is ~40 lines: read JSONL, wrap it in the payload
above, POST, check status, print `{inserted, updated}`.

## Authentication

The ingest endpoint accepts writes from our CI and nowhere else. Two feasible
approaches; we can pick at implementation time:

### Option 1: GitHub OIDC + pre-shared JWKS validation

- The CI job already gets an OIDC token from GitHub (we use that for AWS IAM
  role assumption today).
- The server validates the token's signature against GitHub's public JWKS and
  checks the `repository` + `ref` claims match `vortex-data/vortex` +
  `refs/heads/develop` (or an allowlist).
- No shared secret to rotate.

### Option 2: Static bearer token in GitHub Secrets

- A long-lived token lives in GitHub Actions secrets.
- Server validates against a hash stored in an env var.
- Easier to set up. Needs manual rotation.

Preference is Option 1 (OIDC, no rotating shared secret). Fall back to
Option 2 if the OIDC flow is annoying to wire up. Either way, the endpoint is
not accessible to randos on the internet.

## DuckDB concurrency model

DuckDB allows one read-write process at a time per database file. Since the
Leptos server is that one process, this is fine:

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

- **Server down when CI tries to POST.** CI retries with backoff (4 attempts,
  like existing `git push` retry policy). After that it fails the job. The
  results aren't lost - they're in `results.json` as a CI artifact and can be
  replayed by hand: `scripts/post-ingest.py --results <downloaded_artifact>`.
- **Bad record in a payload.** Classifier rejects just that record; the rest
  of the payload is accepted. The response lists rejected records so the
  operator can see what needs a classifier fix.
- **EBS volume fails.** Restore from the latest S3 backup; if that's stale,
  re-run the migrator from `data.json.gz`.
- **Ingest endpoint DoS'd.** It's behind CloudFront / ALB with rate limits
  and bearer-token auth; not reachable without credentials in normal use.

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

Same classifier, same hash function, same DuckDB shape. Different entry
points:

- **`/api/ingest`** - one payload per CI run, many small POSTs over time.
- **Migrator binary** - reads `data.json.gz` + `commits.json` once, batches
  everything into the same classifier, and either (a) POSTs in bulk to the
  ingest endpoint or (b) writes directly to the DuckDB file when running
  locally without a server. Option (a) is the preferred dev loop; see
  [`06-migration.md`](./06-migration.md).
