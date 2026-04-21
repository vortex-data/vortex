<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 07 - Ongoing ingestion pipeline

After the one-shot historical migration (see [`06-migration.md`](./06-migration.md)),
we need an ongoing flow that gets new benchmark runs into the DuckDB file on S3.
This replaces today's `scripts/cat-s3.sh` + `data.json.gz` append pattern.

## Requirements

- **Every merge to `develop`** triggers N parallel benchmark jobs (random-access,
  compress, and each SQL suite). Each job produces its own `results.json`.
- All of these need to land in the DB without racing.
- Missing a single job's data is a bug, not a graceful degradation. The
  ingester retries.
- The website can tolerate a 1-10 minute delay between a run finishing and the
  new data being visible.

## Two reasonable designs

### Option A: CI writes directly to `bench.duckdb` on S3 (CAS)

Each CI job:
1. Produces `results.json`.
2. Downloads `bench.duckdb` from S3 (with ETag).
3. Opens it, INSERTs new rows, closes.
4. Uploads back with `--if-match <etag>`.
5. On 412, retries.

Pros: simple, keeps the "one source of truth, on S3" model, no new
infrastructure.

Cons: DuckDB is not designed to be mutated by many concurrent writers fighting
over an S3 object. The compare-and-swap is at the **whole file** level, which
means under contention every writer serializes on downloading+re-uploading the
entire DB. For our scale (single-digit MB DB, ~20 concurrent benchmark jobs
per commit) this is *probably* fine, but it's a known scaling ceiling.

### Option B: CI writes per-run shards, a separate step merges

Each CI job:
1. Produces `results.json`.
2. Uploads to `s3://vortex-ci-benchmark-results/pending/<run_id>/<job_id>.json`.
   No concurrency; every path is unique.

A scheduled ingester (hourly, or triggered by bench workflow completion):
1. Lists `s3://vortex-ci-benchmark-results/pending/`.
2. Downloads the current `bench.duckdb`.
3. INSERTs rows from each pending shard.
4. Uploads the new `bench.duckdb`.
5. Moves processed shards to `processed/<date>/`.

Pros: no CAS contention, shards are immutable and re-ingestible, failure
recovery is trivial (just re-run the merger).

Cons: extra hop, slightly more infra. The merger is either a GitHub Action on
a schedule or a tiny Lambda/EC2 cron.

### Recommendation

**Start with Option B.** Simpler concurrency story, easier to reason about.
Option A is an optimization we can take later if B's latency is a problem
(which it shouldn't be; we're not latency-bound).

## Concrete shape of Option B

### Pending shard format

Keep it simple: the existing `results.json` JSONL, as produced by
`vortex-bench`, with one extra wrapping line or a sidecar file carrying
run-level metadata.

```text
s3://vortex-ci-benchmark-results/pending/<github_run_id>/<benchmark_id>/meta.json
s3://vortex-ci-benchmark-results/pending/<github_run_id>/<benchmark_id>/results.json
```

`meta.json`:

```jsonc
{
  "run_id":         "12345678901",
  "benchmark_id":   "random-access-bench",
  "commit_sha":     "<40-hex>",
  "started_at":     "2026-04-21T12:34:56Z",
  "hardware_class": "bench-dedicated",      // from runs-on runner spec
  "schema_version": 1                       // so we can evolve the ingester
}
```

The benchmark emitters already know `commit_sha`; they just stop appending to
`data.json.gz` and instead write these two files.

### Upload from CI

Replace the `bash scripts/cat-s3.sh vortex-ci-benchmark-results data.json.gz
results.json` step with two S3 put-objects (no CAS needed since the run_id +
benchmark_id path is unique).

Keep the existing AWS OIDC + role setup. The ingester role needs additional
`s3:ListBucket`, `s3:DeleteObject` (or `s3:CopyObject`), and `s3:PutObject`
for the pending/processed prefixes + the `bench.duckdb` key.

### Merger job

A new GitHub workflow (`.github/workflows/ingest-benchmarks.yml`) on a schedule
(`cron: every 10 minutes`) plus `workflow_dispatch` for manual runs.

- Checks if any shards exist in `pending/`. If not, exits.
- Downloads `bench.duckdb` to local disk.
- For each shard: parse meta.json, parse results.json, classify, INSERT.
- Writes `bench.duckdb` back.
- Moves shards to `processed/<YYYY-MM-DD>/`.

The merger binary is shared code with the migrator - 80% the same classifier.
Keep them in one crate with two binaries.

### Commit metadata ingestion

Today `scripts/commit-json.sh` + `scripts/cat-s3.sh` appends to `commits.json`.
In v3, either:

- Keep that flow and have the merger also pick up new commits (simplest), or
- Have the merger fetch the commit from the GitHub API directly given a SHA
  (more robust; no JSON to maintain).

Pick the first for the initial v3; swap to the second later if the shell
script is flaky.

## Non-negotiables

- **Idempotent merges.** Running the merger twice over the same pending shard
  must not duplicate rows. The deterministic measurement_id from
  [`05-schema.md`](./05-schema.md) makes this easy.
- **Observable.** The merger logs counts of rows inserted per shard, per
  metric_kind, and per benchmark_id. If a merge inserts 0 rows for a benchmark
  that usually produces hundreds, that should be a visible signal (log + alert).
- **Backpressure.** If the merger is down for a day, pending shards pile up
  but nothing is lost. The merger drains them on next run.

## What about the Leptos server's view of the DB

The server just polls S3 for a new `bench.duckdb`:

- Every M minutes (start with 5), `HEAD` the object, compare ETag.
- If changed, download to a temp path, open a read-only DuckDB handle, swap it
  into the shared app state, drop the old one.
- Server never writes.

## Failure modes

- **Merger fails on a bad shard.** Log, move the shard to a `quarantine/` prefix,
  continue with the rest.
- **DuckDB upload fails.** Retry; don't move shards out of pending until the
  upload succeeds.
- **Concurrent merger runs (shouldn't happen with a cron-gated workflow but
  just in case).** Use an S3 lock object (`locks/ingester.lock` with TTL) or
  rely on GitHub's workflow concurrency controls (`concurrency: { group:
  ingester }`). The latter is simpler.
