# v4 RDS re-migration runbook (2026-06-16) — freshen stale prod data

**Status: DONE -- EXECUTED + VERIFIED 2026-06-16. This was a PROD RDS WRITE (atomic full replace).**
Kept for the record and for the next re-migration. The run used the **release** binary
(`cargo build --release -p vortex-bench-migrate`; the debug profile's bundled DuckDB does not link
here, but release does -- its `libduckdb.a` is already compiled). It was run with
`--allow-missing-file-sizes` because 3 `file-sizes-*-s3.json.gz` sources (`tpch-s3`, `tpch-s3-10`,
`fineweb-s3`) 403 from the public bucket. Results: 4597 commits, 4,919,122 query rows, verify clean
(0 diffs), R1 check clean (0 NULL `commit_timestamp`, newest 2026-06-16 across all 6 datasets). See
the spine's SESSION HANDOFF 2026-06-16c for the full record.

> **2026-06-16 CORRECTIONS (read before trusting any earlier version of this runbook).** While
> prepping the run I read the loader (`benchmarks-website/migrate/src/postgres.rs`) and the schema
> migrations. The earlier runbook was wrong on two load-bearing points:
>
> 1. **The loader is NOT a "full replace" — it is append-only over primary keys.** `postgres::load`
>    only `COPY ... FROM STDIN`s into the existing tables; it never `TRUNCATE`s. Every table has a
>    `PRIMARY KEY` (`measurement_id`, and `commit_sha` for `commits`; see
>    `migrations/001_initial_schema.sql`). Re-running the old `load` against the already-populated
>    prod DB would abort on the first duplicate key and roll the whole transaction back (this is
>    exactly what the `rehearsal_mid_load_failure_rolls_back_to_empty` e2e test demonstrates). It was
>    written as a ONE-SHOT seed into an EMPTY v4 DB. **Fix applied this session:** a new `--replace`
>    flag on `load` that `TRUNCATE`s all six tables as the first statement *inside* the load
>    transaction, making the re-load an atomic replace (a mid-load failure rolls back to the ORIGINAL
>    data, never empty). See "Code change" below — it is in the working tree, uncommitted.
>
> 2. **The load must connect as the RDS MASTER (`postgres`), NOT `migrator`.** The six data tables
>    are owned by the RDS master. `TRUNCATE` requires table ownership; `COPY`/INSERT requires INSERT.
>    `migrator` has NO data-table DML (only ledger + schema CREATE/USAGE — `migrations/002`/`003`).
>    `bench_ingest` has SELECT/INSERT/UPDATE but TRUNCATE is *explicitly withheld* (`migrations/004`).
>    Only the master can do both. This matches the README + `main.rs` Load help ("operator-local
>    master-password DSN") and the original PR-5.0 prod-load path. The master password lives in AWS
>    Secrets Manager.

## Why
v4 RDS data is stale: newest commit was ~2026-06-10, today is 2026-06-16 (~6 days behind). v4 only
gets data via the migration (the CI -> v4-Postgres ingest cutover, PR-5.1, is held off), so
re-migrating is currently the only way to freshen it. This is a STOPGAP; the durable fix is finishing
PR-5.1. After the next develop runs, v4 drifts stale again until that cutover lands.

## Source decision: v2's live S3, NOT the v3 backups
- **v2 public S3** (`vortex-ci-benchmark-results`) is the live source CI appends to every develop run.
  Verified current earlier this session: newest commit `2026-06-16T15:21:24+01:00`. The migrate `run`
  mode reads exactly this (`data.json.gz` / `commits.json` / `file-sizes-*.json.gz`). Public bucket,
  no credentials needed for `run`.
- **v3 backups** (`s3://vortex-benchmark-results-database/v3-backups/`) REJECTED: AccessDenied from
  the bench-prod profile (they live in account 375504701696, not the bench account 245040174862), and
  their freshness is uncertain (only produced while the v3 EC2 server runs; develop #8362 removed the
  v3 server/ops). Both paths funnel through the same `load` step anyway.

## Code change made this session (uncommitted — user commits)
Adds an atomic-replace path to the loader so a re-migration into a populated target is safe.
`cargo check -p vortex-bench-migrate` is GREEN. Files:
- `benchmarks-website/migrate/src/postgres.rs` — `load()` gains a `replace: bool` param; when true it
  `TRUNCATE`s all six tables (list built from `TABLE_SPECS`) as the first statement of the existing
  load transaction. No `CASCADE` (no FKs at alpha). Doc comment updated.
- `benchmarks-website/migrate/src/main.rs` — `Load` subcommand gains `--replace` (default false),
  passed through to `postgres::load`.
- `benchmarks-website/migrate/tests/postgres_e2e.rs` — updated the two existing `load(...)` call
  sites to pass `false`; added `rehearsal_replace_load_reseeds_a_populated_target` (Docker-gated,
  CI-run): proves a plain re-load over a populated target fails on duplicate keys, the `--replace`
  re-load succeeds with exact fixture counts, and `commit_timestamp` is repopulated.

## Build gotcha (why this wasn't run locally)
`cargo build -p vortex-bench-migrate` FAILS in the local sandbox at linking the bundled DuckDB native
static lib: `could not find native static library 'duckdb'` while compiling `libduckdb-sys` (the
`bundled` feature compiles DuckDB's C++ amalgamation). `cargo check` passes — the failure is purely
the native link step, not the Rust code. Build + run in an environment where the bundled DuckDB
compiles/links (CI, or a non-sandboxed shell with a working C/C++ toolchain). The binary lands at
`target/debug/vortex-bench-migrate` (or use `cargo run -p vortex-bench-migrate --`).

## Tool: `benchmarks-website/migrate/` (Rust crate `vortex-bench-migrate`)
Exists on `ct/bench-v4` (develop deleted it via #8362). Subcommands (see `migrate/README.md`,
`src/main.rs`):
```
migrate run    --output fresh.duckdb                                                    # v2 S3 -> fresh v3 DuckDB (current to today)
migrate load   --duckdb fresh.duckdb --postgres-target <DSN> --ca-cert <pem> --replace  # -> v4 RDS, ONE atomic txn: TRUNCATE all six, then COPY
migrate verify --duckdb fresh.duckdb --postgres-target <DSN> --ca-cert <pem>            # value-verify loaded rows == DuckDB per measurement_id
```
The load copies `measurement_id` verbatim from DuckDB and, after the COPYs, runs the post-COPY UPDATE
that denormalizes `query_measurements.commit_timestamp` from `commits.timestamp` (the read-path R1
sort key). `--replace` makes it a full atomic replace; WITHOUT `--replace` it aborts on the first
duplicate PK against the populated prod DB.

## Connection (RDS) — connect as the MASTER role `postgres`
- Endpoint: `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`, DB `vortex_bench`.
- DSN shape: `postgresql://postgres:<MASTER_PW>@<endpoint>:5432/vortex_bench?sslmode=require`
  (URL-encode the password if it has reserved chars). `--ca-cert <global-bundle.pem>` is what actually
  verifies the host; `sslmode=require` is the libpq knob tokio-postgres understands (the rustls
  connector + CA bundle is verify-full-equivalent — see the `connect_postgres` comment in
  `postgres.rs`).
- **Master password**: in AWS Secrets Manager (the RDS-managed master secret). The CONFIRMED secret
  ARN (2026-06-16) is
  `arn:aws:secretsmanager:us-east-1:245040174862:secret:rds!db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690-egkQgW`
  (DbiResourceId `db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690`). Do NOT guess the id and do NOT
  `list-secrets` (the auto-mode classifier blocks secret listing as scouting; an earlier guessed id
  `rds!db-4VPTDACTRQHOS24WEIR3TNC2M4` returned `ResourceNotFoundException`). Resolve it
  deterministically from instance metadata:
  `aws rds describe-db-instances --profile bench-prod --region us-east-1 --db-instance-identifier vortex-bench-prod --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text`,
  then
  `aws secretsmanager get-secret-value --secret-id <arn> --profile bench-prod --region us-east-1 --query SecretString --output text`.
  Do NOT echo it into the chat/logs; assign to a shell var and interpolate into the DSN.
- RDS CA bundle for `--ca-cert`: `curl -sSO https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem`.
- AWS access: profile `bench-prod` (connor-aws-cli, account 245040174862). See memories
  `project_bench_aws_access`, `project_bench_rds_profiling_access`. Instance is `db.r7g.large` (16 GiB).

## Execution checklist
0. **Pre-load RDS snapshot** (insurance before the atomic replace; the rollback path if anything goes
   wrong): `aws rds create-db-snapshot --profile bench-prod --region us-east-1 --db-instance-identifier vortex-bench-prod --db-snapshot-identifier vortex-bench-prod-pre-remigration-20260616`
   (wait for `available`, or use the existing PITR window). RDS PITR is the documented rollback path.
1. Build the migrate crate in a working build env: `cargo build -p vortex-bench-migrate` (use
   `RUSTC_WRAPPER=` only for the exact `sccache: Operation not permitted` error — that is NOT the
   bundled-DuckDB link error above).
2. `migrate run --output /tmp/fresh.duckdb` — reads v2 public S3. Eyeball the summary: per-table row
   counts and that the uncategorized fraction is under the 5% gate (the tool bails if not). Confirm
   the newest commit looks like ~2026-06-16.
3. Fetch the RDS CA bundle (step in Connection above) for `--ca-cert`.
4. Retrieve the master password (Connection above) into a shell var; build the DSN.
5. `migrate load --duckdb /tmp/fresh.duckdb --postgres-target "$DSN" --ca-cert global-bundle.pem --replace`
   — THE PROD WRITE. One transaction: TRUNCATE all six, COPY all six, denormalize `commit_timestamp`.
   Eyeball the printed per-table counts against step 2.
6. `migrate verify --duckdb /tmp/fresh.duckdb --postgres-target "$DSN" --ca-cert global-bundle.pem`
   — must exit 0 (clean). Non-zero = a presence/value diff; investigate before declaring success.
7. **R1 invariant check (load-bearing):** the post-COPY UPDATE must have populated `commit_timestamp`
   (a NULL silently drops chart rows). The loader does this in-txn, and the e2e test asserts it, but
   verify on the live system: hit the live read API for a big group (e.g. `?n=100`) and confirm charts
   render and the newest commit is ~2026-06-16, not blank/stale.
8. **BUST THE CACHE (required -- a manual load does NOT auto-refresh the site).** The site reads
   the default `?n=100` chart data AND group summaries through the Vercel Data Cache (`bench-data`
   tag, 24h backstop) + edge CDN (`s-maxage=300`). A manual `load` bypasses `post-ingest.py`'s
   `/api/revalidate` hook, AND that hook is unwired anyway (gated on `BENCH_REVALIDATE_TOKEN` +
   `BENCH_SITE_BASE_URL`), so nothing invalidates the cache -- the site shows STALE data + STALE
   summaries up to 24h until you bust it. Do ONE of:
   - **Vercel CLI (cleanest, tag-scoped; CLI is authed as `connor-6267` as of 2026-06-16):**
     `vercel cache invalidate --tag bench-data` (serves stale + revalidates in background), or
     `vercel cache purge --type data` for the whole project's Data Cache.
   - **Dashboard:** project -> **CDN** (sidebar; the Data Cache controls live UNDER CDN, the
     non-obvious part) -> **Caches** -> Purge cache -> All content -> **Runtime and Data Cache** ->
     Purge + confirm. The CDN layer (`s-maxage=300`) then catches up within ~5 min, or purge it in
     the same step.
   Then verify: load a default `?n=100` big-group view and confirm the newest commit is ~today.

## Gotchas
- Prod write: confirmed by the user, but take the snapshot (step 0) first; `--replace` is atomic so a
  failed load rolls back to the original data, but the snapshot covers operator error too.
- Connect as MASTER (`postgres`), not `migrator`/`bench_ingest` — TRUNCATE needs table ownership.
- The migrate crate's bundled DuckDB does not link in the local sandbox — run elsewhere (Build gotcha).
- This does NOT fix steady-state freshness — v4 drifts stale again after the next develop runs until
  the PR-5.1 ingest cutover lands.
