<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# `migrations/` — benchmarks-website Postgres DDL

Each `*.sql` file in this directory is a forward-only migration applied in
filename order by [`scripts/migrate-schema.py`](../scripts/migrate-schema.py).
The runner tracks applied migrations in the `_applied_migrations` table on the
target database and is idempotent: re-running a migration set whose filenames
all appear in `_applied_migrations` is a no-op.

## Naming convention

`NNN_<short_snake_case_description>.sql`, where `NNN` is a 3-digit
zero-padded sequence number (`001`, `002`, …). The runner applies in
lexicographic order, which equals numeric order under this convention.

## Authoring rules

- One conceptual change per file. Bundle related DDL only if rolling them
  out separately would leave the database in an unusable state.
- Statements run inside a single transaction per file, full stop. The
  runner does not currently support `CREATE INDEX CONCURRENTLY` or any
  other DDL that cannot run inside a transaction block. If a future
  migration genuinely needs CONCURRENTLY, extend the runner first (e.g.
  honor a `-- migrate-schema: no-transaction` directive) rather than splitting
  the DDL across files -- a half-applied non-transactional migration is
  hard to recover from at 2 AM.
- Never edit a file after it has been applied to production. To revise
  an earlier migration, write a new migration that supersedes the prior
  state.

## Initial files

- `001_initial_schema.sql` — six tables plus dim-leading composite indexes
  following the read-path chart-query filter columns (PR-1.3).
- `002_iam_db_user.sql` — `CREATE ROLE` for the `migrator` IAM-auth user that
  the schema-deploy workflow (`.github/workflows/schema-deploy.yml`, PR-1.4)
  assumes into via direct IAM to the public instance endpoint (PR-1.3).
- `003_migrator_ledger_grant.sql` — grants `migrator` `SELECT, INSERT` (only,
  no `DELETE`/`UPDATE`) on `public._applied_migrations` so a migrator-role
  apply can record/read the ledger against the master-owned bootstrap (PR-1.4).
- `004_ingest_role.sql` — `CREATE ROLE` for the least-privilege `bench_ingest`
  IAM-auth ingest user, its `USAGE` + per-table `SELECT, INSERT, UPDATE` grants on
  the six data tables, and a default-privilege rule so future `migrator`-created
  tables auto-grant the ingest role (PR-2.1).
- `005_read_role.sql` — `CREATE ROLE` for the read-only `bench_read` user the v4
  Next.js read service on Vercel authenticates as, its `USAGE` + per-table
  `SELECT`-only grants on the six data tables, and a default-privilege rule so
  future `migrator`-created tables auto-grant the read role. Unlike 002/004 it
  carries **NO `rds_iam` grant** and idempotently **revokes** any pre-existing
  `rds_iam` membership: the read service authenticates with a static password
  (Vercel has no AWS credentials to mint IAM tokens), and on RDS `rds_iam`
  membership forces IAM-only auth. Added during the Phase-4 operator-gate work.
- `006_read_path_perf.sql` — denormalizes `commits.timestamp` onto
  `query_measurements.commit_timestamp` (with a one-time backfill) and adds the
  read-path-perf indexes (PR-5.1.5). Applied by the RDS master (it ALTERs a
  master-owned table), hence its `requires-superuser` marker.
- `007_summary_covering_index.sql` — rebuilds the summary index with
  `INCLUDE (value_ns)` so the latest-per-series read is index-only (PR-5.1.5).
  Same `requires-superuser` marker as 006. Note: 006/007's in-file rationale
  describes the `DISTINCT ON` read path that shipped first; the read service
  later moved to recursive-CTE skip scans (the PR-5.1.5 cold-render fix) that
  use the same indexes, and the files are frozen post-apply (see Authoring
  rules), so their prose is historical. The current consumer is documented in
  `benchmarks-website/web/lib/summary.ts` / `queries.ts`.

This README + the runner ship in PR-1.2; `001`/`002` land in PR-1.3, `003` in
PR-1.4, `004` in PR-2.1, `005` in the Phase-4 operator-gate work (the v4
read-service identity), and `006`/`007` in PR-5.1.5 (read-path perf).

## Re-stamping `commit_timestamp` (operational note)

`query_measurements.commit_timestamp` is a denormalized copy of
`commits.timestamp` (006). Writers stamp it at insert; 006 backfilled
pre-existing rows. If rows are ever observed unstamped or drifted (a writer
predating the stamp, or a commits re-ingest that changed a timestamp), the
repair is the drift-tolerant form of the backfill. Run it as a role that holds
`UPDATE` on `query_measurements` **and** `SELECT` on `commits` (the `FROM
commits` clause reads `commits.timestamp`) — the RDS master, or any role granted
both; the least-privilege `bench_read` role cannot run it:

```sql
UPDATE query_measurements q
   SET commit_timestamp = c.timestamp
  FROM commits c
 WHERE c.commit_sha = q.commit_sha
   AND q.commit_timestamp IS DISTINCT FROM c.timestamp;
```

(`IS DISTINCT FROM` also fills NULLs, so this strictly supersedes the
NULL-only 006 form. Follow a large repair with `VACUUM ANALYZE
query_measurements` — the 006 prod backfill showed the planner needs it.)

## Bootstrap ordering — `requires-superuser` migrations

Migrations whose header carries a `-- migrate-schema: requires-superuser`
marker comment (currently `002`, `004`, `005`, `006`, and `007`; the marker in
the file is authoritative, not this list) must be applied by a master-capable
role. Before applying a marked file, the runner
([`scripts/migrate-schema.py`](../scripts/migrate-schema.py)) asserts the
connected role is **master-capable** — the capability proxy is `rolsuper OR
rolcreaterole` (a true superuser locally, or the RDS master, which has
`CREATEROLE`). If the connected role has neither, the runner raises
`PermissionError` and refuses to apply the file, rather than failing partway
through its privileged `CREATE ROLE` / `GRANT` / `ALTER DEFAULT PRIVILEGES` /
master-owned-table DDL statements.

The ordering contract this enforces: **apply every marked migration as the RDS
master before any `migrator`-role deploy.** The `migrator` IAM user that
`schema-deploy.yml` assumes into is itself created by `002` and is not
master-capable, so it cannot apply the marked files; those must land first
under the master. `001` (plain DDL) and `003` (a ledger grant) carry no marker
and apply under either role. A future migration that needs a master-capable
role (another `CREATE ROLE`, `ALTER DEFAULT PRIVILEGES`, DDL on master-owned
tables, or other superuser-only DDL) should carry
`-- migrate-schema: requires-superuser` on a comment line so the same
preflight guards it.
