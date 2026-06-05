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
  honor a `-- migrate: no-transaction` directive) rather than splitting
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

This README + the runner ship in PR-1.2; `001`/`002` land in PR-1.3 and `003`
in PR-1.4.
