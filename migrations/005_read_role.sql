-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- migrate-schema: requires-superuser
-- This migration creates a login role (`CREATE ROLE bench_read`) and runs
-- `ALTER DEFAULT PRIVILEGES FOR ROLE migrator` -- both requiring a
-- master-capable executing role, exactly like 004. The marker makes
-- `migrate-schema.py` reject a non-master `apply` loudly and early.

-- Create the `bench_read` login role used by the v4 read service
-- (`benchmarks-website/web/`, the Next.js app on Vercel). The read service only
-- ever SELECTs (chart payloads, group discovery, summaries, health row
-- counts), so this role is read-only by construction: USAGE on the schema plus
-- SELECT on the six data tables, nothing else. It is deliberately SEPARATE
-- from `bench_ingest` (004): the read path runs on third-party serverless
-- infrastructure (Vercel), so it gets an identity that cannot write even if
-- its credential leaks.
--
-- Authentication: STATIC PASSWORD, set OUT-OF-BAND by the operator
-- (`ALTER ROLE bench_read PASSWORD '...'` as master; never in a committed
-- migration). Deliberately NO `rds_iam` grant, unlike 002/004: on RDS,
-- membership in `rds_iam` makes IAM authentication MANDATORY (password auth
-- fails with "PAM authentication failed"), and the Vercel runtime has no AWS
-- credentials to mint IAM tokens with. If the read path later moves to IAM
-- (for example via Vercel-to-AWS OIDC federation), a follow-up migration
-- grants `rds_iam` at switch time, which atomically disables the password.
--
-- Idempotent and substrate-portable, matching 002/004: `CREATE ROLE` is
-- guarded (roles are cluster-global); the `rds_iam` REVOKE that enforces the
-- password-auth invariant on a PRE-EXISTING role is guarded behind an existence
-- check (a no-op on vanilla Postgres) and a REVOKE of a non-member is itself a
-- no-op; and re-granting an existing privilege is a Postgres no-op.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bench_read') THEN
        CREATE ROLE bench_read WITH LOGIN;
    END IF;
    -- Enforce the no-`rds_iam` (password-auth) invariant idempotently. The
    -- `CREATE ROLE` guard above only covers a FRESH role; a `bench_read` that
    -- pre-exists as a member of `rds_iam` (an earlier apply that granted it, or
    -- manual setup) would otherwise survive a green re-apply still IAM-only,
    -- silently breaking the read service's password auth. Guarded behind the
    -- `rds_iam` existence check so this is a no-op on vanilla Postgres (local dev
    -- + the testcontainer suite); a `REVOKE` of a non-member is itself a harmless
    -- no-op (NOTICE only), so the statement is fully idempotent.
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rds_iam') THEN
        REVOKE rds_iam FROM bench_read;
    END IF;
END$$;

-- `bench_read` must reach objects in `public` but must NOT create any:
-- USAGE only, never CREATE.
GRANT USAGE ON SCHEMA public TO bench_read;

-- SELECT-only on the six data tables. INSERT/UPDATE/DELETE/TRUNCATE are all
-- withheld: the read service never writes. The `_applied_migrations` ledger is
-- intentionally excluded (reading migration state is an operator concern, not
-- a read-service one).
GRANT SELECT ON commits TO bench_read;
GRANT SELECT ON query_measurements TO bench_read;
GRANT SELECT ON compression_times TO bench_read;
GRANT SELECT ON compression_sizes TO bench_read;
GRANT SELECT ON random_access_times TO bench_read;
GRANT SELECT ON vector_search_runs TO bench_read;

-- Future data tables are added by `migrator`-run migrations, so default-
-- privilege the read role on objects `migrator` creates, exactly as 004 does
-- for the ingest role (see 004's comment for the full rds_superuser /
-- createrole_self_grant rationale behind the self-grant dance; the same
-- single-bootstrap-master precondition applies: the master applying this held
-- ADMIN on `migrator` since it created it in 002).
DO $$
DECLARE
    self_granted boolean := false;
BEGIN
    IF NOT pg_has_role(current_user, 'migrator', 'USAGE') THEN
        GRANT migrator TO CURRENT_USER WITH INHERIT TRUE;
        self_granted := true;
    END IF;

    ALTER DEFAULT PRIVILEGES FOR ROLE migrator IN SCHEMA public
        GRANT SELECT ON TABLES TO bench_read;

    IF self_granted THEN
        REVOKE migrator FROM CURRENT_USER;
    END IF;
END$$;
