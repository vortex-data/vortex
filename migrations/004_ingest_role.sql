-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- migrate-schema: requires-superuser
-- This migration creates a login role (`CREATE ROLE bench_ingest`), self-grants
-- `migrator` membership, and runs `ALTER DEFAULT PRIVILEGES FOR ROLE migrator` --
-- all requiring a master-capable executing role. It is a one-time bootstrap
-- migration the RDS master applies alongside 002 (see the header below). The
-- marker makes `migrate-schema.py` reject a non-master `apply` loudly and early,
-- before the `DO` blocks below would otherwise roll back with InsufficientPrivilege.

-- Create the `bench_ingest` login role used by the CI dual-write ingest path
-- (Phase 2) and grant it data-DML-only access to the six data tables. This role
-- is deliberately SEPARATE from the schema-deploy `migrator` role (002): the
-- ~14-writer ingest path runs on every push against a publicly-accessible
-- instance, so it gets a least-privilege identity that can write data but never
-- run DDL, migrations, or role changes. The OIDC role `GitHubBenchmarkIngestRole`
-- (provisioned in `benchmarks-website/infra/provision.sh`) scopes
-- `rds-db:connect` to this `bench_ingest` user; see
-- `benchmarks-website/infra/README.md`.
--
-- Bootstrapping / ownership: the six data tables (001) are owned by the RDS
-- master (they are created during the master bootstrap), and only a table's owner
-- can GRANT on it. This migration must therefore be applied AS the master,
-- alongside 002/003 in the one-time bootstrap; the `schema-deploy` workflow (which
-- connects AS `migrator`) then records it as already-applied and never re-runs it.
-- `migrator` cannot grant on master-owned tables, which is why the grant lives
-- here in a bootstrap-class migration rather than being issued by the deploy path.
-- One statement here is NOT runnable by ownership alone: the trailing
-- `ALTER DEFAULT PRIVILEGES FOR ROLE migrator` requires the executing role to hold
-- `migrator`'s privileges, which the non-superuser RDS master does not by default.
-- See the comment directly above that statement for how it self-grants the
-- membership it needs for the duration of the grant.
--
-- Idempotent and substrate-portable, matching 002: `CREATE ROLE` is guarded
-- (roles are cluster-global), the `rds_iam` grant is guarded behind an existence
-- check so this also applies on vanilla Postgres (local dev + the testcontainer
-- suite), and re-granting an existing privilege is a Postgres no-op.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bench_ingest') THEN
        CREATE ROLE bench_ingest WITH LOGIN;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rds_iam') THEN
        GRANT rds_iam TO bench_ingest;
    END IF;
END$$;

-- `bench_ingest` must reach objects in `public` but must NOT create any (no DDL):
-- USAGE only, never CREATE.
GRANT USAGE ON SCHEMA public TO bench_ingest;

-- Data-DML-only on the six data tables: SELECT + INSERT + UPDATE support the
-- `INSERT ... ON CONFLICT (measurement_id | commit_sha) DO UPDATE` upsert write
-- path (which needs both INSERT and UPDATE) plus read-back / reconciliation
-- (SELECT). DELETE and TRUNCATE are deliberately withheld -- the ingest path
-- never removes rows. The `_applied_migrations` ledger is intentionally excluded
-- (ingest is not a migration role), so the grant is enumerated per data table
-- rather than `ON ALL TABLES IN SCHEMA public`.
GRANT SELECT, INSERT, UPDATE ON commits TO bench_ingest;
GRANT SELECT, INSERT, UPDATE ON query_measurements TO bench_ingest;
GRANT SELECT, INSERT, UPDATE ON compression_times TO bench_ingest;
GRANT SELECT, INSERT, UPDATE ON compression_sizes TO bench_ingest;
GRANT SELECT, INSERT, UPDATE ON random_access_times TO bench_ingest;
GRANT SELECT, INSERT, UPDATE ON vector_search_runs TO bench_ingest;

-- Future data tables are added by `migrator`-run migrations (post-bootstrap
-- schema deploys connect AS `migrator`), so those tables are owned by `migrator`.
-- Default-privilege the ingest role on objects `migrator` creates so a new fact
-- table does not require a follow-up explicit grant. (The existing master-owned
-- tables above still need the explicit grants -- default privileges are not
-- retroactive.) ALTERing existing master-owned tables AS `migrator` remains a
-- separate, deferred ownership concern, tracked in the plan's Deferred work.
--
-- `ALTER DEFAULT PRIVILEGES FOR ROLE migrator` requires the executing role to hold
-- the privileges of `migrator` (has_privs_of_role). On real RDS the bootstrap
-- master is `rds_superuser`, which is NOT a true superuser: when it created
-- `migrator` (002) PostgreSQL 16 auto-granted it membership WITH ADMIN TRUE,
-- INHERIT FALSE, SET FALSE (the `createrole_self_grant` default), so the master
-- neither inherits `migrator` nor can `SET ROLE migrator`. A bare ADP FOR ROLE, or
-- a `SET ROLE migrator` wrapper, therefore fails with InsufficientPrivilege and
-- rolls back the whole 004 bootstrap. (A true superuser, used by local dev and the
-- testcontainer suite, bypasses the check, which is why those tests pass while the
-- prod master bootstrap would fail.) The master does hold ADMIN on `migrator`, so
-- it can self-grant the INHERIT membership the ADP needs for the duration of this
-- statement, then revoke it to restore the prior role graph. The grant/revoke is
-- skipped when the executing role already has `migrator`'s privileges (a true
-- superuser, or a cluster whose `createrole_self_grant` includes `inherit`), so
-- this is a no-op there.
--
-- Precondition: the self-grant relies on the executing role holding ADMIN on
-- `migrator`, which holds because the SAME master created `migrator` in 002
-- during this one bootstrap (a CREATEROLE creator gets ADMIN on the role it
-- creates). This is the single-bootstrap-master ordering: 002 and 004 are applied
-- together by the one master. A role that did NOT create `migrator` and lacks
-- ADMIN on it would fail at the `GRANT` below, not the ADP.
DO $$
DECLARE
    self_granted boolean := false;
BEGIN
    IF NOT pg_has_role(current_user, 'migrator', 'USAGE') THEN
        GRANT migrator TO CURRENT_USER WITH INHERIT TRUE;
        self_granted := true;
    END IF;

    ALTER DEFAULT PRIVILEGES FOR ROLE migrator IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE ON TABLES TO bench_ingest;

    IF self_granted THEN
        REVOKE migrator FROM CURRENT_USER;
    END IF;
END$$;
