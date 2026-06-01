-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- Create the `migrator` login role used by the CI schema-deploy workflow.
-- CI authenticates to RDS Proxy with a short-lived IAM auth token (no
-- password); on real RDS, membership in the built-in `rds_iam` role is what
-- binds a Postgres role to IAM-token authentication. The OIDC role
-- `GitHubBenchmarkSchemaRole` provisioned in PR-1.1 already scopes
-- `rds-db:connect` to this `migrator` user; see
-- `benchmarks-website/infra/README.md`.
--
-- Bootstrapping: the FIRST `migrate-schema.py apply` is run by the RDS master
-- user (operator bootstrap, or the first PR-1.4 schema-deploy run), which is
-- what creates this role. Once `migrator` exists, subsequent schema deploys
-- connect AS `migrator`; this migration is already recorded in
-- `public._applied_migrations` by then and never re-runs, so `migrator` itself
-- never needs the `CREATEROLE` privilege.
--
-- Idempotent and substrate-portable. `CREATE ROLE` is guarded because roles are
-- cluster-global and survive a database drop, so a re-run against a reused
-- cluster must not error. The `rds_iam` grant is guarded behind an existence
-- check so this migration also applies cleanly on a vanilla Postgres (local dev
-- and the runner's testcontainer suite), where the `rds_iam` role does not
-- exist.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'migrator') THEN
        CREATE ROLE migrator WITH LOGIN;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rds_iam') THEN
        GRANT rds_iam TO migrator;
    END IF;
END$$;

-- `migrator` runs forward-only DDL migrations, so it needs to create objects in
-- the `public` schema. Re-granting is a no-op, so this statement is idempotent.
GRANT CREATE, USAGE ON SCHEMA public TO migrator;
