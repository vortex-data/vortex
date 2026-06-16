<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# benchmarks-website/infra

AWS infrastructure for the v4 benchmarks-website (RDS Postgres + RDS Proxy + GitHub Actions OIDC). One-shot bootstrap, idempotent re-run, fully scripted.

This directory is created in PR-1.1 and lives until Phase 5; the v3 ops at `benchmarks-website/ops/` is decommissioned separately. The Vercel reader's deploy hooks live with the Next.js app at `benchmarks-website/web/` (created in PR-4.1) and not here.

## What `provision.sh` builds

Single one-shot script. Provisions, in order:

1. The default-VPC subnet group covering every default subnet in `us-east-1` (`vortex-bench-subnet-group`).
2. A security group `vortex-bench-sg` with inbound TCP 5432 from `0.0.0.0/0` — IAM auth is the gate, not network ACLs.
3. An RDS Postgres instance `vortex-bench-prod` on `db.t4g.micro`, Postgres 16, 20 GiB GP3 storage, IAM auth enabled, RDS-managed master password (auto-rotated, stored in Secrets Manager), publicly accessible, single-AZ, 35-day backup window.
4. An RDS Proxy `vortex-bench-proxy` in front of the instance, `IAMAuth=REQUIRED`, TLS required, pulling the master credential from the Secrets-Manager-managed secret via a service-linked IAM role.
5. The GitHub OIDC provider `token.actions.githubusercontent.com` (account-scoped — created once if not present).
6. An IAM role `GitHubBenchmarkSchemaRole` trusted to GitHub Actions OIDC for the `vortex-data/vortex` repo (branches `develop` + `ct/bench-v4`) with `sts:AssumeRoleWithWebIdentity`. Permission policy: `rds-db:connect` scoped to the `migrator` Postgres user on the **instance** resource only (CI schema deploys connect to the public instance endpoint). The dead proxy grant this role carried through PR-1.6 was dropped in PR-2.1's least-privilege cleanup; the VPC-internal proxy serves only the Vercel reader.
7. An IAM role `GitHubBenchmarkIngestRole` (PR-2.1) trusted to the same OIDC provider, repo, and branches. Permission policy: `rds-db:connect` scoped to the `bench_ingest` Postgres user on the instance resource only. This is the dedicated least-privilege identity for the Phase-2 CI dual-write ingest path, deliberately separate from the schema-deploy `migrator` identity so the high-frequency ingest path can write data but never run DDL or migrations.

The `migrator` Postgres user is created by `migrations/002_iam_db_user.sql` (PR-1.3) and the `bench_ingest` user by `migrations/004_ingest_role.sql` (PR-2.1); each OIDC role's permission ARN is pre-scoped to its user. `002` is applied as the RDS master because it creates a role (it needs `CREATEROLE` and grants the `rds_iam` / schema privileges the master holds), while `003` and `004` are applied as the master because they additionally grant on master-owned objects (the ledger and the six data tables, respectively). All three run during the one-time bootstrap; the migrator-run `schema-deploy` path then records them as already-applied.

## Prerequisites

| Tool | Verification |
|---|---|
| AWS CLI v2 | `aws --version` reports `aws-cli/2.x` |
| `jq` | `jq --version` returns a version |
| Authenticated to account `245040174862` | `aws sts get-caller-identity` returns that account |
| IAM permissions on the acting principal | `rds:Create*`, `rds:Describe*`, `rds:Register*`, `iam:CreateRole`, `iam:CreatePolicy`, `iam:PutRolePolicy`, `iam:CreateOpenIDConnectProvider`, `iam:GetOpenIDConnectProvider`, `iam:UpdateAssumeRolePolicy`, `iam:GetRole`, `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress`, `ec2:DescribeVpcs`, `ec2:DescribeSubnets`, `ec2:DescribeSecurityGroups`, `ec2:CreateTags` |

If you SSO into account `245040174862` with `PowerUserAccess` or `AdministratorAccess`, you have everything you need. Confirm with:

```sh
aws sso login --profile bench            # refresh SSO session for the bench profile
aws sts get-caller-identity --profile bench   # Account should be 245040174862
```

The `bench` profile name follows the convention established in the operator's SSO setup; if you used a different profile name, substitute it (or `export AWS_PROFILE=bench` before running `./provision.sh`).

## One-command run

```sh
cd benchmarks-website/infra
./provision.sh
```

Expected duration: 5–12 minutes (the RDS instance + RDS Proxy creation each take a few minutes; the script blocks on `aws rds wait db-instance-available` for the instance and polls `describe-db-proxies` every 15s for the proxy — AWS CLI v2 has no built-in `db-proxy-available` waiter).

Expected end state: prints a summary block with the GitHub repo-variable values to set (instance endpoint, region, DB name, role ARN), plus the proxy endpoint as a separate value to carry into Vercel env config (PR-4.2). The proxy endpoint is **not** a GitHub variable.

## Idempotency

Every mutating step is gated by an existence check (`describe-*`, `get-role`, etc.) — re-running the script after a successful run is a no-op; re-running after a partial failure resumes from the first uncompleted step. Safe to interrupt at any point and re-invoke.

## Customizing

Every name / class / engine version / region is set at the top of `provision.sh` via `readonly` declarations with `${ENV:-default}` fallbacks. Override at invocation time:

```sh
TARGET_REGION=us-east-2 \
DB_INSTANCE_CLASS=db.t4g.small \
DB_ENGINE_VERSION=17.2 \
./provision.sh
```

## After provisioning — set GitHub Actions repo vars

The script prints the exact `gh variable set` commands; copy them from its output. The vars (not secrets — these are non-sensitive):

| Variable | Value | Consumed by |
|---|---|---|
| `RDS_BENCH_INSTANCE_ENDPOINT` | the public RDS **instance** hostname | CI writers — PR-1.4 schema-deploy.yml; PR-2.4 ingest workflows (direct IAM) |
| `RDS_BENCH_REGION` | `us-east-1` (or override) | All AWS-CLI invocations from CI |
| `RDS_BENCH_DB_NAME` | `vortex_bench` | All Postgres connections |
| `GH_BENCH_SCHEMA_ROLE_ARN` | the schema-deploy OIDC role ARN | `.github/workflows/schema-deploy.yml` |
| `GH_BENCH_INGEST_ROLE_ARN` | the ingest OIDC role ARN | PR-2.4 dual-write ingest workflows (`bench.yml` / `sql-benchmarks.yml` / `v3-commit-metadata.yml`) |

The RDS Proxy hostname is deliberately **not** a GitHub Actions variable: it is
VPC-internal, serves only the PR-4.2 Next.js reader on Vercel, and Vercel does
not read GitHub repo variables. `provision.sh` prints it under a separate "carry
into Vercel env (PR-4.2)" step; set it as a Vercel project env var when PR-4.2
lands. Vercel-to-VPC reachability for the proxy (VPC peering / PrivateLink / a
public-facing proxy) is itself an open PR-4.2 design item, not yet provisioned.

No secrets are needed — IAM auth is the credential. The master password is RDS-managed; it is used only for the one-time bootstrap apply (see "Schema deploys + one-time bootstrap" below), never for steady-state CI.

## Schema deploys + one-time bootstrap

Schema migrations under `migrations/` are applied by
`.github/workflows/schema-deploy.yml` (wired in PR-1.4). The workflow is
`workflow_dispatch` only: an operator triggers it manually from the GitHub
Actions UI, optionally with `dry_run: true` to report drift via
`migrate-schema.py status` without applying. That manual trigger is the deploy
gate. Per the 2026-05-29 deploy-model decision, the intended deferred change is
to ALSO trigger `apply` on push under `paths: migrations/** +
scripts/migrate-schema.py` (PR merge becomes the deploy gate) -- NOT a GitHub
Environment / manual-approval gate; execution safety comes from the per-PR
testcontainer migration test, not a human click. The runner path is included in
`paths` so a runner-only change still triggers a deploy.

Steady-state deploys connect to the public RDS **instance** endpoint
(`RDS_BENCH_INSTANCE_ENDPOINT`) as the `migrator` role using a short-lived IAM
auth token (generated client-side via `aws rds generate-db-auth-token`, signed
from the OIDC-assumed `GitHubBenchmarkSchemaRole` credentials; no password, no
IAM permission beyond `rds-db:connect`). The RDS Proxy is VPC-internal and
unreachable from off-VPC GitHub runners, so CI never uses it. TLS is
`verify-full` against Amazon's published RDS root CA bundle.

### One-time bootstrap (operator, as master)

CI can only IAM-auth as `migrator`, and `migrator` does not exist until
migration `002` creates it. So the FIRST apply must be run once by the RDS
master user, out-of-band. Two endpoint details matter:

- Use the public **instance** endpoint (`RDS_BENCH_INSTANCE_ENDPOINT`, from
  `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod
  --query 'DBInstances[0].Endpoint.Address'`) -- the same endpoint steady-state
  CI writers use. The RDS Proxy is VPC-internal and is configured
  `IAMAuth=REQUIRED` (it rejects the master password anyway), so it is never
  used for bootstrap or CI.
- The master username is the one `provision.sh` set on the instance; retrieve
  the RDS-managed master password from Secrets Manager.

`PGSSLMODE=verify-full` is mandatory here: the bootstrap transmits the master
password, so the RDS server certificate MUST be verified (a bare `require` only
encrypts, it does not authenticate the server -- a MITM could capture the
master password). Run these commands from the repository root (the `scripts/`
path below is repo-root-relative, not relative to `benchmarks-website/infra/`).
Download the CA bundle first:

```sh
curl -fsSL https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem \
  -o /tmp/rds-global-bundle.pem
export PGHOST=<instance-endpoint>           # RDS_BENCH_INSTANCE_ENDPOINT, NOT the proxy
export PGPORT=5432
export PGDATABASE=<RDS_BENCH_DB_NAME>
export PGUSER=<master-username>
# Fetch the RDS-managed master password from Secrets Manager. Assign first, then
# export, so a failed fetch or parse is FATAL: a bare `export PGPASSWORD=$(...)`
# returns 0 even when the substitution fails (export's own exit status wins),
# which would silently continue with an empty password. `jq -er` exits non-zero
# on a missing key. Non-interactive + copy-paste-safe: no interactive `read` to
# consume pasted lines, no `stty -echo` to strand the terminal, and command
# substitution preserves any password metacharacter. provision.sh prints the
# master secret ARN; substitute it here:
master_secret=$(aws secretsmanager get-secret-value \
  --secret-id <master-secret-arn> --query SecretString --output text) || exit 1
PGPASSWORD=$(printf '%s' "$master_secret" | jq -er '.password') || exit 1
export PGPASSWORD
export PGSSLMODE=verify-full
export PGSSLROOTCERT=/tmp/rds-global-bundle.pem
uv run --no-project scripts/migrate-schema.py apply
```

This applies `001` (schema) and every migration carrying the
`-- migrate-schema: requires-superuser` marker — currently `002`/`004`/`005`
(role creation + grants) and `006`/`007` (DDL on the master-owned
`query_measurements` table). The marker in each file is authoritative; the
runner refuses to apply a marked file under a non-master role. The single source
of truth for the bootstrap-ordering contract is
[`migrations/README.md`](../../migrations/README.md) § "Bootstrap ordering —
`requires-superuser` migrations"; apply every marked migration as the master
here. `003` (a ledger grant) carries no marker. After the bootstrap, *unmarked*
migrations are applied by the `schema-deploy` workflow as `migrator` against the
public instance endpoint (`RDS_BENCH_INSTANCE_ENDPOINT`) with direct IAM; the
master password is not needed again for those.

Because the bootstrap runs as master, the schema objects and the ledger are
master-owned; `003` grants `migrator` the ledger access it needs to record and
read applied migrations. A migration that `ALTER`s or adds an index to an
existing master-owned table (rather than only `CREATE`-ing new objects) must
itself carry the `requires-superuser` marker and be master-applied here — this
is what `006`/`007` do (PR-5.1.5). `migrator`'s `CREATE` on the schema suffices
only for new-object migrations.

The per-PR testcontainer migration test (`scripts/test_migrate_schema.py`)
applies every migration as a single owning role, so it does NOT model the
master/`migrator` ownership split and cannot catch a non-additive migration that
would fail in production. Until PR-2.1 resolves the role-ownership model,
additivity must be confirmed by inspection.

## Acceptance criteria for PR-1.1

Two checks must pass after running `provision.sh`:

```sh
aws rds describe-db-instances \
  --db-instance-identifier vortex-bench-prod \
  --query 'DBInstances[0].[DBInstanceStatus,IAMDatabaseAuthenticationEnabled]'
# expected: available  True

aws rds describe-db-proxies \
  --db-proxy-name vortex-bench-proxy \
  --query 'DBProxies[0].[Status,Endpoint]'
# expected: available  <proxy-endpoint>.proxy-<resource>.us-east-1.rds.amazonaws.com
```

## Tear-down (not part of PR-1.1; documented here for completeness)

```sh
# These commands use the DEFAULT resource names. provision.sh lets you override
# every name via `${ENV:-default}` (PROXY_ROLE_NAME, DB_PROXY_NAME,
# DB_INSTANCE_IDENTIFIER, DB_SUBNET_GROUP_NAME, DB_SECURITY_GROUP_NAME,
# SCHEMA_ROLE_NAME, ...); if you set
# any override at provision time, substitute the same value in the matching
# command below.
aws rds deregister-db-proxy-targets --db-proxy-name vortex-bench-proxy \
  --target-group-name default --db-instance-identifiers vortex-bench-prod
aws rds delete-db-proxy --db-proxy-name vortex-bench-proxy
aws iam delete-role-policy --role-name vortex-bench-proxy-role --policy-name read-master-secret
aws iam delete-role --role-name vortex-bench-proxy-role
aws rds delete-db-instance --db-instance-identifier vortex-bench-prod \
  --skip-final-snapshot --delete-automated-backups
aws iam delete-role-policy --role-name GitHubBenchmarkSchemaRole --policy-name rds-db-connect-migrator
aws iam delete-role --role-name GitHubBenchmarkSchemaRole
aws iam delete-role-policy --role-name GitHubBenchmarkIngestRole --policy-name rds-db-connect-ingest
aws iam delete-role --role-name GitHubBenchmarkIngestRole
aws ec2 delete-security-group --group-name vortex-bench-sg
aws rds delete-db-subnet-group --db-subnet-group-name vortex-bench-subnet-group
# OIDC provider (account-scoped): only delete if no other workflow uses it
# aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::245040174862:oidc-provider/token.actions.githubusercontent.com
```

Tear-down does NOT delete the Secrets-Manager-managed master password — that's owned by RDS and is removed automatically when the instance is deleted.

## Cost

Steady-state monthly bill once provisioned (rough). NOTE: prod `vortex-bench-prod` was upsized to
`db.r7g.large` (16 GiB) on 2026-06-16 so the load-all `?n=all` working set stays resident in cache
(see `.big-plans/ct__bench-v4-loadall-scope.md`); the `provision.sh` bootstrap default remains
`db.t4g.micro`.

| Item | Cost |
|---|---|
| RDS `db.r7g.large` (current prod; bootstrap default `db.t4g.micro` ~$13) | ~$174 |
| RDS Proxy | ~$11 (1 RDS Proxy unit at $0.015/hr) |
| 20 GiB GP3 storage | ~$2.30 |
| Backup storage (35-day window, ~20 GiB) | ~$2 |
| Secrets Manager (1 secret) | ~$0.40 |
| Data transfer (CI ingests + reader fetches) | <$1 |
| **Total** | **~$30/month** |
