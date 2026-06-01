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
6. An IAM role `GitHubBenchmarkSchemaRole` trusted to GitHub Actions OIDC for the `vortex-data/vortex` repo with `sts:AssumeRoleWithWebIdentity`. Permission policy: `rds-db:connect` scoped to the future `migrator` Postgres user via the proxy.

The `migrator` Postgres user itself is created in PR-1.3 by `migrations/002_iam_db_user.sql`. The OIDC role's permission ARN is already pre-scoped to it; no further IAM work after PR-1.3 lands.

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

Expected end state: prints a summary block with the proxy endpoint and the IAM role ARN to copy into GitHub repo variables.

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
| `RDS_BENCH_ENDPOINT` | the RDS Proxy hostname | PR-2.2 ingest workflows; PR-1.4 schema-deploy.yml; PR-4.2 Next.js reader |
| `RDS_BENCH_REGION` | `us-east-1` (or override) | All AWS-CLI invocations from CI |
| `RDS_BENCH_DB_NAME` | `vortex_bench` | All Postgres connections |
| `GH_BENCH_SCHEMA_ROLE_ARN` | the OIDC role ARN | `.github/workflows/schema-deploy.yml` |

No secrets are needed — IAM auth is the credential. The master password is RDS-managed and never used after initial provisioning.

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
aws rds deregister-db-proxy-targets --db-proxy-name vortex-bench-proxy \
  --target-group-name default --db-instance-identifiers vortex-bench-prod
aws rds delete-db-proxy --db-proxy-name vortex-bench-proxy
aws iam delete-role-policy --role-name vortex-bench-proxy-role --policy-name read-master-secret
aws iam delete-role --role-name vortex-bench-proxy-role
aws rds delete-db-instance --db-instance-identifier vortex-bench-prod \
  --skip-final-snapshot --delete-automated-backups
aws iam delete-role-policy --role-name GitHubBenchmarkSchemaRole --policy-name rds-db-connect-migrator
aws iam delete-role --role-name GitHubBenchmarkSchemaRole
aws ec2 delete-security-group --group-name vortex-bench-sg
aws rds delete-db-subnet-group --db-subnet-group-name vortex-bench-subnet-group
# OIDC provider (account-scoped): only delete if no other workflow uses it
# aws iam delete-open-id-connect-provider --open-id-connect-provider-arn arn:aws:iam::245040174862:oidc-provider/token.actions.githubusercontent.com
```

Tear-down does NOT delete the Secrets-Manager-managed master password — that's owned by RDS and is removed automatically when the instance is deleted.

## Cost

Steady-state monthly bill once provisioned (rough):

| Item | Cost |
|---|---|
| RDS `db.t4g.micro` | ~$13 |
| RDS Proxy | ~$11 (1 RDS Proxy unit at $0.015/hr) |
| 20 GiB GP3 storage | ~$2.30 |
| Backup storage (35-day window, ~20 GiB) | ~$2 |
| Secrets Manager (1 secret) | ~$0.40 |
| Data transfer (CI ingests + reader fetches) | <$1 |
| **Total** | **~$30/month** |
