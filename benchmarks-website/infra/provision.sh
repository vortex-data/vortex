#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# One-time bootstrap of the benchmarks-website v4 AWS infrastructure:
# RDS Postgres + RDS Proxy + the GitHub Actions OIDC role for schema
# deploys. Idempotent: safe to re-run; each step checks for an existing
# resource and skips creation if found. See infra/README.md for the
# full operator runbook.
#
# Prerequisites (verified at start of script):
#   1. AWS CLI v2 installed and authenticated (e.g. `aws sso login`).
#   2. `jq` installed (used to parse aws-cli JSON output).
#   3. The acting principal has permission to create RDS instances, RDS
#      proxies, IAM roles + policies, EC2 security groups, and DB
#      subnet groups in the target account / region.
#
# What this script provisions, in order:
#   1. Verifies AWS identity + region (account `375504701696` in
#      `us-east-1` by default; both overridable via env).
#   2. Identifies the default VPC and its subnets.
#   3. Creates a DB subnet group `vortex-bench-subnet-group` covering
#      every subnet in the default VPC.
#   4. Creates a security group `vortex-bench-sg` allowing inbound
#      TCP 5432 from anywhere (`0.0.0.0/0`) - IAM auth is the gate, not
#      network ACLs.
#   5. Creates the RDS Postgres instance `vortex-bench-prod`:
#      db.t4g.micro, postgres 16, IAM auth enabled, RDS-managed master
#      password (auto-rotated, stored in Secrets Manager).
#   6. Waits for the instance to reach `available`.
#   7. Creates the RDS Proxy `vortex-bench-proxy` in front of the RDS
#      instance, also with IAM auth.
#   8. Creates the IAM role `GitHubBenchmarkSchemaRole` trusted to
#      `token.actions.githubusercontent.com` for the
#      `vortex-data/vortex` repo, with `rds-db:connect` on the future
#      `migrator` Postgres user (created by migration 002 in PR-1.3).
#   9. Prints a summary with the endpoint, role ARN, and the GitHub
#      Actions vars to set.
#
# Re-run safety: every aws-cli mutation goes through `ensure_*`
# functions that check for an existing resource first. A partially
# failed run can be resumed by re-invoking the script.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration. Override via environment as needed.
# -----------------------------------------------------------------------------

readonly TARGET_ACCOUNT="${TARGET_ACCOUNT:-245040174862}"
readonly TARGET_REGION="${TARGET_REGION:-us-east-1}"
readonly DB_INSTANCE_IDENTIFIER="${DB_INSTANCE_IDENTIFIER:-vortex-bench-prod}"
readonly DB_PROXY_NAME="${DB_PROXY_NAME:-vortex-bench-proxy}"
readonly DB_SUBNET_GROUP_NAME="${DB_SUBNET_GROUP_NAME:-vortex-bench-subnet-group}"
readonly DB_SECURITY_GROUP_NAME="${DB_SECURITY_GROUP_NAME:-vortex-bench-sg}"
readonly DB_NAME="${DB_NAME:-vortex_bench}"
readonly DB_INSTANCE_CLASS="${DB_INSTANCE_CLASS:-db.t4g.micro}"
readonly DB_ENGINE_VERSION="${DB_ENGINE_VERSION:-16.4}"
readonly DB_ALLOCATED_STORAGE_GB="${DB_ALLOCATED_STORAGE_GB:-20}"
readonly DB_MASTER_USERNAME="${DB_MASTER_USERNAME:-postgres}"
readonly SCHEMA_ROLE_NAME="${SCHEMA_ROLE_NAME:-GitHubBenchmarkSchemaRole}"
readonly GITHUB_REPO="${GITHUB_REPO:-vortex-data/vortex}"
# Postgres role created by migrations/002 in PR-1.3; the OIDC role's
# rds-db:connect permission is scoped to this user.
readonly PG_MIGRATOR_ROLE="${PG_MIGRATOR_ROLE:-migrator}"

readonly TAG_KEY_PROJECT="Project"
readonly TAG_VAL_PROJECT="vortex-benchmarks"
readonly TAG_KEY_OWNER="ManagedBy"
readonly TAG_VAL_OWNER="benchmarks-website/infra/provision.sh"

# -----------------------------------------------------------------------------
# Logging.
# -----------------------------------------------------------------------------

log()  { printf '[%s] %s\n'  "$(date -u +%H:%M:%SZ)" "$*" >&2; }
warn() { printf '[%s] WARN: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die()  { printf '[%s] FATAL: %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Step 0: Verify prerequisites.
# -----------------------------------------------------------------------------

verify_prereqs() {
    log "Step 0: Verifying prerequisites."

    command -v aws >/dev/null || die "aws CLI not found. Install via 'brew install awscli'."
    command -v jq  >/dev/null || die "jq not found. Install via 'brew install jq'."

    local aws_version_raw aws_major
    aws_version_raw=$(aws --version 2>&1)
    aws_major=$(printf '%s\n' "$aws_version_raw" | sed -E -n 's|.*aws-cli/([0-9]+).*|\1|p')
    [[ "$aws_major" =~ ^[0-9]+$ ]] \
        || die "Could not parse aws CLI version from: ${aws_version_raw}"
    [ "$aws_major" -ge 2 ] || die "aws CLI v2 required; found v${aws_major}."

    local caller_account caller_err
    # PID-suffixed path is portable across GNU coreutils (Amazon Linux,
    # CloudShell) and BSD (macOS) mktemp variants. GNU mktemp -t requires
    # X's in the template; BSD mktemp -t takes a plain prefix. Sidestep
    # both by constructing the path directly.
    caller_err="${TMPDIR:-/tmp}/provision-caller-err.$$"
    # shellcheck disable=SC2064
    trap "rm -f '${caller_err}'" EXIT
    caller_account=$(aws sts get-caller-identity --query 'Account' --output text 2>"$caller_err") \
        || die "aws sts get-caller-identity failed: $(cat "$caller_err")"
    [ "$caller_account" = "$TARGET_ACCOUNT" ] \
        || die "Authenticated to account ${caller_account}, but TARGET_ACCOUNT=${TARGET_ACCOUNT}."

    export AWS_REGION="$TARGET_REGION"
    export AWS_DEFAULT_REGION="$TARGET_REGION"
    log "  ok: account ${TARGET_ACCOUNT}, region ${TARGET_REGION}."
}

# -----------------------------------------------------------------------------
# Step 1: Identify the default VPC and its subnets.
# -----------------------------------------------------------------------------

discover_default_vpc() {
    log "Step 1: Discovering default VPC and subnets."

    DEFAULT_VPC_ID=$(aws ec2 describe-vpcs \
        --filters Name=is-default,Values=true \
        --query 'Vpcs[0].VpcId' --output text)
    [ "$DEFAULT_VPC_ID" != "None" ] || die "No default VPC found in ${TARGET_REGION}."
    log "  default VPC: ${DEFAULT_VPC_ID}"

    # bash-3.2-compatible read loop (macOS ships bash 3.2; `mapfile` is bash 4+).
    DEFAULT_SUBNETS=()
    while IFS= read -r _subnet; do
        DEFAULT_SUBNETS+=("$_subnet")
    done < <(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=${DEFAULT_VPC_ID}" \
        --query 'Subnets[].SubnetId' --output text | tr '\t' '\n')
    [ "${#DEFAULT_SUBNETS[@]}" -ge 2 ] \
        || die "Found ${#DEFAULT_SUBNETS[@]} subnets in default VPC; need at least 2 for RDS subnet group."
    log "  subnets: ${DEFAULT_SUBNETS[*]}"
}

# -----------------------------------------------------------------------------
# Step 2: DB subnet group.
# -----------------------------------------------------------------------------

ensure_db_subnet_group() {
    log "Step 2: DB subnet group ${DB_SUBNET_GROUP_NAME}."

    if aws rds describe-db-subnet-groups --db-subnet-group-name "$DB_SUBNET_GROUP_NAME" \
            >/dev/null 2>&1; then
        log "  exists; skipping."
        return 0
    fi

    aws rds create-db-subnet-group \
        --db-subnet-group-name "$DB_SUBNET_GROUP_NAME" \
        --db-subnet-group-description "vortex-benchmarks RDS subnet group (default VPC)" \
        --subnet-ids "${DEFAULT_SUBNETS[@]}" \
        --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
               "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}" \
        >/dev/null
    log "  created."
}

# -----------------------------------------------------------------------------
# Step 3: Security group (open 5432; IAM auth is the gate).
# -----------------------------------------------------------------------------

ensure_security_group() {
    log "Step 3: Security group ${DB_SECURITY_GROUP_NAME}."

    DB_SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=${DB_SECURITY_GROUP_NAME}" \
                  "Name=vpc-id,Values=${DEFAULT_VPC_ID}" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

    if [ "$DB_SG_ID" != "None" ]; then
        log "  exists: ${DB_SG_ID}; skipping creation."
    else
        DB_SG_ID=$(aws ec2 create-security-group \
            --group-name "$DB_SECURITY_GROUP_NAME" \
            --description "vortex-benchmarks Postgres ingress (IAM-auth gated)" \
            --vpc-id "$DEFAULT_VPC_ID" \
            --query 'GroupId' --output text)
        log "  created: ${DB_SG_ID}"

        aws ec2 create-tags --resources "$DB_SG_ID" \
            --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
                   "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}"
    fi

    # Authorize inbound 5432 from anywhere if not already.
    local existing_rule
    existing_rule=$(aws ec2 describe-security-groups --group-ids "$DB_SG_ID" \
        --query 'SecurityGroups[0].IpPermissions[?FromPort==`5432`]' --output json)
    if [ "$(echo "$existing_rule" | jq 'length')" -eq 0 ]; then
        aws ec2 authorize-security-group-ingress \
            --group-id "$DB_SG_ID" \
            --protocol tcp --port 5432 --cidr 0.0.0.0/0 \
            >/dev/null
        log "  inbound 5432 from 0.0.0.0/0 authorized."
    else
        log "  inbound 5432 already authorized; skipping."
    fi
}

# -----------------------------------------------------------------------------
# Step 4: RDS Postgres instance.
# -----------------------------------------------------------------------------

ensure_rds_instance() {
    log "Step 4: RDS instance ${DB_INSTANCE_IDENTIFIER}."

    if aws rds describe-db-instances --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
            >/dev/null 2>&1; then
        log "  exists; skipping creation."
    else
        aws rds create-db-instance \
            --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
            --db-instance-class "$DB_INSTANCE_CLASS" \
            --engine postgres \
            --engine-version "$DB_ENGINE_VERSION" \
            --allocated-storage "$DB_ALLOCATED_STORAGE_GB" \
            --storage-type gp3 \
            --master-username "$DB_MASTER_USERNAME" \
            --manage-master-user-password \
            --db-name "$DB_NAME" \
            --db-subnet-group-name "$DB_SUBNET_GROUP_NAME" \
            --vpc-security-group-ids "$DB_SG_ID" \
            --enable-iam-database-authentication \
            --backup-retention-period 35 \
            --publicly-accessible \
            --no-multi-az \
            --no-auto-minor-version-upgrade \
            --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
                   "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}" \
            >/dev/null
        log "  creation initiated; waiting for instance to become available (typ. 5-10 min)."
    fi

    aws rds wait db-instance-available --db-instance-identifier "$DB_INSTANCE_IDENTIFIER"
    log "  available."

    DB_ENDPOINT=$(aws rds describe-db-instances \
        --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
        --query 'DBInstances[0].Endpoint.Address' --output text)
    DB_INSTANCE_ARN=$(aws rds describe-db-instances \
        --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
        --query 'DBInstances[0].DBInstanceArn' --output text)
    DB_RESOURCE_ID=$(aws rds describe-db-instances \
        --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
        --query 'DBInstances[0].DbiResourceId' --output text)
    DB_MASTER_SECRET_ARN=$(aws rds describe-db-instances \
        --db-instance-identifier "$DB_INSTANCE_IDENTIFIER" \
        --query 'DBInstances[0].MasterUserSecret.SecretArn' --output text)
    log "  endpoint: ${DB_ENDPOINT}"
    log "  resource-id: ${DB_RESOURCE_ID}"
}

# -----------------------------------------------------------------------------
# Step 5: RDS Proxy.
# -----------------------------------------------------------------------------

ensure_rds_proxy() {
    log "Step 5: RDS Proxy ${DB_PROXY_NAME}."

    # Proxy needs its own IAM role to read the master secret.
    local proxy_role_name="vortex-bench-proxy-role"
    local proxy_role_arn

    if proxy_role_arn=$(aws iam get-role --role-name "$proxy_role_name" \
            --query 'Role.Arn' --output text 2>/dev/null); then
        log "  proxy IAM role exists: ${proxy_role_arn}"
    else
        proxy_role_arn=$(aws iam create-role \
            --role-name "$proxy_role_name" \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "rds.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
                   "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}" \
            --query 'Role.Arn' --output text)
        log "  proxy IAM role created: ${proxy_role_arn}"
    fi

    aws iam put-role-policy \
        --role-name "$proxy_role_name" \
        --policy-name "read-master-secret" \
        --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": "${DB_MASTER_SECRET_ARN}"
  }]
}
EOF
)" >/dev/null
    log "  proxy IAM role policy attached."

    if aws rds describe-db-proxies --db-proxy-name "$DB_PROXY_NAME" >/dev/null 2>&1; then
        log "  proxy exists; skipping creation."
    else
        aws rds create-db-proxy \
            --db-proxy-name "$DB_PROXY_NAME" \
            --engine-family POSTGRESQL \
            --auth "AuthScheme=SECRETS,SecretArn=${DB_MASTER_SECRET_ARN},IAMAuth=REQUIRED" \
            --role-arn "$proxy_role_arn" \
            --vpc-subnet-ids "${DEFAULT_SUBNETS[@]}" \
            --vpc-security-group-ids "$DB_SG_ID" \
            --require-tls \
            --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
                   "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}" \
            >/dev/null
        log "  proxy creation initiated."
    fi

    # aws cli v2 has no built-in `wait db-proxy-available` waiter; poll
    # `describe-db-proxies` until Status == "available" (15 min cap).
    local proxy_status proxy_elapsed=0 proxy_timeout=900 proxy_interval=15
    while [ "$proxy_elapsed" -lt "$proxy_timeout" ]; do
        proxy_status=$(aws rds describe-db-proxies --db-proxy-name "$DB_PROXY_NAME" \
            --query 'DBProxies[0].Status' --output text 2>/dev/null || echo "unknown")
        case "$proxy_status" in
            available)
                break
                ;;
            creating|modifying|reactivating)
                sleep "$proxy_interval"
                proxy_elapsed=$((proxy_elapsed + proxy_interval))
                ;;
            *)
                die "proxy entered unexpected status '${proxy_status}'; aborting"
                ;;
        esac
    done
    [ "$proxy_status" = "available" ] \
        || die "timeout waiting for proxy to become available (last status: ${proxy_status})"
    log "  proxy available."

    # Register the RDS instance as a target.
    local target_groups
    target_groups=$(aws rds describe-db-proxy-target-groups \
        --db-proxy-name "$DB_PROXY_NAME" --output json)
    if [ "$(echo "$target_groups" | jq '.TargetGroups | length')" -eq 0 ] \
       || ! aws rds describe-db-proxy-targets --db-proxy-name "$DB_PROXY_NAME" \
            --query 'Targets[?RdsResourceId==`'"$DB_RESOURCE_ID"'`]' --output json \
            | jq -e 'length > 0' >/dev/null; then
        aws rds register-db-proxy-targets \
            --db-proxy-name "$DB_PROXY_NAME" \
            --target-group-name default \
            --db-instance-identifiers "$DB_INSTANCE_IDENTIFIER" \
            >/dev/null
        log "  RDS instance registered as proxy target."
    else
        log "  proxy already targets the RDS instance; skipping."
    fi

    PROXY_ENDPOINT=$(aws rds describe-db-proxies --db-proxy-name "$DB_PROXY_NAME" \
        --query 'DBProxies[0].Endpoint' --output text)
    log "  proxy endpoint: ${PROXY_ENDPOINT}"
}

# -----------------------------------------------------------------------------
# Step 6: GitHub Actions OIDC IAM role for schema deploys.
# -----------------------------------------------------------------------------

ensure_oidc_provider() {
    # The GitHub OIDC provider is account-scoped; one provider per account.
    local provider_url="token.actions.githubusercontent.com"
    local provider_arn="arn:aws:iam::${TARGET_ACCOUNT}:oidc-provider/${provider_url}"

    if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$provider_arn" \
            >/dev/null 2>&1; then
        log "  OIDC provider exists; skipping creation."
    else
        aws iam create-open-id-connect-provider \
            --url "https://${provider_url}" \
            --client-id-list "sts.amazonaws.com" \
            --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
            >/dev/null
        log "  OIDC provider created."
    fi
    OIDC_PROVIDER_ARN="$provider_arn"
}

ensure_schema_role() {
    log "Step 6: GitHub Actions OIDC role ${SCHEMA_ROLE_NAME}."
    ensure_oidc_provider

    # Trust-policy sub-claim is scoped to the specific branches the
    # schema-deploy.yml workflow runs on (`develop` + `ct/bench-v4`).
    # Without a `schema-deploy` GitHub Environment (operator lacks
    # repo-admin to create one), this sub-claim restriction is the
    # primary gate against unauthorized OIDC role assumption.
    local trust_policy
    trust_policy=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${OIDC_PROVIDER_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike":   {"token.actions.githubusercontent.com:sub": [
        "repo:${GITHUB_REPO}:ref:refs/heads/develop",
        "repo:${GITHUB_REPO}:ref:refs/heads/ct/bench-v4"
      ]}
    }
  }]
}
EOF
)

    if SCHEMA_ROLE_ARN=$(aws iam get-role --role-name "$SCHEMA_ROLE_NAME" \
            --query 'Role.Arn' --output text 2>/dev/null); then
        log "  exists: ${SCHEMA_ROLE_ARN}; updating trust policy."
        aws iam update-assume-role-policy \
            --role-name "$SCHEMA_ROLE_NAME" \
            --policy-document "$trust_policy"
    else
        SCHEMA_ROLE_ARN=$(aws iam create-role \
            --role-name "$SCHEMA_ROLE_NAME" \
            --assume-role-policy-document "$trust_policy" \
            --description "GitHub Actions OIDC role for benchmarks-website Postgres schema deploys" \
            --tags "Key=${TAG_KEY_PROJECT},Value=${TAG_VAL_PROJECT}" \
                   "Key=${TAG_KEY_OWNER},Value=${TAG_VAL_OWNER}" \
            --query 'Role.Arn' --output text)
        log "  created: ${SCHEMA_ROLE_ARN}"
    fi

    # rds-db:connect for the migrator Postgres user via the proxy resource.
    # The Postgres user `migrator` is created in PR-1.3 migrations/002.
    #
    # DBProxyArn shape: `arn:aws:rds:<region>:<account>:db-proxy:prx-<id>` —
    # so `awk -F: '{print $NF}'` already returns `prx-<id>` including the
    # `prx-` prefix. Do NOT prepend another `prx-` (would produce
    # `prx-prx-<id>` and break IAM token issuance through the proxy).
    local rds_arn_account="$TARGET_ACCOUNT"
    local proxy_resource_id
    proxy_resource_id=$(aws rds describe-db-proxies --db-proxy-name "$DB_PROXY_NAME" \
        --query 'DBProxies[0].DBProxyArn' --output text | awk -F: '{print $NF}')

    local permissions_policy
    permissions_policy=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["rds-db:connect"],
    "Resource": [
      "arn:aws:rds-db:${TARGET_REGION}:${rds_arn_account}:dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}",
      "arn:aws:rds-db:${TARGET_REGION}:${rds_arn_account}:dbuser:${proxy_resource_id}/${PG_MIGRATOR_ROLE}"
    ]
  }]
}
EOF
)

    aws iam put-role-policy \
        --role-name "$SCHEMA_ROLE_NAME" \
        --policy-name "rds-db-connect-migrator" \
        --policy-document "$permissions_policy" \
        >/dev/null
    log "  inline permissions policy applied (rds-db:connect for ${PG_MIGRATOR_ROLE})."
}

# -----------------------------------------------------------------------------
# Step 7: Summary.
# -----------------------------------------------------------------------------

print_summary() {
    cat <<EOF

=========================================================================
PROVISIONING COMPLETE.

RDS instance     : ${DB_INSTANCE_IDENTIFIER}
RDS endpoint     : ${DB_ENDPOINT}:5432
RDS resource-id  : ${DB_RESOURCE_ID}
Master secret    : ${DB_MASTER_SECRET_ARN}
RDS Proxy        : ${DB_PROXY_NAME}
Proxy endpoint   : ${PROXY_ENDPOINT}:5432
Schema role ARN  : ${SCHEMA_ROLE_ARN}

NEXT STEPS:

1. Set these GitHub Actions repository VARIABLES (not secrets):
     gh variable set RDS_BENCH_ENDPOINT --body "${PROXY_ENDPOINT}"
     gh variable set RDS_BENCH_REGION   --body "${TARGET_REGION}"
     gh variable set RDS_BENCH_DB_NAME  --body "${DB_NAME}"
     gh variable set GH_BENCH_SCHEMA_ROLE_ARN --body "${SCHEMA_ROLE_ARN}"

2. Verify acceptance criteria for PR-1.1:
     aws rds describe-db-instances --db-instance-identifier ${DB_INSTANCE_IDENTIFIER} \\
       --query 'DBInstances[0].[DBInstanceStatus,IAMDatabaseAuthenticationEnabled]'
     # expected: available  True
     aws rds describe-db-proxies --db-proxy-name ${DB_PROXY_NAME} \\
       --query 'DBProxies[0].[Status,Endpoint]'
     # expected: available  <endpoint>

3. PR-1.3 will create the Postgres '${PG_MIGRATOR_ROLE}' user via
   migrations/002. The OIDC role's permission policy is already
   scoped to that user; nothing more to do here.

=========================================================================
EOF
}

# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------

main() {
    verify_prereqs
    discover_default_vpc
    ensure_db_subnet_group
    ensure_security_group
    ensure_rds_instance
    ensure_rds_proxy
    ensure_schema_role
    print_summary
}

main "$@"
