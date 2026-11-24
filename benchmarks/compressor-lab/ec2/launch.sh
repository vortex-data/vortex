#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Start EC2 workers for a pushed plan. Each instance runs bootstrap.sh, works until no shard is
# left, and terminates itself. Every instance is tagged `vx-lab-plan=<plan id>`, so `stop.sh`
# can find them.
#
# Usage: launch.sh --store s3://bucket/lab --plan-id <id> [--count 4] [--type c8g.4xlarge]
#                  [--on-demand] [--stage facts|time] [--max-hours 12] [--profile <iam profile>]
#                  [--subnet <id>] [--security-group <id>] [--key <name>] [--ami <id>]
#                  [--repo <git url>] [--data-url s3://...] [--dry-run]
#
# Workers default to spot c8g (arm64). For timing use `--stage time --on-demand --count 1`.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
COUNT=1 TYPE=c8g.4xlarge SPOT=1 STAGE=facts MAX_HOURS=12 PROFILE= SUBNET= SG= KEY= AMI=
REPO=https://github.com/vortex-data/vortex DATA_URL= STORE= PLAN_ID= DRY_RUN=
while (($#)); do
    case $1 in
        --store) STORE=$2; shift ;;
        --plan-id) PLAN_ID=$2; shift ;;
        --count) COUNT=$2; shift ;;
        --type) TYPE=$2; shift ;;
        --on-demand) SPOT= ;;
        --stage) STAGE=$2; shift ;;
        --max-hours) MAX_HOURS=$2; shift ;;
        --profile) PROFILE=$2; shift ;;
        --subnet) SUBNET=$2; shift ;;
        --security-group) SG=$2; shift ;;
        --key) KEY=$2; shift ;;
        --ami) AMI=$2; shift ;;
        --repo) REPO=$2; shift ;;
        --data-url) DATA_URL=$2; shift ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[[ -n $STORE && -n $PLAN_ID ]] || { echo "--store and --plan-id are required" >&2; exit 2; }

if [[ -z $AMI ]]; then
    ARCH=$(aws ec2 describe-instance-types --instance-types "$TYPE" \
        --query 'InstanceTypes[0].ProcessorInfo.SupportedArchitectures[0]' --output text)
    AMI=$(aws ssm get-parameter --output text --query Parameter.Value \
        --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-$ARCH")
fi

USER_DATA=$(mktemp)
trap 'rm -f "$USER_DATA"' EXIT
sed -e "s|__STORE__|$STORE|" -e "s|__PLAN_ID__|$PLAN_ID|" -e "s|__REPO__|$REPO|" \
    -e "s|__WORKERS__||" -e "s|__MAX_HOURS__|$MAX_HOURS|" -e "s|__STAGE__|$STAGE|" \
    -e "s|__DATA_URL__|$DATA_URL|" "$HERE/bootstrap.sh" >"$USER_DATA"

ARGS=(
    --image-id "$AMI" --instance-type "$TYPE" --count "$COUNT"
    --instance-initiated-shutdown-behavior terminate
    --user-data "file://$USER_DATA"
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}'
    --tag-specifications
    "ResourceType=instance,Tags=[{Key=Name,Value=vx-lab-$STAGE},{Key=vx-lab-plan,Value=$PLAN_ID},{Key=vx-lab-stage,Value=$STAGE}]"
)
[[ -n $SPOT ]] && ARGS+=(--instance-market-options
    'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
[[ -n $PROFILE ]] && ARGS+=(--iam-instance-profile "Name=$PROFILE")
[[ -n $SUBNET ]] && ARGS+=(--subnet-id "$SUBNET")
[[ -n $SG ]] && ARGS+=(--security-group-ids "$SG")
[[ -n $KEY ]] && ARGS+=(--key-name "$KEY")

if [[ -n $DRY_RUN ]]; then
    printf 'aws ec2 run-instances'; printf ' %q' "${ARGS[@]}"; echo
    exit 0
fi
aws ec2 run-instances "${ARGS[@]}" --query 'Instances[].InstanceId' --output text
