#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# List or terminate the EC2 instances launch.sh started for a plan.
#
# Usage: stop.sh --plan-id <id> [--terminate]
# Without --terminate it only lists them. Shards held by terminated workers are claimed again
# once their leases expire.
set -euo pipefail

PLAN_ID= TERMINATE=
while (($#)); do
    case $1 in
        --plan-id) PLAN_ID=$2; shift ;;
        --terminate) TERMINATE=1 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[[ -n $PLAN_ID ]] || { echo "--plan-id is required" >&2; exit 2; }

IDS=$(aws ec2 describe-instances --output text \
    --filters "Name=tag:vx-lab-plan,Values=$PLAN_ID" \
    "Name=instance-state-name,Values=pending,running,stopping" \
    --query 'Reservations[].Instances[].InstanceId')
if [[ -z $IDS ]]; then
    echo "no running instances for plan $PLAN_ID"
    exit 0
fi
aws ec2 describe-instances --instance-ids $IDS --output table \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime,Tags[?Key==`vx-lab-stage`]|[0].Value]'
if [[ -n $TERMINATE ]]; then
    # shellcheck disable=SC2086
    aws ec2 terminate-instances --instance-ids $IDS --output text \
        --query 'TerminatingInstances[].InstanceId'
fi
