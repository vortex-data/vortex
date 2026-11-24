#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# EC2 user-data for a vx-lab worker: fetch the code a pushed plan was made with, run
# `vx-lab work` on every core until no shard is left, then power off. Launch instances with
# `--instance-initiated-shutdown-behavior terminate` so powering off ends the bill.
#
# `launch.sh` fills in the settings below. They can also be set in the environment.
#   STORE        the store holding the pushed plan, e.g. s3://my-bucket/lab (S3 needs the
#                S3 store backend)
#   PLAN_ID      the pushed plan
#   REPO         the git repository the lab is built from
#   WORKERS      worker processes; defaults to one per core
#   MAX_HOURS    hard limit on the instance's life
#   STAGE        `facts` (default) runs `vx-lab work`; `time` runs the timing stage once
#   DATA_URL     optional S3 prefix synced to /opt/vx-lab/data, for Parquet sources
set -euo pipefail

: "${STORE:=__STORE__}"
: "${PLAN_ID:=__PLAN_ID__}"
: "${REPO:=__REPO__}"
: "${WORKERS:=__WORKERS__}"
: "${MAX_HOURS:=__MAX_HOURS__}"
: "${STAGE:=__STAGE__}"
: "${DATA_URL:=__DATA_URL__}"
[[ $WORKERS == __*__ || -z $WORKERS ]] && WORKERS=$(nproc)
[[ $MAX_HOURS == __*__ || -z $MAX_HOURS ]] && MAX_HOURS=12
[[ $STAGE == __*__ || -z $STAGE ]] && STAGE=facts
[[ $DATA_URL == __*__ ]] && DATA_URL=

# Whatever happens below, the instance does not outlive MAX_HOURS.
shutdown -h "+$((MAX_HOURS * 60))" || true
trap 'shutdown -h now' EXIT

export HOME=${HOME:-/root}
if ! command -v git >/dev/null || ! command -v cc >/dev/null || ! command -v aws >/dev/null; then
    if command -v dnf >/dev/null; then
        dnf install -y -q git gcc gcc-c++ make python3 awscli
    else
        apt-get update -q && apt-get install -y -q git build-essential python3 awscli
    fi
fi

LAB=/opt/vx-lab
SRC=$LAB/src
exec > >(tee -a /var/log/vx-lab.log) 2>&1
echo "vx-lab bootstrap: plan $PLAN_ID, store $STORE, $WORKERS workers, stage $STAGE"

fetch() { # fetch <store path> <local file>
    if [[ $STORE == s3://* ]]; then
        aws s3 cp --only-show-errors "$STORE/$1" "$2"
    else
        cp "$STORE/$1" "$2"
    fi
}

mkdir -p "$LAB"
fetch "queue/$PLAN_ID/plan/plan.json" "$LAB/plan.json"
COMMIT=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["identity"]["git_commit"])' \
    "$LAB/plan.json")
ARCH=$(uname -m)

# The binary checks its own commit against the checkout it was built in, so the checkout
# always lives at $SRC, on the build machine and on every worker.
if [[ ! -d $SRC/.git ]]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$REPO"
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
git -C "$SRC" checkout -q --detach FETCH_HEAD

BIN=$SRC/target/release/vx-lab
CACHED="bin/$COMMIT-$ARCH/vx-lab"
mkdir -p "$(dirname "$BIN")"
if ! fetch "$CACHED" "$BIN" 2>/dev/null; then
    echo "no cached binary for $COMMIT on $ARCH; building"
    if ! command -v cargo >/dev/null; then
        curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    (cd "$SRC" && cargo build --release -p compressor-lab --bin vx-lab)
    # Share it, so later instances skip the build. Racing builders write the same bytes.
    if [[ $STORE == s3://* ]]; then
        aws s3 cp --only-show-errors "$BIN" "$STORE/$CACHED"
    else
        mkdir -p "$STORE/$(dirname "$CACHED")" && cp "$BIN" "$STORE/$CACHED"
    fi
fi
chmod +x "$BIN"

DATA_ROOT=$LAB/data
mkdir -p "$DATA_ROOT"
if [[ -n $DATA_URL ]]; then
    aws s3 sync --only-show-errors "$DATA_URL" "$DATA_ROOT"
fi

case $STAGE in
    facts)
        "$BIN" work --store "$STORE" --plan-id "$PLAN_ID" --workers "$WORKERS" \
            --data-root "$DATA_ROOT"
        ;;
    time)
        # Timing needs the plan as a directory and a quiet machine: one process, nothing else.
        mkdir -p "$LAB/plan"
        cp "$LAB/plan.json" "$LAB/plan/plan.json"
        fetch "queue/$PLAN_ID/plan/tasks.jsonl" "$LAB/plan/tasks.jsonl"
        "$BIN" run --plan "$LAB/plan" --store "$STORE" --stages time --data-root "$DATA_ROOT"
        ;;
    *)
        echo "unknown STAGE $STAGE" >&2
        exit 1
        ;;
esac
"$BIN" status --store "$STORE" --plan-id "$PLAN_ID" || true
echo "vx-lab bootstrap: finished"
