#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Plan, generate facts in parallel shards, time on this machine, materialize and train.
# Every step is resumable: re-running this script only does work that isn't done yet, and each
# run of the time stage adds another timing sample for this machine.
#
# Usage: pipeline.sh <spec.toml> <work-dir> [shards]
set -euo pipefail

SPEC=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
WORK=$2
SHARDS=${3:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LAB="$ROOT/target/release/vx-lab"
DATA_ROOT=$(dirname "$SPEC")

cargo build --release -p compressor-lab --manifest-path "$ROOT/Cargo.toml"
mkdir -p "$WORK"

"$LAB" plan create --spec "$SPEC" --out "$WORK/plan" ${ALLOW_DIRTY:+--allow-dirty}

# Facts: one process per shard. A chunk's tasks share a shard, so shards never wait on each other.
for ((i = 0; i < SHARDS; i++)); do
    "$LAB" run --plan "$WORK/plan" --store "$WORK/store" --data-root "$DATA_ROOT" \
        --shard "$i/$SHARDS" ${ALLOW_DIRTY:+--allow-dirty} &
done
wait

# Timings: one process, nothing else running. Run it again (or on other machines) for more samples.
"$LAB" run --plan "$WORK/plan" --store "$WORK/store" --stages time ${ALLOW_DIRTY:+--allow-dirty}

"$LAB" plan show --plan "$WORK/plan" --store "$WORK/store"
"$LAB" materialize --plan "$WORK/plan" --store "$WORK/store" --out "$WORK/dataset"

uv run --no-project --with pandas --with scikit-learn python -W ignore \
    "$ROOT/benchmarks/compressor-feasibility/train.py" "$WORK/dataset" --export "$WORK/models"
