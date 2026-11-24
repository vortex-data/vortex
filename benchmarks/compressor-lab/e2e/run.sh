#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# End-to-end test of vx-lab on TPC-H with a local store:
#   plan -> parallel fact shards -> (re-run is a no-op) -> verify -> work queue into a second store
#   (must give the same facts) -> time -> materialize -> vx-analyze report.
# Each step is recorded in <work>/report.tsv (step, status, seconds, log) and the script keeps
# going after a failure so one run shows everything that breaks.
#
# Usage: e2e/run.sh <work-dir> [shards]
# Env:   SPEC (default specs/tpch-sf1-e2e.toml), TPCH_DIR (default <work>/tpch-sf1),
#        FACT_STAGES (default: the binary's default), TIME_RUNS (default 1), ANALYSIS (a command
#        run with the dataset dir and work dir as arguments), SKIP_BUILD=1.
set -uo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mkdir -p "$1" && cd "$1" && pwd)
SHARDS=${2:-4}
SPEC_SRC=${SPEC:-$HERE/specs/tpch-sf1-e2e.toml}
TPCH_DIR=${TPCH_DIR:-$WORK/tpch-sf1}
LAB="$ROOT/target/release/vx-lab"
REPORT="$WORK/report.tsv"
LOGS="$WORK/logs"
mkdir -p "$LOGS"
: > "$REPORT"

step() {
    local name=$1; shift
    local log="$LOGS/$name.log" start end status
    start=$(date +%s.%N)
    echo "== $name: $*" | tee "$log"
    if "$@" >> "$log" 2>&1; then status=ok; else status="FAIL($?)"; fi
    end=$(date +%s.%N)
    printf '%s\t%s\t%.1f\t%s\n' "$name" "$status" "$(echo "$end - $start" | bc)" "$log" | tee -a "$REPORT"
    [[ $status == ok ]]
}

[[ -n ${SKIP_BUILD:-} ]] || step build cargo build --release -p compressor-lab --manifest-path "$ROOT/Cargo.toml" || exit 1
step tpch-gen "$HERE/e2e/gen-tpch.sh" "$TPCH_DIR" 1 || exit 1

# The spec reads data/tpch-sf1 next to itself: stage a copy of the spec in the work dir.
mkdir -p "$WORK/spec/data"
ln -sfn "$TPCH_DIR" "$WORK/spec/data/tpch-sf1"
cp "$SPEC_SRC" "$WORK/spec/spec.toml"
SPEC="$WORK/spec/spec.toml"
DATA_ROOT="$WORK/spec"
DIRTY=(--allow-dirty)
STAGE_ARGS=()
[[ -n ${FACT_STAGES:-} ]] && STAGE_ARGS=(--stages "$FACT_STAGES")

step plan-create "$LAB" plan create --spec "$SPEC" --out "$WORK/plan" "${DIRTY[@]}" || exit 1
step plan-recreate-unchanged bash -c "'$LAB' plan create --spec '$SPEC' --out '$WORK/plan' --allow-dirty | tee /dev/stderr | grep -qi unchanged"

shards() {
    local pids=() rc=0
    for ((i = 0; i < SHARDS; i++)); do
        "$LAB" run --plan "$WORK/plan" --store "$WORK/store" --data-root "$DATA_ROOT" \
            --shard "$i/$SHARDS" "${STAGE_ARGS[@]}" "${DIRTY[@]}" > "$LOGS/shard-$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    tail -n 3 "$LOGS"/shard-*.log
    return $rc
}
step facts-shards shards
step facts-rerun-noop "$LAB" run --plan "$WORK/plan" --store "$WORK/store" --data-root "$DATA_ROOT" "${STAGE_ARGS[@]}" "${DIRTY[@]}"
# Chunk and features must reproduce exactly. Exhaustive candidates depend on in-shard decode
# timings, so their mismatches are reported but not failed on.
verify() {
    "$LAB" run --plan "$WORK/plan" --store "$WORK/store" --data-root "$DATA_ROOT" "${STAGE_ARGS[@]}" --verify "${DIRTY[@]}" 2>&1 |
        grep -v '^nondeterminism: candidates' | tee "$LOGS/verify-summary.log"
    ! grep -E '^(chunk|features):.* mismatched [1-9]' "$LOGS/verify-summary.log"
}
step facts-verify verify

# The launcher path: push the plan to a second store, drain it with local workers, and check that
# every fact matches the sharded run.
queue() {
    "$LAB" plan push --plan "$WORK/plan" --store "$WORK/store-queue" --shards "$SHARDS" &&
    "$LAB" work --store "$WORK/store-queue" --plan "$WORK/plan" --workers 2 --data-root "$DATA_ROOT" "${DIRTY[@]}" &&
    "$LAB" status --store "$WORK/store-queue" --plan "$WORK/plan"
}
same_facts() {
    local a b
    a=$("$LAB" store ledger --store "$WORK/store" --kinds chunk,features) || return 1
    b=$("$LAB" store ledger --store "$WORK/store-queue" --kinds chunk,features) || return 1
    echo "sharded: $(wc -l <<< "$a") chunk/features facts, queue: $(wc -l <<< "$b")"
    diff <(echo "$a") <(echo "$b") | head -20
    [[ -n "$a" && "$a" == "$b" ]]
}
step queue-work queue
# Exhaustive candidates depend on in-shard decode timings, so only chunk and features must match.
step queue-same-facts same_facts

for ((r = 1; r <= ${TIME_RUNS:-1}; r++)); do
    step "time-$r" "$LAB" run --plan "$WORK/plan" --store "$WORK/store" --data-root "$DATA_ROOT" --stages time "${DIRTY[@]}"
done

step plan-show "$LAB" plan show --plan "$WORK/plan" --store "$WORK/store"
step materialize "$LAB" materialize --plan "$WORK/plan" --store "$WORK/store" --out "$WORK/dataset"

# Every candidate row should either succeed or say why it does not apply; panics are bugs.
step no-panics bash -c "! grep -i ',panic' '$WORK/dataset/rows-'*.csv | head -5 | grep ."
if [[ -f "$HERE/analysis/pyproject.toml" ]]; then
    step analysis-report uv run --project "$HERE/analysis" vx-analyze report "$WORK/dataset" --out "$WORK/report.html"
fi
if [[ -n ${ANALYSIS:-} ]]; then
    step analysis $ANALYSIS "$WORK/dataset" "$WORK"
elif [[ -f "$ROOT/benchmarks/compressor-feasibility/train.py" ]]; then
    step analysis-train uv run --no-project --with pandas --with scikit-learn python -W ignore \
        "$ROOT/benchmarks/compressor-feasibility/train.py" "$WORK/dataset" --export "$WORK/models"
fi

# Compaction must not change any result; GC (dry run) must find nothing live to delete.
step store-compact "$LAB" store compact --store "$WORK/store"
step store-compact-same bash -c "'$LAB' materialize --plan '$WORK/plan' --store '$WORK/store' --out '$WORK/dataset-compacted' >/dev/null && diff -r -x dataset.json '$WORK/dataset' '$WORK/dataset-compacted'"
step store-gc-dry-run "$LAB" store gc --store "$WORK/store" --force
step store-size du -sh "$WORK/store" "$WORK/store"/*
echo; cat "$REPORT"
! grep -q FAIL "$REPORT"
