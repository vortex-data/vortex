#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -Eeu -o pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

if [[ -f /tmp/vortex-benchmark.env ]]; then
    # shellcheck disable=SC1091
    source /tmp/vortex-benchmark.env
fi

if [[ -z "${BENCH_CPUS:-}" ]]; then
    if command -v numactl >/dev/null 2>&1; then
        # All CPUs on NUMA node 0, skipping CPUs 0-1 to avoid OS interference
        BENCH_CPUS=$(numactl --hardware | awk '/^node 0 cpus:/{sep=""; for(i=4;i<=NF;i++){if($i+0>1){printf "%s%s",sep,$i; sep=","}}}')
    else
        BENCH_CPUS="2-$(($(nproc) - 1))"
    fi
fi

# Optionally cap the benchmark at its first BENCH_MAX_CPUS CPUs, so a run on a large
# machine measures a fixed core budget.
if [[ -n "${BENCH_MAX_CPUS:-}" ]]; then
    capped=()
    IFS=, read -ra parts <<< "$BENCH_CPUS"
    for part in "${parts[@]}"; do
        for cpu in $(seq "${part%-*}" "${part#*-}"); do
            (( ${#capped[@]} < BENCH_MAX_CPUS )) && capped+=("$cpu")
        done
    done
    BENCH_CPUS="$(IFS=,; echo "${capped[*]}")"
fi

if command -v numactl >/dev/null 2>&1; then
    # --all is required: setup-benchmark.sh pins every running process (including the
    # CI runner this shell descends from) to the housekeeping CPUs, and libnuma parses
    # cpu lists relative to the inherited affinity mask unless told to consider all.
    exec numactl --all --physcpubind="$BENCH_CPUS" --membind=0 "$@"
fi

exec taskset -c "$BENCH_CPUS" "$@"
