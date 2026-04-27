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
        cpu_count="$(nproc)"
        BENCH_CPUS="2-$((cpu_count - 1))"
    fi
fi

if command -v numactl >/dev/null 2>&1; then
    exec numactl --physcpubind="$BENCH_CPUS" --membind=0 "$@"
fi

exec taskset -c "$BENCH_CPUS" "$@"
