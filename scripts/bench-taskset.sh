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
    cpu_count="$(nproc)"
    BENCH_CPUS="2-$((cpu_count - 1))"
fi

if command -v numactl >/dev/null 2>&1; then
    exec numactl --physcpubind="$BENCH_CPUS" --localalloc "$@"
fi

exec taskset -c "$BENCH_CPUS" "$@"
