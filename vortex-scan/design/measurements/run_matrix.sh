#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# Full V1-vs-V2 timing matrix. Re-run unchanged against another build by pointing BIN_DIR at it.
#   ROOT=<worktree with vortex-bench/data> BIN_DIR=<dir with duckdb-bench, datafusion-bench> ./run_matrix.sh [hot|cold|all]
S=${0:A:h}
what=${1:-all}
step() { echo "=== $(date '+%H:%M:%S') load: $(uptime | sed 's/.*load averages: //') :: $*" | tee -a $S/matrix.log; python3 $S/bench_ab.py "$@" >> $S/matrix.log 2>&1 || echo "FAILED: $*" | tee -a $S/matrix.log; }
if [[ $what == hot || $what == all ]]; then
  step hot duckdb tpch 1.0 5 10
  step hot duckdb clickbench - 3 5
  step hot duckdb tpch 10.0 3 5
  step hot datafusion tpch 1.0 5 10
  step hot datafusion clickbench - 3 5
  step hot datafusion tpch 10.0 3 5
fi
if [[ $what == cold || $what == all ]]; then
  step cold duckdb tpch 1.0 3
  step cold duckdb clickbench - 3
  step cold duckdb tpch 10.0 3
  step cold datafusion tpch 1.0 3
  step cold datafusion clickbench - 3
  step cold datafusion tpch 10.0 3
fi
echo "=== $(date '+%H:%M:%S') matrix done" | tee -a $S/matrix.log
