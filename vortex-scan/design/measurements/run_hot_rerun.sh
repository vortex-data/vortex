#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# Second hot pass (the first ran while the machine was busier), then the warm-first-iteration baseline.
S=${0:A:h}
step() { echo "=== $(date '+%H:%M:%S') load: $(uptime | sed 's/.*load averages: //') :: RERUN2 $*" | tee -a $S/matrix.log; python3 $S/bench_ab.py "$@" >> $S/matrix.log 2>&1 || echo "FAILED: $*" | tee -a $S/matrix.log; sleep 20; }
first() { echo "=== $(date '+%H:%M:%S') load: $(uptime | sed 's/.*load averages: //') :: WARMFIRST $*" | tee -a $S/matrix.log; python3 $S/bench_first.py "$@" >> $S/matrix.log 2>&1 || echo "FAILED: first $*" | tee -a $S/matrix.log; }
echo "\n# ===== SECOND HOT PASS (RERUN2) =====" >> $S/results.md
step hot duckdb tpch 1.0 5 10
step hot duckdb tpch 10.0 4 5
step hot datafusion tpch 1.0 5 10
step hot datafusion tpch 10.0 4 5
step hot duckdb clickbench - 3 5
step hot datafusion clickbench - 3 5
for e in duckdb datafusion; do
  first $e tpch 1.0 3
  first $e tpch 10.0 3
  first $e clickbench - 3
done
echo "=== $(date '+%H:%M:%S') rerun done" | tee -a $S/matrix.log
