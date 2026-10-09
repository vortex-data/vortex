#!/bin/zsh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# Per-query Samply attribution sweep (hot), V1 and V2, both engines.
S=${0:A:h}
REST=2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22
for e in duckdb datafusion; do
  echo "=== $(date '+%H:%M:%S') $e tpch 10.0 load: $(uptime | sed 's/.*load averages: //')"
  python3 $S/prof_sweep.py $e tpch 10.0 V1,V2 $REST --secs 2
  echo "=== $(date '+%H:%M:%S') $e clickbench load: $(uptime | sed 's/.*load averages: //')"
  python3 $S/prof_sweep.py $e clickbench - V1,V2 --secs 2
  echo "=== $(date '+%H:%M:%S') $e tpch 1.0 load: $(uptime | sed 's/.*load averages: //')"
  python3 $S/prof_sweep.py $e tpch 1.0 V1,V2 --secs 2
done
echo "=== $(date '+%H:%M:%S') sweeps done"
