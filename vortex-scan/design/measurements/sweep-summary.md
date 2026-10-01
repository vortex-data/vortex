<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->


## datafusion-clickbench--

### datafusion-clickbench-- V1: 43 queries
scan share of wall (per-round): median 29%, min 5%, max 99%, mean 41%
scan share of on-CPU: median 38%, mean 49%
threads blocked inside scan, share of busy thread-time: median 5%, mean 6%
  if scan cost falls by 100%: geomean time ratio 0.417 (58% faster); queries reaching >=30% faster: 21/43
  if scan cost falls by 50%: geomean time ratio 0.777 (22% faster); queries reaching >=30% faster: 14/43
  if scan cost falls by 30%: geomean time ratio 0.872 (13% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 13.0%; decode 16.0%; filter/expr kernels 4.6%; pruning 1.3%; sched/layout/dispatch 9.6%; convert to engine 0.5%; blocked in scan (IO wait/handoff) 5.7%; engine on-CPU 49.3%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 2.4%, max 12.4%; io-pool threads: median 109, max 378

### datafusion-clickbench-- V2: 43 queries
scan share of wall (per-round): median 32%, min 3%, max 98%, mean 41%
scan share of on-CPU: median 50%, mean 48%
threads blocked inside scan, share of busy thread-time: median 6%, mean 7%
  if scan cost falls by 100%: geomean time ratio 0.430 (57% faster); queries reaching >=30% faster: 22/43
  if scan cost falls by 50%: geomean time ratio 0.778 (22% faster); queries reaching >=30% faster: 12/43
  if scan cost falls by 30%: geomean time ratio 0.872 (13% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 10.4%; decode 16.8%; filter/expr kernels 5.3%; pruning 0.8%; sched/layout/dispatch 9.9%; convert to engine 0.4%; blocked in scan (IO wait/handoff) 7.2%; engine on-CPU 49.1%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 2.9%, max 12.0%; io-pool threads: median 130, max 342

## datafusion-tpch-1.0

### datafusion-tpch-1.0 V1: 22 queries
scan share of wall (per-round): median 36%, min 14%, max 86%, mean 39%
scan share of on-CPU: median 43%, mean 47%
threads blocked inside scan, share of busy thread-time: median 1%, mean 2%
  if scan cost falls by 100%: geomean time ratio 0.576 (42% faster); queries reaching >=30% faster: 12/22
  if scan cost falls by 50%: geomean time ratio 0.802 (20% faster); queries reaching >=30% faster: 2/22
  if scan cost falls by 30%: geomean time ratio 0.883 (12% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 19.3%; decode 9.4%; filter/expr kernels 6.8%; pruning 0.9%; sched/layout/dispatch 7.8%; convert to engine 1.5%; blocked in scan (IO wait/handoff) 2.5%; engine on-CPU 51.9%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.4%, max 1.6%; io-pool threads: median 36, max 81

### datafusion-tpch-1.0 V2: 22 queries
scan share of wall (per-round): median 38%, min 15%, max 86%, mean 39%
scan share of on-CPU: median 50%, mean 54%
threads blocked inside scan, share of busy thread-time: median 1%, mean 1%
  if scan cost falls by 100%: geomean time ratio 0.569 (43% faster); queries reaching >=30% faster: 14/22
  if scan cost falls by 50%: geomean time ratio 0.798 (20% faster); queries reaching >=30% faster: 2/22
  if scan cost falls by 30%: geomean time ratio 0.880 (12% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 27.2%; decode 8.1%; filter/expr kernels 6.5%; pruning 0.4%; sched/layout/dispatch 9.5%; convert to engine 1.4%; blocked in scan (IO wait/handoff) 1.2%; engine on-CPU 45.7%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.4%, max 1.1%; io-pool threads: median 48, max 170

## datafusion-tpch-10.0

### datafusion-tpch-10.0 V1: 22 queries
scan share of wall (per-round): median 24%, min 7%, max 94%, mean 33%
scan share of on-CPU: median 33%, mean 42%
threads blocked inside scan, share of busy thread-time: median 1%, mean 2%
  if scan cost falls by 100%: geomean time ratio 0.601 (40% faster); queries reaching >=30% faster: 9/22
  if scan cost falls by 50%: geomean time ratio 0.829 (17% faster); queries reaching >=30% faster: 3/22
  if scan cost falls by 30%: geomean time ratio 0.900 (10% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 19.4%; decode 8.8%; filter/expr kernels 4.9%; pruning 0.4%; sched/layout/dispatch 5.3%; convert to engine 1.8%; blocked in scan (IO wait/handoff) 2.0%; engine on-CPU 57.4%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 1.0%, max 5.5%; io-pool threads: median 149, max 220

### datafusion-tpch-10.0 V2: 22 queries
scan share of wall (per-round): median 26%, min 8%, max 95%, mean 33%
scan share of on-CPU: median 37%, mean 43%
threads blocked inside scan, share of busy thread-time: median 2%, mean 3%
  if scan cost falls by 100%: geomean time ratio 0.592 (41% faster); queries reaching >=30% faster: 8/22
  if scan cost falls by 50%: geomean time ratio 0.828 (17% faster); queries reaching >=30% faster: 2/22
  if scan cost falls by 30%: geomean time ratio 0.899 (10% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 18.6%; decode 7.8%; filter/expr kernels 4.6%; pruning 0.1%; sched/layout/dispatch 7.9%; convert to engine 1.9%; blocked in scan (IO wait/handoff) 2.6%; engine on-CPU 56.4%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.5%, max 1.7%; io-pool threads: median 180, max 253

## duckdb-clickbench--

### duckdb-clickbench-- V1: 43 queries
scan share of wall (per-round): median 71%, min 15%, max 99%, mean 65%
scan share of on-CPU: median 63%, mean 59%
threads blocked inside scan, share of busy thread-time: median 43%, mean 39%
  if scan cost falls by 100%: geomean time ratio 0.199 (80% faster); queries reaching >=30% faster: 36/43
  if scan cost falls by 50%: geomean time ratio 0.661 (34% faster); queries reaching >=30% faster: 26/43
  if scan cost falls by 30%: geomean time ratio 0.801 (20% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 8.2%; decode 10.5%; filter/expr kernels 3.0%; pruning 0.5%; sched/layout/dispatch 7.2%; convert to engine 1.8%; blocked in scan (IO wait/handoff) 38.6%; engine on-CPU 30.1%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 2.5%, max 9.7%; io-pool threads: median 14, max 65

### duckdb-clickbench-- V2: 43 queries
scan share of wall (per-round): median 69%, min 13%, max 100%, mean 61%
scan share of on-CPU: median 63%, mean 59%
threads blocked inside scan, share of busy thread-time: median 30%, mean 29%
  if scan cost falls by 100%: geomean time ratio 0.239 (76% faster); queries reaching >=30% faster: 34/43
  if scan cost falls by 50%: geomean time ratio 0.681 (32% faster); queries reaching >=30% faster: 26/43
  if scan cost falls by 30%: geomean time ratio 0.812 (19% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 9.4%; decode 12.8%; filter/expr kernels 4.1%; pruning 0.3%; sched/layout/dispatch 8.0%; convert to engine 2.5%; blocked in scan (IO wait/handoff) 29.3%; engine on-CPU 33.6%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 6.6%, max 20.5%; io-pool threads: median 19, max 92

## duckdb-tpch-1.0

### duckdb-tpch-1.0 V1: 22 queries
scan share of wall (per-round): median 50%, min 20%, max 91%, mean 53%
scan share of on-CPU: median 49%, mean 49%
threads blocked inside scan, share of busy thread-time: median 47%, mean 47%
  if scan cost falls by 100%: geomean time ratio 0.435 (56% faster); queries reaching >=30% faster: 21/22
  if scan cost falls by 50%: geomean time ratio 0.731 (27% faster); queries reaching >=30% faster: 7/22
  if scan cost falls by 30%: geomean time ratio 0.840 (16% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 5.9%; decode 7.8%; filter/expr kernels 4.3%; pruning 0.4%; sched/layout/dispatch 4.5%; convert to engine 1.0%; blocked in scan (IO wait/handoff) 46.6%; engine on-CPU 29.5%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.9%, max 4.3%; io-pool threads: median 5, max 13

### duckdb-tpch-1.0 V2: 22 queries
scan share of wall (per-round): median 41%, min 20%, max 87%, mean 43%
scan share of on-CPU: median 45%, mean 46%
threads blocked inside scan, share of busy thread-time: median 31%, mean 29%
  if scan cost falls by 100%: geomean time ratio 0.544 (46% faster); queries reaching >=30% faster: 18/22
  if scan cost falls by 50%: geomean time ratio 0.782 (22% faster); queries reaching >=30% faster: 1/22
  if scan cost falls by 30%: geomean time ratio 0.870 (13% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 6.8%; decode 12.0%; filter/expr kernels 6.3%; pruning 0.2%; sched/layout/dispatch 5.0%; convert to engine 1.8%; blocked in scan (IO wait/handoff) 28.6%; engine on-CPU 39.4%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 4.7%, max 15.1%; io-pool threads: median 33, max 45

## duckdb-tpch-10.0

### duckdb-tpch-10.0 V1: 22 queries
scan share of wall (per-round): median 63%, min 24%, max 98%, mean 62%
scan share of on-CPU: median 50%, mean 50%
threads blocked inside scan, share of busy thread-time: median 40%, mean 41%
  if scan cost falls by 100%: geomean time ratio 0.315 (68% faster); queries reaching >=30% faster: 21/22
  if scan cost falls by 50%: geomean time ratio 0.682 (32% faster); queries reaching >=30% faster: 14/22
  if scan cost falls by 30%: geomean time ratio 0.811 (19% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 7.3%; decode 9.6%; filter/expr kernels 4.7%; pruning 0.3%; sched/layout/dispatch 4.6%; convert to engine 1.2%; blocked in scan (IO wait/handoff) 40.6%; engine on-CPU 31.7%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 1.0%, max 4.2%; io-pool threads: median 8, max 13

### duckdb-tpch-10.0 V2: 22 queries
scan share of wall (per-round): median 54%, min 20%, max 96%, mean 54%
scan share of on-CPU: median 49%, mean 49%
threads blocked inside scan, share of busy thread-time: median 23%, mean 23%
  if scan cost falls by 100%: geomean time ratio 0.413 (59% faster); queries reaching >=30% faster: 20/22
  if scan cost falls by 50%: geomean time ratio 0.726 (27% faster); queries reaching >=30% faster: 8/22
  if scan cost falls by 30%: geomean time ratio 0.837 (16% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 8.8%; decode 13.8%; filter/expr kernels 6.3%; pruning 0.2%; sched/layout/dispatch 5.5%; convert to engine 2.2%; blocked in scan (IO wait/handoff) 23.0%; engine on-CPU 40.1%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 7.9%, max 19.8%; io-pool threads: median 35, max 43
