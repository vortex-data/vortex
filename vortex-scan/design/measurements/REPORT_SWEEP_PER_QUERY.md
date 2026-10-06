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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | 0.6 | 0.6 | 5 | 36 | 5 | 29 | 1 | 0 | 4 | 0 | 61 |
| q1 | 9.5 | 10.4 | 82 | 94 | 6 | 18 | 11 | 24 | 30 | 2 | 6 |
| q2 | 29.4 | 32.7 | 60 | 66 | 7 | 7 | 35 | 4 | 13 | 1 | 32 |
| q3 | 28.9 | 32.2 | 71 | 77 | 12 | 27 | 24 | 4 | 11 | 1 | 20 |
| q4 | 296.0 | 267.9 | 9 | 8 | 2 | 2 | 3 | 0 | 2 | 0 | 90 |
| q5 | 317.2 | 284.4 | 14 | 13 | 5 | 3 | 6 | 0 | 2 | 0 | 83 |
| q6 | 0.9 | 0.6 | 5 | 25 | 6 | 16 | 3 | 0 | 4 | 0 | 71 |
| q7 | 14.7 | 13.8 | 78 | 91 | 5 | 11 | 17 | 23 | 31 | 2 | 8 |
| q8 | 396.2 | 381.5 | 11 | 11 | 2 | 3 | 5 | 1 | 2 | 0 | 87 |
| q9 | 467.1 | 455.0 | 9 | 14 | 3 | 2 | 7 | 1 | 3 | 0 | 83 |
| q10 | 62.0 | 60.8 | 63 | 68 | 9 | 11 | 27 | 7 | 14 | 1 | 29 |
| q11 | 81.4 | 72.6 | 63 | 68 | 11 | 10 | 28 | 7 | 14 | 1 | 28 |
| q12 | 278.1 | 240.8 | 19 | 17 | 5 | 3 | 8 | 2 | 3 | 0 | 78 |
| q13 | 439.2 | 383.8 | 20 | 20 | 2 | 4 | 9 | 2 | 4 | 0 | 78 |
| q14 | 288.3 | 254.8 | 20 | 20 | 6 | 3 | 9 | 2 | 4 | 0 | 75 |
| q15 | 349.1 | 310.6 | 7 | 6 | 2 | 2 | 2 | 0 | 1 | 0 | 92 |
| q16 | 751.0 | 652.6 | 9 | 7 | 4 | 2 | 3 | 0 | 2 | 0 | 89 |
| q17 | 759.5 | 640.6 | 8 | 7 | 2 | 2 | 3 | 0 | 1 | 0 | 91 |
| q18 | 1528.6 | 1408.3 | 7 | 8 | 1 | 3 | 3 | 0 | 2 | 0 | 91 |
| q19 | 24.0 | 18.4 | 88 | 96 | 9 | 32 | 22 | 11 | 19 | 2 | 3 |
| q20 | 595.5 | 596.7 | 99 | 100 | 3 | 50 | 44 | 0 | 2 | 0 | 0 |
| q21 | 462.3 | 380.2 | 96 | 99 | 14 | 18 | 42 | 13 | 11 | 0 | 1 |
| q22 | 508.3 | 446.2 | 97 | 99 | 8 | 60 | 18 | 7 | 6 | 0 | 1 |
| q23 | 463.3 | 433.8 | 95 | 99 | 5 | 55 | 16 | 5 | 18 | 0 | 1 |
| q24 | 35.3 | 29.4 | 87 | 97 | 14 | 12 | 40 | 8 | 20 | 1 | 3 |
| q25 | 69.0 | 59.5 | 91 | 95 | 11 | 14 | 43 | 9 | 17 | 1 | 5 |
| q26 | 34.7 | 29.8 | 86 | 96 | 12 | 13 | 39 | 8 | 21 | 1 | 4 |
| q27 | 464.2 | 433.7 | 70 | 72 | 11 | 13 | 37 | 2 | 10 | 0 | 25 |
| q28 | 2690.4 | 2520.0 | 8 | 8 | 1 | 2 | 4 | 0 | 1 | 0 | 91 |
| q29 | 30.2 | 28.5 | 41 | 81 | 5 | 9 | 49 | 5 | 12 | 1 | 18 |
| q30 | 252.6 | 232.6 | 37 | 37 | 6 | 6 | 15 | 4 | 8 | 0 | 59 |
| q31 | 260.6 | 235.1 | 46 | 48 | 5 | 14 | 17 | 5 | 10 | 0 | 49 |
| q32 | 1133.4 | 1186.7 | 8 | 8 | 1 | 2 | 3 | 0 | 2 | 0 | 92 |
| q33 | 1599.2 | 1722.3 | 12 | 13 | 2 | 3 | 8 | 0 | 2 | 0 | 85 |
| q34 | 1673.3 | 1671.3 | 12 | 14 | 3 | 4 | 7 | 0 | 2 | 0 | 84 |
| q35 | 320.7 | 321.3 | 6 | 6 | 1 | 1 | 3 | 0 | 1 | 0 | 93 |
| q36 | 30.0 | 27.0 | 16 | 36 | 5 | 9 | 11 | 4 | 8 | 0 | 60 |
| q37 | 14.6 | 14.9 | 18 | 38 | 6 | 8 | 9 | 5 | 11 | 0 | 58 |
| q38 | 8.5 | 8.6 | 59 | 83 | 11 | 20 | 22 | 8 | 19 | 0 | 15 |
| q39 | 55.0 | 57.8 | 11 | 25 | 4 | 5 | 8 | 2 | 8 | 0 | 71 |
| q40 | 6.1 | 6.1 | 42 | 65 | 4 | 21 | 6 | 9 | 19 | 1 | 34 |
| q41 | 5.9 | 5.8 | 45 | 68 | 5 | 18 | 12 | 8 | 19 | 1 | 30 |
| q42 | 6.6 | 6.1 | 29 | 54 | 4 | 12 | 7 | 8 | 18 | 0 | 44 |

### datafusion-clickbench-- V2: 43 queries
scan share of wall (per-round): median 32%, min 3%, max 98%, mean 41%
scan share of on-CPU: median 50%, mean 48%
threads blocked inside scan, share of busy thread-time: median 6%, mean 7%
  if scan cost falls by 100%: geomean time ratio 0.430 (57% faster); queries reaching >=30% faster: 22/43
  if scan cost falls by 50%: geomean time ratio 0.778 (22% faster); queries reaching >=30% faster: 12/43
  if scan cost falls by 30%: geomean time ratio 0.872 (13% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 10.4%; decode 16.8%; filter/expr kernels 5.3%; pruning 0.8%; sched/layout/dispatch 9.9%; convert to engine 0.4%; blocked in scan (IO wait/handoff) 7.2%; engine on-CPU 49.1%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 2.9%, max 12.0%; io-pool threads: median 130, max 342
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | 0.6 | 0.6 | 3 | 11 | 12 | 4 | 1 | 0 | 4 | 0 | 78 |
| q1 | 9.5 | 9.1 | 81 | 95 | 6 | 20 | 11 | 27 | 25 | 3 | 5 |
| q2 | 29.4 | 33.2 | 59 | 62 | 8 | 8 | 35 | 4 | 9 | 1 | 35 |
| q3 | 28.9 | 33.3 | 69 | 74 | 16 | 19 | 27 | 3 | 11 | 1 | 22 |
| q4 | 296.0 | 268.8 | 10 | 6 | 5 | 2 | 2 | 0 | 1 | 0 | 89 |
| q5 | 317.2 | 403.2 | 17 | 14 | 6 | 2 | 7 | 0 | 3 | 0 | 81 |
| q6 | 0.9 | 0.6 | 3 | 10 | 15 | 3 | 2 | 0 | 3 | 0 | 76 |
| q7 | 14.7 | 10.4 | 73 | 90 | 5 | 17 | 14 | 27 | 24 | 2 | 10 |
| q8 | 396.2 | 379.3 | 12 | 9 | 4 | 2 | 4 | 1 | 2 | 0 | 87 |
| q9 | 467.1 | 439.9 | 11 | 14 | 7 | 3 | 7 | 1 | 2 | 0 | 81 |
| q10 | 62.0 | 62.3 | 59 | 65 | 11 | 13 | 25 | 6 | 13 | 1 | 31 |
| q11 | 81.4 | 68.9 | 58 | 65 | 10 | 11 | 27 | 7 | 12 | 0 | 32 |
| q12 | 278.1 | 241.3 | 19 | 16 | 7 | 3 | 7 | 1 | 3 | 0 | 78 |
| q13 | 439.2 | 399.7 | 20 | 20 | 2 | 5 | 9 | 2 | 4 | 0 | 78 |
| q14 | 288.3 | 249.7 | 21 | 18 | 9 | 3 | 8 | 2 | 3 | 0 | 75 |
| q15 | 349.1 | 307.1 | 9 | 6 | 4 | 2 | 2 | 0 | 1 | 0 | 90 |
| q16 | 751.0 | 653.1 | 8 | 7 | 3 | 2 | 3 | 0 | 1 | 0 | 90 |
| q17 | 759.5 | 650.8 | 10 | 8 | 5 | 2 | 3 | 0 | 2 | 0 | 88 |
| q18 | 1528.6 | 1343.9 | 6 | 5 | 2 | 1 | 2 | 0 | 1 | 0 | 93 |
| q19 | 24.0 | 20.8 | 90 | 97 | 8 | 30 | 28 | 9 | 20 | 1 | 2 |
| q20 | 595.5 | 352.8 | 98 | 99 | 15 | 14 | 46 | 15 | 9 | 0 | 1 |
| q21 | 462.3 | 383.1 | 97 | 99 | 12 | 19 | 45 | 10 | 12 | 0 | 1 |
| q22 | 508.3 | 496.3 | 96 | 99 | 15 | 26 | 34 | 8 | 14 | 0 | 1 |
| q23 | 463.3 | 607.5 | 94 | 99 | 14 | 31 | 19 | 8 | 26 | 0 | 1 |
| q24 | 35.3 | 28.9 | 85 | 96 | 18 | 16 | 35 | 8 | 19 | 0 | 3 |
| q25 | 69.0 | 57.1 | 89 | 95 | 11 | 15 | 41 | 10 | 16 | 1 | 5 |
| q26 | 34.7 | 30.2 | 85 | 96 | 16 | 15 | 35 | 8 | 20 | 1 | 3 |
| q27 | 464.2 | 447.8 | 73 | 74 | 12 | 13 | 36 | 2 | 12 | 0 | 23 |
| q28 | 2690.4 | 2466.4 | 8 | 8 | 1 | 2 | 5 | 0 | 1 | 0 | 91 |
| q29 | 30.2 | 31.7 | 41 | 76 | 4 | 10 | 47 | 6 | 9 | 1 | 23 |
| q30 | 252.6 | 228.9 | 34 | 32 | 9 | 7 | 13 | 3 | 6 | 0 | 61 |
| q31 | 260.6 | 242.2 | 51 | 53 | 8 | 16 | 17 | 4 | 11 | 0 | 44 |
| q32 | 1133.4 | 1194.0 | 10 | 10 | 1 | 4 | 4 | 1 | 2 | 0 | 89 |
| q33 | 1599.2 | 1682.6 | 14 | 13 | 5 | 2 | 8 | 0 | 2 | 0 | 83 |
| q34 | 1673.3 | 1726.3 | 11 | 13 | 2 | 4 | 7 | 0 | 2 | 0 | 85 |
| q35 | 320.7 | 311.0 | 7 | 5 | 2 | 1 | 2 | 0 | 1 | 0 | 93 |
| q36 | 30.0 | 28.2 | 17 | 38 | 4 | 8 | 13 | 4 | 10 | 0 | 60 |
| q37 | 14.6 | 16.0 | 21 | 50 | 4 | 8 | 15 | 8 | 15 | 0 | 48 |
| q38 | 8.5 | 10.5 | 58 | 84 | 5 | 17 | 28 | 7 | 24 | 0 | 15 |
| q39 | 55.0 | 58.9 | 10 | 24 | 3 | 6 | 9 | 2 | 6 | 0 | 73 |
| q40 | 6.1 | 6.4 | 44 | 70 | 2 | 23 | 8 | 12 | 22 | 0 | 29 |
| q41 | 5.9 | 6.0 | 45 | 75 | 1 | 19 | 21 | 11 | 19 | 0 | 24 |
| q42 | 6.6 | 6.0 | 32 | 63 | 1 | 17 | 12 | 10 | 19 | 1 | 37 |

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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 43.2 | 47.2 | 27 | 36 | 1 | 6 | 5 | 13 | 6 | 5 | 64 |
| q2 | 13.2 | 10.7 | 29 | 32 | 1 | 14 | 7 | 1 | 6 | 3 | 67 |
| q3 | 19.2 | 20.1 | 39 | 42 | 1 | 12 | 14 | 6 | 7 | 3 | 57 |
| q4 | 13.2 | 13.7 | 38 | 51 | 1 | 24 | 15 | 5 | 7 | 0 | 48 |
| q5 | 38.8 | 39.9 | 25 | 27 | 0 | 11 | 7 | 1 | 5 | 3 | 72 |
| q6 | 7.4 | 7.0 | 86 | 95 | 11 | 34 | 15 | 17 | 15 | 1 | 5 |
| q7 | 40.6 | 42.5 | 27 | 28 | 1 | 9 | 8 | 4 | 5 | 1 | 71 |
| q8 | 25.1 | 23.1 | 38 | 59 | 1 | 33 | 10 | 4 | 8 | 3 | 40 |
| q9 | 37.5 | 35.3 | 27 | 44 | 1 | 27 | 8 | 1 | 5 | 3 | 55 |
| q10 | 24.8 | 20.8 | 43 | 51 | 2 | 21 | 14 | 4 | 8 | 1 | 48 |
| q11 | 9.1 | 7.7 | 39 | 48 | 0 | 27 | 8 | 1 | 7 | 4 | 52 |
| q12 | 13.7 | 14.1 | 56 | 70 | 6 | 35 | 11 | 7 | 12 | 0 | 28 |
| q13 | 21.6 | 19.5 | 54 | 69 | 1 | 11 | 8 | 45 | 5 | 0 | 30 |
| q14 | 9.7 | 8.0 | 53 | 70 | 7 | 31 | 11 | 7 | 14 | 1 | 28 |
| q15 | 14.4 | 14.2 | 53 | 72 | 6 | 33 | 12 | 9 | 12 | 1 | 26 |
| q16 | 10.5 | 9.6 | 19 | 16 | 0 | 5 | 4 | 2 | 4 | 0 | 84 |
| q17 | 61.6 | 53.8 | 18 | 24 | 1 | 13 | 2 | 0 | 6 | 2 | 76 |
| q18 | 71.7 | 61.7 | 14 | 17 | 0 | 10 | 3 | 0 | 2 | 1 | 83 |
| q19 | 11.8 | 9.3 | 74 | 89 | 10 | 26 | 18 | 13 | 18 | 1 | 10 |
| q20 | 18.1 | 17.3 | 35 | 41 | 2 | 19 | 7 | 5 | 8 | 0 | 58 |
| q21 | 55.4 | 48.5 | 26 | 30 | 0 | 12 | 10 | 3 | 4 | 0 | 70 |
| q22 | 8.0 | 6.3 | 28 | 32 | 1 | 14 | 7 | 2 | 7 | 1 | 67 |

### datafusion-tpch-1.0 V2: 22 queries
scan share of wall (per-round): median 38%, min 15%, max 86%, mean 39%
scan share of on-CPU: median 50%, mean 54%
threads blocked inside scan, share of busy thread-time: median 1%, mean 1%
  if scan cost falls by 100%: geomean time ratio 0.569 (43% faster); queries reaching >=30% faster: 14/22
  if scan cost falls by 50%: geomean time ratio 0.798 (20% faster); queries reaching >=30% faster: 2/22
  if scan cost falls by 30%: geomean time ratio 0.880 (12% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 27.2%; decode 8.1%; filter/expr kernels 6.5%; pruning 0.4%; sched/layout/dispatch 9.5%; convert to engine 1.4%; blocked in scan (IO wait/handoff) 1.2%; engine on-CPU 45.7%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.4%, max 1.1%; io-pool threads: median 48, max 170
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 43.2 | 53.3 | 27 | 38 | 1 | 10 | 4 | 12 | 7 | 4 | 61 |
| q2 | 13.2 | 10.9 | 34 | 40 | 1 | 19 | 8 | 2 | 8 | 2 | 60 |
| q3 | 19.2 | 20.2 | 39 | 46 | 1 | 16 | 11 | 6 | 9 | 3 | 53 |
| q4 | 13.2 | 15.0 | 40 | 62 | 4 | 34 | 12 | 4 | 9 | 0 | 37 |
| q5 | 38.8 | 41.0 | 23 | 33 | 1 | 18 | 6 | 1 | 5 | 3 | 67 |
| q6 | 7.4 | 5.9 | 86 | 96 | 2 | 41 | 13 | 16 | 23 | 0 | 4 |
| q7 | 40.6 | 39.2 | 25 | 30 | 0 | 12 | 7 | 4 | 6 | 1 | 69 |
| q8 | 25.1 | 24.5 | 39 | 71 | 1 | 47 | 9 | 2 | 9 | 3 | 29 |
| q9 | 37.5 | 37.7 | 29 | 53 | 1 | 36 | 7 | 1 | 7 | 3 | 47 |
| q10 | 24.8 | 21.0 | 43 | 58 | 1 | 31 | 11 | 3 | 10 | 1 | 42 |
| q11 | 9.1 | 8.3 | 45 | 60 | 0 | 40 | 8 | 1 | 7 | 3 | 40 |
| q12 | 13.7 | 13.7 | 49 | 76 | 1 | 48 | 7 | 10 | 10 | 0 | 23 |
| q13 | 21.6 | 26.2 | 57 | 74 | 2 | 17 | 7 | 39 | 9 | 0 | 26 |
| q14 | 9.7 | 7.8 | 50 | 74 | 1 | 41 | 10 | 6 | 15 | 1 | 25 |
| q15 | 14.4 | 13.5 | 54 | 77 | 2 | 42 | 11 | 7 | 15 | 1 | 23 |
| q16 | 10.5 | 9.7 | 20 | 22 | 1 | 9 | 4 | 4 | 5 | 0 | 77 |
| q17 | 61.6 | 54.2 | 18 | 30 | 1 | 19 | 2 | 0 | 6 | 2 | 69 |
| q18 | 71.7 | 64.2 | 15 | 24 | 1 | 15 | 3 | 0 | 3 | 2 | 75 |
| q19 | 11.8 | 9.3 | 75 | 93 | 2 | 38 | 14 | 16 | 22 | 1 | 7 |
| q20 | 18.1 | 18.0 | 38 | 45 | 1 | 22 | 8 | 4 | 10 | 0 | 54 |
| q21 | 55.4 | 51.4 | 28 | 41 | 1 | 24 | 10 | 2 | 5 | 0 | 59 |
| q22 | 8.0 | 7.6 | 32 | 41 | 1 | 21 | 6 | 3 | 10 | 1 | 58 |

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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 500.9 | 369.8 | 39 | 49 | 1 | 6 | 22 | 12 | 4 | 5 | 50 |
| q2 | 83.5 | 112.3 | 19 | 22 | 1 | 8 | 7 | 1 | 4 | 2 | 77 |
| q3 | 179.6 | 168.9 | 31 | 49 | 1 | 28 | 9 | 3 | 6 | 2 | 51 |
| q4 | 99.5 | 95.8 | 36 | 54 | 1 | 27 | 16 | 4 | 6 | 0 | 46 |
| q5 | 365.3 | 346.3 | 18 | 27 | 0 | 7 | 7 | 1 | 6 | 6 | 72 |
| q6 | 42.7 | 36.1 | 94 | 98 | 7 | 47 | 15 | 16 | 12 | 0 | 2 |
| q7 | 515.9 | 460.3 | 12 | 17 | 0 | 8 | 5 | 2 | 2 | 0 | 83 |
| q8 | 311.7 | 292.7 | 23 | 31 | 1 | 7 | 11 | 1 | 6 | 5 | 68 |
| q9 | 613.2 | 553.4 | 18 | 27 | 0 | 9 | 7 | 1 | 4 | 6 | 73 |
| q10 | 221.6 | 199.0 | 26 | 34 | 1 | 13 | 12 | 3 | 5 | 1 | 65 |
| q11 | 70.0 | 63.4 | 17 | 23 | 0 | 12 | 5 | 0 | 3 | 2 | 77 |
| q12 | 110.3 | 106.0 | 48 | 71 | 7 | 46 | 9 | 4 | 7 | 0 | 27 |
| q13 | 246.3 | 219.1 | 46 | 51 | 1 | 10 | 6 | 31 | 3 | 0 | 48 |
| q14 | 59.7 | 56.4 | 47 | 71 | 4 | 48 | 10 | 4 | 6 | 0 | 28 |
| q15 | 103.2 | 91.8 | 61 | 83 | 4 | 56 | 9 | 5 | 8 | 0 | 16 |
| q16 | 65.3 | 54.8 | 15 | 14 | 0 | 3 | 6 | 2 | 3 | 0 | 86 |
| q17 | 674.4 | 600.9 | 15 | 19 | 0 | 5 | 3 | 1 | 5 | 6 | 80 |
| q18 | 950.7 | 786.5 | 7 | 12 | 0 | 4 | 3 | 1 | 2 | 2 | 88 |
| q19 | 65.9 | 58.7 | 82 | 92 | 8 | 45 | 14 | 9 | 13 | 1 | 8 |
| q20 | 163.0 | 140.9 | 28 | 39 | 2 | 21 | 7 | 4 | 5 | 0 | 60 |
| q21 | 675.5 | 601.1 | 14 | 20 | 1 | 8 | 7 | 2 | 3 | 0 | 80 |
| q22 | 58.1 | 46.7 | 21 | 20 | 1 | 9 | 5 | 1 | 3 | 0 | 80 |

### datafusion-tpch-10.0 V2: 22 queries
scan share of wall (per-round): median 26%, min 8%, max 95%, mean 33%
scan share of on-CPU: median 37%, mean 43%
threads blocked inside scan, share of busy thread-time: median 2%, mean 3%
  if scan cost falls by 100%: geomean time ratio 0.592 (41% faster); queries reaching >=30% faster: 8/22
  if scan cost falls by 50%: geomean time ratio 0.828 (17% faster); queries reaching >=30% faster: 2/22
  if scan cost falls by 30%: geomean time ratio 0.899 (10% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 18.6%; decode 7.8%; filter/expr kernels 4.6%; pruning 0.1%; sched/layout/dispatch 7.9%; convert to engine 1.9%; blocked in scan (IO wait/handoff) 2.6%; engine on-CPU 56.4%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 0.5%, max 1.7%; io-pool threads: median 180, max 253
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 500.9 | 399.8 | 42 | 53 | 2 | 9 | 20 | 11 | 6 | 5 | 46 |
| q2 | 83.5 | 111.0 | 19 | 28 | 0 | 15 | 6 | 0 | 5 | 2 | 72 |
| q3 | 179.6 | 181.9 | 27 | 45 | 2 | 21 | 9 | 3 | 9 | 2 | 54 |
| q4 | 99.5 | 111.1 | 31 | 46 | 3 | 20 | 14 | 3 | 8 | 0 | 52 |
| q5 | 365.3 | 358.0 | 18 | 26 | 1 | 6 | 6 | 0 | 8 | 6 | 73 |
| q6 | 42.7 | 36.1 | 95 | 99 | 6 | 46 | 12 | 13 | 21 | 0 | 1 |
| q7 | 515.9 | 469.9 | 17 | 15 | 4 | 7 | 4 | 2 | 2 | 0 | 82 |
| q8 | 311.7 | 289.0 | 24 | 35 | 1 | 8 | 10 | 1 | 9 | 6 | 64 |
| q9 | 613.2 | 565.5 | 20 | 28 | 1 | 6 | 7 | 1 | 6 | 7 | 71 |
| q10 | 221.6 | 218.9 | 28 | 38 | 2 | 16 | 10 | 2 | 8 | 1 | 60 |
| q11 | 70.0 | 65.0 | 18 | 29 | 0 | 19 | 4 | 0 | 3 | 2 | 71 |
| q12 | 110.3 | 141.0 | 41 | 59 | 8 | 22 | 11 | 7 | 13 | 0 | 38 |
| q13 | 246.3 | 222.5 | 45 | 50 | 4 | 5 | 5 | 32 | 5 | 0 | 48 |
| q14 | 59.7 | 57.2 | 46 | 73 | 3 | 51 | 8 | 3 | 8 | 0 | 26 |
| q15 | 103.2 | 92.3 | 59 | 86 | 6 | 61 | 6 | 3 | 10 | 0 | 13 |
| q16 | 65.3 | 54.9 | 16 | 20 | 0 | 7 | 6 | 3 | 4 | 0 | 80 |
| q17 | 674.4 | 628.8 | 14 | 19 | 1 | 4 | 3 | 1 | 5 | 7 | 80 |
| q18 | 950.7 | 765.1 | 8 | 12 | 0 | 4 | 2 | 1 | 3 | 2 | 88 |
| q19 | 65.9 | 74.2 | 85 | 94 | 8 | 44 | 11 | 8 | 22 | 0 | 5 |
| q20 | 163.0 | 141.4 | 28 | 39 | 2 | 21 | 6 | 4 | 7 | 0 | 59 |
| q21 | 675.5 | 599.2 | 15 | 17 | 1 | 4 | 7 | 2 | 4 | 0 | 82 |
| q22 | 58.1 | 48.3 | 22 | 25 | 2 | 14 | 4 | 1 | 5 | 0 | 74 |

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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | 1.0 | 0.7 | 60 | 62 | 21 | 24 | 2 | 1 | 7 | 14 | 30 |
| q1 | 12.7 | 13.5 | 95 | 98 | 53 | 6 | 6 | 12 | 19 | 1 | 1 |
| q2 | 17.1 | 17.7 | 96 | 99 | 57 | 8 | 11 | 5 | 17 | 0 | 1 |
| q3 | 45.2 | 49.2 | 86 | 68 | 60 | 9 | 9 | 1 | 5 | 3 | 13 |
| q4 | 163.2 | 177.2 | 34 | 18 | 21 | 5 | 5 | 0 | 3 | 1 | 64 |
| q5 | 161.0 | 161.2 | 45 | 30 | 24 | 7 | 9 | 1 | 4 | 3 | 53 |
| q6 | 1.2 | 0.8 | 61 | 70 | 46 | 15 | 8 | 3 | 7 | 5 | 16 |
| q7 | 16.6 | 16.0 | 89 | 91 | 44 | 5 | 10 | 12 | 21 | 2 | 5 |
| q8 | 225.1 | 233.1 | 38 | 23 | 21 | 5 | 7 | 1 | 3 | 2 | 61 |
| q9 | 328.5 | 341.2 | 30 | 21 | 13 | 3 | 9 | 0 | 3 | 2 | 70 |
| q10 | 68.3 | 75.1 | 83 | 69 | 54 | 8 | 12 | 3 | 8 | 1 | 14 |
| q11 | 79.6 | 84.7 | 83 | 70 | 49 | 7 | 15 | 3 | 8 | 1 | 15 |
| q12 | 179.3 | 174.5 | 48 | 34 | 24 | 4 | 13 | 2 | 5 | 1 | 50 |
| q13 | 365.8 | 360.4 | 45 | 29 | 27 | 4 | 10 | 2 | 4 | 1 | 52 |
| q14 | 221.5 | 204.2 | 53 | 37 | 28 | 4 | 13 | 3 | 5 | 1 | 45 |
| q15 | 252.5 | 236.4 | 26 | 13 | 16 | 3 | 4 | 0 | 2 | 1 | 73 |
| q16 | 468.2 | 458.4 | 30 | 20 | 14 | 4 | 7 | 1 | 3 | 3 | 69 |
| q17 | 353.8 | 346.8 | 32 | 21 | 16 | 4 | 7 | 0 | 4 | 3 | 66 |
| q18 | 795.5 | 821.1 | 25 | 15 | 14 | 3 | 5 | 0 | 2 | 2 | 74 |
| q19 | 24.3 | 28.5 | 97 | 99 | 68 | 12 | 7 | 3 | 7 | 1 | 0 |
| q20 | 504.2 | 531.2 | 99 | 100 | 28 | 21 | 49 | 0 | 2 | 0 | 0 |
| q21 | 383.7 | 487.7 | 99 | 99 | 52 | 11 | 24 | 7 | 5 | 0 | 0 |
| q22 | 672.5 | 687.7 | 99 | 97 | 64 | 11 | 14 | 5 | 4 | 0 | 1 |
| q23 | 73.8 | 86.6 | 95 | 98 | 38 | 19 | 30 | 2 | 8 | 2 | 2 |
| q24 | 20.8 | 24.6 | 94 | 97 | 64 | 10 | 8 | 4 | 10 | 2 | 1 |
| q25 | 55.4 | 65.9 | 99 | 99 | 52 | 6 | 28 | 4 | 8 | 1 | 0 |
| q26 | 17.4 | 20.7 | 94 | 95 | 59 | 8 | 14 | 4 | 10 | 2 | 2 |
| q27 | 236.9 | 277.6 | 88 | 76 | 53 | 23 | 4 | 2 | 6 | 1 | 11 |
| q28 | 3280.0 | 3581.5 | 15 | 9 | 7 | 3 | 4 | 0 | 1 | 0 | 84 |
| q29 | 19.9 | 21.3 | 82 | 89 | 68 | 8 | 6 | 4 | 10 | 0 | 4 |
| q30 | 168.4 | 185.4 | 66 | 53 | 34 | 7 | 16 | 3 | 8 | 1 | 31 |
| q31 | 250.0 | 258.9 | 73 | 57 | 42 | 11 | 12 | 2 | 7 | 1 | 25 |
| q32 | 898.2 | 890.3 | 21 | 13 | 12 | 3 | 4 | 0 | 2 | 1 | 76 |
| q33 | 835.2 | 963.2 | 53 | 38 | 26 | 7 | 16 | 0 | 3 | 1 | 46 |
| q34 | 1035.8 | 1072.8 | 45 | 30 | 23 | 6 | 13 | 0 | 3 | 1 | 54 |
| q35 | 308.6 | 341.9 | 18 | 9 | 11 | 2 | 3 | 0 | 2 | 1 | 81 |
| q36 | 14.3 | 15.1 | 71 | 61 | 48 | 7 | 7 | 4 | 10 | 2 | 20 |
| q37 | 7.4 | 7.8 | 73 | 75 | 50 | 8 | 5 | 7 | 14 | 3 | 13 |
| q38 | 11.9 | 12.4 | 84 | 88 | 59 | 7 | 9 | 5 | 12 | 2 | 5 |
| q39 | 27.5 | 32.4 | 63 | 49 | 43 | 6 | 9 | 3 | 8 | 1 | 29 |
| q40 | 7.5 | 7.6 | 78 | 84 | 56 | 10 | 3 | 7 | 14 | 2 | 7 |
| q41 | 7.7 | 7.9 | 74 | 83 | 56 | 10 | 4 | 6 | 13 | 2 | 7 |
| q42 | 7.2 | 7.9 | 64 | 63 | 45 | 9 | 3 | 6 | 13 | 2 | 20 |

### duckdb-clickbench-- V2: 43 queries
scan share of wall (per-round): median 69%, min 13%, max 100%, mean 61%
scan share of on-CPU: median 63%, mean 59%
threads blocked inside scan, share of busy thread-time: median 30%, mean 29%
  if scan cost falls by 100%: geomean time ratio 0.239 (76% faster); queries reaching >=30% faster: 34/43
  if scan cost falls by 50%: geomean time ratio 0.681 (32% faster); queries reaching >=30% faster: 26/43
  if scan cost falls by 30%: geomean time ratio 0.812 (19% faster); queries reaching >=30% faster: 0/43
  mean share of busy thread-time per query: io read + dispatch 9.4%; decode 12.8%; filter/expr kernels 4.1%; pruning 0.3%; sched/layout/dispatch 8.0%; convert to engine 2.5%; blocked in scan (IO wait/handoff) 29.3%; engine on-CPU 33.6%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 6.6%, max 20.5%; io-pool threads: median 19, max 92
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | 1.0 | 0.7 | 61 | 61 | 14 | 26 | 2 | 1 | 7 | 17 | 34 |
| q1 | 12.7 | 10.0 | 92 | 98 | 44 | 8 | 7 | 18 | 18 | 1 | 1 |
| q2 | 17.1 | 10.2 | 92 | 98 | 41 | 17 | 14 | 12 | 14 | 0 | 1 |
| q3 | 45.2 | 32.2 | 75 | 66 | 32 | 13 | 11 | 1 | 7 | 13 | 23 |
| q4 | 163.2 | 159.2 | 26 | 18 | 11 | 4 | 4 | 0 | 3 | 5 | 73 |
| q5 | 161.0 | 150.2 | 37 | 28 | 14 | 4 | 11 | 0 | 4 | 3 | 62 |
| q6 | 1.2 | 0.8 | 62 | 75 | 25 | 15 | 18 | 4 | 11 | 8 | 19 |
| q7 | 16.6 | 11.2 | 84 | 89 | 41 | 7 | 8 | 18 | 17 | 1 | 6 |
| q8 | 225.1 | 230.9 | 28 | 21 | 10 | 5 | 7 | 0 | 3 | 4 | 71 |
| q9 | 328.5 | 338.7 | 25 | 20 | 7 | 4 | 8 | 0 | 3 | 3 | 74 |
| q10 | 68.3 | 73.6 | 76 | 66 | 38 | 7 | 19 | 5 | 9 | 1 | 21 |
| q11 | 79.6 | 64.2 | 76 | 71 | 28 | 9 | 25 | 6 | 10 | 1 | 21 |
| q12 | 179.3 | 165.3 | 43 | 34 | 17 | 4 | 15 | 3 | 5 | 1 | 55 |
| q13 | 365.8 | 347.0 | 42 | 32 | 19 | 6 | 12 | 2 | 4 | 1 | 56 |
| q14 | 221.5 | 189.7 | 44 | 34 | 17 | 4 | 14 | 4 | 5 | 1 | 55 |
| q15 | 252.5 | 217.1 | 20 | 14 | 9 | 3 | 3 | 0 | 2 | 3 | 79 |
| q16 | 468.2 | 474.1 | 25 | 18 | 10 | 4 | 6 | 0 | 3 | 3 | 74 |
| q17 | 353.8 | 355.5 | 30 | 22 | 12 | 5 | 7 | 0 | 3 | 3 | 68 |
| q18 | 795.5 | 817.5 | 23 | 17 | 8 | 4 | 7 | 0 | 3 | 2 | 76 |
| q19 | 24.3 | 21.2 | 96 | 99 | 47 | 16 | 18 | 6 | 12 | 1 | 0 |
| q20 | 504.2 | 620.9 | 100 | 100 | 28 | 5 | 64 | 0 | 3 | 0 | 0 |
| q21 | 383.7 | 437.7 | 99 | 100 | 41 | 16 | 28 | 7 | 7 | 0 | 0 |
| q22 | 672.5 | 697.9 | 98 | 97 | 45 | 22 | 18 | 5 | 7 | 1 | 1 |
| q23 | 73.8 | 93.8 | 94 | 97 | 35 | 11 | 36 | 2 | 12 | 2 | 2 |
| q24 | 20.8 | 21.5 | 92 | 97 | 52 | 15 | 10 | 5 | 14 | 2 | 1 |
| q25 | 55.4 | 47.9 | 97 | 99 | 38 | 16 | 28 | 6 | 10 | 1 | 1 |
| q26 | 17.4 | 17.6 | 93 | 96 | 47 | 14 | 16 | 5 | 14 | 2 | 2 |
| q27 | 236.9 | 269.4 | 80 | 64 | 46 | 15 | 5 | 2 | 10 | 2 | 19 |
| q28 | 3280.0 | 3755.0 | 13 | 7 | 6 | 1 | 4 | 0 | 1 | 0 | 87 |
| q29 | 19.9 | 13.2 | 70 | 88 | 44 | 15 | 12 | 10 | 12 | 0 | 6 |
| q30 | 168.4 | 203.4 | 66 | 55 | 29 | 10 | 16 | 4 | 7 | 1 | 32 |
| q31 | 250.0 | 264.4 | 71 | 62 | 31 | 16 | 15 | 3 | 7 | 1 | 26 |
| q32 | 898.2 | 929.0 | 15 | 11 | 7 | 3 | 4 | 0 | 2 | 1 | 83 |
| q33 | 835.2 | 975.5 | 43 | 31 | 20 | 6 | 14 | 0 | 4 | 1 | 55 |
| q34 | 1035.8 | 1047.2 | 40 | 28 | 17 | 5 | 13 | 0 | 4 | 1 | 59 |
| q35 | 308.6 | 335.3 | 13 | 7 | 6 | 2 | 1 | 0 | 1 | 2 | 87 |
| q36 | 14.3 | 14.6 | 69 | 62 | 41 | 8 | 9 | 5 | 12 | 2 | 23 |
| q37 | 7.4 | 8.4 | 73 | 77 | 47 | 9 | 6 | 8 | 14 | 3 | 12 |
| q38 | 11.9 | 12.0 | 81 | 87 | 52 | 9 | 11 | 6 | 13 | 2 | 6 |
| q39 | 27.5 | 28.1 | 55 | 47 | 30 | 8 | 11 | 3 | 9 | 2 | 37 |
| q40 | 7.5 | 8.2 | 78 | 84 | 54 | 10 | 3 | 8 | 14 | 2 | 7 |
| q41 | 7.7 | 8.1 | 72 | 83 | 51 | 10 | 6 | 8 | 14 | 2 | 8 |
| q42 | 7.2 | 8.1 | 61 | 63 | 43 | 9 | 4 | 7 | 13 | 3 | 21 |

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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 19.5 | 22.4 | 37 | 22 | 23 | 5 | 3 | 2 | 3 | 4 | 60 |
| q2 | 10.3 | 10.7 | 44 | 39 | 45 | 4 | 7 | 3 | 5 | 1 | 34 |
| q3 | 14.2 | 17.3 | 59 | 55 | 52 | 7 | 9 | 4 | 5 | 1 | 21 |
| q4 | 14.7 | 17.3 | 48 | 48 | 41 | 6 | 14 | 3 | 5 | 0 | 31 |
| q5 | 20.3 | 22.9 | 53 | 47 | 50 | 6 | 9 | 2 | 4 | 1 | 27 |
| q6 | 5.8 | 6.6 | 91 | 96 | 64 | 10 | 8 | 8 | 8 | 0 | 1 |
| q7 | 17.4 | 19.7 | 64 | 63 | 56 | 7 | 10 | 5 | 5 | 1 | 16 |
| q8 | 20.2 | 22.4 | 68 | 67 | 58 | 7 | 11 | 4 | 4 | 1 | 14 |
| q9 | 38.0 | 44.9 | 41 | 28 | 32 | 4 | 8 | 2 | 3 | 1 | 49 |
| q10 | 22.7 | 26.0 | 47 | 37 | 44 | 5 | 8 | 2 | 4 | 1 | 35 |
| q11 | 6.3 | 5.9 | 47 | 42 | 40 | 5 | 10 | 3 | 5 | 2 | 35 |
| q12 | 14.0 | 16.2 | 75 | 74 | 58 | 10 | 8 | 4 | 8 | 0 | 11 |
| q13 | 22.8 | 28.9 | 54 | 57 | 26 | 3 | 7 | 29 | 3 | 0 | 32 |
| q14 | 10.7 | 11.5 | 65 | 66 | 70 | 7 | 5 | 2 | 5 | 0 | 10 |
| q15 | 9.4 | 10.5 | 61 | 53 | 55 | 8 | 5 | 5 | 5 | 1 | 21 |
| q16 | 10.9 | 11.9 | 20 | 14 | 16 | 2 | 4 | 2 | 3 | 0 | 73 |
| q17 | 17.0 | 18.2 | 71 | 61 | 68 | 7 | 6 | 2 | 3 | 2 | 13 |
| q18 | 24.9 | 30.5 | 42 | 25 | 38 | 5 | 6 | 0 | 3 | 2 | 46 |
| q19 | 14.3 | 15.8 | 45 | 54 | 55 | 8 | 7 | 2 | 6 | 0 | 21 |
| q20 | 17.6 | 19.9 | 58 | 51 | 52 | 6 | 9 | 4 | 4 | 1 | 24 |
| q21 | 47.7 | 59.9 | 39 | 45 | 40 | 6 | 13 | 3 | 5 | 0 | 33 |
| q22 | 8.0 | 8.7 | 34 | 23 | 42 | 3 | 4 | 2 | 3 | 1 | 45 |

### duckdb-tpch-1.0 V2: 22 queries
scan share of wall (per-round): median 41%, min 20%, max 87%, mean 43%
scan share of on-CPU: median 45%, mean 46%
threads blocked inside scan, share of busy thread-time: median 31%, mean 29%
  if scan cost falls by 100%: geomean time ratio 0.544 (46% faster); queries reaching >=30% faster: 18/22
  if scan cost falls by 50%: geomean time ratio 0.782 (22% faster); queries reaching >=30% faster: 1/22
  if scan cost falls by 30%: geomean time ratio 0.870 (13% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 6.8%; decode 12.0%; filter/expr kernels 6.3%; pruning 0.2%; sched/layout/dispatch 5.0%; convert to engine 1.8%; blocked in scan (IO wait/handoff) 28.6%; engine on-CPU 39.4%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 4.7%, max 15.1%; io-pool threads: median 33, max 45
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 19.5 | 23.6 | 28 | 21 | 16 | 5 | 3 | 2 | 3 | 4 | 67 |
| q2 | 10.3 | 10.1 | 43 | 45 | 17 | 5 | 17 | 6 | 7 | 2 | 46 |
| q3 | 14.2 | 15.4 | 45 | 51 | 31 | 7 | 15 | 6 | 6 | 2 | 34 |
| q4 | 14.7 | 15.9 | 36 | 41 | 25 | 6 | 16 | 3 | 4 | 1 | 44 |
| q5 | 20.3 | 20.8 | 39 | 44 | 27 | 5 | 15 | 4 | 5 | 2 | 41 |
| q6 | 5.8 | 4.8 | 87 | 95 | 47 | 13 | 15 | 13 | 8 | 0 | 3 |
| q7 | 17.4 | 17.0 | 49 | 59 | 30 | 7 | 17 | 10 | 5 | 1 | 29 |
| q8 | 20.2 | 19.0 | 54 | 66 | 31 | 7 | 22 | 7 | 6 | 2 | 24 |
| q9 | 38.0 | 42.3 | 31 | 25 | 17 | 3 | 11 | 2 | 4 | 2 | 62 |
| q10 | 22.7 | 25.8 | 39 | 37 | 31 | 6 | 11 | 3 | 4 | 1 | 43 |
| q11 | 6.3 | 6.0 | 43 | 40 | 34 | 7 | 9 | 3 | 5 | 2 | 40 |
| q12 | 14.0 | 11.9 | 54 | 62 | 33 | 10 | 14 | 10 | 7 | 1 | 26 |
| q13 | 22.8 | 30.8 | 52 | 58 | 24 | 6 | 7 | 27 | 3 | 0 | 32 |
| q14 | 10.7 | 9.6 | 38 | 59 | 41 | 12 | 12 | 5 | 6 | 0 | 25 |
| q15 | 9.4 | 8.8 | 50 | 52 | 32 | 9 | 10 | 9 | 5 | 2 | 33 |
| q16 | 10.9 | 12.4 | 20 | 13 | 10 | 2 | 4 | 2 | 2 | 2 | 78 |
| q17 | 17.0 | 13.6 | 59 | 64 | 36 | 9 | 17 | 6 | 6 | 3 | 23 |
| q18 | 24.9 | 28.7 | 33 | 26 | 19 | 4 | 6 | 0 | 3 | 7 | 60 |
| q19 | 14.3 | 17.1 | 28 | 44 | 36 | 12 | 7 | 4 | 5 | 1 | 35 |
| q20 | 17.6 | 16.9 | 49 | 51 | 31 | 5 | 15 | 8 | 5 | 2 | 34 |
| q21 | 47.7 | 60.8 | 28 | 38 | 25 | 4 | 15 | 3 | 4 | 2 | 47 |
| q22 | 8.0 | 9.0 | 38 | 31 | 39 | 6 | 6 | 2 | 4 | 1 | 42 |

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
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 186.6 | 181.7 | 37 | 24 | 18 | 5 | 3 | 3 | 3 | 4 | 63 |
| q2 | 34.3 | 34.4 | 61 | 53 | 45 | 5 | 15 | 2 | 5 | 1 | 26 |
| q3 | 124.7 | 119.1 | 62 | 48 | 38 | 8 | 11 | 4 | 5 | 1 | 32 |
| q4 | 119.5 | 121.9 | 55 | 52 | 33 | 7 | 18 | 3 | 5 | 1 | 32 |
| q5 | 158.4 | 142.3 | 67 | 49 | 42 | 8 | 12 | 3 | 4 | 2 | 30 |
| q6 | 44.8 | 44.0 | 98 | 99 | 57 | 11 | 11 | 11 | 9 | 0 | 1 |
| q7 | 139.3 | 127.5 | 75 | 59 | 51 | 8 | 11 | 5 | 4 | 1 | 20 |
| q8 | 169.7 | 160.4 | 58 | 44 | 38 | 8 | 12 | 1 | 4 | 2 | 35 |
| q9 | 364.9 | 364.5 | 39 | 26 | 20 | 5 | 10 | 1 | 3 | 2 | 59 |
| q10 | 156.6 | 160.2 | 62 | 46 | 39 | 7 | 12 | 3 | 5 | 1 | 33 |
| q11 | 29.7 | 28.0 | 69 | 50 | 54 | 7 | 9 | 2 | 3 | 2 | 23 |
| q12 | 105.3 | 106.3 | 86 | 82 | 52 | 12 | 11 | 5 | 10 | 1 | 9 |
| q13 | 245.7 | 222.5 | 65 | 59 | 22 | 6 | 7 | 30 | 2 | 0 | 32 |
| q14 | 63.3 | 62.1 | 81 | 67 | 62 | 10 | 6 | 3 | 5 | 0 | 13 |
| q15 | 83.3 | 82.4 | 77 | 53 | 59 | 8 | 5 | 4 | 4 | 1 | 19 |
| q16 | 48.1 | 48.1 | 24 | 18 | 20 | 3 | 5 | 2 | 3 | 0 | 66 |
| q17 | 130.7 | 119.2 | 71 | 47 | 50 | 8 | 6 | 3 | 4 | 3 | 27 |
| q18 | 235.9 | 232.0 | 48 | 28 | 34 | 6 | 7 | 2 | 3 | 1 | 48 |
| q19 | 76.8 | 77.4 | 79 | 69 | 52 | 11 | 8 | 5 | 8 | 1 | 15 |
| q20 | 140.4 | 118.0 | 72 | 56 | 46 | 7 | 12 | 5 | 4 | 1 | 24 |
| q21 | 444.5 | 415.2 | 50 | 43 | 28 | 7 | 16 | 3 | 4 | 0 | 41 |
| q22 | 55.9 | 50.5 | 41 | 24 | 33 | 5 | 6 | 1 | 3 | 1 | 51 |

### duckdb-tpch-10.0 V2: 22 queries
scan share of wall (per-round): median 54%, min 20%, max 96%, mean 54%
scan share of on-CPU: median 49%, mean 49%
threads blocked inside scan, share of busy thread-time: median 23%, mean 23%
  if scan cost falls by 100%: geomean time ratio 0.413 (59% faster); queries reaching >=30% faster: 20/22
  if scan cost falls by 50%: geomean time ratio 0.726 (27% faster); queries reaching >=30% faster: 8/22
  if scan cost falls by 30%: geomean time ratio 0.837 (16% faster); queries reaching >=30% faster: 0/22
  mean share of busy thread-time per query: io read + dispatch 8.8%; decode 13.8%; filter/expr kernels 6.3%; pruning 0.2%; sched/layout/dispatch 5.5%; convert to engine 2.2%; blocked in scan (IO wait/handoff) 23.0%; engine on-CPU 40.1%
  spin/yield CPU (threadCPUDelta) as share of all CPU: mean 7.9%, max 19.8%; io-pool threads: median 35, max 43
| q | hot ms | profiled ms | scan wall % | scan cpu % | blocked-in-scan % busy | io % | decode % | filter % | sched % | convert % | engine % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q1 | 186.6 | 204.7 | 39 | 25 | 19 | 5 | 3 | 3 | 5 | 4 | 61 |
| q2 | 34.3 | 30.2 | 52 | 54 | 25 | 6 | 22 | 4 | 6 | 2 | 34 |
| q3 | 124.7 | 111.8 | 51 | 45 | 22 | 8 | 15 | 6 | 5 | 2 | 43 |
| q4 | 119.5 | 115.7 | 46 | 48 | 23 | 8 | 20 | 3 | 5 | 1 | 40 |
| q5 | 158.4 | 149.4 | 53 | 45 | 22 | 6 | 18 | 4 | 5 | 2 | 43 |
| q6 | 44.8 | 30.3 | 96 | 98 | 37 | 18 | 17 | 17 | 10 | 0 | 1 |
| q7 | 139.3 | 110.2 | 64 | 61 | 23 | 8 | 22 | 10 | 5 | 1 | 30 |
| q8 | 169.7 | 147.0 | 46 | 43 | 19 | 6 | 15 | 2 | 5 | 6 | 46 |
| q9 | 364.9 | 347.4 | 30 | 23 | 9 | 3 | 11 | 1 | 3 | 3 | 70 |
| q10 | 156.6 | 155.0 | 55 | 47 | 24 | 10 | 15 | 3 | 6 | 2 | 40 |
| q11 | 29.7 | 23.0 | 57 | 53 | 25 | 8 | 19 | 5 | 5 | 3 | 35 |
| q12 | 105.3 | 90.4 | 79 | 80 | 33 | 20 | 15 | 7 | 10 | 1 | 14 |
| q13 | 245.7 | 227.1 | 59 | 55 | 18 | 4 | 6 | 32 | 3 | 0 | 37 |
| q14 | 63.3 | 47.2 | 67 | 63 | 36 | 18 | 10 | 5 | 7 | 1 | 24 |
| q15 | 83.3 | 63.0 | 61 | 54 | 26 | 11 | 11 | 10 | 6 | 2 | 34 |
| q16 | 48.1 | 48.6 | 20 | 18 | 9 | 2 | 6 | 2 | 4 | 2 | 74 |
| q17 | 130.7 | 95.8 | 61 | 50 | 25 | 9 | 14 | 6 | 5 | 4 | 38 |
| q18 | 235.9 | 222.2 | 37 | 29 | 14 | 4 | 7 | 2 | 4 | 8 | 61 |
| q19 | 76.8 | 71.9 | 72 | 69 | 36 | 17 | 11 | 5 | 10 | 1 | 20 |
| q20 | 140.4 | 102.8 | 61 | 57 | 24 | 7 | 19 | 9 | 6 | 2 | 33 |
| q21 | 444.5 | 391.0 | 40 | 38 | 15 | 5 | 17 | 4 | 4 | 2 | 52 |
| q22 | 55.9 | 51.4 | 38 | 33 | 22 | 10 | 8 | 2 | 5 | 1 | 52 |
