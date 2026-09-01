# Morsel Prototype: P1 Evaluation Output

Raw output of `cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval`.
The analysis, and the list of what this run does *not* establish, is in
[`morsel-prototype-p1-findings.md`](morsel-prototype-p1-findings.md).


host: 4 logical cores; segments in memory; 1000000 rows per workload; 5 alternating iterations, median reported

## string-heavy — FineWeb-shaped: wide text plus scalars, five disagreeing chunkings

250000 rows, 62 natural splits

### SH1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 38.137ms | 1.00x | 250000 | 11.336ms | — | — | — | — | — |
| A' V1 (tokio x4) | 11.965ms | 0.31x | 250000 | 1.860ms | — | — | — | — | — |
| D  morsel (x1, splits) | 17.487ms | 0.46x | 250000 | 0.601ms | 62 | 121 | 121 | 121 | 189 |
| D  morsel (x1, splits, no-reuse) | 33.327ms | 0.87x | 250000 | 0.600ms | 62 | 310 | 310 | 310 | 0 |
| D  morsel (x4, splits) | 8.801ms | 0.23x | 250000 | 0.913ms | 62 | 207 | 207 | 158 | 152 |
| D  morsel (x4, 65536r) | 7.892ms | 0.21x | 250000 | 4.738ms | 4 | 121 | 121 | 121 | 0 |
| D  morsel (x4, splits, parallel) | 8.753ms | 0.23x | 250000 | 0.803ms | 62 | 196 | 196 | 156 | 154 |

### SH2 lowcard-eq

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 12.915ms | 1.00x | 31301 | 6.770ms | — | — | — | — | — |
| A' V1 (tokio x4) | 4.866ms | 0.38x | 31301 | 2.504ms | — | — | — | — | — |
| D  morsel (x1, splits) | 7.850ms | 0.61x | 31301 | 0.557ms | 31 | 55 | 55 | 55 | 38 |
| D  morsel (x1, splits, no-reuse) | 13.214ms | 1.02x | 31301 | 0.390ms | 31 | 93 | 93 | 93 | 0 |
| D  morsel (x4, splits) | 4.772ms | 0.37x | 31301 | 0.751ms | 31 | 86 | 86 | 73 | 20 |
| D  morsel (x4, 65536r) | 2.537ms | 0.20x | 31301 | 2.052ms | 4 | 55 | 55 | 55 | 0 |
| D  morsel (x4, splits, parallel) | 3.570ms | 0.28x | 31301 | 0.567ms | 31 | 88 | 88 | 74 | 19 |

### SH3 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 39.345ms | 1.00x | 2496 | 10.610ms | — | — | — | — | — |
| A' V1 (tokio x4) | 12.968ms | 0.33x | 2496 | 3.357ms | — | — | — | — | — |
| D  morsel (x1, splits) | 20.714ms | 0.53x | 2496 | 0.769ms | 62 | 117 | 117 | 116 | 130 |
| D  morsel (x1, splits, no-reuse) | 37.144ms | 0.94x | 2496 | 0.677ms | 62 | 248 | 248 | 246 | 0 |
| D  morsel (x4, splits) | 8.668ms | 0.22x | 2496 | 0.997ms | 62 | 191 | 191 | 167 | 79 |
| D  morsel (x4, 65536r) | 6.456ms | 0.16x | 2496 | 4.836ms | 4 | 117 | 117 | 117 | 0 |
| D  morsel (x4, splits, parallel) | 8.657ms | 0.22x | 2496 | 0.796ms | 62 | 171 | 171 | 151 | 95 |

### SH4 selective

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 11.297ms | 1.00x | 40 | 2.589ms | — | — | — | — | — |
| A' V1 (tokio x4) | 4.758ms | 0.42x | 40 | 1.379ms | — | — | — | — | — |
| D  morsel (x1, splits) | 9.170ms | 0.81x | 40 | 0.517ms | 62 | 134 | 130 | 69 | 139 |
| D  morsel (x1, splits, no-reuse) | 9.767ms | 0.86x | 40 | 0.441ms | 62 | 310 | 248 | 208 | 0 |
| D  morsel (x4, splits) | 3.184ms | 0.28x | 40 | 0.707ms | 62 | 157 | 148 | 78 | 130 |
| D  morsel (x4, 65536r) | 4.789ms | 0.42x | 40 | 3.549ms | 4 | 117 | 113 | 113 | 4 |
| D  morsel (x4, splits, parallel) | 3.025ms | 0.27x | 40 | 0.592ms | 62 | 150 | 145 | 74 | 134 |

### SH5 empty

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.910ms | 1.00x | 0 | — | — | — | — | — | — |
| A' V1 (tokio x4) | 1.487ms | 1.63x | 0 | — | — | — | — | — | — |
| D  morsel (x1, splits) | 0.511ms | 0.56x | 0 | — | 62 | 128 | 128 | 4 | 58 |
| D  morsel (x1, splits, no-reuse) | 0.517ms | 0.57x | 0 | — | 62 | 186 | 186 | 62 | 0 |
| D  morsel (x4, splits) | 0.666ms | 0.73x | 0 | — | 62 | 133 | 133 | 8 | 54 |
| D  morsel (x4, 65536r) | 0.422ms | 0.46x | 0 | — | 4 | 97 | 97 | 4 | 0 |
| D  morsel (x4, splits, parallel) | 0.595ms | 0.65x | 0 | — | 62 | 132 | 132 | 8 | 54 |

### SH6 narrow-project

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.558ms | 1.00x | 125382 | 0.548ms | — | — | — | — | — |
| A' V1 (tokio x4) | 0.545ms | 0.98x | 125382 | 0.478ms | — | — | — | — | — |
| D  morsel (x1, splits) | 0.363ms | 0.65x | 125382 | 0.041ms | 16 | 20 | 20 | 20 | 12 |
| D  morsel (x1, splits, no-reuse) | 0.400ms | 0.72x | 125382 | 0.025ms | 16 | 32 | 32 | 32 | 0 |
| D  morsel (x4, splits) | 0.470ms | 0.84x | 125382 | 0.212ms | 16 | 28 | 28 | 21 | 11 |
| D  morsel (x4, 65536r) | 0.337ms | 0.60x | 125382 | 0.212ms | 4 | 20 | 20 | 20 | 0 |
| D  morsel (x4, splits, parallel) | 0.366ms | 0.66x | 125382 | 0.126ms | 16 | 24 | 24 | 20 | 12 |

## wide-numeric — ClickBench-shaped: 20 narrow integer columns, five disagreeing chunkings

1000000 rows, 228 natural splits

### WN1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 25.740ms | 1.00x | 1000000 | 6.038ms | — | — | — | — | — |
| A' V1 (tokio x4) | 28.834ms | 1.12x | 1000000 | 6.263ms | — | — | — | — | — |
| D  morsel (x1, splits) | 9.609ms | 0.37x | 1000000 | 0.495ms | 228 | 1332 | 1332 | 1332 | 3228 |
| D  morsel (x1, splits, no-reuse) | 12.084ms | 0.47x | 1000000 | 0.158ms | 228 | 4560 | 4560 | 4560 | 0 |
| D  morsel (x4, splits) | 7.870ms | 0.31x | 1000000 | 0.568ms | 228 | 2649 | 2649 | 1504 | 3056 |
| D  morsel (x4, 65536r) | 3.111ms | 0.12x | 1000000 | 0.909ms | 16 | 1451 | 1451 | 1334 | 142 |
| D  morsel (x4, splits, parallel) | 7.745ms | 0.30x | 1000000 | 0.563ms | 228 | 2530 | 2530 | 1452 | 3108 |

### WN2 point-filter

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 3.296ms | 1.00x | 2 | 1.826ms | — | — | — | — | — |
| A' V1 (tokio x4) | 3.477ms | 1.05x | 2 | 2.036ms | — | — | — | — | — |
| D  morsel (x1, splits) | 1.890ms | 0.57x | 2 | 0.665ms | 147 | 389 | 340 | 53 | 100 |
| D  morsel (x1, splits, no-reuse) | 1.813ms | 0.55x | 2 | 0.702ms | 147 | 588 | 441 | 153 | 0 |
| D  morsel (x4, splits) | 1.313ms | 0.40x | 2 | 0.592ms | 147 | 441 | 373 | 70 | 83 |
| D  morsel (x4, 65536r) | 0.947ms | 0.29x | 2 | 0.472ms | 16 | 254 | 205 | 69 | 20 |
| D  morsel (x4, splits, parallel) | 1.345ms | 0.41x | 2 | 0.702ms | 147 | 438 | 373 | 65 | 88 |

### WN3 dashboard

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 11.126ms | 1.00x | 874895 | 2.150ms | — | — | — | — | — |
| A' V1 (tokio x4) | 9.485ms | 0.85x | 874895 | 1.471ms | — | — | — | — | — |
| D  morsel (x1, splits) | 5.238ms | 0.47x | 874895 | 0.198ms | 204 | 517 | 455 | 455 | 973 |
| D  morsel (x1, splits, no-reuse) | 5.975ms | 0.54x | 874895 | 0.080ms | 204 | 1428 | 1224 | 1428 | 0 |
| D  morsel (x4, splits) | 3.660ms | 0.33x | 874895 | 0.409ms | 204 | 875 | 791 | 497 | 931 |
| D  morsel (x4, 65536r) | 1.871ms | 0.17x | 874895 | 0.399ms | 16 | 555 | 493 | 455 | 100 |
| D  morsel (x4, splits, parallel) | 3.557ms | 0.32x | 874895 | 0.325ms | 204 | 901 | 817 | 501 | 927 |

### WN4 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 40.545ms | 1.00x | 15441 | 7.455ms | — | — | — | — | — |
| A' V1 (tokio x4) | 27.190ms | 0.67x | 15441 | 5.847ms | — | — | — | — | — |
| D  morsel (x1, splits) | 15.600ms | 0.38x | 15441 | 0.594ms | 228 | 1425 | 1332 | 1332 | 3684 |
| D  morsel (x1, splits, no-reuse) | 21.340ms | 0.53x | 15441 | 0.287ms | 228 | 5016 | 4560 | 5016 | 0 |
| D  morsel (x4, splits) | 9.562ms | 0.24x | 15441 | 0.733ms | 228 | 3155 | 2985 | 1522 | 3494 |
| D  morsel (x4, 65536r) | 4.675ms | 0.12x | 15441 | 1.220ms | 16 | 1563 | 1470 | 1335 | 234 |
| D  morsel (x4, splits, parallel) | 9.073ms | 0.22x | 15441 | 0.591ms | 228 | 3069 | 2919 | 1484 | 3532 |

### WN5 selective-wide

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 11.239ms | 1.00x | 10 | 3.530ms | — | — | — | — | — |
| A' V1 (tokio x4) | 14.897ms | 1.33x | 10 | 4.370ms | — | — | — | — | — |
| D  morsel (x1, splits) | 7.328ms | 0.65x | 10 | 0.417ms | 228 | 4117 | 3984 | 311 | 334 |
| D  morsel (x1, splits, no-reuse) | 6.613ms | 0.59x | 10 | 0.331ms | 228 | 5016 | 4560 | 645 | 0 |
| D  morsel (x4, splits) | 5.577ms | 0.50x | 10 | 0.648ms | 228 | 4595 | 4333 | 322 | 323 |
| D  morsel (x4, 65536r) | 3.111ms | 0.28x | 10 | 0.904ms | 16 | 1603 | 1461 | 713 | 112 |
| D  morsel (x4, splits, parallel) | 5.298ms | 0.47x | 10 | 0.563ms | 228 | 4599 | 4340 | 324 | 332 |

### WN6 packed

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 4.101ms | 1.00x | 250021 | 0.793ms | — | — | — | — | — |
| A' V1 (tokio x4) | 3.583ms | 0.87x | 250021 | 0.705ms | — | — | — | — | — |
| D  morsel (x1, splits) | 2.411ms | 0.59x | 250021 | 0.069ms | 147 | 193 | 193 | 193 | 248 |
| D  morsel (x1, splits, no-reuse) | 2.486ms | 0.61x | 250021 | 0.042ms | 147 | 441 | 441 | 441 | 0 |
| D  morsel (x4, splits) | 1.362ms | 0.33x | 250021 | 0.240ms | 147 | 342 | 342 | 230 | 211 |
| D  morsel (x4, 65536r) | 0.900ms | 0.22x | 250021 | 0.275ms | 16 | 206 | 206 | 195 | 20 |
| D  morsel (x4, splits, parallel) | 1.266ms | 0.31x | 250021 | 0.193ms | 147 | 339 | 339 | 218 | 223 |

## narrow-analytic — TPC-H Q6/Q1-shaped: conjunctive range filter, narrow projection

1000000 rows, 49 natural splits

### NA1 q6-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 5.706ms | 1.00x | 30093 | 2.032ms | — | — | — | — | — |
| A' V1 (tokio x4) | 3.087ms | 0.54x | 30093 | 1.279ms | — | — | — | — | — |
| D  morsel (x1, splits) | 4.429ms | 0.78x | 30093 | 0.260ms | 49 | 124 | 78 | 78 | 216 |
| D  morsel (x1, splits, no-reuse) | 4.475ms | 0.78x | 30093 | 0.168ms | 49 | 294 | 196 | 294 | 0 |
| D  morsel (x4, splits) | 1.777ms | 0.31x | 30093 | 0.350ms | 49 | 227 | 155 | 83 | 211 |
| D  morsel (x4, 65536r) | 1.568ms | 0.27x | 30093 | 0.471ms | 16 | 146 | 89 | 79 | 89 |
| D  morsel (x4, splits, parallel) | 1.726ms | 0.30x | 30093 | 0.257ms | 49 | 225 | 152 | 80 | 214 |

### NA2 q1-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 2.622ms | 1.00x | 857083 | 1.041ms | — | — | — | — | — |
| A' V1 (tokio x4) | 2.099ms | 0.80x | 857083 | 0.632ms | — | — | — | — | — |
| D  morsel (x1, splits) | 1.665ms | 0.64x | 857083 | 0.084ms | 49 | 78 | 78 | 78 | 118 |
| D  morsel (x1, splits, no-reuse) | 1.628ms | 0.62x | 857083 | 0.050ms | 49 | 196 | 196 | 196 | 0 |
| D  morsel (x4, splits) | 1.066ms | 0.41x | 857083 | 0.237ms | 49 | 142 | 142 | 89 | 107 |
| D  morsel (x4, 65536r) | 0.814ms | 0.31x | 857083 | 0.200ms | 16 | 89 | 89 | 78 | 22 |
| D  morsel (x4, splits, parallel) | 1.054ms | 0.40x | 857083 | 0.162ms | 49 | 144 | 144 | 83 | 113 |

### NA3 scan-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 1.502ms | 1.00x | 1000000 | 0.679ms | — | — | — | — | — |
| A' V1 (tokio x4) | 2.009ms | 1.34x | 1000000 | 0.647ms | — | — | — | — | — |
| D  morsel (x1, splits) | 0.457ms | 0.30x | 1000000 | 0.034ms | 49 | 78 | 78 | 78 | 118 |
| D  morsel (x1, splits, no-reuse) | 0.547ms | 0.36x | 1000000 | 0.029ms | 49 | 196 | 196 | 196 | 0 |
| D  morsel (x4, splits) | 0.640ms | 0.43x | 1000000 | 0.164ms | 49 | 132 | 132 | 99 | 97 |
| D  morsel (x4, 65536r) | 0.467ms | 0.31x | 1000000 | 0.171ms | 16 | 97 | 97 | 87 | 13 |
| D  morsel (x4, splits, parallel) | 0.569ms | 0.38x | 1000000 | 0.154ms | 49 | 151 | 151 | 97 | 99 |

All configurations matched the V1 oracle.
