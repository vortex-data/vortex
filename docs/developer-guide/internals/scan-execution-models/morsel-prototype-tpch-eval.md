# Morsel Prototype: TPC-H Evaluation Output

Raw output of
`cargo run --release -p vortex-morsel --features _test-harness --bin tpch-eval -- 1`.
Analysis in [`morsel-prototype-tpch-findings.md`](morsel-prototype-tpch-findings.md).


lineitem SF=1: 6001215 rows (6001215 generated), 16 columns,          733 natural splits; generated in 3585.745ms, written in 5451.060ms
written through the btrblocks compressing pipeline (repartition 8192 rows -> coalesce 1048576B -> compress -> buffer -> chunk -> flat); no zone maps, no dict layout
host: 4 logical cores; segments in memory; 5 alternating iterations, median reported

schema: {l_orderkey=i64, l_partkey=i64, l_suppkey=i64, l_linenumber=i32, l_quantity=decimal(15,2), l_extendedprice=decimal(15,2), l_discount=decimal(15,2), l_tax=decimal(15,2), l_returnflag=utf8, l_linestatus=utf8, l_shipdate=vortex.date[days](i32), l_commitdate=vortex.date[days](i32), l_receiptdate=vortex.date[days](i32), l_shipinstruct=utf8, l_shipmode=utf8, l_comment=utf8}

### Q6 — 114160 rows out (1.90% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 44.759ms | 1.00x | 8.267ms | — | — | — | — |
| A' V1 (tokio x4) | 14.743ms | 0.33x | 2.900ms | — | — | — | — |
| D  morsel (x1, splits) | 41.588ms | 0.93x | 0.776ms | 92 | 368 | 368 | 276 |
| D  morsel (x1, splits, no-reuse) | 41.467ms | 0.93x | 0.726ms | 92 | 368 | 644 | 0 |
| D  morsel (x4, splits) | 14.528ms | 0.32x | 1.156ms | 92 | 368 | 368 | 276 |
| D  morsel (x4, 65536r) | 14.651ms | 0.33x | 0.822ms | 92 | 368 | 368 | 276 |

### Q1 — 5916591 rows out (98.59% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 16.307ms | 1.00x | 3.927ms | — | — | — | — |
| A' V1 (tokio x4) | 8.228ms | 0.50x | 2.013ms | — | — | — | — |
| D  morsel (x1, splits) | 12.391ms | 0.76x | 0.305ms | 92 | 644 | 644 | 0 |
| D  morsel (x1, splits, no-reuse) | 12.918ms | 0.79x | 0.225ms | 92 | 644 | 644 | 0 |
| D  morsel (x4, splits) | 5.174ms | 0.32x | 0.749ms | 92 | 644 | 644 | 0 |
| D  morsel (x4, 65536r) | 4.634ms | 0.28x | 0.563ms | 92 | 644 | 644 | 0 |

### Q14 — 75983 rows out (1.27% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 16.998ms | 1.00x | 3.460ms | — | — | — | — |
| A' V1 (tokio x4) | 7.422ms | 0.44x | 1.816ms | — | — | — | — |
| D  morsel (x1, splits) | 14.697ms | 0.86x | 0.323ms | 92 | 368 | 368 | 92 |
| D  morsel (x1, splits, no-reuse) | 14.911ms | 0.88x | 0.218ms | 92 | 368 | 460 | 0 |
| D  morsel (x4, splits) | 6.170ms | 0.36x | 0.694ms | 92 | 368 | 368 | 92 |
| D  morsel (x4, 65536r) | 4.737ms | 0.28x | 0.542ms | 92 | 368 | 368 | 92 |

### Q15 — 225954 rows out (3.77% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 17.128ms | 1.00x | 3.695ms | — | — | — | — |
| A' V1 (tokio x4) | 8.545ms | 0.50x | 2.295ms | — | — | — | — |
| D  morsel (x1, splits) | 15.063ms | 0.88x | 0.352ms | 92 | 368 | 368 | 92 |
| D  morsel (x1, splits, no-reuse) | 14.946ms | 0.87x | 0.224ms | 92 | 368 | 460 | 0 |
| D  morsel (x4, splits) | 6.564ms | 0.38x | 0.805ms | 92 | 368 | 368 | 92 |
| D  morsel (x4, 65536r) | 6.046ms | 0.35x | 0.459ms | 92 | 368 | 368 | 92 |

### Q12 — 108434 rows out (1.81% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 41.920ms | 1.00x | 8.474ms | — | — | — | — |
| A' V1 (tokio x4) | 16.027ms | 0.38x | 3.709ms | — | — | — | — |
| D  morsel (x1, splits) | 34.916ms | 0.83x | 0.721ms | 92 | 460 | 460 | 460 |
| D  morsel (x1, splits, no-reuse) | 35.334ms | 0.84x | 0.553ms | 92 | 460 | 920 | 0 |
| D  morsel (x4, splits) | 13.331ms | 0.32x | 1.059ms | 92 | 460 | 460 | 460 |
| D  morsel (x4, 65536r) | 13.440ms | 0.32x | 0.874ms | 92 | 460 | 460 | 460 |

### Q19 — 3599028 rows out (59.97% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 47.726ms | 1.00x | 5.003ms | — | — | — | — |
| A' V1 (tokio x4) | 28.291ms | 0.59x | 4.396ms | — | — | — | — |
| D  morsel (x1, splits) | 26.218ms | 0.55x | 0.564ms | 366 | 826 | 826 | 2102 |
| D  morsel (x1, splits, no-reuse) | 33.077ms | 0.69x | 0.341ms | 366 | 2196 | 2928 | 0 |
| D  morsel (x4, splits) | 12.773ms | 0.27x | 0.662ms | 366 | 1994 | 1111 | 1817 |
| D  morsel (x4, 65536r) | 6.447ms | 0.14x | 0.637ms | 92 | 826 | 826 | 184 |

### scan-6col — 6001215 rows out (100.00% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 4.752ms | 1.00x | 1.649ms | — | — | — | — |
| A' V1 (tokio x4) | 4.903ms | 1.03x | 1.930ms | — | — | — | — |
| D  morsel (x1, splits) | 2.236ms | 0.47x | 0.118ms | 92 | 552 | 552 | 0 |
| D  morsel (x1, splits, no-reuse) | 2.015ms | 0.42x | 0.052ms | 92 | 552 | 552 | 0 |
| D  morsel (x4, splits) | 1.890ms | 0.40x | 0.378ms | 92 | 552 | 552 | 0 |
| D  morsel (x4, 65536r) | 1.702ms | 0.36x | 0.308ms | 92 | 552 | 552 | 0 |

### selective — 260 rows out (0.00% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 22.878ms | 1.00x | 4.761ms | — | — | — | — |
| A' V1 (tokio x4) | 9.329ms | 0.41x | 2.353ms | — | — | — | — |
| D  morsel (x1, splits) | 19.650ms | 0.86x | 0.440ms | 92 | 460 | 446 | 92 |
| D  morsel (x1, splits, no-reuse) | 19.339ms | 0.85x | 0.290ms | 92 | 460 | 538 | 0 |
| D  morsel (x4, splits) | 8.646ms | 0.38x | 0.944ms | 92 | 460 | 446 | 92 |
| D  morsel (x4, 65536r) | 7.325ms | 0.32x | 0.528ms | 92 | 460 | 446 | 92 |

Every configuration reproduced V1's dtype, row count and ordered content exactly.
