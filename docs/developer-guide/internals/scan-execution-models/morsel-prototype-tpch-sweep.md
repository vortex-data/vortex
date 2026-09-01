# Morsel Prototype: Scaling Sweep Output

Raw output of
`TPCH_SWEEP=1 cargo run --release -p vortex-morsel --features _test-harness --bin tpch-eval -- 1`.

Three sweeps: driving threads against physical cores (including oversubscription), V1's
concurrent-unit count (workers x per-worker split concurrency, to check the baseline is not
poorly tuned), and morsel size. Analysis in
[`morsel-prototype-tpch-findings.md`](morsel-prototype-tpch-findings.md).


lineitem SF=1: 6001215 rows (6001215 generated), 16 columns,          733 natural splits; generated in 3674.915ms, written in 5606.148ms
written through the btrblocks compressing pipeline (repartition 8192 rows -> coalesce 1048576B -> compress -> buffer -> chunk -> flat); no zone maps, no dict layout
host: 4 logical cores; segments in memory; 5 alternating iterations, median reported

schema: {l_orderkey=i64, l_partkey=i64, l_suppkey=i64, l_linenumber=i32, l_quantity=decimal(15,2), l_extendedprice=decimal(15,2), l_discount=decimal(15,2), l_tax=decimal(15,2), l_returnflag=utf8, l_linestatus=utf8, l_shipdate=vortex.date[days](i32), l_commitdate=vortex.date[days](i32), l_receiptdate=vortex.date[days](i32), l_shipinstruct=utf8, l_shipmode=utf8, l_comment=utf8}

## Driving threads vs cores (4 physical cores, 1 thread per core)

Morsel driver: one morsel in flight per thread. `x4` is one thread per physical core; beyond that the host is oversubscribed.

| query | D x1 | D x2 | D x4 | D x8 | D x16 | best | vs D x4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q6 | 38.033ms | 20.706ms | 10.393ms | 11.839ms | 13.621ms | x4 | 1.00x |
| Q1 | 11.894ms | 6.787ms | 3.786ms | 4.152ms | 5.098ms | x4 | 1.00x |
| Q14 | 14.117ms | 7.988ms | 4.336ms | 4.734ms | 5.476ms | x4 | 1.00x |
| Q15 | 14.533ms | 8.111ms | 4.392ms | 4.693ms | 5.520ms | x4 | 1.00x |
| Q12 | 34.983ms | 18.174ms | 9.470ms | 10.630ms | 11.968ms | x4 | 1.00x |
| Q19 | 24.800ms | 18.831ms | 11.201ms | 11.063ms | 10.852ms | x16 | 0.97x |
| scan-6col | 2.044ms | 1.665ms | 1.293ms | 1.528ms | 2.310ms | x4 | 1.00x |
| selective | 19.359ms | 10.372ms | 5.661ms | 6.142ms | 7.368ms | x4 | 1.00x |

## V1 concurrent units: 4 workers x per-worker split concurrency

V1's parallelism is workers x concurrency. This sweeps the second factor to check the baseline is not simply poorly tuned.

| query | V1 x1 | tok4 c=1 | tok4 c=2 | tok4 c=4 | tok4 c=8 | tok4 c=16 | best |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q6 | 41.525ms | 17.583ms | 15.833ms | 14.205ms | 12.844ms | 12.736ms | 12.736ms |
| Q1 | 14.269ms | 10.856ms | 7.097ms | 6.565ms | 6.456ms | 6.441ms | 6.441ms |
| Q14 | 16.838ms | 9.347ms | 7.264ms | 7.032ms | 6.854ms | 6.730ms | 6.730ms |
| Q15 | 23.859ms | 8.655ms | 7.324ms | 6.633ms | 6.282ms | 5.972ms | 5.972ms |
| Q12 | 41.678ms | 17.855ms | 15.575ms | 13.537ms | 12.960ms | 12.736ms | 12.736ms |
| Q19 | 41.429ms | 35.985ms | 28.621ms | 25.137ms | 24.198ms | 23.542ms | 23.542ms |
| scan-6col | 4.356ms | 5.437ms | 4.547ms | 4.195ms | 3.886ms | 3.924ms | 3.886ms |
| selective | 22.879ms | 10.780ms | 9.222ms | 8.570ms | 8.094ms | 7.592ms | 7.592ms |

## Morsel size at 4 threads

| query | morsels@splits | splits | 16k | 64k | 256k | 1M |
|---|--:|--:|--:|--:|--:|--:|
| Q6 | 92 | 11.206ms | 11.678ms | 11.431ms | 11.665ms | 13.951ms |
| Q1 | 92 | 3.733ms | 3.727ms | 3.757ms | 4.036ms | 4.362ms |
| Q14 | 92 | 4.287ms | 4.043ms | 4.215ms | 4.325ms | 5.384ms |
| Q15 | 92 | 4.243ms | 4.354ms | 4.229ms | 4.185ms | 5.447ms |
| Q12 | 92 | 9.316ms | 9.356ms | 9.336ms | 13.353ms | 22.662ms |
| Q19 | 366 | 11.331ms | 11.591ms | 4.693ms | 4.513ms | 5.665ms |
| scan-6col | 92 | 1.416ms | 1.398ms | 1.466ms | 1.279ms | 1.266ms |
| selective | 92 | 6.097ms | 5.715ms | 5.891ms | 6.068ms | 6.797ms |
