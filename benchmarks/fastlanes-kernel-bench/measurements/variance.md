# Run-to-run variance (matrix_run1 vs matrix_run2)

Compared **720** cells (full T x W x SIMD x variant grid).

## Distribution

| percentile | variance % |
|---:|---:|
| p50 | 1.55 |
| p75 | 5.32 |
| p90 | 13.39 |
| p99 | 45.77 |
| max | 83.46 |

**58 cells (= 8.1%) exceed 15% variance and are flagged as noisy.** These cells should not be cited in conclusions.

## Noisy cells (>15% variance)

| T | W | SIMD | variant | run1 ns | run2 ns | variance % |
|---|---:|---|---|---:|---:|---:|
| u8 | 4 | zmm | fused_for | 8.3 | 15.2 | 83.5 |
| u8 | 5 | zmm | bare_unpack | 19.6 | 11.0 | 78.3 |
| u8 | 2 | zmm | bare_unpack | 7.7 | 13.3 | 72.5 |
| u8 | 2 | zmm | fused_for | 13.3 | 7.8 | 71.2 |
| u8 | 6 | zmm | fused_for | 11.9 | 20.2 | 70.0 |
| u32 | 22 | zmm | bare_unpack | 89.7 | 59.7 | 50.2 |
| u16 | 9 | ymm | bare_unpack | 42.7 | 28.7 | 48.9 |
| u8 | 1 | zmm | fused_for | 14.5 | 9.9 | 46.1 |
| u8 | 1 | zmm | bare_unpack | 9.7 | 14.0 | 44.5 |
| u8 | 7 | zmm | fused_for | 24.7 | 17.3 | 42.6 |
| u16 | 12 | ymm | bare_unpack | 38.7 | 28.6 | 35.1 |
| u32 | 1 | zmm | bare_unpack | 63.7 | 47.6 | 33.8 |
| u32 | 26 | zmm | bare_unpack | 61.7 | 81.4 | 31.9 |
| u32 | 1 | zmm | fused_for | 47.7 | 62.7 | 31.3 |
| u16 | 1 | zmm | fused_for | 21.2 | 27.7 | 30.6 |
| u32 | 15 | ymm | fused_for | 64.1 | 80.7 | 25.9 |
| u32 | 7 | zmm | fused_for | 52.3 | 65.7 | 25.6 |
| u32 | 13 | zmm | bare_unpack | 67.4 | 53.7 | 25.5 |
| u16 | 11 | ymm | bare_unpack | 28.6 | 35.7 | 24.7 |
| u16 | 8 | ymm | fused_for | 35.5 | 28.6 | 24.3 |
| u16 | 10 | ymm | bare_unpack | 28.6 | 35.5 | 24.2 |
| u32 | 2 | sse2 | fused_for | 83.5 | 103.7 | 24.2 |
| u16 | 1 | ymm | fused_for | 35.5 | 28.6 | 24.2 |
| u32 | 13 | ymm | fused_for | 79.7 | 64.2 | 24.1 |
| u32 | 25 | sse2 | fused_for | 159.7 | 128.7 | 24.1 |
| u32 | 8 | sse2 | fused_for | 94.7 | 76.4 | 24.0 |
| u32 | 21 | zmm | bare_unpack | 74.0 | 59.7 | 23.9 |
| u16 | 14 | ymm | bare_unpack | 35.4 | 28.6 | 23.8 |
| u16 | 8 | ymm | bare_unpack | 35.4 | 28.7 | 23.5 |
| u16 | 2 | ymm | bare_unpack | 35.3 | 28.6 | 23.4 |
| u32 | 12 | sse2 | fused_for | 113.7 | 92.6 | 22.7 |
| u32 | 18 | zmm | bare_unpack | 81.7 | 66.7 | 22.5 |
| u32 | 30 | ymm | fused_for | 78.2 | 94.7 | 21.1 |
| u16 | 5 | ymm | fused_for | 29.4 | 35.6 | 21.1 |
| u16 | 1 | zmm | bare_unpack | 27.7 | 22.9 | 20.9 |
| u32 | 30 | ymm | bare_unpack | 77.7 | 64.3 | 20.8 |
| u64 | 22 | zmm | bare_unpack | 120.1 | 99.7 | 20.5 |
| u32 | 22 | sse2 | fused_for | 140.7 | 116.8 | 20.5 |
| u32 | 3 | zmm | fused_for | 48.8 | 58.7 | 20.3 |
| u32 | 23 | sse2 | bare_unpack | 130.7 | 108.9 | 20.0 |
| u64 | 19 | zmm | bare_unpack | 98.7 | 118.4 | 20.0 |
| u16 | 11 | ymm | fused_for | 42.8 | 35.8 | 19.7 |
| u64 | 44 | zmm | bare_unpack | 165.9 | 138.6 | 19.7 |
| u16 | 3 | zmm | fused_for | 23.2 | 27.6 | 19.2 |
| u64 | 16 | zmm | bare_unpack | 114.6 | 96.7 | 18.5 |
| u32 | 9 | sse2 | fused_for | 118.7 | 100.2 | 18.5 |
| u32 | 7 | ymm | fused_for | 64.2 | 75.7 | 17.9 |
| u32 | 27 | ymm | fused_for | 77.0 | 90.7 | 17.8 |
| u64 | 40 | zmm | bare_unpack | 157.6 | 134.6 | 17.1 |
| u16 | 13 | sse2 | bare_unpack | 50.1 | 58.7 | 17.1 |
| u64 | 18 | zmm | bare_unpack | 115.3 | 98.7 | 16.9 |
| u16 | 5 | zmm | bare_unpack | 39.4 | 33.8 | 16.4 |
| u32 | 16 | sse2 | fused_for | 76.4 | 88.7 | 16.1 |
| u64 | 33 | zmm | bare_unpack | 120.6 | 139.2 | 15.4 |
| u64 | 29 | zmm | bare_unpack | 123.7 | 142.6 | 15.3 |
| u16 | 7 | ymm | fused_for | 35.6 | 30.9 | 15.2 |
| u32 | 1 | sse2 | bare_unpack | 87.7 | 76.2 | 15.1 |
| u32 | 27 | sse2 | bare_unpack | 130.7 | 113.6 | 15.1 |

## All cells

Sorted by descending variance %. First 50 rows only; full grid is in the CSVs.

| T | W | SIMD | variant | run1 ns | run2 ns | variance % |
|---|---:|---|---|---:|---:|---:|
| u8 | 4 | zmm | fused_for | 8.3 | 15.2 | 83.5 |
| u8 | 5 | zmm | bare_unpack | 19.6 | 11.0 | 78.3 |
| u8 | 2 | zmm | bare_unpack | 7.7 | 13.3 | 72.5 |
| u8 | 2 | zmm | fused_for | 13.3 | 7.8 | 71.2 |
| u8 | 6 | zmm | fused_for | 11.9 | 20.2 | 70.0 |
| u32 | 22 | zmm | bare_unpack | 89.7 | 59.7 | 50.2 |
| u16 | 9 | ymm | bare_unpack | 42.7 | 28.7 | 48.9 |
| u8 | 1 | zmm | fused_for | 14.5 | 9.9 | 46.1 |
| u8 | 1 | zmm | bare_unpack | 9.7 | 14.0 | 44.5 |
| u8 | 7 | zmm | fused_for | 24.7 | 17.3 | 42.6 |
| u16 | 12 | ymm | bare_unpack | 38.7 | 28.6 | 35.1 |
| u32 | 1 | zmm | bare_unpack | 63.7 | 47.6 | 33.8 |
| u32 | 26 | zmm | bare_unpack | 61.7 | 81.4 | 31.9 |
| u32 | 1 | zmm | fused_for | 47.7 | 62.7 | 31.3 |
| u16 | 1 | zmm | fused_for | 21.2 | 27.7 | 30.6 |
| u32 | 15 | ymm | fused_for | 64.1 | 80.7 | 25.9 |
| u32 | 7 | zmm | fused_for | 52.3 | 65.7 | 25.6 |
| u32 | 13 | zmm | bare_unpack | 67.4 | 53.7 | 25.5 |
| u16 | 11 | ymm | bare_unpack | 28.6 | 35.7 | 24.7 |
| u16 | 8 | ymm | fused_for | 35.5 | 28.6 | 24.3 |
| u16 | 10 | ymm | bare_unpack | 28.6 | 35.5 | 24.2 |
| u32 | 2 | sse2 | fused_for | 83.5 | 103.7 | 24.2 |
| u16 | 1 | ymm | fused_for | 35.5 | 28.6 | 24.2 |
| u32 | 13 | ymm | fused_for | 79.7 | 64.2 | 24.1 |
| u32 | 25 | sse2 | fused_for | 159.7 | 128.7 | 24.1 |
| u32 | 8 | sse2 | fused_for | 94.7 | 76.4 | 24.0 |
| u32 | 21 | zmm | bare_unpack | 74.0 | 59.7 | 23.9 |
| u16 | 14 | ymm | bare_unpack | 35.4 | 28.6 | 23.8 |
| u16 | 8 | ymm | bare_unpack | 35.4 | 28.7 | 23.5 |
| u16 | 2 | ymm | bare_unpack | 35.3 | 28.6 | 23.4 |
| u32 | 12 | sse2 | fused_for | 113.7 | 92.6 | 22.7 |
| u32 | 18 | zmm | bare_unpack | 81.7 | 66.7 | 22.5 |
| u32 | 30 | ymm | fused_for | 78.2 | 94.7 | 21.1 |
| u16 | 5 | ymm | fused_for | 29.4 | 35.6 | 21.1 |
| u16 | 1 | zmm | bare_unpack | 27.7 | 22.9 | 20.9 |
| u32 | 30 | ymm | bare_unpack | 77.7 | 64.3 | 20.8 |
| u64 | 22 | zmm | bare_unpack | 120.1 | 99.7 | 20.5 |
| u32 | 22 | sse2 | fused_for | 140.7 | 116.8 | 20.5 |
| u32 | 3 | zmm | fused_for | 48.8 | 58.7 | 20.3 |
| u32 | 23 | sse2 | bare_unpack | 130.7 | 108.9 | 20.0 |
| u64 | 19 | zmm | bare_unpack | 98.7 | 118.4 | 20.0 |
| u16 | 11 | ymm | fused_for | 42.8 | 35.8 | 19.7 |
| u64 | 44 | zmm | bare_unpack | 165.9 | 138.6 | 19.7 |
| u16 | 3 | zmm | fused_for | 23.2 | 27.6 | 19.2 |
| u64 | 16 | zmm | bare_unpack | 114.6 | 96.7 | 18.5 |
| u32 | 9 | sse2 | fused_for | 118.7 | 100.2 | 18.5 |
| u32 | 7 | ymm | fused_for | 64.2 | 75.7 | 17.9 |
| u32 | 27 | ymm | fused_for | 77.0 | 90.7 | 17.8 |
| u64 | 40 | zmm | bare_unpack | 157.6 | 134.6 | 17.1 |
| u16 | 13 | sse2 | bare_unpack | 50.1 | 58.7 | 17.1 |
