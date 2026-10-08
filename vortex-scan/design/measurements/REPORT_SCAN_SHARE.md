<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

| engine-suite | cfg | n | scan share of wall: median | mean | time-weighted | >50% | >80% | <20% | worker-only: median | time-weighted | blocked-in-scan mean | geomean ratio if scan -100% | -50% | -30% | scan cut needed for 0.70 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| datafusion-clickbench-- | V1 | 43 | 29% | 41% | 24% | 16 | 9 | 19 | 38% | 25% | 6% | 0.42 | 0.78 | 0.87 | 65% |
| datafusion-clickbench-- | V2 | 43 | 32% | 41% | 25% | 17 | 9 | 19 | 49% | 26% | 7% | 0.43 | 0.78 | 0.87 | 65% |
| datafusion-tpch-1.0 | V1 | 22 | 36% | 39% | 31% | 6 | 1 | 3 | 31% | 30% | 2% | 0.58 | 0.80 | 0.88 | 75% |
| datafusion-tpch-1.0 | V2 | 22 | 38% | 39% | 32% | 5 | 1 | 2 | 33% | 32% | 1% | 0.57 | 0.80 | 0.88 | 73% |
| datafusion-tpch-10.0 | V1 | 22 | 24% | 33% | 22% | 3 | 2 | 9 | 26% | 23% | 2% | 0.60 | 0.83 | 0.90 | 83% |
| datafusion-tpch-10.0 | V2 | 22 | 26% | 33% | 23% | 3 | 2 | 9 | 28% | 26% | 3% | 0.59 | 0.83 | 0.90 | 82% |
| duckdb-clickbench-- | V1 | 43 | 71% | 65% | 43% | 29 | 17 | 2 | 79% | 42% | 39% | 0.20 | 0.66 | 0.80 | 45% |
| duckdb-clickbench-- | V2 | 43 | 69% | 61% | 38% | 27 | 12 | 3 | 76% | 38% | 29% | 0.24 | 0.68 | 0.81 | 48% |
| duckdb-tpch-1.0 | V1 | 22 | 50% | 53% | 50% | 11 | 1 | 0 | 70% | 67% | 47% | 0.44 | 0.73 | 0.84 | 56% |
| duckdb-tpch-1.0 | V2 | 22 | 41% | 43% | 39% | 6 | 1 | 1 | 59% | 54% | 29% | 0.54 | 0.78 | 0.87 | 69% |
| duckdb-tpch-10.0 | V1 | 22 | 63% | 62% | 58% | 16 | 3 | 0 | 67% | 61% | 41% | 0.32 | 0.68 | 0.81 | 48% |
| duckdb-tpch-10.0 | V2 | 22 | 54% | 54% | 48% | 14 | 1 | 1 | 58% | 51% | 23% | 0.41 | 0.73 | 0.84 | 55% |
