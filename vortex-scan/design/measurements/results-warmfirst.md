<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->


(NOT EVICTED: warm page cache, fresh process per query, -i 1)

## COLD duckdb tpch sf=1.0 — 3 reps, fresh process per query, data files (546 MB) evicted from the page cache before every process (load avg 31.1-37.6) [cold-duckdb-tpch-1.0-1790784258]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 22.7 | 24.6 | 1.083 | 168.9 | 168.9 | 1.000 |
| tpch_q02/duckdb:vortex-file-compressed | 13.8 | 13.6 | 0.984 | 168.9 | 168.9 | 1.000 |
| tpch_q03/duckdb:vortex-file-compressed | 17.8 | 20.1 | 1.128 | 168.9 | 168.9 | 1.000 |
| tpch_q04/duckdb:vortex-file-compressed | 17.8 | 18.4 | 1.033 | 168.9 | 168.9 | 1.000 |
| tpch_q05/duckdb:vortex-file-compressed | 24.0 | 23.5 | 0.976 | 168.9 | 168.9 | 1.000 |
| tpch_q06/duckdb:vortex-file-compressed | 7.8 | 5.9 | 0.757 | 169.3 | 169.3 | 1.000 |
| tpch_q07/duckdb:vortex-file-compressed | 21.2 | 20.3 | 0.961 | 169.3 | 169.7 | 1.002 |
| tpch_q08/duckdb:vortex-file-compressed | 24.8 | 22.6 | 0.912 | 169.7 | 169.7 | 1.000 |
| tpch_q09/duckdb:vortex-file-compressed | 45.5 | 47.3 | 1.039 | 169.7 | 169.7 | 1.000 |
| tpch_q10/duckdb:vortex-file-compressed | 28.9 | 26.0 | 0.898 | 169.7 | 169.7 | 1.000 |
| tpch_q11/duckdb:vortex-file-compressed | 9.6 | 10.9 | 1.144 | 169.7 | 169.7 | 1.000 |
| tpch_q12/duckdb:vortex-file-compressed | 17.5 | 13.6 | 0.776 | 169.7 | 169.7 | 1.000 |
| tpch_q13/duckdb:vortex-file-compressed | 26.2 | 32.7 | 1.250 | 169.7 | 169.7 | 1.000 |
| tpch_q14/duckdb:vortex-file-compressed | 12.4 | 12.7 | 1.025 | 169.7 | 169.7 | 1.000 |
| tpch_q15/duckdb:vortex-file-compressed | 11.4 | 12.0 | 1.049 | 169.7 | 169.7 | 1.000 |
| tpch_q16/duckdb:vortex-file-compressed | 14.5 | 15.1 | 1.039 | 169.7 | 169.7 | 1.000 |
| tpch_q17/duckdb:vortex-file-compressed | 21.7 | 18.8 | 0.867 | 169.7 | 169.7 | 1.000 |
| tpch_q18/duckdb:vortex-file-compressed | 29.7 | 28.9 | 0.974 | 169.7 | 169.7 | 1.000 |
| tpch_q19/duckdb:vortex-file-compressed | 17.6 | 20.4 | 1.159 | 169.7 | 169.7 | 1.000 |
| tpch_q20/duckdb:vortex-file-compressed | 23.1 | 21.6 | 0.935 | 169.7 | 169.7 | 1.000 |
| tpch_q21/duckdb:vortex-file-compressed | 54.6 | 53.1 | 0.973 | 170.2 | 170.6 | 1.002 |
| tpch_q22/duckdb:vortex-file-compressed | 10.6 | 11.4 | 1.081 | 170.6 | 170.6 | 1.000 |
| **geomean** | **19.2** | **19.1** | **0.995** | 3730 (sum) | 3731 (sum) | 1.000 |
| **sum** | 473 | 473 | 1.001 | | | |

(NOT EVICTED: warm page cache, fresh process per query, -i 1)

## COLD duckdb tpch sf=10.0 — 3 reps, fresh process per query, data files (2850 MB) evicted from the page cache before every process (load avg 20.7-31.1) [cold-duckdb-tpch-10.0-1790784272]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 170.6 | 183.1 | 1.073 | 1709.3 | 1709.9 | 1.000 |
| tpch_q02/duckdb:vortex-file-compressed | 37.3 | 34.2 | 0.917 | 1709.9 | 1709.9 | 1.000 |
| tpch_q03/duckdb:vortex-file-compressed | 120.1 | 104.2 | 0.868 | 1709.9 | 1709.9 | 1.000 |
| tpch_q04/duckdb:vortex-file-compressed | 111.1 | 102.5 | 0.922 | 1710.2 | 1710.2 | 1.000 |
| tpch_q05/duckdb:vortex-file-compressed | 145.9 | 129.7 | 0.889 | 1710.2 | 1710.6 | 1.000 |
| tpch_q06/duckdb:vortex-file-compressed | 41.1 | 27.6 | 0.672 | 1717.9 | 1710.6 | 0.996 |
| tpch_q07/duckdb:vortex-file-compressed | 120.6 | 107.3 | 0.890 | 1717.9 | 1720.1 | 1.001 |
| tpch_q08/duckdb:vortex-file-compressed | 151.6 | 137.5 | 0.907 | 1721.5 | 1721.5 | 1.000 |
| tpch_q09/duckdb:vortex-file-compressed | 337.7 | 349.2 | 1.034 | 1721.7 | 1722.8 | 1.001 |
| tpch_q10/duckdb:vortex-file-compressed | 163.9 | 163.6 | 0.998 | 1723.3 | 1723.2 | 1.000 |
| tpch_q11/duckdb:vortex-file-compressed | 30.1 | 26.7 | 0.889 | 1723.3 | 1723.7 | 1.000 |
| tpch_q12/duckdb:vortex-file-compressed | 100.0 | 85.6 | 0.856 | 1724.2 | 1723.7 | 1.000 |
| tpch_q13/duckdb:vortex-file-compressed | 206.0 | 207.3 | 1.006 | 1724.8 | 1725.9 | 1.001 |
| tpch_q14/duckdb:vortex-file-compressed | 63.2 | 46.8 | 0.741 | 1725.9 | 1725.9 | 1.000 |
| tpch_q15/duckdb:vortex-file-compressed | 77.4 | 59.9 | 0.774 | 1725.9 | 1725.9 | 1.000 |
| tpch_q16/duckdb:vortex-file-compressed | 48.6 | 49.4 | 1.016 | 1725.9 | 1725.9 | 1.000 |
| tpch_q17/duckdb:vortex-file-compressed | 111.2 | 91.8 | 0.826 | 1725.9 | 1726.6 | 1.000 |
| tpch_q18/duckdb:vortex-file-compressed | 214.7 | 213.9 | 0.996 | 1727.1 | 1727.1 | 1.000 |
| tpch_q19/duckdb:vortex-file-compressed | 81.0 | 75.7 | 0.934 | 1727.3 | 1727.3 | 1.000 |
| tpch_q20/duckdb:vortex-file-compressed | 113.1 | 98.7 | 0.873 | 1730.1 | 1729.5 | 1.000 |
| tpch_q21/duckdb:vortex-file-compressed | 394.0 | 370.3 | 0.940 | 1731.3 | 1732.9 | 1.001 |
| tpch_q22/duckdb:vortex-file-compressed | 50.7 | 52.6 | 1.038 | 1732.9 | 1732.9 | 1.000 |
| **geomean** | **105.6** | **95.7** | **0.906** | 37876 (sum) | 37876 (sum) | 1.000 |
| **sum** | 2890 | 2718 | 0.940 | | | |

(NOT EVICTED: warm page cache, fresh process per query, -i 1)
