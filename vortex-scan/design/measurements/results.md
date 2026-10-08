<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->


## HOT duckdb tpch sf=1.0 — 5 interleaved pairs x 10 iters, first iter of each process dropped (load avg 2.7-11.2) [hot-duckdb-tpch-1.0-1790778136]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 19.51 | 20.15 | 1.032 | 18.50 | 18.66 | 1.009 |
| tpch_q02/duckdb:vortex-file-compressed | 10.31 | 9.43 | 0.914 | 9.61 | 8.63 | 0.898 |
| tpch_q03/duckdb:vortex-file-compressed | 14.23 | 12.34 | 0.867 | 13.44 | 11.36 | 0.845 |
| tpch_q04/duckdb:vortex-file-compressed | 14.70 | 13.57 | 0.923 | 13.43 | 12.38 | 0.922 |
| tpch_q05/duckdb:vortex-file-compressed | 20.27 | 18.21 | 0.899 | 19.16 | 17.06 | 0.890 |
| tpch_q06/duckdb:vortex-file-compressed | 5.81 | 3.96 | 0.682 | 5.46 | 3.66 | 0.670 |
| tpch_q07/duckdb:vortex-file-compressed | 17.44 | 14.96 | 0.858 | 16.20 | 13.64 | 0.842 |
| tpch_q08/duckdb:vortex-file-compressed | 20.21 | 16.89 | 0.836 | 19.39 | 15.89 | 0.820 |
| tpch_q09/duckdb:vortex-file-compressed | 38.02 | 36.39 | 0.957 | 36.07 | 34.52 | 0.957 |
| tpch_q10/duckdb:vortex-file-compressed | 22.71 | 20.15 | 0.887 | 21.32 | 18.97 | 0.890 |
| tpch_q11/duckdb:vortex-file-compressed | 6.27 | 5.93 | 0.945 | 5.18 | 5.02 | 0.969 |
| tpch_q12/duckdb:vortex-file-compressed | 13.97 | 9.84 | 0.704 | 12.82 | 9.05 | 0.706 |
| tpch_q13/duckdb:vortex-file-compressed | 22.77 | 24.66 | 1.083 | 21.75 | 23.44 | 1.078 |
| tpch_q14/duckdb:vortex-file-compressed | 10.68 | 8.45 | 0.791 | 9.70 | 8.11 | 0.836 |
| tpch_q15/duckdb:vortex-file-compressed | 9.38 | 7.83 | 0.834 | 8.74 | 7.27 | 0.832 |
| tpch_q16/duckdb:vortex-file-compressed | 10.89 | 11.63 | 1.068 | 10.16 | 10.96 | 1.079 |
| tpch_q17/duckdb:vortex-file-compressed | 17.01 | 12.44 | 0.731 | 15.94 | 11.67 | 0.732 |
| tpch_q18/duckdb:vortex-file-compressed | 24.87 | 24.45 | 0.983 | 23.57 | 22.18 | 0.941 |
| tpch_q19/duckdb:vortex-file-compressed | 14.26 | 13.97 | 0.980 | 12.99 | 12.64 | 0.973 |
| tpch_q20/duckdb:vortex-file-compressed | 17.57 | 15.74 | 0.896 | 16.48 | 14.75 | 0.895 |
| tpch_q21/duckdb:vortex-file-compressed | 47.74 | 43.75 | 0.916 | 45.93 | 41.34 | 0.900 |
| tpch_q22/duckdb:vortex-file-compressed | 8.05 | 8.21 | 1.021 | 7.27 | 7.31 | 1.005 |
| **geomean** | **15.40** | **13.76** | **0.894** | **14.32** | **12.73** | **0.889** |
| **sum** | 387 | 353 | 0.913 | 363 | 329 | 0.905 |

V1 per process (median of 5): wall 5.04s, cpu 30.36s (user 26.36 sys 4.10), instructions 204.89G

V2 per process (median of 5): wall 4.71s, cpu 29.94s (user 25.37 sys 4.56), instructions 205.06G

V2/V1 per process: instructions 1.001, cpu 0.986, wall 0.935

## HOT duckdb clickbench sf=- — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 19.7-34.4) [hot-duckdb-clickbench---1790778186]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/duckdb:vortex-file-compressed | 1.04 | 1.34 | 1.288 | 0.77 | 0.82 | 1.060 |
| clickbench_q01/duckdb:vortex-file-compressed | 12.72 | 9.19 | 0.722 | 11.70 | 8.43 | 0.721 |
| clickbench_q02/duckdb:vortex-file-compressed | 17.10 | 10.03 | 0.587 | 16.36 | 9.04 | 0.552 |
| clickbench_q03/duckdb:vortex-file-compressed | 45.19 | 31.07 | 0.688 | 43.62 | 30.31 | 0.695 |
| clickbench_q04/duckdb:vortex-file-compressed | 163.17 | 162.79 | 0.998 | 154.80 | 150.86 | 0.975 |
| clickbench_q05/duckdb:vortex-file-compressed | 161.04 | 168.80 | 1.048 | 142.62 | 157.62 | 1.105 |
| clickbench_q06/duckdb:vortex-file-compressed | 1.18 | 1.24 | 1.053 | 0.93 | 1.02 | 1.094 |
| clickbench_q07/duckdb:vortex-file-compressed | 16.64 | 11.02 | 0.662 | 14.44 | 10.26 | 0.710 |
| clickbench_q08/duckdb:vortex-file-compressed | 225.08 | 229.42 | 1.019 | 204.52 | 223.69 | 1.094 |
| clickbench_q09/duckdb:vortex-file-compressed | 328.51 | 339.41 | 1.033 | 298.21 | 325.92 | 1.093 |
| clickbench_q10/duckdb:vortex-file-compressed | 68.31 | 51.11 | 0.748 | 63.35 | 48.63 | 0.768 |
| clickbench_q11/duckdb:vortex-file-compressed | 79.57 | 59.44 | 0.747 | 72.50 | 57.80 | 0.797 |
| clickbench_q12/duckdb:vortex-file-compressed | 179.32 | 170.51 | 0.951 | 150.75 | 164.54 | 1.091 |
| clickbench_q13/duckdb:vortex-file-compressed | 365.81 | 366.71 | 1.002 | 313.53 | 340.73 | 1.087 |
| clickbench_q14/duckdb:vortex-file-compressed | 221.50 | 196.81 | 0.889 | 177.18 | 182.96 | 1.033 |
| clickbench_q15/duckdb:vortex-file-compressed | 252.54 | 224.28 | 0.888 | 203.37 | 207.90 | 1.022 |
| clickbench_q16/duckdb:vortex-file-compressed | 468.21 | 514.25 | 1.098 | 413.26 | 478.96 | 1.159 |
| clickbench_q17/duckdb:vortex-file-compressed | 353.78 | 397.41 | 1.123 | 315.20 | 341.17 | 1.082 |
| clickbench_q18/duckdb:vortex-file-compressed | 795.54 | 834.66 | 1.049 | 719.37 | 777.16 | 1.080 |
| clickbench_q19/duckdb:vortex-file-compressed | 24.29 | 23.54 | 0.969 | 23.69 | 19.70 | 0.832 |
| clickbench_q20/duckdb:vortex-file-compressed | 504.17 | 502.70 | 0.997 | 493.14 | 479.57 | 0.972 |
| clickbench_q21/duckdb:vortex-file-compressed | 383.67 | 370.66 | 0.966 | 349.66 | 345.36 | 0.988 |
| clickbench_q22/duckdb:vortex-file-compressed | 672.53 | 664.54 | 0.988 | 649.36 | 629.33 | 0.969 |
| clickbench_q23/duckdb:vortex-file-compressed | 73.84 | 83.54 | 1.131 | 69.35 | 78.88 | 1.137 |
| clickbench_q24/duckdb:vortex-file-compressed | 20.84 | 18.73 | 0.899 | 19.95 | 17.56 | 0.880 |
| clickbench_q25/duckdb:vortex-file-compressed | 55.41 | 43.16 | 0.779 | 53.93 | 38.76 | 0.719 |
| clickbench_q26/duckdb:vortex-file-compressed | 17.39 | 15.77 | 0.907 | 16.86 | 14.76 | 0.875 |
| clickbench_q27/duckdb:vortex-file-compressed | 236.91 | 213.80 | 0.902 | 228.30 | 200.40 | 0.878 |
| clickbench_q28/duckdb:vortex-file-compressed | 3280.01 | 3258.82 | 0.994 | 3128.52 | 3109.94 | 0.994 |
| clickbench_q29/duckdb:vortex-file-compressed | 19.90 | 12.92 | 0.649 | 19.08 | 12.12 | 0.635 |
| clickbench_q30/duckdb:vortex-file-compressed | 168.36 | 175.32 | 1.041 | 160.63 | 170.85 | 1.064 |
| clickbench_q31/duckdb:vortex-file-compressed | 250.01 | 265.37 | 1.061 | 229.30 | 248.15 | 1.082 |
| clickbench_q32/duckdb:vortex-file-compressed | 898.24 | 942.48 | 1.049 | 785.99 | 816.93 | 1.039 |
| clickbench_q33/duckdb:vortex-file-compressed | 835.19 | 850.20 | 1.018 | 796.62 | 812.04 | 1.019 |
| clickbench_q34/duckdb:vortex-file-compressed | 1035.81 | 965.61 | 0.932 | 899.15 | 898.32 | 0.999 |
| clickbench_q35/duckdb:vortex-file-compressed | 308.62 | 310.27 | 1.005 | 290.24 | 293.13 | 1.010 |
| clickbench_q36/duckdb:vortex-file-compressed | 14.30 | 13.92 | 0.973 | 13.20 | 13.44 | 1.018 |
| clickbench_q37/duckdb:vortex-file-compressed | 7.39 | 8.36 | 1.131 | 7.11 | 7.84 | 1.104 |
| clickbench_q38/duckdb:vortex-file-compressed | 11.89 | 10.99 | 0.925 | 10.90 | 10.21 | 0.937 |
| clickbench_q39/duckdb:vortex-file-compressed | 27.51 | 24.34 | 0.885 | 24.29 | 22.90 | 0.943 |
| clickbench_q40/duckdb:vortex-file-compressed | 7.54 | 7.45 | 0.989 | 6.98 | 7.19 | 1.030 |
| clickbench_q41/duckdb:vortex-file-compressed | 7.68 | 7.21 | 0.939 | 7.05 | 6.97 | 0.989 |
| clickbench_q42/duckdb:vortex-file-compressed | 7.18 | 7.36 | 1.026 | 6.76 | 6.70 | 0.992 |
| **geomean** | **80.86** | **75.66** | **0.936** | **73.64** | **69.82** | **0.948** |
| **sum** | 12625 | 12577 | 0.996 | 11607 | 11779 | 1.015 |

V1 per process (median of 3): wall 65.86s, cpu 729.02s (user 672.48 sys 56.07), instructions 5837.84G

V2 per process (median of 3): wall 66.68s, cpu 744.30s (user 664.41 sys 79.89), instructions 5751.68G

V2/V1 per process: instructions 0.985, cpu 1.021, wall 1.012

## HOT duckdb tpch sf=10.0 — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 24.5-37.2) [hot-duckdb-tpch-10.0-1790778603] **[SUPERSEDED — the cloned duckdb.db held views pointing at the vortex-scan-feature copy of the same files, so this run read those files and the cold run was NOT evicted; see the rerun below]**
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 185.37 | 195.25 | 1.053 | 173.13 | 184.58 | 1.066 |
| tpch_q02/duckdb:vortex-file-compressed | 33.35 | 29.96 | 0.898 | 31.54 | 27.69 | 0.878 |
| tpch_q03/duckdb:vortex-file-compressed | 115.81 | 102.61 | 0.886 | 105.50 | 95.67 | 0.907 |
| tpch_q04/duckdb:vortex-file-compressed | 121.02 | 107.49 | 0.888 | 112.37 | 101.75 | 0.905 |
| tpch_q05/duckdb:vortex-file-compressed | 157.63 | 132.25 | 0.839 | 139.77 | 127.68 | 0.914 |
| tpch_q06/duckdb:vortex-file-compressed | 45.56 | 30.15 | 0.662 | 42.17 | 27.65 | 0.656 |
| tpch_q07/duckdb:vortex-file-compressed | 139.28 | 108.57 | 0.780 | 130.44 | 104.63 | 0.802 |
| tpch_q08/duckdb:vortex-file-compressed | 173.57 | 157.18 | 0.906 | 163.55 | 141.71 | 0.866 |
| tpch_q09/duckdb:vortex-file-compressed | 404.95 | 352.90 | 0.871 | 354.44 | 336.60 | 0.950 |
| tpch_q10/duckdb:vortex-file-compressed | 168.86 | 157.17 | 0.931 | 141.88 | 141.79 | 0.999 |
| tpch_q11/duckdb:vortex-file-compressed | 31.41 | 26.47 | 0.842 | 26.66 | 23.29 | 0.874 |
| tpch_q12/duckdb:vortex-file-compressed | 114.76 | 85.78 | 0.747 | 94.55 | 81.90 | 0.866 |
| tpch_q13/duckdb:vortex-file-compressed | 262.14 | 245.52 | 0.937 | 225.15 | 231.90 | 1.030 |
| tpch_q14/duckdb:vortex-file-compressed | 61.15 | 45.16 | 0.739 | 58.46 | 41.80 | 0.715 |
| tpch_q15/duckdb:vortex-file-compressed | 83.92 | 66.46 | 0.792 | 73.03 | 60.33 | 0.826 |
| tpch_q16/duckdb:vortex-file-compressed | 63.54 | 49.11 | 0.773 | 46.84 | 45.28 | 0.967 |
| tpch_q17/duckdb:vortex-file-compressed | 136.65 | 107.48 | 0.787 | 110.19 | 95.55 | 0.867 |
| tpch_q18/duckdb:vortex-file-compressed | 272.65 | 247.70 | 0.909 | 217.24 | 223.39 | 1.028 |
| tpch_q19/duckdb:vortex-file-compressed | 86.14 | 73.21 | 0.850 | 71.48 | 70.35 | 0.984 |
| tpch_q20/duckdb:vortex-file-compressed | 144.01 | 120.77 | 0.839 | 116.27 | 108.03 | 0.929 |
| tpch_q21/duckdb:vortex-file-compressed | 457.39 | 388.28 | 0.849 | 406.02 | 374.90 | 0.923 |
| tpch_q22/duckdb:vortex-file-compressed | 57.54 | 55.47 | 0.964 | 53.62 | 49.93 | 0.931 |
| **geomean** | **118.06** | **100.06** | **0.848** | **103.23** | **92.75** | **0.898** |
| **sum** | 3317 | 2885 | 0.870 | 2894 | 2696 | 0.932 |

V1 per process (median of 3): wall 17.64s, cpu 162.72s (user 146.19 sys 16.53), instructions 862.38G

V2 per process (median of 3): wall 15.23s, cpu 163.52s (user 141.71 sys 21.81), instructions 878.43G

V2/V1 per process: instructions 1.019, cpu 1.005, wall 0.863

## HOT datafusion tpch sf=1.0 — 5 interleaved pairs x 10 iters, first iter of each process dropped (load avg 22.3-31.9) [hot-datafusion-tpch-1.0-1790778709]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 43.24 | 44.35 | 1.026 | 34.43 | 34.54 | 1.003 |
| tpch_q02/datafusion:vortex-file-compressed | 13.17 | 13.59 | 1.032 | 10.88 | 12.09 | 1.112 |
| tpch_q03/datafusion:vortex-file-compressed | 19.18 | 19.34 | 1.009 | 16.67 | 16.96 | 1.017 |
| tpch_q04/datafusion:vortex-file-compressed | 13.19 | 13.76 | 1.044 | 11.76 | 12.28 | 1.045 |
| tpch_q05/datafusion:vortex-file-compressed | 38.82 | 38.03 | 0.980 | 30.21 | 29.73 | 0.984 |
| tpch_q06/datafusion:vortex-file-compressed | 7.45 | 7.11 | 0.954 | 5.95 | 6.01 | 1.010 |
| tpch_q07/datafusion:vortex-file-compressed | 40.65 | 39.87 | 0.981 | 33.31 | 35.78 | 1.074 |
| tpch_q08/datafusion:vortex-file-compressed | 25.11 | 24.81 | 0.988 | 20.88 | 22.40 | 1.073 |
| tpch_q09/datafusion:vortex-file-compressed | 37.49 | 38.46 | 1.026 | 30.71 | 32.90 | 1.071 |
| tpch_q10/datafusion:vortex-file-compressed | 24.84 | 25.65 | 1.033 | 21.01 | 23.52 | 1.119 |
| tpch_q11/datafusion:vortex-file-compressed | 9.07 | 9.68 | 1.068 | 7.76 | 8.57 | 1.104 |
| tpch_q12/datafusion:vortex-file-compressed | 13.72 | 14.05 | 1.025 | 12.14 | 12.06 | 0.993 |
| tpch_q13/datafusion:vortex-file-compressed | 21.57 | 24.37 | 1.129 | 17.06 | 20.65 | 1.211 |
| tpch_q14/datafusion:vortex-file-compressed | 9.66 | 9.95 | 1.030 | 8.18 | 8.59 | 1.050 |
| tpch_q15/datafusion:vortex-file-compressed | 14.38 | 15.83 | 1.101 | 12.24 | 13.39 | 1.094 |
| tpch_q16/datafusion:vortex-file-compressed | 10.55 | 11.40 | 1.081 | 9.82 | 9.72 | 0.990 |
| tpch_q17/datafusion:vortex-file-compressed | 61.65 | 66.25 | 1.075 | 53.00 | 50.89 | 0.960 |
| tpch_q18/datafusion:vortex-file-compressed | 71.74 | 76.26 | 1.063 | 61.02 | 65.06 | 1.066 |
| tpch_q19/datafusion:vortex-file-compressed | 11.78 | 11.72 | 0.995 | 9.59 | 10.07 | 1.051 |
| tpch_q20/datafusion:vortex-file-compressed | 18.12 | 19.96 | 1.102 | 15.82 | 17.03 | 1.076 |
| tpch_q21/datafusion:vortex-file-compressed | 55.39 | 61.23 | 1.105 | 49.74 | 54.23 | 1.090 |
| tpch_q22/datafusion:vortex-file-compressed | 7.99 | 8.82 | 1.103 | 7.01 | 7.51 | 1.072 |
| **geomean** | **20.40** | **21.26** | **1.042** | **17.24** | **18.21** | **1.056** |
| **sum** | 569 | 594 | 1.045 | 479 | 504 | 1.052 |

V1 per process (median of 5): wall 5.85s, cpu 47.82s (user 41.74 sys 6.08), instructions 231.55G

V2 per process (median of 5): wall 6.17s, cpu 49.97s (user 42.44 sys 7.44), instructions 235.00G

V2/V1 per process: instructions 1.015, cpu 1.045, wall 1.055

## HOT datafusion clickbench sf=- — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 27.6-77.3) [hot-datafusion-clickbench---1790778773]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/datafusion:vortex-file-compressed | 0.60 | 0.62 | 1.021 | 0.52 | 0.56 | 1.081 |
| clickbench_q01/datafusion:vortex-file-compressed | 9.45 | 8.77 | 0.928 | 8.35 | 8.01 | 0.958 |
| clickbench_q02/datafusion:vortex-file-compressed | 29.37 | 29.65 | 1.010 | 27.98 | 27.19 | 0.972 |
| clickbench_q03/datafusion:vortex-file-compressed | 28.93 | 34.27 | 1.185 | 27.36 | 29.88 | 1.092 |
| clickbench_q04/datafusion:vortex-file-compressed | 296.01 | 313.31 | 1.058 | 268.54 | 288.13 | 1.073 |
| clickbench_q05/datafusion:vortex-file-compressed | 317.24 | 333.17 | 1.050 | 308.95 | 313.41 | 1.014 |
| clickbench_q06/datafusion:vortex-file-compressed | 0.90 | 0.88 | 0.978 | 0.76 | 0.80 | 1.052 |
| clickbench_q07/datafusion:vortex-file-compressed | 14.75 | 12.46 | 0.845 | 14.01 | 10.69 | 0.763 |
| clickbench_q08/datafusion:vortex-file-compressed | 396.17 | 413.10 | 1.043 | 386.34 | 397.33 | 1.028 |
| clickbench_q09/datafusion:vortex-file-compressed | 467.13 | 484.08 | 1.036 | 451.40 | 462.24 | 1.024 |
| clickbench_q10/datafusion:vortex-file-compressed | 61.99 | 68.11 | 1.099 | 56.93 | 57.31 | 1.007 |
| clickbench_q11/datafusion:vortex-file-compressed | 81.41 | 88.83 | 1.091 | 79.62 | 76.44 | 0.960 |
| clickbench_q12/datafusion:vortex-file-compressed | 278.08 | 325.80 | 1.172 | 260.91 | 264.38 | 1.013 |
| clickbench_q13/datafusion:vortex-file-compressed | 439.21 | 478.64 | 1.090 | 417.13 | 429.00 | 1.028 |
| clickbench_q14/datafusion:vortex-file-compressed | 288.27 | 291.22 | 1.010 | 273.16 | 272.08 | 0.996 |
| clickbench_q15/datafusion:vortex-file-compressed | 349.06 | 357.00 | 1.023 | 327.68 | 338.31 | 1.032 |
| clickbench_q16/datafusion:vortex-file-compressed | 751.02 | 778.31 | 1.036 | 730.29 | 743.05 | 1.017 |
| clickbench_q17/datafusion:vortex-file-compressed | 759.53 | 766.72 | 1.009 | 709.05 | 739.38 | 1.043 |
| clickbench_q18/datafusion:vortex-file-compressed | 1528.56 | 1590.53 | 1.041 | 1476.17 | 1516.64 | 1.027 |
| clickbench_q19/datafusion:vortex-file-compressed | 24.02 | 25.64 | 1.067 | 19.22 | 21.19 | 1.103 |
| clickbench_q20/datafusion:vortex-file-compressed | 595.48 | 411.16 | 0.690 | 553.44 | 366.74 | 0.663 |
| clickbench_q21/datafusion:vortex-file-compressed | 462.32 | 448.02 | 0.969 | 421.52 | 421.42 | 1.000 |
| clickbench_q22/datafusion:vortex-file-compressed | 508.30 | 570.10 | 1.122 | 481.55 | 527.41 | 1.095 |
| clickbench_q23/datafusion:vortex-file-compressed | 463.34 | 654.29 | 1.412 | 429.46 | 606.81 | 1.413 |
| clickbench_q24/datafusion:vortex-file-compressed | 35.27 | 35.08 | 0.994 | 31.67 | 31.75 | 1.003 |
| clickbench_q25/datafusion:vortex-file-compressed | 69.02 | 71.06 | 1.030 | 56.12 | 60.54 | 1.079 |
| clickbench_q26/datafusion:vortex-file-compressed | 34.72 | 32.52 | 0.937 | 29.76 | 28.99 | 0.974 |
| clickbench_q27/datafusion:vortex-file-compressed | 464.22 | 500.74 | 1.079 | 438.74 | 484.33 | 1.104 |
| clickbench_q28/datafusion:vortex-file-compressed | 2690.38 | 2779.22 | 1.033 | 2554.85 | 2568.12 | 1.005 |
| clickbench_q29/datafusion:vortex-file-compressed | 30.17 | 36.16 | 1.198 | 28.72 | 32.50 | 1.132 |
| clickbench_q30/datafusion:vortex-file-compressed | 252.65 | 262.61 | 1.039 | 241.74 | 245.68 | 1.016 |
| clickbench_q31/datafusion:vortex-file-compressed | 260.57 | 277.13 | 1.064 | 243.99 | 224.12 | 0.919 |
| clickbench_q32/datafusion:vortex-file-compressed | 1133.38 | 1197.74 | 1.057 | 1091.75 | 1104.18 | 1.011 |
| clickbench_q33/datafusion:vortex-file-compressed | 1599.16 | 1644.53 | 1.028 | 1547.48 | 1551.02 | 1.002 |
| clickbench_q34/datafusion:vortex-file-compressed | 1673.29 | 1631.57 | 0.975 | 1565.64 | 1573.36 | 1.005 |
| clickbench_q35/datafusion:vortex-file-compressed | 320.69 | 309.21 | 0.964 | 299.53 | 298.23 | 0.996 |
| clickbench_q36/datafusion:vortex-file-compressed | 29.96 | 29.78 | 0.994 | 24.77 | 25.27 | 1.020 |
| clickbench_q37/datafusion:vortex-file-compressed | 14.57 | 14.99 | 1.029 | 14.08 | 14.36 | 1.020 |
| clickbench_q38/datafusion:vortex-file-compressed | 8.52 | 9.97 | 1.170 | 7.88 | 9.06 | 1.150 |
| clickbench_q39/datafusion:vortex-file-compressed | 55.00 | 55.37 | 1.007 | 51.55 | 52.39 | 1.016 |
| clickbench_q40/datafusion:vortex-file-compressed | 6.06 | 6.11 | 1.008 | 5.85 | 5.43 | 0.927 |
| clickbench_q41/datafusion:vortex-file-compressed | 5.86 | 5.73 | 0.979 | 5.66 | 5.43 | 0.961 |
| clickbench_q42/datafusion:vortex-file-compressed | 6.59 | 5.94 | 0.902 | 5.88 | 5.76 | 0.980 |
| **geomean** | **105.29** | **108.33** | **1.029** | **97.60** | **98.83** | **1.013** |
| **sum** | 16841 | 17390 | 1.033 | 15976 | 16239 | 1.016 |

V1 per process (median of 3): wall 84.44s, cpu 887.40s (user 771.41 sys 115.99), instructions 4898.79G

V2 per process (median of 3): wall 89.06s, cpu 918.68s (user 778.18 sys 140.50), instructions 4828.93G

V2/V1 per process: instructions 0.986, cpu 1.035, wall 1.055

## HOT datafusion tpch sf=10.0 — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 34.0-46.3) [hot-datafusion-tpch-10.0-1790779342]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 500.92 | 553.63 | 1.105 | 477.29 | 486.97 | 1.020 |
| tpch_q02/datafusion:vortex-file-compressed | 83.50 | 84.19 | 1.008 | 73.70 | 74.00 | 1.004 |
| tpch_q03/datafusion:vortex-file-compressed | 179.58 | 181.44 | 1.010 | 170.60 | 171.55 | 1.006 |
| tpch_q04/datafusion:vortex-file-compressed | 99.54 | 98.03 | 0.985 | 90.84 | 90.92 | 1.001 |
| tpch_q05/datafusion:vortex-file-compressed | 365.27 | 354.27 | 0.970 | 344.63 | 338.14 | 0.981 |
| tpch_q06/datafusion:vortex-file-compressed | 42.72 | 41.41 | 0.969 | 41.47 | 37.99 | 0.916 |
| tpch_q07/datafusion:vortex-file-compressed | 515.90 | 504.94 | 0.979 | 490.67 | 474.48 | 0.967 |
| tpch_q08/datafusion:vortex-file-compressed | 311.68 | 316.88 | 1.017 | 291.06 | 297.93 | 1.024 |
| tpch_q09/datafusion:vortex-file-compressed | 613.22 | 620.08 | 1.011 | 592.99 | 573.89 | 0.968 |
| tpch_q10/datafusion:vortex-file-compressed | 221.57 | 238.19 | 1.075 | 212.78 | 216.97 | 1.020 |
| tpch_q11/datafusion:vortex-file-compressed | 70.04 | 71.92 | 1.027 | 66.30 | 66.14 | 0.998 |
| tpch_q12/datafusion:vortex-file-compressed | 110.34 | 108.72 | 0.985 | 105.78 | 100.82 | 0.953 |
| tpch_q13/datafusion:vortex-file-compressed | 246.30 | 251.18 | 1.020 | 241.61 | 228.72 | 0.947 |
| tpch_q14/datafusion:vortex-file-compressed | 59.73 | 59.20 | 0.991 | 58.35 | 53.99 | 0.925 |
| tpch_q15/datafusion:vortex-file-compressed | 103.22 | 105.47 | 1.022 | 95.51 | 99.17 | 1.038 |
| tpch_q16/datafusion:vortex-file-compressed | 65.30 | 63.85 | 0.978 | 58.78 | 58.28 | 0.991 |
| tpch_q17/datafusion:vortex-file-compressed | 674.41 | 674.51 | 1.000 | 653.26 | 642.43 | 0.983 |
| tpch_q18/datafusion:vortex-file-compressed | 950.71 | 921.77 | 0.970 | 900.01 | 879.79 | 0.978 |
| tpch_q19/datafusion:vortex-file-compressed | 65.87 | 83.41 | 1.266 | 58.49 | 75.90 | 1.298 |
| tpch_q20/datafusion:vortex-file-compressed | 163.03 | 164.25 | 1.007 | 151.81 | 155.03 | 1.021 |
| tpch_q21/datafusion:vortex-file-compressed | 675.51 | 671.56 | 0.994 | 641.67 | 634.57 | 0.989 |
| tpch_q22/datafusion:vortex-file-compressed | 58.14 | 57.51 | 0.989 | 54.87 | 53.59 | 0.977 |
| **geomean** | **182.02** | **184.84** | **1.015** | **171.54** | **171.18** | **0.998** |
| **sum** | 6176 | 6226 | 1.008 | 5872 | 5811 | 0.990 |

V1 per process (median of 3): wall 32.06s, cpu 302.80s (user 264.04 sys 38.93), instructions 1406.74G

V2 per process (median of 3): wall 31.90s, cpu 304.33s (user 265.56 sys 39.07), instructions 1425.76G

V2/V1 per process: instructions 1.014, cpu 1.005, wall 0.995

## COLD duckdb tpch sf=1.0 — 3 reps, fresh process per query, data files (546 MB) evicted from the page cache before every process (load avg 25.3-30.8) [cold-duckdb-tpch-1.0-1790779546]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 26.6 | 27.7 | 1.039 | 46.5 | 46.1 | 0.991 |
| tpch_q02/duckdb:vortex-file-compressed | 13.5 | 13.4 | 0.991 | 5.1 | 5.1 | 1.000 |
| tpch_q03/duckdb:vortex-file-compressed | 23.8 | 22.0 | 0.923 | 50.8 | 50.8 | 1.000 |
| tpch_q04/duckdb:vortex-file-compressed | 23.2 | 20.0 | 0.864 | 35.8 | 35.7 | 0.999 |
| tpch_q05/duckdb:vortex-file-compressed | 32.0 | 27.7 | 0.866 | 52.9 | 52.9 | 1.000 |
| tpch_q06/duckdb:vortex-file-compressed | 14.0 | 10.3 | 0.738 | 40.9 | 40.9 | 1.000 |
| tpch_q07/duckdb:vortex-file-compressed | 28.1 | 25.7 | 0.914 | 62.7 | 59.6 | 0.951 |
| tpch_q08/duckdb:vortex-file-compressed | 31.6 | 30.4 | 0.963 | 68.5 | 66.6 | 0.972 |
| tpch_q09/duckdb:vortex-file-compressed | 54.5 | 52.5 | 0.962 | 78.7 | 78.7 | 1.000 |
| tpch_q10/duckdb:vortex-file-compressed | 35.4 | 31.4 | 0.887 | 52.7 | 52.6 | 0.998 |
| tpch_q11/duckdb:vortex-file-compressed | 9.4 | 9.8 | 1.046 | 6.5 | 5.9 | 0.907 |
| tpch_q12/duckdb:vortex-file-compressed | 22.7 | 16.5 | 0.726 | 44.8 | 44.7 | 0.999 |
| tpch_q13/duckdb:vortex-file-compressed | 30.9 | 32.6 | 1.056 | 35.6 | 35.6 | 1.000 |
| tpch_q14/duckdb:vortex-file-compressed | 19.2 | 16.7 | 0.869 | 45.2 | 44.8 | 0.992 |
| tpch_q15/duckdb:vortex-file-compressed | 16.5 | 13.4 | 0.812 | 42.0 | 42.0 | 1.000 |
| tpch_q16/duckdb:vortex-file-compressed | 13.8 | 14.2 | 1.033 | 3.3 | 3.3 | 1.000 |
| tpch_q17/duckdb:vortex-file-compressed | 26.1 | 23.0 | 0.881 | 42.7 | 43.0 | 1.008 |
| tpch_q18/duckdb:vortex-file-compressed | 34.6 | 33.6 | 0.971 | 36.3 | 36.0 | 0.990 |
| tpch_q19/duckdb:vortex-file-compressed | 25.4 | 24.0 | 0.944 | 49.8 | 49.8 | 1.000 |
| tpch_q20/duckdb:vortex-file-compressed | 27.7 | 24.3 | 0.876 | 49.9 | 50.1 | 1.004 |
| tpch_q21/duckdb:vortex-file-compressed | 61.3 | 54.8 | 0.894 | 44.6 | 44.7 | 1.003 |
| tpch_q22/duckdb:vortex-file-compressed | 10.8 | 11.9 | 1.095 | 6.0 | 6.0 | 1.000 |
| **geomean** | **23.7** | **21.8** | **0.920** | 901 (sum) | 895 (sum) | 0.993 |
| **sum** | 581 | 536 | 0.922 | | | |

## COLD duckdb clickbench sf=- — 3 reps, fresh process per query, data files (15115 MB) evicted from the page cache before every process (load avg 17.4-38.5) [cold-duckdb-clickbench---1790779563]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/duckdb:vortex-file-compressed | 30.0 | 26.2 | 0.873 | 9.8 | 9.8 | 1.000 |
| clickbench_q01/duckdb:vortex-file-compressed | 68.3 | 61.1 | 0.894 | 15.3 | 15.3 | 0.997 |
| clickbench_q02/duckdb:vortex-file-compressed | 77.9 | 60.8 | 0.781 | 83.1 | 83.1 | 1.000 |
| clickbench_q03/duckdb:vortex-file-compressed | 172.1 | 136.1 | 0.791 | 345.7 | 342.5 | 0.991 |
| clickbench_q04/duckdb:vortex-file-compressed | 288.6 | 235.0 | 0.814 | 345.7 | 342.6 | 0.991 |
| clickbench_q05/duckdb:vortex-file-compressed | 270.4 | 237.2 | 0.877 | 342.0 | 342.0 | 1.000 |
| clickbench_q06/duckdb:vortex-file-compressed | 57.9 | 48.7 | 0.841 | 11.8 | 11.8 | 1.000 |
| clickbench_q07/duckdb:vortex-file-compressed | 69.0 | 63.3 | 0.918 | 15.3 | 15.3 | 1.000 |
| clickbench_q08/duckdb:vortex-file-compressed | 355.5 | 301.3 | 0.848 | 431.2 | 426.6 | 0.989 |
| clickbench_q09/duckdb:vortex-file-compressed | 489.4 | 443.6 | 0.906 | 503.4 | 503.0 | 0.999 |
| clickbench_q10/duckdb:vortex-file-compressed | 245.4 | 179.1 | 0.730 | 356.5 | 353.9 | 0.993 |
| clickbench_q11/duckdb:vortex-file-compressed | 266.1 | 185.5 | 0.697 | 368.5 | 365.9 | 0.993 |
| clickbench_q12/duckdb:vortex-file-compressed | 306.4 | 271.5 | 0.886 | 344.1 | 344.1 | 1.000 |
| clickbench_q13/duckdb:vortex-file-compressed | 615.7 | 523.9 | 0.851 | 682.5 | 682.3 | 1.000 |
| clickbench_q14/duckdb:vortex-file-compressed | 360.7 | 304.3 | 0.844 | 374.2 | 374.9 | 1.002 |
| clickbench_q15/duckdb:vortex-file-compressed | 330.2 | 286.5 | 0.868 | 345.7 | 344.1 | 0.995 |
| clickbench_q16/duckdb:vortex-file-compressed | 647.1 | 594.3 | 0.918 | 678.7 | 680.9 | 1.003 |
| clickbench_q17/duckdb:vortex-file-compressed | 510.3 | 487.7 | 0.956 | 678.1 | 681.3 | 1.005 |
| clickbench_q18/duckdb:vortex-file-compressed | 1044.5 | 990.9 | 0.949 | 926.2 | 927.1 | 1.001 |
| clickbench_q19/duckdb:vortex-file-compressed | 144.8 | 133.5 | 0.922 | 276.7 | 257.3 | 0.930 |
| clickbench_q20/duckdb:vortex-file-compressed | 1126.9 | 1093.1 | 0.970 | 3019.9 | 3063.1 | 1.014 |
| clickbench_q21/duckdb:vortex-file-compressed | 1375.4 | 1207.9 | 0.878 | 3882.4 | 3659.5 | 0.943 |
| clickbench_q22/duckdb:vortex-file-compressed | 1965.9 | 1993.9 | 1.014 | 5186.7 | 6307.4 | 1.216 |
| clickbench_q23/duckdb:vortex-file-compressed | 320.0 | 317.1 | 0.991 | 752.6 | 754.0 | 1.002 |
| clickbench_q24/duckdb:vortex-file-compressed | 91.8 | 107.1 | 1.167 | 97.5 | 122.3 | 1.254 |
| clickbench_q25/duckdb:vortex-file-compressed | 196.8 | 176.4 | 0.896 | 344.1 | 344.1 | 1.000 |
| clickbench_q26/duckdb:vortex-file-compressed | 84.2 | 96.0 | 1.140 | 88.7 | 110.3 | 1.243 |
| clickbench_q27/duckdb:vortex-file-compressed | 1199.2 | 1044.6 | 0.871 | 3400.6 | 3085.6 | 0.907 |
| clickbench_q28/duckdb:vortex-file-compressed | 3647.9 | 3506.6 | 0.961 | 2785.8 | 2405.2 | 0.863 |
| clickbench_q29/duckdb:vortex-file-compressed | 75.9 | 66.1 | 0.871 | 79.4 | 79.4 | 1.000 |
| clickbench_q30/duckdb:vortex-file-compressed | 410.5 | 350.5 | 0.854 | 671.9 | 673.7 | 1.003 |
| clickbench_q31/duckdb:vortex-file-compressed | 653.3 | 558.8 | 0.855 | 1425.9 | 1421.5 | 0.997 |
| clickbench_q32/duckdb:vortex-file-compressed | 1159.2 | 1097.2 | 0.947 | 1105.1 | 1087.6 | 0.984 |
| clickbench_q33/duckdb:vortex-file-compressed | 1596.2 | 1467.0 | 0.919 | 3398.5 | 3057.0 | 0.900 |
| clickbench_q34/duckdb:vortex-file-compressed | 1868.4 | 1589.9 | 0.851 | 3397.8 | 3065.7 | 0.902 |
| clickbench_q35/duckdb:vortex-file-compressed | 369.1 | 338.9 | 0.918 | 212.1 | 223.9 | 1.056 |
| clickbench_q36/duckdb:vortex-file-compressed | 51.1 | 50.0 | 0.978 | 47.2 | 45.0 | 0.952 |
| clickbench_q37/duckdb:vortex-file-compressed | 36.9 | 41.5 | 1.125 | 15.9 | 16.8 | 1.052 |
| clickbench_q38/duckdb:vortex-file-compressed | 47.9 | 49.2 | 1.027 | 45.1 | 45.6 | 1.011 |
| clickbench_q39/duckdb:vortex-file-compressed | 70.5 | 63.1 | 0.896 | 77.1 | 75.1 | 0.973 |
| clickbench_q40/duckdb:vortex-file-compressed | 36.9 | 43.8 | 1.185 | 24.9 | 24.4 | 0.982 |
| clickbench_q41/duckdb:vortex-file-compressed | 38.3 | 46.3 | 1.210 | 20.6 | 23.6 | 1.146 |
| clickbench_q42/duckdb:vortex-file-compressed | 33.9 | 38.7 | 1.140 | 12.7 | 17.1 | 1.340 |
| **geomean** | **245.2** | **225.4** | **0.919** | 37227 (sum) | 36812 (sum) | 0.989 |
| **sum** | 22806 | 20915 | 0.917 | | | |

## COLD duckdb tpch sf=10.0 — 3 reps, fresh process per query, data files (2850 MB) evicted from the page cache before every process (load avg 25.4-31.0) [cold-duckdb-tpch-10.0-1790779940] **[SUPERSEDED — the cloned duckdb.db held views pointing at the vortex-scan-feature copy of the same files, so this run read those files and the cold run was NOT evicted; see the rerun below]**
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 181.9 | 215.9 | 1.187 | 1.6 | 1.6 | 1.000 |
| tpch_q02/duckdb:vortex-file-compressed | 43.1 | 35.7 | 0.829 | 1.6 | 1.6 | 1.000 |
| tpch_q03/duckdb:vortex-file-compressed | 143.0 | 114.3 | 0.800 | 1.6 | 1.6 | 1.000 |
| tpch_q04/duckdb:vortex-file-compressed | 121.4 | 116.8 | 0.962 | 1.6 | 1.6 | 1.000 |
| tpch_q05/duckdb:vortex-file-compressed | 167.1 | 142.2 | 0.851 | 1.6 | 1.6 | 1.000 |
| tpch_q06/duckdb:vortex-file-compressed | 42.0 | 31.1 | 0.742 | 1.6 | 1.6 | 1.000 |
| tpch_q07/duckdb:vortex-file-compressed | 154.5 | 120.5 | 0.780 | 1.6 | 1.6 | 1.000 |
| tpch_q08/duckdb:vortex-file-compressed | 176.4 | 161.8 | 0.917 | 1.6 | 1.6 | 1.000 |
| tpch_q09/duckdb:vortex-file-compressed | 384.7 | 367.6 | 0.955 | 1.6 | 1.6 | 1.000 |
| tpch_q10/duckdb:vortex-file-compressed | 175.7 | 184.0 | 1.047 | 1.6 | 1.6 | 1.000 |
| tpch_q11/duckdb:vortex-file-compressed | 31.7 | 29.8 | 0.941 | 1.6 | 1.6 | 1.000 |
| tpch_q12/duckdb:vortex-file-compressed | 103.0 | 91.1 | 0.884 | 1.6 | 1.6 | 1.000 |
| tpch_q13/duckdb:vortex-file-compressed | 233.1 | 227.1 | 0.974 | 1.6 | 1.6 | 1.000 |
| tpch_q14/duckdb:vortex-file-compressed | 73.6 | 50.2 | 0.681 | 1.6 | 1.6 | 1.000 |
| tpch_q15/duckdb:vortex-file-compressed | 85.2 | 70.6 | 0.829 | 1.6 | 1.6 | 1.000 |
| tpch_q16/duckdb:vortex-file-compressed | 50.3 | 53.7 | 1.068 | 1.6 | 1.6 | 1.000 |
| tpch_q17/duckdb:vortex-file-compressed | 134.4 | 98.9 | 0.735 | 1.6 | 1.6 | 1.000 |
| tpch_q18/duckdb:vortex-file-compressed | 247.2 | 229.6 | 0.929 | 1.6 | 1.6 | 1.000 |
| tpch_q19/duckdb:vortex-file-compressed | 83.7 | 84.0 | 1.003 | 1.6 | 1.6 | 1.000 |
| tpch_q20/duckdb:vortex-file-compressed | 136.7 | 110.1 | 0.805 | 1.6 | 1.6 | 1.000 |
| tpch_q21/duckdb:vortex-file-compressed | 448.7 | 408.1 | 0.910 | 1.6 | 1.6 | 1.000 |
| tpch_q22/duckdb:vortex-file-compressed | 54.7 | 54.2 | 0.990 | 1.6 | 1.6 | 1.000 |
| **geomean** | **118.2** | **105.5** | **0.893** | 34 (sum) | 34 (sum) | 1.000 |
| **sum** | 3272 | 2997 | 0.916 | | | |

## COLD datafusion tpch sf=1.0 — 3 reps, fresh process per query, data files (546 MB) evicted from the page cache before every process (load avg 23.5-25.4) [cold-datafusion-tpch-1.0-1790779989]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 45.8 | 55.7 | 1.217 | 45.4 | 45.4 | 1.000 |
| tpch_q02/datafusion:vortex-file-compressed | 16.2 | 16.3 | 1.003 | 4.6 | 4.6 | 1.000 |
| tpch_q03/datafusion:vortex-file-compressed | 30.9 | 32.0 | 1.036 | 50.2 | 50.2 | 1.000 |
| tpch_q04/datafusion:vortex-file-compressed | 20.4 | 20.9 | 1.026 | 35.1 | 35.1 | 1.000 |
| tpch_q05/datafusion:vortex-file-compressed | 48.7 | 47.4 | 0.975 | 52.2 | 52.2 | 1.000 |
| tpch_q06/datafusion:vortex-file-compressed | 14.7 | 13.8 | 0.936 | 40.1 | 40.1 | 1.000 |
| tpch_q07/datafusion:vortex-file-compressed | 54.9 | 51.7 | 0.942 | 59.0 | 59.0 | 1.000 |
| tpch_q08/datafusion:vortex-file-compressed | 35.5 | 37.8 | 1.066 | 66.0 | 66.0 | 1.000 |
| tpch_q09/datafusion:vortex-file-compressed | 50.5 | 55.6 | 1.102 | 78.1 | 78.1 | 1.000 |
| tpch_q10/datafusion:vortex-file-compressed | 31.7 | 31.4 | 0.989 | 52.1 | 52.1 | 1.000 |
| tpch_q11/datafusion:vortex-file-compressed | 13.5 | 13.0 | 0.965 | 5.1 | 5.1 | 1.000 |
| tpch_q12/datafusion:vortex-file-compressed | 22.2 | 22.6 | 1.014 | 44.5 | 44.1 | 0.992 |
| tpch_q13/datafusion:vortex-file-compressed | 23.4 | 23.2 | 0.995 | 35.0 | 35.0 | 1.000 |
| tpch_q14/datafusion:vortex-file-compressed | 16.8 | 17.6 | 1.045 | 44.2 | 44.2 | 1.000 |
| tpch_q15/datafusion:vortex-file-compressed | 22.9 | 23.2 | 1.012 | 41.4 | 41.4 | 1.000 |
| tpch_q16/datafusion:vortex-file-compressed | 14.7 | 14.6 | 0.994 | 2.8 | 2.8 | 1.000 |
| tpch_q17/datafusion:vortex-file-compressed | 68.9 | 67.7 | 0.983 | 42.0 | 42.4 | 1.008 |
| tpch_q18/datafusion:vortex-file-compressed | 80.5 | 79.1 | 0.983 | 35.9 | 35.8 | 0.996 |
| tpch_q19/datafusion:vortex-file-compressed | 18.8 | 18.5 | 0.986 | 49.2 | 49.2 | 1.000 |
| tpch_q20/datafusion:vortex-file-compressed | 26.8 | 27.7 | 1.030 | 49.4 | 49.6 | 1.004 |
| tpch_q21/datafusion:vortex-file-compressed | 58.5 | 65.1 | 1.113 | 43.7 | 43.7 | 1.000 |
| tpch_q22/datafusion:vortex-file-compressed | 11.4 | 11.9 | 1.040 | 5.4 | 5.4 | 1.000 |
| **geomean** | **28.2** | **28.7** | **1.019** | 881 (sum) | 881 (sum) | 1.000 |
| **sum** | 728 | 747 | 1.026 | | | |

## COLD datafusion clickbench sf=- — 3 reps, fresh process per query, data files (15115 MB) evicted from the page cache before every process (load avg 17.1-37.8) [cold-datafusion-clickbench---1790780001]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/datafusion:vortex-file-compressed | 19.1 | 24.1 | 1.266 | 9.6 | 9.6 | 1.000 |
| clickbench_q01/datafusion:vortex-file-compressed | 55.0 | 55.9 | 1.017 | 15.1 | 15.0 | 0.995 |
| clickbench_q02/datafusion:vortex-file-compressed | 63.3 | 61.0 | 0.963 | 82.8 | 84.6 | 1.021 |
| clickbench_q03/datafusion:vortex-file-compressed | 112.0 | 113.2 | 1.011 | 344.3 | 343.8 | 0.999 |
| clickbench_q04/datafusion:vortex-file-compressed | 295.4 | 295.8 | 1.001 | 343.1 | 346.6 | 1.010 |
| clickbench_q05/datafusion:vortex-file-compressed | 306.2 | 313.2 | 1.023 | 341.8 | 341.8 | 1.000 |
| clickbench_q06/datafusion:vortex-file-compressed | 21.1 | 18.9 | 0.894 | 9.6 | 9.6 | 1.000 |
| clickbench_q07/datafusion:vortex-file-compressed | 54.9 | 55.3 | 1.007 | 15.0 | 15.0 | 1.000 |
| clickbench_q08/datafusion:vortex-file-compressed | 393.3 | 391.8 | 0.996 | 426.6 | 427.2 | 1.001 |
| clickbench_q09/datafusion:vortex-file-compressed | 464.1 | 470.2 | 1.013 | 499.4 | 499.4 | 1.000 |
| clickbench_q10/datafusion:vortex-file-compressed | 156.8 | 141.0 | 0.899 | 355.2 | 353.6 | 0.995 |
| clickbench_q11/datafusion:vortex-file-compressed | 168.5 | 157.7 | 0.936 | 371.7 | 365.6 | 0.984 |
| clickbench_q12/datafusion:vortex-file-compressed | 273.1 | 286.3 | 1.049 | 343.8 | 343.8 | 1.000 |
| clickbench_q13/datafusion:vortex-file-compressed | 427.8 | 413.1 | 0.966 | 680.9 | 679.9 | 0.998 |
| clickbench_q14/datafusion:vortex-file-compressed | 294.4 | 311.0 | 1.056 | 373.9 | 373.9 | 1.000 |
| clickbench_q15/datafusion:vortex-file-compressed | 330.1 | 339.2 | 1.027 | 344.8 | 345.4 | 1.002 |
| clickbench_q16/datafusion:vortex-file-compressed | 689.0 | 678.7 | 0.985 | 677.2 | 677.3 | 1.000 |
| clickbench_q17/datafusion:vortex-file-compressed | 686.9 | 684.3 | 0.996 | 677.8 | 677.3 | 0.999 |
| clickbench_q18/datafusion:vortex-file-compressed | 1365.9 | 1313.6 | 0.962 | 924.1 | 926.3 | 1.002 |
| clickbench_q19/datafusion:vortex-file-compressed | 116.3 | 123.5 | 1.062 | 279.8 | 276.5 | 0.988 |
| clickbench_q20/datafusion:vortex-file-compressed | 886.0 | 892.0 | 1.007 | 3256.6 | 3257.6 | 1.000 |
| clickbench_q21/datafusion:vortex-file-compressed | 1033.6 | 1030.9 | 0.997 | 3784.9 | 3757.5 | 0.993 |
| clickbench_q22/datafusion:vortex-file-compressed | 1418.7 | 1726.7 | 1.217 | 5225.4 | 6360.4 | 1.217 |
| clickbench_q23/datafusion:vortex-file-compressed | 1261.5 | 1350.0 | 1.070 | 4526.2 | 4733.8 | 1.046 |
| clickbench_q24/datafusion:vortex-file-compressed | 86.4 | 88.0 | 1.018 | 208.2 | 207.6 | 0.997 |
| clickbench_q25/datafusion:vortex-file-compressed | 142.7 | 146.7 | 1.028 | 343.8 | 343.8 | 1.000 |
| clickbench_q26/datafusion:vortex-file-compressed | 92.0 | 86.7 | 0.942 | 208.2 | 211.0 | 1.013 |
| clickbench_q27/datafusion:vortex-file-compressed | 928.8 | 906.0 | 0.976 | 3375.9 | 3268.4 | 0.968 |
| clickbench_q28/datafusion:vortex-file-compressed | 2300.3 | 2287.5 | 0.994 | 2753.6 | 2719.9 | 0.988 |
| clickbench_q29/datafusion:vortex-file-compressed | 59.2 | 68.0 | 1.149 | 79.1 | 80.9 | 1.022 |
| clickbench_q30/datafusion:vortex-file-compressed | 313.9 | 302.3 | 0.963 | 672.3 | 673.1 | 1.001 |
| clickbench_q31/datafusion:vortex-file-compressed | 455.4 | 438.6 | 0.963 | 1431.3 | 1425.5 | 0.996 |
| clickbench_q32/datafusion:vortex-file-compressed | 1049.4 | 1071.6 | 1.021 | 1082.2 | 1085.1 | 1.003 |
| clickbench_q33/datafusion:vortex-file-compressed | 1596.4 | 1603.4 | 1.004 | 3393.4 | 3351.8 | 0.988 |
| clickbench_q34/datafusion:vortex-file-compressed | 1598.7 | 1626.0 | 1.017 | 3401.7 | 3353.2 | 0.986 |
| clickbench_q35/datafusion:vortex-file-compressed | 323.4 | 309.7 | 0.958 | 214.1 | 215.4 | 1.006 |
| clickbench_q36/datafusion:vortex-file-compressed | 56.6 | 54.9 | 0.971 | 45.9 | 46.3 | 1.009 |
| clickbench_q37/datafusion:vortex-file-compressed | 41.4 | 35.1 | 0.848 | 15.6 | 16.6 | 1.060 |
| clickbench_q38/datafusion:vortex-file-compressed | 39.5 | 44.2 | 1.119 | 45.9 | 46.3 | 1.008 |
| clickbench_q39/datafusion:vortex-file-compressed | 87.7 | 95.7 | 1.092 | 81.7 | 75.2 | 0.921 |
| clickbench_q40/datafusion:vortex-file-compressed | 35.0 | 36.2 | 1.034 | 24.6 | 26.2 | 1.067 |
| clickbench_q41/datafusion:vortex-file-compressed | 33.7 | 36.1 | 1.071 | 20.9 | 24.2 | 1.157 |
| clickbench_q42/datafusion:vortex-file-compressed | 30.4 | 33.7 | 1.110 | 12.5 | 15.5 | 1.249 |
| **geomean** | **216.6** | **219.5** | **1.014** | 41321 (sum) | 42408 (sum) | 1.026 |
| **sum** | 20164 | 20518 | 1.018 | | | |

## COLD datafusion tpch sf=10.0 — 3 reps, fresh process per query, data files (2850 MB) evicted from the page cache before every process (load avg 16.6-22.4) [cold-datafusion-tpch-10.0-1790780359]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 403.1 | 477.9 | 1.186 | 401.9 | 401.9 | 1.000 |
| tpch_q02/datafusion:vortex-file-compressed | 88.0 | 88.8 | 1.008 | 45.7 | 45.7 | 1.000 |
| tpch_q03/datafusion:vortex-file-compressed | 196.1 | 216.8 | 1.105 | 502.6 | 502.6 | 1.000 |
| tpch_q04/datafusion:vortex-file-compressed | 111.2 | 107.5 | 0.966 | 346.6 | 346.6 | 1.000 |
| tpch_q05/datafusion:vortex-file-compressed | 431.4 | 414.9 | 0.962 | 540.7 | 540.7 | 1.000 |
| tpch_q06/datafusion:vortex-file-compressed | 82.7 | 78.0 | 0.943 | 348.5 | 348.5 | 1.000 |
| tpch_q07/datafusion:vortex-file-compressed | 467.8 | 481.8 | 1.030 | 608.5 | 608.5 | 1.000 |
| tpch_q08/datafusion:vortex-file-compressed | 429.6 | 442.3 | 1.030 | 701.0 | 701.0 | 1.000 |
| tpch_q09/datafusion:vortex-file-compressed | 755.0 | 700.3 | 0.928 | 766.6 | 766.6 | 1.000 |
| tpch_q10/datafusion:vortex-file-compressed | 226.6 | 241.9 | 1.067 | 536.5 | 536.5 | 1.000 |
| tpch_q11/datafusion:vortex-file-compressed | 67.6 | 73.7 | 1.090 | 48.5 | 48.5 | 1.000 |
| tpch_q12/datafusion:vortex-file-compressed | 144.3 | 142.6 | 0.988 | 437.2 | 437.1 | 1.000 |
| tpch_q13/datafusion:vortex-file-compressed | 213.2 | 214.3 | 1.005 | 361.6 | 361.6 | 1.000 |
| tpch_q14/datafusion:vortex-file-compressed | 110.0 | 109.8 | 0.998 | 463.5 | 463.5 | 1.000 |
| tpch_q15/datafusion:vortex-file-compressed | 137.7 | 145.0 | 1.053 | 435.5 | 435.5 | 1.000 |
| tpch_q16/datafusion:vortex-file-compressed | 69.4 | 64.8 | 0.934 | 24.6 | 24.6 | 1.000 |
| tpch_q17/datafusion:vortex-file-compressed | 643.5 | 649.5 | 1.009 | 388.8 | 388.8 | 1.000 |
| tpch_q18/datafusion:vortex-file-compressed | 790.3 | 802.8 | 1.016 | 304.1 | 304.1 | 1.000 |
| tpch_q19/datafusion:vortex-file-compressed | 104.3 | 123.2 | 1.181 | 459.2 | 459.3 | 1.000 |
| tpch_q20/datafusion:vortex-file-compressed | 177.9 | 187.2 | 1.052 | 487.1 | 487.1 | 1.000 |
| tpch_q21/datafusion:vortex-file-compressed | 602.4 | 594.7 | 0.987 | 451.0 | 451.0 | 1.000 |
| tpch_q22/datafusion:vortex-file-compressed | 52.3 | 57.0 | 1.089 | 57.9 | 57.9 | 1.000 |
| **geomean** | **203.5** | **208.9** | **1.026** | 8717 (sum) | 8718 (sum) | 1.000 |
| **sum** | 6305 | 6415 | 1.018 | | | |

## HOT duckdb tpch sf=10.0 — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 21.4-34.1) [hot-duckdb-tpch-10.0-1790780481]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 187.13 | 202.33 | 1.081 | 180.39 | 190.69 | 1.057 |
| tpch_q02/duckdb:vortex-file-compressed | 34.82 | 31.08 | 0.893 | 33.30 | 28.96 | 0.870 |
| tpch_q03/duckdb:vortex-file-compressed | 127.68 | 108.98 | 0.854 | 115.70 | 102.76 | 0.888 |
| tpch_q04/duckdb:vortex-file-compressed | 119.07 | 113.21 | 0.951 | 115.15 | 106.78 | 0.927 |
| tpch_q05/duckdb:vortex-file-compressed | 158.39 | 132.48 | 0.836 | 143.50 | 122.84 | 0.856 |
| tpch_q06/duckdb:vortex-file-compressed | 43.81 | 29.40 | 0.671 | 42.17 | 27.73 | 0.658 |
| tpch_q07/duckdb:vortex-file-compressed | 139.70 | 109.23 | 0.782 | 129.04 | 103.64 | 0.803 |
| tpch_q08/duckdb:vortex-file-compressed | 165.29 | 149.40 | 0.904 | 154.30 | 144.56 | 0.937 |
| tpch_q09/duckdb:vortex-file-compressed | 357.12 | 350.58 | 0.982 | 346.55 | 342.79 | 0.989 |
| tpch_q10/duckdb:vortex-file-compressed | 152.25 | 160.48 | 1.054 | 144.05 | 146.75 | 1.019 |
| tpch_q11/duckdb:vortex-file-compressed | 29.10 | 24.39 | 0.838 | 28.24 | 23.37 | 0.828 |
| tpch_q12/duckdb:vortex-file-compressed | 103.98 | 88.29 | 0.849 | 100.59 | 86.70 | 0.862 |
| tpch_q13/duckdb:vortex-file-compressed | 237.14 | 241.45 | 1.018 | 216.12 | 221.59 | 1.025 |
| tpch_q14/duckdb:vortex-file-compressed | 65.26 | 44.70 | 0.685 | 60.79 | 43.49 | 0.715 |
| tpch_q15/duckdb:vortex-file-compressed | 82.96 | 64.74 | 0.780 | 80.05 | 59.41 | 0.742 |
| tpch_q16/duckdb:vortex-file-compressed | 47.29 | 47.65 | 1.008 | 45.78 | 45.26 | 0.989 |
| tpch_q17/duckdb:vortex-file-compressed | 126.08 | 97.32 | 0.772 | 116.05 | 93.80 | 0.808 |
| tpch_q18/duckdb:vortex-file-compressed | 233.13 | 229.83 | 0.986 | 216.33 | 215.91 | 0.998 |
| tpch_q19/duckdb:vortex-file-compressed | 75.48 | 74.01 | 0.981 | 73.09 | 67.02 | 0.917 |
| tpch_q20/duckdb:vortex-file-compressed | 133.63 | 122.29 | 0.915 | 117.15 | 109.29 | 0.933 |
| tpch_q21/duckdb:vortex-file-compressed | 430.35 | 422.41 | 0.982 | 411.23 | 397.53 | 0.967 |
| tpch_q22/duckdb:vortex-file-compressed | 52.70 | 54.79 | 1.040 | 49.41 | 51.11 | 1.034 |
| **geomean** | **111.36** | **99.68** | **0.895** | **104.92** | **93.83** | **0.894** |
| **sum** | 3102 | 2899 | 0.934 | 2919 | 2732 | 0.936 |

V1 per process (median of 3): wall 16.50s, cpu 138.92s (user 123.65 sys 15.27), instructions 864.18G

V2 per process (median of 3): wall 15.29s, cpu 141.53s (user 122.36 sys 19.17), instructions 877.21G

V2/V1 per process: instructions 1.015, cpu 1.019, wall 0.927

## COLD duckdb tpch sf=10.0 — 3 reps, fresh process per query, data files (2850 MB) evicted from the page cache before every process (load avg 27.4-34.7) [cold-duckdb-tpch-10.0-1790780583]
| query | V1 cold ms | V2 cold ms | V2/V1 | V1 MB paged in | V2 MB paged in | V2/V1 MB |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 216.0 | 222.9 | 1.032 | 407.9 | 406.2 | 0.996 |
| tpch_q02/duckdb:vortex-file-compressed | 47.0 | 39.2 | 0.834 | 51.3 | 47.3 | 0.922 |
| tpch_q03/duckdb:vortex-file-compressed | 177.3 | 144.3 | 0.814 | 504.4 | 504.0 | 0.999 |
| tpch_q04/duckdb:vortex-file-compressed | 160.1 | 131.4 | 0.821 | 348.7 | 348.9 | 1.001 |
| tpch_q05/duckdb:vortex-file-compressed | 212.2 | 159.1 | 0.750 | 541.3 | 541.5 | 1.000 |
| tpch_q06/duckdb:vortex-file-compressed | 96.8 | 65.4 | 0.676 | 349.8 | 349.3 | 0.999 |
| tpch_q07/duckdb:vortex-file-compressed | 198.5 | 149.1 | 0.751 | 610.0 | 609.4 | 0.999 |
| tpch_q08/duckdb:vortex-file-compressed | 235.4 | 176.7 | 0.751 | 706.0 | 702.5 | 0.995 |
| tpch_q09/duckdb:vortex-file-compressed | 428.5 | 395.9 | 0.924 | 776.0 | 770.7 | 0.993 |
| tpch_q10/duckdb:vortex-file-compressed | 234.0 | 216.8 | 0.927 | 536.6 | 537.4 | 1.002 |
| tpch_q11/duckdb:vortex-file-compressed | 39.1 | 33.8 | 0.863 | 51.3 | 49.9 | 0.972 |
| tpch_q12/duckdb:vortex-file-compressed | 154.1 | 131.3 | 0.852 | 439.5 | 438.4 | 0.997 |
| tpch_q13/duckdb:vortex-file-compressed | 259.0 | 241.5 | 0.933 | 362.9 | 363.1 | 1.000 |
| tpch_q14/duckdb:vortex-file-compressed | 127.6 | 99.5 | 0.780 | 464.7 | 464.3 | 0.999 |
| tpch_q15/duckdb:vortex-file-compressed | 141.3 | 90.7 | 0.642 | 436.3 | 436.3 | 1.000 |
| tpch_q16/duckdb:vortex-file-compressed | 53.9 | 55.8 | 1.036 | 26.6 | 27.1 | 1.019 |
| tpch_q17/duckdb:vortex-file-compressed | 167.3 | 122.6 | 0.733 | 389.6 | 392.4 | 1.007 |
| tpch_q18/duckdb:vortex-file-compressed | 266.8 | 258.0 | 0.967 | 304.6 | 306.9 | 1.008 |
| tpch_q19/duckdb:vortex-file-compressed | 148.9 | 138.7 | 0.931 | 460.0 | 460.0 | 1.000 |
| tpch_q20/duckdb:vortex-file-compressed | 178.8 | 128.6 | 0.719 | 491.4 | 488.2 | 0.993 |
| tpch_q21/duckdb:vortex-file-compressed | 477.8 | 422.5 | 0.884 | 459.4 | 460.0 | 1.001 |
| tpch_q22/duckdb:vortex-file-compressed | 61.1 | 62.0 | 1.015 | 59.2 | 60.3 | 1.019 |
| **geomean** | **154.6** | **129.8** | **0.840** | 8778 (sum) | 8764 (sum) | 0.998 |
| **sum** | 4081 | 3486 | 0.854 | | | |

# ===== SECOND HOT PASS (RERUN2) =====

## HOT duckdb tpch sf=1.0 — 5 interleaved pairs x 10 iters, first iter of each process dropped (load avg 12.8-16.1) [hot-duckdb-tpch-1.0-1790782673]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 22.48 | 22.71 | 1.010 | 19.88 | 20.84 | 1.048 |
| tpch_q02/duckdb:vortex-file-compressed | 10.49 | 9.99 | 0.953 | 9.33 | 8.91 | 0.955 |
| tpch_q03/duckdb:vortex-file-compressed | 15.91 | 13.32 | 0.837 | 14.03 | 11.85 | 0.845 |
| tpch_q04/duckdb:vortex-file-compressed | 16.50 | 15.30 | 0.927 | 14.77 | 13.65 | 0.924 |
| tpch_q05/duckdb:vortex-file-compressed | 22.59 | 20.72 | 0.917 | 19.53 | 17.96 | 0.920 |
| tpch_q06/duckdb:vortex-file-compressed | 6.70 | 5.17 | 0.771 | 5.68 | 4.02 | 0.707 |
| tpch_q07/duckdb:vortex-file-compressed | 19.10 | 16.37 | 0.857 | 17.16 | 14.25 | 0.830 |
| tpch_q08/duckdb:vortex-file-compressed | 22.06 | 18.10 | 0.821 | 19.46 | 16.33 | 0.839 |
| tpch_q09/duckdb:vortex-file-compressed | 41.85 | 40.04 | 0.957 | 37.58 | 36.11 | 0.961 |
| tpch_q10/duckdb:vortex-file-compressed | 24.47 | 22.04 | 0.901 | 22.13 | 19.82 | 0.896 |
| tpch_q11/duckdb:vortex-file-compressed | 6.29 | 6.22 | 0.990 | 5.33 | 5.59 | 1.048 |
| tpch_q12/duckdb:vortex-file-compressed | 15.11 | 11.00 | 0.728 | 13.87 | 9.72 | 0.701 |
| tpch_q13/duckdb:vortex-file-compressed | 25.27 | 28.11 | 1.112 | 22.64 | 25.36 | 1.120 |
| tpch_q14/duckdb:vortex-file-compressed | 11.26 | 8.96 | 0.796 | 10.22 | 8.10 | 0.792 |
| tpch_q15/duckdb:vortex-file-compressed | 10.21 | 8.48 | 0.831 | 9.06 | 7.70 | 0.850 |
| tpch_q16/duckdb:vortex-file-compressed | 11.39 | 12.02 | 1.056 | 10.34 | 10.92 | 1.056 |
| tpch_q17/duckdb:vortex-file-compressed | 18.03 | 13.30 | 0.738 | 16.15 | 11.95 | 0.740 |
| tpch_q18/duckdb:vortex-file-compressed | 28.33 | 26.98 | 0.952 | 26.02 | 23.59 | 0.906 |
| tpch_q19/duckdb:vortex-file-compressed | 15.19 | 14.91 | 0.981 | 14.20 | 13.43 | 0.946 |
| tpch_q20/duckdb:vortex-file-compressed | 18.94 | 16.51 | 0.872 | 16.98 | 15.41 | 0.907 |
| tpch_q21/duckdb:vortex-file-compressed | 52.19 | 48.66 | 0.933 | 47.86 | 45.11 | 0.943 |
| tpch_q22/duckdb:vortex-file-compressed | 8.89 | 8.66 | 0.975 | 7.87 | 7.88 | 1.001 |
| **geomean** | **16.77** | **15.09** | **0.900** | **14.98** | **13.47** | **0.899** |
| **sum** | 423 | 388 | 0.916 | 380 | 348 | 0.917 |

V1 per process (median of 5): wall 5.52s, cpu 29.82s (user 25.62 sys 4.20), instructions 206.35G

V2 per process (median of 5): wall 5.17s, cpu 29.50s (user 24.90 sys 4.59), instructions 205.89G

V2/V1 per process: instructions 0.998, cpu 0.989, wall 0.937

## HOT duckdb tpch sf=10.0 — 4 interleaved pairs x 5 iters, first iter of each process dropped (load avg 11.3-76.5) [hot-duckdb-tpch-10.0-1790782747]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/duckdb:vortex-file-compressed | 174.74 | 187.47 | 1.073 | 166.94 | 179.90 | 1.078 |
| tpch_q02/duckdb:vortex-file-compressed | 33.47 | 29.78 | 0.890 | 31.73 | 28.00 | 0.883 |
| tpch_q03/duckdb:vortex-file-compressed | 114.88 | 98.17 | 0.854 | 107.32 | 93.42 | 0.870 |
| tpch_q04/duckdb:vortex-file-compressed | 117.80 | 104.27 | 0.885 | 106.60 | 98.82 | 0.927 |
| tpch_q05/duckdb:vortex-file-compressed | 141.65 | 121.18 | 0.856 | 135.99 | 112.86 | 0.830 |
| tpch_q06/duckdb:vortex-file-compressed | 41.83 | 27.90 | 0.667 | 39.43 | 25.75 | 0.653 |
| tpch_q07/duckdb:vortex-file-compressed | 120.81 | 101.54 | 0.840 | 112.68 | 97.91 | 0.869 |
| tpch_q08/duckdb:vortex-file-compressed | 151.17 | 134.22 | 0.888 | 142.83 | 127.97 | 0.896 |
| tpch_q09/duckdb:vortex-file-compressed | 333.42 | 339.22 | 1.017 | 318.11 | 328.72 | 1.033 |
| tpch_q10/duckdb:vortex-file-compressed | 140.46 | 143.75 | 1.023 | 130.91 | 131.28 | 1.003 |
| tpch_q11/duckdb:vortex-file-compressed | 27.69 | 23.08 | 0.833 | 26.26 | 21.63 | 0.824 |
| tpch_q12/duckdb:vortex-file-compressed | 98.50 | 82.90 | 0.842 | 93.13 | 77.24 | 0.829 |
| tpch_q13/duckdb:vortex-file-compressed | 213.60 | 220.49 | 1.032 | 195.97 | 205.16 | 1.047 |
| tpch_q14/duckdb:vortex-file-compressed | 62.50 | 42.86 | 0.686 | 57.55 | 40.51 | 0.704 |
| tpch_q15/duckdb:vortex-file-compressed | 78.82 | 58.20 | 0.738 | 75.27 | 55.71 | 0.740 |
| tpch_q16/duckdb:vortex-file-compressed | 47.75 | 45.22 | 0.947 | 41.61 | 42.26 | 1.016 |
| tpch_q17/duckdb:vortex-file-compressed | 115.26 | 92.26 | 0.800 | 110.82 | 86.14 | 0.777 |
| tpch_q18/duckdb:vortex-file-compressed | 220.60 | 214.32 | 0.972 | 212.10 | 206.54 | 0.974 |
| tpch_q19/duckdb:vortex-file-compressed | 73.56 | 65.94 | 0.897 | 71.28 | 61.88 | 0.868 |
| tpch_q20/duckdb:vortex-file-compressed | 116.24 | 105.16 | 0.905 | 110.07 | 99.30 | 0.902 |
| tpch_q21/duckdb:vortex-file-compressed | 394.56 | 366.36 | 0.929 | 374.86 | 346.70 | 0.925 |
| tpch_q22/duckdb:vortex-file-compressed | 48.45 | 49.99 | 1.032 | 46.85 | 47.74 | 1.019 |
| **geomean** | **103.83** | **91.83** | **0.884** | **97.77** | **86.69** | **0.887** |
| **sum** | 2868 | 2654 | 0.926 | 2708 | 2515 | 0.929 |

V1 per process (median of 4): wall 15.19s, cpu 140.32s (user 125.12 sys 15.36), instructions 863.55G

V2 per process (median of 4): wall 14.21s, cpu 147.93s (user 127.12 sys 20.73), instructions 879.47G

V2/V1 per process: instructions 1.018, cpu 1.054, wall 0.936

## HOT datafusion tpch sf=1.0 — 5 interleaved pairs x 10 iters, first iter of each process dropped (load avg 26.9-35.5) [hot-datafusion-tpch-1.0-1790782910]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 37.33 | 38.03 | 1.019 | 29.17 | 33.18 | 1.137 |
| tpch_q02/datafusion:vortex-file-compressed | 10.59 | 11.46 | 1.082 | 9.39 | 10.29 | 1.096 |
| tpch_q03/datafusion:vortex-file-compressed | 17.47 | 17.84 | 1.021 | 16.22 | 16.68 | 1.028 |
| tpch_q04/datafusion:vortex-file-compressed | 11.83 | 12.66 | 1.070 | 10.67 | 11.55 | 1.082 |
| tpch_q05/datafusion:vortex-file-compressed | 30.66 | 31.93 | 1.041 | 27.92 | 28.47 | 1.020 |
| tpch_q06/datafusion:vortex-file-compressed | 6.21 | 6.10 | 0.983 | 5.42 | 4.95 | 0.913 |
| tpch_q07/datafusion:vortex-file-compressed | 33.27 | 35.91 | 1.079 | 30.79 | 30.88 | 1.003 |
| tpch_q08/datafusion:vortex-file-compressed | 21.04 | 22.68 | 1.078 | 19.25 | 20.30 | 1.054 |
| tpch_q09/datafusion:vortex-file-compressed | 30.26 | 33.53 | 1.108 | 28.44 | 30.19 | 1.061 |
| tpch_q10/datafusion:vortex-file-compressed | 20.24 | 21.56 | 1.065 | 18.77 | 19.85 | 1.057 |
| tpch_q11/datafusion:vortex-file-compressed | 7.56 | 8.39 | 1.111 | 7.00 | 7.65 | 1.093 |
| tpch_q12/datafusion:vortex-file-compressed | 12.47 | 12.82 | 1.028 | 11.57 | 11.20 | 0.968 |
| tpch_q13/datafusion:vortex-file-compressed | 18.48 | 20.39 | 1.103 | 15.72 | 17.76 | 1.130 |
| tpch_q14/datafusion:vortex-file-compressed | 7.56 | 7.75 | 1.025 | 6.53 | 6.99 | 1.070 |
| tpch_q15/datafusion:vortex-file-compressed | 12.11 | 12.62 | 1.042 | 10.91 | 11.45 | 1.050 |
| tpch_q16/datafusion:vortex-file-compressed | 9.23 | 9.29 | 1.006 | 8.60 | 8.75 | 1.018 |
| tpch_q17/datafusion:vortex-file-compressed | 49.85 | 51.52 | 1.034 | 45.08 | 45.20 | 1.003 |
| tpch_q18/datafusion:vortex-file-compressed | 61.87 | 62.94 | 1.017 | 55.11 | 57.31 | 1.040 |
| tpch_q19/datafusion:vortex-file-compressed | 10.13 | 9.93 | 0.980 | 8.53 | 8.59 | 1.007 |
| tpch_q20/datafusion:vortex-file-compressed | 16.40 | 16.92 | 1.032 | 15.27 | 15.30 | 1.002 |
| tpch_q21/datafusion:vortex-file-compressed | 48.39 | 51.45 | 1.063 | 44.42 | 44.39 | 0.999 |
| tpch_q22/datafusion:vortex-file-compressed | 7.16 | 7.46 | 1.042 | 6.26 | 6.72 | 1.073 |
| **geomean** | **17.31** | **18.11** | **1.046** | **15.52** | **16.14** | **1.040** |
| **sum** | 480 | 503 | 1.048 | 431 | 448 | 1.038 |

V1 per process (median of 5): wall 5.04s, cpu 40.38s (user 35.16 sys 5.22), instructions 231.40G

V2 per process (median of 5): wall 5.33s, cpu 42.72s (user 36.12 sys 6.39), instructions 234.78G

V2/V1 per process: instructions 1.015, cpu 1.058, wall 1.058

## HOT datafusion tpch sf=10.0 — 4 interleaved pairs x 5 iters, first iter of each process dropped (load avg 18.0-39.8) [hot-datafusion-tpch-10.0-1790782983]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| tpch_q01/datafusion:vortex-file-compressed | 441.23 | 441.40 | 1.000 | 392.70 | 384.21 | 0.978 |
| tpch_q02/datafusion:vortex-file-compressed | 70.98 | 69.37 | 0.977 | 66.57 | 65.60 | 0.985 |
| tpch_q03/datafusion:vortex-file-compressed | 170.18 | 161.21 | 0.947 | 150.51 | 142.88 | 0.949 |
| tpch_q04/datafusion:vortex-file-compressed | 88.47 | 87.68 | 0.991 | 79.23 | 77.84 | 0.982 |
| tpch_q05/datafusion:vortex-file-compressed | 311.59 | 304.01 | 0.976 | 288.90 | 271.68 | 0.940 |
| tpch_q06/datafusion:vortex-file-compressed | 36.61 | 36.62 | 1.000 | 34.85 | 34.36 | 0.986 |
| tpch_q07/datafusion:vortex-file-compressed | 443.68 | 435.83 | 0.982 | 410.62 | 399.05 | 0.972 |
| tpch_q08/datafusion:vortex-file-compressed | 267.54 | 287.93 | 1.076 | 244.26 | 258.35 | 1.058 |
| tpch_q09/datafusion:vortex-file-compressed | 535.76 | 546.40 | 1.020 | 500.22 | 494.11 | 0.988 |
| tpch_q10/datafusion:vortex-file-compressed | 189.90 | 200.93 | 1.058 | 171.28 | 179.74 | 1.049 |
| tpch_q11/datafusion:vortex-file-compressed | 59.30 | 61.67 | 1.040 | 54.90 | 56.77 | 1.034 |
| tpch_q12/datafusion:vortex-file-compressed | 92.70 | 94.58 | 1.020 | 86.44 | 87.67 | 1.014 |
| tpch_q13/datafusion:vortex-file-compressed | 204.04 | 206.72 | 1.013 | 192.86 | 194.32 | 1.008 |
| tpch_q14/datafusion:vortex-file-compressed | 51.89 | 53.99 | 1.040 | 44.88 | 48.47 | 1.080 |
| tpch_q15/datafusion:vortex-file-compressed | 82.38 | 86.90 | 1.055 | 78.70 | 83.26 | 1.058 |
| tpch_q16/datafusion:vortex-file-compressed | 52.43 | 53.59 | 1.022 | 49.58 | 49.03 | 0.989 |
| tpch_q17/datafusion:vortex-file-compressed | 575.49 | 547.75 | 0.952 | 517.91 | 516.23 | 0.997 |
| tpch_q18/datafusion:vortex-file-compressed | 783.58 | 753.66 | 0.962 | 709.16 | 726.56 | 1.025 |
| tpch_q19/datafusion:vortex-file-compressed | 59.12 | 70.08 | 1.185 | 51.88 | 66.39 | 1.280 |
| tpch_q20/datafusion:vortex-file-compressed | 139.24 | 138.60 | 0.995 | 126.58 | 132.93 | 1.050 |
| tpch_q21/datafusion:vortex-file-compressed | 631.73 | 566.57 | 0.897 | 559.72 | 547.65 | 0.978 |
| tpch_q22/datafusion:vortex-file-compressed | 50.62 | 48.25 | 0.953 | 45.31 | 46.38 | 1.024 |
| **geomean** | **156.63** | **157.55** | **1.006** | **143.08** | **145.54** | **1.017** |
| **sum** | 5338 | 5254 | 0.984 | 4857 | 4863 | 1.001 |

V1 per process (median of 4): wall 27.32s, cpu 278.37s (user 242.47 sys 36.55), instructions 1403.40G

V2 per process (median of 4): wall 27.28s, cpu 282.19s (user 244.92 sys 36.58), instructions 1424.83G

V2/V1 per process: instructions 1.015, cpu 1.014, wall 0.999

## HOT duckdb clickbench sf=- — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 17.8-31.1) [hot-duckdb-clickbench---1790783234]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/duckdb:vortex-file-compressed | 1.04 | 1.29 | 1.243 | 0.76 | 0.76 | 0.999 |
| clickbench_q01/duckdb:vortex-file-compressed | 13.53 | 9.10 | 0.672 | 12.15 | 8.59 | 0.707 |
| clickbench_q02/duckdb:vortex-file-compressed | 17.95 | 9.36 | 0.522 | 16.92 | 8.80 | 0.520 |
| clickbench_q03/duckdb:vortex-file-compressed | 46.22 | 30.39 | 0.657 | 45.26 | 29.75 | 0.657 |
| clickbench_q04/duckdb:vortex-file-compressed | 196.53 | 163.27 | 0.831 | 176.13 | 145.58 | 0.827 |
| clickbench_q05/duckdb:vortex-file-compressed | 194.39 | 171.37 | 0.882 | 182.78 | 167.46 | 0.916 |
| clickbench_q06/duckdb:vortex-file-compressed | 1.43 | 1.58 | 1.105 | 1.18 | 1.23 | 1.044 |
| clickbench_q07/duckdb:vortex-file-compressed | 17.21 | 12.02 | 0.699 | 14.91 | 10.40 | 0.697 |
| clickbench_q08/duckdb:vortex-file-compressed | 255.75 | 261.79 | 1.024 | 244.31 | 233.17 | 0.954 |
| clickbench_q09/duckdb:vortex-file-compressed | 371.98 | 350.09 | 0.941 | 352.21 | 341.18 | 0.969 |
| clickbench_q10/duckdb:vortex-file-compressed | 72.22 | 52.75 | 0.730 | 67.05 | 50.18 | 0.748 |
| clickbench_q11/duckdb:vortex-file-compressed | 82.51 | 63.06 | 0.764 | 78.04 | 59.47 | 0.762 |
| clickbench_q12/duckdb:vortex-file-compressed | 199.58 | 178.73 | 0.896 | 189.72 | 173.82 | 0.916 |
| clickbench_q13/duckdb:vortex-file-compressed | 419.99 | 386.21 | 0.920 | 386.18 | 363.58 | 0.941 |
| clickbench_q14/duckdb:vortex-file-compressed | 243.04 | 204.51 | 0.841 | 222.43 | 197.70 | 0.889 |
| clickbench_q15/duckdb:vortex-file-compressed | 253.74 | 230.54 | 0.909 | 240.28 | 220.16 | 0.916 |
| clickbench_q16/duckdb:vortex-file-compressed | 521.80 | 519.18 | 0.995 | 491.35 | 500.29 | 1.018 |
| clickbench_q17/duckdb:vortex-file-compressed | 394.43 | 377.45 | 0.957 | 371.31 | 372.59 | 1.003 |
| clickbench_q18/duckdb:vortex-file-compressed | 952.57 | 856.88 | 0.900 | 876.07 | 837.19 | 0.956 |
| clickbench_q19/duckdb:vortex-file-compressed | 27.24 | 21.76 | 0.799 | 24.21 | 19.59 | 0.809 |
| clickbench_q20/duckdb:vortex-file-compressed | 539.10 | 505.27 | 0.937 | 508.47 | 494.46 | 0.972 |
| clickbench_q21/duckdb:vortex-file-compressed | 451.48 | 393.51 | 0.872 | 423.04 | 379.71 | 0.898 |
| clickbench_q22/duckdb:vortex-file-compressed | 723.99 | 703.84 | 0.972 | 684.07 | 654.91 | 0.957 |
| clickbench_q23/duckdb:vortex-file-compressed | 84.33 | 85.03 | 1.008 | 78.95 | 81.09 | 1.027 |
| clickbench_q24/duckdb:vortex-file-compressed | 22.60 | 19.01 | 0.841 | 20.59 | 17.55 | 0.852 |
| clickbench_q25/duckdb:vortex-file-compressed | 62.68 | 43.55 | 0.695 | 59.32 | 41.45 | 0.699 |
| clickbench_q26/duckdb:vortex-file-compressed | 19.78 | 17.03 | 0.861 | 17.90 | 16.14 | 0.902 |
| clickbench_q27/duckdb:vortex-file-compressed | 295.96 | 213.33 | 0.721 | 246.54 | 209.98 | 0.852 |
| clickbench_q28/duckdb:vortex-file-compressed | 3681.88 | 3253.83 | 0.884 | 3418.18 | 3134.56 | 0.917 |
| clickbench_q29/duckdb:vortex-file-compressed | 20.88 | 13.09 | 0.627 | 19.63 | 12.15 | 0.619 |
| clickbench_q30/duckdb:vortex-file-compressed | 194.69 | 186.46 | 0.958 | 177.93 | 171.31 | 0.963 |
| clickbench_q31/duckdb:vortex-file-compressed | 268.96 | 269.10 | 1.001 | 250.69 | 258.68 | 1.032 |
| clickbench_q32/duckdb:vortex-file-compressed | 1002.25 | 934.79 | 0.933 | 905.18 | 895.95 | 0.990 |
| clickbench_q33/duckdb:vortex-file-compressed | 1006.29 | 932.76 | 0.927 | 903.83 | 885.47 | 0.980 |
| clickbench_q34/duckdb:vortex-file-compressed | 1134.08 | 1048.13 | 0.924 | 1044.16 | 1020.28 | 0.977 |
| clickbench_q35/duckdb:vortex-file-compressed | 355.18 | 336.70 | 0.948 | 323.43 | 321.23 | 0.993 |
| clickbench_q36/duckdb:vortex-file-compressed | 16.01 | 14.72 | 0.919 | 14.00 | 14.08 | 1.006 |
| clickbench_q37/duckdb:vortex-file-compressed | 8.01 | 8.09 | 1.009 | 7.18 | 7.74 | 1.078 |
| clickbench_q38/duckdb:vortex-file-compressed | 11.84 | 10.96 | 0.926 | 11.15 | 10.47 | 0.940 |
| clickbench_q39/duckdb:vortex-file-compressed | 27.18 | 25.77 | 0.948 | 25.55 | 24.14 | 0.945 |
| clickbench_q40/duckdb:vortex-file-compressed | 7.22 | 7.87 | 1.089 | 6.96 | 7.13 | 1.025 |
| clickbench_q41/duckdb:vortex-file-compressed | 7.40 | 7.61 | 1.027 | 7.03 | 7.26 | 1.032 |
| clickbench_q42/duckdb:vortex-file-compressed | 7.52 | 7.39 | 0.984 | 6.80 | 7.24 | 1.064 |
| **geomean** | **88.68** | **77.96** | **0.879** | **81.29** | **72.81** | **0.896** |
| **sum** | 14232 | 12939 | 0.909 | 13154 | 12414 | 0.944 |

V1 per process (median of 3): wall 72.85s, cpu 815.71s (user 756.11 sys 59.60), instructions 5841.59G

V2 per process (median of 3): wall 66.85s, cpu 798.33s (user 717.81 sys 80.52), instructions 5748.51G

V2/V1 per process: instructions 0.984, cpu 0.979, wall 0.918

## HOT datafusion clickbench sf=- — 3 interleaved pairs x 5 iters, first iter of each process dropped (load avg 17.1-51.5) [hot-datafusion-clickbench---1790783710]
| query | V1 median ms | V2 median ms | V2/V1 median | V1 min ms | V2 min ms | V2/V1 min |
|---|---:|---:|---:|---:|---:|---:|
| clickbench_q00/datafusion:vortex-file-compressed | 0.67 | 0.62 | 0.923 | 0.51 | 0.54 | 1.058 |
| clickbench_q01/datafusion:vortex-file-compressed | 16.67 | 8.00 | 0.480 | 8.24 | 7.21 | 0.875 |
| clickbench_q02/datafusion:vortex-file-compressed | 42.31 | 25.35 | 0.599 | 25.89 | 24.40 | 0.943 |
| clickbench_q03/datafusion:vortex-file-compressed | 35.31 | 30.40 | 0.861 | 24.99 | 25.88 | 1.036 |
| clickbench_q04/datafusion:vortex-file-compressed | 319.55 | 270.67 | 0.847 | 244.54 | 214.44 | 0.877 |
| clickbench_q05/datafusion:vortex-file-compressed | 378.53 | 291.44 | 0.770 | 284.39 | 272.28 | 0.957 |
| clickbench_q06/datafusion:vortex-file-compressed | 0.82 | 0.79 | 0.968 | 0.75 | 0.75 | 1.007 |
| clickbench_q07/datafusion:vortex-file-compressed | 17.07 | 11.64 | 0.682 | 13.12 | 9.92 | 0.756 |
| clickbench_q08/datafusion:vortex-file-compressed | 472.12 | 350.76 | 0.743 | 357.91 | 341.46 | 0.954 |
| clickbench_q09/datafusion:vortex-file-compressed | 463.47 | 418.20 | 0.902 | 423.50 | 400.71 | 0.946 |
| clickbench_q10/datafusion:vortex-file-compressed | 64.06 | 53.21 | 0.831 | 52.59 | 49.51 | 0.941 |
| clickbench_q11/datafusion:vortex-file-compressed | 76.44 | 70.00 | 0.916 | 73.30 | 64.95 | 0.886 |
| clickbench_q12/datafusion:vortex-file-compressed | 256.59 | 240.45 | 0.937 | 242.98 | 232.02 | 0.955 |
| clickbench_q13/datafusion:vortex-file-compressed | 468.71 | 387.67 | 0.827 | 387.92 | 369.09 | 0.951 |
| clickbench_q14/datafusion:vortex-file-compressed | 278.87 | 263.03 | 0.943 | 255.46 | 246.44 | 0.965 |
| clickbench_q15/datafusion:vortex-file-compressed | 324.41 | 335.30 | 1.034 | 296.45 | 293.88 | 0.991 |
| clickbench_q16/datafusion:vortex-file-compressed | 753.90 | 699.01 | 0.927 | 664.60 | 645.41 | 0.971 |
| clickbench_q17/datafusion:vortex-file-compressed | 684.76 | 696.15 | 1.017 | 658.37 | 633.44 | 0.962 |
| clickbench_q18/datafusion:vortex-file-compressed | 1494.92 | 1418.23 | 0.949 | 1348.33 | 1311.65 | 0.973 |
| clickbench_q19/datafusion:vortex-file-compressed | 19.33 | 29.43 | 1.523 | 16.11 | 19.48 | 1.209 |
| clickbench_q20/datafusion:vortex-file-compressed | 596.52 | 389.36 | 0.653 | 497.75 | 323.61 | 0.650 |
| clickbench_q21/datafusion:vortex-file-compressed | 491.63 | 487.16 | 0.991 | 375.34 | 364.32 | 0.971 |
| clickbench_q22/datafusion:vortex-file-compressed | 588.76 | 532.06 | 0.904 | 427.32 | 462.13 | 1.081 |
| clickbench_q23/datafusion:vortex-file-compressed | 564.11 | 569.85 | 1.010 | 425.73 | 524.74 | 1.233 |
| clickbench_q24/datafusion:vortex-file-compressed | 38.68 | 30.48 | 0.788 | 32.23 | 27.19 | 0.843 |
| clickbench_q25/datafusion:vortex-file-compressed | 83.32 | 62.10 | 0.745 | 67.10 | 52.80 | 0.787 |
| clickbench_q26/datafusion:vortex-file-compressed | 36.35 | 29.74 | 0.818 | 28.75 | 27.15 | 0.944 |
| clickbench_q27/datafusion:vortex-file-compressed | 498.07 | 423.77 | 0.851 | 434.06 | 406.70 | 0.937 |
| clickbench_q28/datafusion:vortex-file-compressed | 2919.74 | 2403.65 | 0.823 | 2306.22 | 2342.60 | 1.016 |
| clickbench_q29/datafusion:vortex-file-compressed | 32.26 | 33.11 | 1.026 | 26.34 | 29.77 | 1.130 |
| clickbench_q30/datafusion:vortex-file-compressed | 254.74 | 231.80 | 0.910 | 212.67 | 208.83 | 0.982 |
| clickbench_q31/datafusion:vortex-file-compressed | 248.97 | 229.25 | 0.921 | 216.37 | 217.01 | 1.003 |
| clickbench_q32/datafusion:vortex-file-compressed | 1128.16 | 1016.44 | 0.901 | 998.85 | 993.60 | 0.995 |
| clickbench_q33/datafusion:vortex-file-compressed | 1551.99 | 1407.50 | 0.907 | 1376.69 | 1384.03 | 1.005 |
| clickbench_q34/datafusion:vortex-file-compressed | 1584.28 | 1489.32 | 0.940 | 1361.17 | 1392.44 | 1.023 |
| clickbench_q35/datafusion:vortex-file-compressed | 305.85 | 303.67 | 0.993 | 289.96 | 263.32 | 0.908 |
| clickbench_q36/datafusion:vortex-file-compressed | 30.30 | 28.76 | 0.949 | 23.50 | 24.85 | 1.058 |
| clickbench_q37/datafusion:vortex-file-compressed | 14.39 | 14.97 | 1.041 | 13.88 | 14.28 | 1.029 |
| clickbench_q38/datafusion:vortex-file-compressed | 8.08 | 9.17 | 1.134 | 7.66 | 8.30 | 1.083 |
| clickbench_q39/datafusion:vortex-file-compressed | 52.18 | 55.85 | 1.070 | 50.02 | 50.62 | 1.012 |
| clickbench_q40/datafusion:vortex-file-compressed | 5.92 | 6.27 | 1.058 | 5.34 | 5.28 | 0.988 |
| clickbench_q41/datafusion:vortex-file-compressed | 5.60 | 5.62 | 1.003 | 5.06 | 5.13 | 1.013 |
| clickbench_q42/datafusion:vortex-file-compressed | 6.09 | 6.09 | 0.999 | 5.60 | 5.37 | 0.960 |
| **geomean** | **109.65** | **98.16** | **0.895** | **91.24** | **88.32** | **0.968** |
| **sum** | 17215 | 15366 | 0.893 | 14568 | 14298 | 0.981 |

V1 per process (median of 3): wall 88.55s, cpu 900.26s (user 785.07 sys 115.19), instructions 4901.50G

V2 per process (median of 3): wall 77.80s, cpu 874.64s (user 744.08 sys 130.56), instructions 4822.47G

V2/V1 per process: instructions 0.984, cpu 0.972, wall 0.879
