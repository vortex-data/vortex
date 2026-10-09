<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

| engine suite | q | cfg | ms under profile | scan share of wall | blocked in scan (IO wait/handoff) | pread | open/close | IO dispatch | decode total | decode detail | filter/expr kernels | pruning | layout/plan/async/dispatch | convert to engine | engine on-CPU | spin CPU share | IO-pool threads |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| duckdb-tpch-10.0 | q6 | V1 | 44.0 | 98% | 57.4 | 9.1 | 0.0 | 1.6 | 10.5 | bitpacking 4.3, other encodings 3.2, FoR 1.4, array deserialise 1.2 | 10.5 | 0.7 | 9.4 | 0.2 | 0.6 | 3.2% | 9 |
| duckdb-tpch-10.0 | q6 | V2 | 30.3 | 96% | 37.1 | 15.9 | 0.0 | 1.9 | 16.7 | bitpacking 8.0, other encodings 5.3, FoR 2.4, array deserialise 0.7 | 16.9 | 0.3 | 9.9 | 0.3 | 1.0 | 19.8% | 35 |
| duckdb-tpch-10.0 | q1 | V1 | 181.7 | 37% | 18.0 | 4.3 | 0.0 | 0.6 | 3.3 | bitpacking 1.5, array deserialise 0.6, other encodings 0.6, FoR 0.3 | 3.1 | 0.2 | 3.4 | 4.2 | 62.7 | 0.5% | 7 |
| duckdb-tpch-10.0 | q1 | V2 | 204.7 | 39% | 18.6 | 3.7 | 0.0 | 1.2 | 3.4 | bitpacking 1.7, array deserialise 0.5, other encodings 0.4, FoR 0.3 | 2.8 | 0.2 | 4.6 | 4.3 | 61.3 | 5.9% | 41 |
| duckdb-tpch-10.0 | q18 | V1 | 232.0 | 48% | 33.5 | 5.2 | 0.0 | 0.3 | 6.8 | run-end 2.7, bitpacking 2.6, FoR 0.6, array deserialise 0.4 | 1.5 | 0.1 | 3.1 | 1.5 | 47.9 | 0.3% | 5 |
| duckdb-tpch-10.0 | q18 | V2 | 222.2 | 37% | 14.4 | 3.1 | 0.0 | 0.7 | 6.9 | bitpacking 4.8, FoR 1.2, array deserialise 0.3, other encodings 0.2 | 2.4 | 0.1 | 3.6 | 8.2 | 60.6 | 2.4% | 33 |
| duckdb-clickbench-- | q20 | V1 | 531.2 | 99% | 28.0 | 20.6 | 0.0 | 0.4 | 48.7 | FSST 47.6, other encodings 0.7, bitpacking 0.3, array deserialise 0.2 | 0.3 | 0.1 | 1.7 | 0.1 | 0.1 | 0.7% | 14 |
| duckdb-clickbench-- | q20 | V2 | 620.9 | 100% | 27.8 | 4.1 | 0.1 | 0.6 | 64.1 | FSST 62.3, other encodings 1.0, bitpacking 0.4, array deserialise 0.3 | 0.4 | 0.1 | 2.7 | 0.1 | 0.1 | 2.1% | 19 |
| duckdb-clickbench-- | q23 | V1 | 86.6 | 95% | 37.9 | 16.7 | 0.7 | 1.2 | 29.8 | FSST 25.2, other encodings 1.6, array deserialise 1.0, bitpacking 0.7 | 2.2 | 0.4 | 8.0 | 1.6 | 1.5 | 7.6% | 65 |
| duckdb-clickbench-- | q23 | V2 | 93.8 | 94% | 35.0 | 7.3 | 0.9 | 2.5 | 35.9 | FSST 30.5, other encodings 1.5, array deserialise 1.2, run-end 0.9 | 2.5 | 0.3 | 11.8 | 1.9 | 1.9 | 12.6% | 73 |
| duckdb-clickbench-- | q32 | V1 | 890.3 | 21% | 12.5 | 2.9 | 0.1 | 0.3 | 4.4 | run-end 2.0, bitpacking 1.6, FoR 0.3, array deserialise 0.3 | 0.2 | 0.1 | 2.4 | 0.9 | 76.4 | 0.5% | 28 |
| duckdb-clickbench-- | q32 | V2 | 929.0 | 15% | 7.3 | 2.7 | 0.1 | 0.4 | 4.2 | run-end 1.8, bitpacking 1.6, FoR 0.3, array deserialise 0.3 | 0.2 | 0.0 | 2.0 | 0.6 | 82.5 | 1.7% | 92 |
| duckdb-clickbench-- | q36 | V1 | 15.1 | 71% | 48.0 | 2.9 | 2.3 | 2.0 | 7.3 | FSST 5.2, other encodings 1.3, array deserialise 0.4, bitpacking 0.3 | 4.2 | 1.5 | 10.0 | 1.6 | 20.2 | 4.0% | 12 |
| duckdb-clickbench-- | q36 | V2 | 14.6 | 69% | 40.7 | 2.5 | 2.9 | 2.6 | 9.4 | FSST 6.5, other encodings 1.7, array deserialise 0.6, bitpacking 0.3 | 4.8 | 0.7 | 11.7 | 1.9 | 22.7 | 4.9% | 39 |
| datafusion-tpch-10.0 | q6 | V1 | 36.1 | 94% | 7.4 | 30.1 | 14.0 | 2.8 | 14.6 | bitpacking 5.8, other encodings 4.9, FoR 1.9, array deserialise 1.4 | 15.5 | 1.7 | 11.7 | 0.4 | 1.8 | 5.5% | 150 |
| datafusion-tpch-10.0 | q6 | V2 | 36.1 | 95% | 6.5 | 32.0 | 10.2 | 3.7 | 12.0 | bitpacking 5.0, other encodings 3.8, FoR 1.7, array deserialise 1.2 | 12.6 | 0.4 | 21.3 | 0.4 | 1.0 | 0.8% | 179 |
| datafusion-tpch-10.0 | q1 | V1 | 369.8 | 39% | 1.0 | 4.3 | 1.5 | 0.7 | 22.2 | other encodings 15.9, FSST 4.4, bitpacking 1.2, array deserialise 0.5 | 11.7 | 0.1 | 3.7 | 4.6 | 50.2 | 0.5% | 111 |
| datafusion-tpch-10.0 | q1 | V2 | 399.8 | 42% | 2.3 | 6.9 | 1.0 | 0.8 | 20.4 | other encodings 14.8, FSST 3.9, bitpacking 1.1, array deserialise 0.3 | 11.2 | 0.1 | 6.2 | 4.7 | 46.3 | 0.3% | 224 |
| datafusion-tpch-10.0 | q18 | V1 | 786.5 | 7% | 0.1 | 2.8 | 1.2 | 0.4 | 2.6 | bitpacking 1.2, run-end 0.9, array deserialise 0.1, FoR 0.1 | 0.8 | 0.0 | 2.2 | 1.5 | 88.4 | 0.1% | 105 |
| datafusion-tpch-10.0 | q18 | V2 | 765.1 | 8% | 0.5 | 2.1 | 1.4 | 0.3 | 2.4 | bitpacking 1.1, run-end 0.8, FoR 0.2, FSST 0.1 | 0.6 | 0.0 | 2.7 | 2.1 | 87.9 | 0.0% | 141 |
| datafusion-clickbench-- | q20 | V1 | 596.7 | 99% | 3.2 | 40.3 | 8.9 | 0.5 | 44.0 | FSST 42.9, other encodings 0.7, array deserialise 0.3, bitpacking 0.2 | 0.4 | 0.1 | 2.2 | 0.0 | 0.3 | 1.6% | 378 |
| datafusion-clickbench-- | q20 | V2 | 352.8 | 98% | 14.7 | 8.6 | 3.4 | 1.9 | 46.4 | FSST 42.7, other encodings 2.1, array deserialise 0.7, bitpacking 0.7 | 15.3 | 0.1 | 8.9 | 0.1 | 0.6 | 2.3% | 299 |
| datafusion-clickbench-- | q23 | V1 | 433.8 | 95% | 5.4 | 49.0 | 4.2 | 1.6 | 15.6 | FSST 6.8, array deserialise 3.9, other encodings 2.4, bitpacking 1.4 | 4.5 | 0.9 | 17.5 | 0.4 | 0.9 | 1.3% | 106 |
| datafusion-clickbench-- | q23 | V2 | 607.5 | 94% | 13.7 | 19.3 | 5.4 | 6.6 | 18.6 | FSST 9.5, other encodings 3.3, array deserialise 2.9, bitpacking 1.5 | 7.7 | 1.1 | 25.9 | 0.5 | 1.3 | 1.4% | 342 |
| datafusion-clickbench-- | q32 | V1 | 1186.7 | 8% | 0.5 | 1.7 | 0.4 | 0.3 | 3.1 | run-end 1.5, bitpacking 1.2, other encodings 0.2, FoR 0.1 | 0.4 | 0.1 | 1.6 | 0.1 | 91.6 | 0.7% | 140 |
| datafusion-clickbench-- | q32 | V2 | 1194.0 | 10% | 1.0 | 2.4 | 1.1 | 0.4 | 3.8 | run-end 2.0, bitpacking 1.3, FoR 0.2, array deserialise 0.2 | 0.5 | 0.0 | 2.0 | 0.1 | 88.6 | 1.1% | 195 |
| datafusion-clickbench-- | q36 | V1 | 27.0 | 16% | 4.9 | 4.0 | 3.6 | 1.6 | 11.1 | FSST 8.3, other encodings 1.5, array deserialise 0.6, bitpacking 0.5 | 3.7 | 2.2 | 8.2 | 0.1 | 60.4 | 1.2% | 62 |
| datafusion-clickbench-- | q36 | V2 | 28.2 | 17% | 3.9 | 3.5 | 1.9 | 2.4 | 12.7 | FSST 7.9, other encodings 2.1, array deserialise 1.9, dict 0.3 | 3.6 | 1.5 | 10.3 | 0.2 | 60.1 | 1.2% | 84 |
