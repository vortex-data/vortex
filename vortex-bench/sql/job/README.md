# Join Order Benchmark (JOB)

The 113 join-heavy queries from Leis et al.,
[*How Good Are Query Optimizers, Really?*](https://www.vldb.org/pvldb/vol9/p204-leis.pdf)
(VLDB 2015), over the 21-table IMDB dataset. Each query joins between 4 and 17 tables with
selective string predicates, so the suite stresses join ordering, filter pushdown, and string
scans rather than raw scan throughput.

The queries live in this directory and are copied byte-for-byte from DuckDB's
[`benchmark/imdb_plan_cost/queries`](https://github.com/duckdb/duckdb/tree/main/benchmark/imdb_plan_cost/queries),
which its `benchmark/imdb` suite runs. The harness downloads the IMDB tables from the Parquet
files that DuckDB's suite loads (about 1.8 GB) and re-encodes them with zstd level 3. See
[`src/job`](../../src/job) for details.

## Query numbering

Benchmark query indices are integers, so the upstream query names are numbered in order:

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| q1 = `1a` | q2 = `1b` | q3 = `1c` | q4 = `1d` | q5 = `2a` | q6 = `2b` | q7 = `2c` | q8 = `2d` |
| q9 = `3a` | q10 = `3b` | q11 = `3c` | q12 = `4a` | q13 = `4b` | q14 = `4c` | q15 = `5a` | q16 = `5b` |
| q17 = `5c` | q18 = `6a` | q19 = `6b` | q20 = `6c` | q21 = `6d` | q22 = `6e` | q23 = `6f` | q24 = `7a` |
| q25 = `7b` | q26 = `7c` | q27 = `8a` | q28 = `8b` | q29 = `8c` | q30 = `8d` | q31 = `9a` | q32 = `9b` |
| q33 = `9c` | q34 = `9d` | q35 = `10a` | q36 = `10b` | q37 = `10c` | q38 = `11a` | q39 = `11b` | q40 = `11c` |
| q41 = `11d` | q42 = `12a` | q43 = `12b` | q44 = `12c` | q45 = `13a` | q46 = `13b` | q47 = `13c` | q48 = `13d` |
| q49 = `14a` | q50 = `14b` | q51 = `14c` | q52 = `15a` | q53 = `15b` | q54 = `15c` | q55 = `15d` | q56 = `16a` |
| q57 = `16b` | q58 = `16c` | q59 = `16d` | q60 = `17a` | q61 = `17b` | q62 = `17c` | q63 = `17d` | q64 = `17e` |
| q65 = `17f` | q66 = `18a` | q67 = `18b` | q68 = `18c` | q69 = `19a` | q70 = `19b` | q71 = `19c` | q72 = `19d` |
| q73 = `20a` | q74 = `20b` | q75 = `20c` | q76 = `21a` | q77 = `21b` | q78 = `21c` | q79 = `22a` | q80 = `22b` |
| q81 = `22c` | q82 = `22d` | q83 = `23a` | q84 = `23b` | q85 = `23c` | q86 = `24a` | q87 = `24b` | q88 = `25a` |
| q89 = `25b` | q90 = `25c` | q91 = `26a` | q92 = `26b` | q93 = `26c` | q94 = `27a` | q95 = `27b` | q96 = `27c` |
| q97 = `28a` | q98 = `28b` | q99 = `28c` | q100 = `29a` | q101 = `29b` | q102 = `29c` | q103 = `30a` | q104 = `30b` |
| q105 = `30c` | q106 = `31a` | q107 = `31b` | q108 = `31c` | q109 = `32a` | q110 = `32b` | q111 = `33a` | q112 = `33b` |
| q113 = `33c` |

## CI variant

CI runs this suite from local NVMe as the `JOB on NVME` PR comment in the `pr-full` and `pr-all`
presets (`action/bench-sql-extended` and `action/bench-all`), comparing DuckDB over Parquet and Vortex files against a native
DuckDB baseline. DataFusion is left out because its join orders make a few queries (`17b`-`17f`)
take tens of seconds each.

## Running locally

```bash
vx-bench run job --engine duckdb --format parquet,vortex,duckdb
```
