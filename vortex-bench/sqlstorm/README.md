# vortex-bench SQLStorm queries

[SQLStorm] is an LLM-generated SQL stress suite — ~62k queries across four
schemas, broad enough to exercise SQL surface that TPC-H and TPC-DS don't.
This directory holds a small, confirmed-working sample (125 queries per
schema, 500 total) that the nightly bench runs against TPC-H and TPC-DS data
generated at SF10 plus two larger datasets we download for the non-TPC
schemas. Queries are pinned at SHA
[`b3bb0b96794a6afe9bb8f3ff2b243562b779c40d`][pinned-sqlstorm].

[SQLStorm]: https://github.com/SQL-Storm/SQLStorm
[pinned-sqlstorm]: https://github.com/SQL-Storm/SQLStorm/tree/b3bb0b96794a6afe9bb8f3ff2b243562b779c40d

## Layout

- `<origin>/<id>.sql` — 125 queries per origin, 4 origins, 500 total.
  `<id>` is the upstream SQLStorm query id (sparse, non-sequential).

| Origin | Source data | Upstream SQLStorm dir |
| --- | --- | --- |
| `tpch` | TPC-H generated at SF10 (`data/tpch/10.0/`) | `v1.0/tpch/` |
| `tpcds` | TPC-DS generated at SF10 (`data/tpcds/10.0/`) | `v1.0/tpcds/` |
| `stackoverflow` | `stackoverflow_math.tar.gz` (~12 GB) from `db.in.tum.de` | `v1.0/stackoverflow/` |
| `job` | `imdb.tzst` from `db.in.tum.de` | `v1.0/job/` |

The benchmark runs strict — a query failure aborts the run rather than
silently dropping a row, so any regression that breaks a query in nightly
is loud. The vendored set was curated to be the intersection of queries
that pass DuckDB and DataFusion against the source data; that is why a
small, confirmed-working sample lives in-tree and the full ~62k SQLStorm
corpus does not.

## Data size (fixed scale)

**There is no SQLStorm scale factor.** Each origin runs at a single fixed
size, and `vx-bench run sqlstorm` does **not** read `--opt scale-factor` —
passing one is silently ignored (it is not an error and changes nothing). The
four origins are sized to sit within the same order of magnitude as JOB:

| Origin | Fixed size | ~Rows (all tables) | ~Parquet |
| --- | --- | --- | --- |
| `stackoverflow` | the `math` tier, ~12 GB | 40 M | 6.1 GB |
| `job` | the full IMDB/JOB snapshot (fixed real dataset) | 74 M | 1.7 GB |
| `tpch` | SF 10 | 87 M | 3.5 GB |
| `tpcds` | SF 10 | 192 M | 3.9 GB |

This mirrors upstream: SQLStorm has no uniform scale knob either. OLAPBench
(the canonical runner) selects size *per origin* — StackOverflow ships at
0 / 1 GB (`dba`) / 12 GB (`math`) / 222 GB, TPC-H/TPC-DS scale via their own
generators, and JOB is fixed. Query *validity* is scale-independent; only row
counts change with size. The fixed points above are set in code — the TPC
scale by `SQLSTORM_TPC_SCALE_FACTOR` (`sqlstorm_benchmark.rs`) and the
StackOverflow tier by the `STACKOVERFLOW` recipe's tarball URL (`data.rs`) —
so changing them means editing those consts (and re-curating, since the
vendored queries are selected to stay short at the configured scale), not
passing a runtime scale factor.

## Refreshing the vendored set

Swaps happen by hand against the pinned SHA above: clone the SQLStorm
corpus at that SHA, pick candidates from `v1.0/<origin>/queries/`, and
verify each runs cleanly on both DuckDB and DataFusion **at the configured
scale** (SF10 / `math`) before vendoring. Candidates must also stay short
— the vendored set is curated to keep each query under ~5 s/engine at scale
so the nightly stays bounded; drop anything slower and refill. One gotcha:
verify against the bench's own DataFusion `SessionContext`, **not**
`datafusion-cli` — the cli decorrelates more subqueries than the harness can
physically plan and reports false-positive passes on queries the harness then
can't actually run.

## Running

The four origins are nightly-only matrix entries in
`.github/workflows/nightly-bench.yml`. Locally:

```
vx-bench run sqlstorm --opt origin=tpch       # tpch | tpcds | stackoverflow | job
```

TPC-H / TPC-DS generate their own SF10 datasets under
`vortex-bench/data/tpch/10.0/` and `vortex-bench/data/tpcds/10.0/` (no longer
shared with the standalone SF1 benchmarks). StackOverflow / JOB download and
convert their upstream tarballs to Parquet under
`vortex-bench/data/sqlstorm/<origin>/parquet/` on first run (idempotent via a
`.success` marker). The StackOverflow `math` tarball is ~12 GB and needs
~30 GB of scratch to extract and load.
