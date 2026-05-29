# vortex-bench SQLStorm queries

[SQLStorm] is an LLM-generated SQL stress suite — ~62k queries across four
schemas, broad enough to exercise SQL surface that TPC-H and TPC-DS don't.
This directory holds a small, confirmed-working sample (125 queries per
schema, 500 total) that the nightly bench runs against vortex-bench's
existing TPC-H/TPC-DS data and two extra datasets we download for the
non-TPC schemas. Queries are pinned at SHA
[`b3bb0b96794a6afe9bb8f3ff2b243562b779c40d`][pinned-sqlstorm].

[SQLStorm]: https://github.com/SQL-Storm/SQLStorm
[pinned-sqlstorm]: https://github.com/SQL-Storm/SQLStorm/tree/b3bb0b96794a6afe9bb8f3ff2b243562b779c40d

## Layout

- `<origin>/<id>.sql` — 125 queries per origin, 4 origins, 500 total.
  `<id>` is the upstream SQLStorm query id (sparse, non-sequential).

| Origin | Source data | Upstream SQLStorm dir |
| --- | --- | --- |
| `tpch` | vortex-bench's existing TPC-H dataset at SF1 | `v1.0/tpch/` |
| `tpcds` | vortex-bench's existing TPC-DS dataset at SF1 | `v1.0/tpcds/` |
| `stackoverflow` | `stackoverflow_dba.tar.gz` (~1 GB) from `db.in.tum.de` | `v1.0/stackoverflow/` |
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
passing one is silently ignored (it is not an error and changes nothing):

| Origin | Fixed size |
| --- | --- |
| `tpch` | SF 1 (reuses vortex-bench's TPC-H dataset) |
| `tpcds` | SF 1 (reuses vortex-bench's TPC-DS dataset) |
| `stackoverflow` | the `dba` dataset, ~1 GB |
| `job` | the full IMDB/JOB snapshot (a single fixed real dataset) |

This mirrors upstream: SQLStorm has no uniform scale knob either. OLAPBench
(the canonical runner) selects size *per origin* — StackOverflow ships at
0 / 1 GB (`dba`) / 12 GB (`math`) / 222 GB, TPC-H/TPC-DS scale via their own
generators, and JOB is fixed. Query *validity* is scale-independent; only row
counts change with size. Scaling an origin up therefore means pointing it at a
larger upstream dataset (a different StackOverflow tarball, or a higher TPC
scale-factor directory), not passing a scale factor — that wiring does not
exist today.

## Refreshing the vendored set

Swaps happen by hand against the pinned SHA above: clone the SQLStorm
corpus at that SHA, pick candidates from `v1.0/<origin>/queries/`, and
verify each runs cleanly on both DuckDB and DataFusion against the source
data before vendoring. One gotcha: verify against the bench's own
DataFusion `SessionContext`, **not** `datafusion-cli` — the cli
decorrelates more subqueries than the harness can physically plan and
reports false-positive passes on queries the harness then can't actually
run.

## Running

The four origins are nightly-only matrix entries in
`.github/workflows/nightly-bench.yml`. Locally:

```
vx-bench run sqlstorm --opt origin=tpch       # tpch | tpcds | stackoverflow | job
```

TPC-H / TPC-DS reuse the existing vortex-bench datasets at SF1.
StackOverflow / JOB download and convert their upstream tarballs to
Parquet under `vortex-bench/data/sqlstorm/<origin>/parquet/` on first
run (idempotent via a `.success` marker).
