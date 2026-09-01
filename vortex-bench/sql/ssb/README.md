# Star Schema Benchmark

The [Star Schema Benchmark](https://www.cs.umb.edu/~poneil/StarSchemaB.PDF) (O'Neil, O'Neil &
Chen) is TPC-H redesigned as a textbook star schema: TPC-H's `lineitem` and `orders` are
denormalized into one wide `lineorder` fact table, joined against four dimensions — `customer`,
`supplier`, `part`, and `dwdate`.

That shape is what makes it worth running alongside TPC-H. Every query is a scan of the fact
table under dimension-derived filters of a known, deliberately varied selectivity, so the suite
isolates filter pushdown, zone-map pruning, and dimension-join throughput rather than mixing them
with TPC-H's subqueries and correlated predicates.

## Queries

The 13 queries are organized into four "flights", each holding the query shape fixed while
tightening the filters. The harness keys queries on a plain index, so the paper's numbering maps
onto `q1.sql` ... `q13.sql` in order:

| File | SSB | Flight |
| --- | --- | --- |
| `q1.sql` | Q1.1 | Flight 1 — single-dimension discount/quantity filter on `lineorder` x `dwdate` |
| `q2.sql` | Q1.2 | |
| `q3.sql` | Q1.3 | |
| `q4.sql` | Q2.1 | Flight 2 — `part` and `supplier` restriction, grouped by year and brand |
| `q5.sql` | Q2.2 | |
| `q6.sql` | Q2.3 | |
| `q7.sql` | Q3.1 | Flight 3 — `customer` x `supplier` geography join, narrowing region to nation to city |
| `q8.sql` | Q3.2 | |
| `q9.sql` | Q3.3 | |
| `q10.sql` | Q3.4 | |
| `q11.sql` | Q4.1 | Flight 4 — all four dimensions, profit aggregation |
| `q12.sql` | Q4.2 | |
| `q13.sql` | Q4.3 | |

Each file names its SSB query in a leading comment.

## Schema notes

The date dimension is registered as **`dwdate`**, not `date`: `date` is a reserved word in both
DataFusion's and DuckDB's parsers. This is the same rename the reference SSB load scripts apply,
for the same reason. Column names (`d_datekey`, `d_year`, ...) are unchanged.

## Data

SSB has no upstream Rust generator, so [`src/ssb/ssbgen/`](../../src/ssb/ssbgen/mod.rs) is an in-house one,
built on the `dbgen` stream primitives the
[`tpchgen`](https://github.com/clflushopt/tpchgen-rs) crate already ports;
[`src/ssb/datagen.rs`](../../src/ssb/datagen.rs) writes it straight to Parquet. No `.tbl`
intermediates, no `cmake`, no C compiler, no `duckdb` CLI.

Only the machinery is shared with TPC-H, not the data. SSB's dimension cardinalities differ
(`customer` is SF x 30k rather than SF x 150k, `supplier` SF x 2k rather than SF x 10k, `part` is
`200000 * floor(1 + log2(SF))`), `dwdate` has no TPC-H analogue, and generating the date dimension
advances the order streams, so even columns drawn from the same stream as TPC-H hold different
values.

Output is byte-for-byte identical to the reference C `dbgen`, including its quirks; the module
docs in `src/ssb/ssbgen/mod.rs` state which reference revision, which behaviours are reproduced
deliberately, and where the port diverges, and `src/ssb/ssbgen/tests.rs` owns the digests that
enforce it.

At SF 10 the source Parquet is ~1.7 GB, dominated by `lineorder`'s 59,986,217 rows, and generation
takes about a sixth as long as building and running the C generator did.

## CI variant

CI runs this suite at scale factor 10 from local NVMe as the `SSB SF=10 on NVME` PR comment,
comparing DataFusion and DuckDB over Parquet, Vortex, and vortex-compact files (plus a native
DuckDB baseline). Like Appian, it is part of the full SQL matrix rather than the quick PR one.

## Running locally

```bash
vx-bench run ssb --engine datafusion,duckdb --format parquet,vortex --opt scale-factor=10.0
```
