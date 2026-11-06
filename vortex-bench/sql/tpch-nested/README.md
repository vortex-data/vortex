# Nested TPC-H benchmark

TPC-H data and queries with a nested schema, for measuring how nested column pruning and
array unnesting behave in each engine and format. The shape follows Apache DataFusion
Comet's [nested benchmark](https://github.com/apache/datafusion-comet/tree/main/benchmarks/nested):
each table keeps its join keys (the `*key` columns) and one partition column (`p_brand`,
`c_mktsegment`, `o_orderdate`, `l_shipdate`) at the top level, and every other column moves
into a one-element `<table>_data` list of structs.

```text
l_orderkey     BIGINT
l_partkey      BIGINT
l_suppkey      BIGINT
l_shipdate     DATE
lineitem_data  LIST<STRUCT<
    l_linenumber: INT,
    l_quantity: DECIMAL(15,2),
    l_extendedprice: DECIMAL(15,2),
    l_discount: DECIMAL(15,2),
    l_tax: DECIMAL(15,2),
    l_returnflag: STRING,
    l_linestatus: STRING,
    l_commitdate: DATE,
    l_receiptdate: DATE,
    l_shipinstruct: STRING,
    l_shipmode: STRING,
    l_comment: STRING>>
```

Every list holds exactly one element, so unnesting it never changes the row count and the
queries return the flat TPC-H answers; the suite validates row counts against the flat suite's.

The queries in `list/q1.sql` ... `list/q22.sql` are the [flat suite's](../tpch) queries with
Comet's rewrite applied: a table whose nested columns a query reads is replaced by a derived
table that unnests its list, and the nested columns are read through it.

```sql
from
    (select l_orderkey, l_partkey, l_suppkey, l_shipdate, unnest(lineitem_data) as lineitem_data from lineitem) as lineitem
where
    lineitem_data.l_quantity < 24
```

Tables whose nested columns a query never reads stay as they are, as in Comet. Comet's queries
are written for Spark SQL (`explode`, `year()`); these use the same SQL as the flat suite so
that the two differ only in shape.

## Data

The flat suite's generator ([`tpchgen`](../../src/tpch/tpchgen.rs)) produces the tables and
nests each batch before writing ([`nest.rs`](../../src/tpch/nest.rs)), so the data has the
flat suite's types, zstd Parquet with 1 Mi-row row groups and the same file splits. Comet's
Spark script instead casts money to `DECIMAL(12,2)`, writes 512 MiB row groups and Hive
partitions the four largest tables by their partition column; none of that is reproduced here.
Vortex files are written with the list layout, so the struct fields of each `<table>_data`
list are stored as separate columns. The V2 scan cannot read the list layout yet, so CI runs
this suite on the V1 scan, like the other list-layout suites (statpopgen, PolarSignals), while
the flat suite runs on V2.

The `--opt shape=<shape>` option selects the nesting. `list` (the default and only shape so
far) is Comet's layout; the data directory and query directory are keyed by it.

## Running locally

```bash
vx-bench run tpch-nested --engine datafusion,duckdb --format parquet,vortex --opt scale-factor=1.0
```

TPC-H is a trademark of the Transaction Processing Performance Council. The queries are
derived from TPC-H under the TPC Fair Use Policy, and results from them are not TPC-H results.
