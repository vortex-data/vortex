# DuckDB

Vortex is a [core extension](https://duckdb.org/docs/stable/core_extensions/vortex) shipped with
DuckDB, available from DuckDB 1.4.2+ on Linux and macOS (amd64, arm64).

## Setup

```sql
INSTALL vortex;
LOAD vortex;
```

## Reading Vortex Files

Use the `read_vortex` function to query a Vortex file:

```sql
SELECT * FROM read_vortex('data.vortex');
```

Filters and projections are pushed down into Vortex, so only the columns and rows needed by the
query are read and decompressed.

```sql
SELECT name, age
FROM read_vortex('data.vortex')
WHERE age > 30;
```

:::{note}
Direct file path syntax (`SELECT * FROM 'data.vortex'`) is coming in an upcoming DuckDB release.
:::

## Writing Vortex Files

Export data to Vortex using the `COPY` statement. The `FORMAT vortex` clause is required —
without it, DuckDB defaults to CSV.

```sql
COPY (SELECT * FROM my_table) TO 'output.vortex' (FORMAT vortex);
```

## Python

The DuckDB Python client works with `read_vortex` the same way:

```python
import duckdb

duckdb.sql("INSTALL vortex")
duckdb.sql("LOAD vortex")

result = duckdb.sql("SELECT * FROM read_vortex('data.vortex') WHERE age > 30")
result.show()
```
