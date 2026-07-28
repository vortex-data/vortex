# DuckDB

Vortex [extension](https://duckdb.org/docs/stable/core_extensions/vortex) is
available from DuckDB 1.4.2+ on Linux and macOS (amd64, arm64).

## Setup

```sql
INSTALL vortex;
LOAD vortex;
```

## Reading files

```sql
SELECT * FROM 'data.vortex';
```

Filters and projections are pushed down into Vortex, so only the columns and rows needed by the
query are read and decompressed.

```sql
SELECT name, age FROM 'data.vortex' WHERE age > 30;
```

## Writing files

Export data to Vortex using the `COPY` statement. The `FORMAT vortex` clause is required —
without it, DuckDB defaults to CSV.

```sql
COPY (SELECT * FROM my_table) TO 'output.vortex' (FORMAT vortex);
```

## Python

The DuckDB Python client works the same way:

```python
import duckdb

duckdb.sql("INSTALL vortex")
duckdb.sql("LOAD vortex")

result = duckdb.sql("SELECT * FROM 'data.vortex' WHERE age > 30")
result.show()
```
