# ClickHouse

[ClickHouse](https://clickhouse.com) reads and writes Vortex files natively, with the
[`Vortex` format](https://clickhouse.com/docs/reference/formats/Vortex), starting with
ClickHouse 26.11. It needs no extension or plugin. The format works in `clickhouse-server` and
in [`clickhouse-local`][clickhouse-local], which queries files without a server.

[clickhouse-local]: https://clickhouse.com/docs/concepts/features/tools-and-utilities/clickhouse-local

## Reading files

ClickHouse picks the `Vortex` format from the `.vortex` extension and infers the schema from the
file:

```sql
SELECT * FROM file('data.vortex');

DESCRIBE file('data.vortex');

-- globbing is also supported
SELECT count() FROM file('data/*.vortex');

-- name the format explicitly when the file has another extension
SELECT * FROM file('data.bin', Vortex);
```

Files in object storage are read with the
[`s3`](https://clickhouse.com/docs/reference/functions/table-functions/s3),
[`gcs`](https://clickhouse.com/docs/reference/functions/table-functions/gcs),
[`azureBlobStorage`](https://clickhouse.com/docs/reference/functions/table-functions/azureBlobStorage),
and [`url`](https://clickhouse.com/docs/reference/functions/table-functions/url) table functions:

```sql
SELECT * FROM s3('https://bucket.s3.amazonaws.com/data/*.vortex');
```

Only the columns used by the query are read. Parts of the `WHERE` condition are pushed down into
the Vortex scan, and the file is read and decoded in parallel.

## Writing files

```sql
-- convert a Parquet file to Vortex
SELECT * FROM file('data.parquet') INTO OUTFILE 'data.vortex';

-- write the result of a query
INSERT INTO FUNCTION file('output.vortex') SELECT * FROM my_table;
```

## Command line

```sh
clickhouse local --query "SELECT * FROM file('data.parquet') INTO OUTFILE 'data.vortex'"
clickhouse local --query "SELECT count() FROM file('data.vortex')"
```

The mapping between Vortex and ClickHouse data types, and the settings of the format, are in the
[ClickHouse documentation](https://clickhouse.com/docs/reference/formats/Vortex).
