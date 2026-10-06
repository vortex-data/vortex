# DuckDB

Vortex [extension](https://duckdb.org/docs/stable/core_extensions/vortex) is
available on Linux and macOS from DuckDB 1.4.2 and on Windows from DuckDB 2.0.

## Setup

```sql
INSTALL vortex; LOAD vortex;
```

## Reading files

```sql
SELECT * FROM 'data.vortex';

# alternative syntax
SELECT * FROM read_vortex('data.vortex');

# globbing is also supported
SELECT * FROM 'foo/*.vortex';

# you can pass lists of files
SELECT * FROM read_vortex(['1.vortex', '2.vortex']);

# hive_partitioning is supported.
# assuming folder structure data/year=2000/1.vortex, data/year=2001/2.vortex
SELECT * FROM 'data/*/*.vortex';

# alternative syntax supports hive_partitioning, filename, and allow_empty
# parameters
SELECT * FROM read_vortex(['1.vortex.csv', '2.vortex'], filename = true);
```

## Writing files

```sql
COPY (SELECT * FROM my_table) TO 'output.vortex';
# you can also get file statistics
COPY (SELECT * FROM my_table) TO 'output.vortex' (FORMAT vortex, RETURN_STATS)
```

Make sure to call `LOAD vortex` before writing any files. Duckdb writes CSV
files by default, so if Vortex extension wasn't loaded prior to `COPY`,
`output.vortex` will be a CSV file. If you want to prevent this from happening,
you can use `COPY ... TO 'output.vortex' (FORMAT vortex)` syntax instead which
will fail if Vortex is not loaded.

## Python client

```python
import duckdb

duckdb.sql("INSTALL vortex")
duckdb.sql("LOAD vortex")

result = duckdb.sql("SELECT * FROM 'data.vortex' WHERE age > 30")
result.show()
```

## Object storage secrets

Vortex does not use DuckDB's secret storage. It uses environment variables via
[`object_store`](https://docs.rs/object_store/latest/object_store) instead.

If you want to read or write from an [S3
bucket](https://docs.rs/object_store/latest/object_store/aws/struct.AmazonS3Builder.html),
set the following variables:

```sh
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="your_bucket_region"
```

For [Google Cloud Storage](https://docs.rs/object_store/latest/object_store/gcp/struct.GoogleCloudStorageBuilder.html),
use a key file variable:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="my service account JSON key file"
```

For [Azure Blob Storage](https://docs.rs/object_store/latest/object_store/azure/struct.MicrosoftAzureBuilder.html),
use

```sh
export AZURE_STORAGE_ACCOUNT="my account"
export AZURE_STORAGE_KEY="my key"
```
