# DuckDB

Vortex [extension](https://duckdb.org/docs/stable/core_extensions/vortex) is
available from DuckDB 1.4.2+ on Linux and macOS (amd64, arm64). Windows support
[is planned](https://github.com/vortex-data/vortex/issues/9569).

## Setup

```sql
INSTALL vortex; LOAD vortex;
```

## Reading files

```sql
SELECT * FROM 'data.vortex';
# this syntax supports arguments like hive_partitioning=true
SELECT * FROM read_vortex('data.vortex');
```

## Writing files

```sql
COPY (SELECT * FROM my_table) TO 'output.vortex';
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
