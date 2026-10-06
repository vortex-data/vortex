# JSONBench

[JSONBench](https://github.com/ClickHouse/JSONBench) runs five analytical queries over a dataset of
Bluesky events stored as JSON documents. Each source file holds one million documents; the
`scale-factor` option selects how many files to use (`--opt scale-factor=10` for 10 million).

## Data

The NDJSON source files are parsed into a single Variant column `data`. A sample of the documents
infers a shredding schema: every object path that is present in at least 10% of the documents and
holds one scalar type in at least 99% of them is shredded into a typed column. Other paths stay in
the residual Variant value. The inference does not know the queries.

The shredded Variant column is written to Parquet (zstd level 3, 122,880-row row groups) with the
Parquet `VARIANT` logical type. The Vortex files are converted from the Parquet files, so both
formats hold the same shredded Variant values.

## Queries

Queries extract fields with `variant_get(data, path, type)`, which returns `NULL` for missing
paths and values that do not cast to `type`. Over Vortex, DataFusion pushes `variant_get` calls
into the scan, which reads shredded paths from their typed columns.

| Query | Description |
| --- | --- |
| Q0 | Top event types |
| Q1 | Top event types with unique users per event type |
| Q2 | Events per hour of the day for posts, reposts and likes |
| Q3 | The three users with the earliest posts |
| Q4 | The three users with the longest posting activity span |
