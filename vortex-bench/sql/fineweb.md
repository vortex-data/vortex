# FineWeb benchmark

A string-heavy benchmark over a sample shard of the HuggingFace
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) web-crawl dataset
(URLs, dates, and full page text). The dataset exercises dictionary and FSST string
compression heavily, and the hand-crafted queries in [`fineweb.sql`](./fineweb.sql)
(numbered from Q0 in file order) focus on string predicates: equality filters, `LIKE`
prefix and containment patterns, and aggregations over string columns.

The harness lives in [`src/fineweb`](../src/fineweb).

## CI variants

CI runs this suite from local NVMe (`FineWeb NVMe`) and from S3 (`FineWeb S3`), comparing
DataFusion and DuckDB over Parquet, Vortex, and vortex-compact files.

## Running locally

```bash
vx-bench run fineweb --engine datafusion,duckdb --format parquet,vortex
```
