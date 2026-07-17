# Public BI benchmark

Real-world BI datasets from the
[Public BI benchmark](https://github.com/cwida/public_bi_benchmark) — anonymized Tableau
workbooks with messy, wide, string-heavy tables. A subset of the datasets (Arade, Bimbo,
CMSprovider, Euro2016, Food, HashTags) also feeds the
[Compression benchmark](../../benchmarks/compress-bench/README.md).

The datasets, schemas, and queries are defined in [`src/public_bi.rs`](../src/public_bi.rs).

## Running locally

```bash
vx-bench run public-bi
```
