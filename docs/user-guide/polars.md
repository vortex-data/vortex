# Polars

:::{warning}
The Polars integration is experimental. Polars' expression API is unstable and not all pushdown
expressions are currently supported.

If you run into any issues or are looking for more features related to this integration, please [file an issue](https://github.com/vortex-data/vortex/issues).
:::

Vortex integrates with Polars via {meth}`.VortexFile.to_polars`, which returns a
{class}`polars.LazyFrame` with column pruning and predicate pushdown.

```{doctest} pycon
>>> import vortex as vx
>>> import pyarrow.parquet as pq
>>>
>>> vx.io.write(pq.read_table("_static/example.parquet"), 'example.vortex')
>>>
>>> lf = vx.open('example.vortex').to_polars()
>>> lf = lf.select('tip_amount', 'fare_amount')
>>> lf = lf.head(3)
>>> lf.collect()
shape: (3, 2)
┌────────────┬─────────────┐
│ tip_amount ┆ fare_amount │
│ ---        ┆ ---         │
│ f64        ┆ f64         │
╞════════════╪═════════════╡
│ 0.0        ┆ 61.8        │
│ 5.1        ┆ 20.5        │
│ 16.54      ┆ 70.0        │
└────────────┴─────────────┘
```
