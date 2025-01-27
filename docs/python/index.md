# Vortex Python

```{toctree}
---
hidden:
---

API <api/index>
```

The Vortex Python bindings provide a Python interface to the Vortex Rust library.
See the parent documentation for details on the Vortex project itself or other language bindings.

## Installation

````{tab} pip
```bash
pip install vortex-array
```
````

````{tab} uv
```bash
uv add vortex-array
```
````

## Guide

Construct a Vortex array from lists of simple Python values:

```{doctest} pycon
>>> import vortex
>>> vtx = vortex.array([1, 2, 3, 4])
>>> vtx.dtype
int(64, False)
```

Python's {obj}`None` represents a missing or null value and changes the dtype of the array from
non-nullable 64-bit integers to nullable 64-bit integers:

```{doctest} pycon
>>> vtx = vortex.array([1, 2, None, 4])
>>> vtx.dtype
int(64, True)
```

A list of {class}`dict` is converted to an array of structures. Missing values may appear at any
level:

```{doctest} pycon

>>> vtx = vortex.array([
...   {'name': 'Joseph', 'age': 25},
...   {'name': None, 'age': 31},
...   {'name': 'Angela', 'age': None},
...   {'name': 'Mikhail', 'age': 57},
...   {'name': None, 'age': None},
...   None,
... ])
>>> vtx.dtype
struct({"age": int(64, True), "name": utf8(True)}, True)
```

{meth}`.Array.to_pylist` converts a Vortex array into a list of Python values.

```{doctest} pycon
>>> vtx.to_pylist()
[{'age': 25, 'name': 'Joseph'}, {'age': 31, 'name': None}, {'age': None, 'name': 'Angela'}, {'age': 57, 'name': 'Mikhail'}, {'age': None, 'name': None}, {'age': None, 'name': None}]
```

### Arrow

The {func}`~vortex.encoding.array` function constructs a Vortex array from an Arrow one without any
copies:

```{doctest} pycon
>>> import pyarrow as pa
>>> arrow = pa.array([1, 2, None, 3])
>>> arrow.type
DataType(int64)
>>> vtx = vortex.array(arrow)
>>> vtx.dtype
int(64, True)
```

{meth}`.Array.to_arrow_array` converts back to an Arrow array:

```{doctest} pycon
>>> vtx.to_arrow_array()
<pyarrow.lib.Int64Array object at ...>
[
1,
2,
null,
3
]
```

If you have a struct array, use {meth}`.Array.to_arrow_table` to construct an Arrow table:

```{doctest} pycon
>>> struct_vtx = vortex.array([
... {'name': 'Joseph', 'age': 25},
... {'name': 'Narendra', 'age': 31},
... {'name': 'Angela', 'age': 33},
... {'name': 'Mikhail', 'age': 57},
... ])
>>> struct_vtx.to_arrow_table()
pyarrow.Table
age: int64
name: string_view
----
age: [[25,31,33,57]]
name: [["Joseph","Narendra","Angela","Mikhail"]]
```

### Pandas

{meth}`.Array.to_pandas_df` converts a Vortex array into a Pandas DataFrame:

```{doctest} pycon
>>> df = struct_vtx.to_pandas_df()
>>> df
      age      name
   0   25    Joseph
   1   31  Narendra
   2   33    Angela
   3   57   Mikhail
```

{func}`~vortex.encoding.array` converts from a Pandas DataFrame into a Vortex array:

```pycon
>>> vortex.array(df).to_arrow_table()
pyarrow.Table
age: int64
name: string_view
----
age: [[25,31,33,57]]
name: [["Joseph","Narendra","Angela","Mikhail"]]
```

### Query Engines

{class}`~vortex.dataset.VortexDataset` implements the {class}`pyarrow.dataset.Dataset` API which
enables many Python-based query engines to pushdown row filters and column projections on Vortex
files. All the query engine examples use the same Vortex file:

```pycon
>>> import vortex
>>> import pyarrow.parquet as pq
>>> vtx = vortex.array(pq.read_table("_static/example.parquet"))
>>> vortex.io.write_path(vtx, 'example.vortex')
>>> ds = vortex.dataset.from_path(
>>> ...     'example.vortex'
>>> ... )
```

#### Polars

```pycon
>>> import polars as pl
>>> lf = pl.scan_pyarrow_dataset(ds)
>>> lf = lf.select('tip_amount', 'fare_amount')
>>> lf = lf.head(3)
>>> lf.collect()
shape: (3, 2)
┌────────────┬─────────────┐
│ tip_amount ┆ fare_amount │
│ --- ┆ --- │
│ f64 ┆ f64 │
╞════════════╪═════════════╡
│ 0.0 ┆ 61.8 │
│ 5.1 ┆ 20.5 │
│ 16.54 ┆ 70.0 │
└────────────┴─────────────┘
```

#### DuckDB

```pycon
>>> import duckdb
>>> duckdb.sql('select ds.tip_amount, ds.fare_amount from ds limit 3').show()
┌────────────┬─────────────┐
│ tip_amount │ fare_amount │
│ double │ double │
├────────────┼─────────────┤
│ 0.0 │ 61.8 │
│ 5.1 │ 20.5 │
│ 16.54 │ 70.0 │
└────────────┴─────────────┘
<BLANKLINE>
```