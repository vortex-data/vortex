# PyArrow

Vortex integrates with PyArrow for reading and writing Vortex files using Arrow tables and record
batch readers.

## Writing Vortex Files

Use {func}`~vortex.io.write` to convert a Parquet file to Vortex. The write function accepts
anything that implements `IntoArrayIterator`, including {class}`pyarrow.Table` and
{class}`pyarrow.RecordBatchReader`:

```{doctest} pycon
>>> import pyarrow.parquet as pq
>>> import vortex as vx
>>>
>>> table = pq.read_table("_static/example.parquet")
>>> vx.io.write(table, 'example.vortex')
```

## Reading Vortex Files

Use {func}`~vortex.open` to lazily open a Vortex file:

```{doctest} pycon
>>> f = vx.open('example.vortex')
>>> len(f)
1000
```

### As an Arrow Table

{meth}`.VortexFile.to_table` collects the scan into a {class}`pyarrow.Table`:

```{doctest} pycon
>>> table = f.to_table()
>>> table.num_rows
1000
```

{meth}`.VortexFile.to_arrow` returns a streaming {class}`pyarrow.RecordBatchReader`.

### Column Projection

Read only the columns you need:

```{doctest} pycon
>>> table = f.to_table(columns=['tip_amount', 'fare_amount'])
>>> table.column_names
['tip_amount', 'fare_amount']
```

### Filters

Vortex expressions are the stable pushdown API. PyVortex plans them against the file schema before
the scan runs:

```{doctest} pycon
>>> table = f.to_table(columns=['tip_amount'], filter=vx.col('tip_amount') > 10)
>>> table.num_rows > 0
True
```

PyArrow compute expressions are accepted as compatibility input. PyVortex converts them through
Substrait, then runs the same Vortex planner:

```{doctest} pycon
>>> import pyarrow.compute as pc
>>> table = f.to_table(columns=['tip_amount'], filter=pc.field('tip_amount') > 10)
>>> table.num_rows > 0
True
```

Use `filter_policy="pushdown"` to raise when a PyArrow expression cannot be pushed into Vortex. Use
`filter_policy="fallback"` to read the rows and apply the PyArrow filter after the scan.

### Streaming Record Batches

Iterate over record batches for streaming processing:

```{doctest} pycon
>>> total = 0
>>> for batch in f.to_arrow():
...     total += batch.num_rows
>>> total
1000
```

## Arrow Interop

The {func}`~vortex.array` function constructs a Vortex array from an Arrow array without copies:

```{doctest} pycon
>>> import pyarrow as pa
>>> arrow = pa.array([1, 2, None, 3])
>>> arr = vx.array(arrow)
>>> arr.dtype
int(64, nullable=True)
```

{meth}`.Array.to_arrow_array` converts back:

```{doctest} pycon
>>> arr.to_arrow_array()
<pyarrow.lib.Int64Array object at ...>
[
1,
2,
null,
3
]
```

Struct arrays convert to Arrow tables with {meth}`.Array.to_arrow_table`:

```{doctest} pycon
>>> struct_arr = vx.array([
... {'name': 'Joseph', 'age': 25},
... {'name': 'Narendra', 'age': 31},
... {'name': 'Angela', 'age': 33},
... {'name': 'Mikhail', 'age': 57},
... ])
>>> struct_arr.to_arrow_table()
pyarrow.Table
age: int64
name: string
----
age: [[25,31,33,57]]
name: [["Joseph","Narendra","Angela","Mikhail"]]
```
