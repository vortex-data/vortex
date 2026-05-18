# Vortex Python

:::{warning}
The Python API surface is not yet complete and is subject to change. Many operations available in
the Rust API are not yet exposed. See the {doc}`/api/python/index` for the full reference.
:::

## Installation

````{tab} pip
```bash
pip install vortex-data
```
````

````{tab} uv
```bash
uv add vortex-data
```
````

## Creating Arrays

{func}`~vortex.array` constructs a Vortex array from Python values:

```{doctest} pycon
>>> import vortex as vx
>>> arr = vx.array([1, 2, 3, 4])
>>> arr.dtype
int(64, nullable=False)
>>> len(arr)
4
```

Python's {obj}`None` represents a missing value and makes the dtype nullable:

```{doctest} pycon
>>> arr = vx.array([1, 2, None, 4])
>>> arr.dtype
int(64, nullable=True)
```

A list of {class}`dict` produces a struct array. Missing values may appear at any level:

```{doctest} pycon
>>> arr = vx.array([
...   {'name': 'Joseph', 'age': 25},
...   {'name': None, 'age': 31},
...   None,
... ])
>>> arr.dtype
struct({"age": int(64, nullable=True), "name": utf8(nullable=True)}, nullable=True)
```

{func}`~vortex.array` also accepts {class}`pyarrow.Array`, {class}`pyarrow.Table`,
{class}`pandas.DataFrame`, and {class}`range` objects.

## DTypes

DType factory functions are available at the top level of the `vortex` module:

```{doctest} pycon
>>> vx.int_(32)
int(32, nullable=False)
>>> vx.utf8(nullable=True)
utf8(nullable=True)
>>> vx.list_(vx.float_(64))
list(float(64, nullable=False), nullable=False)
>>> vx.struct({'x': vx.int_(32), 'y': vx.int_(32)})
struct({"x": int(32, nullable=False), "y": int(32, nullable=False)}, nullable=False)
```

Available types: {func}`~vortex.null`, {func}`~vortex.bool_`,
{func}`~vortex.int_`, {func}`~vortex.uint`, {func}`~vortex.float_`,
{func}`~vortex.decimal`, {func}`~vortex.utf8`, {func}`~vortex.binary`,
{func}`~vortex.struct`, {func}`~vortex.list_`,
{func}`~vortex.fixed_size_list`, {func}`~vortex.date`,
{func}`~vortex.time`, {func}`~vortex.timestamp`.

## Array Operations

### Element Access

```{doctest} pycon
>>> arr = vx.array([10, 20, 30, 40, 50])
>>> arr.scalar_at(0).as_py()
10
>>> arr.to_arrow_array().to_pylist()
[10, 20, 30, 40, 50]
```

### Slicing and Selection

```{doctest} pycon
>>> arr.slice(1, 3).to_arrow_array().to_pylist()
[20, 30]
>>> indices = vx.array([0, 2, 4])
>>> arr.take(indices).to_arrow_array().to_pylist()
[10, 30, 50]
```

### Filtering

```{doctest} pycon
>>> mask = vx.array([True, False, True, False, True])
>>> arr.filter(mask).to_arrow_array().to_pylist()
[10, 30, 50]
```

### Comparisons

```{doctest} pycon
>>> other = vx.array([10, 25, 25, 45, 50])
>>> (arr > other).to_arrow_array().to_pylist()
[False, False, True, False, False]
```

## Expressions

The `vortex.expr` module provides expressions for filtering and projecting. These
are primarily used with {meth}`.VortexFile.scan` and {meth}`.VortexFile.to_arrow` but can also be
applied directly:

```{doctest} pycon
>>> import vortex.expr as ve
>>>
>>> arr = vx.array([
...     {'name': 'Alice', 'age': 30},
...     {'name': 'Bob', 'age': 25},
...     {'name': 'Carol', 'age': 35},
... ])
>>> expr = ve.column('age') > 28
>>> arr.apply(expr).to_arrow_array().to_pylist()
[True, False, True]
```

## VortexFile

{func}`~vortex.open` lazily opens a Vortex file for reading:

```{doctest} pycon
>>> import pyarrow.parquet as pq
>>>
>>> vx.io.write(pq.read_table("_static/example.parquet"), 'example.vortex')
>>>
>>> f = vx.open('example.vortex')
>>> len(f)
1000
```

Use {meth}`.VortexFile.scan` to read data with optional projection, filtering, and limit:

```{doctest} pycon
>>> result = f.scan(['tip_amount'], limit=3).read_all()
>>> result.to_arrow_array()
<pyarrow.lib.StructArray object at ...>
-- is_valid: all not null
-- child 0 type: double
  [
    0,
    5.1,
    16.54
  ]
```

## ArrayIterator

{class}`.ArrayIterator` streams batches of arrays from a scan or other source. It supports
iteration, collecting into a single array, and conversion to Arrow.

{meth}`.ArrayIterator.read_all` collects all batches into a single in-memory {class}`.Array`:

```{doctest} pycon
>>> arr = f.scan(['tip_amount'], limit=5).read_all()
>>> len(arr)
5
```

{meth}`.ArrayIterator.to_arrow` converts to a {class}`pyarrow.RecordBatchReader` for use with
Arrow-based tools:

```{doctest} pycon
>>> reader = f.scan(['tip_amount']).to_arrow()
>>> reader.schema
tip_amount: double
>>> table = reader.read_all()
>>> len(table)
1000
```

## Threading Model

Vortex uses a shared runtime behind the Python API. When no background workers are configured, the
Python thread that is reading from a scan also polls the Vortex work needed to produce each batch.
This means multiple Python threads can make progress independently as long as each thread owns the
reader it is consuming:

```python
from concurrent.futures import ThreadPoolExecutor

import pyarrow.compute as pc
import vortex as vx


def sum_column(path: str, column: str) -> int | float:
    reader = vx.open(path).to_arrow([column], batch_size=64_000)
    total = 0

    for batch in reader:
        value = pc.sum(batch.column(column)).as_py()
        if value is not None:
            total += value

    return total


columns = ["tip_amount", "fare_amount", "total_amount"]
with ThreadPoolExecutor(max_workers=len(columns)) as threads:
    totals = list(threads.map(lambda column: sum_column("example.vortex", column), columns))
```

By default Vortex starts a background worker pool sized to `available_parallelism() - 1`.
Set `VORTEX_MAX_THREADS=n` to pin the pool to a specific size at startup. To adjust the pool
at runtime, use {func}`~vortex.set_worker_threads`; passing `None` resets it to the default:

```python
import vortex as vx

previous_workers = vx.worker_threads()
vx.set_worker_threads(None)  # reset to available_parallelism() - 1

try:
    reader = vx.open("example.vortex").to_arrow(batch_size=64_000)
    table = reader.read_all()
finally:
    vx.set_worker_threads(previous_workers)
```

## Conversion

Arrays convert to other formats:

| Method                          | Result                       |
|---------------------------------|------------------------------|
| {meth}`.Array.to_arrow_array`   | {class}`pyarrow.Array`       |
| {meth}`.Array.to_arrow_table`   | {class}`pyarrow.Table`       |
| {meth}`.Array.to_numpy`         | {class}`numpy.ndarray`       |
| {meth}`.Array.to_pandas`        | {class}`pandas.DataFrame`    |
| {meth}`.Array.to_pylist`        | {class}`list`                |
