# PyArrow

## Getting Started

First, install if you haven't already:

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

Construct a Vortex array from lists of simple Python values:

```{doctest} pycon
>>> import vortex as vx
>>> arr = vx.array([1, 2, 3, 4])
>>> arr.dtype
int(64, nullable=False)
```

Python's {obj}`None` represents a missing or null value and changes the dtype of the array from
non-nullable 64-bit integers to nullable 64-bit integers:

```{doctest} pycon
>>> arr = vx.array([1, 2, None, 4])
>>> arr.dtype
int(64, nullable=True)
```

A list of {class}`dict` is converted to an array of structures. Missing values may appear at any
level:

```{doctest} pycon

>>> arr = vx.array([
...   {'name': 'Joseph', 'age': 25},
...   {'name': None, 'age': 31},
...   {'name': 'Angela', 'age': None},
...   {'name': 'Mikhail', 'age': 57},
...   {'name': None, 'age': None},
...   None,
... ])
>>> arr.dtype
struct({"age": int(64, nullable=True), "name": utf8(nullable=True)}, nullable=True)
```

{meth}`.Array.to_pylist` converts a Vortex array into a list of Python values.

```{doctest} pycon
>>> arr.to_pylist()
[{'age': 25, 'name': 'Joseph'}, {'age': 31, 'name': None}, {'age': None, 'name': 'Angela'}, {'age': 57, 'name': 'Mikhail'}, {'age': None, 'name': None}, {'age': None, 'name': None}]
```

## Arrow

The {func}`~vortex.array` function constructs a Vortex array from an Arrow one without any
copies:

```{doctest} pycon
>>> import pyarrow as pa
>>> arrow = pa.array([1, 2, None, 3])
>>> arrow.type
DataType(int64)
>>> arr = vx.array(arrow)
>>> arr.dtype
int(64, nullable=True)
```

{meth}`.Array.to_arrow_array` converts back to an Arrow array:

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

If you have a struct array, use {meth}`.Array.to_arrow_table` to construct an Arrow table:

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

