# Pandas

Vortex in-memory arrays can be converted to and from Pandas DataFrames.

## Reading a Vortex File into Pandas

To read a Vortex file into a Pandas DataFrame, open the file, scan the data into memory, and
convert:

```{doctest} pycon
>>> import vortex as vx
>>> import pyarrow.parquet as pq
>>>
>>> vx.io.write(pq.read_table("_static/example.parquet"), 'example.vortex')
>>>
>>> f = vx.open('example.vortex')
>>> df = f.scan().read_all().to_pandas()
>>> df[['tip_amount', 'fare_amount']].head(3)
   tip_amount  fare_amount
0         0.0         61.8
1         5.1         20.5
2        16.54         70.0
```

{meth}`.VortexFile.scan` returns an {class}`.ArrayIterator` that streams batches from disk.
{meth}`.ArrayIterator.read_all` collects all batches into a single in-memory {class}`.Array`, and
{meth}`.Array.to_pandas` converts it to a DataFrame.

## Converting In-Memory Arrays

{meth}`.Array.to_pandas` converts any struct-typed Vortex array into a Pandas DataFrame:

```{doctest} pycon
>>> struct_arr = vx.array([
... {'name': 'Joseph', 'age': 25},
... {'name': 'Narendra', 'age': 31},
... {'name': 'Angela', 'age': 33},
... {'name': 'Mikhail', 'age': 57},
... ])
>>> struct_arr.to_pandas()
      age      name
   0   25    Joseph
   1   31  Narendra
   2   33    Angela
   3   57   Mikhail
```

{func}`~vortex.array` converts from a Pandas DataFrame into a Vortex array:

```{doctest} pycon
>>> import pandas as pd
>>>
>>> df = pd.DataFrame({'age': [25, 31, 33, 57], 'name': ['Joseph', 'Narendra', 'Angela', 'Mikhail']})
>>> vx.array(df).to_arrow_table()
pyarrow.Table
age: int64
name: string_view
----
age: [[25,31,33,57]]
name: [["Joseph","Narendra","Angela","Mikhail"]]
```
