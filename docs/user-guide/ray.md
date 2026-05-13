# Ray Data

Vortex provides a Ray Data datasource for reading Vortex files in distributed Ray pipelines.

<!-- Ray does not start correctly in a `uv run` environment. Instead, `source .venv/bin/activate`
and then run `make -C docs doctest` -->

```{doctest} pycon
>>> import vortex as vx
>>> import pyarrow.parquet as pq
>>> import os
>>>
>>> os.makedirs("ray_data", exist_ok=True)
>>> table = pq.read_table("_static/example.parquet")
>>>
>>> vx.io.write(table, 'ray_data/example-01.vortex')
>>> vx.io.write(table, 'ray_data/example-02.vortex')
>>> vx.io.write(table, 'ray_data/example-03.vortex')
>>> from vortex.ray.datasource import VortexDatasource
>>> from ray.data import read_datasource
>>> ds = read_datasource(VortexDatasource(url='ray_data')) # doctest: +SKIP
>>> ds.to_pandas() # doctest: +SKIP
      VendorID tpep_pickup_datetime  ... congestion_surcharge  Airport_fee
0            1  2023-11-01 00:03:03  ...                  0.0         1.75
1            1  2023-11-01 00:03:28  ...                  2.5         0.00
2            2  2023-10-31 23:58:05  ...                  2.5         1.75
3            2  2023-11-01 00:03:50  ...                  2.5         0.00
4            2  2023-11-01 00:06:30  ...                  2.5         0.00
...        ...                  ...  ...                  ...          ...
2995         1  2023-11-01 00:09:20  ...                  2.5         0.00
2996         2  2023-11-01 00:16:03  ...                  2.5         0.00
2997         2  2023-11-01 00:32:42  ...                  2.5         0.00
2998         1  2023-11-01 00:04:52  ...                  2.5         0.00
2999         1  2023-11-01 00:18:56  ...                  2.5         0.00
<BLANKLINE>
[3000 rows x 19 columns]
```
