:tocdepth: 1

Python
======

Install
-------

::

   pip install vortex-data

Write
-----

:func:`~vortex.io.write` writes anything Arrow-shaped, such as a :class:`pyarrow.Table` or a
:class:`pyarrow.RecordBatchReader`, to a Vortex file:

.. doctest::

   >>> import pyarrow as pa
   >>> import vortex as vx
   >>> people = pa.table({"name": ["Joseph", "Angela", "Mikhail", "Ana"], "age": [25, 31, 57, 42]})
   >>> vx.io.write(people, "people.vortex")

To convert a Parquet file, read it with Arrow first. You can use your own Parquet file or download
the `example file used here <https://docs.vortex.dev/_static/example.parquet>`__.
:func:`~vortex.array` turns the table into an uncompressed Vortex array:

.. doctest::

   >>> import pyarrow.parquet as pq
   >>> parquet = pq.read_table("_static/example.parquet")
   >>> vtx = vx.array(parquet)
   >>> vtx.nbytes
   141024
   >>> vx.io.write(vtx, "example.vortex")  # doctest: +SKIP

Small Vortex files (this one is just 71KiB) currently have substantial overhead relative to their
size. This will be addressed shortly. On files with at least tens of megabytes of data, Vortex is
similar to or smaller than Parquet.

.. doctest::

   >>> from os.path import getsize
   >>> getsize("example.vortex") / getsize("_static/example.parquet") # doctest: +SKIP
   2.0...

Read
----

:func:`~vortex.open` opens a file lazily: it reads the footer and nothing else.
:meth:`~vortex.VortexFile.to_arrow` reads it as a :class:`pyarrow.RecordBatchReader`, which
streams record batches instead of loading the whole file at once:

.. doctest::

   >>> f = vx.open("people.vortex")
   >>> len(f)
   4
   >>> f.to_arrow().read_all().to_pydict()
   {'name': ['Joseph', 'Angela', 'Mikhail', 'Ana'], 'age': [25, 31, 57, 42]}

Pass column names to read only those columns, and an ``expr`` to keep only the rows that match it.
Vortex reads data in proportion to the rows that pass the filter and the columns selected, and the
filter columns need not be selected:

.. doctest::

   >>> import vortex.expr as ve
   >>> f.to_arrow(["name"], expr=ve.column("age") > 35).read_all().to_pydict()
   {'name': ['Mikhail', 'Ana']}

Filters compare columns with plain Python values, and combine with ``&``, ``|`` and ``~``. See
:doc:`/api/python/expr` for the full set of expressions:

.. doctest::

   >>> thirties_and_forties = (ve.column("age") >= 30) & (ve.column("age") < 50)
   >>> f.to_arrow(["name"], expr=thirties_and_forties).read_all().to_pydict()
   {'name': ['Angela', 'Ana']}

Vortex is architected for fast random access, in many cases hundreds of times faster than Parquet.
If an external index gives you the rows you need, pass their indices, which must be sorted and
unique, to skip the IO and decoding of every other row:

.. doctest::

   >>> f.to_arrow(indices=vx.array([0, 2])).read_all().to_pydict()
   {'name': ['Joseph', 'Mikhail'], 'age': [25, 57]}

pandas
------

:meth:`~vortex.VortexFile.scan` reads the file as Vortex arrays, which convert to pandas, Polars,
NumPy and Arrow:

.. doctest::

   >>> df = f.scan().read_all().to_pandas()
   >>> df.shape
   (4, 2)
   >>> float(df["age"].mean())
   38.75

Polars
------

:meth:`~vortex.VortexFile.to_polars` reads the file as a :class:`polars.LazyFrame`. Polars pushes
the columns it needs and the filters it can into the Vortex scan:

.. doctest::

   >>> import polars as pl
   >>> f.to_polars().filter(pl.col("age") > 35).select("name").collect()["name"].to_list()
   ['Mikhail', 'Ana']

DuckDB
------

DuckDB's Vortex extension reads Vortex files, including globs of many files, straight from SQL. See
:doc:`/user-guide/duckdb`.

To query a file opened in Python, :meth:`~vortex.VortexFile.to_dataset` reads it as a
:class:`pyarrow.dataset.Dataset`, which DuckDB and other engines scan with column selection and
filter pushdown. Register the dataset with DuckDB instead of first reading the file into an Arrow
table, so that DuckDB reads only what the query needs:

.. doctest::

   >>> import duckdb
   >>> con = duckdb.connect()
   >>> _ = con.register("people", f.to_dataset())
   >>> con.sql("SELECT name FROM people WHERE age > 35 ORDER BY name").fetchall()
   [('Ana',), ('Mikhail',)]

The dataset also accepts Arrow compute expressions as filters:

.. doctest::

   >>> import pyarrow.compute as pc
   >>> f.to_dataset().to_table(columns=["name"], filter=pc.field("age") > 35).to_pydict()
   {'name': ['Mikhail', 'Ana']}

Next steps
----------

The user guides cover each integration in more depth: :doc:`/user-guide/pyarrow`,
:doc:`/user-guide/pandas`, :doc:`/user-guide/polars` and :doc:`/user-guide/duckdb`. The
:doc:`/api/python/index` documents every function and class.
