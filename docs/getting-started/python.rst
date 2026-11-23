:tocdepth: 1

Python
======

Install
-------

::

   pip install vortex-data

The pandas, Polars and DuckDB examples on this page need those packages. Install them as extras::

   pip install "vortex-data[pandas,polars,duckdb]"

Write
-----

:func:`~vortex.io.write` writes anything Arrow-shaped, such as a :class:`pyarrow.Table` or a
:class:`pyarrow.RecordBatchReader`, to a Vortex file:

.. doctest::

   >>> import pyarrow as pa
   >>> import vortex as vx
   >>> people = pa.table({"name": ["Joseph", "Angela", "Mikhail", "Ana"], "age": [25, 31, 57, 42]})
   >>> vx.io.write(people, "people.vortex")

To convert a Parquet file, read it with Arrow and write the table. You can use your own Parquet file
or download the `example file used here <https://docs.vortex.dev/_static/example.parquet>`__:

.. doctest::

   >>> import pyarrow.parquet as pq
   >>> vx.io.write(pq.read_table("_static/example.parquet"), "example.vortex")  # doctest: +SKIP

:func:`~vortex.io.write` compresses the data with options that balance file size and read speed.
To prioritize a small file instead, write with the compact options of
:class:`~vortex.io.VortexWriteOptions`:

.. doctest::

   >>> vx.io.VortexWriteOptions.compact().write(people, "people-compact.vortex")

Read
----

:func:`~vortex.open` opens a file lazily: it reads the footer and nothing else. The footer holds
the row count and the schema:

.. doctest::

   >>> f = vx.open("people.vortex")
   >>> len(f)
   4
   >>> f.dtype
   struct({"name": utf8(nullable=True), "age": int(64, nullable=True)}, nullable=False)

:func:`~vortex.open` also accepts a URL. Vortex infers the object store from the URL, or you can
pass one from :doc:`/api/python/store` as ``store``:

.. code-block:: python

   f = vx.open("s3://bucket/people.vortex")

:meth:`~vortex.VortexFile.to_arrow` reads the file as a :class:`pyarrow.RecordBatchReader`, which
streams record batches instead of loading the whole file at once. Pass ``limit`` to read only the
first rows:

.. doctest::

   >>> f.to_arrow().read_all().to_pydict()
   {'name': ['Joseph', 'Angela', 'Mikhail', 'Ana'], 'age': [25, 31, 57, 42]}
   >>> f.to_arrow(limit=2).read_all().to_pydict()
   {'name': ['Joseph', 'Angela'], 'age': [25, 31]}

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

Vortex is architected for fast random access, with reads about 100x faster than Parquet in the
`benchmarks <https://bench.vortex.dev>`__. If an external index gives you the rows you need, pass their indices, which must be sorted and
unique, to skip the IO and decoding of every other row:

.. doctest::

   >>> f.to_arrow(indices=vx.array([0, 2])).read_all().to_pydict()
   {'name': ['Joseph', 'Mikhail'], 'age': [25, 57]}

pandas
------

:meth:`~vortex.VortexFile.scan` reads the file as Vortex arrays, which convert to pandas, Polars,
NumPy and Arrow. It takes the same column selection, ``expr``, ``limit`` and ``indices`` as
:meth:`~vortex.VortexFile.to_arrow`, so the DataFrame holds only the data you need:

.. doctest::

   >>> df = f.scan(["name", "age"], expr=ve.column("age") > 35).read_all().to_pandas()
   >>> df
         name  age
   0  Mikhail   57
   1      Ana   42

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

Many files
----------

:func:`~vortex.open_dataset` opens a directory, a glob pattern, an object store prefix or a list of
files as one :class:`~vortex.dataset.VortexDataset`. Every file must have the same schema; opening
the dataset reads only the footer of the first file and opens the rest lazily when scanned:

.. doctest::

   >>> vx.io.write(people, "team-a.vortex")
   >>> vx.io.write(pa.table({"name": ["Wei", "Priya"], "age": [38, 29]}), "team-b.vortex")
   >>> teams = vx.open_dataset("team-*.vortex")
   >>> teams.file_count
   2

The dataset reads the same way as the dataset of one file, with column selection and filters pushed
into the scan of every file:

.. doctest::

   >>> _ = con.register("teams", teams)
   >>> con.sql("SELECT name FROM teams WHERE age > 35 ORDER BY name").fetchall()
   [('Ana',), ('Mikhail',), ('Wei',)]
   >>> teams.to_polars().filter(pl.col("age") < 30).select("name").collect()["name"].to_list()
   ['Joseph', 'Priya']

A read keeps up to ``concurrency`` splits in flight across all the files, so many small files read
as concurrently as one large file. Query engines such as DuckDB and Polars do not set it, so set it
on the dataset to tune their reads, for example to keep more requests in flight against an object
store:

.. code-block:: python

   teams = vx.open_dataset("s3://bucket/teams/", concurrency=32)

Next steps
----------

The user guides cover each integration in more depth: :doc:`/user-guide/pyarrow`,
:doc:`/user-guide/pandas`, :doc:`/user-guide/polars` and :doc:`/user-guide/duckdb`.
:doc:`/user-guide/vortex-python` covers arrays and dtypes, and :doc:`/api/python/store/huggingface`
covers reading Vortex files with Hugging Face ``datasets``. The :doc:`/api/python/index` documents
every function and class.

For files shared between applications or Vortex versions, see :doc:`/specs/versioning`.
