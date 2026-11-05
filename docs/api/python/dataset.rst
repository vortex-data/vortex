Dataset
=======

Vortex files implement the Arrow Dataset interface permitting efficient use of a Vortex file within
query engines like DuckDB and Polars. In particular, Vortex will read data proportional to the
number of rows passing a filter condition and the number of columns in a selection. For most Vortex
encodings, this property holds true even when the filter condition specifies a single row.

Use :func:`~vortex.open_dataset` to read many Vortex files as one dataset: a directory, a glob
pattern, an object store prefix or a list of files. Its ``concurrency`` sets how many splits a read
fetches and decodes ahead, across all the files, including reads that a query engine drives.

.. code-block:: python

    import duckdb
    import vortex as vx

    failures = vx.open_dataset("data_vortex/**/*.vortex", concurrency=32)
    duckdb.sql("SELECT date, count(*) FROM failures WHERE failure = 1 GROUP BY date").fetchall()

.. autosummary::
   :nosignatures:

   ~vortex.dataset.open_dataset
   ~vortex.dataset.VortexDataset
   ~vortex.dataset.VortexScanner
   ~vortex.dataset.VortexFragment

.. raw:: html

   <hr>

.. automodule:: vortex.dataset
    :members:
