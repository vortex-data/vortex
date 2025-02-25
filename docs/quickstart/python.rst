:tocdepth: 1

Python Quickstart
=================

Install
-------

::

   pip install vortex-array

Convert
-------

You can either use your own Parquet file or download the `example used here
<https://spiraldb.github.io/vortex/docs/_static/example.parquet>`__.

Use Arrow to read a Parquet file and then use :func:`~vortex.array` to construct an uncompressed
Vortex array:

.. doctest::

   >>> import pyarrow.parquet as pq
   >>> import vortex as vx
   >>> parquet = pq.read_table("_static/example.parquet")
   >>> vtx = vx.array(parquet)
   >>> vtx.nbytes
   141025

Compress
--------

Use :func:`~vortex.compress` to compress the Vortex array and check the relative size:

.. doctest::

   >>> cvtx = vx.compress(vtx)
   >>> cvtx.nbytes
   14415
   >>> cvtx.nbytes / vtx.nbytes
   0.10...

Vortex uses nearly ten times fewer bytes than Arrow. Fewer bytes means more of your data fits in
cache and RAM.

Write
-----

Use :func:`~vortex.io.write_path` to write the Vortex array to disk:

.. doctest::

   >>> vortex.io.write_path(cvtx, "example.vortex")

Small Vortex files (this one is just 71KiB) currently have substantial overhead relative to their
size. This will be addressed shortly. On files with at least tens of megabytes of data, Vortex is
similar to or smaller than Parquet.

.. doctest::

   >>> from os.path import getsize
   >>> getsize("example.vortex") / getsize("_static/example.parquet") # doctest: +SKIP
   2.0...

Read
----

Use :func:`~vortex.io.read_path` to read the Vortex array from disk:

.. doctest::

   >>> cvtx = vortex.io.read_path("example.vortex")
