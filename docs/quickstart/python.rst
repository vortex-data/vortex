:tocdepth: 1

Python Quickstart
=================

Install
-------

::

   pip install vortex-data

Convert
-------

You can either use your own Parquet file or download the `example used here
<https://docs.vortex.dev/_static/example.parquet>`__.

Use Arrow to read a Parquet file and then use :func:`~vortex.array` to construct an uncompressed
Vortex array:

.. doctest::

   >>> import pyarrow.parquet as pq
   >>> import vortex as vx
   >>> parquet = pq.read_table("_static/example.parquet")
   >>> vtx = vx.array(parquet)
   >>> vtx.nbytes
   141024

Write
-----

Use :func:`~vortex.io.write` to write the Vortex array to disk:

.. doctest::

   >>> vortex.io.write(cvtx, "example.vortex") # doctest: +SKIP

Small Vortex files (this one is just 71KiB) currently have substantial overhead relative to their
size. This will be addressed shortly. On files with at least tens of megabytes of data, Vortex is
similar to or smaller than Parquet.

.. doctest::

   >>> from os.path import getsize
   >>> getsize("example.vortex") / getsize("_static/example.parquet") # doctest: +SKIP
   2.0...

Read
----

Use :func:`~vortex.open` to open and read the Vortex array from disk:

.. doctest::

   >>> cvtx = vortex.open("example.vortex").scan().read_all()


Vortex is architected to achieve fast random access, in many cases hundreds of times faster
than what can be achieved with Parquet.

If you have an external index that gives you specific rows to pull out of the Vortex file, you can skip a lot more
IO and decoding and read just the data that is relevant to you:

.. doctest::

    >>> vf = vortex.open("example.vortex")
    >>> # row indices must be ordered and unique
    >>> result = vf.scan(indices=vortex.array([1, 2, 10])).read_all()
    >>> assert len(result) == 3