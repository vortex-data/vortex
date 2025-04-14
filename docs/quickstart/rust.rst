:tocdepth: 1

Rust Quickstart
===============

Install
-------

Install vortex and all the first-party array encodings::

   cargo add vortex

Convert
-------

You can either use your own Parquet file or download the `example used here
<https://spiraldb.github.io/vortex/docs/_static/example.parquet>`__.

Use Arrow to read a Parquet file and then construct an uncompressed Vortex array:

.. literalinclude:: ../../vortex/src/lib.rs
    :language: rust
    :dedent:
    :start-after: [convert]
    :end-before: [convert]

Compress
--------

Use the sampling compressor to compress the Vortex array and check the relative size:

.. literalinclude:: ../../vortex/src/lib.rs
    :language: rust
    :dedent:
    :start-after: [compress]
    :end-before: [compress]


Write
-----

Reading and writing both require an async runtime; in this example we use Tokio. The
VortexFileWriter knows how to write Vortex arrays to disk:

.. literalinclude:: ../../vortex/src/lib.rs
    :language: rust
    :dedent:
    :start-after: [write]
    :end-before: [write]

Read
----

.. literalinclude:: ../../vortex/src/lib.rs
    :language: rust
    :dedent:
    :start-after: [read]
    :end-before: [read]
