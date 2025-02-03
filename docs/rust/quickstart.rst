Quickstart
==========

The reference implementation exposes both a Rust and Python API. A C API is currently in progress.

- :ref:`Quickstart for Python <python-quickstart>`
- :ref:`Quickstart for Rust <rust-quickstart>`

.. _python-quickstart:

Python
------

Install
^^^^^^^

::

   pip install vortex-array

Convert
^^^^^^^

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
   141055

Compress
^^^^^^^^

Use :func:`~vortex.compress` to compress the Vortex array and check the relative size:

.. doctest::

   >>> cvtx = vx.compress(vtx)
   >>> cvtx.nbytes
   15768
   >>> cvtx.nbytes / vtx.nbytes
   0.11...

Vortex uses nearly ten times fewer bytes than Arrow. Fewer bytes means more of your data fits in
cache and RAM.

Write
^^^^^

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
^^^^

Use :func:`~vortex.io.read_path` to read the Vortex array from disk:

.. doctest::

   >>> cvtx = vortex.io.read_path("example.vortex")

.. _rust-quickstart:

Rust
----

Install
^^^^^^^

Install vortex and all the first-party array encodings::

   cargo add vortex

Convert
^^^^^^^

You can either use your own Parquet file or download the `example used here
<https://spiraldb.github.io/vortex/docs/_static/example.parquet>`__.

Use Arrow to read a Parquet file and then construct an uncompressed Vortex array:

.. code-block:: rust

   use std::fs::File;

   use arrow_array::RecordBatchReader;
   use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
   use vortex::array::ChunkedArray;
   use vortex::arrow::FromArrowType;
   use vortex::{Array, IntoArray};
   use vortex::dtype::DType;

   let reader =
       ParquetRecordBatchReaderBuilder::try_new(File::open("_static/example.parquet").unwrap())
           .unwrap()
           .build()
           .unwrap();
   let dtype = DType::from_arrow(reader.schema());
   let chunks = reader
       .map(|x| Array::try_from(x.unwrap()).unwrap())
       .collect::<Vec<_>>();
   let vtx = ChunkedArray::try_new(chunks, dtype).unwrap().into_array();

Compress
^^^^^^^^

Use the sampling compressor to compress the Vortex array and check the relative size:

.. code-block:: rust

   use std::collections::HashSet;

   use vortex::sampling_compressor::{SamplingCompressor, DEFAULT_COMPRESSORS};

   let compressor = SamplingCompressor::new(HashSet::from(*DEFAULT_COMPRESSORS));
   let cvtx = compressor.compress(&vtx, None).unwrap().into_array();
   println!("{}", cvtx.nbytes());

Write
^^^^^

Reading and writing both require an async runtime; in this example we use Tokio. The
VortexFileWriter knows how to write Vortex arrays to disk:

.. code-block:: rust

   use std::path::Path;

   use tokio::fs::File as TokioFile;
   use vortex_serde::file::write::writer::VortexFileWriter;

   let file = TokioFile::create(Path::new("example.vortex"))
       .await
       .unwrap();
   let writer = VortexFileWriter::new(file)
       .write_array_columns(cvtx.clone())
       .await
       .unwrap();
   writer.finalize().await.unwrap();

Read
^^^^

.. code-block:: rust

   use futures::TryStreamExt;
   use vortex::sampling_compressor::ALL_COMPRESSORS_CONTEXT;
   use vortex_serde::file::read::builder::{VortexReadBuilder, LayoutDeserializer};

   let file = TokioFile::open(Path::new("example.vortex")).await.unwrap();
   let builder = VortexReadBuilder::new(
       file,
       LayoutDeserializer::new(
           ALL_COMPRESSORS_CONTEXT.clone(),
           LayoutContext::default().into(),
       ),
   );

   let stream = builder.build().await.unwrap();
   let dtype = stream.schema().clone().into();
   let vecs: Vec<Array> = stream.try_collect().await.unwrap();
   let cvtx = ChunkedArray::try_new(vecs, dtype)
       .unwrap()
       .into_array();

   println!("{}", cvtx.nbytes());
