File Format
===========

Intuition
---------

The Vortex file format has both *layouts*, which describe how different chunks of columns are stored
relative to one another, and *encodings* which describe the byte representation of a contiguous
sequence of values. A layout describes how to contiguously store one or more arrays as is necessary
for storing an array on disk or transmitting it over the wire. An encoding defines one binary
representation for memory, disk, and the wire.

.. _file-format--layouts:

Layouts
^^^^^^^

Vortex arrays have the same binary representation in-memory, on-disk, and over-the-wire; however,
all the rows of all the columns are not necessarily contiguously laid out. Vortex has three kinds of
*layouts* which recursively compose: the *flat layout*, the *column layout*, and the *chunked
layout*.

The flat layout is a contiguous sequence of bytes. Any Vortex array encoding can be serialized into
the flat layout.

The column layout lays out each column of a struct-typed array as a separate sequence of bytes. Each
column may or may not recursively use a chunked layout. Column layouts permit readers to push-down
column projections.

The chunked layout lays out an array as a sequence of row chunks. Each chunk may have a different
size. A chunked layout permits reader to push-down row filters based on statistics which we describe
later. Note that, if the laid out array is a struct array, each column uses the same chunk
size. This is equivalent to Parquet's row groups.

A few examples of concrete layouts:

1. Chunked of struct of chunked of flat: essentially a Parquet layout with row groups in which each
   column's values are contiguously stored in pages.
2. Struct of chunked of flat: eliminates row groups, retaining only pages.
3. Struct of flat: prevents row filter push-down because each array is, to the layout, an opaque
   sequence of bytes.

The chunked layout stores, per chunk, metadata necessary for effective row filtering such as
sortedness, constancy, the minimum value, the maximum value, and the number of null rows. Readers
consult these metadata tables to avoid reading chunks without relevant data.

.. card::

   .. figure:: _static/file-format-2024-10-23-1642.svg
      :width: 800px
      :alt: A schematic of the file format

   +++

   The Vortex file format has five sections: data, statistics, schema, footer, and postscript. The
   postscript describes the locating of the schema and layout which in turn describe how to
   interpret the data and metadata. The schema describes the logical type. The metadata contains
   information necessary for row filtering.

.. _included-codecs:

Encodings
^^^^^^^^^

- Most of the Arrow encodings.
- Chunked, a sequence of arrays.
- Constant, a value and a length.
- Sparse, a value plus a pair of arrays representing exceptions: an array of indices and of values.
- FastLanes Frame-of-Reference, BitPacking, and Delta.
- Fast Static Symbol Table (FSST).
- Adapative Lossless Floating Point (ALP).
- ALP Real Double (ALP-RD).
- ByteBool, one byte per Boolean value.
- ZigZag.

Specification
-------------

TODO!
