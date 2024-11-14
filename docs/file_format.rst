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
all the rows of all the columns are not necessarily contiguously laid out. Vortex currently has
three kinds of *layouts* which recursively compose: the *flat layout*, the *columnar layout*, and
the *chunked layout*.

The flat layout is a contiguous sequence of bytes. *Any* Vortex array encoding (including
struct-typed arrays) can be serialized into the flat layout.

The columnar layout lays out each column of a struct-typed array as a separate sequence of bytes. Each
column may or may not recursively use a chunked layout. Columnar layouts permit readers to push-down
column projections.

The chunked layout lays out an array as a sequence of row chunks. Each chunk may have a different
size. A chunked layout permits reader to push-down row filters based on statistics and/or row offsets.
Note that if the laid out array is a struct array, each column will use the same chunk size. This is
equivalent to Parquet's row groups.

A few examples of concrete layouts:

1. Chunked of columnar of chunked of flat: essentially a Parquet layout with row groups in which each
   column's values are contiguously stored in pages. Note that in this case, the pages within each
   "row group" may be of different sizes / do not have to be aligned.
2. Columnar of chunked of flat: eliminates row groups, retaining only pages.
3. Columnar of flat: prevents row filter push-down because each column is an opaque sequence of bytes.

The chunked layout has an optional child that corresponds to a Vortex `StructArray` of per-chunk
statistics (sometimes referred to as a "statistics table"), which contains metadata necessary for
effective row filtering such as sortedness, the minimum value, the maximum value, and the number of
null rows. Other statistics (e.g., sortedness) are stored inline with the data.

The current writer implementation writes all such "metadata" IPC messages after writing all of the
"data" IPC messages (allowing us to maximize the probability that metadata pruning can proceed
after the first read from disk / object storage). The location of the metadata messages is encoded
in the layout, which is then serialized just before the very end of the file.

One implication of this is that the precise location of the metadata is not itself part of the file
format specification. Instead, it is fully described by the layout.

.. card::

   .. figure:: _static/file-format-2024-10-23-1642.svg
      :width: 800px
      :alt: A schematic of the file format

   +++

   The Vortex file format has two top-level sections:

   1. Data (typically array IPC messages, followed by statistics, though that's a writer implementation detail),
   2. Footer, which contains the schema (i.e., the logical type), the layout, a postscript (containing offsets), and an 8-byte end-of-file marker.

.. _included-codecs:

Encodings
^^^^^^^^^

- Most of the Arrow encodings.
- Chunked: a sequence of arrays.
- Constant: a value and a length.
- Sparse: a default value, plus a pair of arrays representing exceptions: an array of indices and of values.
- FastLanes Frame-of-Reference, BitPacking, and Delta.
- Fast Static Symbol Table (FSST).
- Adapative Lossless Floating Point (ALP).
- ALP Real Double (ALP-RD).
- ByteBool: one byte per Boolean value.
- ZigZag.

Specification
-------------

TODO!
