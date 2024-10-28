.. Vortex documentation master file, created by
   sphinx-quickstart on Wed Aug 28 10:10:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Wide, Fast & Compact. Pick Three.
==================================

.. grid:: 1 1 2 2
    :gutter: 4 4 4 4

    .. grid-item-card:: The File Format
       :link: file_format
       :link-type: doc

       Currently just a schematic. Specification forthcoming.

    .. grid-item-card:: The Rust API
       :link: https://spiraldb.github.io/vortex/docs2/rust/doc/vortex

       The primary interface to the Vortex toolkit.

    .. grid-item-card:: Quickstart
       :link: quickstart
       :link-type: doc

       For end-users looking to read and write Vortex files.

    .. grid-item-card:: The Benchmarks
       :link: https://bench.vortex.dev/

       Random access, throughput, and TPC-H.


Vortex is a fast & extensible columnar file format that is based around state-of-the-art research
from the database community. It is built around cascading compression with lightweight encodings (no
block compression), allowing for both efficient random access and extremely fast decompression.

Vortex also includes an accompanying in-memory format for these (recursively) compressed arrays,
that is zero-copy compatible with Apache Arrow in uncompressed form. Taken together, the Vortex
library is a useful toolkit with compressed Arrow data in-memory, on-disk, & over-the-wire.

Vortex aspires to succeed Apache Parquet by pushing the Pareto frontier outwards: 1-2x faster
writes, 2-10x faster scans, and 100-200x faster random access reads, while preserving the same
approximate compression ratio as Parquet v2 with zstd.

Its features include:

- A zero-copy data layout for disk, memory, and the wire.
- Kernels for computing on, filtering, slicing, indexing, and projecting compressed arrays.
- Builtin state-of-the-art codecs including FastLanes (integer bit-packing), ALP (floating point),
  and FSST (strings).
- Support for custom user-implemented codecs.
- Support for, but no requirement for, row groups.
- A read sub-system supporting filter and projection pushdown.

Vortex's flexible layout empowers writers to choose the right layout for their setting: fast writes,
fast reads, small files, few columns, many columns, over-sized columns, etc.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   quickstart
   guide
   file_format
   api/index
   Rust API <https://spiraldb.github.io/vortex/docs2/rust/doc/vortex>
