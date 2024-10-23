.. Vortex documentation master file, created by
   sphinx-quickstart on Wed Aug 28 10:10:21 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Wide, Fast & Compact. Pick Three.
==================================

.. grid:: 1 1 2 2
    :gutter: 4 4 4 4

    .. grid-item-card::  Row groups? Your choice.
       :link: file-format--layouts
       :link-type: ref

       Bring your wide schemas, images, and videos.

    .. grid-item-card::  All your favorite query engines.
       :link: query-engine-integration
       :link-type: ref

       Query pushdown in Pandas, Polars, DuckDB, & chDB.

    .. grid-item-card::  200x lower latency random reads.

       Block compression is for chumps.

    .. grid-item-card:: Zero copy reads.

       It's called a processor, not a Xerox.

    .. grid-item-card::  Batteries-included.
       :columns: 12 12 12 12
       :link: included-codecs
       :link-type: ref

       Cutting-edge codecs: FSST, ALP, FastLanes, and more.

Vortex is a fast, extensible, lightweight-compressed, and random-access columnar file format as well
as a library for working with compressed Apache Arrow arrays in-memory, on-disk, and
over-the-wire. Vortex aspires to succeed Apache Parquet by delivering two orders of magnitude faster
random-access and faster scans without sacrificing compression ratio nor write throughput. Its
features include:

- A zero-copy data layout for disk, memory, and the wire.
- Kernels for computing on, filtering, slicing, indexing, and projecting compressed arrays.
- Builtin state-of-the-art codecs including FastLanes (integer bit-packing), ALP (floating point),
  and FSST (strings).
- Support for custom user-implemented codecs.
- Support for, but no requirement for, row groups.
- A read sub-system supporting filter and projection pushdown.

Spiral's flexible layout empowers writers to choose the right layout for their setting: fast writes,
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
