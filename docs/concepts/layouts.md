# Layouts

Layouts are the out-of-memory equivalent of [Vortex arrays](/concepts/arrays). They are similarly hierarchical,
with an associated vtable, metadata, dtype, children, and lazy buffers known as "segments".

The tree-structure of a layout can be serialized and persisted. During deserialization, the layout is bound to a
segment source that can lazily fetch the data buffers as needed. This abstraction allows Vortex to implement highly
efficient columnar scans over any block storage including local disk, object stores, remote caches like Redis,
Postgres block storage, and more. 

In fact, the [Vortex file format](/concepts/file-format) is just a serialized layout tree with the data segments 
stored in the same file.

## Built-in Layouts

As with arrays, Vortex provides a number of built-in layouts, and users can define their own custom layouts.

| Name               | Description                                                                                             |
|--------------------|---------------------------------------------------------------------------------------------------------|
| `FlatLayout`       | A layout that holds a single serialized Vortex array.                                                   |
| `StructLayout`     | A layout that holds a collection of named child layouts, corresponding to an associated `StructDType`.  |
| `ChunkedLayout `   | A layout that holds a collection of row-wise partitioned child layouts.                                 | 
| `DictionaryLayout` | A layout that shares a single dictionary of values with a child layout holding indices.                 |
| `ZonedLayout`      | A layout that stores a zone-map of statistics to perform filter pruning.                                |


### Example: Parquet Row Groups

Layouts can be composed together in arbitrary hierarchical structures. This allows users of Vortex to configure 
writers that model the performance characteristics of other file formats or storage systems.

As an example, suppose we want to replicate the behavior of Parquet row groups in Vortex. We would define a layout that
looked something like:

* `ChunkedLayout(ChunkBy::RowCount(100_000))` - at the top-level, we define row-groups of at most 100k rows.
    * `StructLayout` - Parquet then splits the row group into individual columns known as column chunks.
        * `ChunkedLayout(ChunkBy::CompressedSize(64k))` - finally, each column chunk is split into pages by compressed
          size.

## Layout Strategies

A `LayoutStrategy` defines how to construct a layout tree from a stream of Vortex arrays. These strategies can 
partition arrays by column, by row-groups, or by any other arbitrary scheme. Some strategies compute pruning stats, 
others apply compression to the data. 

For segment sinks that are locality-aware, such as a Vortex file, layout strategies can make use of sequence IDs.
These are powerful logical clocks that allow layouts to parallelize writes and compression tasks while maintaining 
full control and determinism over where segments are written into the file.
