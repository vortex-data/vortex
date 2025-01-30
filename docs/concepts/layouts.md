# Vortex Layouts

Layouts share many similarities with [Vortex Arrays](/concepts/arrays). They are hierarchical, they have an associated
vtable, and they have some number of buffers. The main difference is that the buffers of a layout are lazily fetched
and remotely stored.

This allows layouts to perform pruning of unused chunks and columns, without tying the logic to a specific file-based
storage format, and without prescribing the column and row partitioning that a Vortex file can use.

In fact, Layouts provide a mechanism to perform efficient scanning of columnar data over any storage medium.
The buffers might live in-memory, in a single file on-disk, split across many files, in a remote Redis, in Postgres
block storage, or anywhere else that you can implement key/value blob storage.

In psuedo-code, a layout might look like this (note that unlike arrays, layouts use u64 lengths to support larger-than
memory data):

```rust
struct LayoutData {
    vtable: LayoutVTable,
    metadata: [u8],
    dtype: DType,
    length: u64,
    children: [LayoutData],
    buffers: [BufferId],
}
```

**Owned vs Viewed**

As with other possibly large recursive data structures in Vortex, layouts can be either _owned_ or _viewed_.
Owned layouts are heap-allocated, while viewed layouts are lazily unwrapped from an underlying FlatBuffer
representation. This allows Vortex to efficiently load and work with very wide schemas without needing to deserialize
the full layout.

## VTable

The vtable of a layout is much smaller than that of an array. It looks something like this:

* `id`: returns the unique identifier for the layout type.
* `metadata`
    * `validate`: validates the layout's metadata buffer.
    * `display`: returns a human-readable representation of the layout metadata.
* `accept`: a function for accepting a `LayoutVisitor` and walking the layout's children.
* `reader`: constructs a `LayoutReader` given an async source of buffers.

## Built-in Layouts

Vortex provides a few built-in layout types, and will continue to add new layouts as compression strategies improve.

### Flat Layout

A `FlatLayout` simply holds a serialized Vortex array. This can be considered the leaf node of a layout tree.

### Struct Layout

A `StructLayout` holds a collection of named child layouts, corresponding to an associated `StructDType`. This layout
assists with pruning by partitioning the evaluation expression into sub-expressions that can be evaluated over each
of the referenced fields.

### Chunked Layout

A `ChunkedLayout` holds a collection of row-wise partitioned child layouts. This layout assists with pruning by
computing statistics for each child chunk and only fetching chunks that are relevant to the expression being
evaluated.

* `chunks: [LayoutData]`: the first `n` children of a `ChunkedLayout` are the chunks themselves.
* `statistics: LayoutData`: the last child is a statistics table, typically a `FlatLayout` (although different
  layouts may be useful if some statistics grow very large, e.g. bloom filters). Each row corresponds to a chunk, and
  the columns hold statistics such as `min`, `max`, `null_count`, that are useful for pruning.

### Future Layouts

There are some additional layouts that we plan to add in the future:

* `DictionaryLayout`: a layout that holds a dictionary of values in one child layout, and a codes array
  (likely chunked) in another child layout.
* `ListLayout`: a layout that separates the offsets and values of a list array into two child layouts, allowing
  for efficient pruning of the values array based on the relevant offsets.
* `MergeLayout` a struct layout that can split fields of a struct across separate layouts, combining the result back
  into a single struct. This can be useful to isolate outsized columns and use a different chunking strategy, without
  impacting the compression or read performance of the other columns.

## Custom Layouts

As with most parts of Vortex, users can define their own layout types. Reach out on the Vortex GitHub Discussions
page if you need help defining a custom layout.

## Layout Writer

A `LayoutWriter` defines a way to serialize a stream of array chunks into a layout tree. The writer is given a
buffer writer that takes a `ByteBuffer` and returns a `BufferId`. These identifiers are used to construct the layout
tree.

The Rust trait looks like this:

:::{literalinclude} ../../vortex-layout/src/strategies/mod.rs
:start-after: [layout writer]
:end-before: [layout writer]
:::

### File-level Compression

While chunk-level compression can be handed off to a compression strategy, i.e. `fn(Array) -> Array`, there
are some compression techniques that benefit from file-level awareness. For example, sharing a dictionary across
all chunks of a column.

To support this with larger-than-memory data these techniques can be implemented inside a `LayoutStrategy`.

For example, a `DictionaryLayoutStrategy` may accumulate a values dictionary in-memory, while flushing chunks of
codes arrays to disk.
If the dictionary grows too large, the strategy can flush the values dictionary, start a new dictionary, and then
wrap both of these `DictionaryLayout` nodes in a new `ChunkedLayout` node.

## Example: Parquet Row Groups

As an example, suppose we want to replicate the behavior of Parquet row groups in Vortex. We would define a layout
strategy that constructed something like the following tree:

* `ChunkedLayout(ChunkBy::RowCount(100_000))` - at the top-level, we define row-groups of at most 100k rows.
    * `StructLayout` - Parquet then splits the row group into individual columns known as column chunks.
        * `ChunkedLayout(ChunkBy::CompressedSize(64k))` - finally, each column chunk is split into pages by compressed
          size.
