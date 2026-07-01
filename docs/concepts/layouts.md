# Layouts

Layouts are the out-of-memory equivalent of [Vortex arrays](/concepts/arrays). A layout describes
how a logical array is organized across children and file segments so that scans can read only the
data they need.

The serialized layout tree is stored in a file footer. During deserialization, Vortex resolves each
layout encoding ID through the session's layout registry and constructs a typed `Layout<V>`:

- common fields are hoisted into `Layout<V>`: dtype, row count, child access, and segment IDs;
- layout-specific metadata lives in `V::LayoutData`;
- the erased `LayoutRef` lets heterogeneous layout nodes live in one tree; and
- child layouts are materialized lazily from the footer FlatBuffer when a scan route asks for them.

A layout does not execute a scan directly. Its vtable expands the typed layout into a
[`ScanPlan`](scanning.md), and the scan runtime prepares evidence, predicate, projection,
statistics, and aggregate work from that node tree.

## Built-in Layouts

As with arrays, Vortex provides a number of built-in layouts, and users can define their own custom
layouts.

| Name               | Description                                                                                            |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `FlatLayout`       | Stores one serialized Vortex array in one segment.                                                     |
| `StructLayout`     | Stores named child layouts corresponding to fields of a struct dtype.                                  |
| `ChunkedLayout`    | Stores row-wise partitioned child layouts and exposes chunk boundaries as natural scan splits.          |
| `DictionaryLayout` | Stores dictionary values in one child and row-domain codes in another child.                           |
| `ZonedLayout`      | Stores a data child plus zone statistics that can produce predicate evidence before reading row data.   |

## Layout Children

Child relationships are part of the layout contract. A child can be:

- a field child, such as one column of a struct;
- a chunk child, covering a row range of the parent;
- a transparent child, such as the data child of a zoned wrapper; or
- an auxiliary child, such as a validity bitmap, dictionary values, or zone statistics.

The parent vtable defines each child's expected dtype and relationship. This lets Vortex validate
lazy child access without deserializing the entire tree up front.

## Layouts and Segments

Layouts refer to data buffers by `SegmentId`. A segment source, such as a Vortex file or an
in-memory buffer, maps those logical segment IDs to bytes. This indirection keeps the layout tree
independent of where the bytes live: local disk, object storage, an embedded buffer, or a remote
cache can all back the same logical layout structure.

The scan path asks prepared reads and prepared evidence handles for segment requests when the
requests are known exactly. The segment source handles caching, coalescing, and in-flight
deduplication.

## Example: Parquet Row Groups

Layouts can be composed together in arbitrary hierarchical structures. This allows writers to model
the performance characteristics of other file formats or storage systems.

As an example, a Parquet-like layout could use:

- `ChunkedLayout(ChunkBy::RowCount(100_000))` at the top level for row groups.
- `StructLayout` inside each row group to split data by column.
- `ChunkedLayout(ChunkBy::CompressedSize(64k))` inside each column for page-like pieces.
- `FlatLayout` leaves that store serialized array chunks.

The scan runtime would still see one `ScanPlan` tree. Column projections route through the struct
node, row-range work routes through chunked nodes, and leaf reads touch only the flat segments needed
for the current morsel.

## Layout Strategies

A `LayoutStrategy` defines how to construct a layout tree from a stream of Vortex arrays. Strategies
can partition arrays by column, by row range, by size, or by any other scheme. Some strategies
compute pruning statistics, and others choose compression or buffering policies for leaf data.

For segment sinks that are locality-aware, such as a Vortex file, layout strategies can use sequence
IDs. These logical clocks let layouts parallelize writes and compression tasks while retaining
deterministic control over where segments are written.
