# Serialization

Vortex uses the same binary representation for arrays in memory, on disk, and over the wire.
Metadata is stored in FlatBuffers for O(1) field access without parsing, and data buffers are
stored separately with alignment guarantees that enable zero-copy reads. Appropriate padding is
written into Vortex files to ensure that segments can be memory-mapped with correct alignment.

## Array Serialization

A serialized array consists of two parts: a FlatBuffer describing the array tree, and a
sequence of data buffers.

The FlatBuffer contains an `ArrayNode` tree where each node records:
- The array ID (as an interned u16 index).
- Array-specific metadata bytes.
- References to child `ArrayNode`s.
- Indices into the buffer table.
- Optional statistics (min, max, null count, sort order, etc.).

The buffer table records each buffer's padding, alignment exponent, compression, and length.
Buffers are laid out contiguously after the metadata, with padding inserted to satisfy each
buffer's alignment requirement.

On the wire, a serialized array is:

```
[padding] [buffer 0] [padding] [buffer 1] ... [flatbuffer] [u32 flatbuffer length]
```

Deserialization constructs an `ArrayParts` value that holds the FlatBuffer and buffer handles
without copying. The array is then decoded by resolving the array ID through the session's
registry and calling `build()` on the corresponding vtable.

## IPC Format

The IPC format wraps serialized arrays in a message-oriented protocol for streaming between
processes. Each message consists of:

```
[u32 flatbuffer length] [flatbuffer Message] [body bytes]
```

Three message types are defined:

- **ArrayMessage** -- a serialized array with its row count and an encoding context that maps
  dictionary indices to encoding IDs.
- **BufferMessage** -- a raw buffer with an alignment exponent, used for transferring individual
  segments.
- **DTypeMessage** -- a serialized dtype, used to communicate the schema before data transfer
  begins.

The IPC format is used both for inter-process communication and as the wire protocol for remote
source execution in the [Scan API](/concepts/scanning).

:::{note}
The IPC format is unstable and subject to change. It does not yet support shared arrays (e.g.
a dictionary shared across multiple chunked arrays), which limits its efficiency for certain
workloads. This is an area of active development.
:::

## Segment Storage

In a Vortex file, data buffers are stored as segments -- contiguous byte ranges at known offsets.
Each segment is described by a `SegmentSpec` containing:
- **offset** -- byte position from the start of the file.
- **length** -- size in bytes.
- **alignment** -- required memory alignment (as a power-of-two exponent).

Layouts reference segments by `SegmentId`, which is an index into the footer's segment table.
This indirection allows the same layout tree to be backed by different segment sources (local
file, object store, in-memory cache, etc.) without changing the layout structure. 

## File Footer

The file footer is the entry point for reading a Vortex file. It is read from the end of the
file and contains everything needed to reconstruct the layout tree and locate data segments.

The last 8 bytes of the file contain:
- A version number (2 bytes).
- The postscript length (2 bytes).
- A magic number (4 bytes).

The postscript locates four regions by offset and length:
- **DType** -- the schema, stored as a FlatBuffer (optional if embedded in the layout).
- **Layout** -- the layout tree, stored as a FlatBuffer.
- **Statistics** -- per-column file-level statistics (optional).
- **Footer** -- dictionaries of encoding IDs, layout IDs, segment specs, and compression
  configs.

The layout FlatBuffer is a tree of `Layout` nodes, each containing an encoding ID, row count,
metadata, child layouts, and segment indices. This tree is deserialized and bound to a segment
source to create a `LayoutReader` that can lazily fetch data on demand.

## FlatBuffers

Vortex uses FlatBuffers rather than Protocol Buffers or a custom binary format because
FlatBuffers support O(1) random access into the serialized data without parsing the entire
message. This is important for wide schemas where only a few columns are accessed per query --
the reader can jump directly to the relevant layout node without deserializing the rest of the
footer.

All FlatBuffers in Vortex are aligned to 8 bytes. Schema definitions live in the
`vortex-flatbuffers` crate and cover arrays, layouts, the file footer, and IPC messages.

## Zero-Copy Design

The alignment and padding system is designed so that serialized buffers can be used directly
as in-memory arrays without copying. When a segment is read from disk or received over the
network, the I/O subsystem allocates an aligned buffer matching the segment's alignment
requirement. The resulting buffer handle can be used directly by the array without
reallocating or copying the data.

This property holds across all three contexts: in-memory arrays, on-disk file segments, and
over-the-wire IPC messages all use the same layout and alignment conventions.
