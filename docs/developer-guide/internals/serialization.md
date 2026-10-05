# Serialization

Vortex stores array-tree metadata in FlatBuffers and data buffers separately, with alignment that
enables zero-copy reads. A serializer can reuse an array's buffers while adapting its metadata or
children to a supported wire format. The in-memory array and its serialized representation can
evolve independently, as described in the [versioning design](../../specs/versioning/design.md).
Padding in Vortex files allows segments to be memory-mapped with the required alignment.

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

`SerializedArray` holds the serialized tree and buffer handles. Decoding resolves the stored wire ID
through the session's plugin registry and calls the plugin's `deserialize` method with the metadata,
buffers, and children. The plugin validates that wire format and constructs an array supported by
the current implementation.

## IPC Format

The IPC format streams serialized arrays between processes: a dtype message, then one array
message per chunk, each carrying a serialized array as its body. IPC writes buffers without
padding, so unlike file reads, IPC reads copy buffers into aligned memory. The
[IPC format specification](../../specs/ipc-format.md) defines the framing and message types.

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

All FlatBuffers in Vortex are aligned to 8 bytes. Each schema definition lives in the crate that
owns the types it describes -- arrays and dtypes in `vortex-array`, layouts in `vortex-layout`, the
file footer in `vortex-file`, and IPC messages in `vortex-ipc` -- next to the generated Rust
bindings, which `build.rs` compiles into `OUT_DIR`. The read/write traits they all share live in
`vortex_array::flatbuffers`.

## Zero-Copy Design

The alignment and padding system allows serialized buffers to be used directly in in-memory arrays
without copying. When a segment is read from disk or received over the network, the I/O subsystem
allocates an aligned buffer matching the segment's alignment requirement. The resulting buffer
handle can be used directly by the array without reallocating or copying the data.

Reusing buffers does not require the reader's array tree to have the same structure as the serialized
tree. A plugin can adapt a historical format while retaining its data buffers.
