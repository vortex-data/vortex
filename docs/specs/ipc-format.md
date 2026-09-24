# IPC Format

The IPC format streams Vortex arrays between processes as a sequence of self-delimiting messages.
Arrays keep their encodings on the wire, so compressed data crosses the process boundary without
being decompressed. The format is implemented by the `vortex-ipc` crate, re-exported as
[`vortex::ipc`](https://docs.rs/vortex/latest/vortex/ipc/index.html).

:::{warning}
The IPC format is unstable. It is not covered by the [versioning](versioning.md) guarantees of the
[file format](file-format.md), and it has a single message version, `V0`, which readers require
exactly. If you want to persist data, please consider writing a Vortex file instead.
:::

## Streams

An IPC stream carries a single logical array as a sequence of chunks:

```
[DType message] [Array message] [Array message] ...
```

1. The first message is a `DTypeMessage` carrying the stream's data type.
2. Every following message is an `ArrayMessage` holding one chunk, whose data type must equal the
   stream's.
3. The stream ends at end-of-input after a complete message. There is no end-of-stream marker, so
   a stream with no chunks is a lone `DTypeMessage`.

## Message framing

Every message has the same framing:

```
<4 bytes>          u32 little-endian header length, H
<H bytes>          Message FlatBuffer
<body_size bytes>  message body
```

The `Message` FlatBuffer is the header, which contains the following fields:

- **`version`** (`MessageVersion`, default `V0`): the message format version. Readers reject any
  value other than `V0`.
- **`header`** (`MessageHeader` union): the message type, which also determines how to interpret
  the body.
- **`body_size`** (`uint64`): the exact length of the body, so a reader always knows where the next
  message starts.

Messages are written back-to-back, with no padding between or within them. This means that
header or body bytes are not guaranteed to land on an aligned offset. Thus, after receiving a
message, a reader copies each FlatBuffer and data buffer into appropriately aligned memory.

## Message types

### DTypeMessage

The body is a [`DType`](dtype-format.md) FlatBuffer (`root_type DType` in `dtype.fbs`). The
`DTypeMessage` table itself has no fields.

### ArrayMessage

The body is a serialized array. The header carries the information needed to decode it:

- **`row_count`** (`uint32`) — the length of the root array.
- **`encodings`** (`[string]`) — the encoding IDs the array references, such as `vortex.primitive`.

The body has the same shape as any other serialized array:

```
[buffer 0] [buffer 1] ... [buffer N-1] [Array FlatBuffer] [u32 little-endian FlatBuffer length]
```

A reader decodes the body from the end:

1. Read the trailing `u32` to find the length of the `Array` FlatBuffer that precedes it. The
   bytes before the FlatBuffer hold the data buffers.
2. Locate each data buffer from the `Array.buffers` table, in order. Buffer `i` starts `padding`
   bytes after the end of buffer `i - 1`, where the first buffer's predecessor ends at offset 0,
   and spans `length` bytes. Align it to 2<sup>`alignment_exponent`</sup> bytes.
3. Decode the `ArrayNode` tree from `Array.root`. Each node's `encoding` is an index into the
   message's `encodings` list, and its `buffers` are indices into the `Array.buffers` table.
4. Decode the root node with the stream's data type and `row_count`. Each encoding derives the data
   types and lengths of its children from its own metadata. Any `stats` present on a node are
   attached to the decoded array.

Each message is self-contained: its `encodings` list and buffer table apply only to that message.

The IPC writer never pads buffers, so `padding` is always 0. Readers should still honor it,
because [Vortex files](file-format.md) use the same array layout, which do pad. The writer always sets
`compression` to `None`. `LZ4` is reserved in the schema but not implemented; current readers
don't check the field.

The `ArrayNode` tree and its statistics are described in
[Serialization](../developer-guide/internals/serialization.md).

### BufferMessage

The body is a single raw byte buffer. `alignment_exponent` is the alignment the receiver must give
it, as a power of two. Readers reject exponents above 16 (64 KiB), since the value is untrusted.

The stream writers never emit a `BufferMessage`, and the stream readers reject one. It is available
only through the message-level API below.

## Rust API

The crate has three layers. Most callers only need the array-stream layer.

| Layer         | Write                                                          | Read                                                                                        |
|---------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Array streams | `stream::ArrayStreamIPC`, `iterator::ArrayIteratorIPC`         | `stream::AsyncIPCReader`, `iterator::SyncIPCReader`                                         |
| Messages      | `messages::AsyncMessageWriter`, `messages::SyncMessageWriter`  | `messages::AsyncMessageReader`, `messages::SyncMessageReader`, `messages::BufMessageReader` |
| Framing       | `messages::MessageEncoder`                                     | `messages::MessageDecoder`                                                                  |

`ArrayStreamIPC` and `ArrayIteratorIPC` are implemented for every `ArrayStream` and
`ArrayIterator`. They write the `DTypeMessage`, then one `ArrayMessage` per item. The iterator and
stream from `ArrayRef::to_array_iterator` and `ArrayRef::to_array_stream` yield one item per chunk
of a chunked array.

`AsyncIPCReader` and `SyncIPCReader` read such a stream back as an `ArrayStream` or `ArrayIterator`.
They fail if the first message is not a `DTypeMessage`, if a later message is not an
`ArrayMessage`, or if a decoded array's data type differs from the stream's.

`MessageDecoder` holds no IO. It reports the total number of bytes it needs to make progress, so
callers can drive it over any transport.

```rust
use std::io::Cursor;

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::iter::ArrayIteratorExt;
use vortex::buffer::buffer;
use vortex::error::VortexResult;
use vortex::ipc::iterator::ArrayIteratorIPC;
use vortex::ipc::iterator::SyncIPCReader;
use vortex::session::VortexSession;

fn round_trip() -> VortexResult<()> {
    let session = VortexSession::default();
    let array = buffer![1i32, 2, 3].into_array();

    let bytes = array.to_array_iterator().write_ipc(Vec::new(), &session)?;

    let reader = SyncIPCReader::try_new(Cursor::new(bytes), &session)?;
    assert_eq!(reader.read_all()?.len(), 3);
    Ok(())
}
```

Decoding resolves encoding IDs through the session, so the reader's session must register every
encoding the writer used.

## Limitations

- **No zero-copy reads.** Buffers are unpadded on the wire, so the reader copies them into aligned
  memory.
- **No state shared across messages.** Each message repeats its own encoding list, and data such
  as a dictionary shared between chunks is sent again with every chunk.
- **Row count limit.** An `ArrayMessage` holds at most `u32::MAX` rows. Larger arrays must be split
  into chunks.
- **No end-of-stream marker.** A stream cut exactly at a message boundary is indistinguishable from
  a complete one. Transports that can drop data must detect truncation themselves.

## FlatBuffer definition

:::{literalinclude} ../../vortex-ipc/flatbuffers/vortex-serde/message.fbs
:::
