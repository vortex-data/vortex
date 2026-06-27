<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# WASM encodings in the Vortex file format

Status: **draft / in-progress implementation**

## Motivation

Vortex encodings are compiled into the reader. Adding a new encoding means shipping a new
release of every reader (Rust, Python, Java, the WASM web reader, DuckDB/DataFusion
integrations, ...). This makes it expensive to:

- experiment with new compression schemes,
- ship dataset-specific or domain-specific encodings,
- read files written by a *newer* writer than the reader.

The goal of this work is to embed the *decoder* for an encoding **inside the file** as a
sandboxed WebAssembly module. A reader that understands the `WasmLayout` can then decode arrays
written with an encoding it has never seen, by running the embedded WASM kernel against the
serialized array and the host's existing decode machinery.

This document describes the on-disk format, the host/guest ABI, and the crate layout.

## Overview

```
┌──────────────────────── Vortex file ─────────────────────────┐
│ MAGIC                                                          │
│ … data segments (child layout, written normally) …            │
│ … WASM kernel segment (one .wasm blob, written at EOF) …      │   ← split off to end
│ DType / Layout / Statistics / Footer flatbuffers              │
│ Postscript + EOF                                              │
└───────────────────────────────────────────────────────────────┘
```

A `WasmLayout` node in the layout tree holds:

1. **child layouts** — the underlying, *physically encoded* arrays. These are read with the
   normal layout/segment machinery and may themselves be `Flat`, `Chunked`, etc.
2. a **segment id** pointing at the embedded **WASM kernel** (the `.wasm` bytes).
3. metadata: the guest encoding id/version and the output `DType`.

At read time the reader:

1. loads the kernel segment, instantiates it once in an embedded WASM VM (`wasmi`), caching the
   compiled `Module` per kernel digest;
2. obtains the serialized array bytes for the wasm-encoded array (flatbuffer header + buffers)
   from the child layout;
3. copies those bytes into the guest's linear memory and calls the guest's exported
   `vx_decode` entrypoint;
4. the guest **parses the array flatbuffer header itself** (using a stripped-down
   `vortex-flatbuffers` compiled into the module, *without the rest of Vortex*), reads its own
   metadata and buffers, and, whenever it needs a decoded child array, calls back into the host
   import `vx_decode_child`;
5. the host satisfies `vx_decode_child` by decoding that child through the `VortexSession`
   (reusing all of Vortex's existing encodings) and copying the resulting **canonical** array
   back into guest memory in a compact wire format;
6. the guest produces a canonical output (buffers + validity + children) and returns it to the
   host, which reconstructs a Vortex `Canonical` array.

The output is expressed in Vortex canonical encodings (`Primitive`, `Bool`, `VarBinView`,
`Struct`, ...). Arrow C Data Interface is an alternative we keep open as a fallback (see
[Output format](#output-format)) but the canonical-buffer wire format is the primary path.

## Crates

Two new crates, kept out of the core dependency graph so that `wasmi` never leaks into
`vortex-array`/`vortex-layout`:

### `vortex-wasm-guest` (the guest SDK)

A small, `no_std`-friendly crate that an encoding author links against when building their
decoder to `wasm32-unknown-unknown`. It depends **only** on `vortex-flatbuffers` (with the
`array` feature), `vortex-error`, and `vortex-buffer` — never on `vortex-array` or the rest of
Vortex. It provides:

- the host/guest ABI (exported entrypoints, imported host functions);
- `ArrayHeader`, a thin wrapper over the generated `vortex-flatbuffers` `Array`/`ArrayNode`
  types for reading the serialized array header (encoding id, metadata bytes, buffer ranges,
  children);
- a `CanonicalMessage` reader/writer for the wire format used to pass canonical arrays across
  the boundary in both directions;
- a `WasmEncoding` trait plus an `export_wasm_encoding!` macro that wires up `vx_alloc` and
  `vx_decode` exports around a user-supplied `decode` function.

### `vortex-wasm` (the host)

Depends on `vortex-layout`, `vortex-array`, `vortex-session`, and `wasmi`. It provides:

- `WasmKernel` — an instantiated, reusable wrapper around a `wasmi::Module` and the host import
  table, exposing `decode(serialized_array, &dyn HostDecoder) -> Canonical`;
- the `CanonicalMessage` host-side serializer/parser (mirror of the guest's);
- `WasmLayout` + `WasmLayoutEncoding` + `WasmLayoutMetadata`, the layout `VTable`
  implementation;
- `WasmReader`, the `LayoutReader` that drives the kernel;
- `WasmLayoutStrategy`, the writer that wraps a child strategy and appends the kernel segment at
  EOF;
- `register_wasm_layout(session)`, registering the encoding so files can be read.

## On-disk layout (`WasmLayout`)

`encoding = "vortex.wasm"`. Stored in the layout flatbuffer like any other layout:

| field      | meaning                                                                 |
|------------|------------------------------------------------------------------------|
| `row_count`| rows produced by the decoded output                                    |
| `metadata` | prost `WasmLayoutMetadata` (see below)                                  |
| `children` | `[data_layout]` — the encoded array(s); index 0 is the transparent data |
| `segments` | `[kernel_segment_id]` — the embedded `.wasm` blob                      |

```protobuf
message WasmLayoutMetadata {
  string  encoding_id = 1;   // guest encoding id, e.g. "acme.delta"
  uint32  abi_version = 2;   // host/guest ABI version
  bytes   output_dtype = 3;  // serialized DType of the decoded output (optional;
                             // falls back to the layout dtype if absent)
}
```

The kernel itself is content-addressed: identical kernels across many `WasmLayout` nodes in one
file should share a single segment (a writer-side dedup keyed on the blob digest). For the first
cut each `WasmLayout` references one kernel segment; dedup is a follow-up.

### Writing the kernel at EOF with `SequencePointer::split_off`

`LayoutStrategy::write_stream` receives an `eof: SequencePointer` guaranteed to sort after every
chunk in the stream. To force the kernel to the very end of the file, the strategy takes a
sequence id from `eof` (via `split_off`) and uses it for the kernel segment write, while the
child data is written with the normal in-stream sequence ids. Because `SegmentSink::write`
calls `SequenceId::collapse().await`, the kernel's segment bytes are flushed only after all
earlier (data) segments — placing the `.wasm` blob at the end of the file. As the trait docs
require, the strategy awaits the child write and the kernel write **concurrently** to avoid the
EOF-deadlock.

## Host / guest ABI (`abi_version = 1`)

All integers little-endian. The single shared linear memory is exported by the guest as
`"memory"`.

### Guest exports (host calls these)

- `vx_alloc(len: i32) -> i32`
  Allocate `len` bytes and return the offset. The host uses this to place inputs and
  host-decoded children into guest memory.
- `vx_decode(input_ptr: i32, input_len: i32) -> i32`
  Decode the serialized array at `[input_ptr, input_ptr+input_len)`. Returns the offset of a
  length-prefixed `CanonicalMessage` (`[u32 len][bytes…]`). A negative return value is an error
  code.

### Host imports (guest calls these), module `"vortex_host"`

- `vx_decode_child(node_index: i32, out_ptr: i32) -> i32`
  Ask the host to decode the child array at `node_index` (an index into the serialized array
  header's `children`, in document order). The host decodes it through the session, encodes the
  result as a `CanonicalMessage`, allocates space in guest memory via `vx_alloc`, copies the
  message in, and writes its `(offset: u32, len: u32)` pair to the 8 bytes at `out_ptr`. Returns
  0 on success, negative on error.
- `vx_host_log(ptr: i32, len: i32)` (optional, debug only)
  Log a UTF-8 string from guest memory.

### `CanonicalMessage` wire format

A single contiguous, self-describing blob with inline buffer bytes (so one copy moves an entire
array across the boundary):

```
CanonicalMessage:
  u8   kind            // 0 Null, 1 Bool, 2 Primitive, 3 VarBinView, 4 Struct  (extensible)
  u8   ptype           // PType discriminant, valid when kind == Primitive
  u8   validity        // 0 NonNullable, 1 AllValid, 2 AllInvalid, 3 Bitmap
  u8   _pad
  u64  length          // logical element count
  u32  nbuffers
  u32  nchildren
  [nbuffers]  { u64 len; u8 alignment_exp; u8[7] pad; bytes[len] }   // buffer table + inline data
  [nchildren] CanonicalMessage                                       // recursive
  // when validity == Bitmap, the bitmap is buffer index 0 by convention
```

`kind`/`ptype`/`validity` map directly onto the Vortex `Canonical` enum and `PType`. The first
implementation supports `Null`, `Bool`, and `Primitive` (with all four validity variants);
`VarBinView` and `Struct` follow. Both sides share the *same byte format*; the guest SDK and the
host each implement an encoder and a decoder for it.

## Reader flow (`WasmReader`)

`WasmReader` builds one child reader for the data layout (propagating
`LayoutReaderContext`). Its `projection_evaluation`:

1. asks the data child reader for the *raw serialized array* of the wasm-encoded node. The data
   child is a `Flat` layout, so the reader requests its segment and forms the
   `SerializedArray` bytes (header + buffers) — but **does not** decode the top node (its
   encoding is the unknown wasm encoding). We add a small `LayoutReader` capability /
   `reader_context` channel to fetch raw serialized bytes rather than a decoded array.
2. runs `WasmKernel::decode(bytes, host_decoder)` on a blocking pool (`spawn` blocking, since
   `wasmi` execution is synchronous and potentially long-running), where `host_decoder`
   implements `vx_decode_child` by decoding the requested child node via
   `SerializedArray::child(...).decode(session)`.
3. wraps the resulting `Canonical` into an `ArrayRef`, applies the requested projection
   expression and mask, and returns it.

Filtering/pruning fall back to the generic "decode then evaluate" path in v1; pushdown into the
WASM kernel is out of scope.

### Why the guest parses the header

The requirement that the guest parse the array flatbuffer header *without the rest of Vortex* is
satisfied by `vortex-flatbuffers`: it only depends on `flatbuffers`, `vortex-buffer`, and
`vortex-error`, and builds for `wasm32-unknown-unknown`. The guest therefore reads `encoding`,
`metadata`, the buffer table, and `children` straight from the flatbuffer, giving the encoding
full control over how it interprets its own metadata and buffers — exactly mirroring what a
native Vortex `VTable::deserialize` would do, but sandboxed.

## Output format

Primary: **Vortex canonical encodings**, transported via `CanonicalMessage`. This keeps the
output in Vortex's native representation with zero extra dependencies in the guest.

Fallback: **Arrow C Data Interface**. `arrow-rs` already implements the C Data Interface and
Vortex has `FromArrowArray`. A guest could instead populate `FFI_ArrowArray`/`FFI_ArrowSchema`
structs in linear memory and return their pointers; the host would translate the wasm-relative
pointers, import via `arrow::ffi`, and convert with `ArrayRef::from_arrow`. We keep the ABI's
`kind` byte extensible so an Arrow-FFI return mode can be added without breaking v1. The
canonical path is preferred because it avoids re-deriving Arrow buffer layouts and pointer
fix-ups inside the sandbox.

## Security & resource limits

`wasmi` is a sandboxed interpreter: no host memory access beyond the explicit imports, no
syscalls. We additionally:

- set fuel/step limits per `vx_decode` call to bound runtime;
- cap guest linear-memory growth;
- validate every guest-returned pointer/length against the current memory size before reading;
- treat any guest trap or malformed `CanonicalMessage` as a decode error (never a host panic).

The kernel is untrusted data from the file, exactly like array bytes; correctness bugs in a
kernel can only corrupt *that array's* values, never host memory.

## Implementation phases

1. **Foundation (in progress):** crate scaffolding, ABI constants, `CanonicalMessage`
   (host+guest) for Null/Bool/Primitive, `WasmKernel` over `wasmi` with `vx_decode_child`, a
   `.wat`-based host test exercising the full ABI round trip.
2. **Layout:** `WasmLayout`/`WasmLayoutEncoding`/metadata, `WasmReader`, registration; raw
   serialized-bytes access from the data child.
3. **Writer:** `WasmLayoutStrategy` writing the kernel at EOF via `split_off`; round-trip a file
   end-to-end with a real Rust guest example.
4. **Breadth:** `VarBinView`/`Struct` in the wire format, kernel dedup, fuel limits, Arrow-FFI
   return mode, pushdown.

## Open questions

- **Raw-bytes access:** the cleanest way for `WasmReader` to obtain the *undecoded* serialized
  array from a child layout. Options: a new optional `LayoutReader` method, a `reader_context`
  flag that makes `FlatReader` hand back bytes, or a dedicated `RawFlatLayout`. Leaning toward a
  narrow trait method.
- **Kernel caching key:** digest of the blob vs. segment id; cross-file caching in a session.
- **Async vs. blocking:** running `wasmi` on the IO runtime's blocking pool vs. a dedicated
  decode pool.
