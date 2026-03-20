# Design: `execute_into_builder` + `CanonicalWriter`

## The Problem

Canonicalizing a `Chunked<FoR<BitPacked<i32>>>` with N chunks does **2N data copies**:

1. Each chunk decompresses into a throwaway `PrimitiveArray` (copy #1 — new allocation)
2. That array is `memcpy`'d into the builder's output buffer (copy #2)

Peak memory is **~2.25x** the final output: compressed source + output buffer + one intermediate
chunk buffer all live simultaneously. The compressed source cannot be freed incrementally because
the entire `ChunkedArray` is held by a single `Arc`.

### Root cause 1: No buffer ownership during decompression

`VTable::execute` takes `&Self::Array` (shared ref). Arrays are `Arc`-wrapped, and the API gives
no way to move buffers out — even when the refcount is 1. Every decompression must borrow the
compressed buffer, allocate a fresh output, and write into it.

ALP already works around this with a clone hack that doesn't help:

```rust
fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    // TODO(joe): take by value
    Ok(ExecutionStep::Done(execute_decompress(array.clone(), ctx)?.into_array()))
}
```

The clone bumps the Arc refcount, so `into_buffer_mut()` always falls back to copying.

### Root cause 2: Chunked reassembly double-copy

`ChunkedArray::_canonicalize` creates a builder then appends each decompressed chunk. Per chunk:
decompress → allocate intermediate → copy into builder → drop intermediate. The intermediate
buffer is pure waste.

Today's code also has three separate branches for leaf/struct/list:

```rust
Ok(match array.dtype() {
    DType::Struct(..) => Canonical::Struct(pack_struct_chunks(..)),
    DType::List(..)   => Canonical::List(swizzle_list_chunks(..)),
    _                 => { /* builder + append loop */ }
})
```

### Root cause 3: Peak memory amplification

During chunked canonicalization, three memory regions are live simultaneously:

| Region | Lifetime | Size |
|--------|----------|------|
| Original `ChunkedArray` (compressed) | Entire execution (pinned by `Arc`) | `compressed_total` |
| Builder output buffer | Growing, pre-allocated at total capacity | `decompressed_total` |
| Current chunk's decompressed form | Per-chunk temporary | `decompressed_chunk` |

For typical compression ratios (4-10x), peak is ~2.1-2.25x of the final decompressed size.

### What we want

- **One data copy** per chunk: decompress directly into the final output buffer
- **One allocation** for the entire column: pre-sized at the top level
- **Uniform recursive dispatch** for all types: no special-case branches for struct/list/leaf

---

## The Design

Two new abstractions, used together:

1. **`CanonicalWriter`** — a pre-allocated, typed accumulator that arrays push data into.
   One variant per canonical type. Composite writers (Struct, List) hold child writers,
   forming a recursive tree that mirrors the DType tree.

2. **`execute_into_builder`** — a vtable method every encoding implements. Instead of
   returning a new `ArrayRef`, the encoding writes its decompressed data directly into
   the caller's `CanonicalWriter`.

### Core API

```rust
/// A pre-allocated writer for building a canonical array.
/// Each canonical type gets its own implementation.
pub trait CanonicalWriter: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn dtype(&self) -> &DType;
    fn len(&self) -> usize;

    /// Default entry: dispatches to the array's vtable.
    fn write(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        array.execute_into_builder(self, ctx)
    }

    /// Consume and produce the final Canonical.
    fn finish(self: Box<Self>, ctx: &mut ExecutionCtx) -> VortexResult<Canonical>;
}

/// Factory: create a writer for the given dtype and exact row count.
/// All buffers are allocated once here — no reallocation during writes.
pub fn canonical_writer(dtype: &DType, len: usize) -> Box<dyn CanonicalWriter>;
```

```rust
/// The vtable method that encodings implement.
trait ExecuteIntoBuilderVTable {
    fn execute_into_builder(
        &self,
        array: &ArrayRef,
        writer: &mut dyn CanonicalWriter,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;
}

/// Default: fall back to existing execute, then write result into writer.
/// Encodings opt into the optimized path incrementally.
fn execute_into_builder_default(
    array: &ArrayRef,
    writer: &mut dyn CanonicalWriter,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical = array.clone().execute::<Canonical>(ctx)?;
    writer.write(&canonical.into_array(), ctx)
}
```

### Unified buffer primitive: `get_init_bytes`

Every data-writing operation an encoding needs is a pattern on one primitive:

```rust
impl PrimitiveWriter {
    /// Reserve `n` elements of uninitialized memory in the output buffer.
    /// Returns a handle the caller writes into however it wants.
    pub fn get_init_bytes(&mut self, n: usize) -> UninitRange<'_>;
}
```

| "Operation" | How it's done |
|---|---|
| Decode into buffer | `get_init_bytes(n)` -> `decode_into(range)` |
| memcpy | `get_init_bytes(n)` -> `copy_from_slice` |
| Fill with scalar | `get_init_bytes(n)` -> loop fill |
| Child writes, then transform in-place | child calls `get_init_bytes(n)`, parent iterates range applying `f` |
| Scatter at indices | `get_init_bytes(n)` -> fill default -> write at indices |
| Apply patches | `range.set_value(idx, val)` on same range |

No separate methods. The encoding's vtable impl calls `get_init_bytes` and does whatever it
wants with the bytes. `map_last_n(n, f)` and `append_validity(mask)` are convenience methods
on top of the same buffer.

---

## Writer Types

### PrimitiveWriter

Wraps a `PrimitiveBuilder<T>` behind a type-erased layer (necessary because the writer is
created from a `DType` at runtime, not a compile-time `T`).

```rust
pub struct PrimitiveWriter {
    dtype: DType,
    inner: Box<dyn PrimitiveWriterInner>,  // TypedPrimitiveWriter<T>
}

struct TypedPrimitiveWriter<T: NativePType> {
    builder: PrimitiveBuilder<T>,  // capacity == total len, never reallocs
}
```

Encodings downcast via `writer.as_any_mut().downcast_mut::<PrimitiveWriter>()`, then access
the typed inner to call `uninit_range()` / `get_init_bytes()`.

`finish()`: freezes the buffer -> `Canonical::Primitive`.

### BoolWriter

Pre-allocated `BitBufferMut` + `LazyBitBufferBuilder` for validity. Size = `ceil(len / 8)`.

`finish()`: freeze -> `Canonical::Bool`.

### VarBinViewWriter

Views buffer is pre-allocated (`len * 16` bytes). Data buffers are zero-copy attached per
chunk via `push_buffer_and_adjusted_views()` — the string data is never copied, only the
16-byte view structs are.

Methods accessed via downcast:
- `push_views(views, buffers)` — push view structs + stash data buffers zero-copy
- `push_bytes(bytes)` — append a single string/binary value

`finish()`: assemble `VarBinViewArray`.

### StructWriter — Recursive

Holds `Vec<Box<dyn CanonicalWriter>>`, one per field. Each field writer is pre-allocated to
`len` rows and can be any writer type.

```rust
pub struct StructWriter {
    field_writers: Vec<Box<dyn CanonicalWriter>>,
    nulls: LazyBitBufferBuilder,
    struct_dtype: StructFields,
    nullability: Nullability,
    rows_written: usize,
}
```

During `write()`: `StructEncoding` decomposes the struct and pushes each field into its
corresponding child writer. Recursion bottoms out at leaf writers.

During `finish()`: each field writer finishes independently -> `Canonical::Struct`.

### ListWriter — Deferred Element Canonicalization

The canonical list form is `ListViewArray` (offsets + sizes + elements). Key design:
**defer element canonicalization to `finish()`**.

```rust
pub struct ListWriter {
    offsets: BufferMut<u64>,         // pre-allocated to `len`
    sizes: BufferMut<u64>,           // pre-allocated to `len`
    nulls: LazyBitBufferBuilder,

    element_chunks: Vec<ArrayRef>,   // stashed, NOT canonicalized yet
    total_elements: usize,

    elem_dtype: DType,
    nullability: Nullability,
}
```

During `write()`: `ListViewEncoding` pushes shifted offsets/sizes and stashes element
`ArrayRef`s. **No recursion, no element data copy.**

During `finish()`: creates one child `CanonicalWriter` for `elem_dtype` sized to
`total_elements`, flushes all stashed element chunks through it, then assembles the
`ListViewArray`.

This is the key insight: **stash-then-recurse**. The offsets/sizes are just integer
arithmetic (cheap). Element canonicalization happens once at `finish()` time, through a
single writer that may itself be a PrimitiveWriter, VarBinViewWriter, or another ListWriter
for nested lists.

### Other Writers

| Writer | Accumulates during `write()` | Does during `finish()` |
|---|---|---|
| **NullWriter** | Increments counter | `NullArray::new(len)` |
| **DecimalWriter** | Same as PrimitiveWriter | Wrap as `DecimalArray` |
| **FixedSizeListWriter** | Stashes element chunks (like ListWriter, no offsets/sizes) | Flush through child writer |
| **ExtensionWriter** | Delegates to inner storage writer | Wrap in `ExtensionArray` |

---

## How Encodings Use the Writer

Every encoding falls into exactly one of three patterns:

### Pattern 1: Direct Decode

The encoding owns compressed bytes and decodes straight into `get_init_bytes`. No children to
execute first.

```
encoding.execute_into_builder(writer):
    range = writer.get_init_bytes(len)
    decode_compressed_bytes_into(range)
    apply_patches(range)
    range.finish()
```

| Encoding | Notes |
|---|---|
| **BitPacking** | `unpack_into_primitive_builder()` already exists — writes to `UninitRange` via FastLanes `unchecked_unpack()`. Patches applied to same range. |
| **FoR (fused)** | `FoRStrategy` applies reference during unpack — single fused pass. Patches shifted by `wrapping_add(ref)`. |
| **Pco** | Decompress pages directly into range. Only needed pages (lazy). Size from metadata. |
| **Zstd (primitive)** | Decompress frames into range. Size from metadata. Aligned allocation. |
| **Sequence** | Compute `base + i * multiplier` directly into range. No children, no input buffers. |
 | **Constant** | `get_init_bytes(n)` + fill with scalar. Already has optimized `append_to_builder`. |
| **ByteBool** | Convert byte->bit directly into `BitBufferMut`. |

### Pattern 2: Execute Child -> Transform In-Place

The child writes into the writer's buffer, then the parent encoding mutates those same bytes.
The buffer flows down through the child and back up — same allocation throughout.

```
encoding.execute_into_builder(writer):
    self.encoded().execute_into_builder(writer, ctx)    // child writes
    writer.map_last_n(len, |v| transform(v))            // parent transforms in-place
    apply_patches(writer, self.patches(), ctx)           // overwrite exceptions
```

| Encoding | Transform | Patches |
|---|---|---|
| **FoR (non-fused)** | `\|v\| v.wrapping_add(reference)` | From inner child, shifted by reference |
| **ALP** | `\|int\| alp_decode(int, e, f)` — int->float transmute via `10^f * 10^-e` formula | Float exceptions that didn't round-trip through ALP |
| **ZigZag** | `\|u\| zigzag_decode(u)` — unsigned->signed bit-flip | None |
| **DecimalByteParts** | None (integer bits ARE the decimal mantissa) | None |

**How ALP works with the builder:** The encoded integers (e.g. `FoR<BitPacked<i64>>`) flow
through `execute_into_builder` all the way down to BitPacking, which decodes into the
PrimitiveWriter's buffer. Then FoR adds the reference in-place. Then ALP transmutes
`i64->f64` in-place. Then patches overwrite specific indices. One buffer, three in-place
passes, zero intermediate allocations.

**Key writer requirement:** `map_last_n(n, f)` — iterate the last `n` values in the buffer
applying `f`. This is how the "pass buffer down, transform on the way back up" pattern works.

### Pattern 3: Execute Children -> Combine

Multiple children must be materialized first (into temp arrays), then combined into the
writer's buffer. The temp arrays are unavoidable (need two+ sources to merge), but the
**output** goes directly into the final buffer.

```
encoding.execute_into_builder(writer):
    let child_a = self.child_a().execute::<PrimitiveArray>(ctx)?
    let child_b = self.child_b().execute::<PrimitiveArray>(ctx)?
    let range = writer.get_init_bytes(len)
    combine_into(child_a, child_b, range)
    range.finish()
```

| Encoding | Children | Combine | Patches |
|---|---|---|---|
| **ALP-RD** | left_parts (dict-decode), right_parts | `from_bits((left << shift) \| right)` | On left codes |
| **Delta** | bases (small), deltas | Undelta + untranspose per 1024-element chunk | None directly |
| **RLE (FastLanes)** | values, indices, offsets | Dictionary scatter: `output[i] = values[indices[i]]` | None |
| **RunEnd** | ends, values | Fill runs: `push_n(value, run_length)` | None |
| **Sparse** | patch indices, patch values | Fill default + scatter at indices | IS patches |
| **DateTimeParts** | days, seconds, subseconds | `days * 86400 * div + seconds * div + subseconds` | None |
| **Dict** | codes, values | Gather: `output[i] = values[codes[i]]` | None |
| **FSST** | codes, symbol table | Decompress code sequences -> push string views | None |
| **Zstd (strings)** | frames | Decompress frames, parse length-prefixed strings | None |

---

## Container Encodings (No Data Transform)

These don't decode — they restructure and delegate:

| Encoding | Strategy |
|---|---|
| **Chunked** | `for chunk in chunks { chunk.execute_into_builder(writer) }` — peel and recurse |
| **Slice** | `inner.slice(range).execute_into_builder(writer)` |
| **Filter** | Materialize mask, write selected rows into writer |
| **Masked** | Delegate child + apply validity mask to writer |

Chunked is the most important: it peels into individual chunks, each of which calls its own
`execute_into_builder`. This is how `Chunked<FoR<BitPacked<i32>>>` resolves to direct
BitPacking decode into the final buffer — no intermediate arrays.

---

## Patches

Patches are always applied to the same `UninitRange` returned by `get_init_bytes`. They are
a scatter-write on the range:

```rust
fn apply_patches(range: &mut UninitRange<T>, patches: &Patches, ctx: &mut ExecutionCtx) {
    let indices = patches.indices().execute::<PrimitiveArray>(ctx)?;
    let values = patches.values().execute::<PrimitiveArray>(ctx)?;
    for (idx, val) in indices.iter().zip(values.iter()) {
        range.set_value(idx, val);
    }
}
```

No extra buffer. Same range, same memory.

- **FoR + BitPacking fused**: patches get reference added via `wrapping_add(ref)` during apply
- **ALP**: patches contain true float values that didn't round-trip. Overwrite after transmute.
- **ALP-RD**: patches on left_parts codes, applied to temp left array before combination.

---

## How Chunked + List Works

### The entry point (replaces `_canonicalize`)

```rust
fn canonicalize_chunked(
    array: &ChunkedArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    if array.nchunks() == 0 { return Ok(Canonical::empty(array.dtype())); }
    if array.nchunks() == 1 { return array.chunks()[0].clone().execute::<Canonical>(ctx); }

    // One call. Works for every dtype.
    let mut writer = canonical_writer(array.dtype(), array.len());
    for chunk in array.chunks() {
        writer.write(chunk, ctx)?;
    }
    writer.finish(ctx)
}
```

One function. All types. No special-case branches for struct/list/leaf.

### How ListWriter works

**During `write()` — lightweight sizing pass (integer arithmetic only):**

Each chunk is executed to `ListViewArray`. The writer:
1. Reads offsets/sizes, shifts offsets by `total_elements` so far
2. Stashes the element `ArrayRef` (zero-copy, just an `Arc::clone`)
3. Accumulates validity

No element data is touched. Total element count is accumulated as a side effect.

**During `finish()` — recursive canonicalization with known sizes:**

1. Creates a child `CanonicalWriter` for `elem_dtype`, sized to `total_elements` (now known)
2. Flushes all stashed element chunks through the child writer
3. Child writer finishes -> canonical elements
4. Assembles `ListViewArray` from offsets + sizes + canonical elements

### Nested lists: `Chunked<List<List<i32>>>`

Each nesting level does the same thing:
- `write()` accumulates offsets/sizes + stashes element refs
- `finish()` creates the next level's writer with the now-known size and recurses

The pattern bottoms out at a leaf writer where `write_chunk` does the actual data copy.

```
outer ListWriter.finish():
  total_elements = 5 inner lists
  elem_writer = ListWriter(List<i32>, 5)     <- inner ListWriter

  for each stashed element chunk:            <- each is a List<i32> or Chunked<List<i32>>
    elem_chunk.execute_into_builder(inner_writer)
    -> push inner offsets/sizes, stash i32 elements

  inner_writer.finish():
    total_leaf_elements = 12
    leaf_writer = PrimitiveWriter(i32, 12)   <- leaf PrimitiveWriter

    for each stashed i32 chunk:
      chunk.execute_into_builder(leaf_writer)
      -> decode directly into BufferMut<i32>  <- ONLY data copy

    leaf_writer.finish() -> Canonical::Primitive
    -> Canonical::List(inner)
  -> Canonical::List(outer)
```

One data copy total. Offsets/sizes are O(n_rows) integer arithmetic.

### The nested chunked case: `Chunked<List<Chunked<List<i32>>>>`

When a ListWriter's stashed elements are themselves `ChunkedArray`s, the Chunked encoding's
`execute_into_builder` peels them:

```
outer_writer.finish():
  elem_writer = ListWriter(List<i32>, total)

  for elem_chunk in stashed:               // each is Chunked<List<i32>>
    elem_chunk.execute_into_builder(elem_writer)
    |
    |  ChunkedEncoding peels:
    |  for sub in chunks:                   // each sub is List<i32>
    |    sub.execute_into_builder(elem_writer)
    |    -> push offsets/sizes, stash i32 elements

  elem_writer.finish():
    leaf_writer = PrimitiveWriter(i32, total_leaf)
    for leaf_chunk in stashed:
      leaf_chunk.execute_into_builder(leaf_writer)
      -> decode into buffer                  <- ONLY data copy
```

No level ever allocates an intermediate canonical array. The Chunked peeling is transparent
— the writer never knows about it.

### VarBinView in the list case: `Chunked<List<Utf8>>`

Elements are `Chunked<VarBinView>` (or `Chunked<FSST>`, etc). At `finish()`:

```
elem_writer = VarBinViewWriter(Utf8, total_elements)

for elem_chunk in stashed:
  elem_chunk.execute_into_builder(elem_writer)
  |
  |  VarBinView: push_buffer_and_adjusted_views()
  |  -> views copied (16 bytes each)
  |  -> data buffers attached zero-copy (just Vec::push)
  |
  |  FSST: decompress codes -> push_bytes per string
```

For VarBinView elements, the data buffers are **never copied** — they're attached to the
builder's completed buffer list. Only the 16-byte view structs need index adjustment for
the buffer offset. This is already how `VarBinViewBuilder` works today.

---

## End-to-End Trace: `Chunked<List<Chunked<FoR<BitPacked<i32>>>>>`

The hardest real-world case. Shows every layer of the design working together.

```
canonicalize_chunked(array, ctx):
|
| writer = canonical_writer(List<i32>, len=4)
|  -> ListWriter { offsets(cap=4), sizes(cap=4), elem_chunks=[], total=0 }
|
| array.execute_into_builder(&mut writer, ctx)
|
| +- ChunkedEncoding::execute_into_builder (outer):
| |  for chunk in chunks:                      // each is List<Chunked<FoR<BP<i32>>>>
| |    chunk.execute_into_builder(writer, ctx)
| |
| |    +- ListViewEncoding::execute_into_builder:
| |    |    lw = writer.downcast::<ListWriter>()
| |    |    base = lw.total_elements
| |    |    lw.push_list_parts(
| |    |      offsets + base,                   // shifted
| |    |      sizes,                            // as-is
| |    |      lv.elements(),                    // stash Chunked<FoR<BP<i32>>>
| |    |      validity,
| |    |    )
| |    |    lw.total_elements += lv.elements().len()
| |    |    // NO RECURSION. Elements stashed untouched.
| |    +-
| +-
|
| // After all chunks, ListWriter holds:
| //   offsets:        [0, 3, 7, 10, ...]          unified, shifted
| //   sizes:          [3, 4, 3, ...]              concatenated
| //   element_chunks: [Chunked<FoR<BP<i32>>>, ...]  stashed
| //   total_elements: 50000
|
| writer.finish(ctx):
|   // NOW create the element writer — we know the exact size
|   elem_writer = PrimitiveWriter(i32, 50000)
|    -> TypedPrimitiveWriter { PrimitiveBuilder<i32>(cap=50000) }
|
|   for chunk in element_chunks:                // each is Chunked<FoR<BP<i32>>>
|     chunk.execute_into_builder(&mut elem_writer, ctx)
|     |
|     |  +- ChunkedEncoding peels:
|     |  |  for sub in chunks:                  // each is FoR<BP<i32>>
|     |  |    sub.execute_into_builder(&mut elem_writer, ctx)
|     |  |
|     |  |    +- FoREncoding (fused path):
|     |  |    |    range = elem_writer.get_init_bytes(sub.len())
|     |  |    |    // FoRStrategy applies reference during unpack
|     |  |    |    BitPacking::decode_with_strategy(packed, FoRStrategy{ref}, range)
|     |  |    |    // <- ONLY DATA COPY. Directly into final buffer.
|     |  |    |    apply_patches(range, patches, |v| v.wrapping_add(ref))
|     |  |    |    range.append_validity(mask)
|     |  |    |    range.finish()
|     |  |    +-
|     |  +-
|     |
|
|   elements = elem_writer.finish()            // freeze BufferMut -> Canonical::Primitive
|
|   -> Canonical::List(ListViewArray {
|       elements,                               // contiguous, one allocation
|       offsets: [0, 3, 7, 10, ...],
|       sizes:   [3, 4, 3, ...],
|       validity,
|     })
```

**Result**: One allocation for all 50,000 element values. One decode pass per BitPacked chunk
(with fused FoR reference addition). Zero intermediate arrays. Offsets/sizes are cheap integer
buffers.

---

## Per-Encoding Reference

### Pattern assignment

| Encoding | Pattern | Writer Type | Details |
|---|---|---|---|
| **BitPacking** | 1 (direct) | PrimitiveWriter | `unpack_into_primitive_builder()` already exists. Writes to `UninitRange` via FastLanes `unchecked_unpack()`. Patches applied to same range at sparse indices. |
| **FoR (fused)** | 1 (direct) | PrimitiveWriter | `FoRStrategy` applies `wrapping_add(reference)` during unpack. Single pass. Patches shifted by reference. |
| **FoR (non-fused)** | 2 (child->transform) | PrimitiveWriter | Child writes unsigned ints, then `map_last_n(\|v\| v.wrapping_add(ref))`. |
| **Delta** | 3 (combine) | PrimitiveWriter | Execute bases (small) + deltas. Undelta + untranspose per 1024-element chunk into range. |
| **RLE (FL)** | 3 (combine) | PrimitiveWriter | Execute values + indices + offsets. Dictionary scatter into range. |
| **ALP** | 2 (child->transform) | PrimitiveWriter | Child writes encoded ints (e.g. `FoR<BP<i64>>`). Then `map_last_n(\|int\| alp_decode(int, e, f))` transmutes int->float in-place. Then patches overwrite float exceptions. |
| **ALP-RD** | 3 (combine) | PrimitiveWriter | Execute left_parts (dict-decode + patch) + right_parts. Combine: `from_bits((left << shift) \| right)`. Left dict-decode needs temp buffer (unavoidable). |
| **ZigZag** | 2 (child->transform) | PrimitiveWriter | Child writes unsigned. Then `map_last_n(\|u\| zigzag_decode(u))` flips to signed. Already uses `into_buffer_mut()` + `map_each_in_place()` today. |
| **RunEnd** | 3 (combine) | any | Execute ends + values. Fill runs into range. Known output size. |
| **Sparse** | 3 (combine) | any | Fill default + scatter patch values at patch indices. The encoding IS patches. |
| **Dict** | 3 (combine) | any | Execute codes + values. Gather: `output[i] = values[codes[i]]`. Random access pattern. |
| **Pco** | 1 (direct) | PrimitiveWriter | Decompress pages directly into range. Lazy — only needed pages. |
| **Zstd (prim)** | 1 (direct) | PrimitiveWriter | Decompress frames into aligned range. Size from metadata. |
| **Zstd (str)** | 3 (combine) | VarBinViewWriter | Decompress frames, parse length-prefixed strings, push views. |
| **FSST** | 3 (combine) | VarBinViewWriter | Decompress code sequences via symbol table -> `push_bytes` per string. Already does bulk allocation. |
| **Sequence** | 1 (direct) | PrimitiveWriter | Compute `base + i * mult` into range. No children, no input buffers. |
| **ByteBool** | 1 (direct) | BoolWriter | Convert byte->bit directly into `BitBufferMut`. |
| **DateTimeParts** | 3 (combine) | PrimitiveWriter | Execute days/seconds/subseconds. Combine: `days * 86400 * div + seconds * div + sub`. |
| **DecimalByteParts** | 2 (child->transform) | DecimalWriter | Execute MSP child. Bits ARE the decimal mantissa. Zero transform. |
| **Constant** | 1 (direct) | any | `get_init_bytes(n)` + fill. Already has optimized `append_to_builder`. |

### Container encodings

| Encoding | Strategy |
|---|---|
| **Chunked** | `for chunk in chunks { chunk.execute_into_builder(writer) }` — peel loop |
| **Primitive** | `memcpy` buffer into `PrimitiveWriter.get_init_bytes()` |
| **Bool** | `memcpy` bits into `BoolWriter` |
| **Null** | Increment `NullWriter.len` |
| **VarBinView** | `push_views(views, buffers)` — stash data buffers zero-copy |
| **VarBin** | Build views from offsets, share data buffer |
| **ListView** | Push shifted offsets/sizes into `ListWriter`, stash elements |
| **List** (offset) | Compute sizes from adjacent offsets, push, stash elements |
| **FixedSizeList** | Stash elements. `finish()` -> element writer of size `len * list_size` |
| **Struct** | Push each field into its child writer |
| **Extension** | Delegate to storage writer |
| **Slice** | `inner.slice(range).execute_into_builder(writer)` |
| **Filter** | Materialize mask, write selected rows |
| **Masked** | Delegate child + apply validity mask |

---

## Writer API Summary

### On the trait (`dyn CanonicalWriter`)

| Method | Purpose |
|---|---|
| `as_any_mut()` | Downcast to concrete writer type |
| `dtype()` | Type of output |
| `len()` | Rows written so far |
| `write(array, ctx)` | Default: dispatches to array's `execute_into_builder` vtable |
| `finish(self, ctx)` | Consume -> `Canonical` |

### On concrete writers (accessed via downcast)

| Writer | Method | Purpose | Used by |
|---|---|---|---|
| `PrimitiveWriter` | `get_init_bytes(n)` -> `UninitRange` | The one primitive for buffer writes | All Pattern 1 & 3 encodings |
| `PrimitiveWriter` | `map_last_n(n, f)` | Transform last n values in-place | FoR, ALP, ZigZag |
| `PrimitiveWriter` | `append_validity(mask)` | Push validity bits | All encodings |
| `ListWriter` | `push_list_parts(offsets, sizes, elements, validity)` | Push metadata + stash elements | ListView, List |
| `VarBinViewWriter` | `push_views(views, buffers)` | Push view structs + stash data buffers | VarBinView, VarBin |
| `VarBinViewWriter` | `push_bytes(bytes)` | Append one string/binary value | FSST, Zstd (str) |
| `StructWriter` | `field_writer(idx)` -> `&mut dyn CanonicalWriter` | Access per-field writer | Struct |

---

## Pre-allocatability by Canonical Type

| Canonical Type | Pre-allocatable? | How size is known |
|---|---|---|
| Primitive | Yes, exact | `len * size_of::<T>()` |
| Bool | Yes, exact | `ceil(len / 8)` bytes |
| Decimal | Yes, exact | `len * decimal_byte_width` |
| Null | Trivial | Counter |
| VarBinView | Views: yes. Data: zero-copy attach | Views = `len * 16`. Data buffers stashed from chunks. |
| Struct | Yes, recurse per field | Each field pre-allocated to `len` |
| FixedSizeList | Yes, exact | Elements = `len * list_size`, recurse |
| List | Deferred to `finish()` | `total_elements` accumulated during `write()` calls, then child writer pre-allocated |
| Extension | Yes, delegates to storage | Same as storage type |

---

## What Changes from Today

| Concern | Today | New |
|---|---|---|
| Dispatch | 3 branches in `_canonicalize` (struct, list, default) | 1 call: `canonical_writer(dtype, len)` |
| Pre-allocation | Builder may realloc for leaf types | Every buffer sized exactly once at construction |
| Struct fields | `pack_struct_chunks` -> `ChunkedArray` per field (still needs canonicalization later) | `StructWriter` recurses all the way to canonical leaves |
| List elements | `swizzle_list_chunks` -> `ChunkedArray` elements (still compressed) | `ListWriter.finish()` recurses to canonical elements with known size |
| Copies per chunk | 2 (decompress + memcpy into builder) | 1 (decode directly into final buffer) |
| Peak memory | ~2.25x decompressed (compressed + output + temp) | ~1.1x decompressed (compressed chunk + output) |
| Encoding opt-in | N/A | Default fallback -> encodings specialize incrementally |

---

## Implementation Order

Priority by throughput impact:

| Phase | What | Why |
|---|---|---|
| 1 | `CanonicalWriter` trait + factory + `PrimitiveWriter` + `ListWriter` | Core infrastructure |
| 1 | `canonicalize_chunked` using writers | Replaces `_canonicalize` |
| 1 | `ChunkedEncoding::execute_into_builder` | Peel loop — enables everything |
| 2 | `BitPackedEncoding::execute_into_builder` | Most common leaf. Already has `unpack_into_primitive_builder`. |
| 2 | `FoREncoding::execute_into_builder` | Usually wraps BitPacking. Fused path is the hot path. |
| 2 | `ALPEncoding::execute_into_builder` | Most common float encoding. Removes clone hack. |
| 3 | Other writers: `BoolWriter`, `StructWriter`, `VarBinViewWriter` | Complete type coverage |
| 3 | Remaining encodings | ZigZag, Delta, RLE, RunEnd, etc. — incremental opt-in |

After phases 1-2, the majority of real-world columnar read throughput is covered:
BitPacking + FoR handles most integers, ALP handles most floats.