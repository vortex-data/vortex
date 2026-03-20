# Design: `CanonicalWriter` and `execute_into_builder`

## Motivation

Today, canonicalizing a `Chunked<FoR<BitPacked<i32>>>` with N chunks does **2N data copies**:

1. Each chunk decompresses into a throwaway `PrimitiveArray` (copy #1)
2. That array is `memcpy`'d into the builder's output buffer (copy #2)

Peak memory is ~2.25x the final output because the compressed source, the output buffer, and
one intermediate chunk buffer all live simultaneously.

See [vtable-execution-problems.md](vtable-execution-problems.md) for the full problem catalog.

This design introduces a `CanonicalWriter` trait and an `execute_into_builder` vtable method
that eliminate the intermediate buffer entirely. Encodings decode **directly into the final
output buffer** — one copy total, one allocation total.

---

## Core API

### The trait

```rust
pub trait CanonicalWriter: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn dtype(&self) -> &DType;
    fn len(&self) -> usize;

    /// Default entry point: dispatches to the array's vtable.
    fn write(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        array.execute_into_builder(self, ctx)
    }

    /// Consume the writer and produce the final Canonical.
    fn finish(self: Box<Self>, ctx: &mut ExecutionCtx) -> VortexResult<Canonical>;
}
```

### The vtable method

```rust
trait ExecuteIntoBuilderVTable {
    fn execute_into_builder(
        &self,
        array: &ArrayRef,
        writer: &mut dyn CanonicalWriter,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;
}
```

Default implementation for encodings that don't specialize:

```rust
fn execute_into_builder(
    &self,
    array: &ArrayRef,
    writer: &mut dyn CanonicalWriter,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let canonical = array.clone().execute::<Canonical>(ctx)?;
    writer.write(&canonical.into_array(), ctx)
}
```

### The unified buffer primitive: `get_init_bytes`

Every data-writing operation an encoding needs is a pattern on one primitive:

```rust
impl PrimitiveWriter {
    /// Reserve `n` elements of uninitialized memory. Returns a handle the
    /// caller writes into however it wants.
    pub fn get_init_bytes(&mut self, n: usize) -> UninitRange<'_>;
}
```

| "Operation" | Implementation |
|---|---|
| Decode into buffer | `get_init_bytes(n)` + `decode_into(range)` |
| memcpy | `get_init_bytes(n)` + `copy_from_slice` |
| Fill with scalar | `get_init_bytes(n)` + loop fill |
| Child writes, then transform in-place | child calls `get_init_bytes(n)`, parent iterates range applying `f` |
| Scatter at indices | `get_init_bytes(n)` + fill default + write at indices |
| Apply patches | `range.set_value(idx, val)` on same range |

There is no separate method for any of these. The encoding's vtable impl calls `get_init_bytes`
and does whatever it wants with the bytes.

---

## Writer types

### `PrimitiveWriter`

Wraps a `PrimitiveBuilder<T>` behind a type-erased layer (necessary because the writer is
created from a `DType` at runtime, not a compile-time `T`).

```rust
pub struct PrimitiveWriter {
    dtype: DType,
    inner: Box<dyn PrimitiveWriterInner>,  // TypedPrimitiveWriter<T> for the correct T
}

struct TypedPrimitiveWriter<T: NativePType> {
    builder: PrimitiveBuilder<T>,
}
```

Encodings downcast via `writer.as_any_mut().downcast_mut::<PrimitiveWriter>()`, then access the
typed inner via a second downcast to call `uninit_range()` etc.

`finish()`: `builder.finish_into_primitive()` -> `Canonical::Primitive`.

### `ListWriter`

The canonical list form is `ListViewArray` (offsets + sizes + elements). The key design choice:
**defer element canonicalization to `finish()`**.

```rust
pub struct ListWriter {
    dtype: DType,
    elem_dtype: DType,

    offsets: BufferMut<u64>,
    sizes: BufferMut<u64>,
    nulls: LazyBitBufferBuilder,

    /// Stashed element arrays — NOT canonicalized yet.
    element_chunks: Vec<ArrayRef>,
    total_elements: usize,
}
```

During `write()`, `ListViewEncoding` pushes shifted offsets/sizes and stashes element `ArrayRef`s.
No recursion, no copy.

During `finish()`, one child `CanonicalWriter` is created for `elem_dtype` and all stashed
element chunks are flushed through it:

```rust
fn finish(self: Box<Self>, ctx: &mut ExecutionCtx) -> VortexResult<Canonical> {
    let mut elem_writer = canonical_writer(&self.elem_dtype, self.total_elements);
    for chunk in &self.element_chunks {
        chunk.execute_into_builder(elem_writer.as_mut(), ctx)?;
    }
    let elements = elem_writer.finish(ctx)?.into_array();
    Ok(Canonical::List(ListViewArray::new_unchecked(
        elements, self.offsets.freeze().into(), self.sizes.freeze().into(),
        self.nulls.finish(),
    )))
}
```

This means nested `Chunked<List<Chunked<BitPacked<i32>>>>` resolves to flat loops over
`get_init_bytes` — one allocation per nesting level.

### Other writers

| Writer | Accumulates during `write()` | Does during `finish()` |
|---|---|---|
| **BoolWriter** | Bits into `BitBufferMut` + validity | `freeze()` -> `BoolArray` |
| **NullWriter** | Increments a counter | `NullArray::new(len)` |
| **DecimalWriter** | Same as PrimitiveWriter (integer mantissa) | Wrap buffer as `DecimalArray` |
| **VarBinViewWriter** | Appends views, stashes data buffers | Assemble `VarBinViewArray` |
| **StructWriter** | One child writer per field | Each field writer finishes independently |
| **FixedSizeListWriter** | Stashes element chunks (like ListWriter but no offsets/sizes) | Flush elements through child writer |
| **ExtensionWriter** | Delegates to inner storage writer | Wrap result in `ExtensionArray` |

---

## Three encoding patterns

Every encoding falls into exactly one pattern:

### Pattern 1: Direct decode

The encoding owns compressed bytes and decodes straight into `get_init_bytes`. No children to
execute first.

**Encodings**: BitPacking, Pco, Zstd (primitive), Sequence, Constant, ByteBool

```
encoding.execute_into_builder(writer):
    range = writer.get_init_bytes(len)
    decode_compressed_bytes_into(range)
    range.finish()
```

### Pattern 2: Execute child, transform in-place

The child writes into the writer's buffer, then the parent encoding mutates those same bytes
in-place (add reference, transmute int->float, flip sign encoding, etc).

**Encodings**: FoR (non-fused), ALP, ZigZag, DecimalByteParts

```
encoding.execute_into_builder(writer):
    // Child writes into the buffer
    self.encoded().execute_into_builder(writer, ctx)
    // Parent transforms those bytes in-place
    writer.map_last_n(len, |v| transform(v))
    // Patches overwrite specific indices
    apply_patches(writer, self.patches(), ctx)
```

The writer exposes `map_last_n(n, f)` which iterates the tail of the buffer applying `f`.
This is just sugar over `get_init_bytes` — the range was already allocated by the child.

### Pattern 3: Execute children, combine

Multiple children must be materialized first (into temp arrays), then combined into the
writer's buffer.

**Encodings**: ALP-RD, Delta, RLE, RunEnd, DateTimeParts, Dict, Sparse

```
encoding.execute_into_builder(writer):
    let child_a = self.child_a().execute::<PrimitiveArray>(ctx)
    let child_b = self.child_b().execute::<PrimitiveArray>(ctx)
    let range = writer.get_init_bytes(len)
    combine_into(child_a, child_b, range)
    range.finish()
```

The temp arrays for children are unavoidable (need two+ sources to merge), but the **output**
goes directly into the final buffer. No intermediate output allocation.

---

## Patches

Patches are always applied to the same `UninitRange` returned by `get_init_bytes`. They are
just a scatter-write on the range:

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

For **FoR + BitPacking fused**: patches get the reference added via `wrapping_add` during apply:

```
apply_patches_with_transform(range, patches, |v| v.wrapping_add(reference))
```

For **ALP**: patches contain the true float values that didn't round-trip through integer
encoding. They overwrite specific indices after the int->float transmute.

---

## Chunked + List trace

To show the design works end-to-end for the hardest case:

### `Chunked<List<Chunked<BitPacked<i32>>>>`

```
canonicalize(array):
    writer = ListWriter::new(List<i32>)
    array.execute_into_builder(&mut writer, ctx)

    // Step 1: Chunked peels
    ChunkedEncoding::execute_into_builder(writer):
        for chunk in chunks:             // each is List<Chunked<BitPacked<i32>>>
            chunk.execute_into_builder(writer, ctx)

    // Step 2: Each ListView pushes metadata, stashes elements
    ListViewEncoding::execute_into_builder(writer):
        lw = writer.downcast::<ListWriter>()
        lw.push_list_parts(
            offsets + base,              // shifted by current element count
            sizes,
            lv.elements(),               // stash Chunked<BitPacked<i32>> as-is
            validity,
        )
        // NO RECURSION. Elements stashed untouched.

    // After all chunks, ListWriter holds:
    //   offsets:        [0, 3, 7, 10, ...]         unified, shifted
    //   sizes:          [3, 4, 3, ...]             concatenated
    //   element_chunks: [Chunked<BP<i32>>, ...]    stashed, not touched

    // Step 3: finish() flushes elements through a single PrimitiveWriter
    writer.finish(ctx):
        elem_writer = PrimitiveWriter::new(i32, total_elements)

        for chunk in element_chunks:     // each is Chunked<BitPacked<i32>>
            chunk.execute_into_builder(&mut elem_writer, ctx)

            // Chunked peels again
            for sub in chunks:           // each is BitPacked<i32>
                sub.execute_into_builder(&mut elem_writer, ctx)

                // BitPacking decodes directly into the final buffer
                range = elem_writer.get_init_bytes(sub.len())
                decode_into(range)       // <- ONLY data copy
                apply_patches(range)
                range.finish()

        elements = elem_writer.finish()  // -> Canonical::Primitive(i32)
        -> Canonical::List(ListViewArray { elements, offsets, sizes, validity })
```

One allocation for all element data. One decode pass per BitPacked chunk. Zero intermediates.

---

## Per-encoding summary

### Canonical types (already the target — push buffers in)

| Encoding | Writer | Strategy |
|---|---|---|
| **Primitive** | PrimitiveWriter | `memcpy` via `get_init_bytes` |
| **Bool** | BoolWriter | `memcpy` bits |
| **Null** | NullWriter | Increment counter |
| **Decimal** | DecimalWriter | `memcpy` integer mantissa |
| **VarBinView** | VarBinViewWriter | Push views + stash data buffers |
| **VarBin** | VarBinViewWriter | Build views from offsets, share data buffer |
| **ListView** | ListWriter | Push shifted offsets/sizes, stash elements |
| **List** (offset) | ListWriter | Compute sizes from adjacent offsets, push, stash elements |
| **FixedSizeList** | FSLWriter | Stash elements |
| **Struct** | StructWriter | Push each field into its child writer |
| **Extension** | inner writer | Delegate to storage array |

### Container encodings (peel, don't decode)

| Encoding | Writer | Strategy |
|---|---|---|
| **Chunked** | any | `for chunk in chunks { chunk.execute_into_builder(writer) }` |
| **Constant** | any | `get_init_bytes(n)` + fill with scalar |
| **Slice** | any | `inner.slice(range).execute_into_builder(writer)` |
| **Filter** | any | Materialize mask, write selected rows |
| **Masked** | any | Delegate child + apply validity mask |
| **Dict** | any | Execute codes+values, scatter: `writer[i] = values[codes[i]]` |

### Compressed encodings — Pattern 1: Direct decode into `get_init_bytes`

| Encoding | Writer | Notes |
|---|---|---|
| **BitPacking** | PrimitiveWriter | `unpack_into_primitive_builder` already exists. Patches applied to same `UninitRange`. |
| **FoR** (fused) | PrimitiveWriter | `FoRStrategy` applies reference during unpack — single pass. Patches shifted by `wrapping_add(ref)`. |
| **Pco** | PrimitiveWriter | Decompress pages directly into range. Only needed pages (lazy). |
| **Zstd** (primitive) | PrimitiveWriter | Decompress frames into range. Size from metadata. |
| **Zstd** (strings) | VarBinViewWriter | Decompress frames, parse length-prefixed strings, push views. |
| **Sequence** | PrimitiveWriter | Compute `base + i * multiplier` directly into range. No children. |
| **ByteBool** | BoolWriter | Convert byte->bit directly into `BitBufferMut`. |

### Compressed encodings — Pattern 2: Execute child, transform in-place

| Encoding | Writer | Transform | Patches |
|---|---|---|---|
| **FoR** (non-fused) | PrimitiveWriter | `map_last_n(\|v\| v.wrapping_add(reference))` | From inner child, shifted by reference |
| **ALP** | PrimitiveWriter | `map_last_n(\|int\| alp_decode(int, e, f))` (int->float transmute) | Float exceptions overwrite specific indices |
| **ZigZag** | PrimitiveWriter | `map_last_n(\|u\| zigzag_decode(u))` (unsigned->signed) | None |
| **DecimalByteParts** | DecimalWriter | None (integer bits ARE the decimal mantissa) | None |

### Compressed encodings — Pattern 3: Execute children, combine into `get_init_bytes`

| Encoding | Writer | Children | Combine |
|---|---|---|---|
| **ALP-RD** | PrimitiveWriter | left_parts (dict-decode + patch), right_parts | `from_bits((left << shift) \| right)` |
| **Delta** | PrimitiveWriter | bases (small), deltas | Undelta + untranspose into range |
| **RLE** (FastLanes) | PrimitiveWriter | values, indices, offsets | Dictionary scatter into range |
| **RunEnd** | any | ends, values | Fill runs into range |
| **Sparse** | any | patch indices, patch values | Fill default + scatter at indices |
| **DateTimeParts** | PrimitiveWriter | days, seconds, subseconds | `days * 86400 * divisor + seconds * divisor + subseconds` |
| **Dict** | any | codes, values | Gather: `output[i] = values[codes[i]]` |
| **FSST** | VarBinViewWriter | codes, symbol table | Decompress each code sequence, push string views |

---

## Writer API surface

The trait is minimal. Concrete writer methods are accessed via downcast.

### On the trait (`dyn CanonicalWriter`)

| Method | Purpose |
|---|---|
| `as_any_mut()` | Downcast to concrete writer type |
| `dtype()` | Type check |
| `len()` | Rows written so far |
| `write(array, ctx)` | Default entry: dispatches to array's vtable |
| `finish(self, ctx)` | Consume and produce `Canonical` |

### On `PrimitiveWriter` (accessed via downcast)

| Method | Purpose | Used by |
|---|---|---|
| `get_init_bytes(n) -> UninitRange` | The one primitive for all buffer writes | All Pattern 1 & 3 encodings |
| `map_last_n(n, f)` | Transform the last n values in-place | FoR, ALP, ZigZag |
| `append_validity(mask)` | Push validity bits | All encodings |

### On `ListWriter` (accessed via downcast)

| Method | Purpose |
|---|---|
| `push_list_parts(offsets, sizes, elements, validity)` | Push metadata + stash elements |

### On `VarBinViewWriter` (accessed via downcast)

| Method | Purpose |
|---|---|
| `push_views(views, buffers)` | Push view structs + stash data buffers |
| `push_bytes(bytes)` | Append a single string/binary value |

### On `StructWriter` (accessed via downcast)

| Method | Purpose |
|---|---|
| `field_writer(idx) -> &mut dyn CanonicalWriter` | Get the writer for field `idx` |
