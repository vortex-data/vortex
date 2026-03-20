# VTable Execution: Ownership, Copy, and Memory Problems

## Context

The core execution loop lives in `vortex-array/src/executor.rs`. The function `execute_until<M>` is
iterative, using an explicit work stack instead of recursion. Every `VTable::execute` takes
`array: &Self::Array` — a shared reference — and returns an `ExecutionStep` (either `Done(ArrayRef)`
or `ExecuteChild(idx, predicate)`).

This document catalogs the problems arising from this design, focused on buffer ownership, copy
overhead, and peak memory.

---

## Problem 1: Cannot Take Ownership of Buffers During Decompression

### Root cause

`VTable::execute` takes `&Self::Array`. Arrays are `Arc`-wrapped (`ArrayRef = Arc<dyn DynArray>`),
and buffers are stored inside the array struct. The API gives no way to move buffers out — even when
the `Arc` refcount is 1.

### Consequence

Every decompression must:
1. **Borrow** the compressed buffer via `&self`
2. **Allocate** a fresh output buffer
3. **Write** decompressed data into the new buffer

The compressed buffer stays alive (pinned by Arc) until the parent `ArrayRef` is dropped, which
typically happens only after the decompressed result is fully constructed.

### Encoding-specific impact

| Encoding | Pattern | Could benefit from ownership? |
|----------|---------|-------------------------------|
| **BitPacking** | Reads packed buffer, writes to new `BufferMut` | **HIGH** — could unpack in-place if buffer were owned |
| **FoR** | Reads encoded child + reference scalar, writes new buffer | **HIGH** — simple element-wise add |
| **Delta** | Reads bases + deltas, writes new buffer | **HIGH** — prefix-sum in-place |
| **ZigZag** | Reads encoded, writes decoded | **HIGH** — trivial bit-flip, perfect for in-place |
| **ALP** | Already works around this: clones the array (`array.clone()`), calls `into_parts()` to take ownership, then `into_buffer_mut()` which attempts zero-copy but falls back to `BufferMut::copy_from` if Arc refcount > 1 | **Already paying for a workaround** |
| **ALP-RD** | Executes both left_parts and right_parts to `PrimitiveArray`, reads them | MEDIUM — two input buffers merged |
| **RunEnd** | Reads ends + values, writes expanded buffer | LOW — output is larger than input |
| **Sparse** | Reads fill value + patch values, writes expanded buffer | LOW — output is larger than input |
| **PCO** | Reads pages sequentially, writes new buffer | LOW — stateful sequential decode |
| **Zstd/ZstdBuffers** | Reads compressed frames, writes new buffer | LOW — zstd requires separate output |
| **FSST** | Reads compressed codes, writes decoded string heap | LOW — output is larger than input |
| **Sequence** | Generates values from start/step | NONE — no input buffer |
| **ByteBool** | Reads byte buffer, writes `BitBuffer` | LOW — different representation |

### The ALP workaround pattern

ALP is the only encoding that already works around this. In `alp/decompress.rs`:
```rust
pub fn execute_decompress(array: ALPArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let (encoded, exponents, patches, dtype) = array.into_parts();
    // ...
    let encoded = encoded.execute::<PrimitiveArray>(ctx)?;
    // ...
    let mut alp_buffer = encoded.into_buffer_mut(); // zero-copy if refcount==1, else copies
    <T>::decode_slice_inplace(alp_buffer.as_mut_slice(), exponents);
}
```

But even this has a catch: `into_buffer_mut()` (`primitive/array/conversion.rs:70`) calls
`try_into_buffer_mut()` which only succeeds zero-copy when the underlying `Bytes` refcount is 1.
Because the `execute` signature takes `&Self::Array`, the VTable call site in `executor.rs:145`
holds `current` while calling `vtable().execute(&current, ctx)`, meaning the Arc is still alive
during execute. ALP's workaround is to clone the array in the VTable's execute impl:
```rust
fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
    // TODO(joe): take by value
    Ok(ExecutionStep::Done(execute_decompress(array.clone(), ctx)?.into_array()))
}
```
This clone bumps the Arc refcount, so `into_buffer_mut` will almost always fall back to copying.

---

## Problem 2: Chunked Array Reassembly Double-Copy

### Root cause

When a `ChunkedArray` with N chunks of a leaf type (primitive, bool, varbinview, decimal) is
executed, `_canonicalize` in `chunked/vtable/canonical.rs` does:

```rust
let mut builder = builder_with_capacity(array.dtype(), array.len());
array.append_to_builder(builder.as_mut(), ctx)?;
builder.finish_into_canonical()
```

Which iterates each chunk:
```rust
fn append_to_builder(array: &ChunkedArray, builder: &mut dyn ArrayBuilder, ctx: &mut ExecutionCtx) {
    for chunk in array.chunks() {
        chunk.append_to_builder(builder, ctx)?; // each chunk: decompress then memcpy into builder
    }
}
```

### Consequence: 2N copies for N chunks

For each chunk:
1. **Copy #1** — Decompress: the encoding's `execute` allocates a new buffer and writes decompressed
   data into it (because it can't take ownership of the compressed buffer, per Problem 1)
2. **Copy #2** — Append: the decompressed chunk's data is `memcpy`'d into the builder's contiguous
   output buffer

The intermediate per-chunk decompressed buffer is ephemeral — allocated, filled, copied into builder,
then dropped. This is pure waste.

### Types affected

| Type | Chunked canonicalize path | Copies |
|------|--------------------------|--------|
| Primitive / Bool / Decimal | builder + `extend_from_slice` | **2N** (decompress + memcpy) |
| VarBinView | builder + view/buffer management | **2N+** (more complex) |
| Struct | `pack_struct_chunks` — wraps fields in new ChunkedArrays | **0** (deferred) |
| List | `swizzle_list_chunks` — keeps elements chunked | **~0** (only offsets copied) |

### Ideal path

If the builder could receive the decompressed buffer directly (zero-copy handoff), or if
decompression could write directly into a pre-allocated region of the final output buffer, this
would go from 2N copies to 0-1 copies.

---

## Problem 3: Peak Memory Amplification

### Root cause

Problems 1 and 2 combine to create a worst-case memory spike during execution of a chunked
compressed array.

### Memory held simultaneously

During chunked canonicalization of N chunks:

| Memory region | Lifetime | Size |
|--------------|----------|------|
| Original `ChunkedArray` (compressed) | Entire execution — held by `Arc` | `compressed_total` |
| Builder output buffer | Growing, allocated at total capacity upfront | `decompressed_total` |
| Current chunk's decompressed form | Per-chunk temporary | `decompressed_chunk_size` |

**Peak memory ≈ `compressed_total` + `decompressed_total` + `decompressed_chunk_size`**

For typical compression ratios (4-10x), this means:
- `compressed_total` ≈ 10-25% of `decompressed_total`
- Peak ≈ **~2.1-2.25x** of the final decompressed size

### The Arc pinning problem

The original compressed `ChunkedArray` cannot be freed incrementally. Even after chunk 0 is
decompressed and copied into the builder, chunk 0's compressed buffer remains alive because
the entire `ChunkedArray` is held by a single `Arc`. There's no way to "release" individual
chunks as they're consumed.

If chunks could be consumed (moved out) one at a time, compressed memory could be freed
incrementally, reducing peak memory to approximately:
`decompressed_total` + `compressed_chunk_size` + `decompressed_chunk_size`

---

## Problem 4: Encodings That Eagerly Execute Children (Bypassing the Scheduler)

Several encodings call `.execute::<T>(ctx)` on their children directly inside their own `execute`
method, bypassing the iterative scheduler's stack-based execution:

| Encoding | What it does | Problem |
|----------|-------------|---------|
| **ZigZag** | `array.encoded().clone().execute(ctx)?` then `zigzag_decode()` | Recursive execute inside execute — scheduler can't optimize child |
| **ALP** | `encoded.execute::<PrimitiveArray>(ctx)?` | Same — child fully executed before ALP decode starts |
| **ALP-RD** | `left_parts().clone().execute(ctx)?` AND `right_parts().clone().execute(ctx)?` | Two recursive executions |
| **Dict** | `values().clone().execute::<Canonical>(ctx)?` AND `codes().clone().execute(ctx)?` | Two recursive executions |
| **Zstd** | `decompress(ctx)?.execute::<ArrayRef>(ctx)` | Decompress then re-enter execute |
| **Filter** | `child.clone().execute(ctx)?` | Recursive child execute |
| **Slice** | `child.clone().execute::<ArrayRef>(ctx)?` then `.slice()` | Recursive child execute |

These patterns:
- Bypass the scheduler's `execute_parent`/`reduce_parent` optimization passes between steps
- Hold the parent array alive during recursive child execution (exacerbating memory)
- Prevent the scheduler from interleaving work or applying cross-step optimizations

By contrast, encodings that return `ExecutionStep::ExecuteChild(idx, predicate)` (like BitPacking
could, but doesn't) let the scheduler manage execution, optimize between steps, and free
intermediate arrays.

---

## Problem 5: `with_child` Rebuilds All Children

When the scheduler pops the stack and calls `parent.with_child(idx, executed_child)`:

```rust
pub fn with_child(&self, child_idx: usize, replacement: ArrayRef) -> VortexResult<ArrayRef> {
    let mut children: Vec<ArrayRef> = self.children(); // clones ALL child Arcs
    children[child_idx] = replacement;
    self.with_children(children)
}
```

For an array with K children (e.g., StructArray with 100 fields), this clones K-1 unrelated `Arc`s
every time. This is O(K) per stack pop, which compounds when the scheduler executes multiple
children of the same parent.

---

## Summary: What Matters Most

| Problem | Impact | Priority |
|---------|--------|----------|
| No buffer ownership → forced alloc per decompress | Extra allocation + prevents in-place decode for BitPacking/FoR/Delta/ZigZag | **HIGH** |
| Chunked reassembly double-copy | 2x data movement for every leaf chunk | **HIGH** |
| Peak memory = compressed + decompressed + temp | ~2x working memory vs theoretical minimum | **HIGH** |
| Eager child execution bypasses scheduler | Missed optimizations, memory pressure | MEDIUM |
| `with_child` O(K) clone | Perf hit for wide structs | LOW |

---

# Proposed Solutions

## Solution 1: Take Arrays by Ownership

Change `VTable::execute` from `fn execute(array: &Self::Array, ...)` to
`fn execute(array: Self::Array, ...)`.

### What this enables

- Encodings can call `into_parts()` to destructure the array and take ownership of buffers
- `into_buffer_mut()` on `PrimitiveArray` can succeed zero-copy (refcount == 1) instead of
  always falling back to `BufferMut::copy_from`
- In-place decode for BitPacking, FoR, Delta, ZigZag, ALP (which already does this via a clone
  workaround)

### What this solves

- **Problem 1** (buffer ownership): Directly. Encodings own their buffers.
- **Problem 3** (peak memory): Partially. Compressed buffers can be freed after decompress if
  the encoding destructs the array.

### What this doesn't solve

- **Problem 2** (chunked double-copy): Ownership alone doesn't help — each chunk still decompresses
  into its own buffer, then gets copied into the builder. You need a destination-aware execute to
  avoid that second copy.

### Impact on the scheduler

The executor currently holds `current: ArrayRef` and passes `&current` to `vtable().execute()`.
With owned execute, the scheduler would `Arc::try_unwrap(current)` or clone-then-drop. The work
stack stores `(ArrayRef, usize, DonePredicate)` — the parent must still be kept alive while a child
is being executed, so parent ownership can't be transferred during child execution. But the *child*
can be owned when it's popped off for execution.

### Canonical type compatibility

All canonical types benefit equally — this is about the compressed/intermediate encodings, not the
output types.

---

## Solution 2: Builder/Output-Slot Execute

Instead of `execute` returning a new `ArrayRef`, pass a mutable output buffer (builder) into
execute so the encoding writes decompressed data directly into the final destination.

For chunked arrays, the caller pre-allocates one builder for the entire chunked array and gives each
chunk a slice/region of that builder to write into. This eliminates the intermediate per-chunk
buffer entirely.

### Canonical type compatibility for builder pre-allocation

| Canonical Type | Can pre-allocate? | Notes |
|---------------|-------------------|-------|
| **NullArray** | Trivial | No buffers needed, just a counter |
| **BoolArray** | **Yes** | Single `BitBufferMut` + validity. Size = `ceil(len / 8)` bytes. Exact. |
| **PrimitiveArray** | **Yes** | Single `BufferMut<T>` + validity. Size = `len * size_of::<T>()`. Exact. |
| **DecimalArray** | **Yes** | Single typed `BufferMut` + validity. Size depends on decimal type but calculable. |
| **StructArray** | **Yes** | Fixed field count known from DType. Each field gets its own builder recursively. |
| **FixedSizeListArray** | **Yes** | Elements capacity = `len * list_size`. Deterministic. |
| **ExtensionArray** | **Yes** | Delegates to storage type's builder. |
| **ListViewArray** | **Partial** | Offsets + sizes = exact (`len` entries each). But **elements** capacity is unknown upfront — total element count across all chunks isn't stored in metadata. Could sum from chunk metadata but that requires an extra pass. |
| **VarBinViewArray** | **Partial** | Views buffer = exact (`len * 16` bytes). But **data buffers** are dynamic — total string byte length is unknown upfront. Strings ≤12 bytes are inlined (no data buffer), longer strings reference external buffers. Cannot pre-allocate data buffers without knowing the distribution. |

### What this solves

- **Problem 2** (chunked double-copy): Directly. Decompression writes into the final buffer — no
  intermediate allocation, no second memcpy. Goes from 2N copies to N copies (just the decompress
  write).
- **Problem 3** (peak memory): Significantly. No temporary per-chunk decompressed buffers. Memory is
  just: compressed source + final output buffer.

### What this doesn't solve

- **Problem 1** (buffer ownership): Partially orthogonal. The encoding still borrows its input, but
  the output goes directly to the destination. The source buffer is still pinned by Arc.

### The hard cases: ListViewArray and VarBinViewArray

**ListViewArray**: The elements child is itself an array that can be arbitrarily deep (lists of
lists). Pre-allocating requires knowing the total element count across all chunks. Options:
- Store total element counts in chunk metadata (requires format change)
- First pass to sum element counts, second pass to decompress (extra traversal)
- Use a growable builder and accept potential reallocation

**VarBinViewArray**: The views buffer is easy (fixed 16 bytes per element). The data buffers are the
problem:
- Short strings (≤12 bytes) are inlined — zero data buffer usage
- Long strings reference external data buffers that grow dynamically
- Total string data size is unknown without scanning all chunks
- Current `VarBinViewBuilder` uses a growth strategy with `in_progress` + `completed` buffers

For both types, an exact pre-allocation is impossible without additional metadata. A reasonable
approach: pre-allocate views/offsets/sizes exactly, use growable buffers for elements/string data.

---

## Solution 3: Callback with Owned Children (State-Machine Execute)

Instead of a single `execute` call, the encoding acts as a state machine. The scheduler calls
execute, the encoding returns `ExecuteChild(idx)`. When the child is done, the scheduler calls
back into the encoding with the **owned** decompressed child (or an encoding-controlled `dyn Any`
state bag). The encoding then decides what to do next.

```rust
// Strawman API
enum ExecutionStep {
    ExecuteChild(usize, DonePredicate),
    Done(ArrayRef),
}

// The callback: encoding receives its decompressed children
fn resume(
    array: Self::Array,       // or &mut Self::Array
    child_idx: usize,
    child: ArrayRef,          // the executed child, OWNED
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionStep>;
```

### What this enables

- The encoding receives decompressed children by ownership, can call `into_buffer_mut()` on them
- Combined with Solution 2 (builder output), the encoding could decompress a child directly into
  a sub-region of a pre-allocated builder
- The scheduler controls the execution order and can run optimizations between steps
- The encoding doesn't need to hold its own children alive during child execution

### What this solves

- **Problem 1** (buffer ownership): Yes — children are owned when returned
- **Problem 2** (chunked double-copy): Yes, if combined with builder output slots
- **Problem 3** (peak memory): Yes — compressed children can be freed as they're consumed
- **Problem 4** (eager child execution): Yes — all child execution goes through the scheduler

### Complexity

This is the most invasive change. Every encoding's execute becomes a multi-step state machine.
For simple encodings (ZigZag, FoR) it's trivial — one child, one resume. For multi-child encodings
(ALP-RD with left+right parts, Dict with values+codes) it requires tracking which children have
been executed.

### Canonical type compatibility

Same as Solution 2 — the builder/output-slot question is orthogonal to the callback mechanism.

---

## Solution Comparison Matrix

| | Solves ownership? | Solves chunked double-copy? | Solves peak memory? | Invasiveness | Encoding complexity |
|---|---|---|---|---|---|
| **1: Owned execute** | **Yes** | No | Partial | Low — signature change + scheduler update | Low — `into_parts()` |
| **2: Builder output** | Partial | **Yes** | **Yes** | Medium — new builder infrastructure | Medium — write to slot instead of return |
| **3: Callback/resume** | **Yes** | Yes (with builder) | **Yes** | High — state machine per encoding | High — multi-step resume logic |
| **1 + 2 combined** | **Yes** | **Yes** | **Yes** | Medium | Medium |

### Recommendation

Solutions 1 and 2 are complementary and together address all three high-priority problems:

- **Solution 1** gives encodings ownership of their buffers → enables in-place decode, frees
  compressed memory after decompress
- **Solution 2** gives the chunked-array executor a way to write directly into the final output →
  eliminates the intermediate per-chunk buffer and the second memcpy

Solution 3 is more powerful but the complexity may not be justified unless there are further
use-cases that require the state-machine model (e.g., streaming execution, partial materialization).

---

# Per-Encoding Analysis

Detailed analysis of every encoding's execute/decompress path and how each solution applies.

## Compressed Encodings

### BitPacking (`encodings/fastlanes/src/bitpacking/`)

**Structure**: `BitPackedArray { packed: BufferHandle, patches: Option<Patches>, validity, bit_width, offset, len }`

**Data flow**:
1. Create `PrimitiveBuilder<T>` with capacity = `len`
2. Get `UninitRange` from builder (uninitialized output slice)
3. `decode_into(uninit_slice)` — FastLanes `BitPacking::unchecked_unpack()` writes 1024-element
   chunks directly into output
4. If patches: execute patch indices + values children, write at sparse positions
5. Finish builder → `PrimitiveArray`

**Allocations**: One output buffer (builder). Patch children executed eagerly (clone + execute).

**Owned execute**: Limited benefit — output is always wider than packed input (different layout),
can't decode in-place. But patches children could avoid Arc clones.

**Builder output**: **Excellent fit**. `decode_into()` already writes to a caller-provided slice.
Could write into a sub-region of a larger pre-allocated buffer. The `UninitRange` pattern is
already close to what's needed.

**Fused with FoR**: When BitPacking is a child of FoR, `fused_decompress()` applies
`FoR::unchecked_unfor_pack()` during unpacking — reference addition happens in the same pass as
bit-unpacking. This already avoids an intermediate buffer.

---

### FoR — Frame of Reference (`encodings/fastlanes/src/for/`)

**Structure**: `FoRArray { encoded: ArrayRef, reference: Scalar }`

**Data flow** (two paths):
- **Fused** (encoded is BitPacked + unsigned reference): `FoRStrategy` applies reference during
  `decode_into()` — single pass, no intermediate. Then applies patches with
  `|v| v.wrapping_add(&ref_)`.
- **Non-fused** (encoded is other): Execute encoded → `PrimitiveArray`, then
  `map_each_in_place(|v| v.wrapping_add(&min))`. In-place mutation, then `freeze()`.

**Allocations**: Fused path = one builder. Non-fused = one intermediate + in-place transform.

**Owned execute**: **High benefit for non-fused path**. `into_buffer_mut()` on the executed child
can be zero-copy if the child's refcount is 1. Currently blocked by Arc clone.

**Builder output**: **Good fit**. Fused path already writes to builder's uninit range. Non-fused
path could write to a pre-allocated region with reference addition.

---

### Delta (`encodings/fastlanes/src/delta/`)

**Structure**: `DeltaArray { bases: ArrayRef, deltas: ArrayRef, offset, len }`

**Data flow**:
1. Execute `bases` and `deltas` children to `PrimitiveArray`
2. Allocate `BufferMut<T>` with capacity = `deltas.len()`
3. For each 1024-element chunk:
   - `Delta::undelta()` with bases for the chunk's SIMD lanes
   - `Transpose::untranspose()` from lane-major to natural order
4. Scalar fallback for remainder (<1024 elements)
5. Slice to `[offset..offset+len]`

**Allocations**: Output buffer + temp `transposed: [T; 1024]` stack array per chunk. Both
children executed eagerly.

**Owned execute**: **Moderate**. Children are always needed, and output is a new buffer. But
owned children avoid Arc overhead.

**Builder output**: **Good fit**. The undelta+untranspose loop could write directly into a
sub-region of a pre-allocated buffer. No intermediate buffer needed.

---

### RLE — Run Length Encoding (`encodings/fastlanes/src/rle/`)

**Structure**: `RLEArray { values: ArrayRef, indices: ArrayRef, values_idx_offsets: ArrayRef, offset, length }`

**Data flow**:
1. Execute all three children to `PrimitiveArray`
2. Allocate `BufferMut<V>` with capacity = `num_chunks * 1024`
3. For each chunk: `V::decode(chunk_values, chunk_indices_1024, output_1024)` — dictionary lookup
   expands indices to values
4. Slice to `[offset..offset+length]`

**Allocations**: Output buffer (always larger than input — expansion encoding).

**Owned execute**: **Low benefit**. Output is always expanded; input buffers are small (dictionary).

**Builder output**: **Excellent fit**. Output size is known (`num_chunks * 1024`). Could write
directly into caller's buffer. Already pre-allocates correctly.

---

### ZigZag (`encodings/zigzag/src/`)

**Structure**: `ZigZagArray { encoded: ArrayRef }`

**Data flow**:
1. `array.encoded().clone().execute::<PrimitiveArray>(ctx)?`
2. `zigzag_decode()` → `into_buffer_mut()` → `map_each_in_place(zigzag_transform)`
3. Freeze buffer → `PrimitiveArray`

**Allocations**: Zero if `into_buffer_mut()` succeeds zero-copy. One copy if refcount > 1.

**Owned execute**: **Model implementation**. Already uses the `into_buffer_mut()` +
`map_each_in_place()` pattern. With owned execute, the clone before `.execute()` can be
eliminated, making `into_buffer_mut()` reliably zero-copy.

**Builder output**: Not needed — in-place transform is already optimal.

---

### ALP (`encodings/alp/src/alp/`)

**Structure**: `ALPArray { encoded: ArrayRef, patches: Option<Patches>, exponents: Exponents, dtype }`

**Data flow** (unchunked):
1. `array.clone()` (workaround for `&self`), `into_parts()`
2. `encoded.execute::<PrimitiveArray>(ctx)?`
3. `encoded.into_buffer_mut()` — **copies because clone bumped refcount**
4. `<T>::decode_slice_inplace(buffer, exponents)` — in-place ALP decode
5. `transmute` buffer to output type (i32→f32 or i64→f64, same size)
6. Apply patches if present

**Data flow** (chunked — with `chunk_offsets`):
Same as above but processes 1024-element chunks, applying patches per-chunk via `patch_chunk()`.

**Allocations**: One buffer (from `into_buffer_mut()`). Currently always copies due to clone
workaround. Patch children executed eagerly.

**Owned execute**: **Critical benefit**. Removes the `.clone()` workaround. `into_buffer_mut()`
becomes zero-copy. The `// TODO(joe): take by value` comment in the code acknowledges this.

**Builder output**: **Good fit**. Could write decoded values directly into a builder sub-region.
The in-place decode pattern means: copy encoded data into output slot, then decode in-place.

---

### ALP-RD (`encodings/alp/src/alp_rd/`)

**Structure**: `ALPRDArray { left_parts: ArrayRef, left_parts_dictionary: Buffer<u16>, right_parts: ArrayRef, right_bit_width, left_parts_patches: Option<Patches> }`

**Data flow**:
1. Execute `left_parts` → `PrimitiveArray<u16>`, `right_parts` → `PrimitiveArray<u32/u64>`
2. Dictionary decode: `BufferMut<u16>::from_iter(left_parts.iter().map(|code| dict[code]))`
   — **always allocates** new buffer
3. Apply left_parts_patches if present
4. `alp_rd_decode_core()`: `right_parts.map_each_in_place(|right| (left << shift) | right)`
   — combines in-place into `right_parts` buffer
5. Freeze → `Buffer<f32/f64>`

**Allocations**: Dictionary decode buffer (always new). Right parts may copy in `into_buffer_mut()`.

**Owned execute**: **High benefit**. Avoids Arc clones on both children. `right_parts.into_buffer_mut()`
becomes zero-copy. Dictionary decode allocation is unavoidable (output type differs from input).

**Builder output**: **Limited**. Dictionary decode must happen before combining. Could write final
combined result into builder, but the intermediate dictionary buffer is unavoidable.

---

### RunEnd (`encodings/runend/src/`)

**Structure**: `RunEndArray { ends: ArrayRef, values: ArrayRef, offset, length }`

**Data flow**:
1. Execute `ends` and `values` children
2. Type-dispatch to kernel (`runend_decode_slice` for primitives, `runend_decode_bools` for bools)
3. Allocate output `BufferMut<T>` with capacity = `length`
4. For each run: `push_n_unchecked(value, run_length)` — repeats value into output
5. Bool optimization: prefill majority bit value, then toggle minority runs

**Allocations**: Output buffer (always larger — expansion). Validity buffer if nullable.

**Owned execute**: **Low benefit**. Output is expanded; input is compact. No in-place opportunity.

**Builder output**: **Excellent fit**. Output size is exactly `length` (known). Could write
directly into a pre-allocated region. The `push_n_unchecked` loop is ideal for writing into
a sub-slice.

---

### Sparse (`encodings/sparse/src/`)

**Structure**: `SparseArray { patches: Patches, fill_value: Scalar }`

**Data flow**:
1. Execute patch indices and values children
2. Create output filled with `fill_value` (via builder or ConstantArray)
3. Overlay patch values at patch indices
4. Merge validity

**Allocations**: Full output buffer (always `array_len` size). Fill operation + sparse overlay.

**Owned execute**: **Low benefit**. Output is full-size regardless. Fill value is a scalar.

**Builder output**: **Excellent fit**. Pre-fill output buffer with fill value, then scatter patch
values at indices. Size is known (`array_len`).

---

### Dict (`vortex-array/src/arrays/dict/`)

**Structure**: `DictArray { codes: ArrayRef, values: ArrayRef }`

**Data flow**:
1. Execute `values` → `Canonical`, `codes` → `PrimitiveArray`
2. `take_canonical(values, codes)` — indexed gather: `output[i] = values[codes[i]]`
3. Type-specific take implementations for each canonical type

**Allocations**: Output buffer = `codes.len()`. Gather operation writes output.

**Owned execute**: **Moderate**. Both children need full execution. Avoids Arc clones.

**Builder output**: **Poor fit**. Gather operation is random-access (not sequential write).
Output order depends on code values.

---

### PCO — Pcodec (`encodings/pco/src/`)

**Structure**: `PcoArray { chunk_metas, pages, metadata (header + frame info), slice_start, slice_stop }`

**Data flow**:
1. Create `FileDecompressor` from header
2. For each chunk: lazy-init `ChunkDecompressor` (stateful)
3. For each page in chunk: create `PageDecompressor`, decompress into growing `BufferMut<T>`
4. Slice final buffer to requested range

**Allocations**: Single growing output buffer. Stateful decompressor state per chunk.

**Owned execute**: **Low benefit**. Pages/metas are read-only byte buffers. Stateful decode can't
reuse input.

**Builder output**: **Good fit**. Could pass pre-allocated `BufferMut<T>` instead of growing one.
Must know final decompressed count (available from metadata).

---

### Zstd (`encodings/zstd/src/`)

**Structure**: `ZstdArray { dictionary, frames: Vec<ByteBuffer>, metadata (frame sizes), validity }`

**Data flow**:
1. Select frames for requested slice
2. Single `ByteBufferMut::with_capacity_aligned()` for all frames
3. Decompress each frame sequentially into the buffer
4. Reconstruct array (PrimitiveArray or VarBinViewArray depending on dtype)

**Allocations**: Single output buffer for all frames. Aligned allocation.

**Owned execute**: **Low benefit**. Zstd decompression always needs separate output.

**Builder output**: **Good fit**. Could decompress directly into a caller-provided aligned buffer.
Size is known from metadata.

---

### ZstdBuffers (`encodings/zstd/src/zstd_buffers.rs`)

**Structure**: `ZstdBuffersArray { compressed_buffers, uncompressed_sizes, buffer_alignments, inner_encoding_id, inner_metadata, children }`

**Data flow**:
1. `decompress_buffers()`: One `ByteBufferMut::with_capacity_aligned()` per buffer
2. `build_inner()`: Reconstruct wrapped array from decompressed buffers + children

**Allocations**: One allocation per buffer. Has `decode_plan()` for pre-computing layout.

**Owned execute**: **Low**. Same as Zstd — decompression needs separate output.

**Builder output**: **Good fit via decode_plan()**. Pre-compute layout, allocate once, decompress
into pre-split regions.

---

### FSST (`encodings/fsst/src/`)

**Structure**: `FSSTArray { symbols, symbol_lengths, codes: VarBinArray, uncompressed_lengths }`

**Data flow**:
1. Execute `uncompressed_lengths` child
2. Sum uncompressed lengths → total decompressed byte count
3. Single `ByteBufferMut::with_capacity(total)` for ALL decompressed string data
4. Decompress each code string using symbol table into contiguous buffer
5. `build_views()` creates `BinaryView` entries pointing into the buffer
6. Return `VarBinViewArray`

**Allocations**: One large byte buffer + one views buffer. Already efficient single-allocation.

**Owned execute**: **Low benefit**. Output is always larger (decompressed strings).

**Builder output**: **Partial**. Views buffer is pre-allocatable. Data buffer could be pre-allocated
if `uncompressed_lengths` sum is known. Currently already does bulk allocation.

---

### Sequence (`encodings/sequence/src/`)

**Structure**: `SequenceArray { base, multiplier, len, dtype }`

**Data flow**: `BufferMut::from_trusted_len_iter(SequenceIter)` — iterator generates values directly
into buffer. No input buffers.

**Owned execute**: Trivial (no input buffers to own).

**Builder output**: **Perfect fit**. Iterator can write directly into any buffer.

---

### ByteBool (`encodings/bytebool/src/`)

**Structure**: `ByteBoolArray { buffer: BufferHandle (bytes), validity }`

**Data flow**: `BitBuffer::from(array.as_slice())` converts byte → bit representation.

**Owned execute**: **Low**. Byte→bit is a format change, can't be in-place.

**Builder output**: **Good fit**. Could write bits directly into a `BitBufferMut`.

---

### DateTimeParts (`encodings/datetime-parts/src/`)

**Structure**: `DateTimePartsArray { days, seconds, subseconds }`

**Data flow**:
1. Execute `days` → cast to i64 → `into_buffer_mut()`
2. In-place: `days[i] *= 86_400 * divisor`
3. Execute `seconds` → if non-constant, add in-place: `days[i] += seconds[i] * divisor`
4. Execute `subseconds` → if non-constant, add in-place: `days[i] += subseconds[i]`

**Allocations**: Reuses days buffer via `into_buffer_mut()`. Multiple in-place passes.

**Owned execute**: **High benefit**. Three children needed. Owned `days` buffer makes
`into_buffer_mut()` reliably zero-copy.

**Builder output**: **Good fit**. Could pre-allocate i64 buffer, write combined result directly.

---

### DecimalByteParts (`encodings/decimal-byte-parts/src/`)

**Structure**: `DecimalBytePartsArray { msp: ArrayRef, decimal_dtype }`

**Data flow**: Execute `msp` → wrap in `DecimalArray` with dtype. Near-zero overhead.

**Owned execute**: Trivial.

**Builder output**: Not needed — essentially a type cast.

---

## Wrapper/Structural Arrays

### Slice (`vortex-array/src/arrays/slice/`)

**Data flow**: Execute child → canonical → slice range (usually zero-copy metadata operation).

**Owned execute**: Moderate — avoids Arc clone on child.

**Builder output**: Low — slicing is already ~free.

---

### Masked (`vortex-array/src/arrays/masked/`)

**Data flow**: Execute child → canonical → combine validity with mask → new canonical with modified
validity.

**Owned execute**: High — avoids Arc clone on child during recursive execution.

**Builder output**: High — could write validity directly instead of allocating new validity bitmap.

---

### Filter (`vortex-array/src/arrays/filter/`)

**Data flow**: Execute child → canonical → filter elements by mask → new smaller array.

**Owned execute**: Low — bulk copy is unavoidable (selecting subset).

**Builder output**: High — each filter function could write directly to builder instead of
allocating new arrays.

---

### Chunked (`vortex-array/src/arrays/chunked/`)

**Data flow** (leaf types): Builder iterates chunks, each chunk decompresses into builder.
**Data flow** (struct): Execute each chunk to StructArray, wrap fields in ChunkedArrays.
**Data flow** (list): Execute each chunk to ListViewArray, concatenate offsets/sizes, keep elements
chunked.

**Owned execute**: High — could consume chunks one at a time, freeing compressed memory
incrementally.

**Builder output**: Critical — this IS the double-copy problem. Builder pre-allocation +
direct-write-into-slot eliminates copy #2.

---

## Master Compatibility Matrix

| Encoding | Children | Output vs Input | Owned Exec | Builder Output | Best Approach |
|----------|----------|----------------|------------|---------------|---------------|
| BitPacking | patches (opt) | wider (unpack) | Low | **Excellent** | Builder (already writes to uninit range) |
| FoR | 1 encoded | same size | **High** | Good | Owned (in-place add) + builder |
| Delta | 2 (bases+deltas) | same size | Moderate | Good | Builder (write undelta directly) |
| RLE | 3 (vals+idx+offsets) | **expanded** | Low | **Excellent** | Builder (known output size) |
| ZigZag | 1 encoded | same size | **Model** | N/A | Owned (in-place transform) |
| ALP | encoded + patches | same size (transmute) | **Critical** | Good | Owned (removes clone hack) |
| ALP-RD | 2 (left+right) + patches | combines two → one | **High** | Limited | Owned (zero-copy right_parts) |
| RunEnd | 2 (ends+values) | **expanded** | Low | **Excellent** | Builder (known output size) |
| Sparse | patches + fill scalar | **expanded** | Low | **Excellent** | Builder (fill + scatter) |
| Dict | 2 (codes+values) | same as codes | Moderate | Poor | Owned + callback |
| PCO | pages (byte buffers) | decompressed | Low | Good | Builder (size from metadata) |
| Zstd | frames (byte buffers) | decompressed | Low | Good | Builder (size from metadata) |
| ZstdBuffers | compressed buffers | decompressed | Low | Good | Builder via decode_plan |
| FSST | codes + lengths | **expanded** strings | Low | Partial | Existing bulk alloc is good |
| Sequence | none | generated | N/A | **Perfect** | Builder (iterator fills) |
| ByteBool | 1 buffer | format change | Low | Good | Builder (bit-pack into target) |
| DateTimeParts | 3 (d/s/ss) | combines three → one | **High** | Good | Owned (in-place multi-pass) |
| DecimalByteParts | 1 (msp) | type cast | Trivial | N/A | Owned (pass-through) |
| Slice | 1 child | zero-copy metadata | Moderate | Low | Owned |
| Masked | 1 child + mask | same + validity | High | High | Both |
| Filter | 1 child + mask | **shrunk** | Low | High | Builder |
| Chunked | N chunks | concatenated | High | **Critical** | Both (the core problem) |
| Shared | 1 source | cached | None | None | N/A (caching layer) |
| Constant | none | expanded | Low | **Excellent** | Builder (already has append_to_builder) |
| ScalarFn | N children | computed | Moderate | Delegated | Depends on function |
| VarBin | offsets + bytes | VarBinView format | Moderate | Low | Owned |
| List | offsets + elements | ListView format | Moderate | Moderate | Owned |
