# Sub-Segment Slicing

## Problem

Today, `FlatReader` reads an **entire segment** from storage, deserializes **every buffer** in it,
decodes the full array, and only *then* applies row-range slicing or filtering. For a segment
containing a 1M-row `Struct { ts: Primitive<i64>, payload: List<u8>, flags: BitPacked<u8> }`,
reading rows 1000..2000 still fetches all ~20 MB of data just to use ~20 KB of it.

Sub-segment slicing lets the IO layer read **only the byte ranges within a segment that are
actually needed** for a given row range or filter mask. The key constraint: we can only afford
**one IO round-trip per segment**, so we must determine all needed byte ranges before fetching.

## Background: How Segments Pack Buffers

A segment is a contiguous byte range in a file. Inside it, buffers are packed sequentially:

```
┌─────────────────────── segment ────────────────────────┐
│ [pad][buf 0][pad][buf 1][pad][buf 2]...[flatbuffer][u32]│
└─────────────────────────────────────────────────────────┘
```

The flatbuffer suffix (the "array tree") describes the encoding tree:

```
Array
  root: ArrayNode
    encoding: u16          ← which encoding (Primitive, BitPacked, Dict, ...)
    metadata: [u8]         ← encoding-specific (bit_width, offset, ptype, ...)
    buffers:  [u16]        ← indices into the global Buffer descriptor list
    children: [ArrayNode]  ← recursive
  buffers: [Buffer]        ← global list: { padding, alignment_exponent, length }
```

Each `ArrayNode.buffers[i]` is an index into `Array.buffers`. The padding and length fields
let us compute the **byte offset of every buffer within the segment** without reading any
buffer data.

When `FLAT_LAYOUT_INLINE_ARRAY_NODE` is enabled, the flatbuffer metadata is stored in layout
metadata rather than in the segment, so the segment contains only data buffers and the
metadata is available before any IO happens.

## Background: SliceReduce

`SliceReduce` is the existing per-encoding trait for metadata-only slicing:

```rust
pub trait SliceReduce: VTable {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>)
        -> VortexResult<Option<ArrayRef>>;
}
```

Encodings implement this to push slices down through their structure. Key examples:

- **Primitive**: `buffer.slice_typed::<T>(range)` — byte-range slice on the values buffer
- **BitPacked**: compute chunk-aligned byte range from `bit_width` and `offset` metadata,
  then `packed.slice(encoded_start..encoded_stop)`
- **List**: keep elements unchanged, slice offsets to `[start..end+1]`, slice validity
- **Struct**: recurse into each field with the same row range
- **Dict**: slice codes, keep values unchanged

The logic for *how a row range maps to buffer byte ranges* already exists per-encoding in
`SliceReduce`. The problem is that `SliceReduce` operates on **live arrays with materialized
buffers**. We need the same logic but operating on **serialized metadata before IO**.

## Design

### New method on `ArrayPlugin`

Add a method to `ArrayPlugin` (the trait used by `SerializedArray::decode()` to dispatch
deserialization per-encoding):

```rust
pub trait ArrayPlugin: 'static + Send + Sync {
    fn id(&self) -> ArrayId;

    fn deserialize(
        &self, dtype: &DType, len: usize, metadata: &[u8],
        buffers: &[BufferHandle], children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef>;

    /// Compute a read plan for this encoding given a row range.
    ///
    /// Returns which byte ranges of this node's buffers are needed, and what
    /// row ranges to propagate to each child. The default reads everything.
    fn plan_read(
        &self,
        metadata: &[u8],
        dtype: &DType,
        len: usize,
        row_range: Range<usize>,
    ) -> VortexResult<ReadPlan> {
        _ = (metadata, dtype, row_range);
        Ok(ReadPlan::all(len))
    }
}
```

The blanket impl `impl<V: VTable> ArrayPlugin for V` forwards to a new optional VTable method
with the same default.

### `ReadPlan`

```rust
/// Describes what an encoding needs from its buffers and children for a given row range.
pub struct ReadPlan {
    pub buffers: Vec<BufferSlice>,
    pub children: Vec<ChildSlice>,
}

/// What byte range of one buffer is needed.
pub enum BufferSlice {
    /// Read only this byte range of the buffer.
    Range(Range<usize>),
    /// Read the entire buffer.
    All,
    /// This buffer is not needed.
    Skip,
}

/// What row range to propagate to one child.
pub enum ChildSlice {
    /// Recurse into this child with the given row range and length.
    Rows { row_range: Range<usize>, len: usize },
    /// Read the full child.
    All,
    /// Skip this child entirely.
    Skip,
}
```

### `SegmentReadPlan` — the coordination point

```rust
/// Tracks refined byte ranges for every buffer in a segment.
/// This is the single place that sees all buffer needs and coordinates IO.
pub struct SegmentReadPlan {
    segment_id: SegmentId,
    /// One entry per buffer in the segment's global buffer list.
    entries: Vec<SegmentBufferEntry>,
}

struct SegmentBufferEntry {
    /// Byte offset of this buffer within the segment (from cumulative padding + lengths).
    segment_offset: usize,
    /// Full byte length of this buffer.
    full_length: usize,
    /// Alignment requirement.
    alignment: Alignment,
    /// The byte range actually needed. Starts as 0..full_length.
    needed: Range<usize>,
}

impl SegmentReadPlan {
    /// Build from flatbuffer metadata. All buffers start fully needed.
    pub fn from_array_tree(segment_id: SegmentId, fb: &fba::Array) -> Self;

    /// Walk the encoding tree, calling plan_read per node, refining buffer ranges.
    pub fn refine(
        &mut self,
        node: &fba::ArrayNode,
        row_range: Range<usize>,
        len: usize,
        dtype: &DType,
        ctx: &ReadContext,
        session: &VortexSession,
    );

    /// Issue coalesced IO for all needed byte ranges. Single round-trip.
    pub async fn fetch(self, source: &dyn SegmentSource) -> VortexResult<Vec<BufferHandle>>;
}
```

#### `refine` walks the tree recursively

```
refine(node, row_range, len, dtype):
    encoding = session.resolve(node.encoding)
    plan = encoding.plan_read(node.metadata, dtype, len, row_range)

    for (i, buffer_slice) in plan.buffers:
        global_idx = node.buffers[i]
        match buffer_slice:
            Range(r) => self.entries[global_idx].narrow(r)
            Skip     => self.entries[global_idx].skip()
            All      => /* leave as-is */

    for (i, child_slice) in plan.children:
        match child_slice:
            Rows { row_range, len } =>
                refine(node.children[i], row_range, len, child_dtype, ...)
            Skip =>
                skip all buffers in child subtree
            All =>
                /* leave child buffers as-is */
```

#### `fetch` coalesces and reads

```
fetch(source):
    // Collect all needed byte ranges mapped to segment offsets
    ranges = entries
        .filter(|e| e.is_needed())
        .map(|e| e.segment_offset + e.needed.start .. e.segment_offset + e.needed.end)

    // Coalesce nearby ranges (reuse existing IoRequestStream logic)
    coalesced = coalesce(ranges)

    // Single IO: request_ranges on the segment
    data = source.request_ranges(segment_id, coalesced).await

    // Slice coalesced results back into individual buffer handles
    entries.map(|e| slice_from_coalesced(data, e))
```

### Extended `SegmentSource`

```rust
pub trait SegmentSource: 'static + Send + Sync {
    /// Existing: fetch an entire segment.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// New: fetch specific byte ranges within a segment, coalesced into minimal reads.
    fn request_ranges(
        &self,
        id: SegmentId,
        ranges: &[Range<usize>],
    ) -> BoxFuture<'static, VortexResult<Vec<BufferHandle>>>;
}
```

`FileSegmentSource` implements `request_ranges` by translating sub-segment byte ranges to
file-level byte ranges using `SegmentSpec.offset`, then feeding them through the existing
`IoRequestStream` coalescing pipeline.

## Per-Encoding `plan_read` Implementations

### Primitive

```rust
fn plan_read(&self, metadata: &[u8], dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    let byte_width = dtype.byte_width();
    Ok(ReadPlan {
        buffers: vec![
            // Values buffer: exact byte range
            BufferSlice::Range(row_range.start * byte_width..row_range.end * byte_width),
        ],
        children: vec![
            // Validity child: propagate row range
            ChildSlice::Rows { row_range: row_range.clone(), len },
        ],
    })
}
```

### Bool

```rust
fn plan_read(&self, _metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    Ok(ReadPlan {
        buffers: vec![
            BufferSlice::Range(row_range.start / 8..row_range.end.div_ceil(8)),
        ],
        children: vec![
            ChildSlice::Rows { row_range: row_range.clone(), len },
        ],
    })
}
```

### BitPacked

```rust
fn plan_read(&self, metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    let meta = BitPackedMetadata::decode(metadata)?;
    let bit_width = meta.bit_width as usize;
    let offset = meta.offset as usize;

    // Align to 1024-element chunk boundaries (same math as SliceReduce)
    let offset_start = row_range.start + offset;
    let offset_stop = row_range.end + offset;
    let block_start = (offset_start / 1024) * 1024;
    let block_stop = offset_stop.div_ceil(1024) * 1024;

    let encoded_start = (block_start / 8) * bit_width;
    let encoded_stop = (block_stop / 8) * bit_width;

    Ok(ReadPlan {
        buffers: vec![
            BufferSlice::Range(encoded_start..encoded_stop),
        ],
        children: vec![
            ChildSlice::All,   // patch_indices: can't refine without data
            ChildSlice::All,   // patch_values
            ChildSlice::All,   // patch_chunk_offsets
            ChildSlice::Rows { row_range: row_range.clone(), len }, // validity
        ],
    })
}
```

### List

```rust
fn plan_read(&self, metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    Ok(ReadPlan {
        buffers: vec![],  // List has no buffers of its own
        children: vec![
            // Elements: CANNOT refine without reading offsets first.
            // Conservative: read all.
            ChildSlice::All,
            // Offsets: n+1 elements, slice to [start..end+1]
            ChildSlice::Rows {
                row_range: row_range.start..row_range.end + 1,
                len: len + 1,
            },
            // Validity: slice to row range
            ChildSlice::Rows { row_range: row_range.clone(), len },
        ],
    })
}
```

List **cannot** refine its elements child in a single round-trip because the element byte
range depends on offset *values*, not just metadata. This is an inherent limitation for
variable-length encodings. The offsets and validity children still benefit.

### Dict

```rust
fn plan_read(&self, _metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    Ok(ReadPlan {
        buffers: vec![],
        children: vec![
            ChildSlice::Rows { row_range: row_range.clone(), len }, // codes: slice
            ChildSlice::All,  // values: need all dictionary entries
        ],
    })
}
```

### Struct

```rust
fn plan_read(&self, metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    let nfields = StructMetadata::decode(metadata)?.nfields;
    Ok(ReadPlan {
        buffers: vec![],
        children: (0..nfields)
            .map(|_| ChildSlice::Rows { row_range: row_range.clone(), len })
            .chain(std::iter::once(
                ChildSlice::Rows { row_range: row_range.clone(), len } // validity
            ))
            .collect(),
    })
}
```

### Masked (validity wrapper)

```rust
fn plan_read(&self, _metadata: &[u8], _dtype: &DType, len: usize, row_range: Range<usize>)
    -> VortexResult<ReadPlan>
{
    Ok(ReadPlan {
        buffers: vec![],
        children: vec![
            ChildSlice::Rows { row_range: row_range.clone(), len }, // inner array
            ChildSlice::Rows { row_range: row_range.clone(), len }, // mask
        ],
    })
}
```

### Default (any encoding without an override)

```rust
fn plan_read(&self, ...) -> VortexResult<ReadPlan> {
    // Conservative: read all buffers, read all children fully.
    Ok(ReadPlan::all(len))
}
```

This means sub-segment slicing is **opt-in**. Encodings that don't implement `plan_read`
still work — they just read everything, same as today.

## Filter Refinement

For filter masks, the conservative approach converts to a bounding row range:

```rust
impl SegmentReadPlan {
    pub fn refine_filter(
        &mut self,
        mask: &Mask,
        node: &fba::ArrayNode,
        len: usize,
        dtype: &DType,
        ctx: &ReadContext,
        session: &VortexSession,
    ) {
        if let Some(range) = mask.bounding_range() {
            self.refine(node, range, len, dtype, ctx, session);
        }
    }
}
```

This is effective when zone-map pruning produces clustered masks (common case). A future
enhancement could handle very sparse masks by computing per-true-bit byte offsets for
fixed-width types.

## Integration with FlatReader

`FlatReader` changes in `projection_evaluation` and `filter_evaluation`:

```rust
// Before (today):
fn projection_evaluation(&self, row_range, expr, mask) {
    let array = self.array_future().await;     // fetch ENTIRE segment
    let array = array.slice(row_range);         // logical slice (cheap but IO already done)
    let array = array.filter(mask);
    array.apply(expr)
}

// After:
fn projection_evaluation(&self, row_range, expr, mask) {
    let fb = parse_flatbuffer(self.layout.array_tree());

    // 1. Build plan — all buffers fully needed
    let mut plan = SegmentReadPlan::from_array_tree(self.layout.segment_id(), &fb);

    // 2. Refine with row range (metadata-only, no IO)
    plan.refine(fb.root(), row_range, self.layout.row_count(), dtype, ctx, session);

    // 3. Optionally refine with filter bounding range
    plan.refine_filter(&mask, fb.root(), ...);

    // 4. Single coordinated fetch
    let buffers = plan.fetch(&self.segment_source).await?;

    // 5. Decode from partial buffers
    let parts = SerializedArray::from_flatbuffer_with_buffers(array_tree, buffers)?;
    let array = parts.decode(dtype, row_range.len(), ctx, session)?;

    // 6. Apply filter + projection as before
    if !mask.all_true() { array = array.filter(mask)?; }
    array.apply(expr)
}
```

Steps 1-3 are pure metadata computation. Step 4 is one IO. Steps 5-6 are CPU-only
with the execution loop running unchanged.

## Interaction with the Execution Loop

The execution loop (`execute_until`) is **unchanged**. It still runs reduce → reduce_parent
→ execute_parent → execute on arrays with materialized buffers. Sub-segment slicing happens
**before** the execution loop, in the async FlatReader layer.

The two systems complement each other:

| Layer | What it does | When it runs |
|-------|-------------|-------------|
| `plan_read` + `SegmentReadPlan` | Determines minimal byte ranges to fetch | Before IO, in FlatReader |
| `SliceReduce` | Pushes slices through live arrays | After IO, in optimizer/executor |
| Execution loop | Decodes encoded arrays to canonical | After IO, CPU-only |

`plan_read` and `SliceReduce` encode **parallel knowledge** — they both know how a row range
maps through an encoding's structure. The difference is the level they operate at:

- `SliceReduce` operates on **live `ArrayView`** with materialized buffers
- `plan_read` operates on **serialized `&[u8]` metadata** without any buffer data

For encodings where the mapping is simple arithmetic (Primitive, Bool, BitPacked), the two
implementations are small and unlikely to diverge. For encodings where the mapping is
data-dependent (List, RLE), `plan_read` is conservative (`All`) while `SliceReduce` can be
precise because it has the data.

## Concrete Savings Example

Schema: `Struct { ts: Primitive<i64>, name: List<VarBin<u8>>, flags: BitPacked<u8, 3> }`
1M rows, reading rows 1000..2000.

| Buffer | Full segment | With plan_read | Savings |
|--------|-------------|---------------|---------|
| ts values (i64) | 8 MB | 8 KB | 99.9% |
| ts validity | 125 KB | ~125 B | 99.9% |
| name offsets (i32) | 4 MB | 4 KB | 99.9% |
| name elements | 50 MB | 50 MB | 0% (data-dependent) |
| name validity | 125 KB | ~125 B | 99.9% |
| flags packed (3-bit) | 375 KB | ~384 B (1 chunk) | 99.9% |
| flags validity | 125 KB | ~125 B | 99.9% |
| **Total** | **~63 MB** | **~50 MB** | **~20%** |

The `name.elements` buffer dominates because List can't refine it. For schemas with mostly
fixed-width columns (timestamps, IDs, metrics), savings are much larger:

Schema: `Struct { ts: Primitive<i64>, id: Primitive<u64>, value: Primitive<f64> }`
1M rows, reading rows 1000..2000.

| Buffer | Full segment | With plan_read | Savings |
|--------|-------------|---------------|---------|
| ts values | 8 MB | 8 KB | 99.9% |
| id values | 8 MB | 8 KB | 99.9% |
| value values | 8 MB | 8 KB | 99.9% |
| 3x validity | 375 KB | ~375 B | 99.9% |
| **Total** | **~24 MB** | **~24 KB** | **99.9%** |

## Incremental Rollout

1. **Phase 1**: Implement `ReadPlan`, `SegmentReadPlan`, `SegmentSource::request_ranges`.
   Add `plan_read` to `ArrayPlugin` with the default (read-all) implementation.
   FlatReader uses the new path; all encodings behave as today.

2. **Phase 2**: Implement `plan_read` for leaf encodings: Primitive, Bool.
   These are trivial and cover the most common large buffers.

3. **Phase 3**: Implement `plan_read` for compressed encodings: BitPacked, Delta, FoR, ALP.
   These have chunk-aligned access patterns derivable from metadata.

4. **Phase 4**: Implement `plan_read` for structural encodings: Struct, Dict, Masked, List.
   These mainly propagate row ranges to children (List conservatively).

5. **Phase 5**: Evaluate whether filter-aware refinement (beyond bounding-range) is worth
   implementing for very sparse masks.

## Open Questions

1. **Relationship between `plan_read` and `SliceReduce`**: Could we generate one from the
   other, or share a common description? The risk of divergence is low for simple encodings
   but worth watching.

2. **Two-pass reads for List**: Could we support an optional second pass where, after reading
   offsets, we refine the elements range and issue a targeted follow-up read? This would only
   be beneficial when the elements buffer is very large relative to the offsets.

3. **`request_ranges` design**: Should this be a new method on `SegmentSource`, or should the
   existing `request` method accept optional byte ranges? The latter is simpler but changes
   the trait's API for all implementors.

4. **Inline array tree requirement**: Sub-segment slicing requires the flatbuffer metadata
   to be available *before* IO. With `FLAT_LAYOUT_INLINE_ARRAY_NODE` this is already the case.
   Without it, the metadata lives inside the segment itself, creating a chicken-and-egg
   problem. Should inlining become the default?

5. **Field mask integration**: FlatReader receives a `FieldMask` from projection pushdown.
   `plan_read` for Struct could use this to `Skip` entire child subtrees for non-projected
   fields, compounding the savings.
