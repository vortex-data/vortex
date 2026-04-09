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
+-------------------------- segment ---------------------------+
| [pad][buf 0][pad][buf 1][pad][buf 2]...[flatbuffer][u32 len] |
+--------------------------------------------------------------+
```

The flatbuffer suffix (the "array tree") describes the encoding tree:

```
Array
  root: ArrayNode
    encoding: u16          <- which encoding (Primitive, BitPacked, Dict, ...)
    metadata: [u8]         <- encoding-specific (bit_width, offset, ptype, ...)
    buffers:  [u16]        <- indices into the global Buffer descriptor list
    children: [ArrayNode]  <- recursive
  buffers: [Buffer]        <- global list: { padding, alignment_exponent, length }
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

Encodings implement this to push slices down through their structure:

- **Primitive**: `buffer_handle().slice_typed::<T>(range)` -- byte-range slice on values buffer
- **BitPacked**: compute chunk-aligned byte range from `bit_width`/`offset` metadata,
  then `packed().slice(encoded_start..encoded_stop)`
- **List**: keep elements unchanged, `offsets.slice(start..end+1)`, `validity.slice(range)`
- **Struct**: recurse into each field with the same row range
- **Dict**: slice codes, keep values unchanged

Critically, **these implementations only call `BufferHandle::slice()` and
`BufferHandle::slice_typed::<T>()`**. Both are zero-copy today -- they adjust byte
offsets without reading data. This property is the foundation of the design.

## Background: The Execution Loop

The execution loop (`execute_until`) iteratively transforms arrays toward canonical form:

```
for each iteration:
    1. Check if done (matches target or is canonical)
    2. Try reduce / reduce_parent   -- metadata-only rewrites
    3. Try execute_parent            -- child-driven fused execution
    4. Try execute                   -- encoding's own decode step
       -> ExecuteSlot(i): push stack, execute child first
       -> Done: continue
```

Steps 2 are metadata-only (no buffer reads). Steps 3-4 may read buffer data. The loop runs
synchronously; all IO must be complete before it starts.

`SliceReduce` is triggered in step 2 via `reduce_parent`: when a `SliceArray` wraps a child
that implements `SliceReduce`, the optimizer calls `SliceReduce::slice()` which pushes the
row range down, calling `.slice()` on the child's buffers.

## Design: Segment Variant on BufferHandle

### Core Idea

Add a third variant to `BufferHandle::Inner`:

```rust
enum Inner {
    Host(ByteBuffer),
    Device(Arc<dyn DeviceBuffer>),
    Segment(SegmentBufferRef),        // <- new: unfetched, refinable
}
```

A `SegmentBufferRef` represents a byte range within a segment that has **not been fetched**.
When `BufferHandle::slice()` is called on a `Segment` handle, it narrows the byte range
instead of slicing real data. **No encoding code changes.**

### `SegmentBufferRef`

```rust
pub struct SegmentBufferRef {
    /// Which segment this buffer lives in.
    segment_id: SegmentId,
    /// Index of this buffer in the segment's global buffer list.
    buffer_index: u32,
    /// Byte offset of this buffer's start within the segment.
    segment_offset: usize,
    /// The byte range within this buffer that is actually needed.
    /// Starts as 0..full_length. Narrowed by slice() calls.
    needed: Range<usize>,
    /// Alignment requirement from the flatbuffer descriptor.
    alignment: Alignment,
    /// Filled when materialized. Shared via Arc so cloned handles see the data.
    resolved: Arc<OnceLock<ByteBuffer>>,
}
```

### How BufferHandle operations work on Segment

```rust
impl BufferHandle {
    pub fn slice(&self, range: Range<usize>) -> Self {
        match &self.0 {
            Inner::Host(host) => BufferHandle::new_host(host.slice(range)),
            Inner::Device(device) => BufferHandle::new_device(device.slice(range)),
            Inner::Segment(seg) => BufferHandle(Inner::Segment(seg.slice(range))),
        }
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            Inner::Host(b) => b.len(),
            Inner::Device(d) => d.len(),
            Inner::Segment(s) => s.needed.len(),
        }
    }

    pub fn to_host_sync(&self) -> ByteBuffer {
        match &self.0 {
            Inner::Host(b) => b.clone(),
            Inner::Device(d) => d.copy_to_host_sync(ALIGNMENT_TO_HOST_COPY),
            Inner::Segment(s) => s.resolved.get()
                .expect("Segment buffer not yet materialized")
                .clone(),
        }
    }
}
```

And on `SegmentBufferRef`:

```rust
impl SegmentBufferRef {
    fn slice(&self, range: Range<usize>) -> Self {
        SegmentBufferRef {
            needed: (self.needed.start + range.start)..(self.needed.start + range.end),
            // New OnceLock -- this narrowed handle gets its own resolution.
            resolved: Arc::new(OnceLock::new()),
            ..*self
        }
    }

    /// The byte range to read from the segment.
    fn segment_byte_range(&self) -> Range<usize> {
        (self.segment_offset + self.needed.start)..(self.segment_offset + self.needed.end)
    }
}
```

### Why existing SliceReduce works unchanged

Trace through `Primitive::slice()`:

```rust
impl SliceReduce for Primitive {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_native_ptype!(array.ptype(), |T| {
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().slice_typed::<T>(range.clone()),
                //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                //    If buffer_handle() returns Segment, slice_typed::<T>
                //    narrows needed by (range.start * sizeof(T))..(range.end * sizeof(T))
                //    Returns a new Segment handle. No data accessed.
                T::PTYPE,
                array.validity()?.slice(range)?,
            )
            .into_array()
        });
        Ok(Some(result))
    }
}
```

Trace through `BitPacked::slice()`:

```rust
impl SliceReduce for BitPacked {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // ... chunk boundary math ...
        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;

        Ok(Some(BitPacked::try_new(
            array.packed().slice(encoded_start..encoded_stop),
            //              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            //              packed() returns Segment handle.
            //              .slice() narrows to chunk-aligned byte range.
            // ...
        )))
    }
}
```

Trace through `List::slice()`:

```rust
impl SliceReduce for List {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(ListArray::new(
            array.elements().clone(),                      // unchanged -- Segment stays full range
            array.offsets().slice(range.start..range.end + 1)?, // narrows offsets Segment
            array.validity()?.slice(range)?,                    // narrows validity Segment
        ).into_array()))
    }
}
```

**Every existing `SliceReduce` implementation becomes a buffer refinement planner for free.**

### Deserialization with Segment handles

Add a new constructor to `SerializedArray`:

```rust
impl SerializedArray {
    /// Create a SerializedArray where buffers are lazy Segment references.
    /// No IO is performed; buffers are SegmentBufferRefs that track byte ranges.
    pub fn from_flatbuffer_with_segment_refs(
        array_tree: ByteBuffer,
        segment_id: SegmentId,
    ) -> VortexResult<Self> {
        let (fb_buffer, flatbuffer_loc) = Self::validate_array_tree(array_tree)?;
        let fb_array = unsafe { fba::root_as_array_unchecked(fb_buffer.as_ref()) };

        let mut offset = 0;
        let buffers = fb_array
            .buffers()
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(idx, fb_buf)| {
                offset += fb_buf.padding() as usize;
                let buffer_len = fb_buf.length() as usize;
                let alignment = Alignment::from_exponent(fb_buf.alignment_exponent());

                let handle = BufferHandle::new_segment(SegmentBufferRef {
                    segment_id,
                    buffer_index: idx as u32,
                    segment_offset: offset,
                    needed: 0..buffer_len,
                    alignment,
                    resolved: Arc::new(OnceLock::new()),
                });

                offset += buffer_len;
                Ok(handle)
            })
            .collect::<VortexResult<Arc<[_]>>>()?;

        Ok(SerializedArray { flatbuffer: fb_buffer, flatbuffer_loc, buffers })
    }
}
```

This mirrors `from_flatbuffer_and_segment` but creates `Segment` handles instead of
slicing real data. The byte offset computation is identical.

### Materialization

After the optimizer refines all Segment handles, a `materialize` step collects them,
issues one coalesced fetch, and fills the `OnceLock`s:

```rust
/// Collect all Segment buffer handles in the array tree, fetch their needed
/// byte ranges in one coalesced IO, and resolve them to Host buffers.
pub async fn materialize(
    array: &ArrayRef,
    source: &dyn SegmentSource,
) -> VortexResult<()> {
    // 1. Walk the array tree, collect all Segment handles
    let segment_refs: Vec<SegmentBufferRef> = array
        .depth_first_traversal()
        .flat_map(|a| a.buffers())
        .filter_map(|bh| bh.as_segment().cloned())
        .collect();

    if segment_refs.is_empty() {
        return Ok(());
    }

    // 2. Group by segment_id, compute byte ranges
    let mut by_segment: HashMap<SegmentId, Vec<&SegmentBufferRef>> = HashMap::new();
    for seg in &segment_refs {
        by_segment.entry(seg.segment_id).or_default().push(seg);
    }

    // 3. For each segment, coalesce and fetch
    for (segment_id, refs) in by_segment {
        let ranges: Vec<Range<usize>> = refs.iter()
            .map(|r| r.segment_byte_range())
            .collect();

        let data = source.request_ranges(segment_id, &ranges).await?;

        // 4. Resolve each handle
        for (seg_ref, buffer) in refs.iter().zip(data) {
            seg_ref.resolved.set(buffer.unwrap_host())
                .expect("buffer already resolved");
        }
    }

    Ok(())
}
```

### The Async Segment Executor

The current execution loop (`execute_until` in `vortex-array`) is synchronous -- it
assumes all buffer data is already in memory. We add a second executor in `vortex-layout`
that is async, knows about `SegmentSource`, and handles the reduce -> fetch -> execute
cycle:

```rust
/// An async executor that can load segment buffers on demand.
///
/// Lives in `vortex-layout` because it needs `SegmentSource` for IO.
/// This is the async counterpart to the sync `execute_until` in `vortex-array`.
pub struct SegmentExecutor {
    source: Arc<dyn SegmentSource>,
    session: VortexSession,
}

impl SegmentExecutor {
    pub fn new(source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self { source, session }
    }

    /// Execute an array that may contain unfetched Segment buffer handles.
    ///
    /// 1. Run optimizer (reduce to fixpoint) -- refines Segment byte ranges
    /// 2. Collect all Segment handles, coalesce, single async fetch
    /// 3. Run sync execute_until on resolved data
    pub async fn execute<M: Matcher>(&self, array: ArrayRef) -> VortexResult<ArrayRef> {
        // Phase 1: Reduce to fixpoint.
        // SliceReduce calls buffer.slice() on Segment handles -> refines byte ranges.
        // This is metadata-only, no IO.
        let array = array.optimize()?;

        // Phase 2: Materialize all Segment handles in one coalesced fetch.
        self.materialize(&array).await?;

        // Phase 3: All buffers are now Host data. Run sync execution.
        let mut ctx = self.session.create_execution_ctx();
        array.execute_until::<M>(&mut ctx)
    }

    /// Collect all Segment buffer handles in the array tree, coalesce nearby
    /// byte ranges within the same segment, fetch in a single IO batch, and
    /// fill each handle's OnceLock with the resolved data.
    async fn materialize(&self, array: &ArrayRef) -> VortexResult<()> {
        // Walk the array tree, collect all Segment handles
        let segment_refs: Vec<SegmentBufferRef> = array
            .depth_first_traversal()
            .flat_map(|a| a.buffer_handles())
            .filter_map(|bh| bh.as_segment().cloned())
            .collect();

        if segment_refs.is_empty() {
            return Ok(());
        }

        // Group by segment_id
        let mut by_segment: HashMap<SegmentId, Vec<SegmentBufferRef>> = HashMap::new();
        for seg in segment_refs {
            by_segment.entry(seg.segment_id).or_default().push(seg);
        }

        // For each segment, coalesce nearby ranges and fetch
        for (segment_id, refs) in by_segment {
            let ranges: Vec<Range<usize>> = refs.iter()
                .map(|r| r.segment_byte_range())
                .collect();

            // Single IO per segment, coalesced
            let data = self.source.request_ranges(segment_id, &ranges).await?;

            // Resolve each handle's OnceLock
            for (seg_ref, buffer) in refs.iter().zip(data) {
                seg_ref.resolved.set(buffer.unwrap_host())
                    .expect("buffer already resolved");
            }
        }

        Ok(())
    }
}
```

### How FlatReader uses the SegmentExecutor

```rust
impl FlatReader {
    fn segment_executor(&self) -> SegmentExecutor {
        SegmentExecutor::new(self.segment_source.clone(), self.session.clone())
    }
}

impl LayoutReader for FlatReader {
    fn projection_evaluation(&self, row_range, expr, mask) {
        let executor = self.segment_executor();
        let array_tree = self.layout.array_tree().unwrap();
        let segment_id = self.layout.segment_id();
        let dtype = self.layout.dtype().clone();
        let row_count = self.layout.row_count() as usize;
        let ctx = self.layout.array_ctx().clone();
        let session = self.session.clone();

        Ok(async move {
            // 1. Deserialize with Segment handles (no IO)
            let parts = SerializedArray::from_flatbuffer_with_segment_refs(
                array_tree, segment_id,
            )?;
            let array = parts.decode(&dtype, row_count, &ctx, &session)?;

            // 2. Apply slice + filter bounding range (refines Segment handles)
            let mut array = array.slice(row_range)?;
            let mask = mask.await?;
            if let Some(bounds) = mask.bounding_range() {
                array = array.slice(bounds)?;
            }

            // 3. Async execute: reduce -> fetch -> execute in one call
            let array = executor.execute::<Columnar>(array).await?;

            // 4. Filter and project on resolved data
            let array = array.filter(mask)?;
            array.apply(&expr)
        }.boxed())
    }
}
```

The `SegmentExecutor` is the single integration point between the array execution
model and the IO model. FlatReader doesn't need to manually orchestrate reduce,
materialize, and execute -- it hands the array to the executor and gets back
resolved data.

### Why a separate executor, not modifying the existing one

The sync executor in `vortex-array` has no dependency on `vortex-layout` or any IO
concepts. It operates purely on in-memory arrays. The async `SegmentExecutor` in
`vortex-layout` adds IO awareness:

| | Sync executor (`vortex-array`) | Async executor (`vortex-layout`) |
|---|---|---|
| **Crate** | `vortex-array` | `vortex-layout` |
| **Async** | No | Yes |
| **Knows about IO** | No | Yes (`SegmentSource`) |
| **Handles Segment buffers** | No (expects Host/Device) | Yes (materialize) |
| **Used by** | Anything with in-memory arrays | FlatReader, layout readers |

The async executor delegates to the sync executor for the actual compute work.
It just wraps it with the reduce-refine and IO materialization steps.

## What Changes, What Doesn't

### Changes

| Component | Crate | Change |
|-----------|-------|--------|
| `BufferHandle` | `vortex-array` | Add `Inner::Segment(SegmentBufferRef)` variant |
| `BufferHandle::slice()` | `vortex-array` | Handle Segment: narrow byte range |
| `BufferHandle::len()` | `vortex-array` | Handle Segment: return needed range length |
| `BufferHandle::to_host*()` | `vortex-array` | Handle Segment: read from OnceLock |
| `SerializedArray` | `vortex-array` | Add `from_flatbuffer_with_segment_refs()` |
| `SegmentSource` | `vortex-layout` | Add `request_ranges()` method |
| `SegmentExecutor` | `vortex-layout` | **New**: async executor with IO awareness |
| `FlatReader` | `vortex-layout` | Use `SegmentExecutor` for lazy load + execute |

### Unchanged

| Component | Why unchanged |
|-----------|---------------|
| Every `SliceReduce` impl | Already calls `buffer.slice()` / `buffer.slice_typed()` |
| Sync execution loop | Runs after materialization, sees only Host buffers |
| `IoRequestStream` coalescing | Reused by `request_ranges` implementation |
| Optimizer / reduce rules | Already runs SliceReduce via `SliceReduceAdaptor` |
| Other `LayoutReader` impls | Only FlatReader changes; Chunked/Zoned delegate to children |

**No new VTable methods. No per-encoding `plan_read`. No logic duplication.**

## How Encodings Participate

Every encoding that already implements `SliceReduce` automatically participates in
sub-segment slicing. The existing `slice()` calls on `BufferHandle` become refinement
operations when the handle is a Segment variant.

Encodings that only implement `SliceKernel` (which needs buffer data) will have their
buffers fetched at full size -- the optimizer can't push slices past them. This is the
correct conservative behavior.

### Per-encoding behavior (all automatic)

| Encoding | What SliceReduce does to its buffers | Effect on Segment handles |
|----------|--------------------------------------|---------------------------|
| Primitive | `buf.slice_typed::<T>(range)` | Narrows to exact byte range |
| Bool | `buf.slice(start/8..ceil(end/8))` | Narrows to bit-aligned bytes |
| BitPacked | `packed.slice(chunk_start..chunk_stop)` | Narrows to chunk-aligned range |
| List | elements unchanged, offsets sliced | Elements full, offsets narrowed |
| Dict | codes sliced, values unchanged | Codes narrowed, values full |
| Struct | recurses into children | Each field refined independently |
| Masked | child + mask both sliced | Both narrowed |

### Encodings without SliceReduce

Some encodings only have `SliceKernel` (needs buffer data to slice):
- **Chunked**: needs offsets to determine chunk boundaries
- **RLE**: needs run-ends to determine boundaries
- **Sparse**: needs indices to determine which values

For these, the `SliceArray` wrapper survives optimization. The buffers stay at full
range. After materialization, the execution loop handles them as it does today.

## Filter Refinement

For filter masks, convert to a bounding row range before slicing:

```rust
let array = array.slice(row_range)?;
// If we know the filter mask's bounding range, slice further:
if let Some(bounding) = mask.bounding_range() {
    array = array.slice(bounding)?;
}
materialize(&array, &segment_source).await?;
```

The second `slice()` composes with the first (Slice(Slice(x)) reduces to Slice(x)
with combined range). Zone-map pruning often produces clustered masks where this
is very effective.

## Concrete Savings

Schema: `Struct { ts: Primitive<i64>, id: Primitive<u64>, value: Primitive<f64> }`
1M rows, reading rows 1000..2000.

| Buffer | Full segment | With sub-segment slicing | Savings |
|--------|-------------|--------------------------|---------|
| ts values (i64) | 8 MB | 8 KB | 99.9% |
| id values (u64) | 8 MB | 8 KB | 99.9% |
| value values (f64) | 8 MB | 8 KB | 99.9% |
| 3x validity | 375 KB | ~375 B | 99.9% |
| **Total** | **~24 MB** | **~24 KB** | **99.9%** |

Schema with variable-length data:
`Struct { ts: Primitive<i64>, name: List<VarBin<u8>>, flags: BitPacked<u8, 3> }`

| Buffer | Full segment | With sub-segment slicing | Savings |
|--------|-------------|--------------------------|---------|
| ts values | 8 MB | 8 KB | 99.9% |
| name offsets | 4 MB | 4 KB | 99.9% |
| name elements | 50 MB | 50 MB | 0% (data-dependent) |
| flags packed | 375 KB | ~384 B | 99.9% |
| **Total** | **~63 MB** | **~50 MB** | **~20%** |

Variable-length types (List) can't refine their data buffer without reading offsets first.
Everything else collapses.

## Incremental Rollout

1. **Phase 1**: Add `Inner::Segment` to `BufferHandle` with `SegmentBufferRef`. Implement
   `slice()`, `len()`, `to_host*()` for the new variant.

2. **Phase 2**: Add `SerializedArray::from_flatbuffer_with_segment_refs()`.
   Add `SegmentSource::request_ranges()` and implement in `FileSegmentSource`.

3. **Phase 3**: Implement `materialize()` and integrate into `FlatReader`.
   At this point, sub-segment slicing works for every encoding that implements
   `SliceReduce` -- no per-encoding work needed.

4. **Phase 4**: Measure. Profile real workloads. Tune the IO coalescing thresholds
   for sub-segment ranges.

5. **Phase 5**: Evaluate filter-aware refinement beyond bounding-range.

## Open Questions

1. **Inline array tree**: Sub-segment slicing requires the flatbuffer metadata before IO.
   `FLAT_LAYOUT_INLINE_ARRAY_NODE` provides this. Should inlining become the default?

2. **OnceLock vs re-creation**: The OnceLock approach allows interior mutability on the
   existing array tree. An alternative is to re-deserialize from the flatbuffer with
   fetched partial buffers (using `from_flatbuffer_and_segment_with_overrides`). The
   OnceLock approach avoids double-deserialization but adds complexity to BufferHandle.

3. **Multi-round materialization**: The `SegmentExecutor` currently does one
   reduce-fetch-execute cycle. If a future encoding needs multiple rounds (e.g., read
   offsets first, then refine data ranges), the executor could loop:
   `reduce -> materialize -> check for remaining Segment handles -> repeat if needed`.
   The `SegmentBufferRef` on `BufferHandle` makes this straightforward -- just check
   if any Segment handles remain after execute and loop back.

4. **Field mask integration**: `FlatReader` receives a `FieldMask` from projection pushdown.
   For Struct columns, non-projected fields could have their Segment handles skipped
   entirely during deserialization, compounding savings.

5. **Executor reuse across evaluations**: The `SegmentExecutor` is cheap to create (just
   holds `Arc` references). But if `FlatReader` runs multiple evaluations on the same
   segment (filter + projection), could the executor cache the materialized segment
   data across calls? Today `FlatReader` uses `SharedArrayFuture` for this -- the
   executor should integrate with that caching.
