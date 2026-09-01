# V1 `LayoutReader`

V1 represents a scan as a stateful tree of layout-specific readers. It is the established behavior
and compatibility baseline for the other models in this comparison.

## End-to-end flow

```text
LayoutRef
  -> LayoutVTable::reader(...)
LayoutReader tree
  -> register_splits(...)
fixed row splits
  -> pruning_evaluation(...)
  -> filter_evaluation(...)
  -> projection_evaluation(...)
ArrayFuture per split
  -> ordered or unordered concurrent stream
```

There is no separate, generic physical-plan tree. The `LayoutReader` tree performs planning and
execution responsibilities together.

## Core contract

The trait in `vortex-layout/src/reader.rs` exposes three evaluation paths over an exact row range:

```rust
fn pruning_evaluation(
    &self,
    row_range: &Range<u64>,
    expr: &BoundExpression,
    mask: Mask,
) -> VortexResult<MaskFuture>;

fn filter_evaluation(
    &self,
    row_range: &Range<u64>,
    expr: &BoundExpression,
    mask: MaskFuture,
) -> VortexResult<MaskFuture>;

fn projection_evaluation(
    &self,
    row_range: &Range<u64>,
    expr: &BoundExpression,
    mask: MaskFuture,
) -> VortexResult<ArrayFuture>;
```

The contracts are intentionally different:

- pruning returns a proof mask whose false rows cannot satisfy the expression; it need not already
  be intersected with the input mask;
- filtering returns a mask equal in dense length to the input range and must intersect its result
  with the input mask; and
- projection returns a compact array whose length is exactly the true count of the resolved input
  mask.

The scan driver chooses the range and therefore the output batch boundary. Readers can specialize
how the request is fulfilled, but they cannot return a shorter or longer row prefix.

## Reader responsibilities

A reader may own or cache:

- layout metadata and decoded indexes;
- lazily constructed child readers;
- expression partitions and layout-specific rewrites;
- segment-read state and shared futures;
- split discovery logic; and
- pruning, filter, and projection implementations.

This concentration of responsibilities is why V1 is capable but difficult to optimize globally.
An optimizer cannot inspect or replace a generic `Take` or `Pack` node because those operations are
implicit in reader implementations.

## Split execution

`vortex-layout/src/scan/tasks.rs` constructs one future for each selected split. Within a split it:

1. starts with the scan selection mask;
2. applies pruning for each filter conjunct;
3. evaluates remaining conjuncts in adaptive order;
4. constructs projection evaluation with the unresolved filter mask; and
5. awaits the final mask and projected array.

Constructing projection before awaiting the filter mask is deliberate. It lets readers register
segment reads early, so predicate and projection paths can share an in-flight request. A reader is
encouraged to defer consuming the mask until I/O has been registered or completed.

This is useful latency hiding, but the I/O scheduler sees the consequences indirectly through
future construction. Required reads, speculative reads, priorities, and byte costs are not part of
the `LayoutReader` interface.

## How composite layouts execute

### Chunked

The chunked reader intersects the caller's exact range with each relevant chunk, slices the mask,
delegates to the child readers, and concatenates their arrays in chunk order. Chunk boundaries also
provide natural split candidates.

### Struct

The struct reader partitions expressions by field, evaluates field readers over the same range and
mask, and packs their compact arrays. It can cache partitioned expressions, but the partitioning is
reader-specific rather than a generic plan rewrite.

### Dictionary

The dictionary reader treats codes and values differently. Codes use the outer row domain. Values
use the dictionary domain and may be read once or restricted to referenced values. This behavior is
specialized inside the reader rather than represented as a generic `Take` operator.

### List

The list reader bridges multiple coordinate systems:

- outer list rows;
- offsets, including the extra terminal offset;
- element rows derived from the selected offsets; and
- optional outer validity.

It also maps split hints from the element domain back to outer rows. This is necessarily heuristic
without reading offsets and illustrates why fixed outer split discovery is awkward for nested data.

### Zoned

The zoned reader evaluates pruning information from zone metadata and delegates surviving value
work to its data child. Pruning and value execution remain methods of the same reader tree.

## Strengths

- It is the mature implementation with broad layout and expression coverage.
- Layout-specific code can make informed decisions using physical metadata.
- Exact range and mask contracts make row-wise parent composition simple.
- Early future construction overlaps and deduplicates I/O in existing segment sources.
- Fixed split tasks provide straightforward concurrency and ordered output.

## Limitations

- Physical planning, expression pushdown, runtime state, and execution are coupled.
- Generic rewrites across layout types are difficult.
- Expressions can be repartitioned at each split boundary.
- Every subtree is paced by an externally selected exact range.
- Fixed split size controls both concurrency and batching, even when a layout has a better natural
  unit.
- Scheduler policy cannot directly reason about logical read bytes or task phase.
- Nested and lookup layouts must force different row domains into one split-oriented API.

## Role in a replacement

V1 should remain the semantic oracle while a new executor is introduced. Differential tests should
compare V1 and the new path for:

- nullable filters and three-valued boolean behavior;
- sparse masks and rank-based compaction;
- dictionary codes and unused values;
- empty lists, null lists, and very large lists;
- selections crossing chunk and zone boundaries;
- row-index offsets; and
- fallible expressions evaluated only on demanded rows.

## Implementation map

- Trait and postconditions: `vortex-layout/src/reader.rs`
- Split task orchestration: `vortex-layout/src/scan/tasks.rs`
- Split discovery: `vortex-layout/src/scan/split_by.rs`
- Chunk slicing and concatenation: `vortex-layout/src/layouts/chunked/reader.rs`
- Struct expression partitioning: `vortex-layout/src/layouts/struct_/reader.rs`
- Dictionary specialization: `vortex-layout/src/layouts/dict/reader.rs`
- List coordinate translation: `vortex-layout/src/layouts/list/reader.rs`
- Zoned pruning: `vortex-layout/src/layouts/zoned/reader.rs`
