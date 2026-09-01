# Current Plan v2 Execution

Plan v2 introduces a layout-independent physical operator tree and executes it through
`vortex-scan-v2`. It separates planning from stored layout identity, but execution is still driven
by exact, externally chosen split ranges.

## End-to-end flow

```text
LayoutRef
  -> vortex_layout::plan::lower
source PlanRef
  -> add RowIdx and Eval operators
  -> optimize projection, filter, and pruning plans
optimized PlanRef trees
  -> collect plan-aware split boundaries
fixed row splits
  -> PlanVTable::execute(range, mask) recursively
ArrayRef per split
  -> ordered or unordered concurrent stream
```

`vortex-scan-v2/src/scan_builder.rs` uses this path directly. The comment in
`vortex-layout/src/plan/lower.rs` still describes lowering as test support, but the scan-v2 builder
is a production caller on this branch.

## Plan representation

`PlanRef` is an `Arc` to one allocation containing:

- the operator ID;
- output dtype;
- row count;
- generic child storage; and
- an erased, operator-specific data tail.

Common field access does not dynamically dispatch. Typed `Plan<V>` views recover the concrete
vtable and `PlanData` when an operator implementation or rewrite needs them.

`PlanChildren` holds ordered `OnceCell<PlanRef>` slots. Layout lowering can install a closure that
owns the source layout and lowers one child on first access. Rewrites replace the generic child
container and call `PlanVTable::with_children` so an operator can validate children and rebuild
derived metadata such as `Concat` row offsets.

This representation is close to immutable, but lazily populated children are interior caches.
Runtime caches should not be added to this object if the plan is to remain safely reusable across
scans.

## Layout lowering

The current lowering function maps stored layout kinds to work-oriented operators:

| Layout | Initial plan operator |
| --- | --- |
| Flat | `SegmentScan` |
| Chunked | `Concat` |
| Struct | `Pack` |
| Dictionary | `Take` |
| List | `ListPack` |
| Zoned or legacy statistics | `Zoned` |

The distinction is important. A `Take` describes lookup work regardless of which layout produced
it, and `Concat` rewrites can match shape without knowing that the source was a chunked layout.

The current lowering function is a central type switch. A stable version should move construction
behind a layout vtable hook or registry so third-party layouts can produce plans without editing a
central module.

## Planning and optimization

The scan builder wraps the source in `RowIdx`, normalizes projection and filter expressions, then
creates `Eval` plans. Generic rules push or simplify work through physical operators. Separate
optimized roots are retained for:

- projection;
- a parallel predicate plan or adaptively ordered conjunct plans; and
- a pruning falsifier that is accepted only when it uses pruning sources.

The optimizer works over common child storage, so rules can replace children without each operator
reimplementing traversal.

## Execution contract

Each operator implements:

```rust
fn execute(
    plan: &Plan<Self>,
    ctx: &PlanExecutionContext,
    row_range: &Range<u64>,
    mask: MaskFuture,
) -> VortexResult<PlanArrayFuture>;
```

The range is in the operator's row domain. The mask length must equal the dense range length. The
returned array length must equal the true count of the resolved mask.

`PlanExecutionContext` currently contains only the segment source and session. Calling `execute`
recursively creates boxed futures for the requested portion of the plan. There is no separately
opened, persistent executor tree with mutable cursors, backpressure state, or parent buffers.

## Operator behavior

### `SegmentScan`

Reads the requested segment range, decodes the array, and applies the requested mask. It is the
physical leaf for flat data.

### `Concat`

Intersects the exact request with all overlapping chunks, slices the mask into child coordinates,
executes those children, then returns one array or a `ChunkedArray` in order.

### `Pack`

Executes every field and optional validity child over the same exact range and mask. Exact child
cardinality makes struct construction straightforward.

### `Take`

Executes codes over the requested outer range and mask. It currently executes the complete values
domain with an all-true mask, then constructs and optimizes a dictionary array. Codes and values
therefore do not share a row coordinate system even though they are children of one operator.

### `ListPack`

Reads `row_count + 1` offsets, derives one contiguous element range, reads all elements in that
range, reconstructs the list, and finally filters outer rows. This preserves list semantics but can
make one outer split expand into a much larger element request.

### `Eval` and row-index operators

`Eval` applies a bound expression to its child result. Row-index operators introduce global row
identity while preserving or partitioning work across compatible children.

### `Zoned`

The normal data path delegates to the data plan. The pruning path reads zone information, produces
a proof, and can cache shared zone state. This cache is an example of runtime state that should move
to a separate execution object in the proposed design.

## Split scheduling

`vortex-scan-v2/src/splits.rs` knows how to descend through specific plan operators to collect
boundaries. Natural spans are subdivided toward roughly 100,000 rows. The repeated scan creates one
future per selected split and applies configured concurrency and output ordering.

Within each split, `vortex-scan-v2/src/tasks.rs` performs:

```text
pruning proof -> residual filter -> projection -> mapper
```

Projection execution is registered before the filter mask is awaited. This preserves V1's ability
to share in-flight reads between filter and projection, but read cost and phase remain implicit in
the futures returned by operators.

## Strengths

- Physical operators are generic and rewriteable.
- Layout identity no longer dictates every optimization rule.
- Plan display and common traversal make the chosen work inspectable.
- Lazy lowering avoids eagerly expanding unused subtrees.
- Exact range and mask contracts make the first executor simple and easy to compare with V1.

## Limitations

- The caller still selects every output boundary.
- Parent and child progress cannot be independent.
- Split collection depends on knowledge of concrete plan operators.
- Recursive future construction is an execution mechanism, not a scheduler-visible execution
  graph.
- Runtime caches can leak into reusable plan data.
- I/O admission, memory pressure, priority, and prefetch policy are not explicit.
- `Take` and `ListPack` expose the difficulty of forcing multiple coordinate domains into one exact
  request contract.

## Best role going forward

Plan v2 should remain the physical intermediate representation. Its `execute` callback should
eventually become, or be complemented by, an `open` callback that creates a per-scan execution
node. The plan would then describe work while the execution node owns cursors, buffers, read
handles, and progress.

## Implementation map

- Plan vtable and execution contract: `vortex-layout/src/plan/vtable.rs`
- `PlanRef` allocation and typed views: `vortex-layout/src/plan/typed.rs`
- Lazy generic children: `vortex-layout/src/plan/children.rs`
- Execution context: `vortex-layout/src/plan/execution.rs`
- Layout lowering: `vortex-layout/src/plan/lower.rs`
- Operator implementations: `vortex-layout/src/plan/plans/`
- Scan planning: `vortex-scan-v2/src/scan_builder.rs`
- Per-split execution: `vortex-scan-v2/src/tasks.rs`
- Split discovery: `vortex-scan-v2/src/splits.rs`
