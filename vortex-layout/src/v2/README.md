# Layout V2

## Overview

The layout tree describes the structure of stored columnar data. It is the same structure at read and
write time — what differs is whether leaf segments are being read from or written to.

A query is expressed as an `Expression` that is pushed into the layout tree. The tree partitions the
expression over its structure: `StructLayout` splits it across fields, `ChunkedLayout` translates
row coordinates, `ZonedLayout` wraps data with pruning metadata, and `Flat` is a leaf that
reads a physical segment and evaluates the expression on it.

Planning is two-phase:

1. **`prepare`** — called once per query. Walks the layout tree, partitions the expression,
   collects row split boundaries, and returns a `SplitPlanner`.
2. **`plan_split`** — called once per split. The `SplitPlanner` builds a DAG of compute nodes
   (the `SplitPlan`) for a specific row range. The scan scheduler executes this DAG by fetching
   segments and firing compute closures.

The scan calls `prepare` once, then calls `plan_split` for each split in the scan window.
The same `prepare` / `plan_split` API is used for both filter and projection expressions — layout
nodes do not know whether their result is used as a filter mask or a projection column. The scan
is the only component that understands query shape: it calls `prepare` separately for filter and
projection, then wires the filter's output mask as an input dependency in the projection's DAG.

---

## Module Structure

```
v2/
  layout/
    vtable.rs       LayoutVTable trait — the core layout interface
    typed.rs        Layout<V> — concrete typed layout wrapper
    erased.rs       LayoutRef — type-erased layout reference
    children.rs     LayoutChild — lazy deserialization of flatbuffer children
    plugin.rs       LayoutPlugin — registry-based deserialization
    session.rs      LayoutSession — session-scoped layout registry
  layouts/
    flat.rs         Flat — leaf layout backed by a single segment
    chunked.rs      Chunked — sequential concatenation with coordinate translation
    struct_.rs      Struct — column fan-out with expression partitioning
    zoned.rs        Zoned — data wrapped with zone map metadata
  planner.rs        PlanBuilder, SplitPlanner, SplitPlan node types
  scan/
    plan.rs         SplitPlan — the per-split execution DAG
    split.rs        Split formation (boundary coalescing and subdivision)
    output.rs       OutputQueue — ordered emission of split results
    mod.rs          Scan — the scan orchestrator
```

---

## Core Trait: `LayoutVTable`

Every layout implements `LayoutVTable`. This trait defines the structural metadata and the
two-phase planning interface.

```rust
trait LayoutVTable: 'static + Sized + Clone + Send + Sync {
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;
    type Plan: 'static + Send;

    fn id(&self) -> LayoutId;
    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType;
    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship;

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &RowSelection,
        row_splits: &mut BTreeSet<u64>,
        builder: &mut PlanBuilder,
    ) -> VortexResult<SplitPlannerRef>;
}
```

### `child_relationship`

Describes how a child's row space relates to its parent:

- **`RowOffset(n)`** — child rows start at offset `n` in the parent's row space. Used by
  `ChunkedLayout` (each chunk at its cumulative offset) and nullable `StructLayout` (validity at
  offset 0).
- **`FieldName(name)`** — child occupies the same row space as the parent but represents a named
  field. Used by `StructLayout` for data columns.
- **`Auxiliary(range)`** — child is in a separate row space (e.g. zone map indices) but its
  node lifetimes are scoped to the parent's row range. Used by `ZonedLayout` for zone maps.

### `prepare`

Called once per query. Responsibilities:

1. Partition the expression for this subtree (e.g. `StructLayout` splits across fields).
2. Register row split boundaries into `row_splits`.
3. Recursively prepare children via `child.prepare(sub_expr, ...)`.
4. Return a `SplitPlannerRef` — an opaque planner that will build per-split DAGs later.

The `PlanBuilder` passed to `prepare` is used to construct nodes shared across splits (not yet
used, reserved for future caching of scan-lifetime nodes).

---

## Two-Phase Planning

### Phase 1: `prepare`

The scan calls `layout.prepare(expr, selection, &mut row_splits, &mut builder)` on the root layout.
This recursively walks the tree, and each node:

- Adds its natural boundaries to `row_splits` (e.g. chunk boundaries, segment start/end).
- Partitions the expression for its children.
- Returns a `SplitPlannerRef` that captures the partitioned expression state.

After `prepare`, the scan calls `form_splits` to convert the boundary set into a list of
`SplitRange`s (coalescing small intervals, subdividing large ones).

### Phase 2: `plan_split`

For each split, the scan calls `planner.plan_split(row_range, selection, &mut builder)`.
Each planner builds a DAG of compute nodes in the shared `PlanBuilder`:

```rust
trait SplitPlanner: Send + Sync {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId>;
}
```

The returned `NodeId` is the root of this subtree's subgraph. The caller wires it as an input
to higher-level compute nodes.

---

## PlanBuilder

The `PlanBuilder` tracks positional context as layouts recurse into children. All builders in a
single `plan_split` call share the same backing `SplitPlan`, so nodes created by different
subtrees end up in the same DAG.

```rust
// Create a child builder by stepping into a child's coordinate space.
let child_builder = builder.step_into(&child_relationship);

// Create a compute node. Segments are fetched by the scheduler;
// input nodes are outputs of other compute nodes.
builder.create_node(NodeOpts {
    inputs: &[upstream_node_id],
    segments: &[segment_id],
    lifetime: builder.row_range_lifetime(local_row_range),
    compute: |inputs| { /* segments first, then input arrays */ },
})
```

**Coordinate translation**: `step_into(RowOffset(n))` shifts the builder's `base_offset` by `n`,
so `row_range_lifetime(local_range)` returns global coordinates. `step_into(FieldName(_))` leaves
coordinates unchanged. `step_into(Auxiliary(range))` enters a separate row space where node
lifetimes are fixed to the parent's range.

**Single DAG**: all child builders created via `step_into` share the same underlying `SplitPlan`
via `Rc<RefCell<SplitPlan>>`. After planning, `builder.take_plan()` extracts the finalized plan
(panics if child builders are still alive).

---

## SplitPlan

The per-split execution DAG. Each node has:

- **Segment dependencies** (`Vec<SegmentId>`) — fetched by the scheduler, delivered as `ByteBuffer`.
- **Node dependencies** (`Vec<NodeId>`) — outputs of upstream compute nodes, delivered as `ArrayRef`.
- **Compute closure** (`FnOnce(Vec<NodeInput>) -> VortexResult<ArrayRef>`) — runs when all
  dependencies are satisfied. Inputs are ordered `[segments..., nodes...]`.
- **Lifetime** — row range or scan-scoped, used for future eviction decisions.

Node states: `Waiting` (dependencies pending) -> `Ready` (all resolved) -> `Complete` (executed).

The scan scheduler drives execution:

1. `pending_segment_ids()` — collect all segments needed by waiting nodes.
2. Fetch segments concurrently via `SegmentSource`.
3. `complete_segment(id, buffer)` — deliver a buffer, returns newly-ready nodes.
4. `execute_node(node_id)` — run compute, propagate output to dependents, returns newly-ready nodes.
5. `is_complete()` / `take_output()` — check and extract the root node's result.

---

## Layout Implementations

### Flat

The leaf layout. Backed by a single physical segment.

**`prepare`**: registers `0` and `row_count` as split boundaries. Captures the segment ID and
expression.

**`plan_split`**: emits one compute node that depends on the segment. The compute closure
deserializes the segment into an array and evaluates the expression via `array.apply(&expr)`.

### Chunked

Concatenates children sequentially. The only layout that changes coordinate systems.

**Metadata**: `chunk_offsets: Vec<u64>` — cumulative row offsets (`num_children + 1` entries).

**`prepare`**: iterates children, skips non-overlapping chunks, translates selection and row splits
between global and chunk-local coordinates. Registers chunk boundaries as split points.

**`plan_split`**:
- 0 overlapping children: returns an empty array node.
- 1 overlapping child: delegates with translated row range (zero-cost).
- N overlapping children: plans each, then adds a concatenation compute node.

### Struct

Fans out to named field children. Partitions the expression across struct fields.

**Metadata**: `child_dtypes: Vec<DType>` — pre-resolved dtypes for each child. If nullable,
child 0 is the validity layout and data fields start at index 1.

**`prepare`**: partitions the expression over struct fields using `compute_partitioned_expr`:

1. Expands `root()` into `pack(a: $.a, b: $.b, ...)`.
2. Partitions into per-field sub-expressions.
3. Single-field fast path: rewrites `$.field` -> `$` and delegates to one child.
4. Multi-field path: prepares all referenced field children and captures the residual root
   expression.

**`plan_split`**:
- **Single field**: delegates to the field's planner. If nullable, adds a validity merge node.
- **Multi field**: plans all field children in parallel, then adds a compute node that assembles
  a `StructArray` and evaluates the root expression.

### Zoned

Wraps a data child with a zone map child for region-level pruning.

**Metadata**: `zone_len: u64` — rows per zone.

**Children**: child 0 is data (`RowOffset(0)`), child 1 is the zone map
(`Auxiliary(0..row_count)`).

**`prepare`**: prepares the data child with the original expression. Zone map pruning
is not yet implemented — when it is, a pruning predicate will be derived from the expression
and used to prepare the zone map child.

**`plan_split`**: plans the data child. When zone map pruning is enabled, also plans a zone map
read and adds a compute node that evaluates the pruning predicate, expands zone-level bits to a
row-level mask, and intersects with the data output.

---

## Lazy Deserialization

Layout trees serialized as flatbuffers are deserialized lazily via `LayoutChild`:

```rust
enum Inner {
    Owned(LayoutRef),              // Already deserialized (or built in-memory)
    Viewed { fb, loc, ids, ... },  // Backed by flatbuffer — deserialized on first access
}
```

`LayoutChild::resolve(dtype)` returns a `LayoutRef`. On first call for a `Viewed` child, it
deserializes from the flatbuffer, looks up the layout plugin in the session registry, and caches
the result. Subsequent calls return the cached `LayoutRef`.

This means a query touching 2 of 1000 columns only deserializes the 2 relevant subtrees.

---

## Plugin System

Layouts are registered in a `LayoutSession` attached to the `VortexSession`. Each `LayoutVTable`
implementation doubles as a `LayoutPlugin` via a blanket impl, providing `id()` for lookup and
`deserialize()` for flatbuffer reconstruction.

```rust
session.layouts2().register(Struct);
session.layouts2().register(Chunked);
```

During deserialization, `LayoutChild` looks up the plugin by the interned layout ID from the
flatbuffer and delegates to `plugin.deserialize(dtype, metadata, children, source, session)`.

---

## Scan

The `Scan` struct ties everything together:

1. Calls `layout.prepare(expr, selection, ...)` to get a `SplitPlannerRef` and row boundaries.
2. Calls `form_splits(boundaries, ...)` to produce `Vec<SplitRange>`.
3. Maintains a sliding window of active splits (default 8).
4. For each active split: calls `planner.plan_split(row_range, ...)` to build a `SplitPlan`.
5. Fetches pending segments concurrently via `SegmentSource`.
6. Executes ready compute nodes until the plan completes.
7. Pushes completed splits to an `OutputQueue` that emits results in split-ID order.

```rust
let scan = Scan::try_new(&layout, &expr, &RowSelection::All, ScanConfig::default())?;
let stream = scan.into_stream();
```

### Split Formation

`form_splits` converts a `BTreeSet<u64>` of boundary points into `Vec<SplitRange>`:

1. Converts consecutive boundary pairs into intervals.
2. Greedily coalesces adjacent small intervals up to `min_split_rows`.
3. Subdivides intervals exceeding `max_split_rows`.
4. Assigns monotonic `SplitId`s.

### Ordered Output

The `OutputQueue` is a priority queue keyed by `SplitId`. Splits complete out of order but are
emitted strictly in ID order. A completed split is held until all preceding splits have been
emitted.
