# LayoutPlan: a pushdown-first execution model for Vortex layouts

**Status:** Proposal / WIP. Scaffolding lives under `vortex-layout/src/v2/`
(`plan.rs`, `chunked.rs`, `demand.rs`, `matcher.rs`, `pushdown.rs`,
`scan.rs`) and is wired in via `Layout::plan(PlanArguments)` in
`vortex-layout/src/layout.rs:71`.

**Scope.** Single-file scanning. No sorting, no shuffle, no
distributed execution. Plan nodes execute inside one process, against
one Vortex file. Anything that requires moving rows between processes
or globally reordering them lives above this layer in the engine that
embeds Vortex.

## Goals

1. **Replace the three-phase `LayoutReader` API**
   (`pruning_evaluation` / `filter_evaluation` / `projection_evaluation`,
   `vortex-layout/src/reader.rs:35`) with a single recursive plan tree
   whose nodes own their own execution.
2. **Remove opaque I/O scheduling.** Today segment fetches happen
   implicitly inside the reader trait's evaluation methods, behind
   `MaskFuture` await timing. Plan nodes must orchestrate their own
   I/O explicitly so we can reason about, schedule, and override it.
3. **Sub-segment reads for wide arrays.** A `FlatPlan` evaluating a
   selective filter should be able to read only the byte ranges its
   surviving rows need, not the whole segment. The trait shape must
   not preclude this.
4. **Partial-aggregate push-down.** `PartialAggregatePlan` is a regular
   `LayoutPlan` node, placed in the tree by engine-specific optimizer
   rules. Pushdown rules within the tree mostly serve to relax
   ordering on its children (parallelism), with `count(*)`/`min`/`max`
   short-circuits from partition stats as a special case.
5. **Two SIP mechanisms, clearly separated.**
   - **Dynamic-expression SIPs** (value-indexed) for everything that
     can be expressed as a predicate. They ride on the expression
     passed to `Layout::plan` as monotone-dynamic leaves and propagate
     by ordinary expression rewriting.
   - **`RowDemand`** (row-indexed, intra-scan, *backwards-flowing*) is
     an optional demand hint: producers learn which rows the consumer
     no longer needs, so they can stop spending effort on them. It
     does not perform filtering — `FilterPlan` does.
6. **Match the DataFusion `ExecutionPlan` shape** closely enough that
   downstream engines (incl. the Spiral physical engine) can drop it
   into a Scan operator with minimal glue.

## Constraints

- **List layouts have two row domains.** Parent rows and elements are
  related by `nest` (offsets array records the boundaries). A
  `LayoutPlan` lives in *one* domain and partitions that domain; it
  never partitions across the nest boundary. See **Row domains and
  partitioning** below.
- **Row counts are not always known statically.** A `LayoutPlan` node
  may not know how many rows its partition will produce until after
  execution. The trait must not require `row_count` at every node.
- **Partitions need a way to report stats.** Reported as an opt-in
  `partition_stats` method that returns whatever the plan can vouch
  for.

## Non-goals

- Removing `LayoutReader` immediately. Two APIs coexist behind a flag
  until parity is reached.
- A cost model or full rule-based optimizer. `PlanPushdownRule` is the
  hook; the first cut applies a fixed sequence of rules.
- A cross-operator SIP bus separate from expressions. All cross-
  operator SIPs ride on dynamic-expression leaves so pushdown through
  layouts uses the same expression-rewriting machinery as static-
  filter pushdown.

## Background

`LayoutReader` (`vortex-layout/src/reader.rs:35`) drives scans via
three implicit-protocol methods called per-split by the task executor
in `vortex-layout/src/scan/tasks.rs`:

- `pruning_evaluation(row_range, expr, mask) -> MaskFuture`
- `filter_evaluation(row_range, expr, mask) -> MaskFuture`
- `projection_evaluation(row_range, expr, mask) -> ArrayFuture`

The split executor calls these per-conjunct, ANDs the `MaskFuture`s,
and pulls projection output once the mask is finalized. Phase
ordering is fixed; intermediate masks cannot inform unrelated work;
I/O is implicit.

`Layout::plan(PlanArguments)` (default-`vortex_bail!`) is the seam we
hang the replacement off.

## Model

```rust
Layout::plan(args: PlanArguments) -> LayoutPlanRef

pub struct PlanArguments {
    pub selection: Selection,
    pub expr: Expression,
    pub ctx: PlanContext,        // demand handle, session-level hints
}

pub trait LayoutPlan : 'static + Send + Sync {
    fn schema(&self) -> &DType;

    fn partition_count(&self) -> usize;
    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats>;

    fn output_ordered(&self) -> bool;
    fn required_input_ordered(&self) -> Vec<bool>;
    fn maintains_input_order(&self) -> Vec<bool>;

    fn repartition(self: Arc<Self>, n: usize) -> VortexResult<LayoutPlanRef>;
    fn children(&self) -> &[LayoutPlanRef];
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef>;

    fn execute(
        &self,
        row_range: Range<u64>,
        session: &VortexSession,
    ) -> VortexResult<SendableArrayStream>;
}

pub struct PartitionStats {
    pub row_range: Range<u64>,                 // physical row range this partition covers
    pub byte_size_estimate: Option<u64>,
    pub column_stats: HashMap<FieldPath, ColumnStats>,
}
```

Deliberate departures from DataFusion's `ExecutionPlan`:

**No trait-level `row_count`.** Forces honesty about list and dynamic-
filter-bearing plans. Row count moves to
`partition_stats().row_range.len()`.

**Stream output only.** `execute` returns a `SendableArrayStream` — no
separate mask channel, no out-of-band coordination. Selection state
lives inside the plan tree (in `FilterPlan` and `FilterArray`), not
on the dataflow edge.

**Binary ordering.** A plan is either row-ordered or not — see
**Ordering**.

**Partitions are labeled row ranges.** `partition_count` /
`partition_stats(i)` describe the natural splits a plan exposes —
each split is a `Range<u64>` in the plan's row coordinate space.
Engines pick which row range to ask for and call `execute(range, …)`;
they're free to ignore `partition_stats` and request any range, but
asking for partition-aligned ranges avoids slicing overhead at the
leaves.

**`execute` takes a `Range<u64>`, not a partition index.** Partitions
were originally an integer abstraction, but heterogeneous chunking
across struct fields (the writer coalesces per-field by bytes) means
"partition 5 of field A" and "partition 5 of field B" don't cover
the same rows. Row ranges are the only universal coordinate. Each
plan node slices its own data on read — `ChunkedPlan.execute(range)`
finds intersecting chunks; `StructPlan.execute(range)` delegates the
range to every child; `FlatPlan.execute(range)` slices the segment.

**Plans are pure descriptions.** No execution caches on the plan —
holding a plan should never hold I/O state. Cross-execute sharing
(dict values reused across N codes-partition reads, segment reused
across sub-range reads of a flat) is **not** carried by the plan; it
arrives later via explicit sharing nodes (see § Tee / Let-Use). Until
those land, those reads are honestly redundant.

## Row domains and partitioning

Partitions are **domain-relative**. There is no global "partition 5";
it is always "partition 5 of plan P, where P emits output in some
specific row domain."

### Nest, not join

A list-typed column has two row domains:

- **Parent rows.** N entries. Each is one list.
- **Elements.** M entries. The flat concatenation of all lists.

The relationship is `nest`: each parent row *contains* a variable-
length collection of elements, with an offsets array recording the
boundaries. `unnest` flattens (one row per (parent, element) pair);
`array_agg` / `collect_list` re-nests. The list layout *is* the
nested form on disk:

```
ListLayout
├── offsets    (Flat, i32 / i64)
└── elements   (some sub-layout, lives in the element row domain)
```

The two children live in different row domains.

### One plan, one domain

A `LayoutPlan` produces output in *one* row domain. Its partitions
partition that domain.

- **A list-scan plan** (schema `List<T>`) has partitions over parent
  rows. Element data is nested inside each batch as nested arrays.
- **An element-scan plan** (schema `T`) — obtained by calling
  `Layout::plan` over the `elements` child directly, or by unnesting
  upstream — has partitions over the element domain.

Element partition K and parent partition K are unrelated.

### Crossing domains is operator work

`Unnest`, `array_agg`, `gather` — all live in the engine's operator
layer, above `LayoutPlan`. The offsets array is the runtime index
that joins the two domains; it is never consulted at plan-construction
time, because each plan stays inside one domain.

```
ListLayout                    (parent-row domain, N partitions)
  → Unnest operator           (element domain, N partitions, variable size)
  → PartialAgg                (element domain, N partitions)
```

The `Unnest` and `PartialAgg` operators live above `LayoutPlan` —
they're shown here only to illustrate that domain crossing is
operator work, not layout work.

## Ordering

A `LayoutPlan` is **row-ordered** or not. Row-ordered means within
each partition rows come out in row-id order, and partitions come out
in partition-id order. There is no sort-by-arbitrary-key ordering at
this layer; widen later if a real use case forces it.

Three trait methods, patterned on DataFusion's `ExecutionPlan` but
binary:

```rust
fn output_ordered(&self) -> bool;
fn required_input_ordered(&self) -> Vec<bool>;
fn maintains_input_order(&self) -> Vec<bool>;
```

Ordering is resolved at **plan-construction time**. After pushdown
rules run, an `EnforceOrdering` pass walks the tree and compares
required-vs-produced. Mismatches are resolved by **wrapping** (insert
a reorder-buffering merge node) or **rejecting** the rewrite that
produced the mismatch. Because we have no sort-by-key, wrapping can
only re-establish row order via chunk-gated buffering — if even that
doesn't work, the rewrite is rejected.

### Per-node ordering contracts

| Node                     | `output_ordered` | `required_input_ordered` | `maintains_input_order` |
|--------------------------|------------------|--------------------------|-------------------------|
| `FlatPlan`               | `true`           | `[]`                     | `[]`                    |
| `StructPlan`             | propagates       | `[true; n]`              | `[true; n]`             |
| `ChunkedPlan` (ordered)  | `true`           | `[true; n]`              | `[true; n]`             |
| `ChunkedPlan` (relaxed)  | `false`          | `[true; n]`              | `[false; n]`            |
| `RowIdxPlan`             | propagates       | `[true]`                 | `[true]`                |
| `CompressedPlan`         | propagates       | `[true]`                 | `[true]`                |
| `FilterPlan`             | propagates       | `[true, true]`           | `[true, false]`         |
| `PartialAggregatePlan`   | `false`          | `[false]`                | `[false]`               |

### Ordered vs. relaxed `ChunkedPlan`

`ChunkedPlan` is constructed ordered by default; a node-specific
method flips it (DataFusion idiom — cf. `RepartitionExec::with_preserve_order`):

```rust
impl ChunkedPlan {
    fn with_preserve_order(self: Arc<Self>, preserve: bool)
        -> Arc<dyn LayoutPlan>;
}
```

At runtime, **relaxed** spawns all children concurrently and emits
from whichever yields first (no buffering). **Ordered** runs either
sequentially or with a chunk-id-gated reorder buffer.

The flag is flipped by:

1. The pushdown rule that introduces a `PartialAggregatePlan` over a
   Chunked, after verifying no ancestor needs ordering.
2. The engine adapter, when constructing plans for an order-
   indifferent consumer.

Two ways rules adjust ordering generally:

- **In-place mode flip** via node-specific methods like
  `with_preserve_order`. Same shape, different mode.
- **Wrapping** with a new node (e.g., a reorder-buffering merge).
  Used when the change is structural.

## SIPs

Two mechanisms. The split is clean.

### Dynamic-expression SIPs (the common case, value-indexed)

Every cross-operator SIP — Bloom filters from a join build,
value-set narrowing, range refinement, dynamic filters — is expressed
as a **monotone-dynamic leaf in the filter expression**.

The engine constructs the expression with the leaves already in
place. The scan builder passes that expression to `Layout::plan`. The
plan tree carries it through layouts via ordinary expression
rewriting (`StructPlan` routes field accesses, `ChunkedPlan`
replicates, etc.). Every node holding the expression sees the same
shared dynamic-value cell.

At execute time, each plan polls the leaf's current snapshot. When
the leaf publishes a tighter value (lattice meet, monotonic),
in-flight evaluations finish on the old snapshot; the next batch
picks up the new one.

The monotone-value protocol (lattice meet, `refines`, kind URI) lives
in `vortex-array/src/expr/` alongside the existing expression
infrastructure. New SIP kinds are registered there. Layouts that know
how to exploit specific kinds (Zoned over Blooms, Dict over value-
sets) get acceleration; others evaluate the expression normally and
still get correctness.

### `RowDemand` (intra-scan, row-indexed, backwards-flowing)

`RowDemand` is a SIP from `FilterPlan` (or any node that knows it will
discard certain rows) back to source plans. It carries: "rows X..Y in
my partition's row space have zero demand downstream — don't bother
producing real values for them."

**It does not filter.** Filtering is `FilterPlan`'s job (see below).
Dropping `RowDemand` entirely is always safe; sources just spend more
work than necessary.

#### Coordinate space

Coordinates are **pre-filter**, in the **scan's input row space**, and
**physically aligned to the data the leaves hold**. Any post-filter
(rank-compressed) scheme would force every producer/consumer to know
the rank function — i.e., to await upstream filter masks before they
could even talk about positions — which kills the opportunistic /
drop-safe / monotonic semantics that make `RowDemand` cheap.

The scan's input row space is whatever `Scan::build` was asked to
cover (the full file, or the partition subset the engine selected for
this open — see "Partial scans" below). It is **not** the file's full
row space when the engine has already restricted what to read.

#### Window setup is one-shot, traversal slices the windows

`RowDemand` windows are allocated once in `Scan::build`, sized by the
dominant Chunked layout's chunks. Producer/consumer handles are then
plumbed through the plan tree at construction time:

- `Chunked` slices its window range and hands each child plan only the
  sub-window covering that chunk's rows.
- `Struct` / `Zoned` / `Dict` pass-through unchanged (their children
  share the parent's row space).
- A `Flat` leaf gets a window range that is exactly aligned to its own
  data, so `consumer.snapshot(range)` returns a mask the leaf can
  intersect with its on-disk row positions directly — no rank ops.

This "slice as we traverse" model keeps coordinates aligned all the
way down. By execute time, every producer/consumer handle is scoped
to the leaf's local rows.

```
pub struct RowDemand {
    spans: Vec<MonotoneMaskCell>,   // one cell per window of the partition
}

pub struct RowDemandProducer { ... }
pub struct RowDemandConsumer { ... }
```

Producer (one method, monotone publish):

- `producer.publish(window, Mask)` meets the new mask into the cell.
  Monotone: bits only go 1→0 (rows newly known not to be needed).

Consumer (queries are over arbitrary `RowRange`s; the implementation
gathers the overlapping windows):

- `consumer.is_empty(range) -> bool` — true iff *no* row in `range` has
  any remaining demand. Cheapest check; sources should call this first
  before doing any per-row work. Implementable as an OR of "any-set" bits
  cached on each window cell.
- `consumer.cardinality(range) -> u64` — popcount of demand over `range`.
  Lets sources branch between dense and sparse evaluation paths (e.g., a
  flat reader picking between full-segment decode + filter vs. gather of
  surviving offsets).
- `consumer.snapshot(range) -> (version, Arc<Mask>)` — the actual demand
  mask sliced to `range`. Cheap when the underlying windows are
  unchanged since the last snapshot (Arc-clone of cached); otherwise
  materializes by stitching window cells.
- `consumer.wait_for_newer(range, version)` — async watch-channel; wakes
  when any window overlapping `range` has a tighter cell than `version`.

The three reads (`is_empty`, `cardinality`, `snapshot`) form a staircase
of cost: cheap → medium → full. Sources should call them in that order
and bail out as early as the answer allows.

Window granularity is the scan builder's call (typically aligned with
the dominant Chunked layout's chunk size). Cells are dropped once
their windows have been emitted by the projection — the producer is
the natural frontier driver.

The cross-conjunct cooperation story flows entirely through
`RowDemand`: conjunct 1 finishes a window, `FilterPlan` (or a
downstream stream-AND) updates `RowDemand` based on the mask so far,
conjunct 2's reader sees zero demand on already-rejected rows and
short-circuits. Same as any other SIP — opportunistic, drop-safe.

## Partial scans

Engines never want a "scan rows X..Y" parameter on `LayoutPlan` —
that's a row coordinate concern that doesn't belong in the trait.
Instead, the two engine boundaries each express partial work using a
primitive that already exists in `LayoutPlan`:

**`ExecutionPlan::execute(partition_idx)` boundary** (e.g., a
DataFusion `VortexExec` whose output partitioning is N). The engine
asks for N partitions and calls `execute(0..N)`. The plan satisfies
this via `LayoutPlan::repartition(n)` — coalesce or split partitions
until `partition_count() == n`. `ChunkedPlan` is the layer that
implements this in practice (chunks are the natural units to merge or
split).

**`FileOpener::open(file)` boundary**. DataFusion has already chosen
its partitioning at the file-range level — each `(file, byte_range)`
pair is one DF partition, and each `open()` call returns a single
stream covering that range. The opener does **not** ask the plan to
repartition; instead it does **partition selection**:

1. Translate `byte_range` to a row range using the file's natural
   splits (already done in v1 via `split_aligned_row_range`).
2. Build the plan once over the whole file.
3. Read `partition_stats(i).row_count` to learn each partition's row
   span.
4. Call `execute()` only on the partitions that fall within the
   requested row range, flatten the resulting streams.

`Scan::build` is therefore agnostic to the engine's partial-scan
model: it always builds a plan over the full file. Slicing happens at
the engine boundary using `partition_count` / `partition_stats` /
`execute` — no row range is plumbed through `PlanArguments`.

`RowDemand` is allocated against the **selected** partitions: if the
opener chose partitions [5..8], the windows cover those three chunks'
rows (in pre-filter, scan-input space — see above), nothing else.
That naturally bounds the per-scan `RowDemand` allocation.

A natural-but-deferred extension: instead of structurally pruning
out-of-scope chunks, mark their windows as `demand=∅` in `RowDemand`
at scan-build time. The plan stays canonical (full file); partial
scans become an opening-state on the demand cells; producers
short-circuit by checking demand. Composes uniformly with filter
pushdown but only pays off once producers actually check demand —
i.e., once `FilterPlan` lands.

## Scan construction

```rust
impl Scan {
    pub fn build(&self) -> VortexResult<LayoutPlanRef> {
        // 1. Decompose the filter.
        let conjuncts: Vec<Expression> = match &self.filter {
            Some(f) => split_top_level_and(f),
            None    => vec![],
        };

        // 2. Allocate the RowDemand. Optional / SIP — but allocate
        //    one up-front and thread it through the PlanContext so
        //    publishers and consumers can wire up at plan time.
        let demand = Arc::new(RowDemand::new(
            self.layout.row_count(),
            WindowSpec::default(),
        ));
        let ctx = PlanContext { demand: Arc::clone(&demand) };

        // 3. Each conjunct → a bool-stream LayoutPlan.
        let conjunct_plans: Vec<LayoutPlanRef> = conjuncts.into_iter()
            .map(|expr| self.layout.plan(PlanArguments {
                selection: self.selection.clone(),
                expr,
                ctx: ctx.clone(),
            }))
            .collect::<VortexResult<_>>()?;

        // 4. AND the conjunct bool streams in lockstep.
        let combined_mask: LayoutPlanRef = match conjunct_plans.len() {
            0 => ConstantBoolPlan::all_true(self.layout.row_count()),
            1 => conjunct_plans.into_iter().next().unwrap(),
            _ => Arc::new(AndBoolStreamsPlan::new(conjunct_plans)),
        };

        // 5. Projection plan.
        let projection = self.layout.plan(PlanArguments {
            selection: self.selection.clone(),
            expr: self.projection.clone(),
            ctx: ctx.clone(),
        })?;

        // 6. Filter wraps projection with mask. Publishes RowDemand.
        let root: LayoutPlanRef = Arc::new(FilterPlan::new(
            projection,
            combined_mask,
            demand.producer(),
        ));

        // 7. Pushdown.
        apply_pushdown_rules(root)
    }
}
```

A few things to notice:

**No "three trees" anymore.** Conjuncts are bool-stream `LayoutPlan`s,
combined by a stream `AndBoolStreamsPlan` zip. The result is a single
mask stream. `FilterPlan` applies it to projection. One tree, top to
bottom.

**`Layout::plan` does the same thing for every expression.** Whether
the expression is a conjunct (returns Bool) or a projection (returns
the projection's type) is just a type difference — the recursion is
the same. Layouts that have stats emit opportunistic `RowDemand`
publishers regardless of which.

**Pruning is emitted by layouts, not built externally.** When
`ZonedLayout::plan` is called with a conjunct expression, it (a)
delegates to its inner layout for the actual evaluation, and (b)
opportunistically constructs a side-effect plan that reads its zone-
map child and publishes to `RowDemand` whenever stats can answer the
conjunct's `checked_pruning_expr`. The scan builder doesn't know or
care which layouts have stats.

**Dict pushdown unlocks codes-zone pruning automatically.** When
`DictLayout::plan` rewrites `col = "Alice"` to `codes IN {17}` and
hands that to `codes_layout.plan(...)`, any Zoned layer inside codes
sees the rewritten expression and emits its own opportunistic
pruning. No special wiring.

## Per-layout `plan` walkthrough

### `StructLayout::plan`

```rust
fn plan(&self, args: PlanArguments) -> VortexResult<LayoutPlanRef> {
    let fields = referenced_field_paths(&args.expr);
    if fields.len() == 1 {
        let f = fields.into_iter().next().unwrap();
        let child_expr = rewrite_field_access(&args.expr, &f);
        return self.field_layout(&f)?.plan(args.with_expr(child_expr));
    }
    let child_plans = fields.iter().map(|f| {
        let child_expr = rewrite_field_access(&args.expr, f);
        self.field_layout(f)?.plan(args.with_expr(child_expr))
    }).collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(StructPlan {
        children: child_plans,
        output_expr: args.expr,
    }))
}
```

Pass-through with field routing. No data of its own.

### `ZonedLayout::plan`

```rust
fn plan(&self, args: PlanArguments) -> VortexResult<LayoutPlanRef> {
    let inner_plan = self.inner.plan(args.clone())?;

    // Opportunistic pruning publisher (SIP-only).
    if let Some(pruning_expr) =
        checked_pruning_expr(&args.expr, &self.dtype())?
    {
        if self.zone_map.satisfies_stats(&pruning_expr) {
            let pruning_plan = ZonedPruningPlan {
                zone_map_plan: self.zone_map.plan(PlanArguments {
                    selection: args.selection.clone(),
                    expr: pruning_expr,
                    ctx: PlanContext::without_demand(),
                })?,
                zone_size: self.zone_size,
                publisher: args.ctx.demand.producer(),
            };
            args.ctx.demand.register_opportunistic(pruning_plan);
        }
    }

    Ok(inner_plan)
}
```

`inner_plan` is what flows back as data. The pruning subtree is a
side-effect publisher to `RowDemand`. `checked_pruning_expr` is
applied to whatever expression Zoned was handed — which may already
be a rewritten form from an enclosing layout's pushdown (this is the
key invariant that makes dict + codes-zones work).

### `DictLayout::plan`

```rust
fn plan(&self, args: PlanArguments) -> VortexResult<LayoutPlanRef> {
    match classify(&args.expr) {
        Passthrough => self.codes_layout.plan(args),

        ValuePredicate(pred) => {
            // Answer the predicate against the dictionary.
            let matching_codes: CodeSet =
                eval_predicate_on_dict(&pred, &self.values_layout)?;

            // Rewrite into the codes domain.
            //   col = "Alice"   →   codes IN {17}
            //   col > "C"       →   codes IN {3, 7, 12, ...}
            let rewritten = Expression::in_set(
                root_codes_expr(),
                matching_codes,
            );

            self.codes_layout.plan(args.with_expr(rewritten))
        }

        ValueProjection => Ok(Arc::new(DictDecodePlan {
            codes:  self.codes_layout.plan(args.with_expr(root_codes_expr()))?,
            values: self.values_layout.clone(),
        })),
    }
}
```

### `DictLayout` with a zone-map on the codes

Layout:

```
DictLayout
├── codes:  Zoned(codes_data, code_zone_map)
└── values: ...
```

When `DictLayout::plan` is called with `col = "Alice"`, it rewrites
to `codes IN {17}` and hands that to `codes_layout.plan(...)`. The
codes layout is a `ZonedLayout`, which:

1. Plans the inner `codes_data` with `codes IN {17}` — that's the
   actual scan.
2. Derives `checked_pruning_expr(codes IN {17}, codes_dtype)` —
   roughly `min(codes) <= 17 AND max(codes) >= 17`.
3. Sees that its code zone-map can answer min/max over codes.
4. Constructs an opportunistic `ZonedPruningPlan` over the code zone-
   map, registered with `RowDemand`.

Two layers of pushdown stacked — dict → codes rewrite, then codes-
Zoned → row-range pruning — with no special wiring at scan-build time.
Every rewrite is visible to whatever stats-bearing layout sits below
it, because the rewrite happens *before* the inner `plan()` runs.

## `FilterPlan` and its pushdown

`FilterPlan(value_plan, mask_plan, demand_producer)` is the only node
that actually performs filtering. At execute time it consumes value
and mask streams in lockstep and emits `FilterArray`-wrapped batches.
It also writes the current mask to `RowDemand` so sources can stop
producing rejected rows.

Pushdown rules fire bottom-up after construction:

**`PushFilterThroughStruct`** — rewrites

```
FilterPlan(StructPlan(child_a, child_b, ...), mask)
   →
StructPlan(
    FilterPlan(child_a, mask_tee_a),
    FilterPlan(child_b, mask_tee_b),
    ...
)
```

The mask is now consumed by N branches (see Tee/CSE below).

**`PushFilterThroughChunked`** — distributes a FilterPlan across each
chunk, so each chunk filters independently.

**`FuseFilterIntoFlat`** — terminal. A flat plan that consumes a mask
can emit only surviving rows directly, and (future) issue sub-segment
reads guided by mask density.

This is where late materialization actually pays off: at the leaves,
not at the root.

## Tee and CommonSubplanElimination

Three fan-out patterns appear, two of which a CSE pass naturally
addresses:

1. **Intra-partition mask fan-out.** `PushFilterThroughStruct` gives
   the mask N consumers (one per struct child). All N consumers run
   inside the same `execute(range, …)` call.
2. **Inter-subtree source sharing.** Filter conjunct `a > 5` and
   projection both read column `a`'s flat layout. Two FlatPlans over
   the same source do redundant I/O.
3. **Cross-partition broadcast.** A `DictDecodePlan` reads its values
   once and uses them across every codes-partition `execute(range)`
   call. The values aren't a separately-occurring subtree, they're a
   one-shot read fanned out to N execute calls.

Patterns 1 and 2 are CSE-friendly — structural plan-equality
identifies the candidates and the rewrite is local. Pattern 3 is
fundamentally different: it spans multiple `execute` calls (one per
partition), and execute calls are independent, so a TeePlan owning a
shared stream-source doesn't fit cleanly. Pattern 3 wants a one-shot
materialised result `Arc`-shared across calls, not a streaming source
with cursors.

The current direction is to fuse all three into a single
`Let p = source` / `Use(p)` primitive (working name `LetPlan` /
`UsePlan`). The same `Let` source can be either a streaming
multi-cursor handle (Tee semantics, for patterns 1 & 2) or a one-shot
broadcast handle (Arc semantics, for pattern 3) — discriminated by
how it's used downstream. A `CommonSubplanElimination` pass
identifies candidates by structural plan-equality (same layout, same
expression, same selection) and rewrites duplicates into the shared
`Let`. Run as a post-pass after all pushdown reaches fixed point.

Until that lands, plans deliberately do not cache execution state
(see § Model: "Plans are pure descriptions"). Redundant reads are
honest, not hidden — they go away when CSE / Let-Use does.

## Partial-aggregate push-down

`PartialAggregatePlan` is a regular `LayoutPlan` node placed by an
engine-specific physical rule (e.g., a DataFusion rule that detects
`AggregateExec(mode=Partial)` over a Vortex scan and rewrites). The
layout itself never originates aggregates — there is no
`Layout::plan_aggregate`.

Pushdown rules within the layout tree mostly relax ordering for
parallelism rather than finishing the aggregate inside the layout:

1. **`PushPartialAggThroughChunked`** — replace `PartialAgg(Chunked)`
   with `Chunked(PartialAgg-per-child)` *and* `with_preserve_order(false)`
   on the Chunked. This is where parallelism is unlocked.
2. **`AnswerFromPartitionStats`** — when the aggregate is answerable
   from `partition_stats` (`count(*)` from `row_count`, `min/max`
   from column stats), replace with a stats-emitting plan.

Group-by versions of the same rules; `AnswerFromPartitionStats`
rarely fires for group-by because group-able stats are uncommon.

## Worked example

Schema: `Struct{event_date: i64, counter_id: i64, url: utf8}`.
Layout: `Chunked(N, Zoned(Struct{Flat, Flat, Flat}))`.
Filter: `event_date >= X AND counter_id = Y`.
Projection: `url`.

After `Scan::build`, before pushdown:

```
FilterPlan
├── value: ChunkedPlan
│             └── StructPlan
│                     └── FlatPlan (url)
└── mask : AndBoolStreamsPlan
              ├── ChunkedPlan
              │     └── StructPlan
              │             └── FlatPlan (event_date)  -- conjunct_1
              └── ChunkedPlan
                    └── StructPlan
                            └── FlatPlan (counter_id) -- conjunct_2

(Side: each ZonedLayout encountered registered an opportunistic
 pruning plan with RowDemand.)
```

After `PushFilterThroughStruct` + `PushFilterThroughChunked` +
`FuseFilterIntoFlat`:

```
ChunkedPlan
  └── StructPlan(per-chunk)
        └── FlatPlan(url, mask)   -- reads only surviving rows
              ↑
              mask: AndBoolStreamsPlan zipped per chunk
```

After `CommonSubplanElimination`:

```
TeePlan(mask)  ─┐
                ├── FlatPlan(url, mask_cursor_1)
                └── (additional cursors if other children need it)
```

## Per-layout `LayoutPlan` stubs

Each layout under `vortex-layout/src/layouts/` gets a sibling plan
node under `vortex-layout/src/v2/`.

| Layout       | Plan node        | Notes |
|--------------|------------------|-------|
| `Flat`       | `FlatPlan`       | Terminal. Reads segment (or sub-segment range) and evaluates the input expression. Fuses with `FilterPlan`. |
| `Chunked`    | `ChunkedPlan`    | `partition_count()` = chunk count; `with_preserve_order(bool)` flips ordered/relaxed. |
| `Struct`     | `StructPlan`     | Pure rewriting. No own data. |
| `Zoned`      | `ZonedPlan`      | Pass-through for filter/projection, plus an opportunistic `ZonedPruningPlan` publisher to `RowDemand`. |
| `Partitioned`| `PartitionedPlan`| Partition-key splitting. |
| `RowIdx`     | `RowIdxPlan`     | Synthesizes row-id column. |
| `Compressed` | `CompressedPlan` | Forwarder; decompresses on read. |
| `Dict`       | `DictPlan`       | Predicate rewrite into codes domain; `DictDecodePlan` for projection. |

Cross-cutting plan nodes (not 1:1 with a layout):

| Node                    | Purpose |
|-------------------------|---------|
| `FilterPlan`            | Applies a mask stream to a value stream. |
| `AndBoolStreamsPlan`    | Zips N bool-stream plans, AND per element. |
| `TeePlan`               | One source, N cursored consumers; bounded buffer. |
| `PartialAggregatePlan`  | Partial aggregate. Engine-placed. |

## `PlanPushdownRule`

```rust
pub trait PlanPushdownRule {
    type Parent: Matcher;
    fn rewrite(&self, parent: LayoutPlanRef) -> VortexResult<RewriteResult>;
}

pub enum RewriteResult {
    Unchanged,
    Rewritten(LayoutPlanRef),
}
```

The `Matcher` trait declares the parent shape a rule applies to.
Rules are stateless; they cannot inspect or mutate `RowDemand` (that's
runtime state) — they can only rewrite plan structure.

Initial rule sequence:

1. `PushFilterThroughStruct`
2. `PushFilterThroughChunked`
3. `FuseFilterIntoFlat`
4. `PushPartialAggThroughChunked` (flips ChunkedPlan via `with_preserve_order(false)`)
5. `AnswerFromPartitionStats`
6. `CommonSubplanElimination` (inserts `TeePlan`s)
7. `EnforceOrdering` (wraps reorder buffers or rejects, per the Ordering section)

Run to fixed point per rule, then proceed to the next.

## Migration plan

PRs land in order on the `ngates/layoutv2-N` stack. The first four
have already shipped:

- **PR1 — Trait skeleton + design doc.** `LayoutPlan` with the full
  method surface, `PartitionStats`, `PlanArguments`, `LAYOUT_PLAN.md`.
- **PR2 — Plan-node scaffolds + V1/V2 toggle.** `FlatPlan`,
  `ChunkedPlan`, `StructPlan`, the `RowDemand` API stub (no
  implementation), `VORTEX_LAYOUT_PLAN_V2` env var.
- **PR3 — Projection-only `Scan` + DF opener wiring.** `Layout::plan`
  for `Flat`/`Chunked`/`Struct`, `Scan::build` for projection-only,
  V2 path wired into the persistent opener, differential test.
- **PR4 — Row-range execute + Zoned/Dict + StructLayout field
  routing.** `LayoutPlan::execute` switched from integer partition to
  `Range<u64>`; `Layout::plan` for `Zoned` (pass-through) and `Dict`
  (`DictDecodePlan`); `StructLayout::plan` does field routing via
  `partition()`; `AlignedArrayStream` k-way zip in `StructPlan`;
  opener uses linear byte→row interp instead of zone alignment.

Remaining (numbering is loose — some of these may split or fuse as
they land):

- **PR5 — `FilterPlan` + `AndBoolStreamsPlan` + minimal `RowDemand`.**
  End-to-end filtered scan. RowDemand windows allocated against the
  scan's input row range (per the "post-selected, pre-filter" model
  in § SIPs / RowDemand). `FilterPlan` publishes per-conjunct masks;
  `AndBoolStreamsPlan` zips the bool streams.
- **PR6 — Pushdown rules.** `PushFilterThroughStruct`,
  `PushFilterThroughChunked`, `FuseFilterIntoFlat`. Run-to-fixed-point
  driver. `EnforceOrdering` post-pass.
- **PR7 — `Let` / `Use` (sharing primitive) + `CommonSubplanElimination`
  pass.** Single primitive replaces the conceptual `TeePlan`: a `Let p
  = source` node materialises a source once and exposes N consumers
  through `Use(p)` references. Discriminates between streaming
  multi-cursor (intra-partition mask fan-out, inter-subtree source
  sharing) and one-shot broadcast (cross-partition dict values, flat
  segments) at construction time. CSE pass identifies candidates by
  structural plan-equality and rewrites to `Let`/`Use`. **Once this
  lands, the redundant per-execute reads from PR4 collapse**
  (DictDecodePlan re-reading values per call, FlatPlan re-fetching
  segments) — that's the main perf-recovery PR.
- **PR8 — Zoned pruning.** `ZonedPruningPlan` that publishes to
  `RowDemand` from zone-map stats. Lights up the
  pruning-via-stats path that `ZonedLayout::plan` currently no-ops.
  Depends on PR5 (RowDemand) and PR7 (Let/Use for zone-map sharing
  across conjuncts).
- **PR9 — `DictPlan` predicate rewrite.** `col = "Alice"` →
  `codes IN {17}` rewriting on the way down; combined with PR8 makes
  codes-zoned pruning work end-to-end.
- **PR10 — `PartialAggregatePlan`** + DataFusion physical rule that
  places it over Vortex scans. `PushPartialAggThroughChunked` +
  `AnswerFromPartitionStats` rules.
- **PR11 — Cut over `ScanBuilder::into_array_stream` to V2 by
  default.** Flip the toggle's default once V2 reaches parity. Delete
  `LayoutReader`.

## Open questions

**Q1. Window granularity for `RowDemand`.** Default to the dominant
Chunked layout's chunk size; fall back to a fixed row count when no
Chunked is present. Allow layouts to advertise a preferred size.

**Q2. Where do dynamic-leaf trait + lattice protocol live?** Probably
`vortex-array/src/expr/` alongside the existing expression
infrastructure. Treats dynamic values as just another leaf kind.

**Q3. Re-evaluation policy on leaf updates.** When a dynamic-leaf
value tightens mid-execution, finish the current batch on the old
snapshot and pick up the new one at the next batch boundary. Same
pattern as `RowDemand` window boundaries.

**Q4. `AnswerFromPartitionStats` × filter pushdown.** Aggregates over
filter-bearing scans need per-partition classification (fully
retained / fully eliminated / partial) before short-circuiting. Run
the rule after filter pushdown has done that classification.

**Q5. Sub-segment reads in `FlatPlan`.** Out of scope for the initial
port; the trait shape doesn't preclude it. Hooks into
`FuseFilterIntoFlat`.

**Q6. List layouts and partition semantics.** With partitions no
longer required to be row-ranges, what does `partition_count` return
for a list scan that will be unnested? Probably the parent-row chunk
count; unnest introduces fresh element-domain partitioning. Revisit
when list scanning lands.

**Q7. `TeePlan` buffer bounds.** Per-tee fixed bound, or session-level
budget? Probably per-tee with a session default. If a slow consumer
stalls fast ones for too long, escalate via a metric — bug to fix in
the rule that introduced the tee, not in the runtime.

**Q8. Coexistence with `LayoutReader`.** Two parallel implementations
behind a feature flag / session option until parity. No code sharing —
they have fundamentally different shapes.
