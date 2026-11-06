# Scan Plans

A scan plan is the physical plan for satisfying one scan query. It is a tree of physical operators
over a row domain, describing the reads and derived work needed to produce that query's result.

## Operators, not layout mirrors

Plan operators describe *what work happens*, not *which layout produced it*. Their identity and
operator-specific state are independent of the source layout kind. The complete plan node is not:
each of its lazy child slots can own hidden source state used to materialize that child on demand.

| Operator | Work |
| --- | --- |
| `SegmentScan` | read one segment and decode it to an array |
| `Concat` | concatenate its children row-wise |
| `Pack` | assemble a struct from an optional validity slot, then one slot per field |
| `Take` | index `values` by `codes` |
| `ListPack` | assemble a list from elements, offsets, and an optional validity slot |
| `Eval` | apply an expression to its child |
| `RowIdx` | offset row numbers into the file's row domain |

Naming operators for what they compute is what lets one rule cover every case. `Concat` of
`Concat` flattens on shape alone, and `Take` over `SegmentScan` is the dictionary pushdown,
regardless of the source layout.

The stored layout tree describes all physical data in a file. A plan is query-specific: it is built
from that tree for one projection, filter, and row domain. Different queries over the same file can
therefore produce different plans.

## Slots

A plan's children live in fixed-position slots, as an array's do. Each operator defines what every
slot position holds, and an absent child is an empty slot rather than a missing entry, so the
positions of the other children never depend on which are present. A non-nullable `Pack` has an
empty validity slot 0 and its fields from slot 1; a pruning `Zoned` plan has an empty data slot and
its zones in slot 1.

Lowering maps a layout's slots to plan slots in one place, and most operators use the layout's
own slot positions. Compiling a plan into a pipeline is the one place slots become children:
pipelines have no absent inputs, so a plan that joins its children compiles its present slots in
slot order, and inlet `i` of its source reads the `i`-th present slot.

## Optimization

Slot replacement is implemented by the common plan container rather than by every operator. It
replaces the external slots, clones `PlanData`, then invokes the operator's
`PlanVTable::with_slots` callback to validate the new slots and refresh derived caches such as
`Concat` row offsets. Rules therefore rewrite the generic tree without reconstructing common
plan fields inside each operator.

Optimization rewrites the initial tree so that each expression is evaluated as close as possible to
the physical data that can satisfy it. Every rewrite must preserve the query result, including its
dtype, row domain, row order, row identity, null behavior, and observable errors.

Planning does not read segment data. It constructs and optimizes a description of the work that a
later execution stage will perform.

## Vtables

Each operator is a small vtable type implementing `PlanVTable`, paired with a `Plan<V>` container
over a shared `PlanRef`. `PlanRef` points to one allocation whose ordinary fields hold the operator
ID, dtype, row count, and lazy child slots. Only the unsized tail containing the vtable and
`V::PlanData` is erased behind `dyn DynPlan`, so common-field reads do not use dynamic dispatch.
`Plan<V>` provides typed access to that operator data through `Deref`.

`PlanVTable` also carries `id` and a `Metadata` codec. Operators with no unrecoverable state
already serialize their metadata; the ones holding a read context or a bound expression return
`None` until those codecs exist.

## Future work

Plans currently stop at construction and optimization. Still to come: a plan registry and foreign
operator placeholder so third-party operators survive a round trip, a serialization envelope, and
an execution stage that walks an optimized plan, reads the referenced segments, and returns the
query result.
