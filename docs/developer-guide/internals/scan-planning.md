# Scan Plans

A scan plan is the physical plan for satisfying one scan query. It is a tree of physical operators
over a row domain, describing the reads and derived work needed to produce that query's result.

## Operators, not layout mirrors

Plan operators describe *what work happens*, not *which layout produced it*. Their identity and
operator-specific state are independent of the source layout kind. The complete plan node is not:
its common lazy-child container can own hidden source state used to materialize individual children
on demand.

| Operator | Work |
| --- | --- |
| `SegmentScan` | read one segment and decode it to an array |
| `Concat` | concatenate its children row-wise |
| `Pack` | assemble a struct from one child per field, plus optional validity |
| `Take` | index `values` by `codes` |
| `ListPack` | assemble a list from elements and offsets, plus optional validity |
| `Eval` | apply an expression to its child |
| `RowIdx` | offset row numbers into the file's row domain |

Naming operators for what they compute is what lets one rule cover every case. `Concat` of
`Concat` flattens on shape alone, and `Take` over `SegmentScan` is the dictionary pushdown,
regardless of the source layout.

The stored layout tree describes all physical data in a file. A plan is query-specific: it is built
from that tree for one projection, filter, and row domain. Different queries over the same file can
therefore produce different plans.

## Optimization

Child replacement is implemented by the common plan container rather than by every operator. It
replaces the external child container, clones `PlanData`, then invokes the operator's
`PlanVTable::with_children` callback to validate the new children and refresh derived caches such
as `Concat` row offsets. Rules therefore rewrite the generic tree without reconstructing common
plan fields inside each operator.

Optimization rewrites the initial tree so that each expression is evaluated as close as possible to
the physical data that can satisfy it. Every rewrite must preserve the query result, including its
dtype, row domain, row order, row identity, null behavior, and observable errors.

Planning does not read segment data. It constructs and optimizes a description of the work that a
later execution stage will perform.

## Vtables

Each operator is a small vtable type implementing `PlanVTable`, paired with a `Plan<V>` container
over a shared `PlanRef`. `PlanRef` points to one allocation whose ordinary fields hold the operator
ID, dtype, row count, and lazy children. Only the unsized tail containing the vtable and
`V::PlanData` is erased behind `dyn DynPlan`, so common-field reads do not use dynamic dispatch.
`Plan<V>` provides typed access to that operator data through `Deref`.

`PlanVTable` also carries `id` and a `Metadata` codec. Operators with no unrecoverable state
already serialize their metadata; the ones holding a read context or a bound expression return
`None` until those codecs exist.

## Execution

Each operator executes over a row range and selection mask. `SegmentScan` reads its segment,
structural operators combine their children, and `Eval` applies the remaining derived work.
`vortex-scan-v2` copies the existing scan orchestration around this API, so the original
`LayoutReader` scanner is untouched while the plan-native path is developed.

## Future work

Still to come: a plan registry and foreign operator placeholder so third-party operators survive
a round trip, and a serialization envelope.
