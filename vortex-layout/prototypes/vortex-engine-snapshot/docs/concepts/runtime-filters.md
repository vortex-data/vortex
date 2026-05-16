# Runtime filters

> **Status:** Accepted.
> **Progress:** Concept-level overview of the typed-filter mechanism
> for sideways information passing. Per-filter semantics, the
> consumer trait, and the exact `Resource` API live in
> [Runtime filters reference](../reference/runtime-filters.md).
> **Open questions:** filter-type catalogue extensions; tightening
> semantics for monotonic publishers; cross-task filter propagation.

## Why typed filters

The engine has no row demand. The patterns analytical engines
typically extract through sideways information passing — limit-
aware pruning, top-K threshold propagation, sort-merge cursor
exchange, hash-join build-side bloom filters, low-cardinality
probe pruning — are expressed instead as **typed filters
published as `Resource`s**.

A filter has three properties:

- **Explicit semantics.** A `RangeFilter<K>` always means "rows
  outside this range can be skipped"; not "I would like these
  rows more."
- **Opt-in at both ends.** Producers declare what filter they
  publish; consumers declare what filter they subscribe to.
- **Pay-per-use.** Operators that don't subscribe to a filter type
  pay nothing.

This is the standard sideways-information-passing (SIP) pattern
used by analytical engines like DuckDB, Velox/Photon, and
Snowflake.

## The starter catalogue

Three filter types cover the bulk of analytical SIP cases.

### `RangeFilter<K>`

A closed-open `[start, end)` interval over an ordered key.

- **Published by**: `Limit`, `TopK` once saturated, `MergeJoin`
  cursors.
- **Consumed by**: ordered sources (skip stripes whose zone map
  falls outside), ordered filters (drop rows outside).
- **Translates**: through structural operators that know how
  their output key maps to their input key. `ParentChildMin`
  translates `RangeFilter<ParentIdx>` to `RangeFilter<ChildIdx>`
  using its offset table.

### `BloomFilter<K>`

Probabilistic set membership.

- **Published by**: `HashJoinBuild`, `Distinct` build phase.
- **Consumed by**: probe-side scans / filters that test each key
  against the bloom.
- **Translates**: not through structural relations. Blooms don't
  compose.

### `KeyListFilter<K>`

Exact set membership for a small list.

- **Published by**: low-cardinality aggregates, bound `IN (...)`
  predicates.
- **Consumed by**: sources / filters that binary-search the list.
- **Stricter than `BloomFilter`** (zero false positives) but
  bounded by published cardinality.

## Mechanism in one sentence per stage

- **Declare** at plan time: the producer's `OperatorProperties`
  names the filter type; the runtime allocates a typed `Resource`.
- **Publish** at run time: producer writes into the resource
  through an API on its `LocalState`. The resource's readiness
  fires on first write (or on each tighten — see open questions).
- **Subscribe** at lane init: consumer obtains a typed
  `ResourceHandle<F>` from `ctx.shared().filters()`.
- **Consume** in `poll_next` / `push_input`: consumer calls
  `try_snapshot()` for opportunistic skipping, or
  `wait_for_publication()` to block until the filter exists.

For most consumers, opportunistic skipping is right: read
everything when no filter exists, clamp reads when one shows up.

## When to use what

| Mechanism | Use for | Don't use for |
|---|---|---|
| Static plan-time push-down | Constant predicates, projections, sort-aware limit | Anything that depends on runtime data |
| `RangeFilter<K>` | Ordered domains: Limit, TopK, sort-merge cursor | Hash-keyed work |
| `BloomFilter<K>` | Probabilistic set membership for hash joins | Ordered ranges |
| `KeyListFilter<K>` | Low-cardinality exact membership | Large key sets (bloom is better) |

If none of the above fits, the option is to add a new filter type
to the catalogue with explicit semantics — not to introduce a
per-row demand mechanism.

## What runtime filters do not do

- **No uniform translation pass.** Each operator type writes its
  own per-filter translator if it wants to forward a filter to
  its input.
- **No cross-task filters yet.** Filters are local to one
  `LoweredPlan`. Cross-task propagation rides on the same
  solution cross-task resource publication eventually gets.
- **No adaptive insertion.** The optimizer decides at plan time
  whether to insert a filter. The filter's runtime cost is paid
  for by the consumer's skip logic being either fast (zone-map
  skip) or no-op (consumer doesn't subscribe).

## Where to read next

- [Runtime filters reference](../reference/runtime-filters.md):
  exact resource API, consumer trait, per-filter semantics.
- [Execution model](execution-model.md): how `Resource`s in
  general flow between pipelines.
