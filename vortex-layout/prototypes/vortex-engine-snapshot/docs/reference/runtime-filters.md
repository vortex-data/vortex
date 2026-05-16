# Runtime filters reference

> **Status:** Draft.
> **Progress:** Catalogues the typed filter mechanism, the publish /
> subscribe / translate flow, and the per-filter semantics. Concept
> overview is in [Runtime filters](../concepts/runtime-filters.md).
> The first concrete operator slice — `RangeFilter` published by
> `Limit`, consumed by `IntSource` — has not landed yet; the catalogue
> below is the target shape.
> **Open questions:** publication granularity (write-once vs.
> monotonically tightening); multi-key prefix `RangeFilter` for
> sort-merge cursor exchange; bloom filter sizing heuristics; whether
> `KeyListFilter` should also carry approximate cardinality counts;
> source-side translator chains and gating on local-state ready.

The engine has no row demand. Sideways information passing is
expressed as **typed filters published as `Resource`s** captured by
sink/source closures across pipeline boundaries. Each filter type
has explicit semantics, opt-in publishers and consumers, and a
typed `Resource` API.

## Catalogue

### `RangeFilter<K>`

```rust
pub struct RangeFilter<K> {
    /// Closed-open `[start, end)` over the key domain.
    pub start: K,
    pub end: K,
}
```

Published by:

- `Limit`: `RangeFilter<RowIdx>(start = 0, end = K)`.
- `TopK` with sorted input: `RangeFilter<RowIdx>(0, K)` once
  saturated.
- `MergeJoin` cursor exchange: `RangeFilter<JoinKey>(min_unread,
  max_unread)`. The slower side publishes its remaining range so
  the faster side can skip ahead.

Consumed by:

- Ordered sources (`IntSource`, `VortexScanSource`): clamp the
  read cursor to the range; for Vortex layouts, skip stripes whose
  zone map falls outside.
- Filter operators on the same key: drop rows outside.

Translation:

- `ParentChildMin` translates `RangeFilter<ParentIdx>(a, b)` to
  `RangeFilter<ChildIdx>(offsets[a], offsets[b])` using its
  offset table.
- Each operator type writes its own translator if it wants to
  forward a filter to its input.

### `BloomFilter<K>`

```rust
pub struct BloomFilter<K> {
    bits:    Vec<u8>,            // bitset, sized by build cardinality
    hashers: Vec<u64>,            // seeds
    _phantom: PhantomData<K>,
}
```

Published by:

- `HashJoinBuild`: bloom over the build-side join keys.
- `Distinct` build phase: bloom over seen keys.

Consumed by:

- Probe-side scans / filters: test each candidate row's key
  against the bloom; drop on miss.

No structural translation. Bloom filters do not compose through
relations the way range filters do.

### `KeyListFilter<K>`

```rust
pub struct KeyListFilter<K> {
    keys: Vec<K>,  // sorted, deduplicated
}
```

Published by:

- Low-cardinality aggregates: exact list of seen group keys.
- Filter with `IN (...)` predicate after the optimizer has bound
  the literal list.

Consumed by:

- Sources / filters that binary-search the list to skip non-member
  rows.

Stricter than `BloomFilter<K>` (zero false positives) but bounded
by published cardinality.

## Publication

A producer operator declares it publishes a filter via its
`OperatorProperties`. At plan time the runtime creates a typed
`Resource` for each declaration. The producer publishes through a
write API on its `LocalState`; the resource's readiness signal
fires on first write (or on each tighten — see open questions).

```rust
// Inside an operator's poll_*:
ctx.shared().filters().publish_range(
    RangeFilter { start: 0, end: self.k },
);
```

The runtime owns the `Resource` lifetime; consumers acquire a
typed handle at lane init.

## Consumption

A consumer operator declares its filter dependencies via
`OperatorProperties`. At lane init the runtime hands the operator
a typed handle:

```rust
let range_handle: ResourceHandle<RangeFilter<RowIdx>> =
    ctx.shared().filters().subscribe_range("parent_range")?;
```

In `poll_next`, the consumer reads the current filter through the
handle:

```rust
pub trait ResourceHandle<F> {
    fn try_snapshot(&self) -> Option<F>;
    fn wait_for_publication(&self) -> WorkHandle<()>;
}
```

- `try_snapshot()` — current value, or `None` if nothing has been
  published yet.
- `wait_for_publication()` — barrier-style signal that fires on
  first publish. Useful for consumers that *require* the filter
  (vs. those that opportunistically use it).

For most consumers (sources that skip when a filter is available
but read everything when it isn't), `try_snapshot` is enough.

## Translation

Operators that forward a filter to their input implement a
per-filter, per-operator translator:

```rust
impl FilterTranslator<RangeFilter<ParentIdx>, RangeFilter<ChildIdx>> for ParentChildMin {
    fn translate(
        &self,
        local: &Self::LocalState,
        output: &RangeFilter<ParentIdx>,
    ) -> Option<RangeFilter<ChildIdx>> {
        let offsets = &local.offsets;
        if output.end > offsets.len() as u64 - 1 { return None; }
        Some(RangeFilter {
            start: offsets[output.start as usize] as ChildIdx,
            end:   offsets[output.end   as usize] as ChildIdx,
        })
    }
}
```

The runtime composes translators when an operator's input is
itself produced by a translator-publishing operator, producing
chained filters that reach the source.

## When to use which mechanism

| Mechanism | Use for | Don't use for |
|---|---|---|
| Static plan rewrite | Constant predicates, projections, sort-aware Limit | Anything that depends on runtime data |
| `RangeFilter<K>` | Ordered domains: Limit, TopK, sorted merge-join cursor | Hash-keyed work |
| `BloomFilter<K>` | Probabilistic set membership for hash joins | Ordered ranges |
| `KeyListFilter<K>` | Low-cardinality exact membership (small `IN`-list, small aggregate keys) | Large key sets (bloom is better) |

## What is not in scope

- **Updateable filters with strict monotonicity guarantees**
  beyond what bloom and range provide. If "tighten only" semantics
  are needed for a filter type, add it explicitly per type.
- **Cross-task filters.** Filters are local to one `LoweredPlan`.
  Cross-task filter propagation rides on the same solution
  cross-task resource publication eventually gets.
- **Adaptive filter introduction** — an optimizer that decides
  whether the filter pays for itself at runtime. The plan inserts
  the filters it knows are correctness-neutral; the consumer's
  skip logic is either fast (zone-map skip) or no-op (consumer
  doesn't subscribe).

## See also

- [Runtime filters (concept)](../concepts/runtime-filters.md).
- [Lowering API](lowering-api.md) for how `Resource`s are
  captured by closures in sink and source nodes.
