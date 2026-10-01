# Aggregate Statistics

Arrays retain finalized aggregate results in `Aggregations`. This replaces the fixed runtime
`ArrayStats`, `StatsSet`, and `Stat` APIs with the same aggregate functions used by execution.

## Cache API

`array.aggregations()` returns an `AggregationsRef` bound to that immutable array. The store has no
implicit strong reference to its input. The key is the full `AggregateFnRef`, including its options.

```rust
let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
let known: Precision<Scalar> = array.aggregations().get_result(&aggregate);
let result: Scalar = array.aggregations().compute_result(&aggregate, ctx)?;
let snapshot: AggregateResults = array.aggregations().snapshot_results();
```

`get_result` reads metadata without execution. `compute_result` returns an exact cached result or
computes one outside the cache lock. Concurrent misses can compute the same result. A failed target
result is not retained, although successful nested computations can retain their own results.

`Precision::Absent` means unknown. `Precision::Exact` includes null, zero, and false results.
`Precision::Inexact` carries a bound defined by the aggregate. An inexact value cannot answer an
exact computation.

`AggregateResults` is an immutable snapshot for file summaries and serialization. Its public
constructor checks unique keys and result dtypes. It does not establish that those values describe
an array. Producers that avoid scanning must prove that their facts describe the exact immutable
input before using the documented unsafe `seed_result` boundary.

## Migrating Callers

Array computations use an aggregate request instead of a fixed statistic:

```rust
// Before
let sum = array.statistics().compute_stat(Stat::Sum, ctx)?;

// After
let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
let sum = array.aggregations().compute_result(&aggregate, ctx)?;
```

Safe arbitrary setters are removed. An encoding that already proves sortedness can seed that fact
for its exact output array through the producer boundary:

```rust
// SAFETY: this producer establishes ascending order for every value in this exact array.
unsafe {
    array.aggregations().seed_result(
        IsSorted.bind(IsSortedOptions { strict: false }),
        Precision::Exact(true.into()),
    )?;
}
```

File selections and lookups use the same full request. Numerical requests must explicitly skip NaNs
because the existing footer has no field for NaN-including options:

```rust
let minimum = Min.bind(NumericalAggregateOpts::skip_nans());
let options = options.with_file_statistics(vec![minimum.clone(), NullCount.bind(EmptyOptions)]);
let known = file.file_stats().unwrap().fields()[field_index].get_result(&minimum);
```

Callers that own a stream retain their accumulator rather than merging finalized snapshots. The
accumulator can reuse a chunk's exact cached result when its aggregate supports `partial_from_result`:

```rust
let sum = Sum.bind(NumericalAggregateOpts::skip_nans());
let mut accumulator = sum.accumulator(&input_dtype)?;
for chunk in chunks {
    accumulator.accumulate(&chunk, ctx)?;
}
let result = accumulator.finish()?;
```

A custom aggregate opts into this reuse only if its finalized result reconstructs a complete partial.
An aggregate with a richer partial leaves the default conversion unchanged.

## Results and Partials

An accumulator owns the state needed to merge chunks. A finalized scalar can replace a partial only
when the aggregate implements `partial_from_result`. The default declines this conversion, even
when the result and state dtypes match.

Min, max, counts, and the original sum support proven conversions. Sum's null result denotes
saturated overflow, while its empty state is zero. Sortedness and constantness decline because their
booleans omit boundary or representative values. Rich states such as `SumV2` also decline.

`StatFn` reads metadata and converts results through this explicit aggregate contract. It does not
scan the input. Missing metadata stays unknown, and supported bounds retain their precision.
Streaming file summaries accumulate typed partials and finalize them only after merging chunks.
An unseen stream has no results, while a received empty batch preserves known count and sum identities.
Nonempty all-null input now retains exact null extrema instead of omitting those fields.

## Array Rewrites

A new representation receives a fresh store. An aggregate opts into transfer with
`is_representation_invariant` when its result depends only on logical values, order, and dtype.
Custom aggregates keep their results on the original representation by default. A rewrite that
changes the input dtype, including nullability, discards inherited results.

`UncompressedSizeInBytes` does not opt in. Some implementations count retained backing buffers, so
equal logical values can have different sizes in different representations. Slice and subset
propagation use separate rules for facts that remain valid, such as known constantness or sortedness.

Generic aggregate results do not prove unchecked constructor invariants. Decimal precision, list
offsets, validity masks, and list-view trimming validate physical values independently before
unchecked construction. Masked and patched array construction and row output validation also inspect
actual validity rather than generic aggregate facts. Decimal casts resolve lazy validity once and
use that same physical mask for precision validation and the emitted array.

## Wire Compatibility

The node, footer, and legacy zone formats retain their field identifiers and scalar encodings.
`stats::compat` translates them into finalized aggregate results and validates historical scalar
types. Legacy extrema use the input dtype on the wire and the nullable aggregate dtype in memory.
Numerical legacy fields always use NaN-skipping options.

Node serialization projects cache snapshots into representable fields. Footer selections reject
functions or options that the existing format cannot represent. Variable-length truncation changes
only a snapshot, leaving exact live cache entries intact. Legacy sortedness and constantness flags
remain final results and never become aggregate partials.

The decoder also preserves hints on an unknown root encoding when the caller supplies its dtype and
length. Opaque foreign children retain the previous decode behavior because their logical types
cannot be recovered without the missing encoding plugin.
