# Migrating statistics to aggregates

Arrays no longer own a statistics cache. `ArrayStats`, `StatsSet`, and the array `statistics()`
accessors have been removed without a deprecation layer. Computation now goes through aggregate
helpers or accumulators, while `FileStatistics` continues to expose stored file summaries.

## Computing aggregates

Replace calls through `array.statistics()` with helpers under `vortex_array::aggregate_fn::fns`:

| Previous call | Replacement |
|---|---|
| `compute_min` and `compute_max` | `min_max(&array, &mut ctx, options)?` |
| `compute_stat(Stat::Sum, ctx)` | `sum(&array, &mut ctx)?` |
| `compute_null_count` | `null_count(&array, &mut ctx)?` |
| `compute_stat(Stat::NaNCount, ctx)` | `nan_count(&array, &mut ctx)?` |
| `compute_is_constant` | `is_constant(&array, &mut ctx)?` |
| `compute_is_sorted` | `is_sorted(&array, &mut ctx)?` |
| `compute_is_strict_sorted` | `is_strict_sorted(&array, &mut ctx)?` |
| `compute_uncompressed_size_in_bytes` | `uncompressed_size_in_bytes(&array, &mut ctx)?` |

The helpers compute results without retaining them on the array. `min_max` computes both extrema
in one call, with options that control whether NaNs are skipped:

```rust
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;

let extrema = min_max(&array, &mut ctx, NumericalAggregateOpts::skip_nans())?;
if let Some(extrema) = extrema {
    let min = extrema.min;
    let max = extrema.max;
}
```

For a stream of chunks, an accumulator retains the state until the result is finalized:

```rust
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum::Sum;

let aggregate = Sum.bind(NumericalAggregateOpts::skip_nans());
let mut accumulator = aggregate.accumulator(&dtype)?;
for chunk in chunks {
    accumulator.accumulate(&chunk, &mut ctx)?;
}
let result = accumulator.final_scalar()?;
```

Encoding-specific aggregate kernels still apply. Extensions that called the removed statistics
methods must use helpers or accumulators instead. Existing aggregate kernels can remain registered
with the session.

## File summaries

### Selecting aggregates

`with_file_statistics` now accepts aggregate functions with their options, replacing fixed
identifiers such as `Stat::Min`:

```rust
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_file::WriteOptionsSessionExt;

let options = session.write_options().with_file_statistics(vec![
    Min.bind(NumericalAggregateOpts::skip_nans()),
    Max.bind(NumericalAggregateOpts::skip_nans()),
    NullCount.bind(EmptyOptions),
]);
```

The default selection is unchanged: minimum, maximum, sum, null count, and NaN count. It is available
as `vortex_array::stats::default_file_aggregates()`. Numerical summaries skip NaNs, and aggregates
that cannot handle a field's input type have no entry. An empty selection disables file summaries.

The writer computes summaries for each top-level struct field, or for the single non-struct column.
When both minimum and maximum are selected, they share one computation. String and binary extrema
are truncated after finalization to the existing variable-length limit. A truncated minimum is a
lower bound, and a truncated maximum is an upper bound. If no safe upper bound exists, the maximum
is omitted.

The existing footer format limits which aggregates and options can be stored. Unsupported selections,
including numerical aggregates configured to include NaNs, are rejected before any file bytes are
written.

### Reading results

`VortexFile::file_stats()` and `Footer::statistics()` remain the entry points for `FileStatistics`.
Replace `stats_sets()` with `fields()`, and look up results by aggregate function:

```rust
// Before.
let minimum = file_stats.stats_sets()[0].get(Stat::Min);

// After.
let minimum = file_stats.fields()[0]
    .get(&Min.bind(NumericalAggregateOpts::skip_nans()));
```

Each field contains immutable `AggregateResults`, with options included in every lookup key. A
NaN-skipping minimum therefore does not match a request for a minimum that includes NaNs.
`AggregateResults::iter()` returns the function/result pairs, and `FileStatistics::get(index)`
returns a field's results and input dtype.

Lookup returns `Precision<Scalar>`:

- `Absent` means the summary is missing.
- `Exact(value)` is a known result. The value can be null, for example when a sum overflows, which
  differs from missing metadata.
- `Inexact(value)` is a bound whose direction depends on the aggregate. A minimum is a lower bound
  on the actual minimum, and a maximum is an upper bound on the actual maximum.

Consumers that require exact results must continue to reject `Inexact` and `Absent` values.

### Finalized results and partial states

File summaries contain finalized results. They must not be passed directly to an accumulator's
state-merge API, even when the scalar types match.

For example, sortedness has a boolean result, but its partial state also records values at chunk
boundaries. Those values are needed to determine whether two sorted chunks are still sorted when
combined. The finalized boolean cannot supply that state.

File and zone binders resolve `stat(...)` expressions from compatible stored metadata. If no summary
is available, the expression returns a typed null without scanning the input array.

## Reusing results

Remove array-statistics setters, `with_stats_set`, and propagation through slices or takes. A caller
that reuses aggregate results can own a cache for one immutable input, keyed by `AggregateFnRef` so
that the options remain part of the key:

```rust
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

struct InputAggregates {
    array: ArrayRef,
    results: HashMap<AggregateFnRef, Scalar>,
}

impl InputAggregates {
    fn get(&mut self, aggregate: &AggregateFnRef, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        if let Some(result) = self.results.get(aggregate) {
            return Ok(result.clone());
        }

        let mut accumulator = aggregate.accumulator(self.array.dtype())?;
        accumulator.accumulate(&self.array, ctx)?;
        let result = accumulator.final_scalar()?;
        self.results.insert(aggregate.clone(), result.clone());

        Ok(result)
    }
}
```

A sample, slice, take, or other derived input needs a fresh cache because its aggregate results can
differ. The compressor retains its existing `ArrayAndStats` cache, which already follows this
ownership model. Migrating that cache to aggregate functions is a separate change.

Array displays and the TUI no longer show cached statistics. `ValidityExtractor` replaces
`StatsExtractor` for structural validity annotations, while the tree display's `stats` option still
controls byte sizes and validity annotations. Neither display computes aggregates.

## File compatibility

Existing files require no rewrite. Historical footer fields are converted to finalized aggregate
results during metadata reads, including sortedness and constantness flags. This conversion reads
no array values and preserves exact values, truncated bounds, scalar types, and missing summaries.

New writers omit `ArrayNode.stats`, and readers ignore those hints in historical arrays, including
nested children and IPC arrays. The FlatBuffers fields and generated `ArrayStats` wire type remain
for compatibility, as does the legacy zone-map reader. The footer format is unchanged, with no new
wire IDs or edition requirement.
