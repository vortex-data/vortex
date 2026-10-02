// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;

use futures::StreamExt;
use itertools::Itertools;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AccumulatorRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnSatisfaction;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::LayoutWriterContext;
use crate::layouts::zoned::default_bounded_stat_max_bytes;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Accumulates file statistics over `stream` as it passes through.
///
/// `stats` are the aggregates to record for every field, or `None` for each field's default (see
/// [`default_pruning_aggregate_fns`]). Only aggregates `ctx` allows are recorded in the nested
/// statistics: an explicitly requested aggregate it forbids fails, like it does for zone maps,
/// while forbidden defaults are left out.
///
/// # Errors
///
/// Returns an error if `stats` contains an aggregate `ctx` forbids.
pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Option<Arc<[AggregateFnRef]>>,
    session: &VortexSession,
    write_legacy_stats: bool,
    ctx: &LayoutWriterContext,
) -> VortexResult<(FileStatsAccumulator, SendableSequentialStream)> {
    let accumulator =
        FileStatsAccumulator::try_new(stream.dtype(), stats, session, write_legacy_stats, ctx)?;
    let stream = SequentialStreamAdapter::new(
        stream.dtype().clone(),
        stream.scan(accumulator.clone(), |acc, item| {
            future::ready(Some(acc.process(item)))
        }),
    )
    .sendable();
    Ok((accumulator, stream))
}

/// One aggregate function's value over a file statistics entry.
#[derive(Clone, Debug)]
pub struct AggregateStat {
    aggregate_fn: AggregateFnRef,
    /// The aggregate's partial state, which is what the footer stores. `None` for values recovered
    /// from the legacy `ArrayStats` footer format, which only stores results.
    partial: Option<Scalar>,
    value: Precision<Scalar>,
}

impl AggregateStat {
    fn from_accumulator(
        aggregate_fn: AggregateFnRef,
        accumulator: &AccumulatorRef,
    ) -> VortexResult<Self> {
        Ok(Self {
            aggregate_fn,
            partial: Some(accumulator.partial_scalar()?),
            value: Precision::exact(accumulator.final_scalar()?),
        })
    }

    /// Rebuilds an aggregate's value from its `partial` state over a field of `dtype`.
    pub fn try_from_partial(
        aggregate_fn: AggregateFnRef,
        partial: Scalar,
        dtype: &DType,
    ) -> VortexResult<Self> {
        let mut accumulator = aggregate_fn.accumulator(dtype)?;
        accumulator.combine_partials(partial.clone())?;
        let value = Precision::exact(accumulator.final_scalar()?);
        Ok(Self {
            aggregate_fn,
            partial: Some(partial),
            value,
        })
    }

    /// An aggregate value without a partial state, such as one read from the legacy `ArrayStats`
    /// footer format.
    pub fn from_value(aggregate_fn: AggregateFnRef, value: Precision<Scalar>) -> Self {
        Self {
            aggregate_fn,
            partial: None,
            value,
        }
    }

    /// The aggregate function this value was computed by.
    pub fn aggregate_fn(&self) -> &AggregateFnRef {
        &self.aggregate_fn
    }

    /// The aggregate's partial state, if known.
    pub fn partial(&self) -> Option<&Scalar> {
        self.partial.as_ref()
    }

    /// The aggregate's value. A null value means the aggregate saw no valid input.
    pub fn value(&self) -> &Precision<Scalar> {
        &self.value
    }
}

/// The aggregates recorded for one file statistics entry.
#[derive(Clone, Debug, Default)]
pub struct AggregateStats {
    aggregates: Vec<AggregateStat>,
}

impl AggregateStats {
    pub fn new(aggregates: Vec<AggregateStat>) -> Self {
        Self { aggregates }
    }

    pub fn is_empty(&self) -> bool {
        self.aggregates.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &AggregateStat> {
        self.aggregates.iter()
    }

    /// Returns the value of `requested`, from the recorded aggregate that best satisfies it, the
    /// way zone maps resolve aggregates: one that satisfies it exactly over one that only
    /// approximates it. The value is exact only if the recorded value is exact and satisfies
    /// `requested` exactly. Absent if no recorded aggregate satisfies `requested`, or if the value
    /// is null.
    pub fn get(&self, requested: &AggregateFnRef) -> Precision<Scalar> {
        let mut approximate = Precision::Absent;
        for stat in &self.aggregates {
            match stat.aggregate_fn.can_satisfy(requested) {
                AggregateFnSatisfaction::Exact => return non_null(stat.value.clone()),
                AggregateFnSatisfaction::Approximate => {
                    approximate = stat.value.clone().into_inexact();
                }
                AggregateFnSatisfaction::No => {}
            }
        }
        non_null(approximate)
    }

    /// Returns the recorded aggregates as a [`StatsSet`], for consumers keyed by [`Stat`].
    pub fn to_stats_set(&self) -> StatsSet {
        let mut stats_set = StatsSet::default();
        for stat in Stat::all() {
            let Some(requested) = stat.aggregate_fn() else {
                continue;
            };
            let value = self.get(&requested).and_then(Scalar::into_value);
            if !value.is_absent() {
                stats_set.set(stat, value);
            }
        }
        stats_set
    }
}

fn non_null(value: Precision<Scalar>) -> Precision<Scalar> {
    value.and_then(|value| (!value.is_null()).then_some(value))
}

/// Accumulates write-time statistics for a single file column.
struct StatsAccumulator {
    aggregates: Vec<(AggregateFnRef, AccumulatorRef)>,
}

impl StatsAccumulator {
    fn new(dtype: &DType, stats: Option<&[AggregateFnRef]>) -> Self {
        if !supports_file_stats(dtype) {
            return Self {
                aggregates: Vec::new(),
            };
        }

        // When no explicit list is requested, each leaf picks its own default the way
        // `default_zoned_aggregate_fns` does for zoned layouts: a byte-bounded min/max for
        // variable-length columns, exact min/max otherwise.
        let default_stats;
        let stats = match stats {
            Some(stats) => stats,
            None => {
                default_stats = default_pruning_aggregate_fns(dtype);
                default_stats.as_slice()
            }
        };

        let mut aggregates = Vec::new();
        for aggregate_fn in stats {
            // A dtype that doesn't support a given aggregate simply fails to build an
            // accumulator, which is silently skipped, matching this stat's absence from the
            // result.
            if let Ok(accumulator) = aggregate_fn.accumulator(dtype) {
                aggregates.push((aggregate_fn.clone(), accumulator));
            }
        }

        Self { aggregates }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        for (_, accumulator) in &mut self.aggregates {
            accumulator.accumulate(array, ctx)?;
        }
        Ok(())
    }

    /// Returns the accumulated aggregates that `allowed` permits.
    ///
    /// A bounded aggregate whose accumulator ended up exact, because no value exceeded its byte
    /// bound, is recorded as the exact aggregate it stands in for (e.g. `BoundedMax` as `Max`).
    /// The footer stores partial states, which don't record that exactness, so recording the exact
    /// aggregate is what keeps it.
    fn aggregate_stats(
        &self,
        allowed: impl Fn(&AggregateFnRef) -> bool,
    ) -> VortexResult<AggregateStats> {
        let mut out = Vec::with_capacity(self.aggregates.len());
        for (aggregate_fn, accumulator) in &self.aggregates {
            if let Some(exact) = exact_counterpart(aggregate_fn)
                && allowed(&exact)
                && accumulator.can_satisfy(&exact).is_exact()
            {
                // The exact counterparts' partial state is their result, which is the bounded
                // aggregate's result here.
                let value = accumulator.final_scalar()?;
                out.push(AggregateStat {
                    aggregate_fn: exact,
                    partial: Some(value.clone()),
                    value: Precision::exact(value),
                });
            } else if allowed(aggregate_fn) {
                out.push(AggregateStat::from_accumulator(
                    aggregate_fn.clone(),
                    accumulator,
                )?);
            }
        }
        Ok(AggregateStats::new(out))
    }
}

/// The exact aggregate a bounded aggregate approximates, if any.
fn exact_counterpart(aggregate_fn: &AggregateFnRef) -> Option<AggregateFnRef> {
    if aggregate_fn.is::<BoundedMax>() {
        Some(Max.bind(NumericalAggregateOpts::skip_nans()))
    } else if aggregate_fn.is::<BoundedMin>() {
        Some(Min.bind(NumericalAggregateOpts::skip_nans()))
    } else {
        None
    }
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn is_varlen_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
}

/// Default file-level pruning aggregates for `dtype`, chosen the way `default_zoned_aggregate_fns`
/// picks zoned aggregates: a byte-bounded min/max (capped at [`default_bounded_stat_max_bytes`])
/// for variable-length columns, and exact min/max otherwise. Callers wanting a different bound pass
/// their own `BoundedMax`/`BoundedMin` instead.
fn default_pruning_aggregate_fns(dtype: &DType) -> Vec<AggregateFnRef> {
    let (max, min) = if is_varlen_dtype(dtype) {
        let max_bytes = default_bounded_stat_max_bytes();
        (
            BoundedMax.bind(BoundedMaxOptions { max_bytes }),
            BoundedMin.bind(BoundedMinOptions { max_bytes }),
        )
    } else {
        (
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Min.bind(NumericalAggregateOpts::skip_nans()),
        )
    };

    vec![
        max,
        min,
        Sum.bind(NumericalAggregateOpts::skip_nans()),
        NullCount.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
    ]
}

/// Computes the post-order sequence of `(FieldPath, DType)` entries that file-level statistics
/// are stored against.
///
/// Each leaf field (including opaque `List`/`FixedSizeList` columns, which are not recursed into)
/// gets one entry. Each **nullable** struct additionally gets a trailing entry, keyed by its own
/// path and dtype, inserted immediately after its children's entries — this carries the struct's
/// own null count. Dtypes that don't support file stats (see [`supports_file_stats`]), such as
/// [`DType::Variant`], are skipped entirely: no entry is emitted for them or anything beneath
/// them.
///
/// This function is the single source of truth for the number and order of stats entries, used
/// both when accumulating stats at write time and when reconstructing them from the footer at
/// read time, so no paths or dtypes need to be persisted in the flatbuffer itself.
pub fn postorder_stats_layout(dtype: &DType) -> Vec<(FieldPath, DType)> {
    let mut out = Vec::new();
    postorder_stats_layout_into(dtype, FieldPath::root(), &mut out);
    out
}

fn postorder_stats_layout_into(dtype: &DType, path: FieldPath, out: &mut Vec<(FieldPath, DType)>) {
    match dtype.as_struct_fields_opt() {
        Some(struct_fields) => {
            for (name, field_dtype) in struct_fields.names().iter().zip(struct_fields.fields()) {
                postorder_stats_layout_into(&field_dtype, path.clone().push(name.clone()), out);
            }
            if dtype.nullability() == Nullability::Nullable {
                out.push((path, dtype.clone()));
            }
        }
        None if !supports_file_stats(dtype) => {}
        None => out.push((path, dtype.clone())),
    }
}

/// A node in the tree of accumulators mirroring [`postorder_stats_layout`]'s walk of a `DType`.
enum StatsNode {
    /// An opaque leaf: a dtype with no addressable children. Curently includes List/FixedSizedList and Map.
    Leaf(StatsAccumulator),
    /// A dtype that does not support file stats (e.g. [`DType::Variant`]); contributes no entries.
    Skipped,
    /// A dtype with addressable children. Currently only built for `DType::Struct`.
    Container {
        /// One child per addressable sub-dtype, in declaration order (one per struct field,
        /// today).
        children: Vec<(Field, StatsNode)>,
        /// Stats computed over the container's own, undecomposed dtype (e.g. a `Struct`'s null
        /// count), reusing `Leaf`/`Skipped` rather than a separate accumulator-plus-flag pair:
        /// `Leaf` when the container is nullable (matching the trailing entry
        /// [`postorder_stats_layout_into`] emits for it), `Skipped` otherwise — which also means a non-nullable struct's null count is never even
        /// accumulated, not just never emitted.
        own: Box<StatsNode>,
    },
}

impl StatsNode {
    fn build(dtype: &DType, stats: Option<&[AggregateFnRef]>) -> Self {
        match dtype.as_struct_fields_opt() {
            Some(struct_fields) => {
                let children = struct_fields
                    .names()
                    .iter()
                    .zip(struct_fields.fields())
                    .map(|(name, field_dtype)| {
                        (Field::Name(name.clone()), Self::build(&field_dtype, stats))
                    })
                    .collect();
                let own = if dtype.nullability() == Nullability::Nullable {
                    Self::Leaf(StatsAccumulator::new(dtype, stats))
                } else {
                    Self::Skipped
                };
                Self::Container {
                    children,
                    own: Box::new(own),
                }
            }
            None if !supports_file_stats(dtype) => Self::Skipped,
            None => Self::Leaf(StatsAccumulator::new(dtype, stats)),
        }
    }

    fn push_chunk(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => acc.push_chunk(array, ctx),
            Self::Container { .. } => {
                let struct_array = array.clone().execute::<StructArray>(ctx)?;
                self.push_struct_chunk(array, &struct_array, ctx)
            }
        }
    }

    /// Pushes a chunk into a `Container` node given an already-executed `StructArray` for `array`.
    ///
    /// Lets callers that already had to execute the chunk to a `StructArray` for another purpose
    /// (e.g. [`FileStatsAccumulator::process`], which also feeds the legacy per-top-level-field
    /// accumulators from the same execution) avoid doing so a second time.
    ///
    /// `Container` is only ever built for `DType::Struct` today, so every child is addressed by
    /// [`Field::Name`] and extracted via [`StructArray::iter_unmasked_fields`]. A future `List`/
    /// `Map` container would need a different extraction here (e.g. flattened elements, or
    /// derived per-row shape arrays), dispatched per child's [`Field`] kind.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not `Self::Container`.
    fn push_struct_chunk(
        &mut self,
        array: &ArrayRef,
        struct_array: &StructArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Self::Container { children, own } = self else {
            vortex_panic!("push_struct_chunk is only called on Container nodes");
        };
        // The container's own `ArrayRef` already carries the validity needed to compute its own
        // stats (e.g. null count), so we push it directly rather than building a synthetic array.
        own.push_chunk(array, ctx)?;
        for ((_, child), field) in children
            .iter_mut()
            .zip_eq(struct_array.iter_unmasked_fields())
        {
            child.push_chunk(field, ctx)?;
        }
        Ok(())
    }

    /// Appends this node's aggregates that `allowed` permits, in the same post-order as
    /// [`postorder_stats_layout`].
    fn collect_aggregate_stats(
        &self,
        allowed: &impl Fn(&AggregateFnRef) -> bool,
        out: &mut Vec<AggregateStats>,
    ) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) => {
                out.push(acc.aggregate_stats(allowed)?);
                Ok(())
            }
            Self::Container { children, own } => {
                for (_, child) in children {
                    child.collect_aggregate_stats(allowed, out)?;
                }
                own.collect_aggregate_stats(allowed, out)
            }
        }
    }
}

/// An array stream processor that computes aggregate statistics for every field, recursing into
/// nested (possibly nullable) structs. See [`postorder_stats_layout`] for the entry ordering.
///
/// Alongside `root`'s post-order nested layout, this also maintains a `legacy` accumulation
/// covering only the top-level struct fields (or the whole array, for a non-struct root),
/// matching the layout file stats had before nested struct support: one entry per top-level
/// field, with no recursion into nested structs. `legacy` is empty (and never accumulated into)
/// when `write_legacy_stats` is `false`.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    root: Arc<Mutex<StatsNode>>,
    legacy: Arc<Mutex<Vec<StatsAccumulator>>>,
    write_legacy_stats: bool,
    writer_ctx: LayoutWriterContext,
    ctx: Arc<Mutex<ExecutionCtx>>,
}

impl FileStatsAccumulator {
    fn try_new(
        dtype: &DType,
        stats: Option<Arc<[AggregateFnRef]>>,
        session: &VortexSession,
        write_legacy_stats: bool,
        writer_ctx: &LayoutWriterContext,
    ) -> VortexResult<Self> {
        for aggregate_fn in stats.iter().flat_map(|stats| stats.iter()) {
            if !writer_ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }

        let root = Arc::new(Mutex::new(StatsNode::build(dtype, stats.as_deref())));

        let legacy = Arc::new(Mutex::new(if write_legacy_stats {
            match dtype.as_struct_fields_opt() {
                Some(struct_fields) => struct_fields
                    .fields()
                    .map(|field_dtype| StatsAccumulator::new(&field_dtype, stats.as_deref()))
                    .collect(),
                None => vec![StatsAccumulator::new(dtype, stats.as_deref())],
            }
        } else {
            Vec::new()
        }));

        Ok(Self {
            root,
            legacy,
            write_legacy_stats,
            writer_ctx: writer_ctx.clone(),
            ctx: Arc::new(Mutex::new(session.create_execution_ctx())),
        })
    }

    fn process(
        &self,
        chunk: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<(SequenceId, ArrayRef)> {
        let (sequence_id, chunk) = chunk?;
        let mut ctx = self.ctx.lock();
        if chunk.dtype().is_struct() {
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            if self.write_legacy_stats {
                for (acc, field) in self
                    .legacy
                    .lock()
                    .iter_mut()
                    .zip_eq(struct_chunk.iter_unmasked_fields())
                {
                    acc.push_chunk(field, &mut ctx)?;
                }
            }
            self.root
                .lock()
                .push_struct_chunk(&chunk, &struct_chunk, &mut ctx)?;
        } else {
            if self.write_legacy_stats {
                self.legacy.lock()[0].push_chunk(&chunk, &mut ctx)?;
            }
            self.root.lock().push_chunk(&chunk, &mut ctx)?;
        }
        Ok((sequence_id, chunk))
    }

    /// Returns the accumulated aggregates of every entry in the post-order nested layout (see
    /// [`postorder_stats_layout`]), leaving out the aggregates the writer context forbids.
    pub fn aggregate_stats(&self) -> VortexResult<Vec<AggregateStats>> {
        let allowed =
            |aggregate_fn: &AggregateFnRef| self.writer_ctx.allows_aggregate(&aggregate_fn.id());
        let mut out = Vec::new();
        self.root
            .lock()
            .collect_aggregate_stats(&allowed, &mut out)?;
        Ok(out)
    }

    /// Returns the legacy top-level-fields-only aggregates (one per top-level struct field, or a
    /// single entry for a non-struct root dtype). Empty if `write_legacy_stats` was `false`.
    ///
    /// The legacy `ArrayStats` format has a fixed slot per [`Stat`] rather than open-ended
    /// aggregates, so it isn't restricted by the writer context.
    pub fn legacy_aggregate_stats(&self) -> VortexResult<Vec<AggregateStats>> {
        self.legacy
            .lock()
            .iter()
            .map(|acc| acc.aggregate_stats(|_| true))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;

    fn stats_set(acc: &StatsAccumulator) -> VortexResult<StatsSet> {
        Ok(acc.aggregate_stats(|_| true)?.to_stats_set())
    }

    fn node_stats_sets(node: &StatsNode) -> VortexResult<Vec<StatsSet>> {
        let mut aggregates = Vec::new();
        node.collect_aggregate_stats(&|_| true, &mut aggregates)?;
        Ok(aggregates
            .iter()
            .map(AggregateStats::to_stats_set)
            .collect())
    }

    fn agg(stat: Stat) -> AggregateFnRef {
        stat.aggregate_fn()
            .vortex_expect("test only uses stats with an aggregate fn")
    }

    #[rstest]
    #[case::all_within_bound(&["short", "shorter"], true, true)]
    #[case::only_min_truncated(&["short", "a value longer than the bound"], true, false)]
    #[case::both_truncated(&["zz value past the bound", "aa value past the bound"], false, false)]
    fn bounded_varlen_min_max_exact_iff_extremum_fits_bound(
        #[values(
            DType::Utf8(Nullability::NonNullable),
            DType::Binary(Nullability::NonNullable)
        )]
        dtype: DType,
        #[case] values: &[&str],
        #[case] max_exact: bool,
        #[case] min_exact: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let max_bytes = NonZeroUsize::new(8).vortex_expect("non-zero");
        let bounded = [
            BoundedMax.bind(BoundedMaxOptions { max_bytes }),
            BoundedMin.bind(BoundedMinOptions { max_bytes }),
        ];
        let mut acc = StatsAccumulator::new(&dtype, Some(&bounded));
        // One chunk per value, so exactness has to survive merging across chunks.
        for value in values {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                dtype.clone(),
                1,
                vortex_buffer::BufferAllocatorRef::statically_allocated(),
            );
            builder.append_value(value);
            acc.push_chunk(&builder.finish(), &mut ctx)?;
        }

        let stats = stats_set(&acc)?;
        assert_eq!(stats.get(Stat::Max).is_exact(), max_exact);
        assert_eq!(stats.get(Stat::Min).is_exact(), min_exact);
        Ok(())
    }

    #[test]
    fn default_pruning_aggregate_fns_uses_bounded_min_max_for_varlen() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let fns = default_pruning_aggregate_fns(&dtype);
        assert!(fns.iter().any(|f| f.is::<BoundedMax>()));
        assert!(fns.iter().any(|f| f.is::<BoundedMin>()));
        assert!(!fns.iter().any(|f| f.is::<Max>()));
        assert!(!fns.iter().any(|f| f.is::<Min>()));
    }

    #[test]
    fn default_pruning_aggregate_fns_uses_plain_min_max_for_fixed_width() {
        let fns = default_pruning_aggregate_fns(&i32_dtype());
        assert!(fns.iter().any(|f| f.is::<Max>()));
        assert!(fns.iter().any(|f| f.is::<Min>()));
        assert!(!fns.iter().any(|f| f.is::<BoundedMax>()));
        assert!(!fns.iter().any(|f| f.is::<BoundedMin>()));
    }

    #[test]
    fn explicit_min_max_on_varlen_is_exact_and_untruncated() -> VortexResult<()> {
        // An explicit override applies literally: unlike the default, plain `Max`/`Min` on a
        // varlen dtype now computes the exact value with no byte-bound truncation.
        let mut ctx = array_session().create_execution_ctx();
        let dtype = DType::Utf8(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("a long value that would have been truncated before");
        builder.append_value("short");

        let mut acc = StatsAccumulator::new(&dtype, Some(&[agg(Stat::Max), agg(Stat::Min)]));
        acc.push_chunk(&builder.finish(), &mut ctx)?;

        let stats = stats_set(&acc)?;
        assert!(matches!(stats.get(Stat::Max), Precision::Exact(_)));
        assert!(matches!(stats.get(Stat::Min), Precision::Exact(_)));
        Ok(())
    }

    #[test]
    fn aggregates_persist_across_multiple_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dtype = i32_dtype();
        let mut acc = StatsAccumulator::new(
            &dtype,
            Some(&[
                agg(Stat::Max),
                agg(Stat::Min),
                agg(Stat::Sum),
                agg(Stat::NullCount),
            ]),
        );

        acc.push_chunk(&buffer![0, 5, 2].into_array(), &mut ctx)?;
        acc.push_chunk(&buffer![7, 1, 3].into_array(), &mut ctx)?;
        acc.push_chunk(&buffer![-4, 9].into_array(), &mut ctx)?;

        let stats = stats_set(&acc)?;
        assert_eq!(
            stats.get(Stat::Max).as_exact(),
            Some(ScalarValue::from(9i32))
        );
        assert_eq!(
            stats.get(Stat::Min).as_exact(),
            Some(ScalarValue::from(-4i32))
        );
        assert_eq!(
            stats.get(Stat::Sum).as_exact(),
            Some(ScalarValue::from(23i64))
        );
        assert_eq!(
            stats.get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(0u64))
        );
        Ok(())
    }

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    #[test]
    fn postorder_layout_flat_struct() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("b", DType::Bool(Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );
        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (
                    FieldPath::from_name("b"),
                    DType::Bool(Nullability::Nullable)
                ),
            ]
        );
    }

    #[test]
    fn postorder_layout_nested_nullable_struct_trails_children() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a").push("b"), i32_dtype()),
                (FieldPath::from_name("a"), inner),
            ]
        );
    }

    #[test]
    fn postorder_layout_non_nullable_nested_struct_has_no_own_entry() {
        let inner = DType::struct_([("b", i32_dtype())], Nullability::NonNullable);
        let dtype = DType::struct_([("a", inner)], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![(FieldPath::from_name("a").push("b"), i32_dtype())]
        );
    }

    #[test]
    fn postorder_layout_nullable_root_struct_gets_trailing_root_entry() {
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::Nullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a"), i32_dtype()),
                (FieldPath::root(), dtype),
            ]
        );
    }

    #[test]
    fn postorder_layout_list_field_is_opaque_leaf() {
        let list_dtype = DType::list(i32_dtype(), Nullability::NonNullable);
        let dtype = DType::struct_([("a", list_dtype.clone())], Nullability::NonNullable);

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), list_dtype)]);
    }

    #[test]
    fn postorder_layout_variant_field_is_skipped() {
        let dtype = DType::struct_(
            [
                ("a", i32_dtype()),
                ("v", DType::Variant(Nullability::NonNullable)),
            ],
            Nullability::NonNullable,
        );

        let layout = postorder_stats_layout(&dtype);
        assert_eq!(layout, vec![(FieldPath::from_name("a"), i32_dtype())]);
    }

    #[test]
    fn postorder_layout_three_level_nesting_with_mixed_nullability() {
        // root (nullable) -> a (non-nullable) -> b (nullable) -> c (leaf). Exercises composition
        // across more than one level, including a non-nullable struct sandwiched between two
        // nullable ones, which should get no entry of its own.
        let b_dtype = DType::struct_([("c", i32_dtype())], Nullability::Nullable);
        let a_dtype = DType::struct_([("b", b_dtype.clone())], Nullability::NonNullable);
        let root_dtype = DType::struct_([("a", a_dtype)], Nullability::Nullable);

        let layout = postorder_stats_layout(&root_dtype);
        assert_eq!(
            layout,
            vec![
                (FieldPath::from_name("a").push("b").push("c"), i32_dtype()),
                (FieldPath::from_name("a").push("b"), b_dtype),
                (FieldPath::root(), root_dtype),
            ]
        );
    }

    #[test]
    fn three_level_nested_struct_accumulates_stats_at_each_level() -> VortexResult<()> {
        // Same shape as `postorder_layout_three_level_nesting_with_mixed_nullability`, but
        // exercises actual accumulation: each nullable level along the path should independently
        // contribute its own null-count entry, in post-order.
        let mut ctx = array_session().create_execution_ctx();

        let leaf_c = buffer![10i32, 20, 30].into_array();
        let b_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let struct_b =
            StructArray::new(FieldNames::from(["c"]), [leaf_c], 3, b_validity).into_array();
        let struct_a = StructArray::new(
            FieldNames::from(["b"]),
            [struct_b],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let root_validity = Validity::Array(BoolArray::from_iter([true, true, false]).into_array());
        let root =
            StructArray::new(FieldNames::from(["a"]), [struct_a], 3, root_validity).into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(root.dtype(), Some(&requested));
        node.push_chunk(&root, &mut ctx)?;

        let stats_sets = node_stats_sets(&node)?;

        // `a.b.c`'s own stats come first (post-order), then `a.b`'s null-count entry, then the
        // root's own null-count entry.
        assert_eq!(stats_sets.len(), 3);
        assert_eq!(
            stats_sets[0].get(Stat::Min).as_exact(),
            Some(ScalarValue::from(10i32))
        );
        assert_eq!(
            stats_sets[0].get(Stat::Max).as_exact(),
            Some(ScalarValue::from(30i32))
        );
        assert_eq!(
            stats_sets[0].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(0u64))
        );
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        assert_eq!(
            stats_sets[2].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        Ok(())
    }

    #[test]
    fn nested_nullable_struct_accumulates_its_own_null_count() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(outer.dtype(), Some(&requested));
        node.push_chunk(&outer, &mut ctx)?;

        let stats_sets = node_stats_sets(&node)?;

        // `a.b`'s stats come first (post-order), then `a`'s own null-count entry.
        assert_eq!(stats_sets.len(), 2);
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::Primitive(PValue::U64(1)))
        );
        Ok(())
    }

    #[test]
    fn legacy_aggregate_stats_cover_only_top_level_fields() -> VortexResult<()> {
        // The legacy accumulation must have exactly one entry per top-level field, built against
        // that field's own dtype without recursing into it, even though it's a nested struct.
        let session = array_session();
        let inner_dtype = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner_dtype)], Nullability::NonNullable);

        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([
                agg(Stat::NullCount),
                agg(Stat::Min),
                agg(Stat::Max),
            ])),
            &session,
            true,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let (mut ptr, _eof) = SequenceId::root().split();
        acc.process(Ok((ptr.advance(), outer)))?;

        let legacy = acc.legacy_aggregate_stats()?;
        assert_eq!(legacy.len(), 1);
        assert_eq!(
            legacy[0].get(&agg(Stat::NullCount)),
            Precision::exact(Scalar::from(1u64))
        );
        assert!(legacy[0].get(&agg(Stat::Min)).is_absent());
        assert!(legacy[0].get(&agg(Stat::Max)).is_absent());
        Ok(())
    }

    #[rstest]
    #[case(DType::struct_([("a", i32_dtype())], Nullability::NonNullable), StructArray::new(FieldNames::from(["a"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::NonNullable).into_array())]
    #[case(i32_dtype(), buffer![1i32, 2, 3].into_array())]
    fn write_legacy_stats_false_skips_legacy_accumulation(
        #[case] dtype: DType,
        #[case] chunk: ArrayRef,
    ) -> VortexResult<()> {
        // With `write_legacy_stats: false`, `legacy` stays empty for both struct and non-struct
        // roots. This must not panic — `process` indexes into `legacy[0]` for a non-struct root
        // and `zip_eq`s it against the struct's fields otherwise, both of which would panic if
        // `legacy` were left empty without also gating those code paths.
        let session = array_session();
        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::Min), agg(Stat::Max)])),
            &session,
            false,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        let (mut ptr, _eof) = SequenceId::root().split();
        acc.process(Ok((ptr.advance(), chunk)))?;

        assert!(acc.legacy_aggregate_stats()?.is_empty());
        Ok(())
    }

    fn utf8_chunk(values: &[&str]) -> ArrayRef {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype,
            values.len(),
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        for value in values {
            builder.append_value(value);
        }
        builder.finish()
    }

    #[rstest]
    #[case::untruncated(&["short", "shorter"], true)]
    #[case::truncated(&["a value longer than the bound"], false)]
    fn exact_bounded_max_is_recorded_as_max(
        #[case] values: &[&str],
        #[case] exact: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let max_bytes = NonZeroUsize::new(8).vortex_expect("non-zero");
        let mut acc = StatsAccumulator::new(
            &DType::Utf8(Nullability::NonNullable),
            Some(&[BoundedMax.bind(BoundedMaxOptions { max_bytes })]),
        );
        acc.push_chunk(&utf8_chunk(values), &mut ctx)?;

        let aggregates = acc.aggregate_stats(|_| true)?;
        let [recorded] = aggregates.iter().collect::<Vec<_>>()[..] else {
            vortex_panic!("expected one recorded aggregate");
        };
        assert_eq!(recorded.aggregate_fn().is::<Max>(), exact);
        assert_eq!(recorded.aggregate_fn().is::<BoundedMax>(), !exact);
        // The stored partial must rebuild the same value on read.
        let partial = recorded
            .partial()
            .vortex_expect("accumulated aggregates have a partial");
        let rebuilt = AggregateStat::try_from_partial(
            recorded.aggregate_fn().clone(),
            partial.clone(),
            &DType::Utf8(Nullability::NonNullable),
        )?;
        assert_eq!(rebuilt.value(), recorded.value());
        Ok(())
    }

    #[test]
    fn exact_bounded_max_stays_bounded_when_max_is_not_allowed() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let max_bytes = NonZeroUsize::new(8).vortex_expect("non-zero");
        let mut acc = StatsAccumulator::new(
            &DType::Utf8(Nullability::NonNullable),
            Some(&[BoundedMax.bind(BoundedMaxOptions { max_bytes })]),
        );
        acc.push_chunk(&utf8_chunk(&["short"]), &mut ctx)?;

        let aggregates = acc.aggregate_stats(|aggregate_fn| !aggregate_fn.is::<Max>())?;
        assert!(
            aggregates
                .iter()
                .all(|stat| stat.aggregate_fn().is::<BoundedMax>())
        );
        Ok(())
    }

    #[test]
    fn get_prefers_an_exact_aggregate_and_skips_nulls() -> VortexResult<()> {
        let max_bytes = NonZeroUsize::new(8).vortex_expect("non-zero");
        let requested = Max.bind(NumericalAggregateOpts::skip_nans());
        let bound = Scalar::utf8("b", Nullability::Nullable);
        let max = Scalar::utf8("a", Nullability::Nullable);
        let bounded = AggregateStat::from_value(
            BoundedMax.bind(BoundedMaxOptions { max_bytes }),
            Precision::exact(bound.clone()),
        );

        let only_bounded = AggregateStats::new(vec![bounded.clone()]);
        assert_eq!(only_bounded.get(&requested), Precision::inexact(bound));

        let both = AggregateStats::new(vec![
            bounded,
            AggregateStat::from_value(requested.clone(), Precision::exact(max.clone())),
        ]);
        assert_eq!(both.get(&requested), Precision::exact(max));

        let null = AggregateStats::new(vec![AggregateStat::from_value(
            requested.clone(),
            Precision::exact(Scalar::null(DType::Utf8(Nullability::Nullable))),
        )]);
        assert!(null.get(&requested).is_absent());
        Ok(())
    }

    #[test]
    fn forbidden_aggregates_are_gated() -> VortexResult<()> {
        let session = array_session();
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::NonNullable);
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_allowed_aggregates(
            HashSet::from_iter([Min.id(), Max.id(), NanCount.id(), NullCount.id()]),
        );

        // An explicitly requested aggregate the context forbids fails the write.
        let error = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::Sum)])),
            &session,
            true,
            &ctx,
        )
        .err()
        .vortex_expect("Sum is not permitted");
        assert!(error.to_string().contains("not permitted by ctx"));

        // The defaults leave it out of the nested aggregates, but the legacy stats keep it.
        let acc = FileStatsAccumulator::try_new(&dtype, None, &session, true, &ctx)?;
        let chunk = StructArray::new(
            FieldNames::from(["a"]),
            [buffer![1i32, 2, 3].into_array()],
            3,
            Validity::NonNullable,
        )
        .into_array();
        let (mut ptr, _eof) = SequenceId::root().split();
        acc.process(Ok((ptr.advance(), chunk)))?;

        let aggregates = acc.aggregate_stats()?;
        assert!(
            aggregates[0]
                .iter()
                .any(|stat| stat.aggregate_fn().is::<Max>())
        );
        assert!(
            !aggregates[0]
                .iter()
                .any(|stat| stat.aggregate_fn().is::<Sum>())
        );
        assert_eq!(
            acc.legacy_aggregate_stats()?[0]
                .get(&agg(Stat::Sum))
                .and_then(Scalar::into_value),
            Precision::exact(ScalarValue::from(6i64))
        );
        Ok(())
    }
}
