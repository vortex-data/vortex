// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::aggregate_fn::demand::struct_field_demand;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
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
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutWriterContext;
use crate::layouts::zoned::default_bounded_stat_max_bytes;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Accumulates file statistics over `stream` as it passes through.
///
/// `stats` are the aggregates to record for every field, or `None` for each field's default (see
/// `default_aggregate_fns`). Only aggregates `ctx` allows are recorded in the nested
/// statistics: an explicitly requested aggregate it forbids fails, like it does for zone maps,
/// while forbidden defaults are left out.
///
/// `write_nested_stats` and `write_legacy_stats` select which footer layouts are recorded; see
/// [`FileStatsAccumulator`].
///
/// # Errors
///
/// Returns an error if nested stats are written and `stats` contains an aggregate `ctx` forbids.
pub fn accumulate_stats(
    stream: SendableSequentialStream,
    stats: Option<Arc<[AggregateFnRef]>>,
    session: &VortexSession,
    write_nested_stats: bool,
    write_legacy_stats: bool,
    ctx: &LayoutWriterContext,
) -> VortexResult<(FileStatsAccumulator, SendableSequentialStream)> {
    let accumulator = FileStatsAccumulator::try_new(
        stream.dtype(),
        stats,
        write_nested_stats,
        write_legacy_stats,
        ctx,
    )?;
    let dtype = stream.dtype().clone();

    // Each chunk's stats are computed into their own accumulators on a CPU task, so chunks are
    // processed in parallel instead of serially in front of the rest of the write pipeline.
    let chunk_accumulator = accumulator.clone();
    let handle = session.handle();

    // Every task clones one context, which shares its kernel snapshot instead of taking a new
    // snapshot of the session for each chunk.
    let exec_ctx = session.create_execution_ctx();
    let concurrency = get_available_parallelism().unwrap_or(1);
    let stream = stream
        .map(move |item| {
            let accumulator = chunk_accumulator.clone();
            let mut exec_ctx = exec_ctx.clone();
            handle.spawn_cpu(move || {
                let (sequence_id, chunk, demand) = item?;
                let stats = accumulator.chunk_stats(&chunk, &demand, &mut exec_ctx)?;
                Ok::<_, VortexError>((sequence_id, chunk, demand, stats))
            })
        })
        .buffered(concurrency);

    // Merge the per-chunk stats in stream order, which keeps order-dependent aggregates correct.
    let merge_accumulator = accumulator.clone();
    let stream = SequentialStreamAdapter::new(
        dtype,
        stream.map(move |item| {
            let (sequence_id, chunk, demand, stats) = item?;
            merge_accumulator.merge(stats)?;
            Ok((sequence_id, chunk, demand))
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
        accumulator.combine_partials(&partial)?;
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
    /// An accumulator that records nothing.
    fn empty() -> Self {
        Self {
            aggregates: Vec::new(),
        }
    }

    /// Builds the accumulators for `stats`, or for the dtype's defaults when `None`, skipping any
    /// aggregate `allowed` rejects so that it is not computed only to be dropped.
    fn new(
        dtype: &DType,
        stats: Option<&[AggregateFnRef]>,
        allowed: &dyn Fn(&AggregateFnRef) -> bool,
    ) -> Self {
        if !supports_file_stats(dtype) {
            return Self::empty();
        }

        // When no explicit list is requested, each leaf picks its own default the way
        // `default_zoned_aggregate_fns` does for zoned layouts: a byte-bounded min/max for
        // variable-length columns, exact min/max otherwise.
        let default_stats;
        let stats = match stats {
            Some(stats) => stats,
            None => {
                default_stats = default_aggregate_fns(dtype);
                default_stats.as_slice()
            }
        };

        let mut aggregates = Vec::new();
        for aggregate_fn in stats.iter().filter(|aggregate_fn| allowed(aggregate_fn)) {
            // A dtype that doesn't support a given aggregate is silently skipped, matching this
            // stat's absence from the result. Check the dtypes first: a failed `accumulator` call
            // builds an error, and an error captures a backtrace, which is expensive for wide
            // schemas.
            if aggregate_fn.return_dtype(dtype).is_none()
                || aggregate_fn.state_dtype(dtype).is_none()
            {
                continue;
            }

            if let Ok(accumulator) = aggregate_fn.accumulator(dtype) {
                aggregates.push((aggregate_fn.clone(), accumulator));
            }
        }

        Self { aggregates }
    }

    /// Accumulate `array` as null wherever `demand` is false.
    fn push_chunk(
        &mut self,
        array: &ArrayRef,
        demand: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for (_, accumulator) in &mut self.aggregates {
            accumulator.accumulate_demanded(array, demand, ctx)?;
        }
        Ok(())
    }

    /// Drains the state of `other`, built for the same dtype and aggregates, into `self`.
    fn merge_from(&mut self, other: &mut Self) -> VortexResult<()> {
        for ((_, accumulator), (_, other)) in self
            .aggregates
            .iter_mut()
            .zip_eq(other.aggregates.iter_mut())
        {
            accumulator.merge_from(other.as_mut())?;
        }
        Ok(())
    }

    /// Returns the accumulated aggregates that `allowed` permits.
    fn aggregate_stats(
        &self,
        allowed: impl Fn(&AggregateFnRef) -> bool,
    ) -> VortexResult<AggregateStats> {
        self.aggregates
            .iter()
            .filter(|(aggregate_fn, _)| allowed(aggregate_fn))
            .map(|(aggregate_fn, accumulator)| {
                AggregateStat::from_accumulator(aggregate_fn.clone(), accumulator)
            })
            .collect::<VortexResult<_>>()
            .map(AggregateStats::new)
    }
}

fn supports_file_stats(dtype: &DType) -> bool {
    !matches!(dtype, DType::Variant(_))
}

fn is_varlen_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
}

/// Default file-level aggregates for `dtype`. The pruning aggregates are chosen the way
/// `default_zoned_aggregate_fns` picks zoned aggregates: a byte-bounded min/max (capped at
/// [`default_bounded_stat_max_bytes`]) for variable-length columns, and exact min/max otherwise.
/// Callers wanting a different bound pass their own `BoundedMax`/`BoundedMin` instead.
fn default_aggregate_fns(dtype: &DType) -> Vec<AggregateFnRef> {
    let (max, min) = if is_varlen_dtype(dtype) {
        // These read back as inexact even when nothing was truncated, until a V2 of the bounded
        // aggregates records that in its partial state.
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

    // `Sum` is left out until the footer records the overflow-aware `SumV2`.
    vec![
        max,
        min,
        NullCount.bind(EmptyOptions),
        NanCount.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
    ]
}

/// Computes the post-order sequence of `(FieldPath, DType)` entries that file-level statistics
/// are stored against.
///
/// Each leaf field (including opaque `List`/`FixedSizeList` columns, which are not recursed into)
/// gets one entry. Each **nullable** struct additionally gets a trailing entry, keyed by its own
/// path and dtype, inserted immediately after its children's entries — this carries the struct's
/// own null count. Dtypes that don't support file stats (see `supports_file_stats`), such as
/// [`DType::Variant`], are skipped entirely: no entry is emitted for them or anything beneath
/// them.
///
/// Entries beneath a nullable struct describe the logical column, which is null wherever any
/// ancestor struct is null, whatever the unmasked child holds there.
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
    /// An opaque leaf: a dtype with no addressable children. Currently includes List/FixedSizeList and Map.
    Leaf(StatsAccumulator),
    /// A dtype that does not support file stats (e.g. [`DType::Variant`]); contributes no entries.
    Skipped,
    /// Stats accumulated only for the legacy top-level-only layout; contributes no nested entries.
    ///
    /// Used for the own stats of a non-nullable top-level struct field, which the nested layout
    /// doesn't record but the legacy layout does.
    LegacyOnly(StatsAccumulator),
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
    /// Builds the root node for a file of `dtype`.
    fn build_root(
        dtype: &DType,
        stats: Option<&[AggregateFnRef]>,
        allowed: &dyn Fn(&AggregateFnRef) -> bool,
        write_nested_stats: bool,
        write_legacy_stats: bool,
    ) -> Self {
        if !write_nested_stats {
            return Self::build_legacy_root(dtype, stats, allowed);
        }

        let mut root = Self::build(dtype, stats, allowed);

        // The legacy layout records each top-level field's stats over its own dtype. The nested
        // layout already accumulates those for every top-level field except a non-nullable struct,
        // whose own stats are only accumulated for the legacy layout.
        if write_legacy_stats
            && let (Self::Container { children, .. }, Some(struct_fields)) =
                (&mut root, dtype.as_struct_fields_opt())
        {
            for ((_, child), field_dtype) in children.iter_mut().zip_eq(struct_fields.fields()) {
                if let Self::Container { own, .. } = child
                    && matches!(**own, Self::Skipped)
                {
                    **own = Self::LegacyOnly(StatsAccumulator::new(&field_dtype, stats, allowed));
                }
            }
        }

        root
    }

    /// Builds a root node that only accumulates the legacy layout's stats: each top-level field's
    /// stats over its own dtype, or the whole file's for a non-struct root. It contributes no
    /// nested entries.
    fn build_legacy_root(
        dtype: &DType,
        stats: Option<&[AggregateFnRef]>,
        allowed: &dyn Fn(&AggregateFnRef) -> bool,
    ) -> Self {
        let Some(struct_fields) = dtype.as_struct_fields_opt() else {
            return Self::LegacyOnly(StatsAccumulator::new(dtype, stats, allowed));
        };
        // The fields of a nullable root are pushed with the root's validity as their demand, so
        // they are recorded as their logical columns, as in `build`.
        let children = struct_fields
            .names()
            .iter()
            .zip(struct_fields.fields())
            .map(|(name, field_dtype)| {
                (
                    Field::Name(name.clone()),
                    Self::LegacyOnly(StatsAccumulator::new(&field_dtype, stats, allowed)),
                )
            })
            .collect();
        Self::Container {
            children,
            own: Box::new(Self::Skipped),
        }
    }

    fn build(
        dtype: &DType,
        stats: Option<&[AggregateFnRef]>,
        allowed: &dyn Fn(&AggregateFnRef) -> bool,
    ) -> Self {
        match dtype.as_struct_fields_opt() {
            Some(struct_fields) => {
                let nullable = dtype.nullability() == Nullability::Nullable;
                let children = struct_fields
                    .names()
                    .iter()
                    .zip(struct_fields.fields())
                    .map(|(name, field_dtype)| {
                        (
                            Field::Name(name.clone()),
                            Self::build(&field_dtype, stats, allowed),
                        )
                    })
                    .collect();
                let own = if nullable {
                    Self::Leaf(StatsAccumulator::new(dtype, stats, allowed))
                } else {
                    Self::Skipped
                };
                Self::Container {
                    children,
                    own: Box::new(own),
                }
            }
            None if !supports_file_stats(dtype) => Self::Skipped,
            None => Self::Leaf(StatsAccumulator::new(dtype, stats, allowed)),
        }
    }

    /// Pushes a chunk of this node's dtype into its accumulators.
    ///
    /// `Container` is only ever built for `DType::Struct` today, so every child is addressed by
    /// [`Field::Name`] and extracted via [`StructArrayExt::iter_unmasked_fields`]. A future `List`/
    /// `Map` container would need a different extraction here (e.g. flattened elements, or
    /// derived per-row shape arrays), dispatched per child's [`Field`] kind.
    ///
    /// The chunk is null wherever `demand` is false, i.e. wherever an ancestor struct is null.
    fn push_chunk(
        &mut self,
        array: &ArrayRef,
        demand: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match self {
            Self::Skipped => Ok(()),
            Self::Leaf(acc) | Self::LegacyOnly(acc) => acc.push_chunk(array, demand, ctx),
            Self::Container { children, own } => {
                // The container's own `ArrayRef` already carries the validity needed to compute
                // its own stats (e.g. null count), so we push it directly rather than building a
                // synthetic array.
                own.push_chunk(array, demand, ctx)?;
                let struct_array = array.clone().execute::<StructArray>(ctx)?;
                // Fields are stored unmasked, so they are pushed with the struct's validity merged
                // into the demand: each field is null wherever the struct or an ancestor is null.
                let field_demand = struct_field_demand(&struct_array, demand, ctx)?;
                for ((_, child), field) in children
                    .iter_mut()
                    .zip_eq(struct_array.iter_unmasked_fields())
                {
                    child.push_chunk(field, &field_demand, ctx)?;
                }
                Ok(())
            }
        }
    }

    /// Drains the state of `other`, a node built for the same dtype and aggregates, into `self`.
    fn merge_from(&mut self, other: &mut Self) -> VortexResult<()> {
        match (self, other) {
            (Self::Skipped, Self::Skipped) => Ok(()),
            (Self::Leaf(acc), Self::Leaf(other))
            | (Self::LegacyOnly(acc), Self::LegacyOnly(other)) => acc.merge_from(other),
            (
                Self::Container { children, own },
                Self::Container {
                    children: other_children,
                    own: other_own,
                },
            ) => {
                own.merge_from(other_own)?;
                for ((_, child), (_, other)) in
                    children.iter_mut().zip_eq(other_children.iter_mut())
                {
                    child.merge_from(other)?;
                }
                Ok(())
            }
            _ => vortex_bail!("Cannot merge file stats nodes of different shapes"),
        }
    }

    /// Appends this node's aggregates that `allowed` permits, in the same post-order as
    /// [`postorder_stats_layout`].
    fn collect_aggregate_stats(
        &self,
        allowed: &impl Fn(&AggregateFnRef) -> bool,
        out: &mut Vec<AggregateStats>,
    ) -> VortexResult<()> {
        match self {
            Self::Skipped | Self::LegacyOnly(_) => Ok(()),
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

    /// Returns this node's stats over its own, undecomposed dtype, as the legacy layout records
    /// them for a top-level field, leaving out the aggregates `allowed` rejects.
    fn legacy_aggregate_stats(
        &self,
        allowed: impl Fn(&AggregateFnRef) -> bool,
    ) -> VortexResult<AggregateStats> {
        match self {
            Self::Skipped => Ok(AggregateStats::default()),
            Self::Leaf(acc) | Self::LegacyOnly(acc) => acc.aggregate_stats(allowed),
            Self::Container { own, .. } => own.legacy_aggregate_stats(allowed),
        }
    }
}

/// An array stream processor that computes aggregate statistics for every field, recursing into
/// nested (possibly nullable) structs. See [`postorder_stats_layout`] for the entry ordering.
///
/// The same accumulators also produce the legacy layout covering only the top-level struct fields
/// (or the whole array, for a non-struct root), matching the layout file stats had before nested
/// struct support: one entry per top-level field, with no recursion into nested structs. When
/// `write_legacy_stats` is `false`, the legacy layout keeps that shape but records nothing. When
/// `write_nested_stats` is `false`, only the legacy layout's stats are accumulated and the nested
/// layout is not produced.
#[derive(Clone)]
pub struct FileStatsAccumulator {
    root: Arc<Mutex<StatsNode>>,
    dtype: DType,
    stats: Option<Arc<[AggregateFnRef]>>,
    write_nested_stats: bool,
    write_legacy_stats: bool,
    writer_ctx: LayoutWriterContext,
}

impl FileStatsAccumulator {
    fn try_new(
        dtype: &DType,
        stats: Option<Arc<[AggregateFnRef]>>,
        write_nested_stats: bool,
        write_legacy_stats: bool,
        writer_ctx: &LayoutWriterContext,
    ) -> VortexResult<Self> {
        for aggregate_fn in stats.iter().flat_map(|stats| stats.iter()) {
            if !writer_ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }

        let root = StatsNode::build_root(
            dtype,
            stats.as_deref(),
            &|aggregate_fn| writer_ctx.allows_aggregate(&aggregate_fn.id()),
            write_nested_stats,
            write_legacy_stats,
        );

        Ok(Self {
            root: Arc::new(Mutex::new(root)),
            dtype: dtype.clone(),
            stats,
            write_nested_stats,
            write_legacy_stats,
            writer_ctx: writer_ctx.clone(),
        })
    }

    /// Computes the stats of one chunk into a new tree of accumulators, to [merge](Self::merge)
    /// later.
    fn chunk_stats(
        &self,
        chunk: &ArrayRef,
        demand: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<StatsNode> {
        let mut node = StatsNode::build_root(
            &self.dtype,
            self.stats.as_deref(),
            &|aggregate_fn| self.writer_ctx.allows_aggregate(&aggregate_fn.id()),
            self.write_nested_stats,
            self.write_legacy_stats,
        );
        node.push_chunk(chunk, demand, ctx)?;
        Ok(node)
    }

    /// Merges the stats of the chunk that follows all chunks merged so far.
    fn merge(&self, mut chunk_stats: StatsNode) -> VortexResult<()> {
        self.root.lock().merge_from(&mut chunk_stats)
    }

    /// Returns the accumulated aggregates of every entry in the post-order nested layout (see
    /// [`postorder_stats_layout`]), leaving out the aggregates the writer context forbids.
    ///
    /// # Errors
    ///
    /// Returns an error if the accumulator was built without nested stats.
    pub fn aggregate_stats(&self) -> VortexResult<Vec<AggregateStats>> {
        if !self.write_nested_stats {
            vortex_bail!("File stats were accumulated without nested stats");
        }
        let allowed =
            |aggregate_fn: &AggregateFnRef| self.writer_ctx.allows_aggregate(&aggregate_fn.id());
        let mut out = Vec::new();
        self.root
            .lock()
            .collect_aggregate_stats(&allowed, &mut out)?;
        Ok(out)
    }

    /// Returns the legacy top-level-fields-only aggregates (one per top-level struct field, or a
    /// single entry for a non-struct root dtype), leaving out the aggregates the writer context
    /// forbids. Each entry is empty if `write_legacy_stats` was `false`.
    pub fn legacy_aggregate_stats(&self) -> VortexResult<Vec<AggregateStats>> {
        let allowed =
            |aggregate_fn: &AggregateFnRef| self.writer_ctx.allows_aggregate(&aggregate_fn.id());
        let root = self.root.lock();
        // Readers that predate nested stats reject a file whose legacy layout doesn't have one
        // entry per top-level field, so entries that aren't recorded are still written, empty.
        let fields = match &*root {
            StatsNode::Container { children, .. } => {
                children.iter().map(|(_, child)| child).collect()
            }
            node => vec![node],
        };
        fields
            .into_iter()
            .map(|node| {
                if self.write_legacy_stats {
                    node.legacy_aggregate_stats(allowed)
                } else {
                    Ok(AggregateStats::default())
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_panic;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;

    /// A stats node or accumulator a chunk can be pushed into with every row demanded.
    trait PushAll {
        fn push_all(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()>;
    }

    impl PushAll for StatsAccumulator {
        fn push_all(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
            self.push_chunk(array, &Mask::new_true(array.len()), ctx)
        }
    }

    impl PushAll for StatsNode {
        fn push_all(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
            self.push_chunk(array, &Mask::new_true(array.len()), ctx)
        }
    }

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Pushes `chunk` through `acc` the way the write stream does.
    fn push(
        acc: &FileStatsAccumulator,
        chunk: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        acc.merge(acc.chunk_stats(chunk, &Mask::new_true(chunk.len()), ctx)?)
    }

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

    /// The footer can't tell an untruncated bound from a truncated one, so bounded min/max are
    /// inexact even when every value fits the bound.
    #[rstest]
    #[case::all_within_bound(&["short", "shorter"])]
    #[case::only_min_truncated(&["short", "a value longer than the bound"])]
    #[case::both_truncated(&["zz value past the bound", "aa value past the bound"])]
    fn bounded_varlen_min_max_are_inexact(
        #[values(
            DType::Utf8(Nullability::NonNullable),
            DType::Binary(Nullability::NonNullable)
        )]
        dtype: DType,
        #[case] values: &[&str],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let max_bytes = NonZeroUsize::new(8).vortex_expect("non-zero");
        let bounded = [
            BoundedMax.bind(BoundedMaxOptions { max_bytes }),
            BoundedMin.bind(BoundedMinOptions { max_bytes }),
        ];
        let mut acc = StatsAccumulator::new(&dtype, Some(&bounded), &|_| true);
        for value in values {
            let mut builder = VarBinViewBuilder::with_capacity_in(
                dtype.clone(),
                1,
                vortex_buffer::BufferAllocatorRef::statically_allocated(),
            );
            builder.append_value(value);
            acc.push_all(&builder.finish(), &mut ctx)?;
        }

        let stats = stats_set(&acc)?;
        assert!(matches!(stats.get(Stat::Max), Precision::Inexact(_)));
        assert!(matches!(stats.get(Stat::Min), Precision::Inexact(_)));
        Ok(())
    }

    #[test]
    fn default_aggregate_fns_uses_bounded_min_max_for_varlen() {
        let dtype = DType::Utf8(Nullability::NonNullable);
        let fns = default_aggregate_fns(&dtype);
        assert!(fns.iter().any(|f| f.is::<BoundedMax>()));
        assert!(fns.iter().any(|f| f.is::<BoundedMin>()));
        assert!(!fns.iter().any(|f| f.is::<Max>()));
        assert!(!fns.iter().any(|f| f.is::<Min>()));
    }

    #[test]
    fn default_aggregate_fns_uses_plain_min_max_for_fixed_width() {
        let fns = default_aggregate_fns(&i32_dtype());
        assert!(fns.iter().any(|f| f.is::<Max>()));
        assert!(fns.iter().any(|f| f.is::<Min>()));
        assert!(!fns.iter().any(|f| f.is::<BoundedMax>()));
        assert!(!fns.iter().any(|f| f.is::<BoundedMin>()));
    }

    #[test]
    fn disallowed_default_aggregates_are_not_accumulated() {
        let acc =
            StatsAccumulator::new(&i32_dtype(), None, &|f| !f.is::<UncompressedSizeInBytes>());
        assert!(
            !acc.aggregates
                .iter()
                .any(|(f, _)| f.is::<UncompressedSizeInBytes>())
        );
        assert!(acc.aggregates.iter().any(|(f, _)| f.is::<Max>()));
    }

    #[test]
    fn explicit_min_max_on_varlen_is_exact_and_untruncated() -> VortexResult<()> {
        // An explicit override applies literally: unlike the default, plain `Max`/`Min` on a
        // varlen dtype now computes the exact value with no byte-bound truncation.
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::NonNullable);
        let mut builder = VarBinViewBuilder::with_capacity_in(
            dtype.clone(),
            2,
            vortex_buffer::BufferAllocatorRef::statically_allocated(),
        );
        builder.append_value("a long value that would have been truncated before");
        builder.append_value("short");

        let mut acc =
            StatsAccumulator::new(&dtype, Some(&[agg(Stat::Max), agg(Stat::Min)]), &|_| true);
        acc.push_all(&builder.finish(), &mut ctx)?;

        let stats = stats_set(&acc)?;
        assert!(matches!(stats.get(Stat::Max), Precision::Exact(_)));
        assert!(matches!(stats.get(Stat::Min), Precision::Exact(_)));
        Ok(())
    }

    #[test]
    fn aggregates_persist_across_multiple_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = i32_dtype();
        let mut acc = StatsAccumulator::new(
            &dtype,
            Some(&[
                agg(Stat::Max),
                agg(Stat::Min),
                agg(Stat::Sum),
                agg(Stat::NullCount),
            ]),
            &|_| true,
        );

        acc.push_all(&buffer![0, 5, 2].into_array(), &mut ctx)?;
        acc.push_all(&buffer![7, 1, 3].into_array(), &mut ctx)?;
        acc.push_all(&buffer![-4, 9].into_array(), &mut ctx)?;

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
    fn three_level_nested_struct_records_entries_beneath_nullable_ancestors() -> VortexResult<()> {
        // Same shape as `postorder_layout_three_level_nesting_with_mixed_nullability`, but
        // exercises actual accumulation. Every entry describes its logical column, which is null
        // wherever any ancestor struct is null.
        let mut ctx = SESSION.create_execution_ctx();

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
        let mut node = StatsNode::build(root.dtype(), Some(&requested), &|_| true);
        node.push_all(&root, &mut ctx)?;

        let stats_sets = node_stats_sets(&node)?;

        // Post-order: `a.b.c`, then `a.b`'s null-count entry, then the root's.
        assert_eq!(stats_sets.len(), 3);
        // `a.b.c` is valid only where both the root and `a.b` are, so 20 and 30 are left out.
        assert_eq!(
            stats_sets[0].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(2u64))
        );
        assert_eq!(
            stats_sets[0].get(Stat::Min).as_exact(),
            Some(ScalarValue::from(10i32))
        );
        assert_eq!(
            stats_sets[0].get(Stat::Max).as_exact(),
            Some(ScalarValue::from(10i32))
        );
        // `a.b` is null in its own null row and in the root's.
        assert_eq!(
            stats_sets[1].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(2u64))
        );
        assert_eq!(
            stats_sets[2].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        Ok(())
    }

    #[test]
    fn children_of_nullable_struct_are_null_under_its_null_rows() -> VortexResult<()> {
        // https://github.com/vortex-data/vortex/issues/9205: `s.a` holds 999 in the row where `s`
        // is null, which its stats leave out and count as null.
        let mut ctx = SESSION.create_execution_ctx();

        let a = buffer![1i32, 999, 3].into_array();
        let s_validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let s = StructArray::new(FieldNames::from(["a"]), [a], 3, s_validity).into_array();
        let root =
            StructArray::new(FieldNames::from(["s"]), [s], 3, Validity::NonNullable).into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(root.dtype(), Some(&requested), &|_| true);
        node.push_all(&root, &mut ctx)?;

        let mut aggregates = Vec::new();
        node.collect_aggregate_stats(&|_| true, &mut aggregates)?;

        // `s.a`, then `s`'s own null-count entry.
        assert_eq!(aggregates.len(), 2);
        assert_eq!(
            aggregates[0].get(&agg(Stat::NullCount)),
            Precision::exact(Scalar::from(1u64))
        );
        assert_eq!(
            aggregates[0].get(&agg(Stat::Min)),
            Precision::exact(Scalar::from(1i32))
        );
        assert_eq!(
            aggregates[0].get(&agg(Stat::Max)),
            Precision::exact(Scalar::from(3i32))
        );
        assert_eq!(
            aggregates[1].get(&agg(Stat::NullCount)),
            Precision::exact(Scalar::from(1u64))
        );
        Ok(())
    }

    #[test]
    fn nullable_children_of_non_nullable_struct_are_recorded() -> VortexResult<()> {
        // `s` is non-nullable, so `s.a`'s own validity is its full logical validity.
        let mut ctx = SESSION.create_execution_ctx();

        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let s =
            StructArray::new(FieldNames::from(["a"]), [a], 3, Validity::NonNullable).into_array();
        let root =
            StructArray::new(FieldNames::from(["s"]), [s], 3, Validity::NonNullable).into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Max)];
        let mut node = StatsNode::build(root.dtype(), Some(&requested), &|_| true);
        node.push_all(&root, &mut ctx)?;

        let stats_sets = node_stats_sets(&node)?;
        assert_eq!(stats_sets.len(), 1);
        assert_eq!(
            stats_sets[0].get(Stat::NullCount).as_exact(),
            Some(ScalarValue::from(1u64))
        );
        assert_eq!(
            stats_sets[0].get(Stat::Max).as_exact(),
            Some(ScalarValue::from(3i32))
        );
        Ok(())
    }

    #[test]
    fn nested_nullable_struct_accumulates_its_own_null_count() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        let requested = [agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)];
        let mut node = StatsNode::build(outer.dtype(), Some(&requested), &|_| true);
        node.push_all(&outer, &mut ctx)?;

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
        let mut exec_ctx = SESSION.create_execution_ctx();
        let inner_dtype = DType::struct_([("b", i32_dtype())], Nullability::Nullable);
        let dtype = DType::struct_([("a", inner_dtype)], Nullability::NonNullable);

        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([
                agg(Stat::NullCount),
                agg(Stat::Min),
                agg(Stat::Max),
            ])),
            true,
            true,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        let b = buffer![1i32, 2, 3].into_array();
        let inner_validity =
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let inner = StructArray::new(FieldNames::from(["b"]), [b], 3, inner_validity).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        push(&acc, &outer, &mut exec_ctx)?;

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

    #[test]
    fn legacy_aggregate_stats_record_non_nullable_struct_field() -> VortexResult<()> {
        // The nested layout has no entry for a non-nullable struct's own stats, but the legacy
        // layout still records them for a top-level field.
        let mut exec_ctx = SESSION.create_execution_ctx();
        let inner_dtype = DType::struct_([("b", i32_dtype())], Nullability::NonNullable);
        let dtype = DType::struct_([("a", inner_dtype)], Nullability::NonNullable);

        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::NullCount)])),
            true,
            true,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        let b = buffer![1i32, 2, 3].into_array();
        let inner =
            StructArray::new(FieldNames::from(["b"]), [b], 3, Validity::NonNullable).into_array();
        let outer = StructArray::new(FieldNames::from(["a"]), [inner], 3, Validity::NonNullable)
            .into_array();

        push(&acc, &outer, &mut exec_ctx)?;

        let legacy = acc.legacy_aggregate_stats()?;
        assert_eq!(legacy.len(), 1);
        assert_eq!(
            legacy[0].get(&agg(Stat::NullCount)),
            Precision::exact(Scalar::from(0u64))
        );
        // Only `a.b` has a nested entry.
        assert_eq!(acc.aggregate_stats()?.len(), 1);
        Ok(())
    }

    #[test]
    fn legacy_aggregate_stats_of_nullable_root_are_null_under_its_null_rows() -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::Nullable);

        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::NullCount), agg(Stat::Max)])),
            true,
            true,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        let a = buffer![1i32, 999, 3].into_array();
        let validity = Validity::Array(BoolArray::from_iter([true, false, true]).into_array());
        let root = StructArray::new(FieldNames::from(["a"]), [a], 3, validity).into_array();

        push(&acc, &root, &mut exec_ctx)?;

        let legacy = acc.legacy_aggregate_stats()?;
        assert_eq!(legacy.len(), 1);
        // `a` holds 999 in the row the root nulls out, which counts as null and is left out.
        assert_eq!(
            legacy[0].get(&agg(Stat::NullCount)),
            Precision::exact(Scalar::from(1u64))
        );
        assert_eq!(
            legacy[0].get(&agg(Stat::Max)),
            Precision::exact(Scalar::from(3i32))
        );
        Ok(())
    }

    #[rstest]
    #[case(DType::struct_([("a", i32_dtype()), ("b", i32_dtype())], Nullability::NonNullable), StructArray::new(FieldNames::from(["a", "b"]), [buffer![1i32, 2, 3].into_array(), buffer![4i32, 5, 6].into_array()], 3, Validity::NonNullable).into_array(), 2)]
    #[case(i32_dtype(), buffer![1i32, 2, 3].into_array(), 1)]
    fn write_legacy_stats_false_keeps_one_empty_entry_per_top_level_field(
        #[case] dtype: DType,
        #[case] chunk: ArrayRef,
        #[case] top_level_fields: usize,
    ) -> VortexResult<()> {
        // Readers that predate nested stats reject a legacy layout without one entry per
        // top-level field, so excluding legacy stats keeps the entries and records nothing.
        let mut exec_ctx = SESSION.create_execution_ctx();
        let acc = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::Min), agg(Stat::Max)])),
            true,
            false,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;

        push(&acc, &chunk, &mut exec_ctx)?;

        let legacy = acc.legacy_aggregate_stats()?;
        assert_eq!(legacy.len(), top_level_fields);
        assert!(legacy.iter().all(AggregateStats::is_empty));
        Ok(())
    }

    #[rstest]
    #[case::flat(
        DType::struct_([("a", i32_dtype())], Nullability::NonNullable),
        StructArray::new(FieldNames::from(["a"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::NonNullable).into_array(),
    )]
    #[case::nested_struct(
        DType::struct_([("a", DType::struct_([("b", i32_dtype())], Nullability::NonNullable))], Nullability::NonNullable),
        StructArray::new(FieldNames::from(["a"]), [StructArray::new(FieldNames::from(["b"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::NonNullable).into_array()], 3, Validity::NonNullable).into_array(),
    )]
    #[case::nullable_nested_struct(
        DType::struct_([("a", DType::struct_([("b", i32_dtype())], Nullability::Nullable))], Nullability::NonNullable),
        StructArray::new(FieldNames::from(["a"]), [StructArray::new(FieldNames::from(["b"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::Array(BoolArray::from_iter([true, false, true]).into_array())).into_array()], 3, Validity::NonNullable).into_array(),
    )]
    #[case::nullable_root(
        DType::struct_([("a", i32_dtype())], Nullability::Nullable),
        StructArray::new(FieldNames::from(["a"]), [buffer![1i32, 2, 3].into_array()], 3, Validity::Array(BoolArray::from_iter([true, false, true]).into_array())).into_array(),
    )]
    #[case::non_struct(i32_dtype(), buffer![1i32, 2, 3].into_array())]
    fn write_nested_stats_false_keeps_legacy_stats(
        #[case] dtype: DType,
        #[case] chunk: ArrayRef,
    ) -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let stats: Arc<[AggregateFnRef]> =
            Arc::from([agg(Stat::NullCount), agg(Stat::Min), agg(Stat::Max)]);
        let ctx = LayoutWriterContext::new(ArrayContext::empty());
        let both =
            FileStatsAccumulator::try_new(&dtype, Some(Arc::clone(&stats)), true, true, &ctx)?;
        let legacy_only = FileStatsAccumulator::try_new(&dtype, Some(stats), false, true, &ctx)?;

        push(&both, &chunk, &mut exec_ctx)?;
        push(&legacy_only, &chunk, &mut exec_ctx)?;

        assert_eq!(
            format!("{:?}", legacy_only.legacy_aggregate_stats()?),
            format!("{:?}", both.legacy_aggregate_stats()?)
        );
        assert!(legacy_only.aggregate_stats().is_err());
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

    #[test]
    fn bounded_max_partial_round_trips() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::NonNullable);
        let max = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: NonZeroUsize::new(8).vortex_expect("non-zero"),
        });
        let mut acc = StatsAccumulator::new(&dtype, Some(std::slice::from_ref(&max)), &|_| true);
        acc.push_all(
            &utf8_chunk(&["short", "a value longer than the bound"]),
            &mut ctx,
        )?;

        let aggregates = acc.aggregate_stats(|_| true)?;
        let [recorded] = aggregates.iter().collect::<Vec<_>>()[..] else {
            vortex_panic!("expected one recorded aggregate");
        };
        assert!(recorded.aggregate_fn().is::<BoundedMax>());
        let partial = recorded
            .partial()
            .vortex_expect("accumulated aggregates have a partial");
        let rebuilt = AggregateStat::try_from_partial(
            recorded.aggregate_fn().clone(),
            partial.clone(),
            &dtype,
        )?;
        assert_eq!(rebuilt.value(), recorded.value());
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

    #[rstest]
    #[case::allowed(true)]
    #[case::forbidden(false)]
    fn default_uncompressed_size_follows_ctx(#[case] allowed: bool) -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::NonNullable);
        let mut aggregates = HashSet::from_iter([Min.id(), Max.id(), NullCount.id()]);
        if allowed {
            aggregates.insert(UncompressedSizeInBytes.id());
        }
        let ctx =
            LayoutWriterContext::new(ArrayContext::empty()).with_allowed_aggregates(aggregates);
        let acc = FileStatsAccumulator::try_new(&dtype, None, true, true, &ctx)?;
        for values in [buffer![1i32, 2, 3], buffer![4i32, 5]] {
            let len = values.len();
            let chunk = StructArray::new(
                FieldNames::from(["a"]),
                [values.into_array()],
                len,
                Validity::NonNullable,
            )
            .into_array();
            push(&acc, &chunk, &mut exec_ctx)?;
        }

        // Five i32 values, which both layouts record only if the editions allow it.
        let size = agg(Stat::UncompressedSizeInBytes);
        let expected = if allowed {
            Precision::exact(ScalarValue::from(20u64))
        } else {
            Precision::Absent
        };
        assert_eq!(
            acc.legacy_aggregate_stats()?[0]
                .get(&size)
                .and_then(Scalar::into_value),
            expected
        );
        assert_eq!(
            acc.aggregate_stats()?[0]
                .get(&size)
                .and_then(Scalar::into_value),
            expected
        );
        Ok(())
    }

    #[test]
    fn forbidden_aggregates_are_gated() -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::NonNullable);
        let ctx = LayoutWriterContext::new(ArrayContext::empty())
            .with_allowed_aggregates(HashSet::from_iter([Min.id(), Max.id(), NanCount.id()]));

        // An explicitly requested aggregate the context forbids fails the write.
        let error = FileStatsAccumulator::try_new(
            &dtype,
            Some(Arc::from([agg(Stat::Sum)])),
            true,
            true,
            &ctx,
        )
        .err()
        .vortex_expect("Sum is not permitted");
        assert!(error.to_string().contains("not permitted by ctx"));

        // The defaults leave a forbidden aggregate out of both the nested and the legacy stats.
        let acc = FileStatsAccumulator::try_new(&dtype, None, true, true, &ctx)?;
        let chunk = StructArray::new(
            FieldNames::from(["a"]),
            [buffer![1i32, 2, 3].into_array()],
            3,
            Validity::NonNullable,
        )
        .into_array();
        push(&acc, &chunk, &mut exec_ctx)?;

        let aggregates = acc.aggregate_stats()?;
        assert!(
            aggregates[0]
                .iter()
                .any(|stat| stat.aggregate_fn().is::<Max>())
        );
        assert!(
            !aggregates[0]
                .iter()
                .any(|stat| stat.aggregate_fn().is::<NullCount>())
        );
        assert_eq!(
            acc.legacy_aggregate_stats()?[0]
                .get(&agg(Stat::NullCount))
                .and_then(Scalar::into_value),
            Precision::Absent
        );
        Ok(())
    }

    #[test]
    fn defaults_do_not_record_sum() -> VortexResult<()> {
        let mut exec_ctx = SESSION.create_execution_ctx();
        let dtype = DType::struct_([("a", i32_dtype())], Nullability::NonNullable);
        let acc = FileStatsAccumulator::try_new(
            &dtype,
            None,
            true,
            true,
            &LayoutWriterContext::new(ArrayContext::empty()),
        )?;
        let chunk = StructArray::new(
            FieldNames::from(["a"]),
            [buffer![1i32, 2, 3].into_array()],
            3,
            Validity::NonNullable,
        )
        .into_array();
        push(&acc, &chunk, &mut exec_ctx)?;

        let sum = agg(Stat::Sum);
        assert!(acc.aggregate_stats()?[0].get(&sum).is_absent());
        assert!(acc.legacy_aggregate_stats()?[0].get(&sum).is_absent());
        Ok(())
    }

    #[rstest]
    #[case::one_chunk_overflows(vec![vec![Some(i64::MAX), Some(1)], vec![Some(2)]], None)]
    #[case::all_chunks_overflow(vec![vec![Some(i64::MAX), Some(1)]; 2], None)]
    #[case::total_overflows(vec![vec![Some(i64::MAX)], vec![Some(1)]], None)]
    #[case::nullable_values(vec![vec![None, Some(3)], vec![Some(4), None]], Some(7))]
    #[case::all_null_chunk(vec![vec![None], vec![Some(5)]], Some(5))]
    fn combines_chunk_sums(
        #[case] chunks: Vec<Vec<Option<i64>>>,
        #[case] expected: Option<i64>,
    ) -> VortexResult<()> {
        // An overflowing chunk saturates the sum's partial state, which stays saturated when
        // merged with later chunks, so the file sum is absent rather than the sum of the chunks
        // that didn't overflow.
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
        let sum = agg(Stat::Sum);
        let mut acc = StatsAccumulator::new(&dtype, Some(std::slice::from_ref(&sum)), &|_| true);
        for chunk in chunks {
            acc.push_all(
                &PrimitiveArray::from_option_iter(chunk).into_array(),
                &mut ctx,
            )?;
        }

        assert_eq!(
            acc.aggregate_stats(|_| true)?.get(&sum),
            expected.map_or(Precision::Absent, |sum| Precision::exact(Scalar::from(sum)))
        );
        Ok(())
    }
}
