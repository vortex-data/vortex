//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::VTable;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::LegacyStatsAccumulator;
use crate::layouts::zoned::LegacyStatsLayout;
use crate::layouts::zoned::Zoned;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_partials;
use crate::layouts::zoned::legacy_stat_values;
use crate::layouts::zoned::schema::default_bounded_stat_max_bytes;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Configuration for building zoned layouts.
///
/// The input stream is assumed to already be partitioned into one chunk per zone, except
/// possibly the final partial zone.
pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: NonZeroUsize,
    /// The aggregate partials to collect for each block.
    ///
    /// If unset, the writer chooses pruning aggregates from the input dtype.
    pub aggregate_fns: Option<Arc<[AggregateFnRef]>>,
    /// Number of chunks to compute aggregate partials in parallel.
    pub concurrency: NonZeroUsize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: unsafe { NonZeroUsize::new_unchecked(8192) },
            aggregate_fns: None,
            concurrency: unsafe {
                NonZeroUsize::new_unchecked(get_available_parallelism().unwrap_or(1))
            },
        }
    }
}

pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
}

impl ZonedStrategy {
    /// Create a writer that emits a data child plus an auxiliary per-zone stats child.
    pub fn new<Child: LayoutStrategy, Stats: LayoutStrategy>(
        child: Child,
        stats: Stats,
        options: ZonedLayoutOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            stats: Arc::new(stats),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ZonedStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let mut aggregate_fns = self
            .options
            .aggregate_fns
            .clone()
            .unwrap_or_else(|| default_zoned_aggregate_fns(stream.dtype(), session));
        // Fields of a nullable struct are written unmasked: a field holds values in rows where the
        // struct is null, which reads through the struct return as nulls. A bound over them, such
        // as a max, still bounds the field as read, only less tightly. A count does not: the null
        // count misses the rows where only the struct is null, so a zone it shows without nulls can
        // still hold rows where `is_null(s.a)` is true. Such a field records only the statistics
        // that hold under masking and its NaN count, which scans read through the struct only as a
        // proof that a zone holds no NaN, and no legacy zone map, whose statistics are fixed.
        //
        // TODO(joe): record every aggregate for fields under a nullable struct. The struct
        //  validity has to travel with each chunk, since repartitioning below the struct writer
        //  re-chunks the stream, and the aggregates then run over `chunk.mask(validity)` while the
        //  data child still receives the unmasked chunk.
        let nullable_ancestor = ctx.has_nullable_ancestor();
        if nullable_ancestor {
            aggregate_fns = aggregates_under_masking(&aggregate_fns);
        }

        // Readers before `vortex.zoned` only know the legacy `vortex.stats` layout. If the write
        // does not permit either layout, the column gets no zone map.
        if !ctx.allows_layout(&Zoned.id()) {
            if ctx.allows_layout(&LegacyStats.id()) && !nullable_ancestor {
                return self
                    .write_legacy_stream(ctx, segment_sink, stream, eof, session)
                    .await;
            }

            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }
        if aggregate_fns.is_empty() {
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let compute_session = session.clone();

        let stats_accumulator = Arc::new(Mutex::new(AggregateStatsAccumulator::new(
            stream.dtype(),
            &aggregate_fns,
        )));
        // The accumulator has dropped the aggregates this dtype cannot hold, leaving the ones
        // this write would record. An aggregate the context forbids fails the write, like a
        // forbidden array or layout: dropping it silently would leave a file that prunes worse
        // than the caller asked for, with nothing in the output saying so.
        let aggregate_fns = stats_accumulator.lock().aggregate_fns();
        for aggregate_fn in aggregate_fns.iter() {
            if !ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }

        let stream_dtype = stream.dtype().clone();
        let concurrency = self.options.concurrency.get();
        let stream = stream
            .map(move |item| {
                let aggregate_fns = Arc::clone(&aggregate_fns);
                let session = compute_session.clone();
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    let partials = aggregate_partials(
                        &chunk,
                        &aggregate_fns,
                        &mut session.create_execution_ctx(),
                    )?;
                    Ok::<_, VortexError>((sequence_id, chunk, partials))
                })
            })
            .buffered(concurrency);

        // Accumulate zone stats in stream order so the auxiliary table stays aligned with the
        // data child.
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let stream = SequentialStreamAdapter::new(
            stream_dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, partials) = item?;
                stats_accumulator2.lock().push_partials(partials)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        let block_size = self.options.block_size;

        // The eof used for the data child should appear _before_ our own stats tables.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        let Some((stats_array, aggregate_fns)) =
            stats_accumulator.lock().as_array(&mut exec_ctx)?
        else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_stream = stats_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, session)
            .await?;

        Ok(
            ZonedLayout::try_new(data_layout, zones_layout, block_size, aggregate_fns)?
                .into_layout(),
        )
    }
}

impl ZonedStrategy {
    /// Writes the data child and a legacy `vortex.stats` zone map, which readers before
    /// `vortex.zoned` can read.
    async fn write_legacy_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let stats_accumulator = Arc::new(Mutex::new(LegacyStatsAccumulator::new(stream.dtype())));
        let stats = stats_accumulator.lock().stats();

        let compute_session = session.clone();
        let stream_dtype = stream.dtype().clone();
        let concurrency = self.options.concurrency.get();
        let stream = stream
            .map(move |item| {
                let stats = Arc::clone(&stats);
                let session = compute_session.clone();
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    let values =
                        legacy_stat_values(&chunk, &stats, &mut session.create_execution_ctx())?;
                    Ok::<_, VortexError>((sequence_id, chunk, values))
                })
            })
            .buffered(concurrency);

        // Add the zone stats in stream order, so that the zone table rows align with the data
        // child.
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let stream = SequentialStreamAdapter::new(
            stream_dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, values) = item?;
                stats_accumulator2.lock().push_values(values)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        // The data child must come before the zone table in the file.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        let Some((stats_array, stats)) = stats_accumulator.lock().as_array(&mut exec_ctx)? else {
            return Ok(data_layout);
        };

        let stats_stream = stats_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, session)
            .await?;

        Ok(
            LegacyStatsLayout::try_new(data_layout, zones_layout, self.options.block_size, stats)?
                .into_layout(),
        )
    }
}

/// Returns the aggregates of `aggregate_fns` that a field beneath a nullable struct records: those
/// whose statistics [hold under masking](holds_under_masking), and the NaN count.
///
/// Float bounds prune only with a proof that the zone holds no NaN, which readers derive from a
/// NaN count of zero. The NaN count of a field also counts the NaNs in the rows the struct hides,
/// so a zone where it is zero holds no NaN as read through the struct either. Through the struct,
/// scans read it only as that proof, and never as a count.
fn aggregates_under_masking(aggregate_fns: &[AggregateFnRef]) -> Arc<[AggregateFnRef]> {
    aggregate_fns
        .iter()
        .filter(|aggregate_fn| holds_under_masking(aggregate_fn) || aggregate_fn.is::<NanCount>())
        .cloned()
        .collect()
}

/// Whether a statistic of `aggregate_fn` over a column also holds, if less tightly, for the column
/// with some of its values replaced by nulls.
///
/// Min and max bound the non-null values, which such a replacement only removes, and so does
/// whether no value is NaN, or whether every value is null. The fields of a nullable struct hold
/// values in rows where the struct is null, so these statistics of a field also hold for the field
/// as read through the struct. Counts do not: a field's null count misses the rows where only the
/// struct is null. Scans read a zone map beneath a nullable struct as covering those hidden values,
/// so they prune with its bounds and never with its counts.
fn holds_under_masking(aggregate_fn: &AggregateFnRef) -> bool {
    aggregate_fn.is::<Min>()
        || aggregate_fn.is::<Max>()
        || aggregate_fn.is::<BoundedMin>()
        || aggregate_fn.is::<BoundedMax>()
        || aggregate_fn.is::<AllNonNan>()
        || aggregate_fn.is::<AllNull>()
}

fn default_zoned_aggregate_fns(dtype: &DType, session: &VortexSession) -> Arc<[AggregateFnRef]> {
    let (max, min) = match dtype {
        DType::Utf8(_) | DType::Binary(_) => (
            BoundedMax.bind(BoundedMaxOptions {
                max_bytes: default_bounded_stat_max_bytes(),
            }),
            BoundedMin.bind(BoundedMinOptions {
                max_bytes: default_bounded_stat_max_bytes(),
            }),
        ),
        _ => (
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Min.bind(NumericalAggregateOpts::skip_nans()),
        ),
    };

    // Sum is deliberately absent: zone maps exist to prune, and a zone sum prunes nothing.
    // Its semantics are also unsettled - null-on-empty was changed in #9113 and reverted in
    // #9324 - so it is not a stat to record in every zone of every file, let alone freeze
    // into an edition. File-level statistics still record a sum by default (see
    // `file_stats::default_pruning_aggregate_fns`).
    let mut aggregate_fns = vec![
        max,
        min,
        NanCount.bind(EmptyOptions),
        NullCount.bind(EmptyOptions),
    ];

    // Stats from spatial extension types are discovered from the registry at runtime instead.
    aggregate_fns.extend(session.aggregate_fns().zone_stat_defaults(dtype));

    aggregate_fns.into()
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnId;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::file_stats::contains_variant;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::schema::aggregate_state_dtype;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::variant(DType::Variant(Nullability::Nullable))]
    #[case::variant_in_struct(DType::Struct(
        StructFields::new(["v"].into(), vec![DType::Variant(Nullability::Nullable)]),
        Nullability::NonNullable,
    ))]
    fn zone_maps_hold_no_variant_state(#[case] dtype: DType) {
        let session = VortexSession::empty().with::<LayoutSession>();
        let accumulator =
            AggregateStatsAccumulator::new(&dtype, &default_zoned_aggregate_fns(&dtype, &session));
        for aggregate_fn in accumulator.aggregate_fns().iter() {
            let state = aggregate_state_dtype(&dtype, aggregate_fn).vortex_expect("kept state");
            assert!(!contains_variant(&state), "{aggregate_fn} keeps a variant");
        }
    }

    /// Write three zones of primitives through `ctx`, returning the aggregates the zoned
    /// layout recorded.
    fn write_zones(ctx: LayoutWriterContext) -> VortexResult<Vec<String>> {
        Ok(write_layout(ctx)?
            .as_::<Zoned>()
            .aggregate_fns()
            .iter()
            .map(|aggregate_fn| aggregate_fn.id().to_string())
            .collect())
    }

    /// Write three zones of primitives through `ctx`, returning the resulting layout.
    fn write_layout(ctx: LayoutWriterContext) -> VortexResult<LayoutRef> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        write_layout_with(&strategy, ctx)
    }

    /// Write three zones of primitives with `strategy` through `ctx`, returning the resulting
    /// layout.
    fn write_layout_with(
        strategy: &ZonedStrategy,
        ctx: LayoutWriterContext,
    ) -> VortexResult<LayoutRef> {
        let (ptr, eof) = SequenceId::root().split();
        let stream = ChunkedArray::from_iter([
            buffer![1, 2, 3].into_array(),
            buffer![4, 5, 6].into_array(),
            buffer![7, 8, 9].into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);

        block_on(|handle: Handle| async move {
            let session = VortexSession::empty()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    Arc::new(TestSegments::default()),
                    stream,
                    eof,
                    &session,
                )
                .await
        })
    }

    #[test]
    fn unrestricted_context_writes_the_default_aggregates() -> VortexResult<()> {
        let written = write_zones(LayoutWriterContext::new(ArrayContext::empty()))?;
        assert!(written.contains(&Min.id().to_string()));
        assert!(written.contains(&Max.id().to_string()));
        assert!(
            !written.contains(&Sum.id().to_string()),
            "wrote {written:?}"
        );
        Ok(())
    }

    /// A field of a nullable struct holds values in rows the struct nulls out. They loosen its
    /// bounds but falsify its counts, so the zone map of an integer field records only the bounds.
    #[test]
    fn a_nullable_ancestor_records_only_bounds() -> VortexResult<()> {
        let written =
            write_zones(LayoutWriterContext::new(ArrayContext::empty()).with_nullable_ancestor())?;
        assert_eq!(written, [Max.id().to_string(), Min.id().to_string()]);
        Ok(())
    }

    /// A legacy zone map records a fixed set of statistics, counts among them, so a field of a
    /// nullable struct gets none.
    #[test]
    fn a_nullable_ancestor_writes_no_legacy_zone_map() -> VortexResult<()> {
        let ctx = LayoutWriterContext::new(ArrayContext::empty())
            .with_allowed_layouts(HashSet::from_iter([LegacyStats.id()]));
        assert!(write_layout(ctx.clone())?.is::<LegacyStats>());
        assert!(!write_layout(ctx.with_nullable_ancestor())?.is::<LegacyStats>());
        Ok(())
    }

    /// Explicitly requested counts are dropped under a nullable ancestor like the default ones,
    /// and a field left with no bound gets no zone map.
    #[test]
    fn a_nullable_ancestor_without_bounds_writes_no_zone_map() -> VortexResult<()> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                aggregate_fns: Some([NullCount.bind(EmptyOptions)].into()),
                ..Default::default()
            },
        );
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_nullable_ancestor();
        assert!(!write_layout_with(&strategy, ctx)?.is::<Zoned>());
        Ok(())
    }

    /// Beneath a nullable struct, a float field records its NaN count with its bounds, since float
    /// bounds prune only with a proof that the zone holds no NaN. Every core edition permits all
    /// three. An integer field holds no NaN, so it records only its bounds.
    #[rstest]
    #[case::float(PType::F64, &[Max.id(), Min.id(), NanCount.id()])]
    #[case::integer(PType::I32, &[Max.id(), Min.id()])]
    fn a_nullable_ancestor_records_the_nan_count_of_floats(
        #[case] ptype: PType,
        #[case] expected: &[AggregateFnId],
    ) {
        let dtype = DType::from(ptype);
        let aggregate_fns =
            aggregates_under_masking(&default_zoned_aggregate_fns(&dtype, &SESSION));
        let recorded = AggregateStatsAccumulator::new(&dtype, &aggregate_fns)
            .aggregate_fns()
            .iter()
            .map(|aggregate_fn| aggregate_fn.id())
            .collect::<Vec<_>>();
        assert_eq!(recorded, expected);
    }

    #[test]
    fn a_permitted_set_covering_the_defaults_writes_them() -> VortexResult<()> {
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_allowed_aggregates(
            HashSet::from_iter([Min.id(), Max.id(), NanCount.id(), NullCount.id()]),
        );
        assert!(write_zones(ctx)?.contains(&Max.id().to_string()));
        Ok(())
    }

    #[test]
    fn a_forbidden_aggregate_fails_the_write() {
        let ctx = LayoutWriterContext::new(ArrayContext::empty())
            .with_allowed_aggregates(HashSet::from_iter([Min.id()]));
        let error = write_zones(ctx).expect_err("the default aggregates are not all permitted");
        assert!(
            error.to_string().contains("not permitted by ctx"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn default_aggregates_bound_variable_length_min_max() {
        let aggregate_fns =
            default_zoned_aggregate_fns(&DType::Utf8(Nullability::NonNullable), &SESSION);

        assert_eq!(
            aggregate_fns[0].as_::<BoundedMax>().max_bytes,
            default_bounded_stat_max_bytes()
        );
        assert_eq!(
            aggregate_fns[1].as_::<BoundedMin>().max_bytes,
            default_bounded_stat_max_bytes()
        );
    }

    #[test]
    fn default_aggregates_keep_fixed_width_min_max_exact() {
        let aggregate_fns = default_zoned_aggregate_fns(&PType::I32.into(), &SESSION);

        assert!(aggregate_fns[0].is::<Max>());
        assert!(aggregate_fns[1].is::<Min>());
        assert!(aggregate_fns[2].is::<NanCount>());
    }

    /// Zone maps never carry a sum, whether or not the dtype could hold one.
    #[rstest]
    #[case::summable(PType::I32.into())]
    #[case::not_summable(DType::Extension(
        Timestamp::new(TimeUnit::Microseconds, Nullability::Nullable).erased(),
    ))]
    fn default_aggregates_never_record_sum(#[case] dtype: DType) {
        let aggregate_fns = default_zoned_aggregate_fns(&dtype, &VortexSession::empty());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| !aggregate_fn.is::<Sum>())
        );
    }
}
