//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
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
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::DemandChannel;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_demanded_partials;
use crate::layouts::zoned::aggregate_partials;
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
        let aggregate_fns = self
            .options
            .aggregate_fns
            .clone()
            .unwrap_or_else(|| default_zoned_aggregate_fns(stream.dtype(), session));
        // Children of a nullable container are written unmasked, so an aggregate over them sees
        // values the logical column does not contain: counts come out wrong and min/max come out
        // wider than the truth. The container publishes which rows the logical column contains,
        // and each zone aggregates only those demanded rows. Zone maps hold exact values only, so
        // without that demand no zone map is written.
        let demand = if ctx.has_nullable_ancestor() {
            let Some(demand) = ctx.claim_demand() else {
                return self
                    .child
                    .write_stream(ctx, segment_sink, stream, eof, session)
                    .await;
            };
            Some(demand)
        } else {
            None
        };
        // The data child receives the same rows, so it is handed the demand this writer takes,
        // e.g. for a list writer to derive the demand on its elements.
        let data_demand = demand.as_ref().map(|_| DemandChannel::new());
        let data_ctx = match &data_demand {
            Some(data_demand) => ctx
                .clone()
                .without_demand()
                .with_demand(data_demand.clone()),
            None => ctx.clone(),
        };
        let ctx = ctx.without_demand();
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
                // Take the demand here rather than on the CPU pool, so chunks take their demand in
                // stream order. `None` means the demand was lost.
                let chunk_demand = match (&item, &demand) {
                    (Ok((_, chunk)), Some(demand)) => Some(demand.take(chunk.len())),
                    _ => None,
                };
                if let (Some(chunk_demand), Some(data_demand)) = (&chunk_demand, &data_demand) {
                    match chunk_demand {
                        Some(chunk_demand) => data_demand.push(chunk_demand.clone()),
                        None => data_demand.lose(),
                    }
                }
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    let mut ctx = session.create_execution_ctx();
                    let partials = match chunk_demand {
                        None => Some(aggregate_partials(&chunk, &aggregate_fns, &mut ctx)?),
                        Some(Some(demand)) => Some(aggregate_demanded_partials(
                            &chunk,
                            &demand,
                            &aggregate_fns,
                            &mut ctx,
                        )?),
                        Some(None) => None,
                    };
                    Ok::<_, VortexError>((sequence_id, chunk, partials))
                })
            })
            .buffered(concurrency);

        // Accumulate zone stats in stream order so the auxiliary table stays aligned with the
        // data child.
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let demand_lost = Arc::new(AtomicBool::new(false));
        let demand_lost2 = Arc::clone(&demand_lost);
        let stream = SequentialStreamAdapter::new(
            stream_dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, partials) = item?;
                match partials {
                    Some(partials) => stats_accumulator2.lock().push_partials(partials)?,
                    None => demand_lost2.store(true, Ordering::Relaxed),
                }
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
                data_ctx,
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        // Without the demand on every zone, the zones' aggregates are not exact.
        if demand_lost.load(Ordering::Relaxed) {
            return Ok(data_layout);
        }

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
    // into an edition. File-level statistics still record `Stat::Sum` via `PRUNING_STATS`.
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
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;
    use crate::test::read_zone_stats;

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
        Ok(write_chunks(
            ctx,
            vec![
                buffer![1, 2, 3].into_array(),
                buffer![4, 5, 6].into_array(),
                buffer![7, 8, 9].into_array(),
            ],
        )?
        .0)
    }

    /// Write `chunks` as zones of three rows through `ctx`, returning the resulting layout and the
    /// segments it was written to.
    fn write_chunks(
        ctx: LayoutWriterContext,
        chunks: Vec<ArrayRef>,
    ) -> VortexResult<(LayoutRef, Arc<TestSegments>)> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let (ptr, eof) = SequenceId::root().split();
        let stream = ChunkedArray::from_iter(chunks)
            .into_array()
            .to_array_stream()
            .sequenced(ptr);

        let segments = Arc::new(TestSegments::default());
        let segments2 = Arc::clone(&segments);
        let layout = block_on(|handle: Handle| async move {
            let session = vortex_array::array_session()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            strategy
                .write_stream(ctx, segments2, stream, eof, &session)
                .await
        })?;
        Ok((layout, segments))
    }

    /// Under a nullable ancestor that publishes its demand, each zone aggregates only the rows
    /// the ancestor demands, and undemanded values never widen the zone's bounds.
    #[test]
    fn demanded_zones_aggregate_only_demanded_rows() -> VortexResult<()> {
        let demand = DemandChannel::new();
        demand.push(Mask::from_iter([true, true, true, false, true]));
        demand.push(Mask::from_iter([true, true, false, false]));
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_demand(demand);

        let (layout, segments) = write_chunks(
            ctx,
            vec![
                PrimitiveArray::from_option_iter([Some(1), Some(2), Some(3)]).into_array(),
                PrimitiveArray::from_option_iter([Some(-40), None, Some(6)]).into_array(),
                PrimitiveArray::from_option_iter([Some(7), Some(80), None]).into_array(),
            ],
        )?;
        assert!(layout.is::<Zoned>());

        let mut exec_ctx = vortex_array::array_session().create_execution_ctx();
        let min = read_zone_stats(&layout, segments.clone(), &Min.bind(Default::default()))?;
        assert_arrays_eq!(
            min,
            PrimitiveArray::from_option_iter([Some(1i32), Some(6), Some(7)]),
            &mut exec_ctx
        );
        let max = read_zone_stats(&layout, segments.clone(), &Max.bind(Default::default()))?;
        assert_arrays_eq!(
            max,
            PrimitiveArray::from_option_iter([Some(3i32), Some(6), Some(7)]),
            &mut exec_ctx
        );
        // The null in the second zone is demanded, the null in the third zone is not.
        let null_count = read_zone_stats(&layout, segments, &NullCount.bind(EmptyOptions))?;
        assert_arrays_eq!(
            null_count,
            PrimitiveArray::from_option_iter([Some(0u64), Some(1), Some(0)]),
            &mut exec_ctx
        );
        Ok(())
    }

    /// A zone whose demand was never published has no exact aggregates, so no zone map is written.
    #[test]
    fn lost_demand_writes_no_zone_map() -> VortexResult<()> {
        let demand = DemandChannel::new();
        demand.push(Mask::new_true(4));
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_demand(demand);

        assert!(!write_layout(ctx)?.is::<Zoned>());
        Ok(())
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

    /// A field of a nullable struct is written unmasked, so no aggregate over it is exact and no
    /// zone map is written.
    #[test]
    fn a_nullable_ancestor_writes_no_zone_map() -> VortexResult<()> {
        let layout =
            write_layout(LayoutWriterContext::new(ArrayContext::empty()).with_nullable_ancestor())?;
        assert!(!layout.is::<Zoned>());
        Ok(())
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
        let aggregate_fns = default_zoned_aggregate_fns(
            &DType::Utf8(Nullability::NonNullable),
            &vortex_array::array_session(),
        );

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
        let aggregate_fns =
            default_zoned_aggregate_fns(&PType::I32.into(), &vortex_array::array_session());

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
        let aggregate_fns = default_zoned_aggregate_fns(&dtype, &vortex_array::array_session());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| !aggregate_fn.is::<Sum>())
        );
    }
}
