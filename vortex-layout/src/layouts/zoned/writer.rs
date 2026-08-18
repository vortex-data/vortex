//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
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
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_partials;
use crate::layouts::zoned::schema::default_bounded_stat_max_bytes;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;

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

impl LayoutStrategy for ZonedStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let aggregate_fns = self
            .options
            .aggregate_fns
            .clone()
            .unwrap_or_else(|| default_zoned_aggregate_fns(&dtype, session));
        let stats_accumulator = AggregateStatsAccumulator::new(&dtype, &aggregate_fns);
        let aggregate_fns = stats_accumulator.aggregate_fns();
        // The accumulator has dropped the aggregates this dtype cannot hold, leaving the ones
        // this write would record. An aggregate the context forbids fails the write, like a
        // forbidden array or layout: dropping it silently would leave a file that prunes worse
        // than the caller asked for, with nothing in the output saying so.
        for aggregate_fn in aggregate_fns.iter() {
            if !ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        let data = self
            .child
            .new_writer(ctx.clone(), Arc::clone(&segment_sink), dtype, session)?;

        Ok(Box::new(ZonedLayoutWriter {
            data,
            stats_strategy: Arc::clone(&self.stats),
            ctx,
            segment_sink,
            session: session.clone(),
            stats_accumulator,
            aggregate_fns,
            buffered_bytes,
            concurrency: self.options.concurrency.get(),
            block_size: self.options.block_size,
            pending: VecDeque::new(),
            stats_sequence: None,
        }))
    }
}

type ZoneFuture = BoxFuture<
    'static,
    VortexResult<(
        SequenceId,
        ArrayRef,
        Vec<Scalar>,
        crate::BufferedBytesReservation,
    )>,
>;

struct ZonedLayoutWriter {
    data: Box<dyn LayoutWriter>,
    stats_strategy: Arc<dyn LayoutStrategy>,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    session: VortexSession,
    stats_accumulator: AggregateStatsAccumulator,
    aggregate_fns: Arc<[AggregateFnRef]>,
    buffered_bytes: crate::BufferedBytesTracker,
    concurrency: usize,
    block_size: NonZeroUsize,
    pending: VecDeque<ZoneFuture>,
    stats_sequence: Option<SequencePointer>,
}

impl ZonedLayoutWriter {
    async fn drain_one(&mut self) -> VortexResult<()> {
        let Some(future) = self.pending.pop_front() else {
            return Ok(());
        };
        let (sequence_id, chunk, partials, reservation) = future.await?;
        self.stats_accumulator.push_partials(partials)?;
        drop(reservation);
        self.data.write(sequence_id, chunk).await
    }
}

#[async_trait]
impl LayoutWriter for ZonedLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let aggregate_fns = Arc::clone(&self.aggregate_fns);
        let session = self.session.clone();
        let reservation = self.buffered_bytes.reserve(chunk.nbytes());
        self.pending.push_back(
            self.session
                .handle()
                .spawn_cpu(move || {
                    let partials = aggregate_partials(
                        &chunk,
                        &aggregate_fns,
                        &mut session.create_execution_ctx(),
                    )?;
                    Ok((sequence_id, chunk, partials, reservation))
                })
                .boxed(),
        );
        if self.pending.len() >= self.concurrency {
            self.drain_one().await?;
        }
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        while !self.pending.is_empty() {
            self.drain_one().await?;
        }
        let mut sequence = sequence_id.descend();
        self.data.finish(sequence.advance()).await?;
        self.stats_sequence = Some(sequence);
        Ok(())
    }

    async fn close(mut self: Box<Self>) -> VortexResult<LayoutRef> {
        let data_layout = self.data.close().await?;
        let Some((stats_array, aggregate_fns)) = self
            .stats_accumulator
            .as_array(&mut self.session.create_execution_ctx())?
        else {
            return Ok(data_layout);
        };

        let stats_array = stats_array.into_array();
        let mut stats = self.stats_strategy.new_writer(
            self.ctx,
            self.segment_sink,
            stats_array.dtype().clone(),
            &self.session,
        )?;
        let mut stats_sequence = self
            .stats_sequence
            .take()
            .vortex_expect("zoned writer must be finished before close");
        stats.write(stats_sequence.advance(), stats_array).await?;
        stats.finish(stats_sequence.advance()).await?;
        let zones_layout = stats.close().await?;
        Ok(
            ZonedLayout::try_new(data_layout, zones_layout, self.block_size, aggregate_fns)?
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
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
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
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    /// Write three zones of primitives through `ctx`, returning the aggregates the zoned
    /// layout recorded.
    fn write_zones(ctx: LayoutWriterContext) -> VortexResult<Vec<String>> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let (ptr, eof) = SequenceId::root().split();
        let stream = ChunkedArray::from_iter([
            buffer![1, 2, 3].into_array(),
            buffer![4, 5, 6].into_array(),
            buffer![7, 8, 9].into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);

        let layout = block_on(|handle: Handle| async move {
            let session = vortex_array::array_session()
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
        })?;

        Ok(layout
            .as_::<Zoned>()
            .aggregate_fns()
            .iter()
            .map(|aggregate_fn| aggregate_fn.id().to_string())
            .collect())
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
