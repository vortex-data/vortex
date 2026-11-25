//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
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
use vortex_array::arrays::ChunkedArray;
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
/// The input chunks need not line up with the zones: the writer cuts the stream into zones of
/// `block_size` rows itself, and only the final zone may be shorter.
pub struct ZonedLayoutOptions {
    /// The number of rows in each zone.
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
        // Fields of a nullable struct are written unmasked, so every aggregate over them sees
        // values the logical column does not contain: counts come out wrong and min/max come out
        // wider than the truth. Zone maps hold exact values only, so until the aggregates are
        // computed over the masked values no zone map is written for such a field.
        //
        // TODO(joe): write zone maps for fields under a nullable struct. The struct validity has
        //  to travel with each chunk, since repartitioning below the struct writer re-chunks the
        //  stream, and the aggregates then run over `chunk.mask(validity)` while the data child
        //  still receives the unmasked chunk.
        if ctx.has_nullable_ancestor() {
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Readers before `vortex.zoned` only know the legacy `vortex.stats` layout. If the write
        // does not permit either layout, the column gets no zone map.
        if !ctx.allows_layout(&Zoned.id()) {
            if ctx.allows_layout(&LegacyStats.id()) {
                return self
                    .write_legacy_stream(ctx, segment_sink, stream, eof, session)
                    .await;
            }

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
        let zones = Arc::new(Mutex::new(ZoneCutter::new(self.options.block_size)));
        let zones2 = Arc::clone(&zones);
        let aggregate_fns2 = Arc::clone(&aggregate_fns);
        let stream = stream
            .map(move |item| {
                // Zones are cut here, in stream order, before the chunks spread across tasks.
                let chunk_zones = zones2.lock().push_item(&item);
                let aggregate_fns = Arc::clone(&aggregate_fns2);
                let session = compute_session.clone();
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    let mut exec_ctx = session.create_execution_ctx();
                    let partials = chunk_zones?
                        .iter()
                        .map(|zone| aggregate_partials(zone, &aggregate_fns, &mut exec_ctx))
                        .collect::<VortexResult<Vec<_>>>()?;
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
                let mut stats_accumulator = stats_accumulator2.lock();
                for zone_partials in partials {
                    stats_accumulator.push_partials(zone_partials)?;
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
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        // The final zone ends with the stream, so it may be short.
        let final_zone = zones.lock().finish()?;
        if let Some(zone) = final_zone {
            let compute_session = session.clone();
            let partials = session
                .handle()
                .spawn_cpu(move || {
                    aggregate_partials(
                        &zone,
                        &aggregate_fns,
                        &mut compute_session.create_execution_ctx(),
                    )
                })
                .await?;
            stats_accumulator.lock().push_partials(partials)?;
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
        let zones = Arc::new(Mutex::new(ZoneCutter::new(self.options.block_size)));
        let zones2 = Arc::clone(&zones);
        let stats2 = Arc::clone(&stats);
        let stream = stream
            .map(move |item| {
                let chunk_zones = zones2.lock().push_item(&item);
                let stats = Arc::clone(&stats2);
                let session = compute_session.clone();
                session.handle().spawn_cpu(move || {
                    let (sequence_id, chunk) = item?;
                    let mut exec_ctx = session.create_execution_ctx();
                    let values = chunk_zones?
                        .iter()
                        .map(|zone| legacy_stat_values(zone, &stats, &mut exec_ctx))
                        .collect::<VortexResult<Vec<_>>>()?;
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
                let mut stats_accumulator = stats_accumulator2.lock();
                for zone_values in values {
                    stats_accumulator.push_values(zone_values)?;
                }
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

        let final_zone = zones.lock().finish()?;
        if let Some(zone) = final_zone {
            let compute_session = session.clone();
            let values = session
                .handle()
                .spawn_cpu(move || {
                    legacy_stat_values(&zone, &stats, &mut compute_session.create_execution_ctx())
                })
                .await?;
            stats_accumulator.lock().push_values(values)?;
        }

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

/// Cuts a stream of chunks into zones of `zone_len` rows, whatever the chunk boundaries.
///
/// A zone within one chunk is a slice of it, and a zone spanning chunks is a [`ChunkedArray`] of
/// their slices. A chunk that is exactly one zone is returned as is.
struct ZoneCutter {
    zone_len: usize,
    /// Slices of the zone that is still receiving rows.
    open_zone: Vec<ArrayRef>,
    open_len: usize,
}

impl ZoneCutter {
    fn new(zone_len: NonZeroUsize) -> Self {
        Self {
            zone_len: zone_len.get(),
            open_zone: Vec::new(),
            open_len: 0,
        }
    }

    /// Like [`Self::push`], passing an errored stream item through untouched.
    fn push_item<S>(&mut self, item: &VortexResult<(S, ArrayRef)>) -> VortexResult<Vec<ArrayRef>> {
        match item {
            Ok((_, chunk)) => self.push(chunk),
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Adds the next chunk of the stream, returning the zones it completes.
    fn push(&mut self, chunk: &ArrayRef) -> VortexResult<Vec<ArrayRef>> {
        let mut zones = Vec::new();
        let mut offset = 0;
        while offset < chunk.len() {
            let piece_len = (self.zone_len - self.open_len).min(chunk.len() - offset);
            let piece = if piece_len == chunk.len() {
                chunk.clone()
            } else {
                chunk.slice(offset..offset + piece_len)?
            };
            self.open_zone.push(piece);
            self.open_len += piece_len;
            offset += piece_len;
            if self.open_len == self.zone_len {
                zones.push(self.take_open_zone()?);
            }
        }
        Ok(zones)
    }

    /// Returns the final, short zone, if the stream did not end on a zone boundary.
    fn finish(&mut self) -> VortexResult<Option<ArrayRef>> {
        if self.open_zone.is_empty() {
            return Ok(None);
        }
        self.take_open_zone().map(Some)
    }

    fn take_open_zone(&mut self) -> VortexResult<ArrayRef> {
        let mut pieces = std::mem::take(&mut self.open_zone);
        self.open_len = 0;
        if pieces.len() == 1 {
            return Ok(pieces.remove(0));
        }
        let dtype = pieces[0].dtype().clone();
        Ok(ChunkedArray::try_new(pieces, dtype)?.into_array())
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
    use vortex_array::MaskFuture;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_expr::root;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::file_stats::contains_variant;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::LegacyStats;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::schema::aggregate_state_dtype;
    use crate::segments::SegmentSource;
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

    fn session_with_handle(handle: Handle) -> VortexSession {
        VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
            .with_handle(handle)
    }

    /// Writes `values`, chunked into `chunk_len` rows, with zones of `zone_len` rows.
    fn write_chunked(
        ctx: LayoutWriterContext,
        values: &[i32],
        chunk_len: usize,
        zone_len: usize,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(zone_len).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let (ptr, eof) = SequenceId::root().split();
        let stream = ChunkedArray::from_iter(
            values
                .chunks(chunk_len)
                .map(|chunk| chunk.iter().copied().collect::<Buffer<i32>>().into_array()),
        )
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
        let segments = Arc::new(TestSegments::default());
        let sink = Arc::clone(&segments);
        let layout = block_on(|handle: Handle| async move {
            strategy
                .write_stream(ctx, sink, stream, eof, &session_with_handle(handle))
                .await
        })?;
        Ok((segments, layout))
    }

    /// Reads the zone table, the second child of a zoned or legacy stats layout.
    fn zone_table(
        layout: &LayoutRef,
        segments: Arc<dyn SegmentSource>,
    ) -> VortexResult<ArrayRef> {
        let zones = layout.children()?.remove(1);
        block_on(|handle: Handle| async move {
            let session = session_with_handle(handle);
            let reader = zones.new_reader("".into(), segments, &session, &Default::default())?;
            let row_count = zones.row_count();
            reader
                .projection_evaluation(
                    &(0..row_count),
                    &root().bind(reader.dtype())?,
                    MaskFuture::new_true(usize::try_from(row_count)?),
                )?
                .await
        })
    }

    /// The zone table is the same however the input is chunked, for both the zoned layout and
    /// the legacy stats layout.
    #[rstest]
    fn zones_need_not_line_up_with_chunks(
        #[values(1, 3, 7, 16, 50)] chunk_len: usize,
        #[values(false, true)] legacy: bool,
    ) -> VortexResult<()> {
        const ZONE_LEN: usize = 5;
        let ctx = || {
            let ctx = LayoutWriterContext::new(ArrayContext::empty());
            if legacy {
                ctx.with_allowed_layouts(HashSet::from_iter([LegacyStats.id()]))
            } else {
                ctx
            }
        };
        // 48 rows, so the final zone is short.
        let values: Vec<i32> = (0..48).map(|i| (i * 37) % 23).collect();

        let (segments, layout) = write_chunked(ctx(), &values, chunk_len, ZONE_LEN)?;
        // Chunks of exactly one zone each are the input the writer used to require.
        let (aligned_segments, aligned) = write_chunked(ctx(), &values, ZONE_LEN, ZONE_LEN)?;

        assert_eq!(layout.is::<LegacyStats>(), legacy);
        let table = zone_table(&layout, segments)?;
        assert_eq!(table.len(), values.len().div_ceil(ZONE_LEN));
        assert_arrays_eq!(
            table,
            zone_table(&aligned, aligned_segments)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[rstest]
    #[case::one_chunk_per_zone(&[3, 3, 3], &[3, 3, 3])]
    #[case::chunks_smaller_than_zones(&[1, 1, 1, 1, 1, 1, 1], &[3, 3, 1])]
    #[case::chunks_larger_than_zones(&[8], &[3, 3, 2])]
    #[case::chunks_unaligned(&[2, 4, 0, 5], &[3, 3, 3, 2])]
    fn zone_cutter_cuts_at_zone_boundaries(
        #[case] chunk_lens: &[usize],
        #[case] zone_lens: &[usize],
    ) -> VortexResult<()> {
        let mut cutter = ZoneCutter::new(NonZeroUsize::new(3).vortex_expect("non zero"));
        let mut zones = Vec::new();
        let mut next = 0i32;
        for &chunk_len in chunk_lens {
            let chunk = (next..next + i32::try_from(chunk_len)?)
                .collect::<Buffer<i32>>()
                .into_array();
            next += i32::try_from(chunk_len)?;
            zones.extend(cutter.push(&chunk)?);
        }
        zones.extend(cutter.finish()?);

        assert_eq!(zones.iter().map(|zone| zone.len()).collect::<Vec<_>>(), zone_lens);
        assert_arrays_eq!(
            ChunkedArray::from_iter(zones).into_array(),
            (0..next).collect::<Buffer<i32>>().into_array(),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }
}
