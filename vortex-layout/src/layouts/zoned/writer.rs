//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use futures::future::try_join_all;
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
use vortex_array::dtype::FieldPath;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_partials;
use crate::layouts::zoned::schema::default_bounded_stat_max_bytes;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// One zone map: a table holding one row of aggregate partials per `zone_len` rows.
#[derive(Clone)]
pub struct ZoneOptions {
    /// Rows per zone.
    pub zone_len: NonZeroUsize,
    /// The aggregate partials to store for each zone.
    ///
    /// If unset, the writer chooses pruning aggregates from the input dtype.
    pub aggregate_fns: Option<Arc<[AggregateFnRef]>>,
}

impl ZoneOptions {
    /// A zone map of `zone_len`-row zones holding the default pruning aggregates.
    pub fn new(zone_len: NonZeroUsize) -> Self {
        Self {
            zone_len,
            aggregate_fns: None,
        }
    }

    /// Store `aggregate_fns` for each zone instead of the default pruning aggregates.
    pub fn with_aggregate_fns(mut self, aggregate_fns: Arc<[AggregateFnRef]>) -> Self {
        self.aggregate_fns = Some(aggregate_fns);
        self
    }
}

/// Builds a [`ZonedStrategy`], choosing the zone maps written for each column.
///
/// By default every column gets one zone map of 8,192-row zones holding the default pruning
/// aggregates.
#[derive(Clone)]
pub struct ZonedStrategyBuilder {
    zones: Vec<ZoneOptions>,
    field_zones: HashMap<FieldPath, Vec<ZoneOptions>>,
    concurrency: NonZeroUsize,
}

impl Default for ZonedStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ZonedStrategyBuilder {
    /// A builder writing the default zone map for every column.
    pub fn new() -> Self {
        Self {
            zones: vec![ZoneOptions::new(unsafe {
                NonZeroUsize::new_unchecked(8192)
            })],
            field_zones: HashMap::new(),
            concurrency: unsafe {
                NonZeroUsize::new_unchecked(get_available_parallelism().unwrap_or(1))
            },
        }
    }

    /// Write `zones` for every column without zone maps of its own. An empty list writes no
    /// zone maps.
    ///
    /// **Note**: a column holds at most one zone map for now, so [`Self::build`] panics if
    /// `zones` has more than one.
    pub fn with_zones(mut self, zones: impl IntoIterator<Item = ZoneOptions>) -> Self {
        self.zones = zones.into_iter().collect();
        self
    }

    /// Write `zones` for `field` and every field nested under it, instead of those set by
    /// [`Self::with_zones`]. An empty list writes no zone maps.
    ///
    /// Where the paths of several overrides lead to a column, the longest one applies.
    ///
    /// **Note**: [`Self::build`] panics if `zones` has more than one zone map.
    pub fn with_field_zones(
        mut self,
        field: impl Into<FieldPath>,
        zones: impl IntoIterator<Item = ZoneOptions>,
    ) -> Self {
        let field = field.into();
        assert!(
            !field.is_root(),
            "Set the zone maps of the root with with_zones, not with_field_zones"
        );
        self.field_zones.insert(field, zones.into_iter().collect());
        self
    }

    /// Set the number of chunks to compute aggregate partials for in parallel.
    pub fn with_concurrency(mut self, concurrency: NonZeroUsize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Build a writer that writes each column through `child` and its zone table through `stats`.
    pub fn build<Child: LayoutStrategy, Stats: LayoutStrategy>(
        self,
        child: Child,
        stats: Stats,
    ) -> ZonedStrategy {
        ZonedStrategy {
            child: Arc::new(child),
            stats: Arc::new(stats),
            zones: single_zone_map(self.zones),
            field_zones: self
                .field_zones
                .into_iter()
                .map(|(field, zones)| (field, single_zone_map(zones)))
                .collect(),
            concurrency: self.concurrency,
        }
    }
}

/// The zoned layout holds one zone map, whose zone length its metadata stores as a `u32`.
fn single_zone_map(zones: Vec<ZoneOptions>) -> Option<ZoneOptions> {
    assert!(zones.len() <= 1, "A column holds at most one zone map");
    let zone = zones.into_iter().next()?;
    assert!(
        u32::try_from(zone.zone_len.get()).is_ok(),
        "Zone length {} does not fit in a u32",
        zone.zone_len
    );
    Some(zone)
}

/// Writes a data child plus an auxiliary per-zone stats child, choosing the zone map of each
/// column by its field path.
///
/// The input chunks need not line up with the zones: the writer cuts each chunk at the zone
/// boundaries and combines the partials of zones that span chunks.
pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    zones: Option<ZoneOptions>,
    field_zones: HashMap<FieldPath, Option<ZoneOptions>>,
    concurrency: NonZeroUsize,
}

impl ZonedStrategy {
    /// The zone map of the column at `field_path`: that of the longest overridden path leading to
    /// it, or else the default.
    fn zones(&self, field_path: &FieldPath) -> Option<&ZoneOptions> {
        let parts = field_path.parts();
        (1..=parts.len())
            .rev()
            .find_map(|len| {
                self.field_zones
                    .get(&FieldPath::from(parts[..len].to_vec()))
            })
            .unwrap_or(&self.zones)
            .as_ref()
    }
}

/// The aggregate partials of the rows of one chunk that fall in one zone.
struct ZonePiece {
    partials: Vec<Scalar>,
    /// Whether these rows end their zone.
    ends_zone: bool,
}

/// Write-time state of the zone table.
struct ZoneTableState {
    table: AggregateStatsAccumulator,
    dtype: DType,
    aggregate_fns: Arc<[AggregateFnRef]>,
    /// Partials of the pieces of the zone that is still receiving rows.
    open_zone: Vec<Vec<Scalar>>,
}

impl ZoneTableState {
    fn push_piece(&mut self, piece: ZonePiece) -> VortexResult<()> {
        self.open_zone.push(piece.partials);
        if piece.ends_zone {
            self.close_zone()?;
        }
        Ok(())
    }

    /// Records the open zone, including a final short one.
    fn close_zone(&mut self) -> VortexResult<()> {
        if self.open_zone.len() <= 1 {
            // A zone within one chunk keeps its partials as computed.
            return match self.open_zone.pop() {
                Some(partials) => self.table.push_partials(partials),
                None => Ok(()),
            };
        }
        let partials = self
            .aggregate_fns
            .iter()
            .enumerate()
            .map(|(idx, aggregate_fn)| {
                let mut accumulator = aggregate_fn.accumulator(&self.dtype)?;
                for piece in &self.open_zone {
                    accumulator.combine_partials(&piece[idx])?;
                }
                accumulator.flush()
            })
            .collect::<VortexResult<Vec<_>>>()?;
        self.open_zone.clear();
        self.table.push_partials(partials)
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
        // Fields of a nullable struct are written unmasked, so every aggregate over them sees
        // values the logical column does not contain: counts come out wrong and min/max come out
        // wider than the truth. Zone maps hold exact values only, so until the aggregates are
        // computed over the masked values no zone map is written for such a field.
        //
        // TODO(joe): write zone maps for fields under a nullable struct. The struct validity has
        //  to travel with each chunk, since repartitioning below the struct writer re-chunks the
        //  stream, and the aggregates then run over `chunk.mask(validity)` while the data child
        //  still receives the unmasked chunk.
        let zone = match self.zones(ctx.field_path()) {
            Some(zone) if !ctx.has_nullable_ancestor() => zone,
            _ => {
                return self
                    .child
                    .write_stream(ctx, segment_sink, stream, eof, session)
                    .await;
            }
        };
        let dtype = stream.dtype().clone();
        let aggregate_fns = zone
            .aggregate_fns
            .clone()
            .unwrap_or_else(|| default_zoned_aggregate_fns(&dtype, session));

        let table = AggregateStatsAccumulator::new(&dtype, &aggregate_fns);
        // The accumulator has dropped the aggregates this dtype cannot hold, leaving the ones
        // this write would record. An aggregate the context forbids fails the write, like a
        // forbidden array or layout: dropping it silently would leave a file that prunes worse
        // than the caller asked for, with nothing in the output saying so.
        let aggregate_fns = table.aggregate_fns();
        for aggregate_fn in aggregate_fns.iter() {
            if !ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }
        let zone_table = Arc::new(Mutex::new(ZoneTableState {
            table,
            dtype: dtype.clone(),
            aggregate_fns: Arc::clone(&aggregate_fns),
            open_zone: Vec::new(),
        }));

        // Each zone piece of a chunk is aggregated on its own task, so a chunk spanning many
        // zones still spreads across cores. Row offsets are assigned in stream order.
        let zone_len = zone.zone_len.get();
        let compute_session = session.clone();
        let mut next_row = 0;
        let stream = stream
            .map(move |item| {
                let chunk_start = next_row;
                if let Ok((_, chunk)) = &item {
                    next_row += chunk.len();
                }
                let aggregate_fns = Arc::clone(&aggregate_fns);
                let session = compute_session.clone();
                async move {
                    let (sequence_id, chunk) = item?;
                    let tasks = zone_pieces(chunk_start, chunk.len(), zone_len)
                        .map(|(rows, ends_zone)| {
                            let piece = if rows.len() == chunk.len() {
                                chunk.clone()
                            } else {
                                chunk.slice(rows)?
                            };
                            let aggregate_fns = Arc::clone(&aggregate_fns);
                            let piece_session = session.clone();
                            Ok(session.handle().spawn_cpu(move || {
                                Ok::<_, VortexError>(ZonePiece {
                                    partials: aggregate_partials(
                                        &piece,
                                        &aggregate_fns,
                                        &mut piece_session.create_execution_ctx(),
                                    )?,
                                    ends_zone,
                                })
                            }))
                        })
                        .collect::<VortexResult<Vec<_>>>()?;
                    let pieces = try_join_all(tasks).await?;
                    Ok::<_, VortexError>((sequence_id, chunk, pieces))
                }
            })
            .buffered(self.concurrency.get());

        // Combine the pieces in stream order so the zone table stays aligned with the data child.
        let zone_table2 = Arc::clone(&zone_table);
        let stream = SequentialStreamAdapter::new(
            dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, pieces) = item?;
                let mut zone_table = zone_table2.lock();
                for piece in pieces {
                    zone_table.push_piece(piece)?;
                }
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

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
        let stats = {
            let mut zone_table = zone_table.lock();
            zone_table.close_zone()?;
            zone_table.table.as_array(&mut exec_ctx)?
        };
        let Some((stats_array, aggregate_fns)) = stats else {
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
            ZonedLayout::try_new(data_layout, zones_layout, zone.zone_len, aggregate_fns)?
                .into_layout(),
        )
    }
}

/// Cuts the rows of a chunk of `chunk_len` rows starting at row `chunk_start` at the zone
/// boundaries, yielding each piece's rows within the chunk and whether the piece ends its zone.
fn zone_pieces(
    chunk_start: usize,
    chunk_len: usize,
    zone_len: usize,
) -> impl Iterator<Item = (Range<usize>, bool)> {
    let chunk_end = chunk_start + chunk_len;
    let mut row = chunk_start;
    std::iter::from_fn(move || {
        if row >= chunk_end {
            return None;
        }
        let zone_end = (row / zone_len + 1) * zone_len;
        let piece_end = zone_end.min(chunk_end);
        let piece = (
            row - chunk_start..piece_end - chunk_start,
            piece_end == zone_end,
        );
        row = piece_end;
        Some(piece)
    })
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::Buffer;
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
    use crate::layouts::table::TableStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

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
        let strategy = ZonedStrategyBuilder::new()
            .with_zones([ZoneOptions::new(
                NonZeroUsize::new(3).vortex_expect("non zero"),
            )])
            .build(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                FlatLayoutStrategy::default(),
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

    fn zoned(zone_len: usize) -> ZonedStrategyBuilder {
        ZonedStrategyBuilder::new().with_zones([ZoneOptions::new(
            NonZeroUsize::new(zone_len).vortex_expect("non zero"),
        )])
    }

    /// `values` as a chunked array of `chunk_len`-row chunks.
    fn chunked(values: &[i32], chunk_len: usize) -> ArrayRef {
        ChunkedArray::from_iter(
            values
                .chunks(chunk_len)
                .map(|chunk| chunk.iter().copied().collect::<Buffer<i32>>().into_array()),
        )
        .into_array()
    }

    fn write(
        strategy: impl LayoutStrategy,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let sink = Arc::clone(&segments);
        let layout = block_on(|handle: Handle| async move {
            let session = VortexSession::empty()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            strategy
                .write_stream(ArrayContext::empty().into(), sink, stream, eof, &session)
                .await
        })?;
        Ok((segments, layout))
    }

    /// Reads the zone table of the zoned `layout`.
    fn zone_table(layout: &LayoutRef, segments: &Arc<dyn SegmentSource>) -> VortexResult<ArrayRef> {
        let zones = &layout.children()?[1];
        block_on(|handle: Handle| async move {
            let session = VortexSession::empty()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            let reader = zones.new_reader(
                "".into(),
                Arc::clone(segments),
                &session,
                &Default::default(),
            )?;
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

    /// The zone table is the same however the input is chunked.
    #[rstest]
    #[case::chunks_smaller_than_zones(3)]
    #[case::chunks_larger_than_zones(16)]
    #[case::chunks_unaligned(7)]
    fn zones_need_not_line_up_with_chunks(#[case] chunk_len: usize) -> VortexResult<()> {
        let values: Vec<i32> = (0..50).map(|i| (i * 37) % 23).collect();
        let strategy = || {
            zoned(5).build(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                FlatLayoutStrategy::default(),
            )
        };
        let (segments, layout) = write(strategy(), chunked(&values, chunk_len))?;
        // Chunks of exactly one zone each never combine partials.
        let (aligned_segments, aligned) = write(strategy(), chunked(&values, 5))?;

        assert_eq!(layout.as_::<Zoned>().zone_len(), 5);
        assert_arrays_eq!(
            zone_table(&layout, &segments)?,
            zone_table(&aligned, &aligned_segments)?,
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    /// Each column takes the zone map of the longest overridden path leading to it.
    #[test]
    fn field_overrides_choose_each_column_zone_map() -> VortexResult<()> {
        let column = || buffer![1i32, 2, 3, 4, 5, 6].into_array();
        let array = StructArray::from_fields(&[
            ("a", column()),
            ("b", column()),
            (
                "c",
                StructArray::from_fields(&[("d", column()), ("e", column())])?.into_array(),
            ),
        ])?
        .into_array();
        let max_only: Arc<[AggregateFnRef]> =
            [Max.bind(NumericalAggregateOpts::skip_nans())].into();
        let zoned = zoned(3)
            .with_field_zones(
                FieldPath::from_name("b"),
                [
                    ZoneOptions::new(NonZeroUsize::new(2).vortex_expect("non zero"))
                        .with_aggregate_fns(max_only),
                ],
            )
            .with_field_zones(FieldPath::from_name("c"), [])
            .with_field_zones(
                FieldPath::from_name("c").push("d"),
                [ZoneOptions::new(
                    NonZeroUsize::new(6).vortex_expect("non zero"),
                )],
            )
            .build(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                FlatLayoutStrategy::default(),
            );
        let (_, layout) = write(
            TableStrategy::new(Arc::new(FlatLayoutStrategy::default()), Arc::new(zoned)),
            array,
        )?;

        let columns = layout.children()?;
        let nested = columns[2].children()?;
        let zone_len = |column: &LayoutRef| column.as_opt::<Zoned>().map(|zoned| zoned.zone_len());
        assert_eq!(zone_len(&columns[0]), Some(3));
        assert_eq!(zone_len(&columns[1]), Some(2));
        assert_eq!(zone_len(&nested[0]), Some(6));
        assert_eq!(zone_len(&nested[1]), None);
        assert_eq!(
            columns[1]
                .as_::<Zoned>()
                .aggregate_fns()
                .iter()
                .map(|aggregate_fn| aggregate_fn.id())
                .collect::<Vec<_>>(),
            [Max.id()]
        );
        Ok(())
    }

    #[test]
    #[should_panic(expected = "at most one zone map")]
    fn several_zone_maps_per_column_are_rejected() {
        let zone = ZoneOptions::new(NonZeroUsize::new(3).vortex_expect("non zero"));
        drop(
            ZonedStrategyBuilder::new()
                .with_zones([zone.clone(), zone])
                .build(FlatLayoutStrategy::default(), FlatLayoutStrategy::default()),
        );
    }
}
