use std::iter;
use std::ops::Range;
use std::sync::{Arc, OnceLock, RwLock};

use arrow_buffer::BooleanBuffer;
use async_once_cell::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::{ContextRef, IntoArrayVariant};
use vortex_dtype::FieldMask;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_mask::Mask;

use crate::layouts::chunked::range_reader::ChunkedRangeReader;
use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutRangeReader, LayoutVTable};

type PruningCache = Arc<OnceCell<Option<BooleanBuffer>>>;

pub struct ChunkedReader {
    layout: Layout,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
    field_mask: Arc<[FieldMask]>,

    /// The cumulative row offsets of each chunk, starting with zero.
    chunk_row_offsets: Vec<u64>,
    /// Cache of chunk readers
    chunk_readers: Vec<OnceLock<Arc<dyn LayoutReader>>>,
    /// State shared across all range readers.
    shared_state: Arc<SharedState>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: Layout,
        ctx: ContextRef,
        field_mask: &[FieldMask],
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != ChunkedLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // The number of chunks
        let mut nchunks = layout.nchildren();
        if layout.metadata().is_some() {
            // The final child is the statistics table.
            nchunks -= 1;
        }

        // Construct a lazy scan for each chunk of the layout.
        let chunk_readers = (0..nchunks).map(|_| OnceLock::new()).collect();

        // Generate the cumulative chunk offsets, relative to the layout's row offset
        let chunk_row_offsets = iter::once(0)
            .chain((0..nchunks).map(|i| layout.child_row_count(i)).scan(
                layout.row_offset(),
                |state, x| {
                    *state += x;
                    Some(*state)
                },
            ))
            .collect();

        // Set up the stats table reader
        let shared_stats = layout
            .metadata()
            .map(|metadata| {
                let present_stats: Arc<[Stat]> = stats_from_bitset_bytes(metadata.as_ref()).into();

                let stats_dtype = StatsTable::dtype_for_stats_table(layout.dtype(), &present_stats);
                let stats_layout = layout.child(nchunks, stats_dtype, 0)?;
                let stats_reader =
                    stats_layout.reader(segments.clone(), ctx.clone(), &[FieldMask::All])?;

                Ok::<_, VortexError>((stats_reader.range_reader(0..nchunks as u64), present_stats))
            })
            .transpose()?;

        Ok(Self {
            layout,
            ctx,
            segments,
            field_mask: field_mask.into(),
            chunk_row_offsets,
            chunk_readers,
            shared_state: Arc::new(SharedState {
                shared_stats,
                stats_table: OnceCell::new(),
                pruning_result: RwLock::new(HashMap::new()),
            }),
        })
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, idx: usize) -> VortexResult<&Arc<dyn LayoutReader>> {
        self.chunk_readers[idx].get_or_try_init(|| {
            let child_layout = self.layout.child(
                idx,
                self.layout.dtype().clone(),
                self.chunk_row_offsets[idx],
            )?;
            child_layout.reader(self.segments.clone(), self.ctx.clone(), &self.field_mask)
        })
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn range_reader(&self, row_range: Range<u64>) -> Arc<dyn LayoutRangeReader> {
        let start_chunk = self
            .chunk_row_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|x| x - 1);
        let end_chunk = self
            .chunk_row_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|x| x);

        let chunks = (start_chunk..end_chunk)
            .map(|i| {
                // Truncate the row range to the chunk.
                // Note that row ranges are not relativized.
                let start = self.chunk_row_offsets[i].max(row_range.start);
                let end = self.chunk_row_offsets[i + 1].min(row_range.end);
                self.child(i)
                    .vortex_expect("not out of bounds")
                    .range_reader(start..end)
            })
            .collect();

        Arc::new(ChunkedRangeReader {
            layout: self.layout.clone(),
            row_range,
            chunk_range: start_chunk..end_chunk,
            chunks,
            shared_state: self.shared_state.clone(),
        }) as _
    }
}

type SharedStats = (Arc<dyn LayoutRangeReader>, Arc<[Stat]>);

/// State that's shared between each chunk range reader.
pub(crate) struct SharedState {
    /// The stats table range reader and present stats for the layout
    shared_stats: Option<SharedStats>,
    /// Shared stats table
    stats_table: OnceCell<Option<StatsTable>>,
    /// A cache of expr -> optional pruning result (applying the pruning expr to the stats table)
    pruning_result: RwLock<HashMap<ExprRef, PruningCache>>,
}

impl SharedState {
    /// Get or initialize the stats table.
    ///
    /// Only the first successful caller will initialize the stats table, all other callers will
    /// resolve to the same result.
    pub(crate) async fn stats_table(&self) -> VortexResult<&Option<StatsTable>> {
        self.stats_table
            .get_or_try_init(async {
                let Some((stats_range_reader, present_stats)) = self.shared_stats.as_ref() else {
                    return Ok(None);
                };

                let stats_array = stats_range_reader
                    .evaluate_expr(
                        Mask::new_true(usize::try_from(stats_range_reader.row_range().end)?),
                        Identity::new_expr(),
                    )
                    .await?;

                Ok(Some(StatsTable::unchecked_new(
                    stats_array,
                    present_stats.clone(),
                )))
            })
            .await
    }

    pub(crate) async fn pruning_mask(&self, expr: &ExprRef) -> VortexResult<Option<BooleanBuffer>> {
        let cell = self
            .pruning_result
            .write()
            .map_err(|_| vortex_err!("poisoned lock"))?
            .entry(expr.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        cell.get_or_try_init(async {
            let pruning_predicate = PruningPredicate::try_new(expr);
            Ok(if let Some(stats_table) = self.stats_table().await? {
                if let Some(predicate) = pruning_predicate {
                    predicate
                        .evaluate(stats_table.array())?
                        .map(|mask| mask.into_bool())
                        .transpose()?
                        .map(|mask| mask.boolean_buffer())
                } else {
                    None
                }
            } else {
                None
            })
        })
        .await
        .cloned()
    }
}
