use std::ops::Range;
use std::rc::Weak;
use std::sync::{Arc, OnceLock, RwLock};

use arrow_buffer::BooleanBuffer;
use async_once_cell::OnceCell;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::stats::stats_from_bitset_bytes;
use vortex_array::{ContextRef, IntoArrayVariant};
use vortex_dtype::{FieldMask, FieldPath};
use vortex_error::{vortex_bail, vortex_err, vortex_panic, VortexExpect, VortexResult};
use vortex_expr::pruning::PruningPredicate;
use vortex_expr::{ExprRef, Identity};
use vortex_scan::RowMask;

use crate::layouts::chunked::range_reader::ChunkedRangeReader;
use crate::layouts::chunked::stats_table::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{ExprEvaluator, Layout, LayoutRangeReader, LayoutVTable};

type PruningCache = Arc<OnceCell<Option<BooleanBuffer>>>;

#[derive(Clone)]
pub struct ChunkedReader {
    layout: Layout,
    shared_state: Arc<SharedState>,
    /// Shared lazy chunk scanners
    chunk_readers: Arc<[OnceLock<Arc<dyn LayoutReader>>]>,
    /// The cumulative row offsets of each chunk, starting with zero.
    chunk_row_offsets: Vec<u64>,
}

impl ChunkedReader {
    pub(super) fn try_new(
        layout: Layout,
        ctx: ContextRef,
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

        // Generate the cumulative chunk offsets
        let chunk_row_offsets = (0..nchunks).map(|i| layout.child_row_count(i)).collect();

        Ok(Self {
            layout: layout.clone(),
            shared_state: Arc::new(SharedState {
                layout: layout.clone(),
                ctx,
                segments,
                stats_table: OnceCell::new(),
                pruning_result: RwLock::new(HashMap::new()),
            }),
            chunk_readers,
            chunk_row_offsets,
        })
    }

    /// Return the number of chunks
    pub(crate) fn nchunks(&self) -> usize {
        self.chunk_readers.len()
    }

    /// Return the child reader for the chunk.
    pub(crate) fn child(&self, idx: usize) -> VortexResult<&Arc<dyn LayoutReader>> {
        self.chunk_readers[idx].get_or_try_init(|| {
            let child_layout = self.layout.child(
                idx,
                self.layout.dtype().clone(),
                self.chunk_row_offsets[idx],
            )?;
            child_layout.reader(
                self.shared_state.segments.clone(),
                self.shared_state.ctx.clone(),
            )
        })
    }
}

impl LayoutReader for ChunkedReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn range_reader(
        &self,
        row_range: Range<u64>,
        field_mask: Arc<[FieldMask]>,
    ) -> Arc<dyn LayoutRangeReader> {
        let start_chunk = self
            .chunk_row_offsets
            .binary_search(&row_range.start)
            .unwrap_or_else(|x| x);
        let end_chunk = self
            .chunk_row_offsets
            .binary_search(&row_range.end)
            .unwrap_or_else(|x| x);
        let start_chunk_offset =
            usize::try_from(row_range.start - self.chunk_row_offsets[start_chunk])
                .vortex_expect("Chunk offset must fit within usize");
        let end_chunk_length =
            usize::try_from(row_range.end - self.chunk_row_offsets[end_chunk + 1])
                .vortex_expect("Chunk length must fit within usize");

        let chunks = (start_chunk..end_chunk)
            .map(|i| {
                // Truncate the row range to the chunk.
                // Note that row ranges are not relativized.
                let row_range = if i == start_chunk {
                    row_range.start..self.chunk_row_offsets[i + 1]
                } else if i == end_chunk {
                    self.chunk_row_offsets[i]..row_range.end
                } else {
                    self.chunk_row_offsets[i]..self.chunk_row_offsets[i + 1]
                };
                self.child(i)
                    .vortex_expect("not out of bounds")
                    .range_reader(row_range, field_mask.clone())
            })
            .collect();

        Arc::new(ChunkedRangeReader {
            chunk_range: start_chunk..end_chunk,
            chunks,
            shared_state: self.shared_state.clone(),
        }) as _
    }
}

/// State that's shared between each chunk range reader.
///
// TODO(ngates): we could make this use Lazy instead of OnceCell. The errors are worse, but the
//  execution model is better?
pub(crate) struct SharedState {
    layout: Layout,
    ctx: ContextRef,
    segments: Arc<dyn AsyncSegmentReader>,
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
    pub(crate) async fn stats_table(&self) -> VortexResult<Option<&StatsTable>> {
        self.stats_table
            .get_or_try_init(async {
                Ok(match self.layout.metadata() {
                    None => None,
                    Some(metadata) => {
                        // The final child is the statistics table.
                        let nchunks = self.layout.nchildren() - 1;

                        // Figure out which stats are present
                        let present_stats = stats_from_bitset_bytes(metadata.as_ref());

                        let layout_dtype = self.layout.dtype();
                        let stats_dtype =
                            StatsTable::dtype_for_stats_table(layout_dtype, &present_stats);
                        let stats_layout = self.layout.child(nchunks, stats_dtype.clone(), self.layout.row_offset())?;

                        let stats_array = stats_layout
                            .reader(self.segments.clone(), self.ctx.clone())?
                            .evaluate_expr(
                                RowMask::new_valid_between(0, nchunks as u64),
                                Identity::new_expr(),
                            )
                            .await?;

                        if &stats_dtype != stats_array.dtype() {
                            vortex_bail!("Expected stats DType {stats_dtype} doesn't match read stats dtype {}", stats_array.dtype())
                        }

                        // SAFETY: This is only fine to call because we perform validation above
                        Some(StatsTable::unchecked_new(
                            stats_array,
                            present_stats.into(),
                        ))
                    }
                })
            })
            .await
            .map(|opt| opt.as_ref())
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
