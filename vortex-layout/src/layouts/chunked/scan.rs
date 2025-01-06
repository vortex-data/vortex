use std::sync::{Arc, OnceLock, RwLock};

use arrow_buffer::BooleanBuffer;
use vortex_array::array::{ChunkedArray, StructArray};
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, ContextRef, IntoArrayData, IntoArrayVariant};
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};
use vortex_expr::pruning::PruningPredicate;

use crate::layouts::chunked::stats::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::scanner::{LayoutScan, Poll, ResolvedScanner, Scan, Scanner, ScannerExt};
use crate::segments::{SegmentId, SegmentReader};
use crate::{LayoutData, LayoutEncoding, RowMask};

/// Captures the scan state of a chunked layout.
///
/// This scan is used to generate multiple scanners, one per row-range. It may even be the case
/// that the caller requests different row ranges for filter operations as for projection
/// operations. As such, it's beneficial for us to re-use some state across different scanners.
///
/// The obvious state to re-use is that of the statistics table. Each range scan polls the same
/// underlying statistics scanner since a scanner must continue to return its result for subsequent
/// polls.
///
/// There is a question about whether we should lazily construct and share all chunk scanners too.
/// We currently create a new chunk scanner for every chunk read by each range scan. This is sort
/// of fine if we assume row ranges are non-overlapping (although if a range overlaps a chunk
/// boundary we will read the chunk twice). However, if we have overlapping row ranges, as can
/// happen if the parent is performing multiple scans (filter + projection), then we may read the
/// same chunk many times.
#[derive(Clone, Debug)]
pub struct ChunkedScan {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
    ctx: ContextRef,
    // Shared stats table scanner
    stats_scanner: Arc<RwLock<Box<dyn Scanner>>>,
    // Cached pruning mask for the scan
    pruning_mask: Arc<OnceLock<Option<BooleanBuffer>>>,
    // Shared lazy chunk scanners
    chunk_scans: Arc<[OnceLock<Box<dyn LayoutScan>>]>,
    // The stats that are present in the layout
    present_stats: Arc<[Stat]>,
}

struct ChunkedStats {
    present_stats: Vec<Stat>,
    scanner: Box<dyn Scanner>,
    // Cached stats table once it's been read from the scanner.
    table: Option<StatsTable>,
}

impl ChunkedScan {
    pub(super) fn try_new(layout: LayoutData, scan: Scan, ctx: ContextRef) -> VortexResult<Self> {
        if layout.encoding().id() != ChunkedLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        // The number of chunks
        let mut nchunks = layout.nchildren();
        if layout.metadata().is_some() {
            // The final child is the statistics table.
            nchunks -= 1;
        }

        // Figure out which stats are present
        let present_stats: Arc<[Stat]> = layout
            .metadata()
            .map(|m| stats_from_bitset_bytes(m.as_ref()))
            .unwrap_or_default()
            .into();

        let stats_scanner = layout
            .metadata()
            .is_some()
            .then(|| {
                let stats_dtype = StatsTable::dtype_for_stats_table(layout.dtype(), &present_stats);
                let stats_layout = layout.child(layout.nchildren() - 1, stats_dtype)?;
                stats_layout
                    .new_scan(Scan::all(), ctx.clone())?
                    .create_scanner(RowMask::new_valid_between(0, nchunks as u64))
            })
            .transpose()?
            .unwrap_or_else(|| {
                // Otherwise we create a default stats array with no columns.
                ResolvedScanner(
                    StructArray::try_new(vec![].into(), vec![], nchunks, Validity::NonNullable)
                        .vortex_expect("cannot fail")
                        .into_array(),
                )
                .boxed()
            });

        // Construct a lazy scan for each chunk of the layout.
        let chunk_scans = (0..nchunks).map(|_| OnceLock::new()).collect();

        // Compute the dtype of the scan.
        let dtype = scan.result_dtype(layout.dtype())?;

        Ok(Self {
            layout,
            scan,
            dtype,
            ctx,
            stats_scanner: Arc::new(RwLock::new(stats_scanner)),
            pruning_mask: Arc::new(OnceLock::new()),
            chunk_scans,
            present_stats,
        })
    }
}

impl LayoutScan for ChunkedScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        Ok(Box::new(ChunkedScanner {
            chunked_scan: self.clone(),
            mask,
            chunk_states: None,
        }) as _)
    }
}

/// A scanner for a chunked layout.
#[derive(Debug)]
struct ChunkedScanner {
    chunked_scan: ChunkedScan,
    mask: RowMask,
    // State for each chunk in the layout
    chunk_states: Option<Vec<ChunkState>>,
}

#[derive(Debug)]
enum ChunkState {
    Pending(Box<dyn Scanner>),
    Resolved(Option<ArrayData>),
}

impl Scanner for ChunkedScanner {
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll> {
        // If we haven't set up our chunk state yet, then fetch the stats table and do so.
        if self.chunk_states.is_none() {
            // First, we grab the stats table
            let stats_table = match self
                .chunked_scan
                .stats_scanner
                .write()
                .map_err(|_| vortex_err!("poisoned"))?
                .poll(segments)?
            {
                Poll::Some(stats_array) => StatsTable::try_new(
                    self.chunked_scan.layout.dtype().clone(),
                    stats_array,
                    self.chunked_scan.present_stats.clone(),
                )?,
                Poll::NeedMore(segments) => {
                    // Otherwise, we need more segments to read the stats table.
                    return Ok(Poll::NeedMore(segments));
                }
            };

            // And compute the pruning predicate
            let pruning_mask = self.chunked_scan.pruning_mask.get_or_try_init(|| {
                Ok::<_, VortexError>(
                    self.chunked_scan
                        .scan
                        .filter
                        .as_ref()
                        .and_then(PruningPredicate::try_new)
                        .and_then(|predicate| predicate.evaluate(stats_table.array()).transpose())
                        .transpose()?
                        .map(|mask| mask.into_bool())
                        .transpose()?
                        .map(|mask| mask.boolean_buffer()),
                )
            })?;

            // Now we can set up the chunk state.
            let mut chunks = Vec::with_capacity(self.chunked_scan.chunk_scans.len());
            let mut row_offset = 0;
            for chunk_idx in 0..self.chunked_scan.chunk_scans.len() {
                let chunk_scan =
                    self.chunked_scan.chunk_scans[chunk_idx].get_or_try_init(|| {
                        self.chunked_scan
                            .layout
                            .child(chunk_idx, self.chunked_scan.layout.dtype().clone())
                            .vortex_expect("Child index out of bounds")
                            .new_scan(
                                self.chunked_scan.scan.clone(),
                                self.chunked_scan.ctx.clone(),
                            )
                    })?;

                // Figure out the row range of the chunk
                let chunk_len = chunk_scan.layout().row_count();
                let chunk_range = row_offset..row_offset + chunk_len;
                row_offset += chunk_len;

                // Try to skip the chunk based on the row-mask
                if self.mask.is_disjoint(chunk_range.clone()) {
                    chunks.push(ChunkState::Resolved(None));
                    continue;
                }

                // Try to skip the chunk based on the pruning predicate
                if let Some(pruning_mask) = pruning_mask {
                    if pruning_mask.value(chunk_idx) {
                        chunks.push(ChunkState::Resolved(None));
                        continue;
                    }
                }

                // Otherwise, we need to read it. So we set up a mask for the chunk range.
                let chunk_mask = self
                    .mask
                    .slice(chunk_range.start, chunk_range.end)?
                    .shift(chunk_range.start)?;
                chunks.push(ChunkState::Pending(chunk_scan.create_scanner(chunk_mask)?));
            }

            self.chunk_states = Some(chunks);
        }

        let chunk_states = self
            .chunk_states
            .as_mut()
            .vortex_expect("chunk state not set");

        // Now we try to read the chunks.
        let mut needed = vec![];
        for chunk_state in chunk_states.iter_mut() {
            match chunk_state {
                ChunkState::Pending(scanner) => match scanner.poll(segments)? {
                    Poll::Some(array) => {
                        // Resolve the chunk
                        *chunk_state = ChunkState::Resolved(Some(array));
                    }
                    Poll::NeedMore(segment_ids) => {
                        // Request more segments
                        needed.extend(segment_ids);
                    }
                },
                ChunkState::Resolved(_) => {
                    // Already resolved
                }
            }
        }

        // If we need more segments, then request them.
        if !needed.is_empty() {
            return Ok(Poll::NeedMore(needed));
        }

        // Otherwise, we've read all the chunks, so we're done.
        let chunks = chunk_states
            .iter_mut()
            .filter_map(|state| match state {
                ChunkState::Resolved(array) => array.take(),
                _ => vortex_panic!(
                    "This is a bug. Missing a chunk array with no more segments to read"
                ),
            })
            .collect::<Vec<_>>();

        Ok(Poll::Some(
            ChunkedArray::try_new(chunks, self.chunked_scan.dtype.clone())?.into_array(),
        ))
    }
}

enum PollStats {
    None,
    Some(StatsTable),
    NeedMore(Vec<SegmentId>),
}

#[cfg(test)]
mod test {
    use std::assert_matches::assert_matches;

    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};
    use vortex_expr::{BinaryExpr, Identity, Literal, Operator};

    use crate::layouts::chunked::scan::{ChunkState, ChunkedScan, ChunkedScanner};
    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::scanner::{Poll, Scan, Scanner};
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::{LayoutData, RowMask};

    /// Create a chunked layout with three chunks of primitive arrays.
    fn chunked_layout() -> (TestSegments, LayoutData) {
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(
            &DType::Primitive(PType::I32, NonNullable),
            Default::default(),
        )
        .push_all(
            &mut segments,
            [
                Ok(buffer![1, 2, 3].into_array()),
                Ok(buffer![4, 5, 6].into_array()),
                Ok(buffer![7, 8, 9].into_array()),
            ],
        )
        .unwrap();
        (segments, layout)
    }

    #[test]
    fn test_chunked_scan() {
        let (segments, layout) = chunked_layout();

        let scan = layout.new_scan(Scan::all(), Default::default()).unwrap();
        let result = segments.do_scan(scan.as_ref()).into_primitive().unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_chunked_pruning_mask() {
        let (segments, layout) = chunked_layout();

        let scan = ChunkedScan::try_new(
            layout,
            Scan {
                projection: Identity::new_expr(),
                filter: Some(BinaryExpr::new_expr(
                    Identity::new_expr(),
                    Operator::Gt,
                    Literal::new_expr(6.into()),
                )),
            },
            Default::default(),
        )
        .unwrap();

        // Populate the stats scanner so that we can compute the pruning mask
        _ = scan.stats_scanner.write().unwrap().poll(&segments).unwrap();

        let mut scanner = ChunkedScanner {
            chunked_scan: scan,
            mask: RowMask::new_valid_between(0, 9),
            chunk_states: None,
        };

        // Then we poll the chunked scanner without any segments so _only_ the stats were
        // available.
        let Poll::NeedMore(_segments) = scanner.poll(&TestSegments::default()).unwrap() else {
            unreachable!()
        };

        // Now we validate that based on the pruning mask, we have excluded the first two chunks
        let chunk_states = scanner.chunk_states.as_ref().unwrap().as_slice();
        assert_matches!(chunk_states[0], ChunkState::Resolved(None));
        assert_matches!(chunk_states[1], ChunkState::Resolved(None));
        assert_matches!(chunk_states[2], ChunkState::Pending(_));
    }
}
