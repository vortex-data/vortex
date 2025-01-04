use std::sync::{Arc, RwLock};

use vortex_array::array::ChunkedArray;
use vortex_array::stats::{stats_from_bitset_bytes, Stat};
use vortex_array::{ArrayData, ContextRef, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::layouts::chunked::stats::StatsTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::scanner::{LayoutScan, Poll, Scan, Scanner};
use crate::segments::{SegmentId, SegmentReader};
use crate::{LayoutData, LayoutEncoding, RowMask};

pub struct ChunkedScan {
    layout: LayoutData,
    scan: Scan,
    dtype: DType,
    ctx: ContextRef,
    stats: Option<Arc<RwLock<ChunkedStats>>>,
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

        // If we have a filter expression and a stats table, then create a scanner for it.
        let stats = match (layout.metadata(), &scan.filter) {
            (Some(metadata), Some(_expr)) => {
                let present_stats = stats_from_bitset_bytes(metadata.as_ref());
                let stats_dtype = StatsTable::dtype_for_stats_table(layout.dtype(), &present_stats);
                let stats_layout = layout.child(layout.nchildren() - 1, stats_dtype)?;

                let stats_count = stats_layout.row_count();
                debug_assert_eq!(
                    stats_count,
                    layout.nchildren() as u64 - 1,
                    "Stats count should match chunk count"
                );

                // TODO(ngates): scan only the columns of the stats table that are referenced by the
                //  filter expression. This doesn't matter at the moment since we always store the stats
                //  with a FlatLayout.
                let scanner = stats_layout
                    .new_scan(Scan::all(), ctx.clone())?
                    .create_scanner(RowMask::new_valid_between(0, stats_count))?;

                Some(Arc::new(RwLock::new(ChunkedStats {
                    present_stats,
                    scanner,
                    table: None,
                })))
            }
            _ => None,
        };

        // Compute the dtype of the scan.
        let dtype = scan.result_dtype(layout.dtype())?;

        Ok(Self {
            layout,
            scan,
            dtype,
            ctx,
            stats,
        })
    }

    /// Returns the number of chunks in the layout.
    fn nchunks(&self) -> usize {
        let mut nchildren = self.layout.nchildren();
        if self.layout.metadata().is_some() {
            // The final child is the statistics table.
            nchildren -= 1;
        }
        nchildren
    }
}

impl LayoutScan for ChunkedScan {
    fn layout(&self) -> &LayoutData {
        &self.layout
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Note that a [`Scanner`] is intended to return a single batch of data, therefore instead
    /// of reading chunks one by one, we attempt to make progress on reading all chunks at the same
    /// time. We therefore do as much pruning as we can now and then read all chunks in parallel.
    fn create_scanner(&self, mask: RowMask) -> VortexResult<Box<dyn Scanner>> {
        let mut chunk_scanners = Vec::with_capacity(self.layout.nchildren());

        let mut row_start = 0;
        for chunk_idx in 0..self.nchunks() {
            let chunk_range = row_start..row_start + self.layout().child_row_count(chunk_idx);
            row_start = chunk_range.end;

            if mask.is_disjoint(chunk_range.clone()) {
                // Skip this chunk if it's not in the mask.
                continue;
            }

            // We defer creating the chunk layout until here. If we only ever read a few rows,
            // this is good thing. However, this layout therefore isn't shared across scanners in
            // case a single chunk overlaps multiple row masks. We could I suppose store OnceCell
            // for each chunk layout in the ChunkedScan, but let's wait to see if it becomes
            // necessary.
            let chunk_layout = self
                .layout
                .child(chunk_idx, self.layout.dtype().clone())
                .vortex_expect("Child index out of bound");
            let chunk_scan = chunk_layout.new_scan(self.scan.clone(), self.ctx.clone())?;
            let chunk_mask = mask
                .clone()
                // TODO(ngates): I would have thought slice would also shift?
                .slice(chunk_range.start, chunk_range.end)?
                .shift(chunk_range.start)?;
            let chunk_scanner = chunk_scan.create_scanner(chunk_mask)?;
            chunk_scanners.push(chunk_scanner);
        }

        Ok(Box::new(ChunkedScanner {
            layout_dtype: self.layout.dtype().clone(),
            scan_dtype: self.dtype.clone(),
            chunk_scanners,
            chunk_state: vec![ChunkState::Initial; self.nchunks()],
            stats: self.stats.clone(),
            stats_table: None,
        }) as _)
    }
}

/// A scanner for a chunked layout.
struct ChunkedScanner {
    layout_dtype: DType,
    scan_dtype: DType,
    chunk_scanners: Vec<Box<dyn Scanner>>,
    chunk_state: Vec<ChunkState>,
    stats: Option<Arc<RwLock<ChunkedStats>>>,
    stats_table: Option<StatsTable>,
}

#[derive(Clone)]
enum ChunkState {
    // In the initial state, we check whether the chunk can be skipped using the stats table.
    Initial,
    // While we poll the chunk, we mark it as pending.
    Pending,
    // If the chunk was skipped with stats, we mark it as None. Otherwise, we include the chunk.
    Resolved(Option<ArrayData>),
}

impl Scanner for ChunkedScanner {
    fn poll(&mut self, segments: &dyn SegmentReader) -> VortexResult<Poll> {
        // First, we try to fetch the stats table.
        match self.update_stats_table(segments)? {
            PollStats::None => {}
            PollStats::Some(stats_table) => {
                self.stats_table = Some(stats_table);
            }
            PollStats::NeedMore(segments) => return Ok(Poll::NeedMore(segments)),
        }

        // Now we try to read the chunks.
        let mut needed = vec![];
        for (chunk_idx, chunk) in self.chunk_scanners.iter_mut().enumerate() {
            if matches!(self.chunk_state[chunk_idx], ChunkState::Initial) {
                // If the chunk can be skipped using the stats_table, then mark it as empty.
                // TODO
                // Otherwise, we proceed to pending
                self.chunk_state[chunk_idx] = ChunkState::Pending;
            }

            match self.chunk_state[chunk_idx] {
                ChunkState::Initial => unreachable!(),
                ChunkState::Pending => {
                    // Otherwise, poll the chunk.
                    match chunk.poll(segments)? {
                        Poll::Some(array) => {
                            // Resolve the chunk
                            self.chunk_state[chunk_idx] = ChunkState::Resolved(Some(array));
                        }
                        Poll::NeedMore(segment_ids) => {
                            needed.extend(segment_ids);
                        }
                    }
                }
                ChunkState::Resolved(_) => {
                    // The chunk is already resolved
                }
            }
        }

        // If we need more segments, then request them.
        if !needed.is_empty() {
            return Ok(Poll::NeedMore(needed));
        }

        // Otherwise, we've read all the chunks, so we're done.
        let chunks = self
            .chunk_state
            .iter_mut()
            .filter_map(|state| match state {
                ChunkState::Resolved(array) => array.take(),
                _ => vortex_panic!(
                    "This is a bug. Missing a chunk array with no more segments to read"
                ),
            })
            .collect::<Vec<_>>();

        Ok(Poll::Some(
            ChunkedArray::try_new(chunks, self.scan_dtype.clone())?.into_array(),
        ))
    }
}

enum PollStats {
    None,
    Some(StatsTable),
    NeedMore(Vec<SegmentId>),
}

impl ChunkedScanner {
    /// Attempt to populate the stats table if not already set.
    /// Returns segments if required, otherwise the table should be considered up-to-date.
    fn update_stats_table(&mut self, segments: &dyn SegmentReader) -> VortexResult<PollStats> {
        let Some(stats) = &self.stats else {
            // If there is no stats layout to poll from, then we know we have no stats.
            self.stats_table.replace(StatsTable::no_stats(
                self.layout_dtype.clone(),
            ))
            // No stats to update.
            return Ok(PollStats::None);
        };

        // Grab a cheap read lock to check if the table is already set
        if let Some(stats_table) = &stats.read().map_err(|_| vortex_err!("poisoned"))?.table {
            // The table is already set, so we're done.
            return Ok(PollStats::Some(stats_table.clone()));
        }

        // Otherwise, grab a write lock and poll the stats table scanner.
        let mut stats = stats.write().map_err(|_| vortex_err!("poisoned"))?;
        match stats.scanner.poll(segments)? {
            // While we hold the lock, we update the stats table to avoid other threads
            // from trying to update the OnceCell.
            Poll::Some(stats_array) => {
                let present_stats = stats.present_stats.clone();
                let stats_table =
                    StatsTable::try_new(self.layout_dtype.clone(), stats_array, present_stats)?;

                // Update the shared stats
                stats.table.replace(stats_table.clone());

                // And return it
                Ok(PollStats::Some(stats_table))
            }
            // Otherwise, we return the segments needed to read the stats table.
            Poll::NeedMore(needed) => Ok(PollStats::NeedMore(needed)),
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::{ArrayDType, ArrayLen, IntoArrayData, IntoArrayVariant};
    use vortex_buffer::buffer;

    use crate::layouts::chunked::writer::ChunkedLayoutWriter;
    use crate::scanner::{Poll, Scan};
    use crate::segments::test::TestSegments;
    use crate::strategies::LayoutWriterExt;
    use crate::{LayoutData, RowMask};

    /// Create a chunked layout with three chunks of `1..=3` primitive arrays.
    fn chunked_layout() -> (TestSegments, LayoutData) {
        let arr = buffer![1, 2, 3].into_array();
        let mut segments = TestSegments::default();
        let layout = ChunkedLayoutWriter::new(arr.dtype(), Default::default())
            .push_all(
                &mut segments,
                [Ok(arr.clone()), Ok(arr.clone()), Ok(arr.clone())],
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
        assert_eq!(result.as_slice::<i32>(), &[1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_chunked_scan_pruned() {
        let (_segments, layout) = chunked_layout();

        let scan = layout.new_scan(Scan::all(), Default::default()).unwrap();
        let full_mask = RowMask::new_valid_between(0, scan.layout().row_count());

        // We poll scanners with empty segments to count how many segments they would need.
        let mut full_scanner = scan.create_scanner(full_mask.clone()).unwrap();
        let Poll::NeedMore(full_segments) = full_scanner.poll(&TestSegments::default()).unwrap()
        else {
            unreachable!()
        };

        // Using a row-mask that only covers one chunk should have 3x fewer segments.
        let mut one_chunk_scanner = scan.create_scanner(full_mask.slice(0, 3).unwrap()).unwrap();
        let Poll::NeedMore(one_segments) =
            one_chunk_scanner.poll(&TestSegments::default()).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(one_segments.len() * 3, full_segments.len());

        // Using a row-mask that covers two chunks should have 2/3 the full_segments.
        let mut two_chunk_scanner = scan.create_scanner(full_mask.slice(2, 5).unwrap()).unwrap();
        let Poll::NeedMore(two_segments) =
            two_chunk_scanner.poll(&TestSegments::default()).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(two_segments.len(), one_segments.len() * 2);
    }
}
