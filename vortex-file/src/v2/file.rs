use std::io::Read;

use futures_util::stream;
use vortex_array::compute::FilterMask;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::scanner::{Poll, Scan};
use vortex_layout::{LayoutData, RowMask};

use crate::v2::footer::{FileLayout, Segment};
use crate::v2::segments::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) read: R,
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: Vec<Segment>,
    pub(crate) segment_cache: SegmentCache,
}

impl<R> VortexFile<R> {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    /// Returns the [`DType`] of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Returns the [`FileLayout`] of the file.
    ///
    /// This can be passed to [`vortex_file::v2::VortexOpenOptions`] to reconstruct a
    /// [`VortexFile`] without re-reading the footer.
    pub fn file_layout(&self) -> FileLayout {
        FileLayout {
            root_layout: self.layout.clone(),
            segments: self.segments.clone(),
        }
    }
}

/// Async implementation of Vortex File.
impl<R: VortexReadAt> VortexFile<R> {
    /// Performs a scan operation over the file.
    pub fn scan(self, scan: Scan) -> VortexResult<impl ArrayStream + 'static> {
        let row_count = self.row_count();

        // TODO(ngates): support thread pool execution.
        //  The plan is to have OpenOptions configure a Rayon ThreadPool for reading. We would
        //  par_iter each of the row masks (based on configured split by size or row count),
        //  launching their `poll` operation onto the thread pool. If a task returns NeedMore,
        //  then the segment IDs are handed to the IO dispatcher and a synchronous latch is
        //  returned. The IO dispatcher has visibility into all requested segments and can perform
        //  coalescing over ranges. Once a coalesced read returns, the dispatcher updates the
        //  segment cache with all read segments (including those that were incidentally read by
        //  in-between the coalesced ranges). A map of segment IDs -> set<latch> then provides
        //  a way for the dispatcher to notify the waiting tasks that their data is ready. When
        //  finished, the tasks push their results in order onto a channel that acts as the
        //  ArrayStream.
        //  This keeps I/O on the current thread (using the caller's existing runtime), while still
        //  enabling a CPU pool for decompression and filtering.

        self.scan_range(scan, RowMask::new_valid_between(0, row_count))
    }

    /// Performs a scan operation over the file.
    pub fn scan_rows<I: IntoIterator<Item = u64>>(
        self,
        scan: Scan,
        indices: I,
    ) -> VortexResult<impl ArrayStream + 'static> {
        let row_count = self.row_count();

        // TODO(ngates): do we only support "take" over usize rows?
        let filter_mask = FilterMask::from_indices(
            usize::try_from(row_count).expect("row count is too large for usize"),
            indices
                .into_iter()
                .map(|i| usize::try_from(i).expect("row index is too large for usize")),
        );
        let row_mask = RowMask::try_new(filter_mask, 0, row_count)?;

        self.scan_range(scan, row_mask)
    }

    /// Performs a scan operation over a [`RowMask`] of the file.
    fn scan_range(self, scan: Scan, row_mask: RowMask) -> VortexResult<impl ArrayStream + 'static> {
        let layout_scan = self.layout.new_scan(scan, self.ctx.clone())?;
        let scan_dtype = layout_scan.dtype().clone();

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let mut scanner = layout_scan.create_scanner(row_mask)?;

        let stream = stream::once(async move {
            let segment_cache = self.segment_cache.clone();
            let segments = self.segments.clone();
            loop {
                match scanner.poll(&segment_cache)? {
                    Poll::Some(array) => return Ok(array),
                    Poll::NeedMore(segment_ids) => {
                        for segment_id in segment_ids {
                            let segment = &segments[*segment_id as usize];
                            let bytes = self
                                .read
                                .read_byte_range(segment.offset, segment.length as u64)
                                .await?;
                            segment_cache.set(segment_id, bytes);
                        }
                    }
                }
            }
        });

        Ok(ArrayStreamAdapter::new(scan_dtype, stream))
    }
}

/// Sync implementation of Vortex File.
impl<R: Read> VortexFile<R> {}
