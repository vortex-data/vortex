use std::io::Read;

use futures_util::stream;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::scanner::{Poll, Scan};
use vortex_layout::{LayoutData, RowMask};

use crate::v2::footer::Segment;
use crate::v2::segments::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) read: R,
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: Vec<Segment>,
    pub(crate) segment_cache: SegmentCache,
}

/// Async implementation of Vortex File.
impl<R: VortexReadAt> VortexFile<R> {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    /// Returns the DType of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Performs a scan operation over the file.
    pub fn scan(&self, scan: Scan) -> VortexResult<impl ArrayStream + '_> {
        let layout_scan = self.layout.new_scan(scan, self.ctx.clone())?;
        let scan_dtype = layout_scan.dtype().clone();

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let row_mask = RowMask::new_valid_between(0, layout_scan.layout().row_count());
        let mut scanner = layout_scan.create_scanner(row_mask)?;

        let stream = stream::once(async move {
            loop {
                match scanner.poll(&self.segment_cache)? {
                    Poll::Some(array) => return Ok(array),
                    Poll::NeedMore(segment_ids) => {
                        for segment_id in segment_ids {
                            let segment = &self.segments[*segment_id as usize];
                            let bytes = self
                                .read
                                .read_byte_range(segment.offset, segment.length as u64)
                                .await?;
                            self.segment_cache.set(segment_id, bytes);
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
