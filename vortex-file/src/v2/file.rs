use std::io::Read;
use std::ops::Range;
use std::sync::Arc;

use futures_util::stream;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::operations::{Operation, Poll};
use vortex_layout::{LayoutData, LayoutReader};
use vortex_scan::Scan;

use crate::v2::footer::Segment;
use crate::v2::segments::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) read: R,
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: Vec<Segment>,
    pub(crate) segment_cache: SegmentCache,
    // TODO(ngates): not yet used by the file reader
    #[allow(dead_code)]
    pub(crate) splits: Vec<Range<u64>>,
}

impl<R> VortexFile<R> {}

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
    pub fn scan(&self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + '_> {
        // Create a shared reader for the scan.
        let reader: Arc<dyn LayoutReader> = self.layout.reader(self.ctx.clone())?;
        let result_dtype = scan.result_dtype(self.dtype())?;

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let stream = stream::once(async move {
            let row_range = 0..reader.layout().row_count();
            let mut range_scan = reader.range_scan(scan.range_scan(row_range)?);

            loop {
                match range_scan.poll(&self.segment_cache)? {
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

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}

/// Sync implementation of Vortex File.
impl<R: Read> VortexFile<R> {}
