use std::io::Read;

use futures_util::stream;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::operations::Poll;
use vortex_layout::scanner::Scan;
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
        let scan_dtype = scan.result_dtype(self.layout.dtype())?;
        let layout_scan = self.layout.reader(scan, self.ctx.clone())?;

        let scan = Scan::new(filter, projection, row_mask);

        row_range
            .par_iter()
            .map(|row_range| {
                let range_scan = scan.range_scan(row_range)
                let evaluators = range_scan.exprs()
                    .iter()
                    .map(|expr| layout.new_evaluator(expr, ctx.clone()))
                    .collect();

                let expr = range_scan.next_expr();
                range_scan.push_result(array);
                range_scan.next_expr();

            })

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let row_mask = RowMask::new_valid_between(0, layout_scan.layout().row_count());
        let mut scanner = layout_scan.create_eval(row_mask)?;

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
