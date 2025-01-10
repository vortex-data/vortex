use std::ops::Range;
use std::sync::Arc;
use std::task::Poll;

use futures::pin_mut;
use futures_util::future::poll_fn;
use futures_util::{stream, TryFutureExt};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::VortexReadAt;
use vortex_layout::{ExprEvaluator, LayoutData, LayoutReader};
use vortex_scan::Scan;

use crate::v2::segments::cache::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: Arc<SegmentCache<R>>,
    // TODO(ngates): not yet used by the file reader
    #[allow(dead_code)]
    pub(crate) splits: Arc<[Range<u64>]>,
}

impl<R> VortexFile<R> {}

/// Async implementation of Vortex File.
impl<R: VortexReadAt + Unpin> VortexFile<R> {
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
        let reader: Arc<dyn LayoutReader> = self
            .layout
            .reader(self.segments.clone(), self.ctx.clone())?;
        let result_dtype = scan.result_dtype(self.dtype())?;
        // For each row-group, we set up a future that will evaluate the scan and post its.

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let stream = stream::once(async move {
            // TODO(ngates): we should launch the evaluate_async onto a worker thread pool.
            let row_range = 0..self.layout.row_count();

            let eval = scan
                .range_scan(row_range)?
                .evaluate_async(|row_mask, expr| reader.evaluate_expr(row_mask, expr));
            pin_mut!(eval);

            poll_fn(|cx| {
                // Now we alternate between polling the eval task and driving the I/O.
                loop {
                    if let Poll::Ready(array) = eval.try_poll_unpin(cx) {
                        return Poll::Ready(array);
                    }
                    let drive = self.segments.drive();
                    pin_mut!(drive);
                    match drive.try_poll_unpin(cx) {
                        Poll::Ready(_) => {}
                        Poll::Pending => return Poll::Pending,
                    }
                }
            })
            .await
        });

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}
