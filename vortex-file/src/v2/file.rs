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
use vortex_layout::{ExprEvaluator, LayoutReader};
use vortex_scan::Scan;

use crate::v2::footer::FileLayout;
use crate::v2::segments::cache::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) ctx: ContextRef,
    pub(crate) file_layout: FileLayout,
    pub(crate) segments: Arc<SegmentCache<R>>,
    // TODO(ngates): not yet used by the file reader
    #[allow(dead_code)]
    pub(crate) splits: Arc<[Range<u64>]>,
}

impl<R> VortexFile<R> {}

/// When the underling `R` is `Clone`, we can clone the `VortexFile`.
// TODO(ngates): remove Clone from VortexReadAt?
impl<R: Clone> Clone for VortexFile<R> {
    fn clone(&self) -> Self {
        Self {
            ctx: self.ctx.clone(),
            file_layout: self.file_layout.clone(),
            segments: self.segments.clone(),
            splits: self.splits.clone(),
        }
    }
}

/// Async implementation of Vortex File.
impl<R: VortexReadAt + Unpin> VortexFile<R> {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.file_layout.row_count()
    }

    /// Returns the DType of the file.
    pub fn dtype(&self) -> &DType {
        self.file_layout.dtype()
    }

    /// Returns the [`FileLayout`] of the file.
    pub fn file_layout(&self) -> &FileLayout {
        &self.file_layout
    }

    /// Performs a scan operation over the file. (TODO(ngates): remove the `Send`)
    pub fn scan(&self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + Send + '_> {
        // Create a shared reader for the scan.
        let reader: Arc<dyn LayoutReader> = self
            .file_layout
            .root_layout
            .reader(self.segments.clone(), self.ctx.clone())?;
        let result_dtype = scan.result_dtype(self.dtype())?;
        // For each row-group, we set up a future that will evaluate the scan and post its.

        // TODO(ngates): we could query the layout for splits and then process them in parallel.
        //  For now, we just scan the entire layout with one mask.
        //  Note that to implement this we would use stream::try_unfold
        let stream = stream::once(async move {
            // TODO(ngates): we should launch the evaluate_async onto a worker thread pool.
            let row_range = 0..self.row_count();

            let eval = scan
                .range_scan(row_range)?
                .evaluate_async(|row_mask, expr| reader.evaluate_expr(row_mask, expr));
            pin_mut!(eval);
            send(eval);

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

fn send<T: Send>(t: T) -> T {
    t
}
