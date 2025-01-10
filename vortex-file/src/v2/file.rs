use std::ops::Range;
use std::sync::Arc;
use std::task::Poll;

use futures::channel::oneshot;
use futures::pin_mut;
use futures_executor::block_on;
use futures_util::{stream, StreamExt, TryFutureExt};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};
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

        let threads = 4;

        let thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| vortex_err!("failed to create thread pool: {}", e))?,
        );

        // For each row-group, we set up a future that will evaluate the scan and post its.
        let batches = stream::iter(self.splits.iter().cloned())
            .map(move |row_range| {
                let (send, recv) = oneshot::channel();

                let thread_pool = thread_pool.clone();
                let row_range = row_range.clone();
                let scan = scan.clone();
                let reader = reader.clone();

                // Launch the scan task onto the thread pool.
                thread_pool.spawn_fifo(move || {
                    let array_result =
                        scan.range_scan(row_range).and_then(|range_scan| {
                            block_on(range_scan.evaluate_async(|row_mask, expr| {
                                reader.evaluate_expr(row_mask, expr)
                            }))
                        });
                    send.send(array_result)
                        .map_err(|_| VortexError::from(vortex_err!("send failed, recv dropped")))
                        .vortex_expect("send_failed, recv dropped");
                });
                recv
            })
            .then(|recv| async move {
                match recv.await {
                    Ok(r) => r,
                    Err(_cancelled) => Err(vortex_err!("recv failed, send dropped")),
                }
            })
            // Make sure we have spawned as many row ranges as we have threads.
            .buffered(threads);

        // We also set up a stream that will drive the segment cache.
        let driver = self.segments.driver();

        let mut stream = stream::select(batches, driver);

        let stream = stream::poll_fn(move |cx| {
            pin_mut!(batches);

            // Now we alternate between polling the batches stream and driving the I/O.
            loop {
                if let Poll::Ready(Some(array)) = batches.poll_next_unpin(cx) {
                    return Poll::Ready(Some(array));
                }

                let drive = self.segments.drive();
                pin_mut!(drive);
                match drive.try_poll_unpin(cx) {
                    Poll::Ready(_) => {}
                    Poll::Pending => return Poll::Pending,
                }
            }
        });

        Ok(ArrayStreamAdapter::new(result_dtype, stream))
    }
}

struct SegmentDriver<B, R> {
    batches: B,
    segments: Arc<SegmentCache<R>>,
}
