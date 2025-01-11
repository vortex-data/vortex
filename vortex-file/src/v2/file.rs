use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::Stream;
use futures_executor::block_on;
use futures_util::{stream, StreamExt, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_io::VortexReadAt;
use vortex_layout::{ExprEvaluator, LayoutData, LayoutReader};
use vortex_scan::Scan;

use crate::v2::segments::cache::SegmentCache;

pub struct VortexFile<R> {
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) segments: SegmentCache<R>,
    pub(crate) splits: Arc<[Range<u64>]>,
    pub(crate) thread_pool: Arc<rayon::ThreadPool>,
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
    pub fn scan(self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + 'static> {
        // Create a shared reader for the scan.
        let reader: Arc<dyn LayoutReader> = self
            .layout
            .reader(self.segments.reader(), self.ctx.clone())?;
        let result_dtype = scan.result_dtype(self.dtype())?;

        // For each row-group, we set up a future that will evaluate the scan and post its.
        let row_group_driver = stream::iter(ArcIter::new(self.splits.clone()))
            .map(move |row_range| {
                let (send, recv) = oneshot::channel();
                let reader = reader.clone();
                let range_scan = scan.clone().range_scan(row_range);

                // Launch the scan task onto the thread pool.
                self.thread_pool.spawn_fifo(move || {
                    let array_result =
                        range_scan.and_then(|range_scan| {
                            block_on(range_scan.evaluate_async(|row_mask, expr| {
                                reader.evaluate_expr(row_mask, expr)
                            }))
                        });
                    // Post the result back to the main thread
                    send.send(array_result)
                        .map_err(|_| vortex_err!("send failed, recv dropped"))
                        .vortex_expect("send_failed, recv dropped");
                });

                recv
            })
            .then(|recv| async move {
                recv.await
                    .unwrap_or_else(|_cancelled| Err(vortex_err!("recv failed, send dropped")))
            });
        // TODO(ngates): we should call buffered(n) on this stream so that is launches multiple
        //  splits to run in parallel. Currently we use block_on, so there's no point this being
        //  any higher than the size of the thread pool. If we switch to running LocalExecutor,
        //  then there may be some value in slightly over-subscribing.

        // Set up an I/O driver that will make progress on 32 I/O requests at a time.
        // TODO(ngates): we should probably have segments hold an Arc'd driver stream internally
        //  so that multiple scans can poll it, while still sharing the same global concurrency
        //  limit?
        let io_driver = self.segments.driver().buffered(32);

        Ok(ArrayStreamAdapter::new(
            result_dtype,
            ScanDriver {
                row_group_driver,
                io_driver,
            },
        ))
    }
}

pin_project! {
    struct ScanDriver<R, S> {
        #[pin]
        row_group_driver: R,
        #[pin]
        io_driver: S,
    }
}

impl<R, S> Stream for ScanDriver<R, S>
where
    R: Stream<Item = VortexResult<ArrayData>>,
    S: Stream<Item = VortexResult<()>>,
{
    type Item = VortexResult<ArrayData>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            // If the row group driver is ready, then we can return the result.
            if let Poll::Ready(r) = this.row_group_driver.try_poll_next_unpin(cx) {
                return Poll::Ready(r);
            }
            // Otherwise, we try to poll the I/O driver.
            // If the I/O driver is not ready, then we return Pending and wait for I/
            // to wake up the driver.
            if matches!(this.io_driver.as_mut().poll_next(cx), Poll::Pending) {
                return Poll::Pending;
            }
        }
    }
}

/// There is no `IntoIterator` for `Arc<[T]>` so to avoid copying into a Vec<T>, we define our own.
/// See <https://users.rust-lang.org/t/arc-to-owning-iterator/115190/11>.
struct ArcIter<T> {
    inner: Arc<[T]>,
    pos: usize,
}

impl<T> ArcIter<T> {
    fn new(inner: Arc<[T]>) -> Self {
        Self { inner, pos: 0 }
    }
}

impl<T: Clone> Iterator for ArcIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        (self.pos < self.inner.len()).then(|| {
            let item = self.inner[self.pos].clone();
            self.pos += 1;
            item
        })
    }
}
