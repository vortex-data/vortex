use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use futures_util::{stream, FutureExt, StreamExt, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::{ArrayData, ContextRef};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_layout::{ExprEvaluator, LayoutData, LayoutReader};
use vortex_scan::Scan;

use crate::v2::exec::ExecDriver;
use crate::v2::io::IoDriver;
use crate::v2::segments::channel::SegmentChannel;

/// A Vortex file ready for reading.
///
/// It is generic over the `IoDriver` implementation enabling us to swap out the I/O subsystem for
/// particular environments. For example, memory mapped files vs object-store. By remaining generic,
/// it allows us to support both `Send` and `?Send` I/O drivers.
pub struct VortexFile<I> {
    pub(crate) ctx: ContextRef,
    pub(crate) layout: LayoutData,
    pub(crate) io_driver: I,
    pub(crate) exec_driver: Arc<dyn ExecDriver>,
    pub(crate) splits: Arc<[Range<u64>]>,
}

/// Async implementation of Vortex File.
impl<I: IoDriver> VortexFile<I> {
    /// Returns the number of rows in the file.
    pub fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    /// Returns the DType of the file.
    pub fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    /// Performs a scan operation over the file.
    pub fn scan(&self, scan: Arc<Scan>) -> VortexResult<impl ArrayStream + 'static + use<'_, I>> {
        let result_dtype = scan.result_dtype(self.dtype())?;

        // Set up a segment channel to collect segment requests from the execution stream.
        let segment_channel = SegmentChannel::new();

        // Now we give one end of the channel to the layout reader...
        let reader: Arc<dyn LayoutReader> = self
            .layout
            .reader(segment_channel.reader(), self.ctx.clone())?;
        let exec_stream = stream::iter(ArcIter::new(self.splits.clone()))
            .map(move |row_range| scan.clone().range_scan(row_range))
            .map(move |range_scan| match range_scan {
                Ok(range_scan) => {
                    let reader = reader.clone();
                    async move {
                        range_scan
                            .evaluate_async(|row_mask, expr| reader.evaluate_expr(row_mask, expr))
                            .await
                    }
                    .boxed()
                }
                Err(e) => futures::future::ready(Err(e)).boxed(),
            })
            .boxed();
        let exec_stream = self.exec_driver.drive(exec_stream);

        // ...and the other end to the segment driver.
        let io_stream = self.io_driver.drive(segment_channel.into_stream());

        Ok(ArrayStreamAdapter::new(
            result_dtype,
            UnifiedDriverStream {
                exec_stream,
                io_stream,
            },
        ))
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

pin_project! {
    /// A [`Stream`] that drives the both the I/O stream and the execution stream concurrently.
    ///
    /// This is sort of like a `select!` implementation, but not quite.
    ///
    /// We can't use `futures::stream::select` because it requires both streams to terminate, and
    /// our I/O stream will never terminate.
    ///
    /// We can't use `futures::stream::zip` because it waits for boths streams to emit an item,
    /// but our execution stream may require multiple I/O operations to complete before it can
    /// return an item.
    struct UnifiedDriverStream<R, S> {
        #[pin]
        exec_stream: R,
        #[pin]
        io_stream: S,
    }
}

impl<R, S> Stream for UnifiedDriverStream<R, S>
where
    R: Stream<Item = VortexResult<ArrayData>>,
    S: Stream<Item = VortexResult<()>>,
{
    type Item = VortexResult<ArrayData>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            // If the exec stream is ready, then we can return the result.
            if let Poll::Ready(r) = this.exec_stream.try_poll_next_unpin(cx) {
                return Poll::Ready(r);
            }
            // Otherwise, we try to poll the I/O stream.
            // If the I/O stream is not ready, then we return Pending and wait for the next wakeup.
            if matches!(this.io_stream.as_mut().poll_next(cx), Poll::Pending) {
                return Poll::Pending;
            }
        }
    }
}
