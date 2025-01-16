use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use futures_util::{stream, FutureExt, StreamExt, TryFutureExt, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_array::compute::FilterMask;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};
use vortex_array::ContextRef;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_expr::{ExprRef, Identity};
use vortex_layout::{ExprEvaluator, LayoutReader};
use vortex_scan::{RowMask, Scanner};

use crate::v2::exec::ExecDriver;
use crate::v2::io::IoDriver;
use crate::v2::segments::channel::SegmentChannel;
use crate::v2::FileLayout;

/// A Vortex file ready for reading.
///
/// It is generic over the `IoDriver` implementation enabling us to swap out the I/O subsystem for
/// particular environments. For example, memory mapped files vs object-store. By remaining generic,
/// it allows us to support both `Send` and `?Send` I/O drivers.
pub struct VortexFile<I> {
    pub(crate) ctx: ContextRef,
    pub(crate) file_layout: FileLayout,
    pub(crate) io_driver: I,
    pub(crate) exec_driver: Arc<dyn ExecDriver>,
    pub(crate) splits: Arc<[Range<u64>]>,
}

pub struct Scan {
    projection: ExprRef,
    filter: Option<ExprRef>,
    row_indices: Option<Buffer<u64>>,
}

impl Scan {
    pub fn all() -> Self {
        Self {
            projection: Identity::new_expr(),
            filter: None,
            row_indices: None,
        }
    }

    pub fn new(projection: ExprRef) -> Self {
        Self {
            projection,
            filter: None,
            row_indices: None,
        }
    }

    pub fn filtered(filter: ExprRef) -> Self {
        Self {
            projection: Identity::new_expr(),
            filter: Some(filter),
            row_indices: None,
        }
    }

    pub fn with_filter(mut self, filter: ExprRef) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_some_filter(mut self, filter: Option<ExprRef>) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_projection(mut self, projection: ExprRef) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_row_indices(mut self, row_indices: Buffer<u64>) -> Self {
        self.row_indices = Some(row_indices);
        self
    }
}

/// Async implementation of Vortex File.
impl<I: IoDriver> VortexFile<I> {
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

    /// Performs a scan operation over the file.
    pub fn scan(&self, scan: Scan) -> VortexResult<impl ArrayStream + 'static + use<'_, I>> {
        let row_masks = ArcIter::new(self.splits.clone()).filter_map(move |row_range| {
            let Some(row_indices) = &scan.row_indices else {
                // If there is no row indices filter, then take the whole range
                return Some(RowMask::new_valid_between(row_range.start, row_range.end));
            };

            // Otherwise, find the indices that are within the row range.
            if row_indices
                .first()
                .is_some_and(|&first| first >= row_range.end)
                || row_indices
                    .last()
                    .is_some_and(|&last| row_range.start >= last)
            {
                return None;
            }

            // For the given row range, find the indices that are within the row_indices.
            let start_idx = row_indices
                .binary_search(&row_range.start)
                .unwrap_or_else(|x| x);
            let end_idx = row_indices
                .binary_search(&row_range.end)
                .unwrap_or_else(|x| x);

            if start_idx == end_idx {
                // No rows in range
                return None;
            }

            // Construct a row mask for the range.
            let filter_mask = FilterMask::from_indices(
                usize::try_from(row_range.end - row_range.start)
                    .vortex_expect("Split ranges are within usize"),
                row_indices[start_idx..end_idx]
                    .iter()
                    .map(|&idx| {
                        usize::try_from(idx - row_range.start).vortex_expect("index within range")
                    })
                    .collect(),
            );
            Some(RowMask::new(filter_mask, row_range.start))
        });

        self.scan_with_masks(row_masks, scan.projection, scan.filter)
    }

    fn scan_with_masks<R>(
        &self,
        row_masks: R,
        projection: ExprRef,
        filter: Option<ExprRef>,
    ) -> VortexResult<impl ArrayStream + 'static + use<'_, I, R>>
    where
        R: Iterator<Item = RowMask> + Send + 'static,
    {
        let scanner = Arc::new(Scanner::new(self.dtype().clone(), projection, filter)?);

        let result_dtype = scanner.result_dtype().clone();

        // Set up a segment channel to collect segment requests from the execution stream.
        let segment_channel = SegmentChannel::new();

        // Create a single LayoutReader that is reused for the entire scan.
        let reader: Arc<dyn LayoutReader> = self
            .file_layout
            .root_layout()
            .reader(segment_channel.reader(), self.ctx.clone())?;

        // Now we give one end of the channel to the layout reader...
        let exec_stream = stream::iter(row_masks)
            .map(
                move |row_mask| match scanner.clone().range_scanner(row_mask) {
                    Ok(range_scan) => {
                        let reader = reader.clone();
                        async move {
                            range_scan
                                .evaluate_async(|row_mask, expr| {
                                    reader.evaluate_expr(row_mask, expr)
                                })
                                .await
                        }
                        .boxed()
                    }
                    Err(e) => futures::future::ready(Err(e)).boxed(),
                },
            )
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

    /// Resolves the requested statistics for the file.
    pub fn statistics(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<impl Future<Output = VortexResult<Vec<StatsSet>>> + 'static + use<'_, I>>
    {
        // Set up a segment channel to collect segment requests from the execution stream.
        let segment_channel = SegmentChannel::new();

        // Create a single LayoutReader that is reused for the entire scan.
        let reader: Arc<dyn LayoutReader> = self
            .file_layout
            .root_layout()
            .reader(segment_channel.reader(), self.ctx.clone())?;

        let exec_future = async move { reader.evaluate_stats(field_paths, stats).await }.boxed();
        let io_stream = self.io_driver.drive(segment_channel.into_stream());

        Ok(UnifiedDriverFuture {
            exec_future,
            io_stream,
        })
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

impl<T, R, S> Stream for UnifiedDriverStream<R, S>
where
    R: Stream<Item = VortexResult<T>>,
    S: Stream<Item = VortexResult<()>>,
{
    type Item = VortexResult<T>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        loop {
            // If the exec stream is ready, then we can return the result.
            // If it's pending, then we try polling the I/O stream.
            if let Poll::Ready(r) = this.exec_stream.try_poll_next_unpin(cx) {
                return Poll::Ready(r);
            }

            match this.io_stream.as_mut().try_poll_next_unpin(cx) {
                // If the I/O stream made progress, it returns Ok.
                Poll::Ready(Some(Ok(()))) => {}
                // If the I/O stream failed, then propagate the error.
                Poll::Ready(Some(Err(result))) => {
                    return Poll::Ready(Some(Err(result)));
                }
                // Unexpected end of stream.
                Poll::Ready(None) => {
                    return Poll::Ready(Some(Err(vortex_err!("unexpected end of I/O stream"))));
                }
                // If the I/O stream is not ready, then we return Pending and wait for the next wakeup.
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    struct UnifiedDriverFuture<R, S> {
        #[pin]
        exec_future: R,
        #[pin]
        io_stream: S,
    }
}

impl<T, R, S> Future for UnifiedDriverFuture<R, S>
where
    R: Future<Output = VortexResult<T>>,
    S: Stream<Item = VortexResult<()>>,
{
    type Output = VortexResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        loop {
            // If the exec stream is ready, then we can return the result.
            // If it's pending, then we try polling the I/O stream.
            if let Poll::Ready(r) = this.exec_future.try_poll_unpin(cx) {
                return Poll::Ready(r);
            }

            match this.io_stream.as_mut().try_poll_next_unpin(cx) {
                // If the I/O stream made progress, it returns Ok.
                Poll::Ready(Some(Ok(()))) => {}
                // If the I/O stream failed, then propagate the error.
                Poll::Ready(Some(Err(result))) => {
                    return Poll::Ready(Err(result));
                }
                // Unexpected end of stream.
                Poll::Ready(None) => {
                    return Poll::Ready(Err(vortex_err!("unexpected end of I/O stream")));
                }
                // If the I/O stream is not ready, then we return Pending and wait for the next wakeup.
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
