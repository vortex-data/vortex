use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, TryFutureExt, TryStreamExt};
use pin_project_lite::pin_project;
use vortex_error::VortexResult;

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
    pub struct UnifiedDriverStream<R, S> {
        #[pin]
        pub exec_stream: R,
        #[pin]
        pub io_stream: S,
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
                    continue;
                }
                // If the I/O stream is not ready, then we return Pending and wait for the next wakeup.
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pin_project! {
    pub struct UnifiedDriverFuture<R, S> {
        #[pin]
        pub exec_future: R,
        #[pin]
        pub io_stream: S,
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
                    continue;
                }
                // If the I/O stream is not ready, then we return Pending and wait for the next wakeup.
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
