use std::future::Future;
use std::pin::Pin;
use std::task::{ready, Context, Poll};

use futures::Stream;
use futures_util::stream::FuturesUnordered;
use pin_project::pin_project;
use tokio::sync::{Semaphore, TryAcquireError};

/// [`Future`] that carries the amount of memory it will require to hold the completed value.
#[pin_project]
struct SizedFut<Fut> {
    #[pin]
    inner: Fut,
    size_in_bytes: usize,
}

impl<Fut: Future> Future for SizedFut<Fut> {
    // We tag the size in bytes alongside the original output
    type Output = (Fut::Output, usize);

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let size_in_bytes = self.size_in_bytes;
        let inner = ready!(self.project().inner.poll(cx));

        Poll::Ready((inner, size_in_bytes))
    }
}

/// A [`Stream`] that can work on several simultaneous requests, capping the amount of memory it
/// uses at any given point.
///
/// It is meant to serve as a buffer between a producer and consumer of IO requests, with built-in
/// backpressure that prevents the producer from materializing more than a specified maximum
/// amount of memory.
///
/// This crate internally makes use of tokio's [Semaphore], and thus is only available with
/// the `tokio` feature enabled.
#[pin_project]
pub struct SizeLimitedStream<Fut> {
    #[pin]
    inflight: FuturesUnordered<SizedFut<Fut>>,
    bytes_available: Semaphore,
}

impl<Fut> SizeLimitedStream<Fut> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inflight: FuturesUnordered::new(),
            bytes_available: Semaphore::new(max_bytes),
        }
    }

    pub fn bytes_available(&self) -> usize {
        self.bytes_available.available_permits()
    }
}

impl<Fut> SizeLimitedStream<Fut>
where
    Fut: Future,
{
    /// Push a future into the queue after reserving `bytes` of capacity.
    ///
    /// This call may need to wait until there is sufficient capacity available in the stream to
    /// begin work on this future.
    pub async fn push(&self, fut: Fut, bytes: usize) {
        // Attempt to acquire enough permits to begin working on a request that will occupy
        // `bytes` amount of memory when it completes.
        // Acquiring the permits is what creates backpressure for the producer.
        self.bytes_available
            .acquire_many(bytes as u32)
            .await
            .unwrap_or_else(|_| unreachable!("pushing to closed semaphore"))
            .forget();

        let sized_fut = SizedFut {
            inner: fut,
            size_in_bytes: bytes,
        };

        // push into the pending queue
        self.inflight.push(sized_fut);
    }

    /// Synchronous push method. This method will attempt to push if there is enough capacity
    /// to begin work on the future immediately.
    ///
    /// If there is not enough capacity, the original future is returned to the caller.
    pub fn try_push(&self, fut: Fut, bytes: usize) -> Result<(), Fut> {
        match self.bytes_available.try_acquire_many(bytes as u32) {
            Ok(permits) => {
                permits.forget();
                let sized_fut = SizedFut {
                    inner: fut,
                    size_in_bytes: bytes,
                };

                self.inflight.push(sized_fut);
                Ok(())
            }
            Err(acquire_err) => match acquire_err {
                TryAcquireError::Closed => {
                    unreachable!("try_pushing to closed semaphore");
                }

                // No permits available, return the future back to the client so they can
                // try again.
                TryAcquireError::NoPermits => Err(fut),
            },
        }
    }
}

impl<Fut> Stream for SizeLimitedStream<Fut>
where
    Fut: Future,
{
    type Item = Fut::Output;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        match ready!(this.inflight.poll_next(cx)) {
            None => Poll::Ready(None),
            Some((result, bytes_read)) => {
                this.bytes_available.add_permits(bytes_read);

                Poll::Ready(Some(result))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{future, io};

    use bytes::Bytes;
    use futures_util::future::BoxFuture;
    use futures_util::{FutureExt, StreamExt};

    use crate::limit::SizeLimitedStream;

    async fn make_future(len: usize) -> Bytes {
        "a".as_bytes().iter().copied().cycle().take(len).collect()
    }

    #[tokio::test]
    async fn test_size_limit() {
        let mut size_limited = SizeLimitedStream::new(10);
        size_limited.push(make_future(5), 5).await;
        size_limited.push(make_future(5), 5).await;

        // Pushing last request should fail, because we have 10 bytes outstanding.
        assert!(size_limited.try_push(make_future(1), 1).is_err());

        // but, we can pop off a finished work item, and then enqueue.
        assert!(size_limited.next().await.is_some());
        assert!(size_limited.try_push(make_future(1), 1).is_ok());
    }

    #[tokio::test]
    async fn test_does_not_leak_permits() {
        let bad_fut: BoxFuture<'static, io::Result<Bytes>> =
            future::ready(Err(io::Error::new(io::ErrorKind::Other, "badness"))).boxed();

        let good_fut: BoxFuture<'static, io::Result<Bytes>> =
            future::ready(Ok(Bytes::from_static("aaaaa".as_bytes()))).boxed();

        let mut size_limited = SizeLimitedStream::new(10);
        size_limited.push(bad_fut, 10).await;

        // attempt to push should fail, as all 10 bytes of capacity is occupied by bad_fut.
        let good_fut = size_limited
            .try_push(good_fut, 5)
            .expect_err("try_push should fail");

        // Even though the result was an error, the 10 bytes of capacity should be returned back to
        // the stream, allowing us to push the next request.
        let next = size_limited.next().await.unwrap();
        assert!(next.is_err());

        assert_eq!(size_limited.bytes_available(), 10);
        assert!(size_limited.try_push(good_fut, 5).is_ok());
    }
}
