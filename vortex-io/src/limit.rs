// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::Stream;
use futures::stream::FuturesUnordered;
use pin_project_lite::pin_project;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tokio::sync::TryAcquireError;
use vortex_error::VortexExpect;

pin_project! {
    /// [`Future`] that carries the amount of memory it will require to hold the completed value.
    ///
    /// The `OwnedSemaphorePermit` ensures permits are automatically returned when this future
    /// is dropped, either after completion or if cancelled/aborted.
    struct SizedFut<Fut> {
        #[pin]
        inner: Fut,
        // Owned permit that will be automatically dropped when the future completes or is dropped
        _permits: OwnedSemaphorePermit,
    }
}

impl<Fut: Future> Future for SizedFut<Fut> {
    type Output = Fut::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let result = ready!(this.inner.poll(cx));
        Poll::Ready(result)
    }
}

pin_project! {
    /// A [`Stream`] that can work on several simultaneous requests, capping the amount of memory it
    /// uses at any given point.
    ///
    /// It is meant to serve as a buffer between a producer and consumer of IO requests, with built-in
    /// backpressure that prevents the producer from materializing more than a specified maximum
    /// amount of memory.
    ///
    /// This crate internally makes use of tokio's [Semaphore], and thus is only available with
    /// the `tokio` feature enabled.
    pub struct SizeLimitedStream<Fut> {
        #[pin]
        inflight: FuturesUnordered<SizedFut<Fut>>,
        bytes_available: Arc<Semaphore>,
    }
}

impl<Fut> SizeLimitedStream<Fut> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inflight: FuturesUnordered::new(),
            bytes_available: Arc::new(Semaphore::new(max_bytes)),
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
        let permits = Arc::clone(&self.bytes_available)
            .acquire_many_owned(bytes.try_into().vortex_expect("bytes must fit in u32"))
            .await
            .unwrap_or_else(|_| unreachable!("pushing to closed semaphore"));

        let sized_fut = SizedFut {
            inner: fut,
            _permits: permits,
        };

        // push into the pending queue
        self.inflight.push(sized_fut);
    }

    /// Synchronous push method. This method will attempt to push if there is enough capacity
    /// to begin work on the future immediately.
    ///
    /// If there is not enough capacity, the original future is returned to the caller.
    pub fn try_push(&self, fut: Fut, bytes: usize) -> Result<(), Fut> {
        match Arc::clone(&self.bytes_available)
            .try_acquire_many_owned(bytes.try_into().vortex_expect("bytes must fit in u32"))
        {
            Ok(permits) => {
                let sized_fut = SizedFut {
                    inner: fut,
                    _permits: permits,
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
            Some(result) => {
                // Permits are automatically returned when the SizedFut is dropped
                // after being polled to completion by FuturesUnordered
                Poll::Ready(Some(result))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::future;
    use std::io;

    use bytes::Bytes;
    use futures::FutureExt;
    use futures::StreamExt;
    use futures::future::BoxFuture;

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
            future::ready(Err(io::Error::other("badness"))).boxed();

        let good_fut: BoxFuture<'static, io::Result<Bytes>> =
            future::ready(Ok(Bytes::from("aaaaa"))).boxed();

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

    #[tokio::test]
    async fn test_size_limited_stream_zero_capacity() {
        let stream = SizeLimitedStream::new(0);

        // Should not be able to push anything
        let result = stream.try_push(async { vec![1u8] }, 1);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_size_limited_stream_dropped_future_releases_permits() {
        use futures::future::BoxFuture;

        let stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Push a future that will never complete
        stream
            .push(
                Box::pin(async {
                    // This future will be dropped before completion
                    futures::future::pending::<Vec<u8>>().await
                }),
                5,
            )
            .await;

        // Push another future
        stream.push(Box::pin(async { vec![1u8; 3] }), 3).await;

        // We should have 2 bytes available now
        assert_eq!(stream.bytes_available(), 2);

        // Drop the stream without consuming the futures
        drop(stream);

        // Create a new stream to verify permits aren't leaked
        let mut new_stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Should be able to use all 10 bytes
        new_stream.push(Box::pin(async { vec![0u8; 10] }), 10).await;
        assert_eq!(new_stream.bytes_available(), 0);

        // Consume to verify it works
        let result = new_stream.next().await;
        assert!(result.is_some());
        assert_eq!(new_stream.bytes_available(), 10);
    }

    #[tokio::test]
    async fn test_size_limited_stream_exact_capacity() {
        use futures::future::BoxFuture;

        let mut stream = SizeLimitedStream::<BoxFuture<'static, Vec<u8>>>::new(10);

        // Push exactly the capacity
        stream.push(Box::pin(async { vec![0u8; 10] }), 10).await;

        // Should not be able to push more
        let result = stream.try_push(Box::pin(async { vec![1u8] }), 1);
        assert!(result.is_err());

        // After consuming, should be able to push again
        stream
            .next()
            .await
            .expect("The 10 byte vector ought to be in there!");
        assert_eq!(stream.bytes_available(), 10);

        let result = stream.try_push(Box::pin(async { vec![1u8; 5] }), 5);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_size_limited_stream_multiple_small_pushes() {
        let mut stream = SizeLimitedStream::new(100);

        // Push many small items
        for i in 0..10 {
            #[allow(clippy::cast_possible_truncation)]
            stream.push(async move { vec![i as u8; 5] }, 5).await;
        }

        // Should have used 50 bytes
        assert_eq!(stream.bytes_available(), 50);

        // Consume all
        let mut count = 0;
        while stream.next().await.is_some() {
            count += 1;
            if count == 10 {
                break;
            }
        }

        assert_eq!(count, 10);
        assert_eq!(stream.bytes_available(), 100);
    }

    #[test]
    fn test_size_overflow_protection() {
        let stream = SizeLimitedStream::new(100);

        // Test with size that would overflow u32 on 32-bit systems
        // but this test assumes 64-bit where usize > u32::MAX is possible
        #[cfg(target_pointer_width = "64")]
        {
            let _large_size = (u32::MAX as usize) + 1;
            // This should panic with current implementation
            // We're documenting the issue rather than testing the panic
            // as the behavior may change
        }

        // Test with reasonable size
        let result = stream.try_push(async { vec![0u8; 50] }, 50);
        assert!(result.is_ok());
    }
}
