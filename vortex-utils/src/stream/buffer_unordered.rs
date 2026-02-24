// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stream for buffering unordered futures from a stream.

use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt as _;
use futures::stream::Fuse;
use futures::stream::FusedStream;
use futures::stream::FuturesUnordered;
use pin_project_lite::pin_project;

pin_project! {
    /// Stream for [`super::StreamExt::buffer_unordered2`].
    #[must_use = "streams do nothing unless polled"]
    pub struct BufferUnordered<S> where S: Stream{
        #[pin]
        stream: Fuse<S>,
        max: AtomicUsize,
        in_progress_queue: FuturesUnordered<S::Item>,
    }
}

impl<S: Stream> BufferUnordered<S> {
    pub(super) fn new(stream: S, concurrency: AtomicUsize) -> Self {
        Self {
            stream: stream.fuse(),
            max: concurrency,
            in_progress_queue: FuturesUnordered::new(),
        }
    }
}

impl<S> Stream for BufferUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    type Item = <S::Item as Future>::Output;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();

        // First up, try to spawn off as many futures as possible by filling up
        // our queue of futures.
        while this.in_progress_queue.len() < this.max.load(Ordering::Relaxed) {
            match this.stream.as_mut().poll_next(cx) {
                Poll::Ready(Some(fut)) => this.in_progress_queue.push(fut),
                Poll::Ready(None) | Poll::Pending => break,
            }
        }

        // Attempt to pull the next value from the in_progress_queue
        match this.in_progress_queue.poll_next_unpin(cx) {
            x @ Poll::Pending | x @ Poll::Ready(Some(_)) => {
                // After pulling the latest value, re-fill before returning it.
                while this.in_progress_queue.len() < this.max.load(Ordering::Relaxed) {
                    match this.stream.as_mut().poll_next(cx) {
                        Poll::Ready(Some(fut)) => this.in_progress_queue.push(fut),
                        Poll::Ready(None) | Poll::Pending => break,
                    }
                }

                return x;
            }
            Poll::Ready(None) => {}
        }

        // If more values are still coming from the stream, we're not done yet
        if this.stream.is_done() {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let queue_len = self.in_progress_queue.len();
        let (lower, upper) = self.stream.size_hint();
        let lower = lower.saturating_add(queue_len);
        let upper = match upper {
            Some(x) => x.checked_add(queue_len),
            None => None,
        };
        (lower, upper)
    }
}

impl<S> FusedStream for BufferUnordered<S>
where
    S: Stream,
    S::Item: Future,
{
    fn is_terminated(&self) -> bool {
        self.in_progress_queue.is_terminated() && self.stream.is_terminated()
    }
}
