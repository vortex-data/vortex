// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A clone of a [`Shared`] future that is safe to drop at any point.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

use futures::FutureExt;
use futures::future::Shared;

/// A clone of a [`Shared`] future that polls again as it is dropped.
///
/// A [`Shared`] future is polled by one clone at a time. A clone that polls it while another is
/// polling only leaves its waker, relying on the polling clone to poll again once woken. A clone
/// dropped after being woken but before polling again would leave the future with nobody to poll
/// it, and every other clone waiting forever. A `PolledShared` that has polled therefore polls
/// again as it is dropped, for as long as each poll is woken.
///
/// Every clone must still be polled again once woken while it is held: a clone polled once with a
/// waker that never polls again, such as a no-op waker, strands the other clones the same way.
pub struct PolledShared<F: Future>
where
    F::Output: Clone,
{
    future: Shared<F>,
    /// Whether the future was polled and has not yet finished.
    polling: bool,
}

impl<F: Future> PolledShared<F>
where
    F::Output: Clone,
{
    /// Wraps one clone of a shared future.
    pub fn new(future: Shared<F>) -> Self {
        Self {
            future,
            polling: false,
        }
    }
}

impl<F: Future> Future for PolledShared<F>
where
    F::Output: Clone,
{
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let poll = self.future.poll_unpin(cx);
        self.polling = poll.is_pending();
        poll
    }
}

impl<F: Future> Drop for PolledShared<F>
where
    F::Output: Clone,
{
    fn drop(&mut self) {
        if !self.polling {
            return;
        }
        // A wake during the last poll could again leave nobody to poll, so poll until a poll is
        // not woken. Each wake is progress of the future, so this ends.
        let woken = Arc::new(Woken::default());
        let waker = Waker::from(Arc::clone(&woken));
        let mut cx = Context::from_waker(&waker);
        while self.future.poll_unpin(&mut cx).is_pending()
            && woken.0.swap(false, Ordering::AcqRel)
        {}
    }
}

/// Records that a waker was woken.
#[derive(Default)]
struct Woken(AtomicBool);

impl Wake for Woken {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::future::poll_fn;
    use std::sync::Mutex;

    use futures::future::BoxFuture;

    use super::*;

    type Handle = PolledShared<BoxFuture<'static, u8>>;

    /// A clone dropped after its poll was woken polls again, so a clone that polled meanwhile
    /// is woken by the finished future rather than waiting forever.
    #[test]
    fn dropped_clone_does_not_strand_the_others() {
        let other: Arc<Mutex<Option<Handle>>> = Arc::default();
        let woken = Arc::new(Woken::default());
        let mut polls = 0;
        let future = {
            let other = Arc::clone(&other);
            let woken = Arc::clone(&woken);
            // Finishes while its first poll is running, during which the other clone polls too,
            // as a clone on another thread would.
            poll_fn(move |cx| {
                polls += 1;
                if polls > 1 {
                    return Poll::Ready(1);
                }
                cx.waker().wake_by_ref();
                let waker = Waker::from(Arc::clone(&woken));
                let mut other = other.lock().unwrap_or_else(|poison| poison.into_inner());
                let other = other.as_mut().expect("the other clone is set");
                assert!(
                    other
                        .poll_unpin(&mut Context::from_waker(&waker))
                        .is_pending()
                );
                Poll::Pending
            })
            .boxed()
            .shared()
        };
        let mut first = PolledShared::new(future.clone());
        *other.lock().unwrap_or_else(|poison| poison.into_inner()) =
            Some(PolledShared::new(future));

        assert!(
            first
                .poll_unpin(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert!(!woken.0.load(Ordering::Acquire));
        drop(first);
        assert!(woken.0.load(Ordering::Acquire));

        let second = other
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
            .expect("the other clone is set");
        assert_eq!(second.now_or_never(), Some(1));
    }
}
