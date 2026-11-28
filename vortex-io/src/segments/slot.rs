// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::SegmentFuture;

struct Slot {
    result: OnceLock<SharedVortexResult<BufferHandle>>,
    /// Told of the waiters until the read completes.
    hooks: Mutex<Option<Box<dyn FillHooks>>>,
    polled: AtomicBool,
    /// Whether the completer was dropped without completing the read.
    abandoned: AtomicBool,
    wakers: Mutex<Vec<Option<Waker>>>,
    /// Increments each time `wakers` is emptied.
    generation: AtomicU64,
}

impl Slot {
    /// Empties the waker list and wakes every waiter that was on it, outside the lock.
    fn wake_all(&self) {
        let wakers = {
            let mut wakers = self.wakers.lock();
            self.generation.fetch_add(1, Ordering::Release);
            std::mem::take(&mut *wakers)
        };
        for waker in wakers.into_iter().flatten() {
            waker.wake();
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.result.get().is_none()
            && let Some(hooks) = self.hooks.get_mut()
        {
            hooks.dropped();
        }
    }
}

/// Told of the waiters of a slot that a [`Completer`] fills.
pub(crate) trait FillHooks: Send + Sync {
    /// A waiter polled the read for the first time while it was pending.
    fn polled(&self);
    /// Every waiter was dropped before the read completed.
    fn dropped(&self);
    /// The completer was dropped without completing the read. Each waiter awaits a future from
    /// here instead.
    fn abandoned(&self) -> SegmentFuture;
}

/// Fills a [`SharedRead`] created by [`SharedRead::new`]. It doesn't keep the read's
/// waiters alive, so the read is cancelled once they have all gone.
pub(crate) struct Completer {
    slot: Weak<Slot>,
    done: bool,
}

impl Completer {
    /// Whether every waiter has been dropped.
    pub(crate) fn is_canceled(&self) -> bool {
        self.slot.strong_count() == 0
    }

    /// Completes the read, waking every waiter.
    pub(crate) fn complete(mut self, result: VortexResult<BufferHandle>) {
        self.done = true;
        if let Some(slot) = self.slot.upgrade() {
            drop(slot.result.set(result.map_err(Arc::new)));
            drop(slot.hooks.lock().take());
            slot.wake_all();
        }
    }
}

impl Drop for Completer {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if let Some(slot) = self.slot.upgrade() {
            if slot.hooks.lock().is_none() {
                drop(slot.result.set(Err(Arc::new(vortex_err!(
                    "Segment read dropped before it completed"
                )))));
            }
            slot.abandoned.store(true, Ordering::Release);
            slot.wake_all();
        }
    }
}

/// One waiter on a read that a [`Completer`] fills, such as a
/// [`FileSegmentSource`](super::FileSegmentSource) read that its driver resolves.
///
/// Each clone is a waiter, and the read is cancelled once every waiter is dropped. The result,
/// error or not, goes to every waiter.
///
/// [`futures::future::Shared`] would serialize many threads awaiting the read: each poll of a
/// pending `Shared` locks the mutex around its waker list, and its completion wakes every waiter
/// while holding that lock. Here, a waiter whose read has finished reads the result without taking
/// a lock, a pending waiter locks the waker list only when it registers a new waker, and the
/// completion takes the wakers out of the list before it wakes them.
pub(crate) struct SharedRead {
    slot: Arc<Slot>,
    /// Where this waiter's waker is in the slot's list, if it has registered one since the list
    /// was last emptied.
    parked: Option<Parked>,
    /// What this waiter awaits instead once the completer has been dropped.
    abandoned: Option<SegmentFuture>,
}

/// A registered waker: its index in the slot's list, the list's generation, and a copy of the
/// waker to tell whether a later poll's waker differs.
struct Parked {
    index: usize,
    generation: u64,
    waker: Waker,
}

impl SharedRead {
    /// A read that the returned [`Completer`] fills, with `hooks` told of its waiters.
    pub(crate) fn new(hooks: Option<Box<dyn FillHooks>>) -> (Self, Completer) {
        let slot = Arc::new(Slot {
            result: OnceLock::new(),
            hooks: Mutex::new(hooks),
            polled: AtomicBool::new(false),
            abandoned: AtomicBool::new(false),
            wakers: Mutex::new(Vec::new()),
            generation: AtomicU64::new(0),
        });
        let completer = Completer {
            slot: Arc::downgrade(&slot),
            done: false,
        };
        (Self::waiter(slot), completer)
    }

    fn waiter(slot: Arc<Slot>) -> Self {
        Self {
            slot,
            parked: None,
            abandoned: None,
        }
    }

    /// A weak handle to the read.
    pub(crate) fn downgrade(&self) -> WeakSharedRead {
        WeakSharedRead(Arc::downgrade(&self.slot))
    }

    fn poll_read(&mut self, cx: &mut Context<'_>) -> Poll<SharedVortexResult<BufferHandle>> {
        if let Some(result) = self.slot.result.get() {
            return Poll::Ready(result.clone());
        }
        if let Some(abandoned) = &mut self.abandoned {
            return abandoned.poll_unpin(cx).map_err(Arc::new);
        }
        if !self.slot.polled.load(Ordering::Relaxed)
            && !self.slot.polled.swap(true, Ordering::Relaxed)
            && let Some(hooks) = &*self.slot.hooks.lock()
        {
            hooks.polled();
        }
        self.park(cx.waker());
        // The completer sets the result, or marks the read abandoned, before it empties the list,
        // so a waiter that registered too late sees it here.
        if let Some(result) = self.slot.result.get() {
            return Poll::Ready(result.clone());
        }
        if self.slot.abandoned.load(Ordering::Acquire) {
            let abandoned = self
                .slot
                .hooks
                .lock()
                .as_ref()
                .map(|hooks| hooks.abandoned());
            if let Some(abandoned) = abandoned {
                return self
                    .abandoned
                    .insert(abandoned)
                    .poll_unpin(cx)
                    .map_err(Arc::new);
            }
        }
        Poll::Pending
    }

    /// Registers this waiter's waker unless it has already registered one that wakes the same
    /// task since the list was last emptied.
    fn park(&mut self, waker: &Waker) {
        if let Some(parked) = &self.parked
            && parked.generation == self.slot.generation.load(Ordering::Acquire)
            && parked.waker.will_wake(waker)
        {
            return;
        }

        let mut wakers = self.slot.wakers.lock();
        let generation = self.slot.generation.load(Ordering::Relaxed);
        let index = match &self.parked {
            Some(parked) if parked.generation == generation => {
                wakers[parked.index] = Some(waker.clone());
                parked.index
            }
            _ => {
                wakers.push(Some(waker.clone()));
                wakers.len() - 1
            }
        };
        drop(wakers);
        self.parked = Some(Parked {
            index,
            generation,
            waker: waker.clone(),
        });
    }
}

impl Drop for SharedRead {
    fn drop(&mut self) {
        if self.slot.result.get().is_some() {
            return;
        }
        if let Some(parked) = &self.parked {
            let mut wakers = self.slot.wakers.lock();
            if parked.generation == self.slot.generation.load(Ordering::Relaxed) {
                wakers[parked.index] = None;
            }
        }
    }
}

/// A weak handle to a [`SharedRead`] that doesn't keep its read alive.
pub(crate) struct WeakSharedRead(Weak<Slot>);

impl WeakSharedRead {
    /// A new waiter on the read, unless every waiter has been dropped.
    pub(crate) fn upgrade(&self) -> Option<SharedRead> {
        self.0.upgrade().map(SharedRead::waiter)
    }
}

impl Clone for SharedRead {
    fn clone(&self) -> Self {
        Self::waiter(Arc::clone(&self.slot))
    }
}

impl Future for SharedRead {
    type Output = SharedVortexResult<BufferHandle>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().poll_read(cx)
    }
}

#[cfg(test)]
mod tests {
    use std::task::Context;

    use futures::FutureExt;
    use futures::executor::block_on;
    use futures::future::join_all;
    use vortex_buffer::ByteBuffer;

    use super::*;

    fn buffer(byte: u8) -> BufferHandle {
        BufferHandle::new_host(ByteBuffer::from(vec![byte; 4]))
    }

    #[test]
    fn completion_reaches_every_waiter() -> VortexResult<()> {
        let (shared, completer) = SharedRead::new(None);
        let mut waiters = (0..16).map(|_| shared.clone()).collect::<Vec<_>>();
        for waiter in &mut waiters {
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
        }
        let threads = waiters
            .into_iter()
            .map(|waiter| std::thread::spawn(move || block_on(waiter)))
            .collect::<Vec<_>>();
        completer.complete(Ok(buffer(7)));
        for thread in threads {
            assert_eq!(
                thread
                    .join()
                    .map_err(|_| vortex_err!("waiter panicked"))??
                    .unwrap_host(),
                ByteBuffer::from(vec![7; 4])
            );
        }
        Ok(())
    }

    #[test]
    fn late_waiters_get_the_result() -> VortexResult<()> {
        let (shared, completer) = SharedRead::new(None);
        completer.complete(Ok(buffer(3)));
        let late = shared
            .downgrade()
            .upgrade()
            .ok_or_else(|| vortex_err!("every waiter dropped"))?;
        assert_eq!(block_on(late)?.unwrap_host(), ByteBuffer::from(vec![3; 4]));
        Ok(())
    }

    #[test]
    fn dropping_every_waiter_cancels_the_read() {
        let (shared, completer) = SharedRead::new(None);
        let weak = shared.downgrade();
        let mut waiter = shared.clone();
        assert!(
            waiter
                .poll_unpin(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(shared);
        assert!(!completer.is_canceled());
        drop(waiter);
        assert!(completer.is_canceled());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn errors_reach_every_waiter() {
        let (shared, completer) = SharedRead::new(None);
        let waiters = (0..8).map(|_| shared.clone()).collect::<Vec<_>>();
        completer.complete(Err(vortex_err!("disk on fire")));
        let results = block_on(join_all(waiters));
        let Err(first) = &results[0] else {
            panic!("the read failed");
        };
        for result in &results {
            assert!(matches!(result, Err(err) if Arc::ptr_eq(err, first)));
        }
        assert!(matches!(
            block_on(shared).map(|_| ()),
            Err(err) if err.to_string().contains("disk on fire")
        ));
    }
}
