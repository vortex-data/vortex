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
use std::task::Wake;
use std::task::Waker;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use super::SegmentFuture;
use super::SegmentId;
use super::SegmentSource;

/// A [`SegmentSource`] that allows multiple requesters to await the same underlying segment
/// request.
pub struct SharedSegmentSource<S> {
    inner: S,
    in_flight: DashMap<SegmentId, WeakSharedSegment>,
}

impl<S: SegmentSource> SharedSegmentSource<S> {
    /// Create a new `SharedSegmentSource` wrapping the provided inner source.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            in_flight: DashMap::with_hasher_and_shard_amount(Default::default(), 64),
        }
    }
}

impl<S: SegmentSource> SegmentSource for SharedSegmentSource<S> {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.request_shared(id).map_err(VortexError::from).boxed()
    }

    fn request_shared(&self, id: SegmentId) -> SharedSegment {
        loop {
            match self.in_flight.entry(id) {
                Entry::Occupied(e) => {
                    if let Some(shared) = e.get().upgrade() {
                        return shared;
                    }
                    // Every waiter has gone, so the read was dropped. Start another.
                    e.remove();
                }
                Entry::Vacant(e) => {
                    let shared = self.inner.request_shared(id);
                    e.insert(shared.downgrade());
                    return shared;
                }
            }
        }
    }
}

/// One segment read awaited by any number of waiters.
///
/// Each clone is a waiter, and the read is dropped, cancelling it, once every waiter is dropped.
/// The result, error or not, goes to every waiter.
///
/// This replaces [`futures::future::Shared`], which serializes many threads awaiting one read: each
/// poll of a pending `Shared` locks the mutex around its waker list, and its completion wakes every
/// waiter while holding that lock. Here, a waiter whose read has finished reads the result without
/// taking a lock, a pending waiter locks the waker list only when it registers a new waker, and the
/// completion takes the wakers out of the list before it wakes them.
///
/// A source that completes its reads elsewhere, such as [`FileSegmentSource`](super::FileSegmentSource)
/// whose read driver resolves them, fills the slot directly through a [`Completer`], so the
/// completion wakes every waiter at once from the driver's thread.
///
/// Otherwise the waiters poll the read future, one at a time, with a waker that wakes every parked
/// waiter. Whichever waiter sees the read finish completes it for everyone.
pub struct SharedSegment {
    slot: Arc<Slot>,
    /// Where this waiter's waker is in the slot's list, if it has registered one since the list
    /// was last emptied.
    parked: Option<Parked>,
}

/// What a waiter's attempt to poll the read did.
enum Drive {
    /// The read completed.
    Done(SharedVortexResult<BufferHandle>),
    /// The read was polled, now or by another waiter.
    Polled,
    /// There was no read to poll.
    Skipped,
}

/// A weak handle to a [`SharedSegment`] that doesn't keep its read alive.
pub struct WeakSharedSegment(Weak<Slot>);

impl WeakSharedSegment {
    /// A new waiter on the read, unless every waiter has been dropped.
    pub fn upgrade(&self) -> Option<SharedSegment> {
        self.0.upgrade().map(SharedSegment::waiter)
    }
}

/// A registered waker: its index in the slot's list, the list's generation, and a copy of the
/// waker to tell whether a later poll's waker differs.
struct Parked {
    index: usize,
    generation: u64,
    waker: Waker,
}

struct Slot {
    result: OnceLock<SharedVortexResult<BufferHandle>>,
    /// The read, until it has completed, for a slot that waiters drive. Waiters only ever try to
    /// lock it, never wait for it. A filled slot has none unless its completer was dropped.
    read: Mutex<Option<SegmentFuture>>,
    /// Notified of a filled slot's waiters, which drive nothing.
    hooks: Option<Box<dyn FillHooks>>,
    polled: AtomicBool,
    parking: Arc<Parking>,
    /// Wakes every parked waiter, for the read to be polled with.
    progress: Waker,
}

/// The parked waiters' wakers. It holds no reference to the read, so a waker the read keeps does
/// not keep the read alive.
#[derive(Default)]
struct Parking {
    wakers: Mutex<Vec<Option<Waker>>>,
    /// Increments each time the list is emptied.
    generation: AtomicU64,
}

impl Parking {
    /// Wakes every parked waiter, outside the lock, leaving them registered.
    fn wake_parked(&self) {
        let wakers = self
            .wakers
            .lock()
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        for waker in wakers {
            waker.wake();
        }
    }

    /// Empties the list and wakes every waiter that was on it, outside the lock.
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

impl Wake for Parking {
    fn wake(self: Arc<Self>) {
        self.wake_parked();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.wake_parked();
    }
}

impl SharedSegment {
    /// Share `read` between any number of waiters.
    pub fn new(read: SegmentFuture) -> Self {
        Self::waiter(Slot::new(Some(read), None))
    }

    /// A read that the returned [`Completer`] fills, with `hooks` told of its waiters.
    pub(crate) fn filled(hooks: Option<Box<dyn FillHooks>>) -> (Self, Completer) {
        let slot = Slot::new(None, hooks);
        let completer = Completer {
            slot: Arc::downgrade(&slot),
            done: false,
        };
        (Self::waiter(slot), completer)
    }

    fn waiter(slot: Arc<Slot>) -> Self {
        Self { slot, parked: None }
    }

    /// A weak handle to the read.
    pub fn downgrade(&self) -> WeakSharedSegment {
        WeakSharedSegment(Arc::downgrade(&self.slot))
    }

    /// Polls the read if no other waiter is polling it, completing it for every waiter if it has
    /// finished.
    fn try_drive(&mut self) -> Drive {
        let Some(mut read) = self.slot.read.try_lock() else {
            // Another waiter is polling the read, and the read's next wakeup wakes this one too.
            return Drive::Polled;
        };
        // The completer sets the result before it empties `read`, so an empty `read` means either
        // the result is there or a completer will fill the slot.
        let Some(future) = read.as_mut() else {
            return Drive::Skipped;
        };
        let mut panicked = PanicGuard {
            slot: &self.slot,
            armed: true,
        };
        let poll = future.poll_unpin(&mut Context::from_waker(&self.slot.progress));
        panicked.armed = false;

        let Poll::Ready(result) = poll else {
            return Drive::Polled;
        };
        let result = self.slot.complete(result.map_err(Arc::new));
        *read = None;
        drop(read);
        // The slot has emptied the list, so this waiter's registration is gone.
        self.parked = None;
        Drive::Done(result)
    }

    /// Registers this waiter's waker unless it has already registered one that wakes the same
    /// task since the list was last emptied.
    fn park(&mut self, waker: &Waker) {
        if let Some(parked) = &self.parked
            && parked.generation == self.slot.parking.generation.load(Ordering::Acquire)
            && parked.waker.will_wake(waker)
        {
            return;
        }

        let mut wakers = self.slot.parking.wakers.lock();
        let generation = self.slot.parking.generation.load(Ordering::Relaxed);
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

impl Slot {
    fn new(read: Option<SegmentFuture>, hooks: Option<Box<dyn FillHooks>>) -> Arc<Self> {
        let parking = Arc::new(Parking::default());
        Arc::new(Self {
            result: OnceLock::new(),
            read: Mutex::new(read),
            hooks,
            polled: AtomicBool::new(false),
            progress: Waker::from(Arc::clone(&parking)),
            parking,
        })
    }

    /// Stores the result and wakes every parked waiter, returning the result that won if another
    /// completion got there first.
    fn complete(
        &self,
        result: SharedVortexResult<BufferHandle>,
    ) -> SharedVortexResult<BufferHandle> {
        let result = self.result.get_or_init(|| result).clone();
        self.parking.wake_all();
        result
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.result.get().is_none()
            && let Some(hooks) = &self.hooks
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
    /// The completer was dropped without completing the read. The returned future, driven by the
    /// waiters, gives their result instead.
    fn abandoned(&self) -> SegmentFuture;
}

/// Fills a [`SharedSegment`] created by [`SharedSegment::filled`]. It doesn't keep the read's
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
            drop(slot.complete(result.map_err(Arc::new)));
        }
    }
}

impl Drop for Completer {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let Some(slot) = self.slot.upgrade() else {
            return;
        };
        if slot.result.get().is_some() {
            return;
        }
        match &slot.hooks {
            // The waiters drive the replacement, starting when this wakes them.
            Some(hooks) => {
                *slot.read.lock() = Some(hooks.abandoned());
                slot.parking.wake_all();
            }
            None => drop(slot.complete(Err(Arc::new(vortex_err!(
                "Segment read dropped before it completed"
            ))))),
        }
    }
}

/// Completes the read with an error, waking every waiter, if polling it panics. The panic itself
/// reaches only the waiter that polled.
struct PanicGuard<'a> {
    slot: &'a Slot,
    armed: bool,
}

impl Drop for PanicGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            drop(
                self.slot
                    .complete(Err(Arc::new(vortex_err!("Shared segment read panicked")))),
            );
        }
    }
}

impl Clone for SharedSegment {
    fn clone(&self) -> Self {
        Self::waiter(Arc::clone(&self.slot))
    }
}

impl Future for SharedSegment {
    type Output = SharedVortexResult<BufferHandle>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if let Some(result) = this.slot.result.get() {
            return Poll::Ready(result.clone());
        }
        if let Some(hooks) = &this.slot.hooks
            && !this.slot.polled.load(Ordering::Relaxed)
            && !this.slot.polled.swap(true, Ordering::Relaxed)
        {
            hooks.polled();
        }
        let drive = this.try_drive();
        if let Drive::Done(result) = drive {
            return Poll::Ready(result);
        }
        this.park(cx.waker());
        // A completion that emptied the list before this waiter registered set the result first,
        // and a dropped completer installed the read to drive first.
        if let Some(result) = this.slot.result.get() {
            return Poll::Ready(result.clone());
        }
        if matches!(drive, Drive::Skipped)
            && let Drive::Done(result) = this.try_drive()
        {
            return Poll::Ready(result);
        }
        Poll::Pending
    }
}

impl Drop for SharedSegment {
    fn drop(&mut self) {
        if self.slot.result.get().is_some() {
            return;
        }
        if let Some(parked) = &self.parked {
            let mut wakers = self.slot.parking.wakers.lock();
            if parked.generation == self.slot.parking.generation.load(Ordering::Relaxed) {
                wakers[parked.index] = None;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_in_result)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Context;
    use std::task::Poll;
    use std::task::Waker;

    use futures::FutureExt;
    use futures::channel::oneshot;
    use futures::executor::block_on;
    use futures::future::join_all;
    use futures::task::ArcWake;
    use futures::task::waker;
    use parking_lot::Mutex;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;

    fn buffer(byte: u8) -> BufferHandle {
        BufferHandle::new_host(ByteBuffer::from(vec![byte; 4]))
    }

    /// A read that completes when the test sends on the returned channel, and records whether it
    /// was dropped.
    fn gated() -> (
        SharedSegment,
        oneshot::Sender<VortexResult<BufferHandle>>,
        Arc<AtomicUsize>,
    ) {
        struct OnDrop(Arc<AtomicUsize>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let (send, recv) = oneshot::channel();
        let dropped = Arc::new(AtomicUsize::new(0));
        let on_drop = OnDrop(Arc::clone(&dropped));
        let read = async move {
            let _on_drop = on_drop;
            recv.await
                .unwrap_or_else(|_| Err(vortex_err!("gate dropped")))
        };
        (SharedSegment::new(read.boxed()), send, dropped)
    }

    #[derive(Default)]
    struct CountingWaker(AtomicUsize);

    impl ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn counting() -> (Arc<CountingWaker>, Waker) {
        let count = Arc::new(CountingWaker::default());
        let waker = waker(Arc::clone(&count));
        (count, waker)
    }

    #[test]
    fn many_waiters_share_one_read() -> VortexResult<()> {
        let reads = Arc::new(AtomicUsize::new(0));
        let shared = {
            let reads = Arc::clone(&reads);
            SharedSegment::new(
                async move {
                    reads.fetch_add(1, Ordering::Relaxed);
                    Ok(buffer(7))
                }
                .boxed(),
            )
        };
        let waiters = (0..64).map(|_| shared.clone()).collect::<Vec<_>>();
        for result in block_on(join_all(waiters)) {
            assert_eq!(result?.unwrap_host(), ByteBuffer::from(vec![7; 4]));
        }
        assert_eq!(reads.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn progress_and_completion_wake_every_parked_waiter() -> VortexResult<()> {
        let (shared, send, _) = gated();
        let mut waiters = (0..8).map(|_| shared.clone()).collect::<Vec<_>>();
        let wakers = (0..8).map(|_| counting()).collect::<Vec<_>>();
        for (waiter, (_, waker)) in waiters.iter_mut().zip(&wakers) {
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(waker))
                    .is_pending()
            );
            // Polling again with the same waker registers nothing new.
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(waker))
                    .is_pending()
            );
        }
        assert_eq!(shared.slot.parking.wakers.lock().len(), 8);

        // The read's wakeup reaches every waiter, so any of them can complete it.
        send.send(Ok(buffer(1))).unwrap();
        for (count, _) in &wakers {
            assert_eq!(count.0.load(Ordering::Relaxed), 1);
        }
        let Poll::Ready(result) = waiters[3].poll_unpin(&mut Context::from_waker(&wakers[3].1))
        else {
            panic!("the read has finished");
        };
        result?;

        // Its completion wakes every waiter again, and empties the list.
        for (count, _) in &wakers {
            assert_eq!(count.0.load(Ordering::Relaxed), 2);
        }
        assert!(shared.slot.parking.wakers.lock().is_empty());
        for (waiter, (_, waker)) in waiters.iter_mut().zip(&wakers) {
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(waker))
                    .is_ready()
            );
        }
        Ok(())
    }

    #[test]
    fn a_filled_read_wakes_every_waiter_once() -> VortexResult<()> {
        let (shared, completer) = SharedSegment::filled(None);
        let mut waiters = (0..8).map(|_| shared.clone()).collect::<Vec<_>>();
        let wakers = (0..8).map(|_| counting()).collect::<Vec<_>>();
        for (waiter, (_, waker)) in waiters.iter_mut().zip(&wakers) {
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(waker))
                    .is_pending()
            );
        }
        assert!(!completer.is_canceled());
        completer.complete(Ok(buffer(2)));
        for (waiter, (count, waker)) in waiters.iter_mut().zip(&wakers) {
            assert_eq!(count.0.load(Ordering::Relaxed), 1);
            let Poll::Ready(result) = waiter.poll_unpin(&mut Context::from_waker(waker)) else {
                panic!("the read has completed");
            };
            assert_eq!(result?.unwrap_host(), ByteBuffer::from(vec![2; 4]));
        }
        Ok(())
    }

    #[test]
    fn a_filled_read_is_canceled_without_waiters() {
        let (shared, completer) = SharedSegment::filled(None);
        let other = shared.clone();
        drop(shared);
        assert!(!completer.is_canceled());
        drop(other);
        assert!(completer.is_canceled());
    }

    #[test]
    fn a_dropped_completer_fails_the_read() {
        let (shared, completer) = SharedSegment::filled(None);
        let mut waiter = shared.clone();
        let (count, waker) = counting();
        assert!(
            waiter
                .poll_unpin(&mut Context::from_waker(&waker))
                .is_pending()
        );
        drop(completer);
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        assert!(block_on(shared).is_err());
        assert!(block_on(waiter).is_err());
    }

    #[test]
    fn late_waiters_get_the_result() -> VortexResult<()> {
        let (shared, send, dropped) = gated();
        send.send(Ok(buffer(3))).unwrap();
        block_on(shared.clone())?;
        assert_eq!(dropped.load(Ordering::Relaxed), 1);

        let late = shared.downgrade().upgrade().expect("a waiter is alive");
        assert_eq!(block_on(late)?.unwrap_host(), ByteBuffer::from(vec![3; 4]));
        assert_eq!(
            block_on(shared)?.unwrap_host(),
            ByteBuffer::from(vec![3; 4])
        );
        Ok(())
    }

    #[test]
    fn dropping_every_waiter_drops_the_read() {
        let (shared, send, dropped) = gated();
        let weak = shared.downgrade();
        let mut waiters = (0..4).map(|_| shared.clone()).collect::<Vec<_>>();
        for waiter in &mut waiters {
            assert!(
                waiter
                    .poll_unpin(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
        }
        drop(shared);
        drop(waiters);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert!(send.is_canceled());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn errors_reach_every_waiter() {
        let (shared, send, _) = gated();
        let waiters = (0..8).map(|_| shared.clone()).collect::<Vec<_>>();
        send.send(Err(vortex_err!("disk on fire"))).unwrap();
        let results = block_on(join_all(waiters));
        let first = results[0].as_ref().unwrap_err();
        for result in &results {
            let err = result.as_ref().unwrap_err();
            assert!(Arc::ptr_eq(err, first));
            assert!(err.to_string().contains("disk on fire"));
        }
    }

    #[test]
    fn the_last_waiter_to_poll_can_drop() -> VortexResult<()> {
        let (shared, send, dropped) = gated();
        let mut last = shared.clone();
        let mut other = shared;
        let (other_count, other_waker) = counting();
        assert!(
            other
                .poll_unpin(&mut Context::from_waker(&other_waker))
                .is_pending()
        );
        assert!(
            last.poll_unpin(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        drop(last);

        send.send(Ok(buffer(9))).unwrap();
        assert_eq!(other_count.0.load(Ordering::Relaxed), 1);
        assert_eq!(block_on(other)?.len(), 4);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn a_panicking_read_fails_the_other_waiters() {
        let shared = SharedSegment::new(
            futures::future::poll_fn(|_| -> Poll<VortexResult<BufferHandle>> {
                panic!("driver panicked")
            })
            .boxed(),
        );
        let mut other = shared.clone();
        let (count, waker) = counting();
        // `other` cannot drive while the panicking waiter holds the read, so it parks first.
        let read = shared.slot.read.lock();
        assert!(
            other
                .poll_unpin(&mut Context::from_waker(&waker))
                .is_pending()
        );
        drop(read);

        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| block_on(shared.clone())));
        assert!(panicked.is_err());
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        let Poll::Ready(Err(err)) = other.poll_unpin(&mut Context::from_waker(&waker)) else {
            panic!("the read failed");
        };
        assert!(err.to_string().contains("panicked"));
    }

    /// Many threads await many reads that complete on another thread; none may miss its wakeup.
    #[test]
    fn concurrent_waiters_never_miss_a_completion() -> VortexResult<()> {
        const READS: usize = 256;
        const THREADS: usize = 8;
        for _ in 0..20 {
            let gates = (0..READS).map(|_| gated()).collect::<Vec<_>>();
            let shared = gates.iter().map(|(s, ..)| s.clone()).collect::<Vec<_>>();
            let sends = Mutex::new(
                gates
                    .into_iter()
                    .map(|(_, send, _)| send)
                    .collect::<Vec<_>>(),
            );
            std::thread::scope(|scope| -> VortexResult<()> {
                let handles = (0..THREADS)
                    .map(|_| {
                        let waiters = shared.clone();
                        scope.spawn(move || block_on(join_all(waiters)))
                    })
                    .collect::<Vec<_>>();
                for send in sends.lock().drain(..) {
                    send.send(Ok(buffer(0))).unwrap();
                }
                drop(shared);
                for handle in handles {
                    for result in handle.join().unwrap() {
                        result?;
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}
