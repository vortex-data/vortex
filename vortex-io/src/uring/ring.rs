// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The shared ring: queues operations as their futures are created, submits on first poll.

use std::ffi::CString;
use std::fmt;
use std::io;
use std::os::fd::FromRawFd;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;
use std::thread::JoinHandle;

use io_uring::IoUring;
use io_uring::opcode;
use io_uring::squeue;
use parking_lot::Mutex;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// The `user_data` of the no-op that wakes the reaper to shut down. It names no operation.
const WAKE_TOKEN: u64 = u64::MAX;

/// `IORING_ENTER_GETEVENTS`, stable kernel ABI.
const ENTER_GETEVENTS: u32 = 1;

/// One io_uring shared by every file opened through it.
///
/// Cloning shares the ring. The ring's reaper thread stops, after every operation in flight has
/// completed, when the last clone and the last reader opened through it are dropped.
#[derive(Clone)]
pub struct UringRing(Arc<Owner>);

impl UringRing {
    /// A ring with room for `entries` queued submissions. The kernel rounds it up to a power of
    /// two. More operations than that can be in flight; a full queue is submitted to make room.
    pub fn new(entries: u32) -> VortexResult<Self> {
        let ring = IoUring::builder()
            .setup_cqsize(entries.saturating_mul(4))
            .build(entries)
            .map_err(|err| vortex_err!("creating an io_uring: {err}"))?;
        let shared = Arc::new(Shared {
            ring,
            queued: Mutex::new(0),
            cq: Mutex::new(()),
            ops: Mutex::default(),
            shutdown: AtomicBool::new(false),
            stats: Counters::default(),
        });
        let reaper = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("vortex-uring-reaper".into())
                .spawn(move || shared.reap_until_shutdown())
                .map_err(|err| vortex_err!("starting the io_uring reaper: {err}"))?
        };
        Ok(Self(Arc::new(Owner {
            shared,
            reaper: Some(reaper),
        })))
    }

    /// What the ring has done so far.
    pub fn stats(&self) -> UringStats {
        self.0.shared.stats.snapshot()
    }

    /// Submits every queued operation now, without waiting for a poll.
    pub fn flush(&self) -> VortexResult<()> {
        self.0.shared.flush()
    }

    pub(super) fn shared(&self) -> &Arc<Shared> {
        &self.0.shared
    }
}

impl fmt::Debug for UringRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UringRing")
            .field("stats", &self.stats())
            .finish()
    }
}

/// Counts of a ring's work, for checking how operations were batched.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UringStats {
    /// Operations queued, including continuations of short reads.
    pub queued: u64,
    /// `io_uring_enter` calls that submitted at least one operation.
    pub submits: u64,
    /// Operations those calls submitted.
    pub submitted: u64,
    /// Completions reaped by a polling thread right after it submitted.
    pub reaped_inline: u64,
    /// Completions reaped by the reaper thread after it waited.
    pub reaped_by_reaper: u64,
    /// `io_uring_enter` calls the reaper made to wait.
    pub waits: u64,
}

#[derive(Default)]
struct Counters {
    queued: AtomicU64,
    submits: AtomicU64,
    submitted: AtomicU64,
    reaped_inline: AtomicU64,
    reaped_by_reaper: AtomicU64,
    waits: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> UringStats {
        UringStats {
            queued: self.queued.load(Ordering::Relaxed),
            submits: self.submits.load(Ordering::Relaxed),
            submitted: self.submitted.load(Ordering::Relaxed),
            reaped_inline: self.reaped_inline.load(Ordering::Relaxed),
            reaped_by_reaper: self.reaped_by_reaper.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
        }
    }
}

struct Owner {
    shared: Arc<Shared>,
    reaper: Option<JoinHandle<()>>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Wake the reaper; it exits once every operation in flight has completed.
        let nop = opcode::Nop::new().build().user_data(WAKE_TOKEN);
        if self.shared.push(nop).and_then(|()| self.shared.flush()).is_err() {
            tracing::warn!("io_uring: could not wake the reaper to shut down");
            return;
        }
        if let Some(reaper) = self.reaper.take()
            && reaper.join().is_err()
        {
            tracing::warn!("io_uring reaper panicked");
        }
    }
}

/// What an operation's memory and descriptors are, kept alive until the kernel completes it.
pub(super) enum Keep {
    /// A read into `buf`'s spare capacity, from `fd`.
    Read {
        buf: ByteBufferMut,
        _fd: Arc<OwnedFd>,
    },
    /// An open of `path`.
    Path(CString),
}

enum State {
    Waiting(Option<Waker>),
    Done(i32),
}

struct Op {
    state: State,
    /// The future was dropped; free the op when it completes.
    abandoned: bool,
    keep: Keep,
}

#[derive(Default)]
struct Ops {
    slots: Vec<Option<Op>>,
    free: Vec<usize>,
    live: usize,
}

impl Ops {
    fn insert(&mut self, op: Op) -> usize {
        self.live += 1;
        if let Some(slot) = self.free.pop() {
            self.slots[slot] = Some(op);
            slot
        } else {
            self.slots.push(Some(op));
            self.slots.len() - 1
        }
    }

    fn remove(&mut self, slot: usize) -> Op {
        let op = self.slots[slot]
            .take()
            .vortex_expect("io_uring: removing a free slot");
        self.free.push(slot);
        self.live -= 1;
        op
    }

    fn get_mut(&mut self, slot: usize) -> &mut Op {
        self.slots[slot]
            .as_mut()
            .vortex_expect("io_uring: operation slot is free")
    }
}

pub(super) struct Shared {
    ring: IoUring,
    /// Operations pushed to the submission queue and not yet submitted. Also serialises pushes.
    queued: Mutex<u32>,
    /// Serialises reaping.
    cq: Mutex<()>,
    ops: Mutex<Ops>,
    shutdown: AtomicBool,
    stats: Counters,
}

impl Shared {
    /// Queues an operation without submitting it. `build` makes the submission from the kept
    /// memory, which stays where it is until the operation completes.
    pub(super) fn queue(
        &self,
        keep: Keep,
        build: impl FnOnce(&mut Keep) -> squeue::Entry,
    ) -> VortexResult<u64> {
        let (slot, entry) = {
            let mut ops = self.ops.lock();
            let slot = ops.insert(Op {
                state: State::Waiting(None),
                abandoned: false,
                keep,
            });
            let entry = build(&mut ops.get_mut(slot).keep);
            (slot, entry)
        };
        let token = slot as u64;
        if let Err(err) = self.push(entry.user_data(token)) {
            self.ops.lock().remove(slot);
            return Err(err);
        }
        self.stats.queued.fetch_add(1, Ordering::Relaxed);
        Ok(token)
    }

    fn push(&self, entry: squeue::Entry) -> VortexResult<()> {
        let mut queued = self.queued.lock();
        loop {
            // SAFETY: `queued` serialises every use of the submission queue.
            let mut sq = unsafe { self.ring.submission_shared() };
            // SAFETY: the memory the entry points to is kept in its op until it completes.
            if unsafe { sq.push(&entry) }.is_ok() {
                sq.sync();
                *queued += 1;
                return Ok(());
            }
            drop(sq);
            self.submit_locked(&mut queued)?;
        }
    }

    /// Submits every queued operation in one `io_uring_enter`.
    pub(super) fn flush(&self) -> VortexResult<()> {
        let mut queued = self.queued.lock();
        self.submit_locked(&mut queued)
    }

    fn submit_locked(&self, queued: &mut u32) -> VortexResult<()> {
        while *queued > 0 {
            match self.ring.submitter().submit() {
                Ok(n) => {
                    self.stats.submits.fetch_add(1, Ordering::Relaxed);
                    self.stats.submitted.fetch_add(n as u64, Ordering::Relaxed);
                    *queued = queued.saturating_sub(n as u32);
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => return Err(vortex_err!("io_uring submit: {err}")),
            }
        }
        Ok(())
    }

    /// Delivers every completion the kernel has posted. Returns how many it delivered.
    fn reap(&self, block: bool) -> u64 {
        let guard = if block {
            Some(self.cq.lock())
        } else {
            self.cq.try_lock()
        };
        let Some(_guard) = guard else {
            return 0;
        };
        // SAFETY: `cq` serialises every use of the completion queue.
        let completions: Vec<(u64, i32)> = unsafe { self.ring.completion_shared() }
            .map(|cqe| (cqe.user_data(), cqe.result()))
            .collect();
        let mut wakers = Vec::new();
        let mut delivered = 0;
        {
            let mut ops = self.ops.lock();
            for (token, result) in completions {
                if token == WAKE_TOKEN {
                    continue;
                }
                delivered += 1;
                let slot = usize::try_from(token).vortex_expect("io_uring token fits usize");
                let op = ops.get_mut(slot);
                if op.abandoned {
                    close_opened(ops.remove(slot).keep, result);
                    continue;
                }
                if let State::Waiting(Some(waker)) =
                    std::mem::replace(&mut op.state, State::Done(result))
                {
                    wakers.push(waker);
                }
            }
        }
        wakers.into_iter().for_each(Waker::wake);
        delivered
    }

    fn reap_until_shutdown(&self) {
        loop {
            // SAFETY: waits for one completion; submits nothing and passes no argument.
            let waited = unsafe {
                self.ring
                    .submitter()
                    .enter::<libc::sigset_t>(0, 1, ENTER_GETEVENTS, None)
            };
            match waited {
                Ok(_) => {}
                Err(err) if err.kind() == io::ErrorKind::Interrupted => {}
                Err(err) => {
                    tracing::error!("io_uring reaper: {err}");
                    return;
                }
            }
            self.stats.waits.fetch_add(1, Ordering::Relaxed);
            let delivered = self.reap(true);
            self.stats
                .reaped_by_reaper
                .fetch_add(delivered, Ordering::Relaxed);
            if self.shutdown.load(Ordering::Acquire) && self.ops.lock().live == 0 {
                return;
            }
        }
    }

    /// Polls the operation `token`: submits whatever is queued, reaps what has completed, and
    /// returns the result and its kept memory once the operation is done.
    pub(super) fn poll_op(
        &self,
        token: u64,
        cx: &mut Context<'_>,
    ) -> Poll<VortexResult<(i32, Keep)>> {
        if let Err(err) = self.flush() {
            return Poll::Ready(Err(err));
        }
        let delivered = self.reap(false);
        self.stats
            .reaped_inline
            .fetch_add(delivered, Ordering::Relaxed);
        let slot = usize::try_from(token).vortex_expect("io_uring token fits usize");
        let mut ops = self.ops.lock();
        let op = ops.get_mut(slot);
        match &mut op.state {
            State::Done(result) => {
                let result = *result;
                let op = ops.remove(slot);
                Poll::Ready(Ok((result, op.keep)))
            }
            State::Waiting(waker) => {
                match waker {
                    Some(waker) if waker.will_wake(cx.waker()) => {}
                    _ => *waker = Some(cx.waker().clone()),
                }
                Poll::Pending
            }
        }
    }

    /// Gives up on `token`: its memory is freed when the kernel completes it.
    pub(super) fn abandon(&self, token: u64) {
        let slot = usize::try_from(token).vortex_expect("io_uring token fits usize");
        let mut ops = self.ops.lock();
        if let State::Done(result) = ops.get_mut(slot).state {
            close_opened(ops.remove(slot).keep, result);
        } else {
            ops.get_mut(slot).abandoned = true;
        }
    }
}

/// Closes the descriptor an abandoned open returned, which nothing else owns.
fn close_opened(keep: Keep, result: i32) {
    if let Keep::Path(_) = keep
        && result >= 0
    {
        // SAFETY: the open's future was dropped, so the new descriptor has no other owner.
        drop(unsafe { OwnedFd::from_raw_fd(result) });
    }
}
