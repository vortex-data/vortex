// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use kanal::Receiver;
use kanal::Sender;
use kanal::unbounded;
use parking_lot::Mutex;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;

const DEFAULT_MAX_THREADS: usize = 500;
const MIN_MAX_THREADS: usize = 1;
const MAX_MAX_THREADS: usize = 10_000;
const KEEP_ALIVE: Duration = Duration::from_millis(500);

/// A dynamically sized pool for blocking I/O owned by a single Vortex runtime.
pub(crate) struct BlockingPool {
    sender: Sender<Job>,
    receiver: Receiver<Job>,
    state: Arc<Mutex<PoolState>>,
    thread_limit: usize,
}

impl Default for BlockingPool {
    fn default() -> Self {
        Self::new(max_threads())
    }
}

impl BlockingPool {
    fn new(thread_limit: usize) -> Self {
        assert!(thread_limit > 0, "blocking thread limit must be non-zero");
        let (sender, receiver) = unbounded();
        Self {
            sender,
            receiver,
            state: Arc::new(Mutex::new(PoolState::default())),
            thread_limit,
        }
    }

    pub(crate) fn spawn(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let cancelled = Arc::new(AtomicBool::new(false));
        if self
            .sender
            .send(Job {
                cancelled: Arc::clone(&cancelled),
                task,
            })
            .is_err()
        {
            vortex_panic!("cannot spawn blocking work on a shut down runtime");
        }
        self.grow();
        Box::new(BlockingAbortHandle { cancelled })
    }

    fn grow(&self) {
        let mut state = self.state.lock();
        // An unbounded kanal send hands work directly to a waiting receiver. A queued job therefore
        // means every existing worker is busy (or transitioning out of an idle wait), so add one
        // worker for this submission while capacity remains.
        if self.sender.is_empty() || state.thread_count >= self.thread_limit {
            return;
        }
        state.thread_count += 1;

        let receiver = self.receiver.clone();
        let worker_state = Arc::clone(&self.state);
        if let Err(error) = std::thread::Builder::new()
            .name("vortex-blocking-io".to_string())
            .spawn(move || worker_loop(receiver, worker_state))
        {
            state.thread_count -= 1;
            vortex_panic!("failed to spawn a blocking I/O worker: {error}");
        }
    }
}

impl Drop for BlockingPool {
    fn drop(&mut self) {
        drop(self.sender.close());
    }
}

#[derive(Default)]
struct PoolState {
    thread_count: usize,
}

struct Job {
    cancelled: Arc<AtomicBool>,
    task: Box<dyn FnOnce() + Send + 'static>,
}

impl Job {
    fn run(self) {
        if !self.cancelled.load(Ordering::Acquire) {
            (self.task)();
        }
    }
}

struct BlockingAbortHandle {
    cancelled: Arc<AtomicBool>,
}

impl AbortHandle for BlockingAbortHandle {
    fn abort(self: Box<Self>) {
        self.cancelled.store(true, Ordering::Release);
    }
}

fn worker_loop(receiver: Receiver<Job>, state: Arc<Mutex<PoolState>>) {
    loop {
        match receiver.recv_timeout(KEEP_ALIVE) {
            Ok(job) => {
                // `Handle::spawn_blocking` catches task panics so they can be propagated to the
                // caller. Keep this boundary too, so an unexpected panic does not corrupt the
                // worker counts and permanently reduce the pool's capacity.
                drop(catch_unwind(AssertUnwindSafe(|| job.run())));
            }
            Err(kanal::ReceiveErrorTimeout::Timeout) => {
                let mut state = state.lock();
                if receiver.is_empty() {
                    state.thread_count -= 1;
                    return;
                }
            }
            Err(kanal::ReceiveErrorTimeout::Closed | kanal::ReceiveErrorTimeout::SendClosed) => {
                let mut state = state.lock();
                state.thread_count -= 1;
                return;
            }
        }
    }
}

fn max_threads() -> usize {
    std::env::var("BLOCKING_MAX_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(MIN_MAX_THREADS, MAX_MAX_THREADS))
        .unwrap_or(DEFAULT_MAX_THREADS)
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::mpsc;

    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;

    #[test]
    fn test_executes_jobs() -> VortexResult<()> {
        let pool = BlockingPool::new(2);
        let (send, recv) = mpsc::sync_channel(1);
        drop(pool.spawn(Box::new(move || {
            let _ = send.send(42);
        })));

        assert_eq!(
            recv.recv_timeout(Duration::from_secs(5))
                .map_err(|error| vortex_err!("blocking job did not finish: {error}"))?,
            42
        );
        Ok(())
    }

    #[test]
    fn test_aborts_queued_jobs() -> VortexResult<()> {
        let pool = BlockingPool::new(1);
        let (started_send, started_recv) = mpsc::sync_channel(1);
        let (release_send, release_recv) = mpsc::sync_channel(1);
        drop(pool.spawn(Box::new(move || {
            let _ = started_send.send(());
            let _ = release_recv.recv();
        })));
        started_recv
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| vortex_err!("first blocking job did not start: {error}"))?;

        let (ran_send, ran_recv) = mpsc::sync_channel(1);
        pool.spawn(Box::new(move || {
            let _ = ran_send.send(());
        }))
        .abort();

        let (done_send, done_recv) = mpsc::sync_channel(1);
        drop(pool.spawn(Box::new(move || {
            let _ = done_send.send(());
        })));
        let _ = release_send.send(());
        done_recv
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| vortex_err!("blocking queue did not drain: {error}"))?;
        assert!(ran_recv.try_recv().is_err());
        Ok(())
    }

    #[test]
    fn test_grows_for_concurrent_jobs() -> VortexResult<()> {
        let pool = BlockingPool::new(2);
        let barrier = Arc::new(Barrier::new(3));
        let (started_send, started_recv) = mpsc::sync_channel(2);

        for _ in 0..2 {
            let barrier = Arc::clone(&barrier);
            let started_send = started_send.clone();
            drop(pool.spawn(Box::new(move || {
                let _ = started_send.send(());
                barrier.wait();
            })));
        }

        for _ in 0..2 {
            started_recv
                .recv_timeout(Duration::from_secs(5))
                .map_err(|error| vortex_err!("blocking job did not start: {error}"))?;
        }
        barrier.wait();
        Ok(())
    }
}
