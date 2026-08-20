// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use crossbeam_channel::TryRecvError;
use crossbeam_channel::unbounded;
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
        let mut state = self.state.lock();
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
        state.queued_job_count += 1;
        if state.queued_job_count > state.idle_thread_count {
            self.grow(&mut state);
        }
        Box::new(BlockingAbortHandle { cancelled })
    }

    fn grow(&self, state: &mut PoolState) {
        if state.thread_count >= self.thread_limit {
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

#[derive(Default)]
struct PoolState {
    // All three counters are updated while holding the pool mutex. Keeping queued work and idle
    // capacity in the same state makes thread growth independent of channel timing.
    thread_count: usize,
    idle_thread_count: usize,
    queued_job_count: usize,
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
        let mut pool_state = state.lock();
        match receiver.try_recv() {
            Ok(job) => {
                pool_state.queued_job_count -= 1;
                drop(pool_state);
                run_job(job);
                continue;
            }
            Err(TryRecvError::Disconnected) => {
                pool_state.thread_count -= 1;
                return;
            }
            Err(TryRecvError::Empty) => {
                pool_state.idle_thread_count += 1;
            }
        }
        drop(pool_state);

        match receiver.recv_timeout(KEEP_ALIVE) {
            Ok(job) => {
                let mut pool_state = state.lock();
                pool_state.idle_thread_count -= 1;
                pool_state.queued_job_count -= 1;
                drop(pool_state);
                run_job(job);
            }
            Err(RecvTimeoutError::Timeout) => {
                let mut pool_state = state.lock();
                match receiver.try_recv() {
                    Ok(job) => {
                        pool_state.idle_thread_count -= 1;
                        pool_state.queued_job_count -= 1;
                        drop(pool_state);
                        run_job(job);
                    }
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                        pool_state.idle_thread_count -= 1;
                        pool_state.thread_count -= 1;
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let mut pool_state = state.lock();
                pool_state.idle_thread_count -= 1;
                pool_state.thread_count -= 1;
                return;
            }
        }
    }
}

fn run_job(job: Job) {
    // `Handle::spawn_blocking` catches task panics so they can be propagated to the caller. Keep
    // this boundary too, so an unexpected panic does not corrupt the worker counts and permanently
    // reduce the pool's capacity.
    drop(catch_unwind(AssertUnwindSafe(|| job.run())));
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
    use std::time::Instant;

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

    #[test]
    fn test_reuses_idle_worker() -> VortexResult<()> {
        let pool = BlockingPool::new(2);
        let (first_send, first_recv) = mpsc::sync_channel(1);
        drop(pool.spawn(Box::new(move || {
            let _ = first_send.send(std::thread::current().id());
        })));
        let first_thread = first_recv
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| vortex_err!("first blocking job did not finish: {error}"))?;

        let deadline = Instant::now() + Duration::from_secs(5);
        while pool.state.lock().idle_thread_count != 1 {
            if Instant::now() >= deadline {
                return Err(vortex_err!("blocking worker did not become idle"));
            }
            std::thread::yield_now();
        }

        let (started_send, started_recv) = mpsc::sync_channel(1);
        let (release_send, release_recv) = mpsc::sync_channel(1);
        drop(pool.spawn(Box::new(move || {
            let _ = started_send.send(std::thread::current().id());
            let _ = release_recv.recv();
        })));
        let second_thread = started_recv
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| vortex_err!("second blocking job did not finish: {error}"))?;

        assert_eq!(first_thread, second_thread);
        let state = pool.state.lock();
        assert_eq!(state.thread_count, 1);
        assert_eq!(state.idle_thread_count, 0);
        assert_eq!(state.queued_job_count, 0);
        drop(state);
        let _ = release_send.send(());
        Ok(())
    }
}
