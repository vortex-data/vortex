//! Sync work-pool runtime — crossbeam-deque + N OS threads running
//! `FnOnce` closures directly.
//!
//! Replaces a shared `smol::Executor` for pipeline dispatch. Each
//! submitted pipeline becomes a closure that `block_on`s its driver
//! future on the worker thread that picked it up. While the closure
//! is running the worker is fully busy with that pipeline — no
//! per-batch async re-dispatch overhead. Fairness across long-running
//! pipelines falls out of OS preemption (the worker pool is sized to
//! `num_cpus`, so any pipeline that exceeds its quantum is
//! pre-empted just like a regular thread).
//!
//! Cooperative epoch yielding (so a CPU-bound pipeline can pause and
//! let queued shorter pipelines run) is intentionally deferred — it
//! becomes interesting in a thread-per-core io_uring runtime where
//! the worker is also driving polling; in the present multi-thread
//! pool the OS handles fairness.

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::{self, JoinHandle};

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use parking::{Parker, Unparker};
use parking_lot::Mutex;

/// A unit of work dispatched onto the pool.
pub(crate) type Job = Box<dyn FnOnce() + Send + 'static>;

/// Sync work pool. `spawn` pushes a closure; the next free worker
/// picks it up and runs it to completion.
pub(crate) struct Runtime {
    inner: Arc<RuntimeInner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

struct RuntimeInner {
    injector: Injector<Job>,
    stealers: Box<[Stealer<Job>]>,
    parkers: Box<[ParkerSlot]>,
    shutdown: AtomicBool,
}

/// Shared half of one worker's park slot. The `Parker` itself lives
/// on the worker thread; the `Unparker` clone here lets `spawn`
/// wake the worker when new work arrives.
struct ParkerSlot {
    unparker: Unparker,
    /// `true` while the worker is parked (or about to park). Cleared
    /// on the unpark side via `swap(false)` so only one waker actually
    /// wakes a given parked worker.
    is_parked: AtomicBool,
}

impl Runtime {
    /// Spin up `n_workers` worker threads.
    pub(crate) fn new(n_workers: usize) -> Arc<Self> {
        assert!(n_workers > 0, "runtime needs at least one worker");

        let injector: Injector<Job> = Injector::new();
        let mut deques: Vec<Worker<Job>> = Vec::with_capacity(n_workers);
        let mut stealers: Vec<Stealer<Job>> = Vec::with_capacity(n_workers);
        let mut parkers: Vec<Parker> = Vec::with_capacity(n_workers);
        let mut parker_slots: Vec<ParkerSlot> = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let w: Worker<Job> = Worker::new_fifo();
            stealers.push(w.stealer());
            deques.push(w);
            let (parker, unparker) = parking::pair();
            parkers.push(parker);
            parker_slots.push(ParkerSlot {
                unparker,
                is_parked: AtomicBool::new(false),
            });
        }

        let inner = Arc::new(RuntimeInner {
            injector,
            stealers: stealers.into_boxed_slice(),
            parkers: parker_slots.into_boxed_slice(),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(n_workers);
        for (idx, (deque, parker)) in deques.into_iter().zip(parkers).enumerate() {
            let inner = Arc::clone(&inner);
            let handle = thread::Builder::new()
                .name(format!("engine-pool-{idx}"))
                .spawn(move || worker_main(idx, deque, parker, inner))
                .expect("spawn engine pool worker");
            handles.push(handle);
        }

        Arc::new(Self {
            inner,
            workers: Mutex::new(handles),
        })
    }

    /// Submit a job for execution. Returns immediately once the job
    /// is queued; the actual work runs on some worker thread.
    pub(crate) fn spawn(&self, job: Job) {
        self.inner.injector.push(job);
        self.inner.wake_one();
    }

    /// Initiate shutdown and join all workers. Pending jobs that are
    /// already in flight finish; queued-but-unstarted jobs may not
    /// run (callers should arrange for `spawn` calls to stop before
    /// shutdown).
    pub(crate) fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Release);
        // Wake every worker so they observe shutdown.
        for slot in self.inner.parkers.iter() {
            if slot.is_parked.swap(false, Ordering::AcqRel) {
                slot.unparker.unpark();
            } else {
                // Worker is awake (or another waker took the slot);
                // still send an unpark so a subsequent park returns
                // immediately.
                slot.unparker.unpark();
            }
        }
        let handles = std::mem::take(&mut *self.workers.lock());
        for h in handles {
            drop(h.join());
        }
    }
}

impl RuntimeInner {
    fn wake_one(&self) {
        for slot in self.parkers.iter() {
            if slot.is_parked.swap(false, Ordering::AcqRel) {
                slot.unparker.unpark();
                return;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn set_user_initiated_qos() {
    // Tell macOS to schedule this thread on performance cores. Without
    // this, default-QoS threads land on efficiency cores, which gives
    // ~6 effective cores on a 6P+12E machine instead of the full 18.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn set_user_initiated_qos() {}

fn worker_main(idx: usize, deque: Worker<Job>, parker: Parker, inner: Arc<RuntimeInner>) {
    set_user_initiated_qos();
    let slot = &inner.parkers[idx];
    loop {
        // Drain available work greedily before considering parking.
        while let Some(job) = find_job(idx, &deque, &inner) {
            job();
        }

        if inner.shutdown.load(Ordering::Acquire) {
            return;
        }

        // No work available — get ready to park. Mark ourselves
        // parked *before* the final check so any concurrent `spawn`
        // is guaranteed to either (a) find us already parked and
        // unpark, or (b) push a job we'll see in the recheck.
        slot.is_parked.store(true, Ordering::Release);

        // Recheck both queues *after* publishing the parked flag.
        if let Some(job) = find_job(idx, &deque, &inner) {
            // Race: caller may have already unparked us, but either
            // way we're awake now.
            slot.is_parked.store(false, Ordering::Release);
            job();
            continue;
        }
        if inner.shutdown.load(Ordering::Acquire) {
            slot.is_parked.store(false, Ordering::Release);
            return;
        }

        parker.park();
        // Either an unparker fired, or shutdown nudged us. The
        // `wake_one` / `shutdown` paths clear the parked flag for us
        // (when they claim it via swap). Make sure it's clear so
        // future park cycles work correctly even if we were woken
        // spuriously.
        slot.is_parked.store(false, Ordering::Release);
    }
}

fn find_job(idx: usize, local: &Worker<Job>, inner: &RuntimeInner) -> Option<Job> {
    local.pop().or_else(|| {
        iter::repeat_with(|| {
            inner.injector.steal_batch_and_pop(local).or_else(|| {
                inner
                    .stealers
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, s)| s.steal())
                    .collect::<Steal<_>>()
            })
        })
        .find(|s| !s.is_retry())
        .and_then(|s| s.success())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn runs_many_jobs_across_workers() {
        let rt = Runtime::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..1000 {
            let counter = Arc::clone(&counter);
            let tx = tx.clone();
            rt.spawn(Box::new(move || {
                counter.fetch_add(1, Ordering::Relaxed);
                let _ = tx.send(());
            }));
        }
        drop(tx);
        // Wait for completion.
        while rx.recv_timeout(Duration::from_secs(5)).is_ok() {}
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
        rt.shutdown();
    }

    #[test]
    fn wake_after_idle() {
        let rt = Runtime::new(2);
        // Let workers go to sleep.
        thread::sleep(Duration::from_millis(20));
        let (tx, rx) = std::sync::mpsc::channel();
        rt.spawn(Box::new(move || {
            let _ = tx.send(42_u32);
        }));
        assert_eq!(rx.recv_timeout(Duration::from_secs(1)), Ok(42));
        rt.shutdown();
    }
}
