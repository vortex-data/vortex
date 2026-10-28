// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::io;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::panic::resume_unwind;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;
use std::thread;
use std::thread::JoinHandle;
use std::thread::Thread;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::TryRecvError;
use futures::FutureExt;
use futures::StreamExt;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::LocalBoxFuture;
use futures::stream::FuturesUnordered;
use rustix::thread::CpuSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::DRIVER;
use super::UringOptions;
use super::driver::Driver;

enum Job {
    Blocking(Box<dyn FnOnce() + Send>),
    Async(Box<dyn FnOnce() -> LocalBoxFuture<'static, ()> + Send>),
}

impl Job {
    fn into_future(self) -> LocalBoxFuture<'static, ()> {
        match self {
            Self::Blocking(work) => async move {
                work();
            }
            .boxed_local(),
            Self::Async(work) => work(),
        }
    }
}

/// A fixed pool of scan workers, each owning one persistent io_uring.
///
/// Jobs never migrate between workers and no worker consumes another worker's completions.
/// Async jobs can overlap up to [`UringOptions::max_active_jobs`] on each worker. Blocking jobs
/// occupy their worker until they return. Workers are pinned unless constructed with inherited
/// affinity.
#[derive(Clone)]
pub struct UringWorkerPool(Arc<Inner>);

struct Inner {
    sender: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    max_active_jobs: usize,
}

struct TaskWake(Sender<()>);

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}

struct Admission {
    active: AtomicUsize,
    workers: AtomicUsize,
    wakes: Vec<Sender<()>>,
}

impl Admission {
    fn workers(&self) -> usize {
        self.workers.load(Ordering::Relaxed).max(1)
    }

    fn wake_all(&self) {
        for wake in &self.wakes {
            let _ = wake.try_send(());
        }
    }

    fn reserve(&self) {
        let previous = self.active.fetch_add(1, Ordering::Relaxed);
        // Crossing an admission threshold can unblock workers awaiting external events.
        if (previous + 1).is_multiple_of(self.workers()) {
            self.wake_all();
        }
    }
}

struct WorkerAdmission(Arc<Admission>);

impl Drop for WorkerAdmission {
    fn drop(&mut self) {
        self.0.workers.fetch_sub(1, Ordering::Relaxed);
        self.0.wake_all();
    }
}

struct ActiveJobs {
    futures: FuturesUnordered<LocalBoxFuture<'static, ()>>,
    admission: Arc<Admission>,
    limit: usize,
}

impl ActiveJobs {
    fn can_admit(&self) -> bool {
        // Leave queued jobs available to less occupied workers. Started futures cannot migrate.
        self.futures.len() < self.limit
            && self.futures.len()
                <= self.admission.active.load(Ordering::Relaxed) / self.admission.workers()
    }

    fn is_empty(&self) -> bool {
        self.futures.is_empty()
    }

    fn push(&mut self, job: Job) {
        let future = job.into_future();
        self.admission.reserve();
        self.futures.push(future);
    }

    fn poll_completed(&mut self, cx: &mut Context<'_>) -> bool {
        if matches!(self.futures.poll_next_unpin(cx), Poll::Ready(Some(()))) {
            self.admission.active.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

impl Drop for ActiveJobs {
    fn drop(&mut self) {
        self.admission
            .active
            .fetch_sub(self.futures.len(), Ordering::Relaxed);
    }
}

fn run_concurrent(
    receiver: Receiver<Job>,
    max_active_jobs: usize,
    admission: Arc<Admission>,
    worker: usize,
    wake_recv: Receiver<()>,
) -> VortexResult<()> {
    let waker = Waker::from(Arc::new(TaskWake(admission.wakes[worker].clone())));
    let mut cx = Context::from_waker(&waker);
    let mut active = ActiveJobs {
        futures: FuturesUnordered::new(),
        admission,
        limit: max_active_jobs,
    };
    let mut closed = false;
    loop {
        while !closed && active.can_admit() {
            match receiver.try_recv() {
                Ok(job) => active.push(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => closed = true,
            }
        }
        if active.is_empty() {
            if closed {
                return Ok(());
            }
            match receiver.recv() {
                Ok(job) => active.push(job),
                Err(_) => closed = true,
            }
            continue;
        }
        // Clear an old notification before polling, so a concurrent wake cannot be lost.
        let _ = wake_recv.try_recv();
        if active.poll_completed(&mut cx) {
            continue;
        }
        let pending_io = DRIVER.with(|slot| -> VortexResult<bool> {
            let mut slot = slot.borrow_mut();
            let driver = slot
                .as_mut()
                .ok_or_else(|| vortex_err!("io_uring worker stopped"))?;
            driver.pump(false)?;
            Ok(driver.has_work())
        })?;
        if wake_recv.try_recv().is_ok() {
            continue;
        }
        if pending_io {
            // Admit work that arrived while polling before entering a native I/O wait.
            if !closed && active.can_admit() {
                match receiver.try_recv() {
                    Ok(job) => {
                        active.push(job);
                        continue;
                    }
                    Err(TryRecvError::Empty) => {}
                    Err(TryRecvError::Disconnected) => closed = true,
                }
            }
            DRIVER.with(|slot| -> VortexResult<()> {
                slot.borrow_mut()
                    .as_mut()
                    .ok_or_else(|| vortex_err!("io_uring worker stopped"))?
                    .pump(true)
            })?;
        } else if closed || !active.can_admit() {
            let _ = wake_recv.recv();
        } else {
            crossbeam_channel::select! {
                recv(receiver) -> job => match job {
                    Ok(job) => active.push(job),
                    Err(_) => closed = true,
                },
                recv(wake_recv) -> _ => {},
            }
        }
    }
}

struct WorkerWake(Thread);

impl Wake for WorkerWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

impl UringWorkerPool {
    /// Starts `workers` threads pinned to distinct CPUs in the caller's allowed affinity mask.
    pub fn new(workers: usize, options: UringOptions) -> VortexResult<Self> {
        Self::start(workers, options, true)
    }

    /// Starts `workers` threads that retain the caller's allowed CPU mask.
    ///
    /// Each ring and its jobs stay on their owning thread, but the scheduler can move that thread
    /// between CPUs. This permits comparing thread-local rings with CPU-local rings independently
    /// of shared-ring coalescing or completed-read retention.
    pub fn with_inherited_affinity(workers: usize, options: UringOptions) -> VortexResult<Self> {
        Self::start(workers, options, false)
    }

    fn start(workers: usize, options: UringOptions, pinned: bool) -> VortexResult<Self> {
        if options.max_active_jobs == 0 {
            vortex_bail!("Active jobs per worker must be positive");
        }
        let allowed = rustix::thread::sched_getaffinity(None).map_err(io::Error::from)?;
        let cpus: Vec<_> = (0..CpuSet::MAX_CPU)
            .filter(|&cpu| allowed.is_set(cpu))
            .collect();
        if workers == 0 || workers > cpus.len() {
            vortex_bail!(
                "Requested {workers} workers but {} CPUs are available",
                cpus.len()
            );
        }
        let (sender, receiver) = crossbeam_channel::unbounded::<Job>();
        let (wakes, wake_receivers): (Vec<_>, Vec<_>) =
            (0..workers).map(|_| crossbeam_channel::bounded(1)).unzip();
        let admission = Arc::new(Admission {
            active: AtomicUsize::new(0),
            workers: AtomicUsize::new(workers),
            wakes,
        });
        let mut inner = Inner {
            sender: Some(sender),
            workers: Vec::new(),
            max_active_jobs: options.max_active_jobs,
        };
        for (worker, (&cpu, wake_recv)) in cpus.iter().take(workers).zip(wake_receivers).enumerate()
        {
            let receiver = receiver.clone();
            let admission = Arc::clone(&admission);
            let (started_send, started_recv) = crossbeam_channel::bounded(1);
            inner.workers.push(
                thread::Builder::new()
                    .name(format!("vortex-uring-{cpu}"))
                    .spawn(move || {
                        let _registration = WorkerAdmission(Arc::clone(&admission));
                        let setup = (|| -> VortexResult<()> {
                            // Kernel offload threads can inherit the submitting thread's affinity.
                            // Start the first one while the scan worker still has the caller's mask.
                            let prepared = if options.eager_kernel_worker {
                                Some(Driver::new(options)?)
                            } else {
                                None
                            };
                            if pinned {
                                let mut affinity = CpuSet::new();
                                affinity.set(cpu);
                                rustix::thread::sched_setaffinity(None, &affinity)
                                    .map_err(io::Error::from)?;
                            }
                            let driver = match prepared {
                                Some(driver) => driver,
                                None => Driver::new(options)?,
                            };
                            DRIVER.with(|slot| *slot.borrow_mut() = Some(driver));
                            Ok(())
                        })();
                        let failed = setup.is_err();
                        drop(started_send.send(setup));
                        if failed {
                            return;
                        }
                        if options.max_active_jobs == 1 {
                            while let Ok(Job::Blocking(job)) = receiver.recv() {
                                job();
                            }
                        } else if let Err(error) = run_concurrent(
                            receiver,
                            options.max_active_jobs,
                            admission,
                            worker,
                            wake_recv,
                        ) {
                            tracing::error!(%error, "io_uring worker stopped");
                        }
                        DRIVER.with(|slot| {
                            slot.borrow_mut().take();
                        });
                    })?,
            );
            started_recv
                .recv()
                .map_err(|_| vortex_err!("io_uring worker stopped during startup"))??;
        }
        Ok(Self(Arc::new(inner)))
    }

    /// Number of scan workers.
    pub fn workers(&self) -> usize {
        self.0.workers.len()
    }

    /// Maximum active async jobs on each worker.
    pub fn max_active_jobs(&self) -> usize {
        self.0.max_active_jobs
    }

    /// Runs a future on one worker while driving that worker's ring between polls.
    ///
    /// The future is constructed on its worker and need not be `Send`. With multiple active
    /// jobs, dropping the returned future cancels its work at the next poll on the worker.
    /// With one active job, cancellation only prevents work that has not started. External
    /// runtimes required by the future must remain running. A native I/O wait may delay an
    /// unrelated external wakeup or admission of a newly queued job.
    pub fn spawn_async<F, Fut, T>(&self, work: F) -> BoxFuture<'static, VortexResult<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = VortexResult<T>> + 'static,
        T: Send + 'static,
    {
        if self.0.max_active_jobs > 1 {
            let (mut sender, receiver) = oneshot::channel();
            let job = Job::Async(Box::new(move || {
                async move {
                    if sender.is_canceled() {
                        return;
                    }
                    let mut future =
                        Box::pin(AssertUnwindSafe(async move { work().await }).catch_unwind());
                    let result = std::future::poll_fn(|cx| {
                        if sender.poll_canceled(cx).is_ready() {
                            return Poll::Ready(None);
                        }
                        future.as_mut().poll(cx).map(Some)
                    })
                    .await;
                    if let Some(result) = result {
                        drop(sender.send(result));
                    }
                }
                .boxed_local()
            }));
            let queued = self
                .0
                .sender
                .as_ref()
                .is_some_and(|sender| sender.send(job).is_ok());
            return async move {
                if !queued {
                    vortex_bail!("io_uring worker pool has stopped");
                }
                match receiver
                    .await
                    .map_err(|_| vortex_err!("io_uring worker dropped its job"))?
                {
                    Ok(result) => result,
                    Err(panic) => resume_unwind(panic),
                }
            }
            .boxed();
        }
        self.spawn(move || {
            let mut future = std::pin::pin!(work());
            let waker = Waker::from(Arc::new(WorkerWake(thread::current())));
            let mut cx = Context::from_waker(&waker);
            loop {
                if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
                    return result;
                }
                let drove_io = DRIVER.with(|slot| -> VortexResult<bool> {
                    let mut slot = slot.borrow_mut();
                    let driver = slot
                        .as_mut()
                        .ok_or_else(|| vortex_err!("io_uring worker stopped"))?;
                    if !driver.has_work() {
                        return Ok(false);
                    }
                    driver.pump(true)?;
                    Ok(true)
                })?;
                if !drove_io {
                    thread::park();
                }
            }
        })
    }

    /// Schedules one scan job. Dropping its future cancels it if execution has not started.
    pub fn spawn<F, T>(&self, work: F) -> BoxFuture<'static, VortexResult<T>>
    where
        F: FnOnce() -> VortexResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let job = Job::Blocking(Box::new(move || {
            if sender.is_canceled() {
                return;
            }
            let result = catch_unwind(AssertUnwindSafe(work));
            drop(sender.send(result));
        }));
        let queued = self
            .0
            .sender
            .as_ref()
            .is_some_and(|sender| sender.send(job).is_ok());
        async move {
            if !queued {
                vortex_bail!("io_uring worker pool has stopped");
            }
            match receiver
                .await
                .map_err(|_| vortex_err!("io_uring worker dropped its job"))?
            {
                Ok(result) => result,
                Err(panic) => resume_unwind(panic),
            }
        }
        .boxed()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            // A job may hold the final pool reference. Its own worker exits after that job.
            if worker.thread().id() != thread::current().id() {
                drop(worker.join());
            }
        }
    }
}
