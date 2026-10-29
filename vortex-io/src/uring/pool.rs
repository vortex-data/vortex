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
/// affinity. Dropping the pool waits for its workers unless called from a native scan worker;
/// in that case closing the queue lets them drain and exit without joining on that worker.
#[derive(Clone)]
pub struct UringWorkerPool(Arc<Inner>);

struct Inner {
    sender: Option<Sender<Job>>,
    control_sender: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
    max_active_jobs: usize,
    max_control_jobs: usize,
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

    fn reserve(&self, limit: usize) {
        let active = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        let workers = self.workers();
        // Crossing an admission threshold can unblock workers awaiting external events.
        // At the per-worker limit, no additional job becomes admissible.
        if active.is_multiple_of(workers) && active / workers < limit {
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
        self.admission.reserve(self.limit);
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

struct JobQueue {
    receiver: Receiver<Job>,
    active: ActiveJobs,
    closed: bool,
}

impl JobQueue {
    fn new(receiver: Receiver<Job>, admission: Arc<Admission>, limit: usize) -> Self {
        Self {
            receiver,
            active: ActiveJobs {
                futures: FuturesUnordered::new(),
                admission,
                limit,
            },
            closed: false,
        }
    }

    fn can_admit(&self) -> bool {
        !self.closed && self.active.can_admit()
    }

    fn try_admit(&mut self) -> bool {
        if self.can_admit() {
            match self.receiver.try_recv() {
                Ok(job) => {
                    self.active.push(job);
                    return true;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => self.closed = true,
            }
        }
        false
    }
}

fn run_concurrent(
    mut queues: Vec<JobQueue>,
    worker: usize,
    wake_recv: Receiver<()>,
    drain_on_idle: bool,
) -> VortexResult<()> {
    let waker = Waker::from(Arc::new(TaskWake(
        queues[0].active.admission.wakes[worker].clone(),
    )));
    let mut cx = Context::from_waker(&waker);
    loop {
        for queue in &mut queues {
            while queue.try_admit() {}
        }
        if queues.iter().all(|queue| queue.active.is_empty()) {
            if drain_on_idle {
                drain_pending_reads()?;
            }
            if queues.iter().all(|queue| queue.closed) {
                return Ok(());
            }
        }
        // Clear an old notification before polling, so a concurrent wake cannot be lost.
        let _ = wake_recv.try_recv();
        let mut completed = false;
        for queue in &mut queues {
            completed |= queue.active.poll_completed(&mut cx);
        }
        if completed {
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
            if queues.iter_mut().any(JobQueue::try_admit) {
                continue;
            }
            DRIVER.with(|slot| -> VortexResult<()> {
                slot.borrow_mut()
                    .as_mut()
                    .ok_or_else(|| vortex_err!("io_uring worker stopped"))?
                    .pump(true)
            })?;
        } else {
            let mut select = crossbeam_channel::Select::new();
            let mut indices = [None; 2];
            for (index, queue) in queues.iter().enumerate() {
                if queue.can_admit() {
                    let selected = select.recv(&queue.receiver);
                    indices[selected] = Some(index);
                }
            }
            let wake_index = select.recv(&wake_recv);
            let selected = select.select();
            let delivery = if selected.index() == wake_index {
                let _ = selected.recv(&wake_recv);
                None
            } else {
                let index = indices[selected.index()]
                    .ok_or_else(|| vortex_err!("Selected an unregistered job queue"))?;
                Some((index, selected.recv(&queues[index].receiver)))
            };
            drop(select);
            if let Some((index, job)) = delivery {
                match job {
                    Ok(job) => queues[index].active.push(job),
                    Err(_) => queues[index].closed = true,
                }
            }
        }
    }
}

pub(super) struct WorkerWake(pub(super) Thread);

impl Wake for WorkerWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn drain_pending_reads() -> VortexResult<()> {
    DRIVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let driver = slot
            .as_mut()
            .ok_or_else(|| vortex_err!("io_uring worker stopped"))?;
        // Finished jobs can leave cancelled requests or reads shared with another worker.
        // Retire them before parking so their buffers and mappings can be released.
        while driver.has_work() {
            if let Err(error) = driver.pump(true) {
                driver.fail_all("Owning ring failed while completing shared reads");
                return Err(error);
            }
        }
        Ok(())
    })
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
        if options.control_workers > 0
            && (options.max_control_jobs == 0 || options.control_workers >= workers)
        {
            vortex_bail!(
                "Reserved control workers require control admission and at least one scan worker"
            );
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
            workers: AtomicUsize::new(workers - options.control_workers),
            wakes,
        });
        let (control_sender, control_receiver) = crossbeam_channel::unbounded::<Job>();
        let control_admission = Arc::new(Admission {
            active: AtomicUsize::new(0),
            workers: AtomicUsize::new(if options.control_workers == 0 {
                workers
            } else {
                options.control_workers
            }),
            wakes: admission.wakes.clone(),
        });
        let mut inner = Inner {
            sender: Some(sender),
            control_sender: (options.max_control_jobs > 0).then_some(control_sender),
            workers: Vec::new(),
            max_active_jobs: options.max_active_jobs,
            max_control_jobs: options.max_control_jobs,
        };
        for (worker, (&cpu, wake_recv)) in cpus.iter().take(workers).zip(wake_receivers).enumerate()
        {
            let serves_control = options.max_control_jobs > 0
                && (options.control_workers == 0 || worker < options.control_workers);
            let serves_scan = options.control_workers == 0 || worker >= options.control_workers;
            let receiver = receiver.clone();
            let admission = Arc::clone(&admission);
            let control_admission = Arc::clone(&control_admission);
            let control_receiver = control_receiver.clone();
            let (started_send, started_recv) = crossbeam_channel::bounded(1);
            inner.workers.push(
                thread::Builder::new()
                    .name(format!("vortex-uring-{cpu}"))
                    .spawn(move || {
                        let _registration =
                            serves_scan.then(|| WorkerAdmission(Arc::clone(&admission)));
                        let _control_registration =
                            serves_control.then(|| WorkerAdmission(Arc::clone(&control_admission)));
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
                        if options.max_active_jobs == 1 && options.max_control_jobs == 0 {
                            while let Ok(Job::Blocking(job)) = receiver.recv() {
                                job();
                                if let Err(error) = drain_pending_reads() {
                                    tracing::error!(%error, "io_uring worker stopped");
                                    break;
                                }
                            }
                        } else {
                            let mut queues = Vec::with_capacity(2);
                            if serves_control {
                                queues.push(JobQueue::new(
                                    control_receiver,
                                    control_admission,
                                    options.max_control_jobs,
                                ));
                            } else {
                                drop(control_receiver);
                            }
                            if serves_scan {
                                queues.push(JobQueue::new(
                                    receiver,
                                    admission,
                                    options.max_active_jobs,
                                ));
                            } else {
                                drop(receiver);
                            }
                            if let Err(error) = run_concurrent(
                                queues,
                                worker,
                                wake_recv,
                                options.reuse_across_workers,
                            ) {
                                tracing::error!(%error, "io_uring worker stopped");
                            }
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

    /// Maximum active control futures on each worker; zero means the control queue is disabled.
    pub fn max_control_jobs(&self) -> usize {
        self.0.max_control_jobs
    }

    /// Runs a control future independently of scan-job admission.
    ///
    /// Requires a positive [`UringOptions::max_control_jobs`]. The future may await scan jobs
    /// submitted with [`Self::spawn_async`], but must not await other control futures. Dropping
    /// its handle cancels it at the next poll on its worker. Blocking work occupies its worker
    /// even when control admission is available.
    pub fn spawn_control<F, Fut, T>(&self, work: F) -> BoxFuture<'static, VortexResult<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = VortexResult<T>> + 'static,
        T: Send + 'static,
    {
        enqueue_async(self.0.control_sender.as_ref(), work)
    }

    /// Runs a future on one worker while driving that worker's ring between polls.
    ///
    /// The future is constructed on its worker and need not be `Send`. With multiple active
    /// jobs or control admission enabled, dropping the returned future cancels its work at the
    /// next poll on the worker. Otherwise cancellation only prevents work that has not started. External
    /// runtimes required by the future must remain running. A native I/O wait may delay an
    /// unrelated external wakeup or admission of a newly queued job.
    pub fn spawn_async<F, Fut, T>(&self, work: F) -> BoxFuture<'static, VortexResult<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = VortexResult<T>> + 'static,
        T: Send + 'static,
    {
        if self.0.max_active_jobs > 1 || self.0.max_control_jobs > 0 {
            return enqueue_async(self.0.sender.as_ref(), work);
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

fn enqueue_async<F, Fut, T>(
    queue: Option<&Sender<Job>>,
    work: F,
) -> BoxFuture<'static, VortexResult<T>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = VortexResult<T>> + 'static,
    T: Send + 'static,
{
    let (mut sender, receiver) = oneshot::channel();
    let job = Job::Async(Box::new(move || {
        async move {
            if sender.is_canceled() {
                return;
            }
            let mut future = Box::pin(AssertUnwindSafe(async move { work().await }).catch_unwind());
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
    let queued = queue.is_some_and(|sender| sender.send(job).is_ok());
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

impl Drop for Inner {
    fn drop(&mut self) {
        self.sender.take();
        self.control_sender.take();
        let on_worker = DRIVER
            .try_with(|slot| match slot.try_borrow() {
                Ok(driver) => driver.is_some(),
                Err(_) => true,
            })
            .unwrap_or(true);
        // A worker in this or another pool can be waiting for reads driven by this thread.
        // Joining it here would prevent this thread from returning to its ring.
        if on_worker {
            self.workers.clear();
            return;
        }
        for worker in self.workers.drain(..) {
            drop(worker.join());
        }
    }
}
