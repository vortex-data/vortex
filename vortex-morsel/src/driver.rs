// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Affinity-owned morsel execution over one shared asynchronous IO service.
//!
//! Each worker owns one arena and at most one active morsel. The arena never crosses a thread
//! boundary. Planning submits all named segment futures to scan-wide required/speculative queues;
//! while its morsel is suspended, a worker polls IO from those queues. Exact ticket completion
//! wakes only the worker whose continuation parked on that ticket. Output order is restored by
//! morsel index after all workers finish.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Wake;
use std::task::Waker;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::unbounded;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::build::ExecPlan;
use crate::build::cut_morsels;
use crate::cells::SharedCells;
use crate::io::IoKey;
use crate::io::IoPlane;
use crate::io::IoPriority;
use crate::io::IoRead;
use crate::io::IoReadPoll;
use crate::io::IoService;
use crate::node::Arena;
use crate::node::ExecPoll;
use crate::node::PlanPoll;
use crate::node::Wait;
use crate::node::WaitSet;
use crate::node::begin_morsel;
use crate::node::poll_execute_morsel;
use crate::node::poll_plan_morsel;
use crate::node::retire_morsel;
use crate::stats::ScanStats;

/// The morsel row ranges for a plan.
///
/// With `target_rows` of zero every natural split is a morsel boundary, which is exactly the V1
/// split set — the fair-comparison default. A larger target coalesces consecutive splits, which
/// is where the executor's ability to straddle chunk boundaries starts to pay.
pub fn morsels(plan: &ExecPlan, target_rows: u64) -> Vec<Range<u64>> {
    cut_morsels(plan.natural_splits(), target_rows)
}

/// One configured run of the morsel executor.
pub struct MorselScan {
    plan: Arc<ExecPlan>,
    segments: Arc<dyn SegmentSource>,
    session: VortexSession,
    morsels: Arc<[Range<u64>]>,
    threads: usize,
    share_decodes: bool,
    completion: Option<CompletionSink>,
    worker_pool: Option<Arc<SharedMorselWorkerPool>>,
}

type CompletionSink = Arc<dyn Fn(usize, Option<ArrayRef>) + Send + Sync>;

struct WorkerRun {
    plan: Arc<ExecPlan>,
    session: VortexSession,
    morsels: Arc<[Range<u64>]>,
    io: Arc<IoService>,
    cells: SharedCells,
    start: Instant,
    completion: Option<CompletionSink>,
}

#[derive(Clone, Copy)]
enum TaskPhase {
    Plan,
    Execute,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WaitToken {
    generation: usize,
    epoch: usize,
}

struct LocalMorsel {
    arena: Arena,
    io: IoPlane,
    phase: TaskPhase,
    index: usize,
    range: Range<u64>,
    active: bool,
    generation: usize,
    wait_epoch: usize,
    waiting: Option<WaitToken>,
    morsel_io_uses_start: u64,
    morsel_io_requests_start: u64,
    morsel_io_batches_start: u64,
    morsel_io_blocks_start: u64,
    stats: ScanStats,
}

struct IoWork {
    queued: AtomicBool,
    running: AtomicBool,
    required: AtomicBool,
    ready: Mutex<Vec<usize>>,
    scheduled: Vec<AtomicBool>,
    completed: Vec<AtomicBool>,
    reads: Vec<IoRead>,
}

enum WorkerSignal {
    Wake(WaitToken),
    Shutdown,
}

struct Scheduler {
    run: Arc<WorkerRun>,
    urgent_tx: Sender<Arc<IoWork>>,
    urgent_rx: Receiver<Arc<IoWork>>,
    ready_tx: Sender<Arc<IoWork>>,
    ready_rx: Receiver<Arc<IoWork>>,
    worker_tx: Vec<Sender<WorkerSignal>>,
    io_work: Mutex<HashMap<IoKey, Arc<IoWork>>>,
    results: Mutex<Vec<(usize, ArrayRef)>>,
    error: Mutex<Option<VortexError>>,
    next_morsel: AtomicUsize,
    remaining: AtomicUsize,
    stopped: AtomicBool,
    io_bytes: AtomicU64,
    io_waits: AtomicU64,
    io_wait_nanos: AtomicU64,
}

enum WorkerMessage {
    Run {
        scheduler: Arc<Scheduler>,
        worker: usize,
        signals: Receiver<WorkerSignal>,
        done: mpsc::Sender<ScanStats>,
    },
    Shutdown,
}

struct Worker {
    messages: mpsc::Sender<WorkerMessage>,
    handle: Option<JoinHandle<()>>,
}

/// A set of ready morsel workers whose lifecycle is outside a timed scan.
struct MorselWorkerPool {
    workers: Vec<Worker>,
}

/// A persistent set of morsel workers that can be shared by successive scans.
///
/// Runs are serialized because every scan dispatches work to every worker. Keeping the workers
/// alive removes thread creation and shutdown from short query execution paths.
pub struct SharedMorselWorkerPool {
    inner: Mutex<MorselWorkerPool>,
    threads: usize,
}

impl SharedMorselWorkerPool {
    /// Start a persistent pool with the requested number of workers.
    pub fn new(threads: usize) -> VortexResult<Self> {
        let threads = threads.max(1);
        Ok(Self {
            inner: Mutex::new(MorselWorkerPool::new(threads)?),
            threads,
        })
    }

    /// The number of workers available to each scan.
    pub fn threads(&self) -> usize {
        self.threads
    }

    fn run(
        &self,
        scheduler: Arc<Scheduler>,
        signals: Vec<Receiver<WorkerSignal>>,
    ) -> VortexResult<Vec<ScanStats>> {
        self.inner.lock().run(scheduler, signals)
    }
}

impl MorselWorkerPool {
    fn new(threads: usize) -> VortexResult<Self> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let mut workers = Vec::with_capacity(threads);

        for idx in 0..threads {
            let (message_tx, message_rx) = mpsc::channel();
            let ready_tx = ready_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("vortex-morsel-{idx}"))
                .spawn(move || {
                    if ready_tx.send(()).is_err() {
                        return;
                    }
                    while let Ok(message) = message_rx.recv() {
                        match message {
                            WorkerMessage::Run {
                                scheduler,
                                worker,
                                signals,
                                done,
                            } => {
                                let stats = scheduler.worker_loop(worker, &signals);
                                let _ = done.send(stats);
                            }
                            WorkerMessage::Shutdown => break,
                        }
                    }
                })
                .map_err(|err| vortex_err!("failed to spawn morsel worker: {err}"))?;
            workers.push(Worker {
                messages: message_tx,
                handle: Some(handle),
            });
        }
        drop(ready_tx);

        for _ in 0..threads {
            ready_rx
                .recv()
                .map_err(|err| vortex_err!("morsel worker failed to start: {err}"))?;
        }
        Ok(Self { workers })
    }

    fn run(
        &self,
        scheduler: Arc<Scheduler>,
        signals: Vec<Receiver<WorkerSignal>>,
    ) -> VortexResult<Vec<ScanStats>> {
        let (done_tx, done_rx) = mpsc::channel();
        for (worker, (thread, signals)) in self.workers.iter().zip(signals).enumerate() {
            thread
                .messages
                .send(WorkerMessage::Run {
                    scheduler: Arc::clone(&scheduler),
                    worker,
                    signals,
                    done: done_tx.clone(),
                })
                .map_err(|err| vortex_err!("failed to dispatch morsel worker: {err}"))?;
        }
        drop(done_tx);

        let mut stats = Vec::with_capacity(self.workers.len());
        for _ in 0..self.workers.len() {
            stats.push(
                done_rx
                    .recv()
                    .map_err(|err| vortex_err!("morsel worker stopped early: {err}"))?,
            );
        }
        Ok(stats)
    }
}

impl Drop for MorselWorkerPool {
    fn drop(&mut self) {
        for worker in &self.workers {
            drop(worker.messages.send(WorkerMessage::Shutdown));
        }
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                drop(handle.join());
            }
        }
    }
}

struct IoWake {
    scheduler: Weak<Scheduler>,
    work: Weak<IoWork>,
    index: usize,
}

struct TaskWake {
    tx: Sender<WorkerSignal>,
    token: WaitToken,
}

impl Wake for TaskWake {
    fn wake(self: Arc<Self>) {
        drop(self.tx.send(WorkerSignal::Wake(self.token)));
    }

    fn wake_by_ref(self: &Arc<Self>) {
        drop(self.tx.send(WorkerSignal::Wake(self.token)));
    }
}

impl Wake for IoWake {
    fn wake(self: Arc<Self>) {
        self.enqueue();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.enqueue();
    }
}

impl IoWake {
    fn enqueue(&self) {
        let (Some(scheduler), Some(work)) = (self.scheduler.upgrade(), self.work.upgrade()) else {
            return;
        };
        if work.completed[self.index].load(Ordering::Acquire)
            || work.scheduled[self.index].swap(true, Ordering::AcqRel)
        {
            return;
        }
        work.ready.lock().push(self.index);
        scheduler.enqueue_io(work);
    }
}

impl Scheduler {
    fn new(run: Arc<WorkerRun>, workers: usize) -> (Arc<Self>, Vec<Receiver<WorkerSignal>>) {
        let (urgent_tx, urgent_rx) = unbounded();
        let (ready_tx, ready_rx) = unbounded();
        let mut worker_tx = Vec::with_capacity(workers);
        let mut worker_rx = Vec::with_capacity(workers);
        for _ in 0..workers {
            let (tx, rx) = unbounded();
            worker_tx.push(tx);
            worker_rx.push(rx);
        }
        let scheduler = Arc::new(Self {
            remaining: AtomicUsize::new(run.morsels.len()),
            run,
            urgent_tx,
            urgent_rx,
            ready_tx,
            ready_rx,
            worker_tx,
            io_work: Mutex::new(HashMap::default()),
            results: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            next_morsel: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            io_bytes: AtomicU64::new(0),
            io_waits: AtomicU64::new(0),
            io_wait_nanos: AtomicU64::new(0),
        });
        if scheduler.run.morsels.is_empty() {
            scheduler.stop();
        }
        (scheduler, worker_rx)
    }

    fn submit_reads(self: &Arc<Self>, mut reads: Vec<IoRead>) -> u64 {
        reads.sort_unstable_by_key(|read| match read.key() {
            IoKey::Segment(id) => *id,
        });
        let (required, speculative): (Vec<_>, Vec<_>) = reads
            .into_iter()
            .partition(|read| read.priority() == IoPriority::Required);
        for read in &speculative {
            self.run.io.issue(read);
        }
        let eager_required = self.run.io.nowait_unsupported();
        if eager_required {
            for read in &required {
                self.run.io.issue(read);
            }
        }
        u64::from(self.submit_io_batch(required, true, eager_required))
            + u64::from(self.submit_io_batch(speculative, false, true))
    }

    fn submit_io_batch(
        self: &Arc<Self>,
        reads: Vec<IoRead>,
        required: bool,
        enqueue: bool,
    ) -> bool {
        if reads.is_empty() {
            return false;
        }
        let read_count = reads.len();
        let work = Arc::new(IoWork {
            queued: AtomicBool::new(false),
            running: AtomicBool::new(false),
            required: AtomicBool::new(required),
            ready: Mutex::new((0..read_count).collect()),
            scheduled: (0..read_count).map(|_| AtomicBool::new(true)).collect(),
            completed: (0..read_count).map(|_| AtomicBool::new(false)).collect(),
            reads,
        });
        {
            let mut io_work = self.io_work.lock();
            for read in &work.reads {
                io_work.insert(read.key(), Arc::clone(&work));
            }
        }
        if enqueue {
            self.enqueue_io(work);
        }
        true
    }

    fn enqueue_io(&self, work: Arc<IoWork>) {
        if self.stopped.load(Ordering::Acquire) || work.queued.swap(true, Ordering::AcqRel) {
            return;
        }
        let tx = if work.required.load(Ordering::Acquire) {
            &self.urgent_tx
        } else {
            &self.ready_tx
        };
        drop(tx.send(work));
    }

    fn promote(&self, key: IoKey) {
        let Some(work) = self.io_work.lock().get(&key).cloned() else {
            return;
        };
        for read in &work.reads {
            self.run.io.issue(read);
        }
        work.required.store(true, Ordering::Release);
        if work.queued.load(Ordering::Acquire) {
            // Leave the normal-queue copy in place and add an urgent copy. The first receiver
            // clears `queued` and owns the poll; the other copy is then a cheap stale dequeue.
            drop(self.urgent_tx.send(work));
        } else {
            self.enqueue_io(work);
        }
    }

    fn park(&self, worker: usize, token: WaitToken, waits: &WaitSet) -> VortexResult<bool> {
        if waits.is_empty() {
            return Err(vortex_err!(
                "execution blocked without naming an exact dependency"
            ));
        }

        let mut parked = false;
        let tx = self
            .worker_tx
            .get(worker)
            .cloned()
            .ok_or_else(|| vortex_err!("blocked on an unknown worker"))?;
        for wait in waits.waits() {
            let Wait::Io(ticket) = wait;
            let read = self
                .run
                .io
                .read(*ticket)
                .ok_or_else(|| vortex_err!("blocked on an unknown IO ticket"))?;
            read.promote();
            self.promote(read.key());
            let waker = Waker::from(Arc::new(TaskWake {
                tx: tx.clone(),
                token,
            }));
            if read.park(waker) {
                parked = true;
            }
        }
        Ok(parked)
    }

    fn run_io(self: &Arc<Self>, work: Arc<IoWork>) -> VortexResult<()> {
        if work.running.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let ready = std::mem::take(&mut *work.ready.lock());
        for index in ready {
            work.scheduled[index].store(false, Ordering::Release);
            if work.completed[index].load(Ordering::Acquire) {
                continue;
            }
            let waker = Waker::from(Arc::new(IoWake {
                scheduler: Arc::downgrade(self),
                work: Arc::downgrade(&work),
                index,
            }));
            match work.reads[index].poll(&waker)? {
                IoReadPoll::Pending => {
                    self.io_waits.fetch_add(1, Ordering::Relaxed);
                }
                IoReadPoll::Ready { bytes, wait_time } => {
                    if work.completed[index].swap(true, Ordering::AcqRel) {
                        continue;
                    }
                    self.io_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
                    self.io_wait_nanos.fetch_add(
                        u64::try_from(wait_time.as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                }
                IoReadPoll::AlreadyReady => {
                    work.completed[index].store(true, Ordering::Release);
                }
            }
        }
        work.queued.store(false, Ordering::Release);
        work.running.store(false, Ordering::Release);
        if !work.ready.lock().is_empty() {
            self.enqueue_io(work);
        }
        Ok(())
    }

    fn try_run_io(self: &Arc<Self>) -> VortexResult<bool> {
        let work = self
            .urgent_rx
            .try_recv()
            .or_else(|_| self.ready_rx.try_recv());
        let Ok(work) = work else {
            return Ok(false);
        };
        self.run_io(work)?;
        Ok(true)
    }

    fn worker_loop(self: &Arc<Self>, worker: usize, signals: &Receiver<WorkerSignal>) -> ScanStats {
        let mut morsel = LocalMorsel::new(&self.run);
        let mut runnable = morsel.assign_next(self);

        loop {
            if self.stopped.load(Ordering::Acquire) {
                break;
            }

            if runnable {
                match morsel.run(self) {
                    Ok(LocalPoll::Runnable) => runnable = true,
                    Ok(LocalPoll::Blocked(waits)) => {
                        let token = morsel.next_wait_token();
                        match self.park(worker, token, &waits) {
                            Ok(true) => {
                                morsel.waiting = Some(token);
                                runnable = false;
                            }
                            Ok(false) => runnable = true,
                            Err(err) => {
                                self.fail(err);
                                break;
                            }
                        }
                    }
                    Ok(LocalPoll::Complete { index, batch }) => {
                        self.complete(index, batch);
                        runnable =
                            !self.stopped.load(Ordering::Acquire) && morsel.assign_next(self);
                    }
                    Err(err) => {
                        self.fail(err);
                        break;
                    }
                }

                if let Err(err) = self.try_run_io() {
                    self.fail(err);
                    break;
                }
                continue;
            }

            crossbeam_channel::select_biased! {
                recv(signals) -> signal => match signal {
                    Ok(WorkerSignal::Wake(token)) if morsel.waiting == Some(token) => {
                        morsel.waiting = None;
                        runnable = true;
                    }
                    Ok(WorkerSignal::Wake(_)) => {}
                    Ok(WorkerSignal::Shutdown) | Err(_) => break,
                },
                recv(self.urgent_rx) -> work => match work {
                    Ok(work) => if let Err(err) = self.run_io(work) {
                        self.fail(err);
                        break;
                    },
                    Err(_) => break,
                },
                recv(self.ready_rx) -> work => match work {
                    Ok(work) => if let Err(err) = self.run_io(work) {
                        self.fail(err);
                        break;
                    },
                    Err(_) => break,
                },
            }
        }
        morsel.stats
    }

    fn complete(&self, index: usize, batch: Option<ArrayRef>) {
        if let Some(completion) = &self.run.completion {
            completion(index, batch);
        } else if let Some(batch) = batch {
            self.results.lock().push((index, batch));
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.stop();
        }
    }

    fn fail(&self, err: VortexError) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            *self.error.lock() = Some(err);
            self.send_shutdown();
        }
    }

    fn stop(&self) {
        if !self.stopped.swap(true, Ordering::AcqRel) {
            self.send_shutdown();
        }
    }

    fn send_shutdown(&self) {
        for tx in &self.worker_tx {
            drop(tx.send(WorkerSignal::Shutdown));
        }
    }

    fn finish(&self, worker_stats: Vec<ScanStats>) -> VortexResult<(Vec<ArrayRef>, ScanStats)> {
        if let Some(err) = self.error.lock().take() {
            return Err(err);
        }

        let mut stats = ScanStats::default();
        for worker in worker_stats {
            stats.merge(&worker);
        }
        stats.io_bytes += self.io_bytes.load(Ordering::Relaxed);
        stats.io_waits = self.io_waits.load(Ordering::Relaxed);
        stats.io_wait_time = Duration::from_nanos(self.io_wait_nanos.load(Ordering::Relaxed));

        let mut results = std::mem::take(&mut *self.results.lock());
        results.sort_unstable_by_key(|(index, _)| *index);
        Ok((results.into_iter().map(|(_, array)| array).collect(), stats))
    }
}

enum LocalPoll {
    Runnable,
    Blocked(WaitSet),
    Complete {
        index: usize,
        batch: Option<ArrayRef>,
    },
}

impl LocalMorsel {
    fn new(run: &WorkerRun) -> Self {
        Self {
            arena: run.plan.instantiate(),
            io: IoPlane::new(Arc::clone(&run.io)),
            phase: TaskPhase::Plan,
            index: 0,
            range: 0..0,
            active: false,
            generation: 0,
            wait_epoch: 0,
            waiting: None,
            morsel_io_uses_start: 0,
            morsel_io_requests_start: 0,
            morsel_io_batches_start: 0,
            morsel_io_blocks_start: 0,
            stats: ScanStats::default(),
        }
    }

    fn assign_next(&mut self, scheduler: &Scheduler) -> bool {
        let index = scheduler.next_morsel.fetch_add(1, Ordering::Relaxed);
        let Some(range) = scheduler.run.morsels.get(index).cloned() else {
            self.active = false;
            return false;
        };

        self.index = index;
        self.range = range.clone();
        self.phase = TaskPhase::Plan;
        self.active = true;
        self.generation = self.generation.wrapping_add(1);
        self.wait_epoch = 0;
        self.waiting = None;
        self.morsel_io_uses_start = self.stats.io_uses;
        self.morsel_io_requests_start = self.stats.io_requests;
        self.morsel_io_batches_start = self.stats.io_batches;
        self.morsel_io_blocks_start = self.stats.execute_io_blocks;
        self.io.clear();
        begin_morsel(&mut self.arena, scheduler.run.plan.root(), range);
        true
    }

    fn next_wait_token(&mut self) -> WaitToken {
        self.wait_epoch = self.wait_epoch.wrapping_add(1);
        WaitToken {
            generation: self.generation,
            epoch: self.wait_epoch,
        }
    }

    fn run(&mut self, scheduler: &Arc<Scheduler>) -> VortexResult<LocalPoll> {
        debug_assert!(self.active);
        match self.phase {
            TaskPhase::Plan => {
                let poll = poll_plan_morsel(
                    &mut self.arena,
                    scheduler.run.plan.root(),
                    &self.io,
                    &scheduler.run.cells,
                    &mut self.stats,
                )?;
                self.stats.io_batches += scheduler.submit_reads(self.io.take_reads());
                match poll {
                    PlanPoll::Item(_) => Ok(LocalPoll::Runnable),
                    PlanPoll::Blocked(waits) => Ok(LocalPoll::Blocked(waits)),
                    PlanPoll::Complete => {
                        self.stats.morsels += 1;
                        self.phase = TaskPhase::Execute;
                        Ok(LocalPoll::Runnable)
                    }
                }
            }
            TaskPhase::Execute => match poll_execute_morsel(
                &mut self.arena,
                scheduler.run.plan.root(),
                &self.range,
                &self.io,
                &scheduler.run.cells,
                &scheduler.run.session,
                &mut self.stats,
            )? {
                ExecPoll::Value(batch) => {
                    let array = batch.value.into_array()?;
                    let array = (!array.is_empty()).then_some(array);
                    self.finish_morsel(scheduler, array)
                }
                ExecPoll::Yield(_) => Ok(LocalPoll::Runnable),
                ExecPoll::Blocked(waits) => {
                    self.stats.execute_io_blocks += 1;
                    Ok(LocalPoll::Blocked(waits))
                }
                ExecPoll::Done => self.finish_morsel(scheduler, None),
            },
        }
    }

    fn finish_morsel(
        &mut self,
        scheduler: &Scheduler,
        batch: Option<ArrayRef>,
    ) -> VortexResult<LocalPoll> {
        retire_morsel(
            &mut self.arena,
            scheduler.run.plan.root(),
            &scheduler.run.cells,
            &mut self.stats,
        );
        self.io.clear();
        if batch.is_some() && self.stats.time_to_first_batch.is_none() {
            self.stats.time_to_first_batch = Some(scheduler.run.start.elapsed());
        }
        let io_uses = self.stats.io_uses - self.morsel_io_uses_start;
        let io_requests = self.stats.io_requests - self.morsel_io_requests_start;
        let io_batches = self.stats.io_batches - self.morsel_io_batches_start;
        let io_blocks = self.stats.execute_io_blocks - self.morsel_io_blocks_start;
        self.stats
            .record_morsel_io(io_uses, io_requests, io_batches, io_blocks);
        self.active = false;
        Ok(LocalPoll::Complete {
            index: self.index,
            batch,
        })
    }
}

impl MorselScan {
    /// Configure a scan over a built plan.
    pub fn new(
        plan: Arc<ExecPlan>,
        segments: Arc<dyn SegmentSource>,
        session: VortexSession,
    ) -> Self {
        let morsels = Arc::from(morsels(&plan, 0));
        Self {
            plan,
            segments,
            session,
            morsels,
            threads: 1,
            share_decodes: true,
            completion: None,
            worker_pool: None,
        }
    }

    /// Set the number of driving threads and affinity-owned active morsels.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Override the morsel cut.
    pub fn with_morsels(mut self, morsels: Vec<Range<u64>>) -> Self {
        self.morsels = Arc::from(morsels);
        self
    }

    /// Enable or disable the leased shared decoded cells.
    pub fn with_share_decodes(mut self, share: bool) -> Self {
        self.share_decodes = share;
        self
    }

    /// Deliver each completed morsel to a sink instead of collecting all outputs.
    pub fn with_completion_sink(
        mut self,
        completion: impl Fn(usize, Option<ArrayRef>) + Send + Sync + 'static,
    ) -> Self {
        self.completion = Some(Arc::new(completion));
        self
    }

    /// Drive this scan with a persistent worker pool.
    pub fn with_worker_pool(mut self, worker_pool: Arc<SharedMorselWorkerPool>) -> Self {
        self.threads = worker_pool.threads;
        self.worker_pool = Some(worker_pool);
        self
    }

    fn lease_counts(&self) -> HashMap<IoKey, usize> {
        let mut counts: HashMap<IoKey, usize> = HashMap::default();
        for (key, range) in self.plan.flat_uses() {
            let overlapping = self
                .morsels
                .iter()
                .filter(|morsel| morsel.start < range.end && range.start < morsel.end)
                .count();
            if overlapping > 0 {
                *counts.entry(key).or_default() += overlapping;
            }
        }
        counts
    }

    /// The morsels this scan will drive.
    pub fn morsel_ranges(&self) -> &[Range<u64>] {
        &self.morsels
    }

    /// Run the scan, returning batches in row order plus the run's counters.
    pub fn run(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats)> {
        let (batches, stats, _) = self.run_timed()?;
        Ok((batches, stats))
    }

    /// Run the scan with worker creation and shutdown outside the measured interval.
    pub(crate) fn run_timed(&self) -> VortexResult<(Vec<ArrayRef>, ScanStats, Duration)> {
        let workers = (self.worker_pool.is_none() && self.threads > 1)
            .then(|| MorselWorkerPool::new(self.threads))
            .transpose()?;
        let start = Instant::now();
        let cells = if self.share_decodes {
            SharedCells::with_leases(self.lease_counts())
        } else {
            SharedCells::disabled()
        };
        let run = Arc::new(WorkerRun {
            plan: Arc::clone(&self.plan),
            session: self.session.clone(),
            morsels: Arc::clone(&self.morsels),
            io: IoService::new(Arc::clone(&self.segments)),
            cells,
            start,
            completion: self.completion.clone(),
        });

        let (scheduler, signals) = Scheduler::new(Arc::clone(&run), self.threads);
        let worker_stats = match (&self.worker_pool, workers.as_ref()) {
            (Some(workers), _) => workers.run(Arc::clone(&scheduler), signals)?,
            (None, Some(workers)) => workers.run(Arc::clone(&scheduler), signals)?,
            (None, None) => vec![scheduler.worker_loop(0, &signals[0])],
        };
        let (batches, stats) = scheduler.finish(worker_stats)?;

        debug_assert_eq!(
            run.cells.live(),
            0,
            "every lease must be released by the end of the scan"
        );

        let wall = start.elapsed();
        drop(workers);
        Ok((batches, stats, wall))
    }
}
