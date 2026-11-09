// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runs many prepared V2 scans on one planning run.
//!
//! A [`ScanScheduler`] takes scans with the position of each in the output of its caller, such
//! as the table position of the first row that a scan of a file covers. It works all of them on one
//! [`Run`], so the chunk nearest to the start of the output always runs first, whichever scan it
//! belongs to. Each scan gets its batches as a stream in row order.
//!
//! The scheduler has no task of its own. The task that polls any of its streams drives the run
//! for all of them: CPU work runs on that task and only reads are awaited. A stream that is
//! polled therefore also makes progress for the scans that nobody polls yet, which is how the
//! scheduler runs ahead.
//!
//! With a runtime [`Handle`], the run on the polling task only plans: each chunk it admits
//! runs on a task of the runtime, with a run of its own, so the chunks of all scans decode in
//! parallel. The order of admission and the budget are the same.
//!
//! How far it runs ahead has two limits. At most
//! [`max_chunks`](ScanScheduler::with_max_chunks) chunks run at one time. With a
//! [`MemoryBudget`], the reads of all scans and the batches that nobody has consumed yet are
//! charged to one account, and while it is full only the work at the lowest position runs. The
//! budget is a soft limit: that lowest work always runs, so a full budget never stops the scan
//! that the caller waits for. Without a budget, the batches of a scan that nobody polls have no
//! limit, so set a budget when the scans are not consumed together.

use std::any::Any;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoService;
use vortex_io::request::IoSource;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_scan::planning::budget::MemoryBudget;
use vortex_scan::planning::budget::MemoryCharge;
use vortex_scan::planning::driver::Batch;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::RootId;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_utils::aliases::hash_map::HashMap;

use crate::scan::v2::DEFAULT_LOCAL_CHUNKS;
use crate::scan::v2::PreparedScan;

/// Runs the scans it is given on one planning run, nearest position first.
///
/// Clones share the run, so every scan of one query can go through the same scheduler.
#[derive(Clone)]
pub struct ScanScheduler {
    shared: Arc<Shared>,
}

struct Shared {
    state: Mutex<State>,
    /// What the chunks on tasks did and the state has not stored yet. A lock of its own guards
    /// it, so a task hands over an event without a wait for the lock of the state.
    events: Mutex<Vec<TaskEvent>>,
    /// The chunks on tasks that have not made their first step. The budget does not hold their
    /// reads yet.
    unstarted: Arc<AtomicUsize>,
    /// The tasks that wait for a read, for released bytes, or for another task to drive the run.
    waiting: Arc<Waiting>,
    budget: Option<Arc<MemoryBudget>>,
    /// Where chunks run, when not on the task that polls.
    handle: Option<Handle>,
}

impl Shared {
    /// Starts the tasks that the state recorded. To start a task wakes a thread of the
    /// runtime, so this runs without the lock of the state.
    ///
    /// The root that plans a scan hands the chunks that it spawns back to the scheduler. A
    /// chunk runs the planners that it spawns itself.
    fn launch(self: &Arc<Self>, pending: Vec<Pending>) {
        let Some(handle) = &self.handle else {
            return;
        };

        for (spawned, start, slot) in pending {
            let Spawned {
                owner,
                position: _,
                planner,
                scope,
                io,
                ordered,
                map,
                queue,
            } = spawned;
            let budget = self.budget.clone();
            // The scheduler owns the task, so the task must not keep the scheduler alive.
            let shared = Arc::downgrade(self);
            let task = handle.spawn(async move {
                let send = |event| {
                    if let Some(shared) = shared.upgrade() {
                        shared.deliver(event);
                    }
                };
                let started = || start.started();
                let spawned = (owner.chunk == PLANNING).then_some(|scope, planner| {
                    send(TaskEvent::Spawned(owner.scan, scope, planner));
                });
                let emit = (!ordered).then_some(|batch| push_ready(&queue, batch));

                let chunk = run_chunk(planner, scope, io, budget, map, started, spawned, emit);
                let batches = chunk.await;
                send(TaskEvent::Finished(owner, batches));
            });

            let mut slot = slot.lock();
            // A chunk that is already removed has no use for its task.
            if !slot.cancelled {
                slot.task = Some(task);
            }
        }
    }

    /// Hands over what a chunk on a task did. The tasks drive the scheduler this way, so a
    /// stream that is polled only takes its batches.
    fn deliver(self: &Arc<Self>, event: TaskEvent) {
        self.events.lock().push(event);
        self.drain();
    }

    /// Stores the events of the tasks, admits the chunks that they made room for, and wakes the
    /// scans that got batches.
    ///
    /// This never waits for the lock of the state: many tasks finish chunks at the same time,
    /// and a task that waits for a lock keeps a thread of the runtime from its work. When
    /// another thread holds the lock, that thread stores the events, because every holder calls
    /// this after it releases the lock.
    fn drain(self: &Arc<Self>) {
        loop {
            if self.events.lock().is_empty() {
                return;
            }
            let Some(mut state) = self.state.try_lock() else {
                return;
            };

            let wakers = state.store_events(self, None);
            // Only a scan that is polled can run work past a full budget, so the streams that
            // are polled admit the chunk that a full budget holds back here.
            let held_back = state.admit(self, true);
            let failed = state.failed.is_some() || held_back;
            let pending = std::mem::take(&mut state.pending);
            drop(state);
            self.launch(pending);

            if failed {
                self.waiting.wake_all();
            }
            for waker in wakers {
                waker.wake();
            }
        }
    }
}

impl Default for ScanScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ScanScheduler {
    /// A scheduler that runs at most [`DEFAULT_LOCAL_CHUNKS`] chunks at one time, with no
    /// memory budget.
    pub fn new() -> Self {
        Self::build(DEFAULT_LOCAL_CHUNKS, None, None)
    }

    /// Runs at most `max_chunks` chunks at one time, over all scans. Set it before the first
    /// scan is added.
    pub fn with_max_chunks(self, max_chunks: usize) -> Self {
        Self::build(
            max_chunks,
            self.shared.budget.clone(),
            self.shared.handle.clone(),
        )
    }

    /// Runs each chunk on a task of `handle`, not on the task that polls. Set it before the
    /// first scan is added.
    pub fn with_handle(self, handle: Handle) -> Self {
        let max_chunks = self.shared.state.lock().max_chunks;
        Self::build(max_chunks, self.shared.budget.clone(), Some(handle))
    }

    /// Charges `budget` for the reads of every scan and for the batches that nobody has consumed
    /// yet, and runs only the lowest position while it is full. Set it before the first scan is
    /// added.
    pub fn with_memory_budget(self, budget: Arc<MemoryBudget>) -> Self {
        let max_chunks = self.shared.state.lock().max_chunks;
        Self::build(max_chunks, Some(budget), self.shared.handle.clone())
    }

    fn build(max_chunks: usize, budget: Option<Arc<MemoryBudget>>, handle: Option<Handle>) -> Self {
        let mut run = Run::new();
        if let Some(budget) = &budget {
            run = run.with_memory_budget(Arc::clone(budget));
        }

        Self {
            shared: Arc::new(Shared {
                events: Mutex::new(Vec::new()),
                unstarted: Arc::new(AtomicUsize::new(0)),
                state: Mutex::new(State {
                    run,
                    scans: HashMap::default(),
                    roots: HashMap::default(),
                    queued: BTreeMap::new(),
                    tasks: HashMap::default(),
                    task_positions: BTreeMap::new(),
                    planning: BTreeMap::new(),
                    running: 0,
                    max_chunks: max_chunks.max(1),
                    pending: Vec::new(),
                    next_scan: 0,
                    next_queued: 0,
                    failed: None,
                }),
                waiting: Arc::new(Waiting::default()),
                budget,
                handle,
            }),
        }
    }

    /// The memory budget that the scans share, when there is one.
    pub fn memory_budget(&self) -> Option<&Arc<MemoryBudget>> {
        self.shared.budget.as_ref()
    }

    /// Adds `scan`, whose first covered row is at `position` in the output of the caller, and
    /// streams its batches in row order. Scans with the same position share the chunk places
    /// evenly.
    ///
    /// The scan starts when any stream of the scheduler is polled. Dropping the stream cancels
    /// the work of the scan.
    pub fn scan(
        &self,
        scan: Arc<PreparedScan>,
        position: u64,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        self.scan_map(scan, position, true, Ok)
    }

    /// Adds `scan` as [`scan`](Self::scan) does, and streams each batch as soon as its chunk
    /// makes it, in no order. A caller that needs no row order gets its first rows sooner.
    pub fn scan_unordered(
        &self,
        scan: Arc<PreparedScan>,
        position: u64,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        self.scan_map(scan, position, false, Ok)
    }

    /// Adds `scan` as [`scan`](Self::scan) does, in row order when `ordered`, and streams what
    /// `map` makes of each batch.
    ///
    /// `map` runs where the batch was decoded, right after it: on the task of the chunk when
    /// the scheduler has a runtime handle. A caller that converts each batch, to Arrow for
    /// example, then converts it in parallel and while its bytes are in the cache of that core.
    pub fn scan_map<T: Send + 'static>(
        &self,
        scan: Arc<PreparedScan>,
        position: u64,
        ordered: bool,
        map: impl Fn(ArrayRef) -> VortexResult<T> + Send + Sync + 'static,
    ) -> BoxStream<'static, VortexResult<T>> {
        let map: MapBatch = Arc::new(move |array| Ok(Box::new(map(array)?) as Item));
        let queue = Arc::new(Mutex::new(Queue::default()));
        let io = scan.io();
        let rows = scan.rows().clone();
        let first_row = rows.start;
        let scope = WorkScope {
            file_ordinal: 0,
            rows,
        };

        let mut state = self.shared.state.lock();
        let id = state.next_scan;
        state.next_scan += 1;

        let owner = Owner {
            scan: id,
            chunk: PLANNING,
        };
        match &self.shared.handle {
            Some(_) => {
                let planner = scan.planner();
                let io = io.session();
                let map = Arc::clone(&map);
                let spawned = Spawned {
                    owner,
                    position,
                    planner,
                    scope,
                    io,
                    ordered,
                    map,
                    queue: Arc::clone(&queue),
                };
                state.spawn(spawned, &self.shared.unstarted);
            }
            None => {
                let root = state
                    .run
                    .admit_at(scan.planner(), scope, io.session(), position);
                state.roots.insert(root, owner);
            }
        }

        let mut held = BTreeMap::new();
        held.insert(PLANNING, Held::default());
        state.scans.insert(
            id,
            ScanState {
                _scan: scan,
                io,
                position,
                first_row,
                held,
                next_chunk: PLANNING + 1,
                ordered,
                map,
                queue: Arc::clone(&queue),
            },
        );
        let pending = std::mem::take(&mut state.pending);
        drop(state);
        self.shared.launch(pending);
        self.shared.drain();

        // A task that waits for the run must see the new scan.
        self.shared.waiting.wake_all();

        let items = ScanStream {
            shared: Arc::clone(&self.shared),
            id,
            queue,
            done: false,
        };
        items
            .map(|item| {
                let item = item?.downcast::<T>();
                let item = item.map_err(|_| vortex_err!("A scheduled scan got a foreign batch"))?;
                Ok(*item)
            })
            .boxed()
    }
}

/// What the function of a scan made of one batch. The stream of the scan knows its type.
type Item = Box<dyn Any + Send>;

/// The function that a scan applies to each of its batches.
type MapBatch = Arc<dyn Fn(ArrayRef) -> VortexResult<Item> + Send + Sync>;

/// The key, in the held batches of a scan, of the root that plans it. Its chunks follow.
const PLANNING: u64 = 0;

/// The wakers of the tasks that wait on the scheduler. It is itself the waker that the scheduler
/// gives to its reads and its budget, so any event wakes every waiting task and one of them
/// drives the run.
#[derive(Default)]
struct Waiting(Mutex<Vec<Waker>>);

impl Waiting {
    fn register(&self, waker: &Waker) {
        let mut wakers = self.0.lock();
        if !wakers.iter().any(|known| known.will_wake(waker)) {
            wakers.push(waker.clone());
        }
    }

    fn wake_all(&self) {
        let wakers = std::mem::take(&mut *self.0.lock());
        for waker in wakers {
            waker.wake();
        }
    }
}

impl Wake for Waiting {
    fn wake(self: Arc<Self>) {
        self.wake_all();
    }
}

/// The scan and the chunk of it that a root of the run belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Owner {
    scan: u64,
    chunk: u64,
}

/// A chunk that a scan spawned and the run has not admitted yet.
struct Queued {
    owner: Owner,
    scope: WorkScope,
    planner: Box<dyn Planner>,
}

/// A chunk, or the root that plans a scan, to run on a task.
struct Spawned {
    owner: Owner,
    position: u64,
    planner: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    ordered: bool,
    map: MapBatch,
    queue: Arc<Mutex<Queue>>,
}

/// A chunk to start on a task, with the mark of its first step and the place for its task.
type Pending = (Spawned, Start, Arc<Mutex<TaskSlot>>);

/// Marks the first step of a chunk on a task. Until then, the budget does not hold the reads of
/// the chunk, so the scheduler keeps bytes free for it.
///
/// The chunk marks its own start without the lock of the scheduler: a start changes only how
/// many bytes the next admission keeps free.
#[derive(Clone)]
struct Start {
    started: Arc<AtomicBool>,
    unstarted: Arc<AtomicUsize>,
}

impl Start {
    /// Counts one more chunk that has not started.
    fn new(unstarted: &Arc<AtomicUsize>) -> Self {
        unstarted.fetch_add(1, Ordering::Relaxed);
        Self {
            started: Arc::new(AtomicBool::new(false)),
            unstarted: Arc::clone(unstarted),
        }
    }

    /// Records that the chunk made its first step, or that the scheduler removed it. Only the
    /// first call counts.
    fn started(&self) {
        if !self.started.swap(true, Ordering::Relaxed) {
            self.unstarted.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// What a chunk on a task did.
enum TaskEvent {
    /// The root that plans a scan spawned a chunk.
    Spawned(u64, WorkScope, Box<dyn Planner>),
    /// The chunk ran to its end, or an error stopped it.
    Finished(Owner, VortexResult<Vec<HeldBatch>>),
}

/// The bytes that the budget keeps free for a chunk on a task that has not made its first step.
/// The reads of that chunk are not charged yet.
const UNSTARTED_CHUNK_BYTES: u64 = 1 << 20;

/// A chunk, or the root that plans a scan, that runs on a task of the runtime.
struct ChunkTask {
    position: u64,
    start: Start,
    slot: Arc<Mutex<TaskSlot>>,
}

/// The task of a chunk. It is stored after the task starts, without the lock of the state.
#[derive(Default)]
struct TaskSlot {
    /// Whether the scheduler removed the chunk.
    cancelled: bool,
    /// Dropping the task cancels the chunk.
    task: Option<Task<()>>,
}

/// The batches of one chunk, held until the chunk is done and every earlier chunk is yielded.
#[derive(Default)]
struct Held {
    done: bool,
    batches: Vec<HeldBatch>,
}

/// One batch with the row it starts at.
struct HeldBatch {
    start: u64,
    item: Item,
    /// The bytes of the batch, charged until the consumer takes it.
    _charge: Option<MemoryCharge>,
}

struct ScanState {
    /// Keeps the file's shared scan cache alive while the scan runs.
    _scan: Arc<PreparedScan>,
    io: Arc<dyn IoService>,
    /// The position of the first row that the scan covers in the output of the caller.
    position: u64,
    /// The first file row that the scan covers.
    first_row: u64,
    /// Batches by chunk. Chunks are spawned in row order, so their keys order their batches.
    held: BTreeMap<u64, Held>,
    next_chunk: u64,
    /// Whether the scan yields its batches in row order. Else each batch is ready at once.
    ordered: bool,
    /// What the scan makes of each batch.
    map: MapBatch,
    /// The batches ready to yield, which the stream of the scan shares.
    queue: Arc<Mutex<Queue>>,
}

/// The batches of one scan that are ready to yield, in row order when the scan is ordered.
///
/// A lock of its own guards it, not the lock of the scheduler: a chunk on a task hands over
/// each batch, and the stream takes each batch, without a wait for the other scans.
#[derive(Default)]
struct Queue {
    ready: VecDeque<HeldBatch>,
    /// The task that waits for the batches of the scan.
    waker: Option<Waker>,
}

/// Adds `batch` to `queue` and wakes the task that waits for it.
fn push_ready(queue: &Mutex<Queue>, batch: HeldBatch) {
    let waker = {
        let mut queue = queue.lock();
        queue.ready.push_back(batch);
        queue.waker.take()
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

impl ScanState {
    /// Moves the batches of every done chunk that no earlier chunk holds back to `ready`.
    fn flush(&mut self) {
        while let Some(entry) = self.held.first_entry() {
            if !entry.get().done {
                return;
            }

            let mut batches = entry.remove().batches;
            // Morsels of one chunk finish in whatever order their reads arrive.
            batches.sort_by_key(|batch| batch.start);
            self.queue.lock().ready.extend(batches);
        }
    }

    fn finished(&self) -> bool {
        self.held.is_empty() && self.queue.lock().ready.is_empty()
    }
}

struct State {
    run: Run,
    scans: HashMap<u64, ScanState>,
    roots: HashMap<RootId, Owner>,
    /// Chunks not yet admitted, by their position and then the order they were spawned in.
    queued: BTreeMap<(u64, u64), Queued>,
    /// The chunks that run on tasks of the runtime.
    tasks: HashMap<Owner, ChunkTask>,
    /// The chunks that run on tasks, by position.
    task_positions: BTreeMap<(u64, Owner), ()>,
    /// The scans whose planning root runs on a task, by position.
    planning: BTreeMap<(u64, u64), ()>,
    /// Chunks admitted and not yet done.
    running: usize,
    max_chunks: usize,
    /// The tasks to start once the lock of the state is released.
    pending: Vec<Pending>,
    next_scan: u64,
    next_queued: u64,
    /// Why the run failed. A failed run fails every scan.
    failed: Option<String>,
}

impl State {
    /// The lowest position of live work: a root of the run, or a chunk on a task.
    fn lowest_position(&self) -> Option<u64> {
        let task = self
            .task_positions
            .keys()
            .next()
            .map(|(position, _)| *position);
        match (self.run.lowest_position(), task) {
            (Some(root), Some(task)) => Some(root.min(task)),
            (root, task) => root.or(task),
        }
    }

    /// Admits queued chunks, lowest position first, while fewer than `max_chunks` run. A full
    /// budget admits only a chunk that no live work precedes, and no chunk when `strict`.
    /// Returns whether a full budget held a chunk back.
    fn admit(&mut self, shared: &Arc<Shared>, strict: bool) -> bool {
        while self.running < self.max_chunks {
            let lowest = self.lowest_position();
            let planning = self.planning.keys().next().map(|(position, _)| *position);
            let Some(key) = self.queued.keys().next().copied() else {
                return false;
            };

            let position = key.0;
            // A scan that still plans at a lower position can spawn a chunk before this one.
            // The last free place is kept for that chunk.
            if self.running + 1 >= self.max_chunks && planning.is_some_and(|at| at < position) {
                return false;
            }

            let ahead = lowest.is_some_and(|lowest| position > lowest);
            let unstarted = shared.unstarted.load(Ordering::Relaxed) as u64;
            let unstarted = unstarted.saturating_mul(UNSTARTED_CHUNK_BYTES);
            let full = shared
                .budget
                .as_deref()
                .is_some_and(|budget| budget.used().saturating_add(unstarted) >= budget.limit());
            if full && (ahead || strict) {
                return true;
            }

            let Some(Queued {
                owner,
                scope,
                planner,
            }) = self.queued.remove(&key)
            else {
                return false;
            };
            // A scan that was dropped leaves its queued chunks behind.
            let Some(scan) = self.scans.get(&owner.scan) else {
                continue;
            };

            // Each root reads through a session of its own, because the run clears a root's
            // session once the root is done.
            let io = scan.io.session();
            self.running += 1;

            if shared.handle.is_none() {
                let root = self.run.admit_at(planner, scope, io, position);
                self.roots.insert(root, owner);
                continue;
            }

            let spawned = Spawned {
                owner,
                position,
                planner,
                scope,
                io,
                ordered: scan.ordered,
                map: Arc::clone(&scan.map),
                queue: Arc::clone(&scan.queue),
            };
            self.spawn(spawned, &shared.unstarted);
        }
        false
    }

    /// Records a chunk, or the root that plans a scan, that runs on a task. The caller starts
    /// the task with [`Shared::launch`] after it releases the lock of the state.
    fn spawn(&mut self, spawned: Spawned, unstarted: &Arc<AtomicUsize>) {
        let (owner, position) = (spawned.owner, spawned.position);
        let slot = Arc::new(Mutex::new(TaskSlot::default()));
        let start = Start::new(unstarted);
        self.pending
            .push((spawned, start.clone(), Arc::clone(&slot)));

        self.tasks.insert(
            owner,
            ChunkTask {
                position,
                start,
                slot,
            },
        );
        self.task_positions.insert((position, owner), ());
        if owner.chunk == PLANNING {
            self.planning.insert((position, owner.scan), ());
        }
    }

    /// Queues a chunk that the root of `scan` spawned.
    fn queue(&mut self, scan: u64, scope: WorkScope, planner: Box<dyn Planner>) {
        let Some(state) = self.scans.get_mut(&scan) else {
            return;
        };
        let chunk = state.next_chunk;
        state.next_chunk += 1;
        state.held.insert(chunk, Held::default());

        let rows = scope.rows.start.saturating_sub(state.first_row);
        let position = state.position.saturating_add(rows);
        let owner = Owner { scan, chunk };
        self.queued.insert(
            (position, self.next_queued),
            Queued {
                owner,
                scope,
                planner,
            },
        );
        self.next_queued += 1;
    }

    /// Removes the task of `owner`, when it has one.
    fn remove_task(&mut self, owner: Owner) -> bool {
        let Some(task) = self.tasks.remove(&owner) else {
            return false;
        };
        self.task_positions.remove(&(task.position, owner));
        if owner.chunk == PLANNING {
            self.planning.remove(&(task.position, owner.scan));
        } else {
            self.running -= 1;
        }
        task.start.started();

        let mut slot = task.slot.lock();
        slot.cancelled = true;
        slot.task = None;
        true
    }

    /// Stores the events that the tasks handed over. Returns the wakers of the scans that got
    /// batches to yield, but not the waker of the scan `polled`, whose task runs now.
    fn store_events(&mut self, shared: &Shared, polled: Option<u64>) -> Vec<Waker> {
        let events = std::mem::take(&mut *shared.events.lock());

        let mut wakers = Vec::new();
        for event in events {
            let Some(scan) = self.store(event) else {
                continue;
            };
            if polled == Some(scan) {
                continue;
            }
            if let Some(scan) = self.scans.get(&scan)
                && let Some(waker) = scan.queue.lock().waker.take()
            {
                wakers.push(waker);
            }
        }
        wakers
    }

    /// Stores what a chunk on a task did. Returns the scan that got batches to yield.
    fn store(&mut self, event: TaskEvent) -> Option<u64> {
        let (owner, batches) = match event {
            TaskEvent::Spawned(scan, scope, planner) => {
                self.queue(scan, scope, planner);
                return None;
            }
            TaskEvent::Finished(owner, batches) => (owner, batches),
        };
        // The chunk of a scan that was dropped has no task left.
        if !self.remove_task(owner) {
            return None;
        }

        let batches = match batches {
            Ok(batches) => batches,
            Err(error) => {
                self.failed = Some(error.to_string());
                return None;
            }
        };
        let scan = self.scans.get_mut(&owner.scan)?;
        if let Some(held) = scan.held.get_mut(&owner.chunk) {
            held.batches = batches;
            held.done = true;
        }
        scan.flush();
        Some(owner.scan)
    }

    /// Stores what the run handed back. Returns the scan that got batches to yield or finished.
    fn route(&mut self, progress: Progress) -> Option<u64> {
        match progress {
            Progress::Batch(Batch {
                root,
                scope,
                array,
                charge,
            }) => {
                let owner = self.roots.get(&root)?;
                let scan = self.scans.get_mut(&owner.scan)?;
                let item = match (scan.map)(array) {
                    Ok(item) => item,
                    Err(error) => {
                        self.failed = Some(error.to_string());
                        return None;
                    }
                };
                let batch = HeldBatch {
                    start: scope.rows.start,
                    item,
                    _charge: charge,
                };
                if !scan.ordered {
                    scan.queue.lock().ready.push_back(batch);
                    return Some(owner.scan);
                }
                scan.held.get_mut(&owner.chunk)?.batches.push(batch);
                None
            }
            Progress::Spawn(root, scope, planner) => {
                let scan = self.roots.get(&root)?.scan;
                self.queue(scan, scope, planner);
                None
            }
            Progress::RootDone(root) => {
                let owner = self.roots.remove(&root)?;
                if owner.chunk != PLANNING {
                    self.running -= 1;
                }

                let scan = self.scans.get_mut(&owner.scan)?;
                scan.held.get_mut(&owner.chunk)?.done = true;
                scan.flush();
                Some(owner.scan)
            }
            Progress::Waiting | Progress::Idle => None,
        }
    }

    /// Removes `scan` and cancels its work. The caller drops what this returns after it
    /// releases the lock of the state: the last reference to a prepared scan can take long to
    /// drop, and the other scans must not wait for it.
    fn remove(&mut self, scan: u64) -> Option<Removed> {
        let state = self.scans.remove(&scan)?;

        let roots: Vec<_> = self
            .roots
            .iter()
            .filter(|(_, owner)| owner.scan == scan)
            .map(|(root, owner)| (*root, owner.chunk))
            .collect();
        for (root, chunk) in roots {
            self.run.cancel(root);
            self.roots.remove(&root);
            if chunk != PLANNING {
                self.running -= 1;
            }
        }
        let queued = self
            .queued
            .extract_if(.., |_, queued| queued.owner.scan == scan)
            .map(|(_, queued)| queued)
            .collect();

        let tasks: Vec<_> = self
            .tasks
            .keys()
            .filter(|owner| owner.scan == scan)
            .copied()
            .collect();
        for owner in tasks {
            self.remove_task(owner);
        }

        Some(Removed {
            _state: state,
            _queued: queued,
        })
    }
}

/// What is left of a scan that the scheduler removed.
struct Removed {
    _state: ScanState,
    _queued: Vec<Queued>,
}

/// Runs a chunk's `planner` over `scope` to its end on a run of its own, and returns its
/// batches. A planner that the chunk spawns in turn runs as another root of the same run.
///
/// Calls `started` after the first step of the run, when a run with a budget has charged the
/// reads that the chunk asked for.
#[expect(
    clippy::too_many_arguments,
    reason = "the parts of one chunk and its three hooks"
)]
async fn run_chunk(
    planner: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    budget: Option<Arc<MemoryBudget>>,
    map: MapBatch,
    started: impl FnOnce(),
    mut spawned: Option<impl FnMut(WorkScope, Box<dyn Planner>)>,
    mut emit: Option<impl FnMut(HeldBatch)>,
) -> VortexResult<Vec<HeldBatch>> {
    let mut run = Run::new();
    // Only a scheduler with a budget waits for the chunk to start.
    let mut started = budget.is_some().then_some(started);
    if let Some(budget) = budget {
        run = run.with_memory_budget(budget);
    }
    run.admit(planner, scope, Arc::clone(&io));

    let mut batches = Vec::new();
    loop {
        let progress = run.advance()?;
        if let Some(started) = started.take() {
            started();
        }

        match progress {
            Progress::Batch(Batch {
                scope,
                array,
                charge,
                ..
            }) => {
                let batch = HeldBatch {
                    start: scope.rows.start,
                    item: map(array)?,
                    _charge: charge,
                };
                match &mut emit {
                    Some(emit) => emit(batch),
                    None => batches.push(batch),
                }
            }
            Progress::Spawn(_, scope, child) => match &mut spawned {
                Some(spawned) => spawned(scope, child),
                None => {
                    run.admit(child, scope, Arc::clone(&io));
                }
            },
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => return Ok(batches),
            Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
}

/// The batches of one scan of a [`ScanScheduler`].
struct ScanStream {
    shared: Arc<Shared>,
    id: u64,
    queue: Arc<Mutex<Queue>>,
    done: bool,
}

impl ScanStream {
    fn poll_scan(&mut self, cx: &mut Context<'_>) -> Poll<Option<VortexResult<Item>>> {
        let shared = Arc::clone(&self.shared);

        // A batch that is ready needs no lock of the scheduler.
        let ready = self.queue.lock().ready.pop_front();
        if let Some(batch) = ready {
            // The charge of the batch ends here: its bytes now belong to the consumer.
            return Poll::Ready(Some(Ok(batch.item)));
        }

        let mut state = shared.state.lock();
        // The events that the tasks handed over while another thread held the lock.
        let wakers = state.store_events(&shared, Some(self.id));
        // Whether this poll gave another scan something, so the task of that scan must run.
        let mut wake = false;
        // Dropped after the lock is released.
        let mut removed = None;

        let result = loop {
            if let Some(message) = &state.failed {
                let error = vortex_err!("The scan scheduler failed: {message}");
                removed = state.remove(self.id);
                break Poll::Ready(Some(Err(error)));
            }

            let Some(scan) = state.scans.get_mut(&self.id) else {
                break Poll::Ready(None);
            };
            if let Some(batch) = self.queue.lock().ready.pop_front() {
                break Poll::Ready(Some(Ok(batch.item)));
            }
            if scan.finished() {
                removed = state.remove(self.id);
                wake = true;
                break Poll::Ready(None);
            }

            state.admit(&shared, false);

            let progress = match state.run.advance() {
                Ok(progress) => progress,
                Err(error) => {
                    state.failed = Some(error.to_string());
                    removed = state.remove(self.id);
                    wake = true;
                    break Poll::Ready(Some(Err(error)));
                }
            };

            match progress {
                Progress::Waiting => {
                    // Register before the poll, so a read that completes meanwhile wakes this
                    // task.
                    shared.waiting.register(cx.waker());
                    let waker = Waker::from(Arc::clone(&shared.waiting));
                    let mut wait = Context::from_waker(&waker);

                    match state.run.poll_completion(&mut wait) {
                        Poll::Ready(Ok(())) => {}
                        Poll::Ready(Err(error)) => {
                            state.failed = Some(error.to_string());
                            removed = state.remove(self.id);
                            wake = true;
                            break Poll::Ready(Some(Err(error)));
                        }
                        Poll::Pending => {
                            if let Some(budget) = &shared.budget
                                && (state.run.is_throttled() || !state.queued.is_empty())
                            {
                                budget.wake_on_release(&wait);
                            }
                            break Poll::Pending;
                        }
                    }
                }
                Progress::Idle if !state.tasks.is_empty() => {
                    // The run has no work. The chunks on tasks wake this task when the scan
                    // gets batches, and released bytes wake it to admit a queued chunk.
                    {
                        let mut queue = self.queue.lock();
                        // A batch that arrived since the last look woke nobody.
                        if !queue.ready.is_empty() {
                            continue;
                        }
                        queue.waker = Some(cx.waker().clone());
                    }
                    shared.waiting.register(cx.waker());
                    if let Some(budget) = &shared.budget
                        && !state.queued.is_empty()
                    {
                        let waker = Waker::from(Arc::clone(&shared.waiting));
                        budget.wake_on_release(&Context::from_waker(&waker));
                    }
                    break Poll::Pending;
                }
                Progress::Idle => {
                    // Every admitted root is done. Queued chunks are admitted on the next turn,
                    // since nothing live precedes them.
                    if state.queued.is_empty() && self.queue.lock().ready.is_empty() {
                        removed = state.remove(self.id);
                        wake = true;
                        break Poll::Ready(None);
                    }
                }
                progress => {
                    if state.route(progress).is_some_and(|scan| scan != self.id) {
                        wake = true;
                    }
                }
            }
        };

        let pending = std::mem::take(&mut state.pending);
        drop(state);
        shared.launch(pending);
        drop(removed);
        if wake {
            shared.waiting.wake_all();
        }
        for waker in wakers {
            waker.wake();
        }
        shared.drain();
        result
    }
}

impl Stream for ScanStream {
    type Item = VortexResult<Item>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        let result = self.poll_scan(cx);
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            self.done = true;
        }
        result
    }
}

impl Drop for ScanStream {
    fn drop(&mut self) {
        let removed = self.shared.state.lock().remove(self.id);
        drop(removed);
        self.shared.drain();
        // Another task may wait on reads that this stream's task was to be woken for.
        self.shared.waiting.wake_all();
    }
}
