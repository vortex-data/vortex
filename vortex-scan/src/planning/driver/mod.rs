// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A single-worker driver for the protocol.
//!
//! A [`Run`] drives any number of root planners, each with its own IO session, and hands their
//! batches to its caller as they are produced. Runnable items are prioritised in DFS preorder,
//! with roots ordered by admission and children by emission. Each visit takes at most one CPU
//! step. A parent's continuation follows its emitted child's entire subtree. Waiting work is
//! skipped and regains its priority when IO makes it runnable. Published IO is registered
//! immediately and state is inspected again before parking. Output follows emission order,
//! not global row order. An earlier runnable branch can starve later branches.
//!
//! A run never blocks: [`Run::advance`] returns [`Progress::Waiting`] when every live item waits
//! for IO, and the caller decides how to wait, by awaiting [`Run::poll_completion`] or by calling
//! [`Run::poll_io`] once it knows reads have finished. [`Driver::run`] is the blocking
//! convenience for one root.
//!
//! Every root reads through its own [`SegmentIoSession`]. The run holds the handles of what its
//! items published: fetches until they are delivered, and announcements and prefetches until the
//! item that published them retires.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use vortex_array::ArrayRef;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::segments::IoHandle;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoKey;
use vortex_io::segments::IoOwnerId;
use vortex_io::segments::IoRequest;
use vortex_io::segments::SegmentIoSession;
use vortex_io::segments::SourceResult;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// Identifies one root admitted to a [`Run`]. Unique within the run, in admission order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RootId(pub u64);

/// One array produced by a morsel, with the scope the morsel was created for.
pub struct Batch {
    /// The root the morsel descends from.
    pub root: RootId,
    /// Scope of the morsel that produced the array.
    pub scope: WorkScope,
    /// The non-empty array.
    pub array: ArrayRef,
}

/// What a call to [`Run::advance`] left behind.
pub enum Progress {
    /// A morsel's batch. One morsel's batches arrive in the order it emitted them.
    Batch(Batch),
    /// Every item descended from the root has retired, and its IO session was dropped.
    RootDone(RootId),
    /// Every live item waits for IO. Await [`Run::poll_completion`], or call [`Run::poll_io`]
    /// once reads have finished.
    Waiting,
    /// Nothing is live: every admitted root has finished.
    Idle,
}

enum Item {
    Planner(Box<dyn Planner>),
    Morsel(Box<dyn Morsel>),
}

/// What one item's `compute()` produced.
enum Output {
    Planner(PlannerOutput),
    Morsel(MorselOutput),
}

/// One live planner or morsel. Boxed in the run queue and the parked map, so scheduling moves a
/// pointer.
struct Work {
    id: IoOwnerId,
    root: RootId,
    scope: WorkScope,
    item: Item,
    /// Stable child-ordinal path, beginning with the root's admission ordinal.
    path: Vec<u64>,
    /// The parent continuation sorts after all previously emitted child subtrees.
    next_child: u64,
    /// Published fetches not yet delivered. Each finishes with its key and result, so a report
    /// from the session yields exactly the fetches that finished.
    fetches: FuturesUnordered<BoxFuture<'static, (IoKey, FetchResult)>>,
    /// The keys of `fetches`, to reject a fetch published while one of the same key is
    /// outstanding.
    outstanding: HashSet<IoKey>,
    /// Announcements and prefetches, held until the item retires.
    hints: Vec<IoHandle>,
}

type FetchResult = Result<SourceResult, Arc<VortexError>>;

impl Work {
    fn priority(&self) -> Vec<u64> {
        let mut priority = self.path.clone();
        priority.push(self.next_child);
        priority
    }

    // Child k takes the continuation's old key as its path. Its subtree then sorts before
    // the parent's new continuation at k + 1, even if descendants are discovered after IO.
    fn child_path(&mut self) -> Vec<u64> {
        let path = self.priority();
        self.next_child += 1;
        path
    }

    fn state(&self) -> State {
        match &self.item {
            Item::Planner(planner) => planner.state(),
            Item::Morsel(morsel) => morsel.state(),
        }
    }

    /// Hands every finished fetch to the item, in the order they finished. A failed fetch fails
    /// the run.
    fn deliver_finished(&mut self) -> VortexResult<()> {
        let mut cx = Context::from_waker(Waker::noop());
        while let Poll::Ready(Some((key, result))) = self.fetches.poll_next_unpin(&mut cx) {
            self.outstanding.remove(&key);
            let result = result.map_err(|error| {
                VortexError::from(error).with_context(format!("reading {key:?}"))
            })?;
            match &mut self.item {
                Item::Planner(planner) => planner.deliver(key, result),
                Item::Morsel(morsel) => morsel.deliver(key, result),
            }
        }
        Ok(())
    }
}

/// Keeps scheduling order separate from lookup for deliveries to runnable work.
#[derive(Default)]
struct ReadyQueue {
    priorities: BTreeMap<Vec<u64>, IoOwnerId>,
    work: HashMap<IoOwnerId, Box<Work>>,
}

impl ReadyQueue {
    fn insert(&mut self, work: Box<Work>) {
        self.priorities.insert(work.priority(), work.id);
        self.work.insert(work.id, work);
    }

    fn pop(&mut self) -> Option<Box<Work>> {
        let (_, owner) = self.priorities.pop_first()?;
        self.work.remove(&owner)
    }

    fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    fn cancel(&mut self, root: RootId) {
        self.work.retain(|_, work| work.root != root);
        self.priorities
            .retain(|_, owner| self.work.contains_key(owner));
    }
}

/// Poll IO even when one high-priority branch keeps producing CPU work.
const IO_POLL_INTERVAL: usize = 64;

/// One admitted root: the session its descendants read through, how many of them are live, and
/// how many of their fetches the session has not reported yet.
struct Root {
    io: Arc<dyn SegmentIoSession>,
    live: usize,
    in_flight: usize,
}

/// A set of root planners advanced by its caller on one thread at a time.
///
/// Dropping a run drops the IO session of every root still live, even when it ends in an error.
#[derive(Default)]
pub struct Run {
    roots: HashMap<RootId, Root>,
    next_root: u64,
    next_owner: u64,
    queue: ReadyQueue,
    parked: HashMap<IoOwnerId, Box<Work>>,
    /// Batches and finished roots not yet returned by [`advance`](Self::advance).
    events: VecDeque<Progress>,
    /// Visits left before the sources are polled for completions again.
    visits_until_poll: usize,
    step_limit: Option<usize>,
    steps: usize,
}

impl Run {
    /// Creates a run with nothing admitted.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fails with an error naming the limit if more than `steps` visits are needed. Tests use it
    /// to catch loops.
    pub fn with_step_limit(mut self, steps: usize) -> Self {
        self.step_limit = Some(steps);
        self
    }

    /// Admits `root` and everything it spawns, reading through `io`.
    ///
    /// The session belongs to the root: the run submits the requests of every descendant to it,
    /// and drops it once they have all retired or the root is cancelled.
    pub fn admit(
        &mut self,
        root: Box<dyn Planner>,
        scope: WorkScope,
        io: Arc<dyn SegmentIoSession>,
    ) -> RootId {
        let id = RootId(self.next_root);
        self.next_root += 1;
        self.roots.insert(
            id,
            Root {
                io,
                live: 0,
                in_flight: 0,
            },
        );
        self.spawn(id, scope, Item::Planner(root), vec![id.0]);
        id
    }

    /// Drops every live item descended from `root` and its session. Batches the root already
    /// produced are still returned, followed by no [`Progress::RootDone`].
    pub fn cancel(&mut self, root: RootId) {
        if self.roots.remove(&root).is_none() {
            return;
        }
        self.queue.cancel(root);
        self.parked.retain(|_, work| work.root != root);
    }

    /// The number of admitted roots that have not finished.
    pub fn live_roots(&self) -> usize {
        self.roots.len()
    }

    /// Runs ready work, taking completions the sources already have, until there is something to
    /// hand back or everything left waits for IO. Never blocks.
    pub fn advance(&mut self) -> VortexResult<Progress> {
        loop {
            if let Some(event) = self.events.pop_front() {
                return Ok(event);
            }
            if self.visits_until_poll == 0 || self.queue.is_empty() {
                self.poll_io()?;
                self.visits_until_poll = IO_POLL_INTERVAL;
                if self.queue.is_empty() {
                    return Ok(if self.parked.is_empty() {
                        Progress::Idle
                    } else {
                        Progress::Waiting
                    });
                }
            }
            self.visits_until_poll -= 1;
            let Some(work) = self.queue.pop() else {
                continue;
            };
            self.steps += 1;
            if let Some(limit) = self.step_limit
                && self.steps > limit
            {
                vortex_bail!("driver exceeded its step limit of {limit}");
            }
            self.visit(work)?;
        }
    }

    /// Waits for a fetch of any root's session to finish, registering `cx` to be woken by the
    /// next one, and delivers every finished fetch.
    ///
    /// Call it after [`advance`](Self::advance) returned [`Progress::Waiting`].
    pub fn poll_completion(&mut self, cx: &mut Context<'_>) -> Poll<VortexResult<()>> {
        if self.roots.values().all(|root| root.in_flight == 0) {
            return Poll::Ready(Err(vortex_err!(
                "the run is waiting for IO with no fetch in flight"
            )));
        }
        match self.poll_sessions(cx) {
            Ok(true) => Poll::Ready(Ok(())),
            Ok(false) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        }
    }

    /// Delivers every fetch the roots' sessions have finished, without blocking.
    pub fn poll_io(&mut self) -> VortexResult<()> {
        self.poll_sessions(&mut Context::from_waker(Waker::noop()))
            .map(drop)
    }

    /// Delivers the fetches every root's session reports finished, returning whether any was.
    fn poll_sessions(&mut self, cx: &mut Context<'_>) -> VortexResult<bool> {
        let mut owners = Vec::new();
        for root in self.roots.values_mut() {
            if root.in_flight == 0 {
                continue;
            }
            let before = owners.len();
            if let Poll::Ready(result) = root.io.poll_ready(cx, &mut owners) {
                result?;
            }
            root.in_flight -= owners.len() - before;
        }
        for owner in &owners {
            self.deliver(*owner)?;
        }
        Ok(!owners.is_empty())
    }

    /// Delivers the finished fetches of `owner`, if it is still live, and requeues it once it no
    /// longer waits.
    fn deliver(&mut self, owner: IoOwnerId) -> VortexResult<()> {
        if let Some(work) = self.parked.get_mut(&owner) {
            work.deliver_finished()?;
            if work.state() == State::Waiting {
                if work.outstanding.is_empty() {
                    vortex_bail!(
                        "work for scope {:?} waits with no outstanding fetch",
                        work.scope
                    );
                }
            } else if let Some(work) = self.parked.remove(&owner) {
                self.queue.insert(work);
            }
            return Ok(());
        }
        // An item that published fetches and had CPU work left is still in the run queue. One
        // that retired or was cancelled is gone, and its fetches are dropped.
        match self.queue.work.get_mut(&owner) {
            Some(work) => work.deliver_finished(),
            None => Ok(()),
        }
    }

    fn spawn(&mut self, root: RootId, scope: WorkScope, item: Item, path: Vec<u64>) {
        let id = IoOwnerId(self.next_owner);
        self.next_owner += 1;
        if let Some(root) = self.roots.get_mut(&root) {
            root.live += 1;
        }
        let work = Box::new(Work {
            id,
            root,
            scope,
            item,
            path,
            next_child: 0,
            fetches: FuturesUnordered::new(),
            outstanding: HashSet::default(),
            hints: Vec::new(),
        });
        self.queue.insert(work);
    }

    /// Retires a finished item, and its root once nothing descended from it is live. The item's
    /// handles are dropped with it.
    fn retire(&mut self, work: &Work) {
        let Some(root) = self.roots.get_mut(&work.root) else {
            return;
        };
        root.live -= 1;
        if root.live == 0 {
            self.roots.remove(&work.root);
            self.events.push_back(Progress::RootDone(work.root));
        }
    }

    /// Parks an item that waits for IO, which is a protocol error with no fetch outstanding.
    fn park(&mut self, work: Box<Work>) -> VortexResult<()> {
        if work.outstanding.is_empty() {
            vortex_bail!(
                "work for scope {:?} waits with no outstanding fetch",
                work.scope
            );
        }
        self.parked.insert(work.id, work);
        Ok(())
    }

    /// Requeues, parks, or retires an item according to the state it reports.
    fn settle(&mut self, work: Box<Work>) -> VortexResult<()> {
        match work.state() {
            State::Done => self.retire(&work),
            State::NeedsCompute => self.queue.insert(work),
            State::Waiting => self.park(work)?,
        }
        Ok(())
    }

    /// Visits one item: computes once if it is ready, and settles it otherwise.
    fn visit(&mut self, mut work: Box<Work>) -> VortexResult<()> {
        let output = match &mut work.item {
            Item::Planner(planner) => match planner.state() {
                State::NeedsCompute => Output::Planner(planner.compute()?),
                _ => return self.settle(work),
            },
            Item::Morsel(morsel) => match morsel.state() {
                State::NeedsCompute => Output::Morsel(morsel.compute()?),
                _ => return self.settle(work),
            },
        };
        match output {
            Output::Planner(PlannerOutput::Done) | Output::Morsel(MorselOutput::Done) => {
                self.retire(&work);
            }
            Output::Planner(PlannerOutput::Continue) | Output::Morsel(MorselOutput::Continue) => {
                self.queue.insert(work);
            }
            Output::Planner(PlannerOutput::NeedsIO(batch))
            | Output::Morsel(MorselOutput::NeedsIO(batch)) => {
                self.submit(&mut work, batch)?;
                self.settle(work)?;
            }
            Output::Planner(PlannerOutput::Planner(scope, child)) => {
                let path = work.child_path();
                self.spawn(work.root, scope, Item::Planner(child), path);
                self.queue.insert(work);
            }
            Output::Planner(PlannerOutput::Morsel(scope, morsel)) => {
                let path = work.child_path();
                self.spawn(work.root, scope, Item::Morsel(morsel), path);
                self.queue.insert(work);
            }
            Output::Morsel(MorselOutput::Batch(array)) => {
                if array.is_empty() {
                    vortex_bail!("morsel for scope {:?} returned an empty batch", work.scope);
                }
                self.events.push_back(Progress::Batch(Batch {
                    root: work.root,
                    scope: work.scope.clone(),
                    array,
                }));
                self.queue.insert(work);
            }
        }
        Ok(())
    }

    /// Registers the requests an item published with its root's session, and keeps their
    /// handles.
    fn submit(&mut self, work: &mut Work, batch: Vec<IoRequest>) -> VortexResult<()> {
        if batch.is_empty() {
            vortex_bail!("compute() published an empty batch");
        }
        for request in &batch {
            if request.intent == IoIntent::Fetch && !work.outstanding.insert(request.key) {
                vortex_bail!(
                    "fetch of {:?} by {:?} was published while it is outstanding",
                    request.key,
                    work.id
                );
            }
        }
        let Some(root) = self.roots.get_mut(&work.root) else {
            vortex_bail!("work {:?} outlived its root {:?}", work.id, work.root);
        };
        let handles = root.io.submit(work.id, &batch)?;
        for (request, handle) in batch.iter().zip(handles) {
            if request.intent != IoIntent::Fetch {
                work.hints.push(handle);
                continue;
            }
            root.in_flight += 1;
            let key = request.key;
            work.fetches.push(
                async move {
                    let result = handle.result().await;
                    // The handle keeps the read registered until it is delivered.
                    drop(handle);
                    (key, result)
                }
                .boxed(),
            );
        }
        Ok(())
    }
}

/// Drives one root to completion on the calling thread, blocking while it waits for IO.
pub struct Driver {
    io: Arc<dyn SegmentIoSession>,
    step_limit: Option<usize>,
}

impl Driver {
    /// Creates a driver that submits every request to `io`.
    pub fn new(io: Arc<dyn SegmentIoSession>) -> Self {
        Self {
            io,
            step_limit: None,
        }
    }

    /// Fails with an error naming the limit if more than `steps` visits are needed. Tests use it
    /// to catch loops.
    pub fn with_step_limit(self, steps: usize) -> Self {
        Self {
            step_limit: Some(steps),
            ..self
        }
    }

    /// Runs `root` and everything it spawns, returning batches in emission order.
    pub fn run(&self, root: Box<dyn Planner>) -> VortexResult<Vec<Batch>> {
        let mut run = Run::new();
        run.step_limit = self.step_limit;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..0,
        };
        run.admit(root, scope, Arc::clone(&self.io));
        let mut batches = Vec::new();
        loop {
            match run.advance()? {
                Progress::Batch(batch) => batches.push(batch),
                Progress::RootDone(_) | Progress::Idle => return Ok(batches),
                Progress::Waiting => block_on_completion(&mut run)?,
            }
        }
    }
}

/// Parks the calling thread until `run` delivers a finished fetch.
fn block_on_completion(run: &mut Run) -> VortexResult<()> {
    struct Unpark(std::thread::Thread);

    impl Wake for Unpark {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(result) = run.poll_completion(&mut cx) {
            return result;
        }
        std::thread::park();
    }
}

#[cfg(test)]
mod tests;
