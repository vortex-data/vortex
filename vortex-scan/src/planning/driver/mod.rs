// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A single-worker driver for the protocol.
//!
//! A [`Run`] drives any number of root planners, each with its own IO session, and hands their
//! batches to its caller as they are produced. Runnable items sit in a FIFO run queue, taking at
//! most one CPU step per visit. Published IO is registered immediately and state is inspected
//! again before parking. Output follows completion order; FIFO scheduling does not imply row
//! ordering. Waiting work is parked, and comes back to the run queue when a delivery makes it
//! runnable, in whatever order the sources finish their reads.
//!
//! A run never blocks: [`Run::advance`] returns [`Progress::Waiting`] when every live item waits
//! for IO, and the caller decides how to wait, by awaiting [`Run::poll_completion`] or by handing
//! a completion it took from a source to [`Run::complete`]. [`Driver::run`] is the blocking
//! convenience for one root.

use std::collections::VecDeque;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::next::PendingPlanner;
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
    /// Every item descended from the root has retired, and its IO session was cleared.
    RootDone(RootId),
    /// Every live item waits for IO. Await [`Run::poll_completion`], or pass the next completion
    /// to [`Run::complete`].
    Waiting,
    /// Nothing is live: every admitted root has finished.
    Idle,
}

enum Item {
    Pending(Box<dyn PendingPlanner>),
    Live(Box<dyn Planner>),
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
    /// Published `Fetch` requests not yet delivered, with the target each must be answered with.
    outstanding: HashMap<IoRequestId, IoTarget>,
}

impl Work {
    fn state(&self) -> State {
        match &self.item {
            // Pending work is started by its next visit.
            Item::Pending(_) => State::NeedsCompute,
            Item::Live(planner) => planner.state(),
            Item::Morsel(morsel) => morsel.state(),
        }
    }

    fn consumer(&mut self) -> VortexResult<&mut dyn IoConsumer> {
        match &mut self.item {
            Item::Live(planner) => Ok(&mut **planner),
            Item::Morsel(morsel) => Ok(&mut **morsel),
            Item::Pending(_) => vortex_bail!("pending work cannot receive IO"),
        }
    }

    /// Checks `result` against the outstanding request it answers and hands it to the item.
    fn deliver(&mut self, request: IoRequestId, result: IoResult) -> VortexResult<()> {
        let Some(target) = self.outstanding.remove(&request) else {
            vortex_bail!(
                "completion for {request:?} of {:?}, which is not outstanding",
                self.id
            );
        };
        if !result.matches(&target) {
            vortex_bail!("IO source answered {target:?} with {}", result.kind());
        }
        self.consumer()?.set_io_result(request, result);
        Ok(())
    }
}

/// One admitted root: the session its descendants read through, and how many of them are live.
struct Root {
    io: Arc<dyn IoSource>,
    live: usize,
}

/// A set of root planners advanced by its caller on one thread at a time.
///
/// Dropping a run clears the IO session of every root still live, even when it ends in an error.
#[derive(Default)]
pub struct Run {
    roots: HashMap<RootId, Root>,
    next_root: u64,
    next_owner: u64,
    queue: VecDeque<Box<Work>>,
    parked: HashMap<IoOwnerId, Box<Work>>,
    /// Batches and finished roots not yet returned by [`advance`](Self::advance).
    events: VecDeque<Progress>,
    /// Visits left before the sources are polled for completions again.
    round: usize,
    step_limit: Option<usize>,
    steps: usize,
}

impl Drop for Run {
    fn drop(&mut self) {
        for root in self.roots.values() {
            root.io.clear();
        }
    }
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
    /// and clears it once they have all retired or the root is cancelled.
    pub fn admit(
        &mut self,
        root: Box<dyn PendingPlanner>,
        scope: WorkScope,
        io: Arc<dyn IoSource>,
    ) -> RootId {
        let id = RootId(self.next_root);
        self.next_root += 1;
        self.roots.insert(id, Root { io, live: 0 });
        self.spawn(id, scope, Item::Pending(root));
        id
    }

    /// Drops every live item descended from `root` and clears its session. Batches the root
    /// already produced are still returned, followed by no [`Progress::RootDone`].
    pub fn cancel(&mut self, root: RootId) {
        let Some(cancelled) = self.roots.remove(&root) else {
            return;
        };
        self.queue.retain(|work| work.root != root);
        self.parked.retain(|_, work| work.root != root);
        cancelled.io.clear();
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
            if self.round == 0 {
                // Every ready item has been visited once: take what the sources finished.
                self.take_completions()?;
                self.round = self.queue.len();
                if self.round == 0 {
                    return Ok(if self.parked.is_empty() {
                        Progress::Idle
                    } else {
                        Progress::Waiting
                    });
                }
            }
            self.round -= 1;
            let Some(work) = self.queue.pop_front() else {
                self.round = 0;
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

    /// Delivers a completion taken from the session of one of the run's roots.
    pub fn complete(&mut self, completion: Completion) -> VortexResult<()> {
        let Completion {
            owner,
            request,
            result,
        } = completion;
        let result = result?;
        if let Entry::Occupied(mut parked) = self.parked.entry(owner) {
            let work = parked.get_mut();
            work.deliver(request, result)?;
            // An item that still waits for other requests stays parked.
            if work.state() != State::Waiting {
                self.queue.push_back(parked.remove());
            } else if work.outstanding.is_empty() {
                vortex_bail!(
                    "work for scope {:?} waits with no outstanding fetch",
                    work.scope
                );
            }
            return Ok(());
        }
        // An item that published fetches and had CPU work left is still in the run queue.
        let Some(work) = self.queue.iter_mut().find(|work| work.id == owner) else {
            vortex_bail!("completion for unknown work {owner:?}");
        };
        work.deliver(request, result)
    }

    /// Waits for a completion from any root's session, registering `cx` to be woken by the next
    /// one, and delivers every completion that is ready.
    ///
    /// Call it after [`advance`](Self::advance) returned [`Progress::Waiting`].
    pub fn poll_completion(&mut self, cx: &mut Context<'_>) -> Poll<VortexResult<()>> {
        let sources: Vec<_> = self
            .roots
            .values()
            .map(|root| Arc::clone(&root.io))
            .collect();
        if sources.is_empty() {
            return Poll::Ready(Err(vortex_err!(
                "the run is waiting for IO with no live root"
            )));
        }
        let mut delivered = false;
        for io in sources {
            match io.poll_completion(cx) {
                Poll::Ready(Ok(completion)) => {
                    if let Err(error) = self.complete(completion) {
                        return Poll::Ready(Err(error));
                    }
                    delivered = true;
                }
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => {}
            }
        }
        if delivered {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    /// Delivers the completions every root's session has ready, without blocking.
    fn take_completions(&mut self) -> VortexResult<()> {
        if self.roots.len() == 1 {
            // The common case, without collecting the sessions.
            let Some(io) = self.roots.values().next().map(|root| Arc::clone(&root.io)) else {
                return Ok(());
            };
            while let Some(completion) = io.poll()? {
                self.complete(completion)?;
            }
            return Ok(());
        }
        let sources: Vec<_> = self
            .roots
            .values()
            .map(|root| Arc::clone(&root.io))
            .collect();
        for io in sources {
            while let Some(completion) = io.poll()? {
                self.complete(completion)?;
            }
        }
        Ok(())
    }

    fn spawn(&mut self, root: RootId, scope: WorkScope, item: Item) {
        let id = IoOwnerId(self.next_owner);
        self.next_owner += 1;
        if let Some(root) = self.roots.get_mut(&root) {
            root.live += 1;
        }
        self.queue.push_back(Box::new(Work {
            id,
            root,
            scope,
            item,
            outstanding: HashMap::default(),
        }));
    }

    /// Retires a finished item, and its root once nothing descended from it is live.
    fn retire(&mut self, work: &Work) {
        let Entry::Occupied(mut root) = self.roots.entry(work.root) else {
            return;
        };
        root.get().io.release(work.id);
        root.get_mut().live -= 1;
        if root.get().live == 0 {
            root.remove().io.clear();
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
            State::NeedsCompute => self.queue.push_back(work),
            State::Waiting => self.park(work)?,
        }
        Ok(())
    }

    /// Visits one item: starts it, or computes once if it is ready.
    fn visit(&mut self, mut work: Box<Work>) -> VortexResult<()> {
        let output = match &mut work.item {
            Item::Pending(_) => {
                let Item::Pending(pending) =
                    std::mem::replace(&mut work.item, Item::Live(Box::new(Finished)))
                else {
                    unreachable!("matched pending above");
                };
                work.item = Item::Live(pending.start()?);
                self.queue.push_back(work);
                return Ok(());
            }
            Item::Live(planner) => match planner.state() {
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
                self.queue.push_back(work);
            }
            Output::Planner(PlannerOutput::NeedsIO(batch))
            | Output::Morsel(MorselOutput::NeedsIO(batch)) => {
                self.submit(&mut work, batch)?;
                self.settle(work)?;
            }
            Output::Planner(PlannerOutput::Planner(scope, child)) => {
                self.spawn(work.root, scope, Item::Pending(child));
                self.queue.push_back(work);
            }
            Output::Planner(PlannerOutput::Morsel(scope, morsel)) => {
                self.spawn(work.root, scope, Item::Morsel(morsel));
                self.queue.push_back(work);
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
                self.queue.push_back(work);
            }
        }
        Ok(())
    }

    /// Registers the requests an item published with its root's session.
    fn submit(&mut self, work: &mut Work, batch: IoBatch) -> VortexResult<()> {
        if batch.is_empty() {
            vortex_bail!("compute() published an empty batch");
        }
        for request in &batch {
            if request.intent == IoIntent::Fetch
                && work
                    .outstanding
                    .insert(request.request, request.target)
                    .is_some()
            {
                vortex_bail!(
                    "fetch {:?} of {:?} was published while it is outstanding",
                    request.request,
                    work.id
                );
            }
        }
        let Some(root) = self.roots.get(&work.root) else {
            vortex_bail!("work {:?} outlived its root {:?}", work.id, work.root);
        };
        root.io.submit(work.id, batch)
    }
}

/// Drives one root to completion on the calling thread, blocking while it waits for IO.
pub struct Driver {
    io: Arc<dyn IoSource>,
    step_limit: Option<usize>,
}

impl Driver {
    /// Creates a driver that submits every request to `io`.
    pub fn new(io: Arc<dyn IoSource>) -> Self {
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
    pub fn run(&self, root: Box<dyn PendingPlanner>) -> VortexResult<Vec<Batch>> {
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
                Progress::Waiting => run.complete(self.io.wait()?)?,
            }
        }
    }
}

/// Placeholder while a pending item is being started.
struct Finished;

impl IoConsumer for Finished {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for Finished {
    fn state(&self) -> State {
        State::Done
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        Ok(PlannerOutput::Done)
    }
}

#[cfg(test)]
mod tests;
