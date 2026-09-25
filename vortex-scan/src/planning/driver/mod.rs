// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A single-worker blocking driver for the protocol.
//!
//! Runnable items sit in a FIFO run queue, taking at most one CPU step per visit. Published
//! IO is registered immediately and state is inspected again before parking. Output follows
//! completion order; FIFO scheduling does not imply row ordering. Waiting work is parked, and
//! comes back to the run queue when one of its requests completes, in whatever order the source
//! finishes them. The driver only blocks when the run queue is empty and something is parked.

use std::collections::VecDeque;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::request::IoOwnerId;
use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::next::PendingPlanner;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// One array produced by a morsel, with the scope the morsel was created for.
pub struct Batch {
    /// Scope of the morsel that produced the array.
    pub scope: WorkScope,
    /// The non-empty array.
    pub array: ArrayRef,
}

/// Drives a tree of planners and morsels to completion on the calling thread.
pub struct Driver {
    io: Arc<dyn IoSource>,
    step_limit: Option<usize>,
}

enum Item {
    Pending(Box<dyn PendingPlanner>),
    Live(Box<dyn Planner>),
    Morsel(Box<dyn Morsel>),
}

struct Work {
    id: IoOwnerId,
    scope: WorkScope,
    item: Item,
    /// Submitted requests not yet completed, with the target each must be answered with.
    outstanding: HashMap<IoRequestId, IoTarget>,
    registered: HashMap<IoRequestId, IoRequest>,
    /// Ids delivered since the item's last `state()`; any of them re-listed is a protocol error.
    delivered: Vec<IoRequestId>,
}

/// A run owns service registrations even when it exits through an error.
struct RunIo<'a>(&'a dyn IoSource);

impl Drop for RunIo<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

impl Work {
    fn new(id: IoOwnerId, scope: WorkScope, item: Item) -> Self {
        Self {
            id,
            scope,
            item,
            registered: HashMap::default(),
            outstanding: HashMap::default(),
            delivered: Vec::new(),
        }
    }

    fn consumer(&mut self) -> VortexResult<&mut dyn IoConsumer> {
        match &mut self.item {
            Item::Live(planner) => Ok(&mut **planner),
            Item::Morsel(morsel) => Ok(&mut **morsel),
            Item::Pending(_) => vortex_bail!("pending work cannot receive IO"),
        }
    }
}

/// What to do with an item after one visit.
enum After {
    Drop,
    Requeue,
    Park,
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
        let _run = RunIo(self.io.as_ref());
        let mut next_id = 0u64;
        let mut fresh = || {
            next_id += 1;
            IoOwnerId(next_id)
        };
        let mut queue = VecDeque::new();
        queue.push_back(Work::new(
            fresh(),
            WorkScope {
                file_ordinal: 0,
                rows: 0..0,
            },
            Item::Pending(root),
        ));
        let mut parked: HashMap<IoOwnerId, Work> = HashMap::default();
        let mut output = Vec::new();
        let mut steps = 0usize;

        loop {
            // Visit each ready item once before checking for IO completions.
            let visits = queue.len();
            for _ in 0..visits {
                let Some(mut work) = queue.pop_front() else {
                    break;
                };
                steps += 1;
                if let Some(limit) = self.step_limit
                    && steps > limit
                {
                    vortex_bail!("driver exceeded its step limit of {limit}");
                }
                match self.visit(&mut work, &mut fresh, &mut queue, &mut output)? {
                    After::Drop => self.io.release(work.id),
                    After::Requeue => queue.push_back(work),
                    After::Park => {
                        parked.insert(work.id, work);
                    }
                }
            }
            if let Some(completion) = self.io.poll()? {
                Self::complete(completion, &mut queue, &mut parked)?;
                continue;
            }
            if !queue.is_empty() {
                continue;
            }
            if parked.is_empty() {
                return Ok(output);
            }
            Self::complete(self.io.wait()?, &mut queue, &mut parked)?;
        }
    }

    fn complete(
        completion: Completion,
        queue: &mut VecDeque<Work>,
        parked: &mut HashMap<IoOwnerId, Work>,
    ) -> VortexResult<()> {
        let Completion {
            owner: id,
            request,
            result,
        } = completion;
        let result = result?;
        let work = parked.remove(&id).or_else(|| {
            let index = queue.iter().position(|work| work.id == id)?;
            queue.remove(index)
        });
        let Some(mut work) = work else {
            vortex_bail!("completion for unknown work {id:?}");
        };
        let Some(outstanding) = work.outstanding.remove(&request) else {
            vortex_bail!("completion for {request:?} of {id:?}, which is not outstanding");
        };
        if !result.matches(&outstanding) {
            vortex_bail!(
                "IO source answered {:?} with {}",
                outstanding,
                result.kind()
            );
        }
        work.consumer()?.set_io_result(request, result);
        work.delivered.push(request);
        queue.push_back(work);
        Ok(())
    }

    /// Visits one item: starts it, submits its batch, or computes once.
    fn visit(
        &self,
        work: &mut Work,
        fresh: &mut impl FnMut() -> IoOwnerId,
        queue: &mut VecDeque<Work>,
        output: &mut Vec<Batch>,
    ) -> VortexResult<After> {
        let state = match &mut work.item {
            Item::Pending(_) => {
                let Item::Pending(pending) =
                    std::mem::replace(&mut work.item, Item::Live(Box::new(Finished)))
                else {
                    unreachable!("matched pending above");
                };
                work.item = Item::Live(pending.start()?);
                return Ok(After::Requeue);
            }
            Item::Live(planner) => planner.state(),
            Item::Morsel(morsel) => morsel.state(),
        };
        match state {
            State::Done => Ok(After::Drop),
            State::NeedsIO(batch) => {
                if batch.iter().any(|request| request.intent != IoIntent::Fetch) {
                    vortex_bail!("State::NeedsIO can only wait for Fetch requests");
                }
                self.submit(work, batch)?;
                Ok(After::Park)
            }
            State::NeedsCompute => {
                work.delivered.clear();
                match &mut work.item {
                    Item::Live(planner) => match planner.compute()? {
                        PlannerOutput::Done => Ok(After::Drop),
                        PlannerOutput::Continue => Ok(After::Requeue),
                        PlannerOutput::NeedsIO(batch) => {
                            self.submit(work, batch)?;
                            self.after_publication(work)
                        }
                        PlannerOutput::Planner(scope, child) => {
                            queue.push_back(Work::new(fresh(), scope, Item::Pending(child)));
                            Ok(After::Requeue)
                        }
                        PlannerOutput::Morsel(scope, morsel) => {
                            queue.push_back(Work::new(fresh(), scope, Item::Morsel(morsel)));
                            Ok(After::Requeue)
                        }
                    },
                    Item::Morsel(morsel) => match morsel.compute()? {
                        MorselOutput::Done => Ok(After::Drop),
                        MorselOutput::Continue => Ok(After::Requeue),
                        MorselOutput::NeedsIO(batch) => {
                            self.submit(work, batch)?;
                            self.after_publication(work)
                        }
                        MorselOutput::Batch(array) => {
                            if array.is_empty() {
                                vortex_bail!(
                                    "morsel for scope {:?} returned an empty batch",
                                    work.scope
                                );
                            }
                            output.push(Batch {
                                scope: work.scope.clone(),
                                array,
                            });
                            Ok(After::Requeue)
                        }
                    },
                    Item::Pending(_) => unreachable!("pending work was started above"),
                }
            }
        }
    }

    fn after_publication(&self, work: &mut Work) -> VortexResult<After> {
        let state = match &work.item {
            Item::Live(planner) => planner.state(),
            Item::Morsel(morsel) => morsel.state(),
            Item::Pending(_) => unreachable!("only live work publishes IO"),
        };
        match state {
            State::Done => Ok(After::Drop),
            State::NeedsCompute => Ok(After::Requeue),
            State::NeedsIO(batch) => {
                if batch.iter().any(|request| request.intent != IoIntent::Fetch) {
                    vortex_bail!("State::NeedsIO can only wait for Fetch requests");
                }
                self.submit(work, batch)?;
                Ok(After::Park)
            }
        }
    }

    /// Submits the requests in `batch` that are not already outstanding.
    fn submit(&self, work: &mut Work, batch: IoBatch) -> VortexResult<()> {
        if batch.is_empty() {
            vortex_bail!("state() reported NeedsIO with an empty batch");
        }
        if let Some(stale) = batch
            .iter()
            .find(|request| work.delivered.contains(&request.request))
        {
            vortex_bail!(
                "request {:?} was delivered on the previous visit but is listed again",
                stale.request
            );
        }
        work.delivered.clear();
        let mut submissions = Vec::new();
        for request in batch {
            if let Some(previous) = work.registered.get(&request.request) {
                if previous.target != request.target {
                    vortex_bail!("changed target for {:?} of {:?}", request.request, work.id);
                }
                if previous.intent == IoIntent::Fetch {
                    if !work.outstanding.contains_key(&request.request) {
                        vortex_bail!("request {:?} was already delivered", request.request);
                    }
                    continue;
                }
                if previous.intent == request.intent
                    || (previous.intent == IoIntent::Prefetch && request.intent == IoIntent::Announce)
                {
                    continue;
                }
            }
            if request.intent == IoIntent::Fetch {
                work.outstanding
                    .insert(request.request, request.target.clone());
            }
            work.registered.insert(request.request, request.clone());
            submissions.push(request);
        }
        if !submissions.is_empty() {
            self.io.submit(work.id, submissions)?;
        }
        Ok(())
    }
}

/// Placeholder while a pending item is being started.
struct Finished;

impl IoConsumer for Finished {
    fn set_io_result(&mut self, _request: IoRequestId, _result: vortex_io::request::IoResult) {}
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
