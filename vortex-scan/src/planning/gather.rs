// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A planner that runs a whole plan inside itself and hands its output to a continuation.
//!
//! Some planning needs data before it can decide what to read next: an index search reads the
//! index, and only then knows which rows of which data files to scan. [`GatherPlanner`] lets that
//! decision be a planner like any other. It runs an inner plan on a private [`Run`], publishes the
//! inner plan's reads as its own fetches, so whoever drives the outer plan does every read, and
//! gathers the inner plan's batches. When the inner plan finishes, a continuation turns the
//! batches into what the planner does next, such as opening the data files.

use std::fmt;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoOwnerId;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_utils::aliases::hash_map::HashMap;

use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// What a [`GatherPlanner`] does once its inner plan has finished, given the inner plan's batches
/// in scope order: nothing more, or one child.
///
/// A child that reads another byte source, such as a data file found through an index, is an
/// [`Open`](PlannerOutput::Open). Any other output is an error.
pub type Then = Box<dyn FnOnce(Vec<ArrayRef>) -> VortexResult<Option<PlannerOutput>> + Send>;

/// Runs an inner plan whose reads become this planner's fetches, then continues with [`Then`].
///
/// The planner reads the same byte source as its inner plan, so it must run over that source's
/// IO service: emit it as an [`Open`](PlannerOutput::Open) of the inner plan's service, or admit
/// it as a root of it.
pub struct GatherPlanner {
    scope: WorkScope,
    run: Run,
    /// Requests the inner plan has submitted and this planner has not yet published.
    outbox: Arc<Mutex<Vec<(IoOwnerId, IoBatch)>>>,
    /// The inner owner and request each published fetch answers.
    routes: HashMap<IoRequestId, (IoOwnerId, IoRequestId)>,
    next_request: u32,
    /// Answers delivered and not yet passed to the inner run.
    delivered: Vec<Completion>,
    batches: Vec<(WorkScope, ArrayRef)>,
    then: Option<Then>,
    /// The continuation has run, and its child, if any, was emitted.
    finished: bool,
}

impl GatherPlanner {
    /// Gathers what `root` produces over `scope`, then calls `then` with it.
    pub fn new(scope: WorkScope, root: Box<dyn Planner>, then: Then) -> Self {
        let outbox = Arc::default();
        let mut run = Run::new();
        run.admit(
            root,
            scope.clone(),
            Arc::new(Forward {
                outbox: Arc::clone(&outbox),
            }),
        );
        Self {
            scope,
            run,
            outbox,
            routes: HashMap::default(),
            next_request: 0,
            delivered: Vec::new(),
            batches: Vec::new(),
            then: Some(then),
            finished: false,
        }
    }

    /// The inner plan's submitted requests, under ids of this planner.
    fn publish(&mut self) -> IoBatch {
        let submitted = std::mem::take(&mut *self.outbox.lock());
        let mut batch = IoBatch::new();
        for (owner, requests) in submitted {
            for request in requests {
                let id = IoRequestId(self.next_request);
                self.next_request += 1;
                // Only fetches are delivered; hints are passed on for the service to use.
                if request.intent == IoIntent::Fetch {
                    self.routes.insert(id, (owner, request.request));
                }
                batch.push(IoRequest {
                    intent: request.intent,
                    request: id,
                    target: request.target,
                });
            }
        }
        batch
    }

    /// Runs the inner plan until it needs a read this planner has not published, or finishes.
    fn advance(&mut self) -> VortexResult<PlannerOutput> {
        for completion in self.delivered.drain(..) {
            self.run.complete(completion)?;
        }
        loop {
            match self.run.advance()? {
                Progress::Batch(batch) => self.batches.push((batch.scope, batch.array)),
                Progress::Spawn(spawned) => {
                    if spawned.io.is_some() {
                        vortex_bail!("a gathered plan cannot open another byte source");
                    }
                    self.run.admit(
                        spawned.planner,
                        spawned.scope,
                        Arc::new(Forward {
                            outbox: Arc::clone(&self.outbox),
                        }),
                    );
                }
                Progress::RootDone(_) => {}
                Progress::Waiting => {
                    let batch = self.publish();
                    if !batch.is_empty() {
                        return Ok(PlannerOutput::NeedsIO(batch));
                    }
                    if self.routes.is_empty() {
                        vortex_bail!("the gathered plan waits with no read outstanding");
                    }
                    // Every read the inner plan waits for is published: wait for its answer.
                    return Ok(PlannerOutput::Continue);
                }
                Progress::Idle => return self.finish(),
            }
        }
    }

    fn finish(&mut self) -> VortexResult<PlannerOutput> {
        self.finished = true;
        let then = self
            .then
            .take()
            .ok_or_else(|| vortex_err!("GatherPlanner finished twice"))?;
        // Inner roots run concurrently, so put their batches back in scope order. The sort is
        // stable: one morsel's batches keep the order it emitted them in.
        self.batches
            .sort_by_key(|(scope, _)| (scope.file_ordinal, scope.rows.start));
        let batches = std::mem::take(&mut self.batches)
            .into_iter()
            .map(|(_, array)| array)
            .collect();
        match then(batches)? {
            None => Ok(PlannerOutput::Done),
            Some(child @ (PlannerOutput::Open(..) | PlannerOutput::Planner(..))) => Ok(child),
            Some(other) => vortex_bail!("a gather continuation returned {other:?}"),
        }
    }
}

impl fmt::Debug for GatherPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatherPlanner")
            .field("scope", &self.scope)
            .field("outstanding", &self.routes.len())
            .field("batches", &self.batches.len())
            .finish()
    }
}

impl IoConsumer for GatherPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        let Some((owner, request)) = self.routes.remove(&request) else {
            vortex_panic!("GatherPlanner: delivery of {request:?}, which is not outstanding");
        };
        self.delivered.push(Completion {
            owner,
            request,
            result: Ok(result),
        });
    }
}

impl Planner for GatherPlanner {
    fn state(&self) -> State {
        if self.finished {
            State::Done
        } else if self.delivered.is_empty()
            && !self.routes.is_empty()
            && self.outbox.lock().is_empty()
        {
            State::Waiting
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.advance()
    }
}

/// The inner run's IO session: it only collects what the inner plan submits, since answers come
/// back through [`GatherPlanner::set_io_result`] and are passed to the run directly.
struct Forward {
    outbox: Arc<Mutex<Vec<(IoOwnerId, IoBatch)>>>,
}

impl IoSource for Forward {
    fn submit(&self, owner: IoOwnerId, batch: IoBatch) -> VortexResult<()> {
        self.outbox.lock().push((owner, batch));
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn poll_completion(&self, _cx: &mut Context<'_>) -> Poll<VortexResult<Completion>> {
        Poll::Ready(Err(vortex_err!(
            "a gathered plan's reads are answered by its planner, not polled"
        )))
    }

    fn wait(&self) -> VortexResult<Completion> {
        vortex_bail!("a gathered plan's reads are answered by its planner, not awaited")
    }

    fn release(&self, owner: IoOwnerId) {
        self.outbox.lock().retain(|(o, _)| *o != owner);
    }

    fn clear(&self) {
        self.outbox.lock().clear();
    }
}
