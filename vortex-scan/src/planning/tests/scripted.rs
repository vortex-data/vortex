// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scripted planners, morsels, and a recording IO session for driver tests.
//!
//! A scripted object follows its script exactly and panics when the driver calls it against the
//! protocol, so a driver bug becomes a test failure that names the script position.

use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

use futures::FutureExt;
use futures::channel::oneshot;
use futures::future;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::segments::IoHandle;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoKey;
use vortex_io::segments::IoOwnerId;
use vortex_io::segments::IoRequest;
use vortex_io::segments::SegmentIoSession;
use vortex_io::segments::SegmentSourceId;
use vortex_io::segments::SourceRequest;
use vortex_io::segments::SourceResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// Shared record of the calls the driver made, as readable one-line events.
#[derive(Clone, Default)]
pub struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    fn push(&self, event: String) {
        self.0.lock().push(event);
    }

    pub fn events(&self) -> Vec<String> {
        self.0.lock().clone()
    }
}

/// One scripted planner step. Every step but `Await` and `Wait` is the output of one `compute()`.
pub enum PlannerStep {
    Done,
    Continue,
    /// Publishes the batch and goes straight on to the next step.
    NeedsIO(Vec<IoRequest>),
    Planner(WorkScope, Vec<PlannerStep>),
    Morsel(WorkScope, Vec<MorselStep>),
    /// Publishes the batch, then waits until every request in it is delivered.
    Io(Vec<IoRequest>),
    /// Waits for fetches an earlier `NeedsIO` step published.
    Await(Vec<IoKey>),
    /// Reports `Waiting` without having published anything.
    Wait,
}

/// One scripted morsel step.
pub enum MorselStep {
    Compute(MorselOutput),
    /// Publishes the batch, then waits until every request in it is delivered.
    Io(Vec<IoRequest>),
}

/// The keys a waiting step still expects.
struct Pending(Vec<IoKey>);

impl Pending {
    fn of(batch: &[IoRequest]) -> Self {
        Self(batch.iter().map(|request| request.key).collect())
    }

    /// Removes a delivered key and returns whether the step has everything it waited for.
    fn deliver(&mut self, name: &str, position: usize, key: IoKey) -> bool {
        let index = self
            .0
            .iter()
            .position(|pending| *pending == key)
            .unwrap_or_else(|| {
                panic!("{name}: delivery of unknown segment {key:?} at step {position}")
            });
        self.0.remove(index);
        self.0.is_empty()
    }
}

pub struct ScriptedPlanner {
    name: &'static str,
    script: Vec<PlannerStep>,
    position: usize,
    /// What the step at `position` waits for, once it has published or received something.
    pending: Option<Pending>,
    log: Log,
}

impl ScriptedPlanner {
    pub fn new(name: &'static str, script: Vec<PlannerStep>, log: &Log) -> Self {
        Self {
            name,
            script,
            position: 0,
            pending: None,
            log: log.clone(),
        }
    }

    /// A fresh scripted planner, boxed as the driver takes it.
    pub fn boxed(name: &'static str, script: Vec<PlannerStep>, log: &Log) -> Box<dyn Planner> {
        Box::new(Self::new(name, script, log))
    }
}

impl ScriptedPlanner {
    fn take_delivery(&mut self, key: IoKey) {
        self.log
            .push(format!("deliver {} {}", self.name, describe(key)));
        let position = self.position;
        if let Some(PlannerStep::Await(keys)) = self.script.get(position)
            && self.pending.is_none()
        {
            self.pending = Some(Pending(keys.clone()));
        }
        let Some(pending) = self.pending.as_mut() else {
            panic!(
                "{}: unexpected delivery at script step {position}",
                self.name
            );
        };
        if pending.deliver(self.name, position, key) {
            self.pending = None;
            self.position += 1;
        }
    }
}

impl Planner for ScriptedPlanner {
    fn deliver(&mut self, key: IoKey, _result: SourceResult) {
        self.take_delivery(key);
    }

    fn state(&self) -> State {
        match self.script.get(self.position) {
            None => State::Done,
            Some(PlannerStep::Io(_)) if self.pending.is_none() => State::NeedsCompute,
            Some(PlannerStep::Io(_) | PlannerStep::Await(_) | PlannerStep::Wait) => State::Waiting,
            Some(_) => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let position = self.position;
        if let Some(PlannerStep::Io(batch)) = self.script.get(position) {
            assert!(
                self.pending.is_none(),
                "{}: compute called while script step {position} waits for IO",
                self.name
            );
            self.log.push(format!("publish {}", self.name));
            // A batch with nothing to wait for is left for the driver to reject.
            self.pending = Some(Pending::of(batch));
            return Ok(PlannerOutput::NeedsIO(batch.clone()));
        }
        self.log.push(format!("compute {}", self.name));
        self.position += 1;
        let Some(slot) = self.script.get_mut(position) else {
            panic!("{}: compute past the end of the script", self.name);
        };
        let step = std::mem::replace(slot, PlannerStep::Done);
        Ok(match step {
            PlannerStep::Done => PlannerOutput::Done,
            PlannerStep::Continue => PlannerOutput::Continue,
            PlannerStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
            PlannerStep::Planner(scope, script) => {
                PlannerOutput::Planner(scope, Self::boxed("child", script, &self.log))
            }
            PlannerStep::Morsel(scope, script) => PlannerOutput::Morsel(
                scope,
                Box::new(ScriptedMorsel::new("morsel", script, &self.log)),
            ),
            PlannerStep::Io(_) | PlannerStep::Await(_) | PlannerStep::Wait => panic!(
                "{}: compute called while script step {position} waits for IO",
                self.name
            ),
        })
    }
}

pub struct ScriptedMorsel {
    name: &'static str,
    script: Vec<MorselStep>,
    position: usize,
    pending: Option<Pending>,
    log: Log,
}

impl ScriptedMorsel {
    pub fn new(name: &'static str, script: Vec<MorselStep>, log: &Log) -> Self {
        Self {
            name,
            script,
            position: 0,
            pending: None,
            log: log.clone(),
        }
    }
}

impl Morsel for ScriptedMorsel {
    fn deliver(&mut self, key: IoKey, _result: SourceResult) {
        self.log
            .push(format!("deliver {} {}", self.name, describe(key)));
        let position = self.position;
        let Some(pending) = self.pending.as_mut() else {
            panic!(
                "{}: unexpected delivery at script step {position}",
                self.name
            );
        };
        if pending.deliver(self.name, position, key) {
            self.pending = None;
            self.position += 1;
        }
    }

    fn state(&self) -> State {
        match self.script.get(self.position) {
            None => State::Done,
            Some(MorselStep::Io(_)) if self.pending.is_some() => State::Waiting,
            Some(_) => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        let position = self.position;
        match self.script.get_mut(position) {
            Some(MorselStep::Io(batch)) => {
                assert!(
                    self.pending.is_none(),
                    "{}: compute called while script step {position} waits for IO",
                    self.name
                );
                let batch = batch.clone();
                self.log.push(format!("publish {}", self.name));
                self.pending = Some(Pending::of(&batch));
                Ok(MorselOutput::NeedsIO(batch))
            }
            Some(MorselStep::Compute(output)) => {
                self.log.push(format!("compute {}", self.name));
                self.position += 1;
                Ok(std::mem::replace(output, MorselOutput::Done))
            }
            None => panic!("{}: compute past the end of the script", self.name),
        }
    }
}

/// How the log names what `key` reads: a segment's id, or `size`.
fn describe(key: IoKey) -> String {
    match key.request {
        SourceRequest::Segment(segment) => segment.to_string(),
        SourceRequest::Size => "size".to_string(),
    }
}

/// The key of `segment` in the one source the driver tests read.
pub fn key(segment: u32) -> IoKey {
    IoKey::segment(SegmentSourceId(0), segment)
}

/// A fetch of `segment`.
pub fn fetch(segment: u32) -> IoRequest {
    IoRequest::new(key(segment), IoIntent::Fetch)
}

type FetchResult = Result<SourceResult, Arc<VortexError>>;

/// Which unfinished fetch the session finishes next.
#[derive(Clone, Copy, Default)]
pub enum Order {
    /// Oldest submission first.
    #[default]
    Fifo,
    /// Newest submission first, so a batch and concurrently parked items complete in reverse.
    Lifo,
}

/// A fetch the session has not finished yet.
struct Unfinished {
    owner: IoOwnerId,
    send: oneshot::Sender<FetchResult>,
    result: FetchResult,
}

/// A [`SegmentIoSession`] answering fetches from canned buffers.
///
/// Fetches finish one per [`poll_ready`](SegmentIoSession::poll_ready), in the configured
/// [`Order`]. A gated session finishes one only after each [`arrive`](Self::arrive), so a test
/// decides when IO lands. Announcements and prefetches never finish. It records every submission
/// and every fetch performed.
#[derive(Default)]
pub struct RecordingSession {
    canned: Mutex<HashMap<IoKey, Result<ByteBuffer, String>>>,
    submissions: Mutex<Vec<(IoOwnerId, IoRequest)>>,
    performed: Mutex<Vec<IoKey>>,
    unfinished: Mutex<Vec<Unfinished>>,
    order: Order,
    gated: bool,
    arrivals: Mutex<usize>,
    waker: Mutex<Option<Waker>>,
}

impl RecordingSession {
    /// Answers segments 0 and 1 with four bytes each.
    pub fn new(order: Order, gated: bool) -> Self {
        let session = Self {
            order,
            gated,
            ..Default::default()
        };
        session.canned_bytes(0, ByteBuffer::from(vec![0u8, 1, 2, 3]));
        session.canned_bytes(1, ByteBuffer::from(vec![4u8, 5, 6, 7]));
        session
    }

    /// Answers `segment` with `bytes`.
    pub fn canned_bytes(&self, segment: u32, bytes: ByteBuffer) {
        self.canned.lock().insert(key(segment), Ok(bytes));
    }

    /// Fails reads of `segment` with `message`.
    pub fn fail(&self, segment: u32, message: &str) {
        self.canned
            .lock()
            .insert(key(segment), Err(message.to_string()));
    }

    /// Lets a gated session finish one more fetch.
    pub fn arrive(&self) {
        *self.arrivals.lock() += 1;
        if let Some(waker) = self.waker.lock().take() {
            waker.wake();
        }
    }

    pub fn submissions(&self) -> Vec<(IoOwnerId, IoRequest)> {
        self.submissions.lock().clone()
    }

    pub fn performed(&self) -> Vec<IoKey> {
        self.performed.lock().clone()
    }
}

impl SegmentIoSession for RecordingSession {
    fn submit(&self, owner: IoOwnerId, batch: &[IoRequest]) -> VortexResult<Vec<IoHandle>> {
        Ok(batch
            .iter()
            .map(|request| {
                self.submissions.lock().push((owner, *request));
                if request.intent != IoIntent::Fetch {
                    return IoHandle::new(future::pending().boxed().shared(), Arc::new(()));
                }
                self.performed.lock().push(request.key);
                let result = match self.canned.lock().get(&request.key) {
                    Some(Ok(bytes)) => {
                        Ok(SourceResult::Bytes(BufferHandle::new_host(bytes.clone())))
                    }
                    Some(Err(message)) => Err(Arc::new(vortex_err!("{message}"))),
                    None => Err(Arc::new(vortex_err!(
                        "no canned answer for {:?}",
                        request.key
                    ))),
                };
                let (send, receive) = oneshot::channel();
                self.unfinished.lock().push(Unfinished {
                    owner,
                    send,
                    result,
                });
                let bytes = receive
                    .map(|result| {
                        result.unwrap_or_else(|_| Err(Arc::new(vortex_err!("session dropped"))))
                    })
                    .boxed()
                    .shared();
                IoHandle::new(bytes, Arc::new(()))
            })
            .collect())
    }

    fn poll_ready(
        &self,
        cx: &mut Context<'_>,
        ready: &mut Vec<IoOwnerId>,
    ) -> Poll<VortexResult<()>> {
        let mut unfinished = self.unfinished.lock();
        if unfinished.is_empty() {
            return Poll::Ready(Err(vortex_err!("RecordingSession: no fetch in flight")));
        }
        if self.gated {
            let mut arrivals = self.arrivals.lock();
            if *arrivals == 0 {
                *self.waker.lock() = Some(cx.waker().clone());
                return Poll::Pending;
            }
            *arrivals -= 1;
        }
        let next = match self.order {
            Order::Fifo => unfinished.remove(0),
            Order::Lifo => unfinished.pop().expect("checked non-empty"),
        };
        // A dropped receiver means its reader was cancelled, which is the reader's concern.
        drop(next.send.send(next.result));
        ready.push(next.owner);
        Poll::Ready(Ok(()))
    }
}

/// A small non-nullable `u32` array of `n` rows.
pub fn array_of(n: u64) -> ArrayRef {
    PrimitiveArray::from_iter(0..u32::try_from(n).expect("small test array")).into_array()
}

/// An execution context over a bare array session, for `assert_arrays_eq!`.
pub fn ctx() -> ExecutionCtx {
    array_session().create_execution_ctx()
}

pub fn scope(rows: Range<u64>) -> WorkScope {
    WorkScope {
        file_ordinal: 0,
        rows,
    }
}
