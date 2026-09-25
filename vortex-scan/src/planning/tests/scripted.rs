// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scripted planners, morsels, and a recording IO source for driver tests.
//!
//! A scripted object follows its script exactly and panics when the driver calls it against the
//! protocol, so a driver bug becomes a test failure that names the script position.

use std::ops::Range;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use vortex_io::request::Completion;
use vortex_io::request::IoBatch;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequest;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::request::IoSource;
use vortex_io::request::IoTarget;
use vortex_io::request::IoOwnerId;
use crate::planning::morsel::Morsel;
use crate::planning::morsel::MorselOutput;
use crate::planning::next::PendingPlanner;
use crate::planning::next::pending;
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

/// One scripted planner step. `Io` is a batch to report from `state()`; everything else is the
/// output of one `compute()`.
pub enum PlannerStep {
    Done,
    Continue,
    NeedsIO(IoBatch),
    Planner(WorkScope, Vec<PlannerStep>),
    Morsel(WorkScope, Vec<MorselStep>),
    Io(IoBatch),
}

/// One scripted morsel step.
pub enum MorselStep {
    Compute(MorselOutput),
    Io(IoBatch),
}

pub struct ScriptedPlanner {
    name: &'static str,
    script: Vec<PlannerStep>,
    position: usize,
    log: Log,
}

impl ScriptedPlanner {
    pub fn new(name: &'static str, script: Vec<PlannerStep>, log: &Log) -> Self {
        Self {
            name,
            script,
            position: 0,
            log: log.clone(),
        }
    }

    /// Wraps a fresh scripted planner as pending work that logs its start.
    pub fn pending(
        name: &'static str,
        script: Vec<PlannerStep>,
        log: &Log,
    ) -> Box<dyn PendingPlanner> {
        let planner = Self::new(name, script, log);
        pending(move || {
            planner.log.push(format!("start {name}"));
            Ok(Box::new(planner) as Box<dyn Planner>)
        })
    }

    fn head(&self) -> Option<&PlannerStep> {
        self.script.get(self.position)
    }
}

impl IoConsumer for ScriptedPlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.log.push(format!(
            "deliver {} {} {}",
            self.name,
            request.0,
            result.kind()
        ));
        let position = self.position;
        let Some(PlannerStep::Io(batch)) = self.script.get_mut(position) else {
            panic!(
                "{}: unexpected delivery at script step {position}",
                self.name
            );
        };
        let index = batch
            .iter()
            .position(|pending| pending.request == request)
            .unwrap_or_else(|| {
                panic!(
                    "{}: delivery of unknown request {request:?} at step {position}",
                    self.name
                )
            });
        batch.remove(index);
        if batch.is_empty() {
            self.position += 1;
        }
    }
}

impl Planner for ScriptedPlanner {
    fn state(&self) -> State {
        match self.head() {
            None => State::Done,
            Some(PlannerStep::Io(batch)) => State::NeedsIO(batch.clone()),
            Some(_) => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.log.push(format!("compute {}", self.name));
        let position = self.position;
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
                PlannerOutput::Planner(scope, Self::pending("child", script, &self.log))
            }
            PlannerStep::Morsel(scope, script) => PlannerOutput::Morsel(
                scope,
                Box::new(ScriptedMorsel::new("morsel", script, &self.log)),
            ),
            PlannerStep::Io(_) => panic!(
                "{}: compute called while script step {position} needs IO",
                self.name
            ),
        })
    }
}

pub struct ScriptedMorsel {
    name: &'static str,
    script: Vec<MorselStep>,
    position: usize,
    log: Log,
}

impl ScriptedMorsel {
    pub fn new(name: &'static str, script: Vec<MorselStep>, log: &Log) -> Self {
        Self {
            name,
            script,
            position: 0,
            log: log.clone(),
        }
    }
}

impl IoConsumer for ScriptedMorsel {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.log.push(format!(
            "deliver {} {} {}",
            self.name,
            request.0,
            result.kind()
        ));
        let position = self.position;
        let Some(MorselStep::Io(batch)) = self.script.get_mut(position) else {
            panic!(
                "{}: unexpected delivery at script step {position}",
                self.name
            );
        };
        let index = batch
            .iter()
            .position(|pending| pending.request == request)
            .unwrap_or_else(|| {
                panic!(
                    "{}: delivery of unknown request {request:?} at step {position}",
                    self.name
                )
            });
        batch.remove(index);
        if batch.is_empty() {
            self.position += 1;
        }
    }
}

impl Morsel for ScriptedMorsel {
    fn state(&self) -> State {
        match self.script.get(self.position) {
            None => State::Done,
            Some(MorselStep::Io(batch)) => State::NeedsIO(batch.clone()),
            Some(MorselStep::Compute(_)) => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        self.log.push(format!("compute {}", self.name));
        let position = self.position;
        self.position += 1;
        match self.script.get_mut(position) {
            Some(MorselStep::Compute(output)) => Ok(std::mem::replace(output, MorselOutput::Done)),
            Some(MorselStep::Io(_)) => panic!(
                "{}: compute called while script step {position} needs IO",
                self.name
            ),
            None => panic!("{}: compute past the end of the script", self.name),
        }
    }
}

enum Canned {
    Bytes(ByteBuffer),
    Size(u64),
    Fail(String),
}

/// Which submitted request `wait` completes first.
#[derive(Clone, Copy, Default)]
pub enum Order {
    /// Oldest submission first.
    #[default]
    Fifo,
    /// Newest submission first, so a batch and concurrently parked items complete in reverse.
    Lifo,
}

/// An [`IoSource`] answering from canned buffers. Every submission is resolved at once and
/// held until `wait`, which hands completions back in the configured [`Order`]. It records the
/// targets in submission order.
#[derive(Default)]
pub struct RecordingIoSource {
    canned: Mutex<HashMap<IoTarget, Canned>>,
    performed: Mutex<Vec<IoTarget>>,
    ready: Mutex<Vec<Completion>>,
    order: Order,
    submissions: Mutex<Vec<(IoOwnerId, IoRequest)>>,
}

impl RecordingIoSource {
    pub fn lifo() -> Self {
        Self {
            order: Order::Lifo,
            ..Default::default()
        }
    }

    /// Answers a range at `offset` with `bytes`.
    pub fn canned_bytes(&self, offset: u64, bytes: ByteBuffer) {
        let len = bytes.len();
        self.canned
            .lock()
            .insert(IoTarget::Range { offset, len }, Canned::Bytes(bytes));
    }

    /// Answers `target` with a size regardless of what it asked for, to test the driver's check.
    pub fn answer_with_size(&self, target: IoTarget, size: u64) {
        self.canned.lock().insert(target, Canned::Size(size));
    }

    pub fn fail(&self, target: IoTarget, message: &str) {
        self.canned
            .lock()
            .insert(target, Canned::Fail(message.to_string()));
    }

    pub fn submissions(&self) -> Vec<(IoOwnerId, IoRequest)> {
        self.submissions.lock().clone()
    }

    pub fn performed(&self) -> Vec<IoTarget> {
        self.performed.lock().clone()
    }
}

impl IoSource for RecordingIoSource {
    fn submit(&self, work: IoOwnerId, batch: Vec<IoRequest>) -> VortexResult<()> {
        for request in batch {
            self.submissions.lock().push((work, request.clone()));
            if request.intent != IoIntent::Fetch {
                continue;
            }
            self.performed.lock().push(request.target.clone());
            let result = match self.canned.lock().get(&request.target) {
                Some(Canned::Bytes(bytes)) => {
                    Ok(IoResult::Bytes(BufferHandle::new_host(bytes.clone())))
                }
                Some(Canned::Size(size)) => Ok(IoResult::Size(*size)),
                Some(Canned::Fail(message)) => Err(vortex_err!("{message}")),
                None => Err(vortex_err!("no canned answer for {:?}", request.target)),
            };
            self.ready.lock().push(Completion {
                owner: work,
                request: request.request,
                result,
            });
        }
        Ok(())
    }

    fn poll(&self) -> VortexResult<Option<Completion>> {
        Ok(None)
    }

    fn release(&self, work: IoOwnerId) {
        self.ready.lock().retain(|completion| completion.owner != work);
    }

    fn clear(&self) {
        self.ready.lock().clear();
    }

    fn wait(&self) -> VortexResult<Completion> {
        let mut ready = self.ready.lock();
        let completion = match self.order {
            Order::Fifo if !ready.is_empty() => Some(ready.remove(0)),
            Order::Fifo => None,
            Order::Lifo => ready.pop(),
        };
        completion.ok_or_else(|| vortex_err!("RecordingIoSource: wait with nothing outstanding"))
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
