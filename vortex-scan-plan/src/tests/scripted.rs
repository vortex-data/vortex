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

use crate::io::IoBatch;
use crate::io::IoConsumer;
use crate::io::IoRequestId;
use crate::io::IoResult;
use crate::io::IoSource;
use crate::io::IoTarget;
use crate::morsel::Morsel;
use crate::morsel::MorselOutput;
use crate::next::PendingPlanner;
use crate::next::pending;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::planner::WorkScope;

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
            variant(&result)
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
            variant(&result)
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

fn variant(result: &IoResult) -> &'static str {
    match result {
        IoResult::Size(_) => "size",
        IoResult::Bytes(_) => "bytes",
    }
}

enum Canned {
    Bytes(ByteBuffer),
    Fail(String),
}

/// An [`IoSource`] answering from canned buffers, recording every target performed.
#[derive(Default)]
pub struct RecordingIoSource {
    canned: Mutex<HashMap<IoTarget, Canned>>,
    performed: Mutex<Vec<IoTarget>>,
}

impl RecordingIoSource {
    /// Answers a four-byte range at `offset` with `bytes`.
    pub fn canned_bytes(&self, offset: u64, bytes: ByteBuffer) {
        let len = bytes.len();
        self.canned
            .lock()
            .insert(IoTarget::Range { offset, len }, Canned::Bytes(bytes));
    }

    pub fn fail(&self, target: IoTarget, message: &str) {
        self.canned
            .lock()
            .insert(target, Canned::Fail(message.to_string()));
    }

    pub fn performed(&self) -> Vec<IoTarget> {
        self.performed.lock().clone()
    }
}

impl IoSource for RecordingIoSource {
    fn perform(&self, target: &IoTarget) -> VortexResult<IoResult> {
        self.performed.lock().push(target.clone());
        match self.canned.lock().get(target) {
            Some(Canned::Bytes(bytes)) => {
                Ok(IoResult::Bytes(BufferHandle::new_host(bytes.clone())))
            }
            Some(Canned::Fail(message)) => Err(vortex_err!("{message}")),
            None => Err(vortex_err!("no canned answer for {target:?}")),
        }
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
