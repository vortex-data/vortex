// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Contracts and execution context for pipeline sources and operators.

use std::collections::VecDeque;

use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::pipeline::Arena;
use super::pipeline::Inlet;
use super::pipeline::PortId;
use crate::segments::SegmentId;

/// The output of one non-blocking compute call.
#[derive(Debug)]
pub enum Step {
    /// No output remains for this input. A source has finished entirely.
    End,
    /// A batch with more computation available without another input or readiness check.
    More(ArrayRef),
    /// The last batch for this input. A stage needs another input; a source needs readiness
    /// checked again.
    Final(ArrayRef),
}

/// Whether a source can compute or needs the scheduler to wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceReady {
    /// The next compute call can produce output or finish without blocking.
    Ready,
    /// Nothing can happen until the condition clears.
    Blocked(SourceBlocked),
}

/// Why a source cannot compute yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceBlocked {
    /// Wait for bytes from newly queued or outstanding reads.
    Io,
    /// The inlet needs more batches or its sink to close it. It must be open and have room
    /// for its sink to push; debug builds check that the sink can still wake this source.
    Inlet(usize),
}

/// The start of a pipeline: produces batches from its inlets and the segments it asks to have
/// read.
pub trait Source: Operator {
    /// Prepares the next [`compute`](Operator::compute) call, after the driver checks outlets.
    /// Returning [`SourceReady::Ready`] guarantees that call can proceed without blocking.
    /// Any input or choice made here can be retained by the source for that call.
    ///
    /// Queue reads with [`Cx::request_read`] and return
    /// `SourceReady::Blocked(SourceBlocked::Io)`. The driver issues newly queued reads
    /// before waiting. Readiness is checked again when bytes are available, including bytes
    /// already delivered; they are available through [`Cx::take_segment`] in arrival order.
    fn ready(&mut self, _cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        Ok(SourceReady::Ready)
    }
}

/// What an operator is handed by the stage below it.
#[derive(Debug)]
pub enum Input {
    /// No new input: a source is ready, or an operator is continuing after [`Step::More`].
    None,
    /// The next batch.
    Batch(ArrayRef),
    /// The stage below has finished: no more batches come. Handed over once; [`Input::None`]
    /// may follow to drain any [`Step::More`] outputs.
    End,
}

/// Computes batches without blocking, either as a source or as a stage transforming input.
pub trait Operator: Send {
    /// Computes one output. [`Step::More`] requests another call with [`Input::None`] before
    /// any new input is supplied. [`Step::End`] and [`Step::Final`] finish the current input;
    /// after [`Input::End`], they finish the stage as well.
    ///
    /// Sources receive only [`Input::None`]. Their first call and each call after
    /// [`Step::Final`] follow [`Source::ready`]; [`Step::End`] finishes the source.
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step>;
}

/// Distinct inlet indices, with constant-time insertion and removal.
#[derive(Default)]
pub(super) struct InletSet {
    indices: Vec<usize>,
    positions: Vec<usize>,
}

impl InletSet {
    pub(super) fn new(inlets: usize) -> Self {
        Self {
            indices: Vec::with_capacity(inlets),
            positions: vec![usize::MAX; inlets],
        }
    }

    pub(super) fn all(inlets: usize) -> Self {
        Self {
            indices: (0..inlets).collect(),
            positions: (0..inlets).collect(),
        }
    }

    pub(super) fn insert(&mut self, index: usize) {
        if self.positions[index] == usize::MAX {
            self.positions[index] = self.indices.len();
            self.indices.push(index);
        }
    }

    pub(super) fn remove(&mut self, index: usize) {
        let position = self.positions[index];
        if position == usize::MAX {
            return;
        }
        self.indices.swap_remove(position);
        if let Some(&moved) = self.indices.get(position) {
            self.positions[moved] = position;
        }
        self.positions[index] = usize::MAX;
    }

    fn pop(&mut self) -> Option<usize> {
        let index = *self.indices.last()?;
        self.remove(index);
        Some(index)
    }

    pub(super) fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub(super) fn clear(&mut self) {
        for &index in &self.indices {
            self.positions[index] = usize::MAX;
        }
        self.indices.clear();
    }
}

/// What a source or stage may touch during one call: its pipeline's inlets, the bytes the
/// source asked for, and the session.
pub struct Cx<'a> {
    pub(super) arena: &'a mut Arena,
    pub(super) inlets: &'a [PortId],
    pub(super) bytes: &'a mut VecDeque<(SegmentId, BufferHandle)>,
    pub(super) reads: &'a mut SmallVec<[SegmentId; 1]>,
    pub(super) session: &'a VortexSession,
    pub(super) exec: &'a mut ExecutionCtx,
    pub(super) touched: &'a mut InletSet,
    pub(super) changed: &'a mut InletSet,
}

impl Cx<'_> {
    /// Takes an inlet index whose sink appended data or closed since the last readiness
    /// check. Each index is reported once, and all inlets are reported on the first check.
    /// Remaining notifications are cleared after [`Source::ready`] returns. A source must
    /// account for its own consumption when caching inlet state.
    pub fn changed_inlet(&mut self) -> Option<usize> {
        self.changed.pop()
    }

    /// Queues a segment read during [`Source::ready`]. Return
    /// `SourceReady::Blocked(SourceBlocked::Io)` after queuing the reads needed for
    /// this call.
    pub fn request_read(&mut self, segment: SegmentId) {
        self.reads.push(segment);
    }

    /// Whether a requested segment has arrived and can be taken.
    pub fn has_bytes(&self) -> bool {
        !self.bytes.is_empty()
    }

    /// The reading side of inlet `index` of this pipeline's source.
    pub fn inlet(&mut self, index: usize) -> Inlet<'_> {
        self.touched.insert(index);
        self.arena.inlet(self.inlets[index])
    }

    /// Takes an arrived segment's ID and bytes, oldest arrival first.
    pub fn take_segment(&mut self) -> Option<(SegmentId, BufferHandle)> {
        self.bytes.pop_front()
    }

    /// The session used for decoding and evaluation.
    pub fn session(&self) -> &VortexSession {
        self.session
    }

    /// The execution context for executing arrays.
    pub fn exec(&mut self) -> &mut ExecutionCtx {
        self.exec
    }
}
