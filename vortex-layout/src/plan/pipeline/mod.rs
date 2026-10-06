// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pipeline execution of physical plans: an experimental second executor beside
//! [`exec`](crate::plan::exec), kept until one of the two is chosen.
//!
//! An [`ExecGraph`](crate::plan::exec::ExecGraph) grows as its nodes run. A [`PipelineGraph`] is
//! compiled whole when it is built: each plan operator adds what it needs through
//! [`PlanVTable::compile`](crate::plan::PlanVTable::compile), as one of three things.
//!
//! - A [`Source`] produces one piece, from one segment or from no read at all.
//! - A [`Transform`] changes each piece that passes through it.
//! - A [`Sink`] collects the pieces of its inputs and produces one piece from them.
//!
//! A pipeline is a source or a sink, the transforms its piece passes through, and the sink port
//! or graph root the piece lands in. A piece runs the length of its pipeline in one step.
//!
//! Every pipeline is known once the graph is built, and so is every read and the number of pieces
//! each sink takes. A sink is ready when it has counted that many, however they arrive.
//!
//! A `PipelineGraph` is driven as an `ExecGraph` is, and shares its request, piece, and output
//! types.

pub(crate) mod ops;

use std::collections::VecDeque;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::plan::PlanRef;
use crate::plan::exec::DecodeCache;
use crate::plan::exec::ExecContext;
use crate::plan::exec::ExecOutput;
use crate::plan::exec::ExecState;
use crate::plan::exec::IoBatch;
use crate::plan::exec::IoRequest;
use crate::plan::exec::IoRequestId;
use crate::plan::exec::Piece;
use crate::plan::exec::Port;
use crate::segments::SegmentId;

/// Produces one piece of a [`PipelineGraph`]: the rows it was added to the graph for.
pub trait Source: Send {
    /// The segment whose bytes the source needs, or `None` when it needs no read.
    fn segment(&self) -> Option<SegmentId>;

    /// Produces the source's rows. `bytes` holds the segment [`segment`](Self::segment) names,
    /// and is `None` when it names none.
    fn produce(&mut self, bytes: Option<BufferHandle>, ctx: &ExecContext)
    -> VortexResult<ArrayRef>;
}

/// Changes each piece passing from a source or sink to where it lands.
///
/// A transform is shared by every pipeline compiled beneath it, so it keeps nothing between
/// pieces.
pub trait Transform: Send + Sync {
    /// Returns the piece that takes the place of `piece`, covering the same rows.
    fn apply(&self, piece: Piece) -> VortexResult<Piece>;
}

/// Collects the pieces of its inputs and produces one piece from them.
pub trait Sink: Send {
    /// Takes a piece from the input on `port`. Pieces arrive in any order.
    fn push(&mut self, port: Port, piece: Piece) -> VortexResult<()>;

    /// Produces the sink's rows, once every piece of every input has been pushed.
    fn finish(&mut self) -> VortexResult<ArrayRef>;
}

/// One stage a piece passes through on the way to where it lands.
#[derive(Clone)]
enum Step {
    /// Moves the piece's rows into the row domain of a parent that concatenates its children.
    Rebase(u64),
    Transform(Arc<dyn Transform>),
}

/// Where the piece of a pipeline lands.
#[derive(Clone, Copy)]
enum Dest {
    Root,
    Sink { sink: usize, port: Port },
}

/// The steps a produced piece passes through, as a range of [`PipelineGraph::steps`], and where
/// it lands.
#[derive(Clone, Copy)]
struct Tail {
    start: usize,
    end: usize,
    dest: Dest,
}

struct SourceSlot {
    /// Taken when the source produces.
    source: Option<Box<dyn Source>>,
    rows: Range<u64>,
    tail: Tail,
    /// Whether the source's read is yet to be delivered.
    awaiting: bool,
}

struct SinkSlot {
    /// Dropped when the sink finishes.
    sink: Option<Box<dyn Sink>>,
    rows: Range<u64>,
    /// Pieces the sink has yet to take.
    remaining: usize,
    tail: Tail,
}

/// Something ready to run.
enum Work {
    /// A source, with the bytes of its read if it made one.
    Source(usize, Option<BufferHandle>),
    /// A piece known when the graph was built.
    Piece(Piece, Tail),
}

/// A running plan, compiled into pipelines when it is built.
pub struct PipelineGraph {
    ctx: ExecContext,
    sources: Vec<SourceSlot>,
    sinks: Vec<SinkSlot>,
    /// The steps of every pipeline, each pipeline's in the order its piece passes through them.
    steps: Vec<Step>,
    ready: Vec<Work>,
    new_io: IoBatch,
    outputs: VecDeque<Piece>,
    /// Pipelines landing in the root whose piece has not arrived.
    root_open: usize,
}

impl PipelineGraph {
    /// Compiles `plan` over `rows`, restricted to `mask`. `row_offset` is the global row index of
    /// the plan's first row, used by row-index plans. Segments found in `decoded` are neither
    /// read nor decoded again, and segments the graph decodes are added to it.
    ///
    /// Construction does no IO. Every read is returned by the first [`compute`](Self::compute).
    pub fn try_new(
        session: VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: Mask,
        row_offset: u64,
        decoded: DecodeCache,
    ) -> VortexResult<Self> {
        let mut graph = Self {
            ctx: ExecContext::new(session, row_offset, decoded),
            sources: Vec::new(),
            sinks: Vec::new(),
            steps: Vec::new(),
            ready: Vec::new(),
            new_io: Vec::new(),
            outputs: VecDeque::new(),
            root_open: 0,
        };
        let mut builder = GraphBuilder {
            graph: &mut graph,
            stack: Vec::new(),
            dest: Dest::Root,
        };
        builder.child(plan, rows, mask)?;
        Ok(graph)
    }

    /// Reports what the graph needs next. Cheap and side-effect free.
    pub fn state(&self) -> ExecState {
        if !self.outputs.is_empty() || !self.ready.is_empty() || !self.new_io.is_empty() {
            ExecState::NeedsCompute
        } else if self.root_open == 0 {
            ExecState::Done
        } else {
            ExecState::Waiting
        }
    }

    /// Runs ready pipelines until a piece reaches the root, the graph's reads should be flushed,
    /// a source has decoded a delivered read, or nothing is ready.
    ///
    /// # Errors
    ///
    /// Returns the error of a source, transform, or sink that failed. The graph cannot be run
    /// further.
    pub fn compute(&mut self) -> VortexResult<ExecOutput> {
        loop {
            if let Some(piece) = self.outputs.pop_front() {
                return Ok(ExecOutput::Piece(piece));
            }
            // Get reads in flight before spending CPU.
            if !self.new_io.is_empty() {
                return Ok(ExecOutput::NeedsIO(mem::take(&mut self.new_io)));
            }
            let Some(work) = self.ready.pop() else {
                return Ok(ExecOutput::Yield);
            };
            // Decoding newly read data is the costly work between reads that other owners' work
            // should interleave with.
            let delivered = self.run(work)?;
            if delivered && self.outputs.is_empty() {
                return Ok(ExecOutput::Yield);
            }
        }
    }

    /// Delivers the result of a request returned by [`compute`](Self::compute).
    pub fn set_io_result(&mut self, id: IoRequestId, result: BufferHandle) -> VortexResult<()> {
        let source = usize::try_from(id.0)
            .ok()
            .filter(|source| self.sources.get(*source).is_some_and(|slot| slot.awaiting))
            .ok_or_else(|| vortex_err!("Unknown IO request {id:?}"))?;
        self.sources[source].awaiting = false;
        self.ready.push(Work::Source(source, Some(result)));
        Ok(())
    }

    /// Runs one piece of work down its pipeline, and on through every sink it completes.
    /// Returns whether a delivered read was decoded.
    fn run(&mut self, work: Work) -> VortexResult<bool> {
        let (mut piece, mut tail, delivered) = match work {
            Work::Source(source, bytes) => {
                let delivered = bytes.is_some();
                let slot = &mut self.sources[source];
                let Some(mut producer) = slot.source.take() else {
                    vortex_bail!("Source {source} produced twice");
                };
                let array = producer.produce(bytes, &self.ctx)?;
                let piece = Piece {
                    rows: slot.rows.clone(),
                    array,
                };
                (piece, slot.tail, delivered)
            }
            Work::Piece(piece, tail) => (piece, tail, false),
        };
        loop {
            for step in &self.steps[tail.start..tail.end] {
                piece = match step {
                    Step::Rebase(offset) => Piece {
                        rows: piece.rows.start + offset..piece.rows.end + offset,
                        array: piece.array,
                    },
                    Step::Transform(transform) => transform.apply(piece)?,
                };
            }
            match tail.dest {
                Dest::Root => {
                    self.outputs.push_back(piece);
                    self.root_open -= 1;
                    return Ok(delivered);
                }
                Dest::Sink { sink, port } => {
                    let slot = &mut self.sinks[sink];
                    let Some(collector) = slot.sink.as_mut() else {
                        vortex_bail!("Sink {sink} took a piece after it finished");
                    };
                    collector.push(port, piece)?;
                    slot.remaining -= 1;
                    if slot.remaining > 0 {
                        return Ok(delivered);
                    }
                    let array = collector.finish()?;
                    slot.sink = None;
                    piece = Piece {
                        rows: slot.rows.clone(),
                        array,
                    };
                    tail = slot.tail;
                }
            }
        }
    }
}

/// Compiles plans into a [`PipelineGraph`].
///
/// A plan adds itself through its [`PlanVTable::compile`](crate::plan::PlanVTable::compile). What
/// it adds feeds the place the builder currently points at: the graph's root to begin with, then
/// the transforms and sink ports its ancestors opened around it.
pub struct GraphBuilder<'a> {
    graph: &'a mut PipelineGraph,
    /// The steps between the plan being compiled and where its pieces land, outermost first.
    stack: Vec<Step>,
    dest: Dest,
}

impl GraphBuilder<'_> {
    /// What the graph gives every source it runs.
    pub fn context(&self) -> &ExecContext {
        &self.graph.ctx
    }

    /// Compiles `plan` over `rows` of its row domain, restricted to `mask`, which is as long as
    /// `rows`.
    pub fn child(&mut self, plan: &PlanRef, rows: Range<u64>, mask: Mask) -> VortexResult<()> {
        plan.compile(rows, mask, self)
    }

    /// Adds a source producing `rows` of the plan being compiled.
    pub fn source(&mut self, rows: Range<u64>, source: Box<dyn Source>) {
        let tail = self.tail();
        let id = self.graph.sources.len();
        let segment = source.segment();
        self.graph.sources.push(SourceSlot {
            source: Some(source),
            rows,
            tail,
            awaiting: segment.is_some(),
        });
        match segment {
            Some(segment_id) => self.graph.new_io.push(IoRequest {
                id: IoRequestId(id as u64),
                segment_id,
            }),
            None => self.graph.ready.push(Work::Source(id, None)),
        }
    }

    /// Adds a piece of the plan being compiled that is already known.
    pub fn piece(&mut self, piece: Piece) {
        let tail = self.tail();
        self.graph.ready.push(Work::Piece(piece, tail));
    }

    /// Passes every piece of what `input` compiles through `transform`.
    pub fn transform(
        &mut self,
        transform: Arc<dyn Transform>,
        input: impl FnOnce(&mut Self) -> VortexResult<()>,
    ) -> VortexResult<()> {
        self.step(Step::Transform(transform), input)
    }

    /// Moves the rows of every piece of what `input` compiles up by `offset`, from the row domain
    /// of a child into that of a parent holding the child at `offset`.
    pub fn rebase(
        &mut self,
        offset: u64,
        input: impl FnOnce(&mut Self) -> VortexResult<()>,
    ) -> VortexResult<()> {
        if offset == 0 {
            return input(self);
        }
        self.step(Step::Rebase(offset), input)
    }

    /// Adds a sink producing `rows` of the plan being compiled from what `inputs` compiles.
    ///
    /// Inside `inputs`, [`port`](Self::port) names the port each input feeds.
    pub fn sink(
        &mut self,
        rows: Range<u64>,
        sink: Box<dyn Sink>,
        inputs: impl FnOnce(&mut Self) -> VortexResult<()>,
    ) -> VortexResult<()> {
        let tail = self.tail();
        let id = self.graph.sinks.len();
        self.graph.sinks.push(SinkSlot {
            sink: Some(sink),
            rows,
            remaining: 0,
            tail,
        });
        // The sink's inputs are pipelines of their own, ending at the sink.
        let stack = mem::take(&mut self.stack);
        let dest = mem::replace(&mut self.dest, Dest::Sink { sink: id, port: 0 });
        let result = inputs(self);
        self.stack = stack;
        self.dest = dest;
        result?;
        vortex_ensure!(
            self.graph.sinks[id].remaining > 0,
            "Sink {id} has no inputs"
        );
        Ok(())
    }

    /// Feeds what `input` compiles to `port` of the sink being added.
    pub fn port(
        &mut self,
        port: Port,
        input: impl FnOnce(&mut Self) -> VortexResult<()>,
    ) -> VortexResult<()> {
        let Dest::Sink { sink, port: outer } = self.dest else {
            vortex_bail!("Port {port} is named outside a sink");
        };
        self.dest = Dest::Sink { sink, port };
        let result = input(self);
        self.dest = Dest::Sink { sink, port: outer };
        result
    }

    fn step(
        &mut self,
        step: Step,
        input: impl FnOnce(&mut Self) -> VortexResult<()>,
    ) -> VortexResult<()> {
        self.stack.push(step);
        let result = input(self);
        self.stack.pop();
        result
    }

    /// Records the pipeline of a piece produced where the builder points, and counts the piece
    /// at where it lands.
    fn tail(&mut self) -> Tail {
        let start = self.graph.steps.len();
        self.graph.steps.extend(self.stack.iter().rev().cloned());
        match self.dest {
            Dest::Root => self.graph.root_open += 1,
            Dest::Sink { sink, .. } => self.graph.sinks[sink].remaining += 1,
        }
        Tail {
            start,
            end: self.graph.steps.len(),
            dest: self.dest,
        }
    }
}
