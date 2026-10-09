// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scan: its pipelines, the driver that runs one, the scheduler that picks which, and the
//! splits whose stages it compiles.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Blocked;
use super::Cx;
use super::DEFAULT_CAPACITY;
use super::Input;
use super::Operator;
use super::Source;
use super::Step;
use super::compile::Chain;
use super::compile::Shares;
use super::port::Arena;
use super::port::PipelineId;
use super::port::PortId;
use super::port::Reader;
use super::port::Slab;
use super::query::QueryRun;
use crate::layouts::zoned::zone_map::ZoneMap;
use crate::plan::PlanRef;
use crate::plan::Query;
use crate::plan::QueryPlan;
use crate::segments::SegmentId;

/// Identifies a read a scan asked its owner for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadId(u64);

/// A segment the scan needs read.
#[derive(Clone, Debug)]
pub struct ReadRequest {
    /// Identifies the read when its bytes are delivered.
    pub id: ReadId,
    /// The segment to read.
    pub segment_id: SegmentId,
}

/// A row range of the plan's domain, and the rows of it to produce.
#[derive(Clone, Debug)]
pub struct Split {
    /// The rows.
    pub rows: Range<u64>,
    /// Which of them to produce; as long as `rows`.
    pub mask: Mask,
}

impl Split {
    /// Every row of `rows`.
    pub fn all(rows: Range<u64>) -> Self {
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            mask: Mask::new_true(len),
            rows,
        }
    }
}

/// What the owner of a [`Scan`] does next.
#[derive(Debug)]
pub enum Turn {
    /// Read this segment and [`deliver`](Scan::deliver) its bytes.
    Read(ReadRequest),
    /// The next rows of split `split`, in row order within the split.
    Output(usize, ArrayRef),
    /// Nothing can run until a read is delivered.
    Waiting,
    /// Every split has produced all its rows.
    Done,
}

/// Where a pipeline is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Runnable,
    Blocked(Blocked),
}

/// A source, its stages, and the ports they read and write.
pub(crate) struct Pipeline {
    source: Box<dyn Source>,
    stages: Vec<Box<dyn Operator>>,
    inlets: Vec<PortId>,
    outlets: SmallVec<[PortId; 1]>,
    bytes: Option<BufferHandle>,
    /// The inlets the last run read, each once.
    touched: Vec<usize>,
    /// For each inlet, whether it is in `touched`.
    listed: Vec<bool>,
    state: State,
    queued: bool,
}

/// What one run of a pipeline came to.
enum Progress {
    /// The source wants this segment read.
    Read(SegmentId),
    /// The pipeline did work and may do more.
    Ran,
    /// The pipeline cannot run until the condition clears.
    Blocked(Blocked),
    /// The pipeline has finished and closed its outlets.
    Done,
}

/// Runs `pipe` until it blocks, finishes, or requests a read.
fn drive(
    pipe: &mut Pipeline,
    arena: &mut Arena,
    session: &VortexSession,
    exec: &mut ExecutionCtx,
) -> VortexResult<Progress> {
    let Pipeline {
        source,
        stages,
        inlets,
        outlets,
        bytes,
        touched,
        listed,
        ..
    } = pipe;
    let mut cx = Cx {
        arena,
        inlets,
        bytes,
        session,
        exec,
        touched,
        listed,
    };
    loop {
        if !outlets.iter().all(|&outlet| cx.arena.has_room(outlet)) {
            return Ok(Progress::Blocked(Blocked::Outlet));
        }
        if let Some(segment) = source.request() {
            return Ok(Progress::Read(segment));
        }
        match source.compute(Input::None, &mut cx)? {
            // A source's input is its inlets and reads, which nothing changes while it runs, so
            // it is asked again at once: it blocks or finishes without a trip through the
            // scheduler, and the outlets' room still bounds what it produces.
            Step::More(batch) | Step::Last(batch) => push(stages, batch, outlets, &mut cx)?,
            Step::Consumed => return Ok(Progress::Ran),
            Step::Blocked(blocked) => return Ok(Progress::Blocked(blocked)),
            Step::Finished => {
                push_end(stages, outlets, &mut cx)?;
                return Ok(Progress::Done);
            }
        }
    }
}

/// Pushes `batch` through `stages` into `outlets`.
fn push(
    stages: &mut [Box<dyn Operator>],
    batch: ArrayRef,
    outlets: &[PortId],
    cx: &mut Cx<'_>,
) -> VortexResult<()> {
    let Some((stage, rest)) = stages.split_first_mut() else {
        push_out(batch, outlets, cx.arena);
        return Ok(());
    };
    let mut input = Input::Chunk(batch);
    loop {
        match stage.compute(input, cx)? {
            Step::More(batch) => {
                push(rest, batch, outlets, cx)?;
                input = Input::None;
            }
            Step::Last(batch) => return push(rest, batch, outlets, cx),
            Step::Consumed => return Ok(()),
            Step::Finished => return push_end(rest, outlets, cx),
            Step::Blocked(_) => vortex_bail!("A pipeline stage blocked; only sources block"),
        }
    }
}

/// Sends the end through `stages`, then closes `outlets`.
fn push_end(
    stages: &mut [Box<dyn Operator>],
    outlets: &[PortId],
    cx: &mut Cx<'_>,
) -> VortexResult<()> {
    let Some((stage, rest)) = stages.split_first_mut() else {
        for &outlet in outlets {
            cx.arena.close(outlet);
        }
        return Ok(());
    };
    loop {
        match stage.compute(Input::End, cx)? {
            Step::More(batch) => push(rest, batch, outlets, cx)?,
            Step::Last(batch) => {
                push(rest, batch, outlets, cx)?;
                return push_end(rest, outlets, cx);
            }
            Step::Consumed | Step::Finished => return push_end(rest, outlets, cx),
            Step::Blocked(_) => vortex_bail!("A pipeline stage blocked; only sources block"),
        }
    }
}

/// Writes `batch` into every outlet: one for a plain pipeline, one per reader for a share.
fn push_out(batch: ArrayRef, outlets: &[PortId], arena: &mut Arena) {
    if let Some((&last, rest)) = outlets.split_last() {
        for &outlet in rest {
            arena.push(outlet, batch.clone());
        }
        arena.push(last, batch);
    }
}

/// Everything a scan's pipelines share: the ports, the pipeline table, the scheduler, the reads
/// in flight, and the shares.
pub(crate) struct Core {
    pub(crate) session: VortexSession,
    pub(crate) exec: ExecutionCtx,
    /// The global row index of plan row zero.
    pub(crate) row_offset: u64,
    pub(crate) arena: Arena,
    pipelines: Slab<Option<Pipeline>>,
    free_pipelines: Vec<PipelineId>,
    /// Runnable pipelines, last in first out, so data a pipeline just produced is consumed
    /// before more is produced.
    ready: Vec<PipelineId>,
    reads: FxHashMap<ReadId, PipelineId>,
    next_read: u64,
    new_reads: VecDeque<ReadRequest>,
    /// Split slots whose stage output has something new, with a flag per slot so a slot is
    /// queued once.
    dirty: Vec<usize>,
    dirty_flags: Vec<bool>,
    pub(crate) shares: Shares,
    /// The zone tables the scan has read, by their plan's address, kept while it runs.
    pub(crate) zone_maps: FxHashMap<usize, Arc<ZoneMap>>,
    /// The zones each proof prunes, by the proof's address, proven once per scan.
    pub(crate) pruned: FxHashMap<usize, Mask>,
    /// The split being compiled, whose shared readers a compile claims. `usize::MAX` when no
    /// split is, as when a source asks for a plan mid-run.
    pub(crate) split: usize,
}

impl Core {
    fn new(session: VortexSession, row_offset: u64, splits: Vec<Range<u64>>) -> Self {
        Self {
            exec: session.create_execution_ctx(),
            session,
            row_offset,
            arena: Arena::default(),
            pipelines: Slab::default(),
            free_pipelines: Vec::new(),
            ready: Vec::new(),
            reads: FxHashMap::default(),
            next_read: 0,
            new_reads: VecDeque::new(),
            dirty: Vec::new(),
            dirty_flags: Vec::new(),
            shares: Shares::new(splits),
            zone_maps: FxHashMap::default(),
            pruned: FxHashMap::default(),
            split: usize::MAX,
        }
    }

    /// Makes `chain` a pipeline writing `outlets`, and queues it to run.
    pub(crate) fn add_pipeline(
        &mut self,
        chain: Chain,
        outlets: SmallVec<[PortId; 1]>,
    ) -> PipelineId {
        let listed = vec![false; chain.inlets.len()];
        let touched = Vec::with_capacity(chain.inlets.len());
        let pipeline = Pipeline {
            source: chain.source,
            stages: chain.stages,
            inlets: chain.inlets,
            outlets,
            bytes: None,
            touched,
            listed,
            state: State::Runnable,
            queued: true,
        };
        let id = match self.free_pipelines.pop() {
            Some(id) => id,
            None => self.pipelines.push(None),
        };
        for &inlet in &pipeline.inlets {
            self.arena.get_mut(inlet).reader = Reader::Pipeline(id);
        }
        for &outlet in &pipeline.outlets {
            self.arena.get_mut(outlet).writer = Some(id);
        }
        self.pipelines[id] = Some(pipeline);
        self.ready.push(id);
        id
    }

    /// Compiles `plan` over `rows`, restricted to `mask`, into pipelines whose output the scan
    /// reads for split `split` in slot `slot`, and returns the output port. `None` when nothing
    /// is selected.
    pub(crate) fn compile_stage(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
        (slot, split): (usize, usize),
    ) -> VortexResult<Option<PortId>> {
        self.split = split;
        let chain = self.compile(plan, rows, mask);
        self.split = usize::MAX;
        let Some(chain) = chain? else {
            return Ok(None);
        };
        let port = self
            .arena
            .create(DEFAULT_CAPACITY, None, Reader::Split(slot));
        self.add_pipeline(chain, smallvec::smallvec![port]);
        Ok(Some(port))
    }

    fn pipeline(&self, id: PipelineId) -> &Pipeline {
        self.pipelines[id]
            .as_ref()
            .unwrap_or_else(|| unreachable!("pipeline {id} is not live"))
    }

    fn pipeline_mut(&mut self, id: PipelineId) -> &mut Pipeline {
        self.pipelines[id]
            .as_mut()
            .unwrap_or_else(|| unreachable!("pipeline {id} is not live"))
    }

    fn make_runnable(&mut self, id: PipelineId) {
        let pipe = self.pipeline_mut(id);
        pipe.state = State::Runnable;
        if !pipe.queued {
            pipe.queued = true;
            self.ready.push(id);
        }
    }

    /// Runs pipeline `id` once and settles what it did.
    fn run(&mut self, id: PipelineId) -> VortexResult<()> {
        let pipe = self.pipelines[id]
            .as_mut()
            .ok_or_else(|| vortex_err!("Pipeline {id} is not live"))?;
        pipe.queued = false;
        let progress = drive(pipe, &mut self.arena, &self.session, &mut self.exec)?;
        match progress {
            Progress::Read(segment_id) => {
                let read = ReadId(self.next_read);
                self.next_read += 1;
                self.reads.insert(read, id);
                self.new_reads.push_back(ReadRequest {
                    id: read,
                    segment_id,
                });
                self.pipeline_mut(id).state = State::Blocked(Blocked::Io);
            }
            Progress::Ran => self.make_runnable(id),
            // Nothing changes a port while its reader computes, so the condition the source
            // reported holds until a neighbour wakes it.
            Progress::Blocked(blocked) => self.pipeline_mut(id).state = State::Blocked(blocked),
            Progress::Done => {}
        }
        self.wake_neighbours(id);
        if matches!(progress, Progress::Done) {
            self.retire(id);
        }
        Ok(())
    }

    /// Wakes the readers of what pipeline `id` wrote and the writers of what it took.
    fn wake_neighbours(&mut self, id: PipelineId) {
        for index in 0..self.pipeline(id).outlets.len() {
            let queue = self.arena.get(self.pipeline(id).outlets[index]);
            if !queue.readable() {
                continue;
            }
            let port = self.pipeline(id).outlets[index];
            match queue.reader {
                Reader::Pipeline(reader) => {
                    // A reader waits on one inlet, so a batch on any other cannot unblock it.
                    let reader_pipe = self.pipeline(reader);
                    if let State::Blocked(Blocked::Inlet(waiting)) = reader_pipe.state
                        && reader_pipe.inlets[waiting] == port
                    {
                        self.make_runnable(reader);
                    }
                }
                Reader::Split(slot) => self.mark_dirty(slot),
                Reader::Unclaimed => {}
            }
        }
        // Only an inlet the run read can have gained room, so a source with many inlets, as a
        // concatenation of many chunks, pays for the inlets it read, not for all of them.
        let touched = std::mem::take(&mut self.pipeline_mut(id).touched);
        for &index in &touched {
            if let Some(writer) = self.arena.get(self.pipeline(id).inlets[index]).writer
                && writer != id
                && self.writer_has_room(writer)
            {
                self.make_runnable(writer);
            }
        }
        let mut touched = touched;
        let pipe = self.pipeline_mut(id);
        for &index in &touched {
            pipe.listed[index] = false;
        }
        touched.clear();
        pipe.touched = touched;
    }

    /// Queues split slot `slot` for the scan to read its stage's output.
    fn mark_dirty(&mut self, slot: usize) {
        if !self.dirty_flags[slot] {
            self.dirty_flags[slot] = true;
            self.dirty.push(slot);
        }
    }

    /// Whether `writer` is blocked on its outlets and every one of them has room now.
    fn writer_has_room(&self, writer: PipelineId) -> bool {
        let pipe = self.pipeline(writer);
        pipe.state == State::Blocked(Blocked::Outlet)
            && pipe
                .outlets
                .iter()
                .all(|&outlet| self.arena.has_room(outlet))
    }

    /// Drops a finished pipeline. Its outlets are closed; its inlets lose their reader.
    fn retire(&mut self, id: PipelineId) {
        if let Some(pipe) = self.pipelines[id].take() {
            for inlet in pipe.inlets {
                self.arena.drop_reader(inlet);
            }
            self.free_pipelines.push(id);
        }
    }

    /// The scan has read a stage's output: wakes its writer if the output was full.
    fn read_output(&mut self, port: PortId) {
        if let Some(writer) = self.arena.get(port).writer
            && self.writer_has_room(writer)
        {
            self.make_runnable(writer);
        }
    }

    fn deliver(&mut self, read: ReadId, bytes: BufferHandle) -> VortexResult<()> {
        let id = self
            .reads
            .remove(&read)
            .ok_or_else(|| vortex_err!("Unknown read {read:?}"))?;
        self.pipeline_mut(id).bytes = Some(bytes);
        self.make_runnable(id);
        Ok(())
    }

    fn describe_blocked(&self) -> String {
        self.pipelines
            .iter()
            .enumerate()
            .filter_map(|(id, pipe)| pipe.as_ref().map(|pipe| format!("{id}: {:?}", pipe.state)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What a scan runs over each split.
#[derive(Clone)]
enum Root {
    /// The plan, under the split's selection.
    Plan(PlanRef),
    /// The query's stages: zone pruning, its conjuncts, then its projection.
    Query(QueryPlan),
}

/// A split being run.
struct Active {
    index: usize,
    /// The output port of the stage running, read by the scan.
    output: Option<PortId>,
    query: Option<QueryRun>,
}

/// Runs a plan over a list of splits.
///
/// A [`QueryPlan`] at the root is run as stages: zone pruning, then each conjunct under the rows
/// the earlier ones kept, then the projection under the rows they all kept. Any other plan runs
/// as one stage under each split's selection.
///
/// Up to [`with_max_active`](Self::with_max_active) splits are compiled at once, in order, so
/// reads of later splits are in flight while earlier ones compute.
pub struct Scan {
    core: Core,
    root: Root,
    splits: Vec<Split>,
    next: usize,
    active: Vec<Option<Active>>,
    live: usize,
    outputs: VecDeque<(usize, ArrayRef)>,
}

impl fmt::Debug for Scan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scan")
            .field("splits", &self.splits.len())
            .field("next", &self.next)
            .field("live", &self.live)
            .finish()
    }
}

impl Scan {
    /// A scan of `plan` over `splits`, with one split active at a time.
    ///
    /// Counts, from the plan, how often each split will read each segment, so segments read
    /// more than once are decoded once. Builds no pipeline and issues no read.
    pub fn try_new(
        session: VortexSession,
        plan: PlanRef,
        splits: Vec<Split>,
    ) -> VortexResult<Self> {
        for split in &splits {
            vortex_ensure!(
                split.rows.start <= split.rows.end
                    && split.rows.end <= plan.row_count()
                    && split.mask.len() as u64 == split.rows.end - split.rows.start,
                "Split {:?} with a mask of {} rows does not fit a plan of {} rows",
                split.rows,
                split.mask.len(),
                plan.row_count()
            );
        }
        let root = match plan.as_opt::<Query>() {
            Some(query) => Root::Query(query.clone()),
            None => Root::Plan(plan),
        };
        let mut core = Core::new(
            session,
            0,
            splits.iter().map(|split| split.rows.clone()).collect(),
        );
        match &root {
            Root::Plan(plan) => core.shares.add(plan, 0..plan.row_count())?,
            Root::Query(query) => QueryRun::reserve(query, &mut core)?,
        }
        core.shares.retain_shared();
        core.dirty_flags = vec![false; 1];
        Ok(Self {
            core,
            root,
            splits,
            next: 0,
            active: vec![None],
            live: 0,
            outputs: VecDeque::new(),
        })
    }

    /// Runs up to `max_active` splits at once.
    pub fn with_max_active(mut self, max_active: usize) -> Self {
        self.active.resize_with(max_active.max(1), || None);
        self.core.dirty_flags = vec![false; self.active.len()];
        self.core.dirty = Vec::with_capacity(self.active.len());
        self
    }

    /// Sets the global row index of plan row zero, which row-index plans add to their rows.
    pub fn with_row_offset(mut self, row_offset: u64) -> Self {
        self.core.row_offset = row_offset;
        self
    }

    /// Runs pipelines until the owner has something to do.
    pub fn step(&mut self) -> VortexResult<Turn> {
        loop {
            if let Some((split, array)) = self.outputs.pop_front() {
                return Ok(Turn::Output(split, array));
            }
            if let Some(read) = self.core.new_reads.pop_front() {
                return Ok(Turn::Read(read));
            }
            if let Some(slot) = self.core.dirty.pop() {
                self.core.dirty_flags[slot] = false;
                self.advance(slot)?;
                continue;
            }
            if self.live < self.active.len() && self.next < self.splits.len() {
                self.activate()?;
                continue;
            }
            if let Some(id) = self.core.ready.pop() {
                self.core.run(id)?;
                continue;
            }
            if !self.core.reads.is_empty() {
                return Ok(Turn::Waiting);
            }
            if self.live == 0 {
                return Ok(Turn::Done);
            }
            vortex_bail!(
                "Scan is stuck with nothing runnable and no read in flight: {}",
                self.core.describe_blocked()
            );
        }
    }

    /// Delivers the bytes of a read [`step`](Self::step) returned.
    pub fn deliver(&mut self, read: ReadId, bytes: BufferHandle) -> VortexResult<()> {
        self.core.deliver(read, bytes)
    }

    /// Starts the next split in a free slot.
    fn activate(&mut self) -> VortexResult<()> {
        let slot = self
            .active
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| vortex_err!("No free split slot"))?;
        let index = self.next;
        self.next += 1;
        let split = self.splits[index].clone();
        let (output, query) = match &self.root {
            Root::Plan(plan) => (
                self.core
                    .compile_stage(plan, split.rows.clone(), &split.mask, (slot, index))?,
                None,
            ),
            Root::Query(plan) => {
                let mut query = QueryRun::new(plan.clone(), split.rows, split.mask);
                (
                    query.next_stage(&mut self.core, (slot, index))?,
                    Some(query),
                )
            }
        };
        if output.is_none() {
            self.core.shares.finish(index, &mut self.core.arena);
        } else {
            self.active[slot] = Some(Active {
                index,
                output,
                query,
            });
            self.live += 1;
        }
        Ok(())
    }

    /// Takes what the stage running in `slot` produced, and moves the split on when the stage
    /// has finished.
    fn advance(&mut self, slot: usize) -> VortexResult<()> {
        let Some(active) = self.active[slot].as_mut() else {
            return Ok(());
        };
        while let Some(port) = active.output {
            while let Some(batch) = self.core.arena.get_mut(port).pop() {
                match active.query.as_mut() {
                    None => self.outputs.push_back((active.index, batch)),
                    Some(query) => {
                        if let Some(batch) = query.accept(batch, &mut self.core)? {
                            self.outputs.push_back((active.index, batch));
                        }
                    }
                }
            }
            self.core.read_output(port);
            if !self.core.arena.get(port).closed() {
                return Ok(());
            }
            self.core.arena.drop_reader(port);
            active.output = match active.query.as_mut() {
                None => None,
                Some(query) => {
                    query.finish_stage(&mut self.core)?;
                    query.next_stage(&mut self.core, (slot, active.index))?
                }
            };
        }
        self.core.shares.finish(active.index, &mut self.core.arena);
        self.active[slot] = None;
        self.live -= 1;
        Ok(())
    }

    /// Ports in use.
    #[cfg(test)]
    pub(crate) fn live_ports(&self) -> usize {
        self.core.arena.live()
    }

    /// Shared ports not yet claimed or dropped.
    #[cfg(test)]
    pub(crate) fn pending_shares(&self) -> usize {
        self.core.shares.pending()
    }
}
