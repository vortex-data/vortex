// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scan: its pipelines, the driver that runs one, the scheduler that picks which, and the
//! splits whose stages it compiles.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;

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

use super::DEFAULT_CAPACITY;
use super::compile::Chain;
use super::port::Arena;
use super::port::PipelineId;
use super::port::PortId;
use super::port::Slab;
use super::port::SourceEnd;
use super::primitive::PrimitiveRead;
use super::values::ValueCache;
use crate::plan::physical::PlanRef;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::InletSet;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
use crate::segments::SegmentId;

#[cfg(debug_assertions)]
mod validate;

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
    /// Optional sparse primitive byte demand. Owners may ignore it and read the whole segment.
    pub primitive: Option<PrimitiveRead>,
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

/// Whether a pipeline can run, and if not, what it waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Runnable,
    /// What its source said it waits for.
    Blocked(SourceBlocked),
    /// Room in a full outlet, found by the driver before the source ran.
    Outlet,
}

/// A source, its stages, and the ports they read and write.
pub(crate) struct Pipeline {
    source: Box<dyn Source>,
    stages: Vec<Box<dyn Operator>>,
    inlets: Vec<PortId>,
    outlets: SmallVec<[PortId; 1]>,
    /// The segments it asked for that have arrived, in order of arrival.
    bytes: VecDeque<(SegmentId, BufferHandle, bool)>,
    /// Reads it asked for that have not arrived, so retiring it cancels them only when any
    /// are left.
    pending: usize,
    /// Once its source has finished, the next stage to finish: stages are finished one at a
    /// time, each when the outlets have room for what it flushes.
    finishing: Option<usize>,
    /// Stages with more output for their current input, resumed deepest first.
    continuing: Vec<usize>,
    /// The source can compute again without checking readiness.
    source_more: bool,
    touched: InletSet,
    changed: InletSet,
    state: State,
    queued: bool,
}

/// What one run of a pipeline came to.
enum Progress {
    /// The source waits for this.
    Blocked(SourceBlocked),
    /// An outlet is full.
    Outlet,
    /// The pipeline has finished and closed its outlets.
    Done,
}

/// Runs `pipe` until it blocks or finishes, collecting any reads its source requests.
fn drive(
    pipe: &mut Pipeline,
    arena: &mut Arena,
    session: &VortexSession,
    exec: &mut ExecutionCtx,
    reads: &mut SmallVec<[(SegmentId, Option<PrimitiveRead>); 1]>,
) -> VortexResult<Progress> {
    let Pipeline {
        source,
        stages,
        inlets,
        outlets,
        bytes,
        finishing,
        continuing,
        source_more,
        touched,
        changed,
        ..
    } = pipe;
    let mut cx = Cx {
        arena,
        inlets,
        bytes,
        reads,
        session,
        exec,
        touched,
        changed,
    };
    loop {
        // Closing needs no room, so a pipeline with nothing left to flush closes at once.
        if *finishing == Some(stages.len()) && continuing.is_empty() {
            for &outlet in outlets.iter() {
                cx.arena.close(outlet);
            }
            return Ok(Progress::Done);
        }
        if !outlets.iter().all(|&outlet| cx.arena.has_room(outlet)) {
            return Ok(Progress::Outlet);
        }
        if let Some(stage) = continuing.pop() {
            let step = stages[stage].compute(Input::None, &mut cx)?;
            push(stages, stage, step, continuing, outlets, &mut cx)?;
            continue;
        }
        if let Some(stage) = *finishing {
            *finishing = Some(stage + 1);
            let step = stages[stage].compute(Input::End, &mut cx)?;
            push(stages, stage, step, continuing, outlets, &mut cx)?;
            continue;
        }
        if !*source_more {
            let ready = source.ready(&mut cx)?;
            cx.changed.clear();
            vortex_ensure!(
                cx.reads.is_empty() || ready == SourceReady::Blocked(SourceBlocked::Io),
                "A source that queues reads must block on IO"
            );
            match ready {
                SourceReady::Ready => {}
                SourceReady::Blocked(blocked) => return Ok(Progress::Blocked(blocked)),
            }
        }
        let step = source.compute(Input::None, &mut cx)?;
        *source_more = matches!(&step, Step::More(_));
        match step {
            Step::More(batch) | Step::Final(batch) => {
                push_batch(stages, 0, batch, continuing, outlets, &mut cx)?;
            }
            Step::End => *finishing = Some(0),
        }
    }
}

/// Records a stage's continuation before pushing its output through the remaining stages.
fn push(
    stages: &mut [Box<dyn Operator>],
    stage: usize,
    step: Step,
    continuing: &mut Vec<usize>,
    outlets: &[PortId],
    cx: &mut Cx<'_>,
) -> VortexResult<()> {
    let batch = match step {
        Step::End => return Ok(()),
        Step::More(batch) => {
            continuing.push(stage);
            batch
        }
        Step::Final(batch) => batch,
    };
    push_batch(stages, stage + 1, batch, continuing, outlets, cx)
}

/// Pushes one batch, retaining continuations so outlet capacity bounds every output.
fn push_batch(
    stages: &mut [Box<dyn Operator>],
    start: usize,
    mut batch: ArrayRef,
    continuing: &mut Vec<usize>,
    outlets: &[PortId],
    cx: &mut Cx<'_>,
) -> VortexResult<()> {
    for (index, stage) in stages.iter_mut().enumerate().skip(start) {
        batch = match stage.compute(Input::Batch(batch), cx)? {
            Step::More(batch) => {
                continuing.push(index);
                batch
            }
            Step::Final(batch) => batch,
            Step::End => return Ok(()),
        };
    }
    push_out(batch, outlets, cx.arena);
    Ok(())
}

/// Writes `batch` into every outlet: one for a plain pipeline, one per claimant for a share.
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
    /// Reads in flight: the pipeline that asked, or `None` once that pipeline has finished and
    /// the read is cancelled, and the segment.
    reads: FxHashMap<ReadId, (Option<PipelineId>, SegmentId)>,
    next_read: u64,
    new_reads: VecDeque<ReadRequest>,
    /// Split slots whose stage output has something new, with a flag per slot so a slot is
    /// queued once.
    dirty: Vec<usize>,
    dirty_flags: Vec<bool>,
    /// The segment of every scan source compiled so far, in compile order.
    pub(crate) segments: Vec<SegmentId>,
    /// Whole values kept for later scans of the same split.
    pub(crate) values: ValueCache,
    pub(crate) segment_cache: Option<super::SegmentCache>,
}

impl Core {
    fn new(session: VortexSession, row_offset: u64) -> Self {
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
            segments: Vec::new(),
            values: ValueCache::default(),
            segment_cache: None,
        }
    }

    /// Makes `chain` a pipeline writing `outlets`, and queues it to run.
    pub(crate) fn add_pipeline(
        &mut self,
        chain: Chain,
        outlets: SmallVec<[PortId; 1]>,
    ) -> PipelineId {
        let touched = InletSet::new(chain.inlets.len());
        let changed = InletSet::all(chain.inlets.len());
        let pipeline = Pipeline {
            continuing: Vec::new(),
            source_more: false,
            source: chain.source,
            stages: chain.stages,
            inlets: chain.inlets,
            outlets,
            bytes: VecDeque::new(),
            pending: 0,
            finishing: None,
            touched,
            changed,
            state: State::Runnable,
            queued: true,
        };
        let id = match self.free_pipelines.pop() {
            Some(id) => id,
            None => self.pipelines.push(None),
        };
        for (index, &inlet) in pipeline.inlets.iter().enumerate() {
            self.arena.get_mut(inlet).source = SourceEnd::Pipeline { id, inlet: index };
        }
        for &outlet in &pipeline.outlets {
            self.arena.get_mut(outlet).sink = Some(id);
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
        slot: usize,
    ) -> VortexResult<Option<PortId>> {
        let Some(chain) = self.compile(plan, rows, mask)? else {
            return Ok(None);
        };
        let port = self
            .arena
            .create(DEFAULT_CAPACITY, None, SourceEnd::Scan(slot));
        self.add_pipeline(chain, smallvec::smallvec![port]);
        #[cfg(debug_assertions)]
        self.validate();
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
        let mut reads = SmallVec::new();
        let progress = drive(
            pipe,
            &mut self.arena,
            &self.session,
            &mut self.exec,
            &mut reads,
        )?;
        for (segment_id, primitive) in reads {
            self.ask(id, segment_id, primitive);
        }
        #[cfg(debug_assertions)]
        self.validate_progress(id, &progress);
        match &progress {
            // A read-ahead source can still have delivered bytes to consume.
            Progress::Blocked(SourceBlocked::Io) if !self.pipeline(id).bytes.is_empty() => {
                self.make_runnable(id);
            }
            // Nothing changes a port while its source's pipeline runs, so the condition the run reported
            // holds until a neighbour wakes it.
            Progress::Blocked(blocked) => self.pipeline_mut(id).state = State::Blocked(*blocked),
            Progress::Outlet => self.pipeline_mut(id).state = State::Outlet,
            Progress::Done => {}
        }
        self.wake_neighbours(id);
        if matches!(progress, Progress::Done) {
            self.retire(id);
        }
        Ok(())
    }

    /// Issues a read of `segment_id`, which pipeline `id`'s source asked for.
    fn ask(&mut self, id: PipelineId, segment_id: SegmentId, primitive: Option<PrimitiveRead>) {
        self.read(id, segment_id, primitive);
        self.pipeline_mut(id).pending += 1;
    }

    /// Wakes the sources of what pipeline `id` pushed and the sinks of what it took.
    fn wake_neighbours(&mut self, id: PipelineId) {
        for index in 0..self.pipeline(id).outlets.len() {
            let port = self.pipeline(id).outlets[index];
            let queue = self.arena.get_mut(port);
            if !queue.take_changed() {
                continue;
            }
            match queue.source {
                SourceEnd::Pipeline { id: source, inlet } => {
                    // Record every change, including inlets the source is not waiting on.
                    let source_pipe = self.pipeline_mut(source);
                    source_pipe.changed.insert(inlet);
                    if source_pipe.state == State::Blocked(SourceBlocked::Inlet(inlet)) {
                        self.make_runnable(source);
                    }
                }
                SourceEnd::Scan(slot) => self.mark_dirty(slot),
                SourceEnd::Unclaimed => {}
            }
        }
        // Only an inlet the run read can have gained room, so a source with many inlets, as a
        // concatenation of many chunks, pays for the inlets it read, not for all of them.
        let mut touched = std::mem::take(&mut self.pipeline_mut(id).touched);
        for &index in touched.indices() {
            if let Some(sink) = self.arena.get(self.pipeline(id).inlets[index]).sink
                && sink != id
                && self.sink_has_room(sink)
            {
                self.make_runnable(sink);
            }
        }
        touched.clear();
        self.pipeline_mut(id).touched = touched;
    }

    /// Queues split slot `slot` for the scan to read its stage's output.
    fn mark_dirty(&mut self, slot: usize) {
        if !self.dirty_flags[slot] {
            self.dirty_flags[slot] = true;
            self.dirty.push(slot);
        }
    }

    /// Whether pipeline `sink` is blocked on its outlets and every one of them has room now.
    fn sink_has_room(&self, sink: PipelineId) -> bool {
        let pipe = self.pipeline(sink);
        pipe.state == State::Outlet
            && pipe
                .outlets
                .iter()
                .all(|&outlet| self.arena.has_room(outlet))
    }

    /// Drops a finished pipeline. Its outlets are closed; its inlets lose their source; the reads
    /// it asked for and never took are cancelled, so their bytes are dropped on arrival instead
    /// of reaching a pipeline that reuses its id.
    fn retire(&mut self, id: PipelineId) {
        if let Some(pipe) = self.pipelines[id].take() {
            for inlet in pipe.inlets {
                self.arena.drop_source(inlet);
            }
            if pipe.pending > 0 {
                self.cancel_reads(id);
            }
            self.free_pipelines.push(id);
        }
    }

    /// Cancels the reads pipeline `id` waits for: they become reads for no pipeline, whose
    /// bytes are dropped on arrival.
    fn cancel_reads(&mut self, id: PipelineId) {
        for (owner, _) in self.reads.values_mut() {
            if *owner == Some(id) {
                *owner = None;
            }
        }
    }

    /// The scan has read a stage's output: wakes its sink if the output was full.
    fn read_output(&mut self, port: PortId) {
        if let Some(sink) = self.arena.get(port).sink
            && self.sink_has_room(sink)
        {
            self.make_runnable(sink);
        }
    }

    fn deliver(&mut self, read: ReadId, bytes: BufferHandle, whole: bool) -> VortexResult<()> {
        let (owner, segment) = self
            .reads
            .remove(&read)
            .ok_or_else(|| vortex_err!("Unknown read {read:?}"))?;
        if let Some(id) = owner {
            let pipe = self.pipeline_mut(id);
            pipe.pending -= 1;
            pipe.bytes.push_back((segment, bytes, whole));
            self.make_runnable(id);
        }
        Ok(())
    }

    /// Issues a read of `segment` for pipeline `owner`.
    fn read(&mut self, owner: PipelineId, segment_id: SegmentId, primitive: Option<PrimitiveRead>) {
        let read = ReadId(self.next_read);
        self.next_read += 1;
        self.reads.insert(read, (Some(owner), segment_id));
        self.new_reads.push_back(ReadRequest {
            id: read,
            segment_id,
            primitive,
        });
    }

    /// Every live pipeline's state, and for one waiting on an inlet, that port and its sink.
    fn describe_blocked(&self) -> String {
        self.pipelines
            .iter()
            .enumerate()
            .filter_map(|(id, pipe)| {
                let pipe = pipe.as_ref()?;
                Some(match pipe.state {
                    State::Blocked(SourceBlocked::Inlet(index)) => {
                        let port = pipe.inlets[index];
                        let queue = self.arena.get(port);
                        format!(
                            "{id}: {:?} (port {port}: {} queued, closed {}, sink {:?})",
                            pipe.state,
                            queue.len(),
                            queue.closed(),
                            queue.sink.map(|sink| (sink, self.pipeline(sink).state)),
                        )
                    }
                    state => format!("{id}: {state:?}"),
                })
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// What a scan runs over each split.
#[derive(Clone)]
enum Root {
    /// The plan, under the split's selection.
    Plan(PlanRef),
}

/// A split being run.
struct Active {
    index: usize,
    /// The output port of the stage running, read by the scan.
    output: Option<PortId>,
}

/// Runs a plan over a list of splits.
///
/// The plan runs as one stage under each split's selection.
///
/// Up to [`with_max_active`](Self::with_max_active) splits are compiled at once, in order, so
/// reads of later splits are in flight while earlier ones compute.
pub struct Scan {
    core: Core,
    root: Option<Root>,
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
    /// Validates split bounds without building a pipeline or issuing a read.
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
        let root = Root::Plan(plan);
        let mut core = Core::new(session, 0);
        core.dirty_flags = vec![false; 1];
        Ok(Self {
            core,
            root: Some(root),
            splits,
            next: 0,
            active: vec![None],
            live: 0,
            outputs: VecDeque::new(),
        })
    }

    /// The segments compiling `plan` over `rows` under `mask` reads, each once, in id order.
    /// Compiling does no IO, so the segments a stage will read are known before it starts.
    pub fn segments(
        session: &VortexSession,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Vec<SegmentId>> {
        let mut core = Core::new(session.clone(), 0);
        core.compile(plan, rows, mask)?;
        let mut segments = core.segments;
        segments.sort_unstable();
        segments.dedup();
        Ok(segments)
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

    /// Shares whole values, such as dictionary values, with the other scans given `values`.
    pub fn with_value_cache(mut self, values: ValueCache) -> Self {
        self.core.values = values;
        self
    }

    pub(crate) fn with_segment_cache(mut self, cache: Option<super::SegmentCache>) -> Self {
        self.core.segment_cache = cache;
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
            // Reads still in flight once every split has finished were cancelled; nothing waits
            // for their bytes.
            if self.live == 0 {
                return Ok(Turn::Done);
            }
            if !self.core.reads.is_empty() {
                return Ok(Turn::Waiting);
            }
            vortex_bail!(
                "Scan is stuck with nothing runnable and no read in flight: {}",
                self.core.describe_blocked()
            );
        }
    }

    /// Delivers the bytes of a read [`step`](Self::step) returned.
    pub fn deliver(&mut self, read: ReadId, bytes: BufferHandle) -> VortexResult<()> {
        self.core.deliver(read, bytes, true)
    }

    /// Delivers a private reconstruction whose unselected rows are not valid segment data.
    pub(crate) fn deliver_partial(
        &mut self,
        read: ReadId,
        bytes: BufferHandle,
    ) -> VortexResult<()> {
        self.core.deliver(read, bytes, false)
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
        let output = match self
            .root
            .as_ref()
            .ok_or_else(|| vortex_err!("Scan has no plan for its next split"))?
        {
            Root::Plan(plan) => {
                self.core
                    .compile_stage(plan, split.rows.clone(), &split.mask, slot)?
            }
        };
        if self.next == self.splits.len() {
            // Compiled operators own what they need. The last split no longer needs the
            // lowering tree or the source layouts retained by its lazy children.
            self.root = None;
        }
        if output.is_some() {
            self.active[slot] = Some(Active { index, output });
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
                self.outputs.push_back((active.index, batch));
            }
            self.core.read_output(port);
            if !self.core.arena.get(port).closed() {
                return Ok(());
            }
            self.core.arena.drop_source(port);
            active.output = None;
        }
        self.active[slot] = None;
        self.live -= 1;
        Ok(())
    }

    /// Ports in use.
    #[cfg(test)]
    pub(crate) fn live_ports(&self) -> usize {
        self.core.arena.live()
    }

    /// Reads in flight that a pipeline waits for.
    #[cfg(test)]
    pub(crate) fn awaited_reads(&self) -> usize {
        self.core
            .reads
            .values()
            .filter(|(owner, _)| owner.is_some())
            .count()
    }
}
