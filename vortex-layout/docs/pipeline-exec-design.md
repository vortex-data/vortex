<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Pipeline executor design

Design for replacing the node-based plan executor in `vortex-layout/src/plan/exec` with a
push-based pipeline executor. This is the state of the design discussion as of this
document; the "Open decisions" section lists what is not yet settled. Nothing here is
implemented.

## 1. Goals

- Execute a `PlanRef` for one split (a row range of one file) as a set of pipelines
  driven from a single thread, issuing IO through the owner rather than awaiting it.
- Make blocking explicit and local: a pipeline blocks on exactly one of IO, an empty
  inlet, or a full outlet, and the scheduler re-runs it when that condition clears.
- Make every batch flow through a port with one writer and one reader, so backpressure
  and lifetime are per port and need no global bookkeeping.
- Make sharing a plan-level fact (`Share` node) and a compile-level mechanism (one
  pipeline, many single-use ports), replacing `DecodeCache`, the per-plan `OnceLock`
  caches (`ZoneCache`, dictionary values), and any prerequisite/provide mechanism.
- Decide everything decidable from the layout at build time: which chunks exist, where
  masks are known, which zones are pruned. No gates, no deferred compilation.
- Keep the plan layer (`PlanRef`, rules, optimizer, lowering) as it is, apart from the
  `Share` node and the pass that inserts it.

## 2. Vocabulary

| Term | Meaning |
|---|---|
| Graph | The compiled form of one plan for one split: a table of pipelines plus the ports they use. One graph per (plan stage, split). |
| Pipeline | One `Source`, zero or more `Operator` stages, and one or more outlet ports. Runs in one thread, never concurrently with itself. |
| Source | The start of a pipeline. Reads from inlet ports and/or requests segments. The only thing that can block. |
| Operator | A stage inside a pipeline. Pure push: takes one input, produces zero or more outputs. Never blocks. |
| Port | A bounded or unbounded queue of `ArrayRef` with exactly one writer pipeline and exactly one reader source. Lives in the scan's arena. |
| Inlet | The reader-side view of a port, borrowed for one `compute` call. |
| Outlet | The writer-side view of a port. Only the driver holds one. |
| Driver | The function that runs one pipeline until it blocks, finishes, or has produced one unit of progress. |
| Scheduler | The loop over pipelines that picks what to drive and tracks why each pipeline is blocked. |
| Owner | The code that owns a graph: forwards segment reads to a `SegmentSource`, delivers bytes, drains the graph's output. The scan layer. |
| Split | A row range of a file handed to the scan by the morsel planner. One graph per split per stage. |
| Stage | One compile of part of a `Query`: zone pruning, one conjunct, or the projection. See §8. |
| Share | A plan node with one child and several parents. Compiled once; its pipeline fans out into one port per parent. See §9. |

## 3. Traits

```rust
/// What a stage is handed on each call.
pub enum Input {
    /// One batch from the stage below (or, for a source, never: sources get `None`).
    Chunk(ArrayRef),
    /// The stage below has finished. Flush whatever is held.
    End,
    /// Nothing new. Sent to a source on every call, and to a stage that returned `More`.
    None,
}

/// What a stage says back.
pub enum Step {
    /// A batch, and more is ready now without further input.
    /// The driver pushes the batch onward and calls again with `Input::None`.
    More(ArrayRef),
    /// A batch, and the input is spent. The driver pushes it onward and returns.
    Last(ArrayRef),
    /// The input was absorbed and nothing is ready. The driver returns.
    Consumed,
    /// Nothing can happen until the condition clears. Sources only.
    Blocked(Blocked),
    /// No more output will ever come. The driver sends `End` to the next stage.
    Finished,
}

pub enum Blocked {
    /// A read was requested (via `request`) and has not been delivered.
    Io,
    /// Inlet `i` is empty and its writer has not closed it.
    Inlet(usize),
    /// An outlet is full. Produced by the driver only, never by a stage.
    Outlet,
}

pub trait Operator: Send {
    fn compute(&mut self, input: Input, cx: &mut Cx<'_>) -> VortexResult<Step>;
}

/// The start of a pipeline.
pub trait Source: Operator {
    /// Number of inlet ports this source reads.
    fn inlet_count(&self) -> usize;

    /// How many batches inlet `inlet` may hold before its writer is `Blocked(Outlet)`.
    /// Asked once at build. An empty port always has room regardless of this value.
    fn capacity(&self, inlet: usize) -> usize {
        DEFAULT_CAPACITY
    }

    /// The next segment this source wants read. `None` when it needs nothing now.
    /// Called before every `compute`. A source that returns `Some` is blocked on IO
    /// until the segment is delivered; it is not computed in the same driver call.
    fn request(&mut self) -> Option<SegmentId>;
}

/// The terminal consumer of a whole graph. Exactly one per graph.
pub trait Sink: Send {
    fn consume(&mut self, batch: ArrayRef) -> VortexResult<()>;
    fn finish(&mut self) -> VortexResult<()>;
}
```

Contract notes:

- A stage returning `More` MUST eventually return something other than `More` on
  repeated `Input::None`. The driver loops on it.
- A stage returning `Finished` is not called again. The driver sends `End` to the next
  stage. A stage that receives `End` must return `Last`, `Consumed`, or `Finished`; it
  may return `More` to flush several batches, after which the driver sends `End` again.
  (`End` is idempotent for a stage: receiving it twice is the same as once.)
- Only a `Source` may return `Blocked`. The driver treats `Blocked` from a stage as a
  bug (`vortex_bail!`).
- `Consumed` from a source means "I did work but produced nothing" (for example it took
  a batch from an inlet and buffered it). The driver returns `Progress::Ran` and the
  scheduler may run it again immediately.
- `Input::None` to a source is the normal input. A source never receives `Chunk` or
  `End`.

### 3.1 `Cx`

```rust
/// Per-call view of what a stage may touch.
pub struct Cx<'a> {
    /// Bytes delivered for requests this source made, by segment id.
    /// A source takes its bytes out of here (`take(seg)`) when it computes.
    pub bytes: &'a mut Delivered,
    inlets: &'a mut [Queue],
    exec: &'a mut ExecutionCtx,
}

impl<'a> Cx<'a> {
    pub fn inlet(&mut self, i: usize) -> Inlet<'_>;
    pub fn exec(&mut self) -> &mut ExecutionCtx;
}

pub struct Delivered {
    /// Keyed by segment id. Removed when taken.
    ready: HashMap<SegmentId, ByteBuffer>,
}
impl Delivered {
    pub fn take(&mut self, seg: SegmentId) -> Option<ByteBuffer>;
}
```

`Delivered` is per pipeline (only the pipeline's source requested anything).

### 3.2 Ports, `Inlet`, `Outlet`

```rust
pub struct Queue {
    batches: VecDeque<ArrayRef>,
    /// `None` means unbounded (share outlets, §9).
    capacity: Option<usize>,
    closed: bool,
}

/// The reader side. Borrowed from the arena for one `compute` call.
pub struct Inlet<'a> { queue: &'a mut Queue }

impl Inlet<'_> {
    /// The oldest batch, in place, so a consumer can slice it down without removing it.
    pub fn peek_mut(&mut self) -> Option<&mut ArrayRef>;
    /// The oldest batch, removed. This is what frees the slot.
    pub fn take(&mut self) -> Option<ArrayRef>;
    /// The writer has finished. Once `closed` and empty, the inlet is at end.
    pub fn closed(&self) -> bool;
    pub fn is_empty(&self) -> bool;
}

/// The writer side. Only the driver holds one.
pub struct Outlet<'a> { queue: &'a mut Queue }

impl Outlet<'_> {
    /// True when `capacity` is `None`, or the queue is empty, or len < capacity.
    pub fn has_room(&self) -> bool;
    pub fn push(&mut self, batch: ArrayRef);
    pub fn close(&mut self);
}
```

Rules:

- Room is counted in batches. A batch that was `peek_mut`-sliced still occupies its
  slot until `take`. So a consumer that is part way through a batch holds its producer
  back, which is intended.
- The driver checks `has_room` on every outlet of a pipeline before running the source.
  The one batch (or `More` sequence) produced by that run may land in a port that is now
  full; that is allowed. So a port holds at most `capacity + 1` batches briefly.
  (See Open decision D3 for whether `More` sequences should be bounded here.)
- A consumer with a row budget (Filter, Pack) uses this pattern:

```rust
let Some(front) = cx.inlet(i).peek_mut() else {
    return Ok(Step::Blocked(Blocked::Inlet(i)));   // only valid in a Source
};
let piece = if front.len() <= budget {
    cx.inlet(i).take().vortex_expect("peeked")
} else {
    let rest = front.slice(budget..front.len())?;
    std::mem::replace(front, rest).slice(0..budget)?
};
```

- `Blocked(Inlet(i))` is returned only when the inlet is empty AND not closed. An empty
  closed inlet is end-of-stream for that inlet, which the source handles (usually by
  finishing, or for a join by finishing once all inlets are at end).

### 3.3 Arena

```rust
pub struct Arena {
    ports: SlotMap<PortId, Queue>,   // or Vec<Option<Queue>> with a free list
}
```

The arena is owned by the scan (one per file scan), not by the graph, so a port
written by a pipeline in one graph can be read by a pipeline in a graph compiled later
(a later Query stage, or a later split). A port is freed when it is closed, empty, and
its reader has been attached and has retired, or when its reader is released (§9.4).

## 4. Pipeline and driver

```rust
pub struct Pipeline {
    source: Box<dyn Source>,
    stages: Vec<Box<dyn Operator>>,
    /// Inlet ports read by `source`, index = inlet index.
    inlets: Vec<PortId>,
    /// Outlet ports written by the end of the pipeline. One for a plain pipeline,
    /// one per reader for a share pipeline (§9).
    outlets: Vec<PortId>,
    delivered: Delivered,
    state: State,
}

pub enum State { Runnable, Blocked(Blocked), Done }

pub enum Progress {
    /// The source wants this segment. The pipeline is now `Blocked(Io)`.
    Read(SegmentId),
    /// The pipeline did work.
    Ran,
    /// The pipeline could not do work (blocked on inlet or outlet).
    Idle,
    /// The pipeline is finished and its outlets are closed.
    Done,
}
```

Driver:

```rust
fn run(&mut self, p: PipelineId) -> VortexResult<Progress> {
    loop {
        if !self.all_outlets_have_room(p) {
            self.set(p, State::Blocked(Blocked::Outlet));
            return Ok(Progress::Idle);
        }
        if let Some(seg) = self.source_mut(p).request() {
            self.set(p, State::Blocked(Blocked::Io));
            return Ok(Progress::Read(seg));
        }
        let step = {
            let (pipe, cx) = self.split_borrow(p);           // pipeline + Cx over arena
            pipe.source.compute(Input::None, &mut cx)?
        };
        match step {
            Step::More(b)    => { self.push(p, 0, b)?; /* loop */ }
            Step::Last(b)    => { self.push(p, 0, b)?; return Ok(Progress::Ran); }
            Step::Consumed   => return Ok(Progress::Ran),
            Step::Blocked(w) => { self.set(p, State::Blocked(w)); return Ok(Progress::Idle); }
            Step::Finished   => { self.push_end(p, 0)?; self.set(p, State::Done); return Ok(Progress::Done); }
        }
    }
}

/// Pushes one batch into stage `i`, and whatever it emits into `i + 1`, and so on,
/// until the batch reaches the outlets.
fn push(&mut self, p: PipelineId, i: usize, batch: ArrayRef) -> VortexResult<()> {
    if i == self.stage_count(p) {
        return self.push_out(p, batch);
    }
    let mut input = Input::Chunk(batch);
    loop {
        let step = self.stage_compute(p, i, input)?;
        match step {
            Step::More(b)    => { self.push(p, i + 1, b)?; input = Input::None; }
            Step::Last(b)    => return self.push(p, i + 1, b),
            Step::Consumed   => return Ok(()),
            Step::Finished   => return self.push_end(p, i + 1),
            Step::Blocked(_) => vortex_bail!("stages do not block"),
        }
    }
}

/// Sends `End` into stage `i` and onward. Closes the outlets at the end.
fn push_end(&mut self, p: PipelineId, i: usize) -> VortexResult<()> {
    if i == self.stage_count(p) {
        for &port in &self.pipelines[p].outlets { self.arena.outlet(port).close(); }
        return Ok(());
    }
    let mut input = Input::End;
    loop {
        match self.stage_compute(p, i, input)? {
            Step::More(b)  => { self.push(p, i + 1, b)?; input = Input::End; }
            Step::Last(b)  => { self.push(p, i + 1, b)?; return self.push_end(p, i + 1); }
            Step::Consumed | Step::Finished => return self.push_end(p, i + 1),
            Step::Blocked(_) => vortex_bail!("stages do not block"),
        }
    }
}

fn push_out(&mut self, p: PipelineId, batch: ArrayRef) -> VortexResult<()> {
    let outlets = &self.pipelines[p].outlets;
    match outlets.len() {
        1 => self.arena.outlet(outlets[0]).push(batch),
        _ => for &port in outlets { self.arena.outlet(port).push(batch.clone()); }
    }
    Ok(())
}
```

Borrowing: `split_borrow` must hand out `&mut Pipeline` and a `Cx` over the arena at the
same time. The pipeline table and the arena are separate fields of the graph, so this is
a plain disjoint borrow. `Cx.inlets` is a slice of `&mut Queue` gathered from the
arena by the pipeline's `inlets` ids (collect into a `SmallVec<[&mut Queue; 4]>` via
`get_disjoint_mut` or an index-sorted split). The first implementation can take the
inlets out of the arena (`std::mem::take`) for the duration of the call and put them
back; measure before optimising.

## 5. Scheduler

```rust
pub struct Graph {
    pipelines: Vec<Pipeline>,
    /// Pipelines whose state is `Runnable`, LIFO (the most recently unblocked runs
    /// first, which keeps data moving toward the output).
    ready: Vec<PipelineId>,
    /// For each port, the pipeline that writes it and the pipeline that reads it,
    /// so a take can unblock the writer and a push can unblock the reader.
    port_writer: HashMap<PortId, PipelineId>,
    port_reader: HashMap<PortId, PipelineId>,
    output: PortId,
}

pub enum Turn {
    /// The owner must read this segment and call `deliver`.
    Read(SegmentId),
    /// A batch is available on the output port.
    Output(ArrayRef),
    /// Nothing is runnable and no read is outstanding: the graph is finished.
    Finished,
    /// Nothing is runnable but reads are outstanding: wait for `deliver`.
    WaitingForIo,
}

impl Graph {
    /// Runs pipelines until something the owner must act on happens.
    pub fn step(&mut self) -> VortexResult<Turn> {
        loop {
            if let Some(b) = self.arena.inlet(self.output).take() {
                self.unblock_writer(self.output);
                return Ok(Turn::Output(b));
            }
            let Some(p) = self.ready.pop() else {
                return Ok(if self.outstanding_reads > 0 { Turn::WaitingForIo }
                          else if self.all_done() { Turn::Finished }
                          else { vortex_bail!("graph is deadlocked: {}", self.describe_blocked()) });
            };
            match self.run(p)? {
                Progress::Read(seg) => { self.outstanding_reads += 1; return Ok(Turn::Read(seg)); }
                Progress::Ran => {
                    // After a run: every outlet this pipeline pushed into may have
                    // woken its reader; every inlet it took from may have woken its writer.
                    self.wake_neighbours(p);
                    self.ready.push(p);            // it is still runnable
                }
                Progress::Idle => { /* state already set to Blocked */ self.wake_neighbours(p); }
                Progress::Done => { self.wake_neighbours(p); }
            }
        }
    }

    pub fn deliver(&mut self, seg: SegmentId, bytes: ByteBuffer) {
        let p = self.requester[seg];
        self.pipelines[p].delivered.ready.insert(seg, bytes);
        self.outstanding_reads -= 1;
        self.make_runnable(p);
    }
}
```

Wake rules (`wake_neighbours`):

- A pipeline in `Blocked(Inlet(i))` becomes runnable when its inlet `i` port receives
  a push or is closed.
- A pipeline in `Blocked(Outlet)` becomes runnable when any of its outlet ports has a
  batch taken (room appeared).
- A pipeline in `Blocked(Io)` becomes runnable on `deliver` of its requested segment.

Implementation: after a run of `p`, iterate `p`'s outlets and inlets, look up the
neighbour pipeline, and if its blocked reason matches, move it to `ready`. This is
O(ports of p) per run, no global scan. Avoid pushing a pipeline onto `ready` twice
(keep a `queued: bool` on each pipeline).

Deadlock: with bounded ports and single-reader semantics the only deadlock is a
diamond into a join with bounded ports on both arms (§9.3). Share outlets are unbounded
precisely to rule it out. The `vortex_bail!` above is the backstop and should name
every blocked pipeline and reason.

The owner loop (scan layer), for one graph:

```rust
loop {
    match graph.step()? {
        Turn::Read(seg) => segment_source.request(seg, /* callback → */ graph.deliver(seg, bytes)),
        Turn::Output(b) => sink.consume(b)?,
        Turn::WaitingForIo => await next delivery,
        Turn::Finished => { sink.finish()?; break; }
    }
}
```

The owner may run several graphs (splits) at once and issue many reads before any is
delivered: reads are issued one per `step` call per pipeline, but `step` keeps running
other pipelines, so a graph with N scan pipelines will issue up to N reads before
blocking on IO. In-flight depth across splits is the owner's policy (today 1 in memory,
4 × cores on disk in the benchmarks).

## 6. Masks, filters, and pruning

### 6.1 Principle

Sources read row ranges, never masks. A mask is data: a stream of `Bool` batches on a
port. The only operator that consumes a mask is `Filter`. The compile places `Filter`
where it is cheapest (as low as the plan's `Filter` node, after `ExpressionZonedRule`
and friends), and it is never pushed below a `SegmentScan`.

### 6.2 `Filter` as a join source

`Filter` is a `Source` with two inlets: data (0) and mask (1). It joins by row position:
it holds a cursor into each stream and emits data rows where the mask is true for the
same row positions. Both streams are over the same row range by construction.

```rust
struct Filter {
    data_pending: Option<ArrayRef>,   // left over after a partial apply
    mask_pending: Option<Mask>,
}
impl Source for Filter {
    fn inlet_count(&self) -> usize { 2 }
    fn request(&mut self) -> Option<SegmentId> { None }
}
impl Operator for Filter {
    fn compute(&mut self, _: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        // 1. Ensure a data batch: pending, or peek inlet 0, else Blocked(Inlet(0)),
        //    unless inlet 0 is closed → Finished.
        // 2. Ensure a mask batch covering at least one row of it: pending, or peek
        //    inlet 1, else Blocked(Inlet(1)).
        // 3. Take n = min(data.len(), mask.len()) rows from each (slice in place via
        //    peek_mut / take as in §3.2).
        // 4. If mask slice is all-true emit data slice; all-false emit nothing
        //    (Consumed) ; else `data.filter(mask)` — the per-batch dense/sparse choice
        //    lives inside the array kernel, not here.
        // 5. Return Last(batch) (more input is needed for the next), or More if both
        //    inlets still have whole batches queued (optional optimisation).
    }
}
```

A `Filter` may also be used as a stage (an `Operator`, not a `Source`) when its mask is
fully known at build and is given to it as a `Mask` in the constructor. This is the
form used for known masks that are small; see 6.3.

### 6.3 Known masks

A mask known at build (the owner's row selection for the split, a zone proof over the
split's zones, the result mask of an earlier Query stage) is written into a port at
compile time: the port is created, the batches pushed, and the port closed. The reader
pipeline sees an ordinary closed inlet. No special "prefilled" port type exists.

For very small known masks the compile may instead construct `Filter` as a stage with
the `Mask` inline. This is an optimisation and not required.

### 6.4 `And`

`And` is a `Source` with N inlets of `Bool` streams, joined by row position, emitting the
conjunction. Used when a `Filter` needs the intersection of several masks (for example
the owner selection and a zone proof). An `And` with one inlet is not built.

### 6.5 Zone pruning at build

`ExpressionZonedRule` already turns `Filter(pred, zoned_column)` into a `Zoned` plan with
a prune proof. In the pipeline model:

- The zone table for a column is a `Share` (§9) because every split reads it. Its
  pipeline is one `SegmentScan` chain.
- A proof (`pred` evaluated against the zone table into one bit per zone) is a `Share`
  parented by every split's Query stage zero. (Open decision D2: whether the proof is
  its own Share or stage zero computes it from the table's Share port each time.)
- At compile of a split, the compile reads the proof port if it is already closed. For
  each chunk of the data column intersecting the split: if every zone of the chunk is
  pruned the chunk's pipeline is not built; if no zone is pruned no mask is attached;
  otherwise a mask port is created from the proof bits expanded over the chunk's rows
  and a `Filter` joins it. If the proof port is not yet closed (first split), the
  compile cannot prune at build, so it builds all chunks with a `Filter` whose mask
  inlet is fed by an `Expand(proof → rows)` pipeline reading the proof port. Both forms
  give identical output; the first is cheaper.

### 6.6 List columns

`ListPack(elements, offsets)` under a mask has two forms chosen in the plan, not at
runtime:

- `Filter(elements, Expand(mask, offsets))`: expand the row mask to an element mask using
  the offsets, filter elements, then pack. Reads all element chunks.
- `Take(elements, offsets_of_selected)`: read elements by offset ranges. Only when the
  selection is sparse enough that the offsets identify few ranges.

`Expand` is an `Operator` stage with the offsets given at construction (offsets are
small and read first via their own pipeline whose port the compile drains before
building the element pipelines, or passed as a known array). The choice is a plan rule
and is the same as today's `ExpressionListPackRule` territory.

### 6.7 Dictionaries

`Take(codes, values)`: `values` is a `Share` (whole-file). The pipeline is
`SegmentScan(codes) → TakeStage(values_port)`. `TakeStage` is an `Operator` constructed
with the values `ArrayRef`, which the compile obtains by draining the values share port
(it is closed by the time any split that needs it compiles, except the first; see §9.5
for the first-reader case).

## 7. Sources and operators to implement

| Name | Kind | Inlets | Notes |
|---|---|---|---|
| `SegmentScan` | Source | 0 | `request` returns its segment id once; `compute` takes bytes from `cx.bytes`, decodes to the array for its row range, returns `Last` then `Finished`. Slicing to the split's sub-range is done here. |
| `PortSource` | Source | 1 | Passes batches from an inlet through (`take` → `Last`). Used to start a pipeline on a port written by another pipeline (for example a Share reader). |
| `Filter` | Source | 2 | §6.2. |
| `And` | Source | N | §6.4. |
| `Pack` | Source | N | Struct join: one inlet per field, aligned by row position, emits `StructArray` of min(available) rows using the §3.2 slicing pattern. Capacity per inlet is 2 by default (see D3). Finishes when all inlets are closed and empty. |
| `Concat` | Source | N | Takes inlets in order: drains inlet 0 to end, then 1, and so on. Capacity 1 per inlet except the current one; this stalls later chunks' producers until needed, which bounds memory. (Alternative: all chunks run freely; measure.) |
| `Eval` | Operator | – | Applies an expression to each batch: `Chunk(b)` → `Last(expr(b))`. |
| `Expand` | Operator | – | Row mask → element mask via offsets (§6.6). |
| `TakeStage` | Operator | – | Codes → values (§6.7). |
| `Slice` | Operator | – | Narrows batches to a sub-range; used when a shared decoded segment is wider than the reader's range (§9.2). |
| `ToBits` | Operator | – | Executes a lazy `Bool` batch to a `Mask`-backed `BoolArray` (used by Query stages so no lazy array outlives its segment). |

Every plan node in `plan/plans/*` maps to one of these at compile: `SegmentScan`→
`SegmentScan`; `Concat`→`Concat`; `Eval`→`Eval` stage on its child's pipeline;
`Filter`→`Filter` source with a mask pipeline; `Pack`→`Pack`; `ListPack`→ §6.6;
`Take`→`TakeStage`; `RowIdx`→ a stage that emits row indices for the batch's range (or a
source with 0 inlets emitting the range in chunks); `Zoned`→ §6.5; `Query`→ §8;
`Share`→ §9.

Compile fuses a chain `Source → Eval → Eval → …` into one pipeline. A new pipeline
starts at every node with more than one child (Pack, Filter, And, Concat) and at every
`Share` child.

## 8. Query as the stage planner

`QueryPlan` holds the projection, the conjuncts, the `FilterExpr` scheduler and the
dense threshold, as today. At execution it is not a node; it is a loop in the owner that
compiles and runs one graph per stage for one split.

```rust
pub struct QueryRun {
    plan: QueryPlan,
    split: Range<u64>,
    mask: Mask,                 // rows still selected, over `split`
    remaining: BitVec,          // conjuncts not yet evaluated
}

impl QueryRun {
    /// Returns the next stage's graph and what to do with its output, or `None` when
    /// the projection has run.
    fn next_stage(&mut self, scan: &mut Scan) -> VortexResult<Option<(Graph, StageKind)>>;
}

enum StageKind { Prune, Conjunct(usize), Projection }
```

Stage order:

1. **Prune.** For each conjunct that has a zone proof (`plan.pruning(i)`), intersect the
   proof expanded over the split with `mask`. If the proof port for a conjunct is not yet
   closed (first split of the file), build a graph whose single pipeline reads the zone
   table share port, evaluates the predicate against it, and writes the proof port; run
   it to completion, then intersect. No data is read in this stage.
2. **Conjuncts.** While `remaining` is non-empty and `mask` is not all-false: ask the
   scheduler for the next conjunct; compile its plan over `split` with `mask` as a known
   mask port (§6.3); run it; fold its output batches into a `BitBufferMut` as they arrive
   (the `ToBits` stage makes each batch a plain bool buffer so nothing lazy survives);
   intersect by rank into `mask`; report selectivity.
3. **Projection.** Compile the projection over `split` with the final `mask`; its output
   port is the split's output.

Dense versus sparse (today's `dense_threshold` and `chunks_with_selected_rows`): see
Open decision D1. Until decided, implement the simple form: the conjunct's graph is
compiled with the current `mask` as its known mask, and `Filter` placement decides per
batch.

Remaining-reader accounting for shares (§9.4): when a conjunct stage compiles a `Share`
that is also referenced by unevaluated conjuncts or the projection, those readers' ports
are created now (so the share's pipeline fans out into them), and are claimed when the
later stage compiles. If the mask goes all-false and the later stages are never compiled,
`QueryRun` drops those ports (`arena.release(port)`).

## 9. Sharing

### 9.1 Plan node

```rust
pub struct Share {
    child: PlanRef,
    id: ShareId,   // stable within a plan; used for display, serialization, and port lookup
}
```

One child, many parents. The only DAG node in a plan. It carries no scope, count, or
cache policy.

**Insertion pass** (`ShareRule`, runs after `optimize`): walk the optimized tree,
counting parents per `PlanRef` by Arc pointer (`as_ptr_key`). Every subtree with more
than one parent is wrapped once in a `Share` and every parent is rewired to the same
`Share` Arc. Walk bottom-up so a shared subtree inside a shared subtree becomes a nested
`Share`. Subtrees that are per-file by construction (zone tables, dictionary values,
proofs) are shared because the layout hands out the same Arc for them to every plan
built from that layout; the pass needs no special knowledge of them. Equality is pointer
equality only: two structurally equal subtrees that the memo did not unify are not
shared, and that is acceptable.

**Display:** first occurrence expanded as `share#<id>: vortex.plan.share(dtype, rows)`
with its subtree; later parents print `share#<id>` only.

**Serialization:** a `shares` table at the top of the plan message, indexed by id; a
parent references an entry by index. Deserialization rebuilds one Arc per entry so
pointer identity survives a round trip.

### 9.2 Compile

The compile keeps a scan-owned map `ShareId → SharePorts`:

```rust
struct SharePorts {
    /// Ports created for this share, one per expected reader, in the order readers
    /// are expected to claim them. `None` once claimed.
    unclaimed: Vec<Option<PortId>>,
    /// The pipeline that writes them, if built in a graph still running.
    writer: Option<(GraphId, PipelineId)>,
}
```

When the compile reaches a `Share`:

1. If `ShareId` is not in the map: compute `expected` (§9.4), create `expected` ports
   with `capacity: None`, build the child's pipeline(s) with `outlets = all those ports`,
   record them. Then fall through to 2.
2. Claim the next unclaimed port for this reader. The reader pipeline starts with a
   `PortSource` on that port (fused with whatever stages the parent needs, for example
   `Slice` when the parent's row range is narrower than the share's, then `Filter` when
   the parent has a mask).

The reader never knows whether the writer is in its own graph, finished three stages ago,
or in another split's graph. All three look like a port with batches in it.

### 9.3 Fan-out and capacity

A share pipeline has `outlets.len() == expected`. `push_out` clones the `ArrayRef` into
each outlet. The data exists once; the ports hold references.

Share outlets are **unbounded** (`capacity: None`). Reasons:

- With bounded outlets a slow reader stalls the writer, which stalls every other reader.
- With bounded outlets a diamond (share → A, share → B, A and B → Pack with capacity 1)
  deadlocks: B blocked on Pack's inlet 1, so B's share port fills, so the share blocks, so
  A gets nothing, so Pack never sees inlet 0.

Cost: readers may diverge by up to the share's whole output. Bounded by split size for
in-split readers; whole-value for cross-split shares, which are small (zone tables,
dictionary values) or already decided to be kept (segments wider than a split).

### 9.4 Expected readers and release

`expected` for a share, at the time the first reader compiles it, is:

```
sum over splits S that overlap the share's row range (this split and later ones) of
    number of parents of the Share in the plan that will be compiled for S
```

where "parents that will be compiled for S" counts a Query's stages as separate
parents: a Share referenced by conjuncts {i, j} and the projection contributes 3 per
split. The scan knows the split list (see D4) and the plan knows its parents, so this is
exact.

Release: a reader that will never claim its port releases it:

- `QueryRun` releases the ports of stages it will not run (mask went all-false, or the
  conjunct scheduler was told to stop).
- The scan releases the ports of splits it truncates (`scan.truncate()`), by walking the
  shares those splits would have read.
- A split pruned to nothing at build still compiles far enough to claim and release.

Freeing: a port is freed when it is closed, empty, and claimed-and-retired or released.
A share's pipeline is freed with its graph. Nothing else is tracked.

### 9.5 First reader of a cross-split share

The first split to compile a file-wide share (zone table, dictionary values, proof) builds
its pipeline in that split's graph. Readers that need the whole value before they can
build anything (zone pruning at build, `TakeStage` construction) handle the not-yet-closed
port in one of two ways:

- Prune: build without pruning, with a streamed `Filter` fed by `Expand(proof port)`
  (§6.5). Correct, slower, only for the first split.
- `TakeStage`: build a `PortSource(values port) → Collect` pipeline and make the codes'
  pipeline's `TakeStage` take the collected values from a shared `OnceCell` the compile
  created. Or simpler: the compile runs a mini-graph for the values share to completion
  before building the split (one extra `step` loop at compile). The simpler form is
  acceptable for the first implementation; dictionary values are small.

### 9.6 What Share replaces

| Today | Replaced by |
|---|---|
| `DecodeCache` with generation window | Share ports for segments wider than a split. |
| `ZoneCache.zone_map` (`OnceLock`) | Zone table Share. |
| `ZoneCache.proofs` | Proof Share (or stage-zero recompute, D2). |
| Dictionary values re-read per split | Values Share. |
| Column decoded once for a conjunct and once for the projection | In-plan Share with a port per stage. |
| `ZonePruneNode` expanding proofs at runtime | Pruning at build from the proof port; `Expand` only for the first split. |

## 10. Scan integration

```rust
pub struct Scan {
    arena: Arena,
    shares: HashMap<ShareId, SharePorts>,
    splits: Vec<Range<u64>>,        // see D4
    plan: PlanRef,                  // optimized, with Shares inserted
    graphs: Vec<(GraphId, SplitId, Graph)>,
    segment_source: Arc<dyn SegmentSource>,
}

impl Scan {
    pub fn add_split(&mut self, rows: Range<u64>) -> SplitId;
    /// Compiles and runs the Query stages for a split, in order, driving IO through
    /// `segment_source`, delivering output to `sink`.
    pub fn run_split(&mut self, split: SplitId, sink: &mut dyn Sink) -> VortexResult<()>;
    /// Drops every split not yet started and releases their share ports.
    pub fn truncate(&mut self);
}
```

`run_split` is the owner loop of §5 wrapped in the stage loop of §8. The scan layer
(`vortex-scan`) calls `add_split` from its morsel planner and `run_split` from its
executor; several `run_split`s may be in flight if the owner drives them with an
`FuturesUnordered` as the benchmarks do today. Since the arena is shared across splits
and the driver is single-threaded, concurrent splits run on one thread interleaved, or
the arena is behind a lock taken per `step`. Start with one thread per scan and
interleaving; measure before adding locking.

## 11. Open decisions

- **D1. Dense vs sparse conjunct reads.** Today a conjunct under a dense mask reads whole
  chunks (`chunks_with_selected_rows`) and one under a sparse mask reads selected rows
  only. With shares, a sparse read leaves sparse batches in the later readers' ports, so
  the projection must re-rank its mask against the stored selection. Options: (a) keep
  per-stage choice and store the selection with the share so later readers re-rank
  (needs a `Selection` alongside each share's batches); (b) always read whole chunks
  when any later reader of the share exists, so every later `Filter` is positional.
  Recommendation: (b). Cost is extra decode on very selective filters, which the dense
  threshold sweep showed is small on compressed data.
- **D2. Proofs as Shares.** Either a proof is its own `Share` (cached file-wide by the
  same mechanism) or stage zero recomputes it from the zone table share each split
  (cheap: one predicate over a few hundred zones). Recommendation: own Share, so the
  mechanism is uniform and the plan shows it.
- **D3. Default capacity.** Capacity 1 on ordinary ports gives lockstep: the next
  segment read is only issued when the consumer has taken the previous batch. Capacity 2
  lets the next read overlap. `More` sequences can overshoot capacity by one run's worth
  of batches. Recommendation: `DEFAULT_CAPACITY = 2`, `Pack` and `Filter` use the
  default, `Concat` uses 1 on non-current inlets.
- **D4. Lazy splits.** `expected` needs the split list. If splits can be added after a
  share's pipeline is built, its ports cannot all be created up front. Options: (a)
  require the full split list before `run_split`; (b) allow `add_split` after start, and
  have a share whose readers may still arrive keep one extra "future readers" port that
  is cloned into a new port on each late `add_split` (batches copied by reference) and
  freed on `truncate`/end-of-splits. Recommendation: (a) for the first implementation;
  the scan layer plans morsels up front today.
- **D5. Memory arbitration.** No global bound on port memory. Acceptable for now;
  revisit with a per-scan budget that lowers capacities or pauses `SegmentScan` requests.
- **D6. Concurrency.** One thread per scan, splits interleaved on it, versus one thread
  per split with the arena locked. Not decided; start with the former.

## 12. Implementation order

1. `Queue`, `Inlet`, `Outlet`, `Arena`, `Pipeline`, `Graph::step`, `deliver`, wake rules,
   with `SegmentScan`, `Eval`, `PortSource`, `Pack`, `Concat`. Test with a hand-built
   graph against `TestSegments`.
2. `Filter`, `And`, known-mask ports, `ToBits`. Test filter join alignment with ragged
   batch sizes on both inlets.
3. Compile from `PlanRef` for plans without `Query`, `Zoned`, `ListPack`, `Take`, `Share`.
   Replace `exec::execute_stream` behind the same signature so `vortex-file` tests run.
4. `Share` node, `ShareRule`, display, proto, compile with fan-out ports, release. Test
   the diamond into `Pack`, and a share read by two splits.
5. `QueryRun` stages with shares across stages. Port the Query tests from
   `exec/tests.rs`.
6. `Zoned` at build, proof shares, first-split `Expand` path. Port the zoned tests.
7. `ListPack`, `Take` with values share.
8. Scan integration: `Scan`, `add_split`, `run_split`, `truncate`; wire `vortex-file`'s
   driver and the `query_exec` benchmark; delete `DecodeCache`, `ZoneCache`,
   `ZonePruneNode`, `QueryNode`.
9. Benchmark against V1 and the current V2 with `query_profile` (interleaved A/B) and
   `query_exec`; profile with callgrind on `run_exec`.
