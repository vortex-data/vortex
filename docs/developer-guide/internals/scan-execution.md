# Scan execution

This page is for implementers who want to add a plan operator, a planner, or a driver to the V2
scan, or assemble a scan of their own from those parts. For what a plan is and how it is
optimized, see [Scan plans](scan-planning.md).

A V2 scan has two levels, and each one composes:

| Level | Composes | Types |
| --- | --- | --- |
| Planner | **work**: what runs, in what order, over which rows | `Planner`, `Morsel` |
| Plan | **computation**: how values are produced from segments | `PlanVTable`, `Operator`, `Source` |

A planner does its work by running plans. The two levels do not map one to one. For example, a
scan of two files can use one planner per file, each with its own plan, or one planner running a
`Concat` of both files' plans. A **driver** steps planners and serves their reads. It is the only
part that decides which thread runs anything and how IO is awaited.

```text
Layout ──new_plan──► Plan ──optimize──► Plan
                                          │ compile, per split
                                          ▼
                         Source → Operator → … → port      (pipelines)
                                          ▲
                                          │ run by
          Planner ──emits──► Planner / Morsel ──► arrays
             ▲
             │ stepped by
          Driver (Run) ──► IoSource
```

Nothing in any of these layers blocks or awaits. Each one returns what it needs next, and the layer
above decides when to call it again.

## Plans

A plan operator is a `PlanVTable` implementation (`vortex_layout::plan::physical`):

| Item | Contract |
| --- | --- |
| `type PlanData` | Operator state. Children never live here; they live in slots so rewrites can find them. |
| `type Metadata` | The serialized form of `PlanData`. |
| `id` | A globally unique `PlanId`. |
| `metadata` | The serializable metadata, or `None` while the operator holds state that cannot be serialized. |
| `with_slots` | Validates replaced slots and refreshes derived data, such as `Concat`'s row offsets. |
| `compile` | Compiles the plan over `rows`, restricted to `mask`, into a pipeline `Chain`, or `None` when `mask` selects no row. Does no IO. |

`compile` either extends a child's chain with stages of its own or joins several children's chains
behind a new source. `Eval` is the simplest case: compile the child, then append a stage that
applies the expression.

```rust
fn compile(
    plan: &Plan<Self>,
    rows: Range<u64>,
    mask: &Mask,
    compiler: &mut Compiler<'_>,
) -> VortexResult<Option<Chain>> {
    let chain = compiler.compile(&plan.child_plan()?, rows, mask)?;
    Ok(chain.map(|chain| chain.with(EvalStage::new(plan.expression().clone()))))
}
```

An operator that combines children calls `compiler.join(chains, source)`. Inlet `i` of `source`
reads the `i`-th present slot.

Layouts build their initial plan in `VTable::new_plan`. The default returns an error, which means
the layout cannot be scanned by V2. `optimize(plan, session)` applies the rewrite rules. To push an
expression through a new operator, register a partition rule for the (expression, plan) pair in the
session's `PartitionSession`.

## Operators and pipelines

A compiled plan is a set of pipelines: one `Source`, zero or more `Operator` stages, and the ports
its last stage writes. A port connects one pipeline's sink to an inlet of another pipeline's
source.

- `Operator::compute(input, cx) -> Step` receives `Input::Batch`, `Input::None` (continue after
  `Step::More`), or `Input::End` (the stage below has finished). It returns `Step::More(array)`
  (call again before new input), `Step::Final(array)` (done with this input), or `Step::End`.
- `Source::ready(cx) -> SourceReady` runs before each source `compute`. Queue segment reads with
  `cx.request_read` and return `Blocked(SourceBlocked::Io)`, or return `Blocked(Inlet(i))` to wait
  on a child pipeline. Delivered bytes are available from `cx.take_segment`.

A pipeline that is blocked on a read, an empty inlet or a full outlet is parked and re-run when that
condition clears. Masks are known at compile time, so a stage that reads rows applies the mask
itself and unselected rows are never decoded.

`pipeline::Scan` runs a plan over a list of `Split`s (a row range and a mask) without doing IO
itself:

```rust
let mut scan = Scan::try_new(session, plan, vec![Split { rows, mask }])?;
loop {
    match scan.step()? {
        Turn::Read(read) => { /* fetch read.segment_id, then scan.deliver(read.id, bytes)? */ }
        Turn::Output(_split, array) => { /* rows of the split, in row order */ }
        Turn::Waiting => { /* nothing runs until a read is delivered */ }
        Turn::Done => break,
    }
}
```

## Planners and morsels

Planners and morsels are the protocol in `vortex_scan::planning`. Both are `Send` state machines
that also implement `IoConsumer`.

```rust
pub trait Planner: IoConsumer + Send {
    fn state(&self) -> State;
    fn compute(&mut self) -> VortexResult<PlannerOutput>;
}

pub trait Morsel: IoConsumer + Send {
    fn state(&self) -> State;
    fn compute(&mut self) -> VortexResult<MorselOutput>;
}
```

Rules for both:

- `state` is cheap and has no side effects: `NeedsCompute`, `Waiting`, or `Done`. Report `Waiting`
  only while one of your fetches is outstanding.
- `compute` is called only when `state` is `NeedsCompute`, and does one step.
- `set_io_result` stores the delivered result. It never computes.
- Every request is published once, from `compute`, as `NeedsIO(batch)`. `Fetch` requests are
  delivered; `Prefetch` and `Announce` are hints and need no delivery.

A planner decides which work is needed. Each `compute` returns one `PlannerOutput`:

| Output | Meaning |
| --- | --- |
| `Continue` | A CPU checkpoint; requeue me. |
| `NeedsIO(batch)` | Requests to publish. |
| `Planner(scope, child)` | A child planner, run by the same run before this planner continues. |
| `Spawn(scope, child)` | An independent child. The run hands it to its owner, who may run it anywhere. |
| `Morsel(scope, morsel)` | Work that produces arrays for `scope`. |
| `Done` | Finished. |

A morsel produces the arrays: `MorselOutput::Batch(array)` (never empty, in row order),
`NeedsIO`, `Continue`, or `Done`.

Stages hand over to their successor through `Next<Input>`, which is built with `next_fn`. A
constructor only stores its input; the successor does its planning in its own `compute`.

The built-in V2 scan composes these planners over a `PreparedScan`:

```text
PrunePlanner ──kept rows──► ChunkPlanner ──per chunk──► FilterPlanner ──surviving rows──► ProjectionPlanner ──► morsels
```

`PrunePlanner::new(scan, rows, next)` and `ChunkPlanner::new(scan, rows, selection, kept)` are
public, so a driver can compose them or put its own planners between them.

## Drivers

A `Run` (`vortex_scan::planning::driver`) steps any number of root planners, each reading through
its own `IoSource`:

| Method | Use |
| --- | --- |
| `admit(root, scope, io) -> RootId` | Add a root planner. |
| `advance() -> Progress` | Take one step. |
| `complete(completion)` | Hand back a completed read taken from the source. |
| `poll_completion(cx)` | Await the next completed read. |
| `cancel(root)` | Drop a root and everything under it. |

`advance` returns `Batch`, `Spawn(scope, planner)`, `RootDone(id)`, `Waiting` (every live item
waits on IO), or `Idle`. A minimal blocking driver handles each one. This is what `Driver::run`
does:

```rust
let mut run = Run::new();
run.admit(root, WorkScope { file_ordinal: 0, rows: 0..0 }, Arc::clone(&io));
let mut batches = Vec::new();
loop {
    match run.advance()? {
        Progress::Batch(batch) => batches.push(batch),
        // Run independent work here as another root, or send it to another thread's run.
        Progress::Spawn(scope, child) => { run.admit(child, scope, Arc::clone(&io)); }
        Progress::RootDone(_) if run.live_roots() > 0 => {}
        Progress::RootDone(_) | Progress::Idle => break,
        // Block for the next read. An async driver awaits `run.poll_completion` instead.
        Progress::Waiting => run.complete(io.wait()?)?,
    }
}
```

The engine drivers differ only in where they put these steps:

- `vortex-file`'s `ScanDriver` steps the root run on `spawn_cpu` tasks and runs each spawned chunk
  on a task of its own.
- DataFusion's driver does the same and converts batches to Arrow on the producing task.
- DuckDB's driver cuts each file into parts and runs one `Run` per DuckDB thread, inline.

## Example: a minimal scan

This sketch builds a filtered scan from the parts above, without using `PreparedScan`. It is meant
to show how the pieces fit together, and it leaves out pruning, chunking, caching and most error
handling.

**1. Plans.** Lower the layout once and derive a filter plan and a projection plan from it:

```rust
let data = layout.new_plan()?;
let filter = optimize(EvalPlan::try_new(predicate.bind(layout.dtype())?, data.clone())?.into(), &session)?;
let projection = optimize(EvalPlan::try_new(projection.bind(layout.dtype())?, data)?.into(), &session)?;
```

**2. Running a plan inside a planner or morsel.** Wrap a `pipeline::Scan` so its reads become
protocol fetches. The built-in scan does the same thing in `ScanAdaptor`:

```rust
struct PlanRun {
    scan: Scan,
    locations: Arc<[SegmentLocation]>,
    // Request id -> the scan's read id.
    outstanding: HashMap<u32, ReadId>,
    next_id: u32,
}

impl PlanRun {
    fn step(&mut self) -> VortexResult<Turn> {
        self.scan.step()
    }

    fn fetch(&mut self, read: ReadRequest) -> IoRequest {
        let loc = self.locations[*read.segment_id as usize];
        let id = self.next_id;
        self.next_id += 1;
        self.outstanding.insert(id, read.id);
        IoRequest {
            intent: IoIntent::Fetch,
            request: IoRequestId(id),
            target: IoTarget::Range { offset: loc.offset, len: loc.length as usize, alignment: loc.alignment },
        }
    }

    fn deliver(&mut self, id: IoRequestId, result: IoResult) -> VortexResult<()> {
        let IoResult::Bytes(bytes) = result else { vortex_bail!("expected bytes") };
        let read = self.outstanding.remove(&id.0).vortex_expect("delivered read was requested");
        self.scan.deliver(read, bytes)
    }
}
```

**3. A filter planner.** The planner runs the filter plan itself, because it needs the result to
decide what comes next. It then emits one projection morsel for the rows that survived:

```rust
struct SimpleFilterPlanner {
    filter: PlanRun,         // the filter plan over Split::all(rows)
    rows: Range<u64>,
    mask: Vec<Mask>,         // filter output, in row order
    waiting: bool,
    pending_error: Option<VortexError>,
    projection: PlanRef,
    locations: Arc<[SegmentLocation]>,
    session: VortexSession,
    done: bool,
}

impl Planner for SimpleFilterPlanner {
    fn state(&self) -> State {
        if self.done { State::Done } else if self.waiting { State::Waiting } else { State::NeedsCompute }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        match self.filter.step()? {
            Turn::Read(read) => {
                self.waiting = true;
                Ok(PlannerOutput::NeedsIO(vec![self.filter.fetch(read)]))
            }
            Turn::Output(_, array) => {
                let mut ctx = self.session.create_execution_ctx();
                self.mask.push(array.fill_null(false)?.execute::<Mask>(&mut ctx)?);
                Ok(PlannerOutput::Continue)
            }
            Turn::Waiting => { self.waiting = true; Ok(PlannerOutput::Continue) }
            Turn::Done => {
                self.done = true;
                let mask = Mask::concat(self.mask.iter())?;
                let split = Split { rows: self.rows.clone(), mask };
                let scan = Scan::try_new(self.session.clone(), self.projection.clone(), vec![split])?;
                let scope = WorkScope { file_ordinal: 0, rows: self.rows.clone() };
                Ok(PlannerOutput::Morsel(scope, Box::new(ProjectionMorsel::new(scan, self.locations.clone()))))
            }
        }
    }
}

impl IoConsumer for SimpleFilterPlanner {
    fn set_io_result(&mut self, id: IoRequestId, result: IoResult) {
        // Store only: a failed delivery is reported by the next compute.
        self.pending_error = self.filter.deliver(id, result).err();
        self.waiting = false;
    }
}
```

`ProjectionMorsel` has the same shape. It steps its own `PlanRun`, returns `NeedsIO` for reads and
`MorselOutput::Batch` for each output array, and returns `Done` after `Turn::Done`.

**4. Driving it.** Admit the planner on a run, as in the driver loop above, with an `IoSource`
that reads the file's byte ranges:

```rust
let root = Box::new(SimpleFilterPlanner::new(filter, projection, 0..row_count, locations, session));
let batches = Driver::new(io).run(root)?;
```

From here, the built-in scan adds pruning ahead of the filter (`PrunePlanner`), cuts rows into
chunks so they can run in parallel (`ChunkPlanner` with `Spawn`), and shares plans and segments
across scans (`PreparedScan`, `ScanCache`).

## Adding a planner: a limit

The V2 `ScanBuilder` has no row limit yet. A limit fits naturally as a planner, because it decides
which work to run and does not change how values are computed. A sketch:

- **Without a filter**, the limit is a row range. Before planning, shrink the scan's rows to the
  first `n` selected rows. No new planner is needed.
- **With a filter**, wrap the chunk stage. A parent planner never sees its children's output,
  because batches go to the driver, so the limit needs shared state: a budget of `n` rows held in
  an `Arc<AtomicU64>`.
  - A `LimitPlanner` emits chunk planners in row order, as children (`PlannerOutput::Planner`).
    Before emitting each one it checks the budget, and returns `Done` once it is spent.
  - Each chunk's projection planner takes rows from the budget for the rows its filter kept, and
    slices its morsel's selection to what it took, so rows past the limit are never decoded.
  - Chunks already in flight when the budget runs out produce nothing, so overshoot only costs
    filter work.
- To keep parallelism, the limit planner can `Spawn` a bounded number of chunks ahead instead.
  Because spawned chunks can finish out of order, they must reserve budget in row order (for
  example, chunk `i` waits until chunks before it have taken theirs) to keep the first `n` rows.
  That is policy inside the planner; the plans and pipelines underneath stay the same.
