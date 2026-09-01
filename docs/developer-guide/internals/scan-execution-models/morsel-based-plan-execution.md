# Morsel-Based Plan Execution

Status: **current working direction (2026-08-26)**. This document is the focus for subsequent
scan-execution design work. Earlier documents in this directory remain useful evidence and
derivation, but this document owns the model described below.

## Documents to use

Use these documents together; treat the remaining files in this directory as historical working
notes unless one of these links points to them:

- **Current direction:** this document owns the morsel, internal `IO | Plan` iterator, exec-node,
  and retirement contracts.
- **Prototype plan:** the [morsel prototype plan](morsel-prototype-plan.md) sequences the
  implementation phases and the real-query experiments that gate them.
- **Current implementation context:** [plan v2](plan-v2.md) describes today's physical operators;
  [scan plans](../scan-planning.md) and [array execution](../execution.md) describe the existing
  planning and array-execution APIs that this proposal would change or reuse.
- **Measured evidence:** the [self-paced experiment findings](self-paced-plan-exec-findings.md) and
  [executor reference](self-paced-executor-reference.md) record the benchmark results, ownership
  lessons, and implemented experimental machinery that should constrain a prototype.
- **Prior derivation:** the [previous consolidated design](scan-execution-design.md) and
  [demand/operator discussion](scan-execution-demand-and-operators.md) contain useful reasoning,
  but this document wins when their contracts disagree with it.
- **Archive map:** the [scan execution model index](index.md) links every earlier proposal, review,
  experiment, and handover.

The short version is:

- divide each root row domain into independent morsels, initially about 128K rows;
- open one stateful execution graph per morsel;
- let every execution node expose a lazy stream of future I/O and internal planning boundaries;
- let the scheduler fetch or prefetch those requests according to their demand;
- drive each node as a resumable CPU operator returning `Value`, `Blocked`, `Yield`, or `Done`; and
- retire the morsel as one lifetime unit, releasing its leases on requests shared across morsels.

This keeps scheduling at a useful granularity while making I/O, planning refinement, and CPU work
visible independently.

## What “each exec node” means

Yes: **one exec node is the per-morsel, stateful instantiation of one physical plan node**. It is
not a thread, an I/O request, or an output batch.

The immutable plan node says what the operator is and owns reusable metadata. Opening that plan
node for a morsel creates an exec node with:

- the part of the morsel in this node's row domain;
- child exec nodes lazily opened for the corresponding child ranges;
- a planning cursor that discovers this node's and its descendants' I/O;
- an execution cursor, buffered child values, and outstanding tickets; and
- a `BEGIN -> WORK -> RETIRE` lifecycle.

“Ownership of part of the plan” therefore means ownership of a **mutable activation over a plan
slice**, not unique ownership of the immutable plan node. A morsel holds a shared `PlanRef` and
exclusively owns the exec arena containing its local state:

~~~rust
struct MorselExec {
    root_plan: PlanRef,       // shared, immutable query plan
    root_exec: NodeId,
    nodes: ExecArena,         // uniquely owned by this morsel
}

struct ExecBase {
    plan: PlanRef,            // operator metadata used by this activation
    rows: Range<u64>,         // the owned slice in this node's domain
    plan_pc: PlanPc,          // local refinement cursor
    exec_pc: ExecPc,          // local value-execution cursor
    children: Vec<NodeId>,    // only children opened for this slice
}
~~~

Refinement reads the shared plan and creates or updates nodes in the morsel's arena. It never
rewrites the shared `PlanRef`. For example, refining a chunked activation finds the chunks that
overlap this morsel, obtains those immutable child plans, maps the intersections into child-local
ranges, and opens only those child activations in `nodes`.

Current `PlanChildren` already separates these concerns partially: `PlanRef::child` may lazily
lower and cache an immutable logical child for the whole query. The new per-morsel refinement sits
after that operation. Morsel-specific ranges, demand, tickets, and progress must live in the exec
arena rather than the shared `PlanChildren` cache.

Conceptually:

```text
immutable PlanNode              per-morsel ExecNode
------------------              -------------------
operator and dtype       open   row range
children                 ---->  child exec nodes
chunk/segment metadata          planning cursor
expressions                     execution cursor and buffers
                                request leases
                                BEGIN -> WORK -> RETIRE
```

The exec node has two independently resumable surfaces:

1. **planning** reveals future work without doing the operator's value computation; and
2. **execution** performs CPU work and produces the logical value stream.

A parent owns traversal of its children on both surfaces. The root's planning stream therefore
covers every request currently discoverable in its subtree, without requiring the scheduler to
understand operator types.

Both `next_plan` and `execute` take `&mut self`. The scheduler guarantees that only one of them is
active for a morsel at a time. They may be interleaved arbitrarily, but cannot concurrently mutate
the same activation. I/O continues independently because it owns stable tickets and cache cells,
not a borrow of the exec node.

This differs from the current `PlanVTable::execute` contract, which recursively constructs a
single future for an exact row range. The current lowering is still a useful starting IR: flat,
chunked, and struct layouts lower to `SegmentScan`, `Concat`, and `Pack`; those plan nodes would
open `FlatExec`, `ChunkedExec`, and `StructExec` instances under this model.

## Morsels

A morsel is a contiguous range in a row domain and the primary unit of parallel ownership:

~~~rust
struct Morsel {
    id: MorselId,
    domain: DomainId,
    rows: Range<u64>,
    kind: MorselKind,
}

enum MorselKind {
    Scan,
    // Later:
    Filter,
    Projection,
}
~~~

The first implementation should use one scan morsel for filter and projection, targeting roughly
131,072 rows. This is a target, not an alignment requirement. A second limit based on estimated
live bytes should prevent very wide projections from making one morsel too large.

Morsel boundaries deliberately do not have to match physical chunk or segment boundaries. When a
physical request straddles two morsels:

1. both morsels register the same stable request key;
2. the read/decode cache joins them to one request cell;
3. each exec node takes a lease and slices the shared array to its local rows; and
4. `RETIRE` releases that morsel's lease. The cache may evict only after its final user retires
   and no returned array retains the underlying buffers.

The request cell is scan-wide; execution progress and slice objects remain morsel-local. This is
the narrow cross-morsel sharing needed to make fixed morsels independent of storage geometry.

## Core contracts

The pseudocode uses an arena-friendly `NodeId`, although direct boxes could implement the same
contract.

~~~rust
trait PlanNode: Send + Sync {
    fn open(&self, morsel: &Morsel, cx: &mut OpenCx) -> NodeId;
}

trait ExecNode {
    /// Reveal one bounded quantum of I/O or further planning.
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll;

    /// Perform bounded CPU/state-machine work.
    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll;

    /// Cancel unused candidates and release all morsel-owned leases and buffers.
    fn retire(&mut self, cx: &mut RetireCx);
}

enum NodePhase {
    Begin,
    Work,
    Retire,
    Done,
}
~~~

The sketches omit the outer `VortexResult<...>` on fallible methods so the state transitions stay
visible. A real trait wraps `PlanPoll` and `ExecPoll` in `VortexResult` and unwinds the morsel
through `RETIRE` on error.

`next_plan` is a pull-based, resumable iterator rather than Rust's ordinary `Iterator`, because it
mutates internal state and some later planning is gated by metadata, offsets, codes, or a child
result.

~~~rust
enum PlanPoll {
    Item(PlanItem),
    Blocked(WaitSet),
    Complete,
}

enum PlanItem {
    /// A coalescing and admission unit containing one or more physical requests.
    Io(IoBatch),

    /// Stop before the node's next internal planning/refinement quantum.
    Plan,
}

struct IoBatch {
    group: IoGroupId,
    requests: Vec<IoUse>,
}

struct IoUse {
    ticket: IoTicket,
    key: IoKey,
    demand: DemandRef,
    estimated_bytes: usize,
}
~~~

`Plan` is a yield marker, not a detached task. Returning it means: **this call stops before doing
the next non-trivial internal planning step; the next call to this same node's `next_plan(&mut
self)` performs that step**. The planning state and all results remain inside the morsel exec.

The two-call boundary is deliberate:

~~~rust
enum ChunkPlanPc {
    BeforeExpand,
    ExpandOnNextCall,
    Children,
    Complete,
}

fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
    loop {
        match self.plan_pc {
            ChunkPlanPc::BeforeExpand => {
                self.plan_pc = ChunkPlanPc::ExpandOnNextCall;
                return PlanPoll::Item(PlanItem::Plan); // no expensive planning yet
            }
            ChunkPlanPc::ExpandOnNextCall => {
                self.open_overlapping_children(cx)?;  // mutates only this morsel exec
                self.plan_pc = ChunkPlanPc::Children;
                // Continue in this call until an IO, Plan, Blocked, or Complete result.
            }
            ChunkPlanPc::Children => return self.next_child_plan(cx),
            ChunkPlanPc::Complete => return PlanPoll::Complete,
        }
    }
}
~~~

If one refinement quantum is itself too large, it performs bounded progress, leaves its program
counter in `ExpandOnNextCall`, and returns another `Plan`. The next call resumes it.

`Plan` is not a vague barrier or ordering marker:

- Grouping is represented by `IoBatch`/`IoGroupId`.
- Hard ordering is represented by the exec node's internal program counter and any tickets it
  waits on.
- Priority is scheduler policy based on demand and whether the morsel has pending internal plan
  work.
- Returning `Plan` lets the scheduler inspect already-visible I/O and other morsels before asking
  this node to pay for the planning quantum.

This captures the intended “look for other I/O before asking me again” behavior without making
correctness depend on an advisory ordering hint.

The caller pulls repeatedly within a small item budget. `IO` registers a batch. `Plan` stops the
pull loop and marks the morsel as having internal planning work; a later pull calls `next_plan`
again and therefore runs it. `Blocked` parks the planning side of the morsel on its wait set, and
`Complete` closes the stream.

Demand is a stable symbol allocated while opening the graph. An I/O use holds the symbol rather
than a copied mask, so the scheduler can sample the newest state immediately before admission.

~~~rust
enum DemandSnapshot {
    Open(Mask),          // conservative; IO is a prefetch candidate
    Sealed(Mask),        // exact; non-empty IO is required
    SealedEmpty,         // this use can be cancelled
}

impl DemandRef {
    fn snapshot(&self) -> DemandSnapshot;
}
~~~

If several uses join the same `IoKey`, the cache performs the request once. The scheduler retains
the demand and priority of every use: one required use makes the physical request required, while
retiring one morsel removes only that use.

Execution returns a logical stream:

~~~rust
enum ExecPoll {
    Done,
    Blocked(WaitSet),
    Yield(Progress),
    Value(ValueBatch),
}

struct ValueBatch {
    /// Dense input rows accounted for, in this node's domain.
    coverage: Range<u64>,
    value: Value,
}

enum Value {
    Array(ArrayRef),
    Mask(Mask),
}

enum Wait {
    Io(IoTicket),
    Fact(FactTicket),
    Cpu(CpuTicket),
    Credit(CreditTicket),
}
~~~

`coverage` is necessary even if the public shape is described as `Value(Array)`. A filter can
consume 32K dense input rows and produce an empty array; without coverage, its parent cannot tell
the difference between progress and no progress.

The execution rules are:

- `Value` commits a non-empty dense coverage prefix. The array itself may be empty after filter.
- `Blocked` is returned only when the node has polled every independently runnable child and none
  can make local progress. Its wait set names every event that can unblock it.
- `Blocked(Io)` raises that request's critical-path priority. It does not by itself seal demand.
- `Yield` is for fairness after actual state progress and a bounded transition budget.
- `Done` means no more values will be produced; the owner must call `retire` exactly once.

The general `WaitSet` is a small extension of `Blocked(IO)`. It is needed when internal planning
waits for a fact or a separately scheduled CPU decode blocks execution. `Plan` itself has no ticket:
the mutable program counter in the morsel exec is the continuation. A first prototype can contain
only `Wait::Io` while keeping the enum extensible.

## Scheduler sketch

Planning and execution are interleaved. They are not global phases.

~~~rust
fn pull_morsel_plan(m: &mut MorselExec, scheduler: &mut Scheduler) {
    if m.phase == NodePhase::Begin {
        m.phase = NodePhase::Work;
    }

    match m.nodes.next_plan(m.root_exec) {
        PlanPoll::Item(PlanItem::Io(batch)) => {
            scheduler.register_io(batch);
            scheduler.mark_plan_runnable(m.id); // another cheap pull may expose more IO
        }
        PlanPoll::Item(PlanItem::Plan) => {
            // next_plan changed its internal PC but did no expensive planning yet.
            // Defer this morsel so already-visible IO and other morsels get a look.
            scheduler.mark_internal_plan_pending(m.id);
        }
        PlanPoll::Blocked(waits) => scheduler.park_plan(m.id, waits),
        PlanPoll::Complete => scheduler.mark_plan_complete(m.id),
    }
}

fn drive_morsel_exec(m: &mut MorselExec, scheduler: &mut Scheduler) {
    scheduler.admit_io_by_demand_and_priority();

    match m.nodes.execute(m.root_exec) {
        ExecPoll::Value(batch) => {
            m.output.push(batch);
            scheduler.mark_exec_runnable(m.id);
        }
        ExecPoll::Blocked(waits) => {
            scheduler.boost_critical_path(&waits);
            scheduler.park_exec(m.id, waits);
        }
        ExecPoll::Yield(progress) => {
            debug_assert!(progress.did_work());
            scheduler.mark_exec_runnable(m.id);
        }
        ExecPoll::Done => {
            m.phase = NodePhase::Retire;
            m.nodes.retire(m.root_exec);
            m.phase = NodePhase::Done;
        }
    }
}
~~~

Admission samples each request's demand:

```text
SealedEmpty   -> cancel the use; do not issue it
Open          -> prefetch if queue depth, bytes, and expected value justify it
Sealed        -> fetch as required work
Blocked(IO)   -> add critical-path priority to that use
```

The scheduler may admit an `IO` batch, repoll a morsel whose planning side is runnable, or drive a
morsel's execution side. It never owns a planning continuation and never recursively inspects a
plan node. The morsel remains the unit receiving the exclusive mutable borrow.

## Shared helpers used below

Composite nodes repeatedly need two helpers.

`PlanMux` polls child planning streams round-robin. It returns an item immediately, remembers
blocked children, and reports `Complete` only when every child is complete. When a child returns
`Plan`, the mux pins its cursor to that child: the next parent pull re-enters the same child and
therefore performs the promised internal refinement before polling siblings again.

`AlignedHeads` buffers at most one value head per child. Given row-equivalent children whose heads
all start at the parent's cursor, it returns the smallest common end and slices longer heads,
retaining their tails. This makes physical chunk boundaries local to the child that owns them.

`finish_children` is the terminal handshake for composite nodes. After the parent has consumed its
entire coverage, it polls each child until the child returns `Done`, retires that child, and only
then returns the parent's `Done`. A child returning `Done` before its required coverage is consumed
is an error.

~~~rust
struct ChildHead {
    batch: ValueBatch,
    consumed: usize,
}

impl ChildHead {
    fn remaining_coverage(&self) -> Range<u64>;
    fn take_through(&mut self, end: u64) -> Value;
    fn exhausted(&self) -> bool;
}
~~~

These helpers are pseudocode conveniences, not proposed public APIs.

## `FLAT`

`FlatExec` is the leaf form of today's flat-layout `SegmentScan`. Planning registers the stable
physical request; execution waits for its shared decoded array and emits the morsel-local slice.

~~~rust
struct FlatExec {
    phase: NodePhase,
    node_rows: Range<u64>,       // physical array coverage in parent coordinates
    output_rows: Range<u64>,     // intersection with this morsel/request
    demand: DemandRef,
    ticket: Option<IoTicket>,
    emitted: bool,
    lease: Option<ArrayLease>,
}

impl ExecNode for FlatExec {
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
        if self.ticket.is_some() {
            return PlanPoll::Complete;
        }

        // The key names the complete stored segment/decode, not the morsel slice.
        // Two morsels crossing this segment therefore join the same cell.
        let key = IoKey::decoded_segment(cx.source(), cx.segment_id(), cx.decode_id());
        let ticket = cx.io_cache().join(key.clone());
        self.ticket = Some(ticket);

        PlanPoll::Item(PlanItem::Io(IoBatch {
            group: cx.current_group(),
            requests: vec![IoUse {
                ticket,
                key,
                demand: self.demand.clone(),
                estimated_bytes: cx.segment_size(),
            }],
        }))
    }

    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll {
        if self.emitted {
            self.phase = NodePhase::Retire;
            return ExecPoll::Done;
        }

        let ticket = self.ticket.expect("planning registers the flat request");
        let array = match cx.io_cache().poll_array(ticket) {
            Poll::Pending => return ExecPoll::Blocked(WaitSet::one(Wait::Io(ticket))),
            Poll::Ready(array_lease) => array_lease,
        };

        let local = (self.output_rows.start - self.node_rows.start)
            ..(self.output_rows.end - self.node_rows.start);
        let value = array.slice(local)?;
        self.lease = Some(array);
        self.emitted = true;

        ExecPoll::Value(ValueBatch {
            coverage: self.output_rows.clone(),
            value: Value::Array(value),
        })
    }

    fn retire(&mut self, cx: &mut RetireCx) {
        self.lease.take();
        if let Some(ticket) = self.ticket.take() {
            cx.io_cache().release_use(ticket);
        }
        self.phase = NodePhase::Done;
    }
}
~~~

The cache may internally split read and decode into separate I/O and CPU tickets. The externally
important property is that both morsels can share the decoded array and perform only cheap slice
work locally. If decode sharing proves too expensive to retain, the same contract can initially
cache bytes and allow duplicate decode; that is a policy change, not an operator change.

## `CHUNKED`

`ChunkedExec` opens only children overlapping the morsel. It maps their local ranges back to one
ordered parent stream. Planning can expose every overlapping child's I/O even while execution is
waiting on the first child.

~~~rust
struct ChunkPart {
    node: NodeId,
    child_rows: Range<u64>,     // child-local
    parent_rows: Range<u64>,    // same rows in the chunked domain
    plan_done: bool,
    head: Option<ChildHead>,
    output_complete: bool,
    done: bool,
}

struct ChunkedExec {
    phase: NodePhase,
    parts: Vec<ChunkPart>,
    plan_pc: ChunkPlanPc,
    plan_cursor: usize,
    poll_cursor: usize,
    emit_part: usize,
    transition_budget: usize,
}

impl ExecNode for ChunkedExec {
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
        loop {
            match self.plan_pc {
                ChunkPlanPc::BeforeExpand => {
                    self.plan_pc = ChunkPlanPc::ExpandOnNextCall;
                    return PlanPoll::Item(PlanItem::Plan);
                }
                ChunkPlanPc::ExpandOnNextCall => {
                    // This is the later call: compute morsel/chunk intersections and open the
                    // matching child activations in this morsel's arena.
                    self.open_overlapping_children(cx)?;
                    self.plan_pc = ChunkPlanPc::Children;
                }
                ChunkPlanPc::Children => {
                    // Round-robin so all overlapping physical leaves become visible.
                    let poll = PlanMux::next(&mut self.parts, &mut self.plan_cursor, cx);
                    if matches!(poll, PlanPoll::Complete) {
                        self.plan_pc = ChunkPlanPc::Complete;
                    }
                    return poll;
                }
                ChunkPlanPc::Complete => return PlanPoll::Complete,
            }
        }
    }

    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll {
        let mut waits = WaitSet::new();
        let mut progress = Progress::none();

        loop {
            while self.emit_part < self.parts.len() && self.parts[self.emit_part].done {
                self.emit_part += 1;
            }
            if self.emit_part == self.parts.len() {
                self.phase = NodePhase::Retire;
                return ExecPoll::Done;
            }

            // Preserve logical row order at emission.
            if let Some(head) = self.parts[self.emit_part].head.take() {
                let batch = head.batch;
                if batch.coverage.end == self.parts[self.emit_part].parent_rows.end {
                    self.parts[self.emit_part].output_complete = true;
                }
                return ExecPoll::Value(batch);
            }

            // Poll all children, including later chunks, so one blocked chunk does not hide
            // independent CPU work. Bounded look-ahead permits one buffered head per child.
            for part in round_robin(&mut self.parts, &mut self.poll_cursor) {
                if part.done || part.head.is_some() {
                    continue;
                }
                match cx.execute(part.node) {
                    ExecPoll::Value(mut batch) => {
                        batch.coverage = map_child_to_parent(batch.coverage, part);
                        part.head = Some(ChildHead::new(batch));
                        progress.record_transition();
                    }
                    ExecPoll::Blocked(child_waits) => waits.extend(child_waits),
                    ExecPoll::Yield(child_progress) => progress += child_progress,
                    ExecPoll::Done => {
                        if !part.output_complete {
                            return cx.error("chunk child ended before covering its parent range");
                        }
                        part.done = true;
                        cx.retire(part.node);
                        progress.record_transition();
                    }
                }
                if progress.transitions() >= self.transition_budget {
                    return ExecPoll::Yield(progress);
                }
            }

            if let Some(head) = self.parts[self.emit_part].head.take() {
                let batch = head.batch;
                if batch.coverage.end == self.parts[self.emit_part].parent_rows.end {
                    self.parts[self.emit_part].output_complete = true;
                }
                return ExecPoll::Value(batch);
            }
            if !waits.is_empty() {
                return ExecPoll::Blocked(waits);
            }
            return ExecPoll::Yield(progress.require_nonzero());
        }
    }

    fn retire(&mut self, cx: &mut RetireCx) {
        for part in &mut self.parts {
            cx.retire_if_needed(part.node);
            part.head.take();
        }
        self.phase = NodePhase::Done;
    }
}
~~~

A simpler first implementation may execute only `emit_part` while still planning every part. The
bounded look-ahead above is the stronger form: it exposes CPU parallelism without allowing
out-of-order output or unbounded buffering.

## `STRUCT`

`StructExec` owns row-equivalent field children. Their I/O and CPU work may progress independently,
but it can emit only the common prefix for which every field has a value. Longer child batches are
sliced and their tails remain buffered.

~~~rust
struct StructField {
    name: FieldName,
    node: NodeId,
    plan_done: bool,
    head: Option<ChildHead>,
    done: bool,
}

struct StructExec {
    phase: NodePhase,
    rows: Range<u64>,
    cursor: u64,
    fields: Vec<StructField>,
    validity: Option<StructField>,
    plan_mux: PlanMux,
    poll_cursor: usize,
}

impl ExecNode for StructExec {
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
        self.plan_mux.next(all_children_mut(self), cx)
    }

    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll {
        if self.cursor == self.rows.end {
            return finish_children(all_children_mut(self), &mut self.phase, cx);
        }

        let mut waits = WaitSet::new();
        let mut progress = Progress::none();

        // Do not return Blocked after the first blocked field. Poll every missing field so
        // racing field reads and CPU work remain visible.
        for child in missing_heads_round_robin(self, &mut self.poll_cursor) {
            match cx.execute(child.node) {
                ExecPoll::Value(batch) => {
                    debug_assert_eq!(batch.coverage.start, self.cursor);
                    child.head = Some(ChildHead::new(batch));
                    progress.record_transition();
                }
                ExecPoll::Blocked(child_waits) => waits.extend(child_waits),
                ExecPoll::Yield(child_progress) => progress += child_progress,
                ExecPoll::Done => return cx.error("struct child ended before the parent range"),
            }
        }

        if all_children_have_heads(self) {
            let end = minimum_head_end(self);
            let fields = self.fields.iter_mut()
                .map(|field| field.head.as_mut().unwrap().take_array_through(end))
                .collect::<Vec<_>>();
            let validity = take_validity_through(&mut self.validity, end);
            drop_exhausted_heads(self);

            let start = self.cursor;
            self.cursor = end;
            return ExecPoll::Value(ValueBatch {
                coverage: start..end,
                value: Value::Array(pack_struct(fields, validity)?),
            });
        }

        if !waits.is_empty() {
            return ExecPoll::Blocked(waits);
        }
        ExecPoll::Yield(progress.require_nonzero())
    }

    fn retire(&mut self, cx: &mut RetireCx) {
        for child in all_children_mut(self) {
            cx.retire_if_needed(child.node);
            child.head.take();
        }
        self.phase = NodePhase::Done;
    }
}
~~~

This is where morsel-local slicing absorbs mismatched physical geometry. If field A returns rows
`[0, 32K)` and field B returns `[0, 8K)`, the struct emits `[0, 8K)`, retains A's `[8K, 32K)` tail,
and next asks B for a batch beginning at 8K.

## `FILTER`

`FilterExec` is the explicit cardinality-changing operator. Its `selection` child produces final
mask batches, normally from `ConjunctParallelExec`; its `values` child produces positional arrays.
It plans both sides so projection I/O can be prefetched, but it does not emit until their coverage
is aligned and the mask for that coverage is final.

~~~rust
struct FilterExec {
    phase: NodePhase,
    rows: Range<u64>,
    cursor: u64,
    selection: NodeId,
    values: NodeId,
    projected_demand: DemandRef,
    mask_head: Option<ChildHead>,
    value_head: Option<ChildHead>,
    plan_mux: PlanMux,
    poll_first: Side,
}

impl ExecNode for FilterExec {
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
        // Both streams become visible. IO under projected_demand is candidate work until the
        // matching mask range seals, then required or cancelled.
        self.plan_mux.next([self.selection, self.values], cx)
    }

    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll {
        if self.cursor == self.rows.end {
            return finish_children(
                [self.selection, self.values],
                &mut self.phase,
                cx,
            );
        }

        let mut waits = WaitSet::new();
        let mut progress = Progress::none();

        if self.mask_head.is_none() {
            match cx.execute(self.selection) {
                ExecPoll::Value(batch @ ValueBatch { value: Value::Mask(_), .. }) => {
                    self.projected_demand.seal(batch.coverage.clone(), batch.mask());
                    self.mask_head = Some(ChildHead::new(batch));
                    progress.record_transition();
                }
                ExecPoll::Blocked(w) => waits.extend(w),
                ExecPoll::Yield(p) => progress += p,
                other => return cx.type_or_early_done_error(other),
            }
        }

        // Positional reads/decode may race the mask. CPU that can trap on dead rows must wait
        // for sealed demand; the node metadata tells ExecCx whether speculative CPU is legal.
        if self.value_head.is_none() &&
            (self.mask_head.is_some() || cx.is_speculation_safe(self.values))
        {
            match cx.execute(self.values) {
                ExecPoll::Value(batch @ ValueBatch { value: Value::Array(_), .. }) => {
                    self.value_head = Some(ChildHead::new(batch));
                    progress.record_transition();
                }
                ExecPoll::Blocked(w) => waits.extend(w),
                ExecPoll::Yield(p) => progress += p,
                other => return cx.type_or_early_done_error(other),
            }
        }

        // An all-false final mask accounts for dense progress without waiting for value IO.
        if let Some(mask_head) = &mut self.mask_head {
            if mask_head.remaining_mask().all_false() {
                let coverage = mask_head.remaining_coverage();
                self.projected_demand.seal_empty(coverage.clone());
                self.cursor = coverage.end;
                self.mask_head = None;
                if let Some(values) = &mut self.value_head {
                    values.discard_through(coverage.end);
                    drop_exhausted(&mut self.value_head);
                } else {
                    cx.skip(self.values, coverage.clone());
                }
                return ExecPoll::Value(ValueBatch {
                    coverage,
                    value: Value::Array(empty_array(cx.output_dtype())),
                });
            }
        }

        if let (Some(mask), Some(values)) = (&mut self.mask_head, &mut self.value_head) {
            let end = mask.remaining_coverage().end.min(values.remaining_coverage().end);
            let coverage = self.cursor..end;
            let mask = mask.take_mask_through(end);
            let values = values.take_array_through(end);
            drop_exhausted(&mut self.mask_head);
            drop_exhausted(&mut self.value_head);
            self.cursor = end;

            return ExecPoll::Value(ValueBatch {
                coverage,
                value: Value::Array(values.filter(mask)?),
            });
        }

        if !waits.is_empty() {
            return ExecPoll::Blocked(waits);
        }
        ExecPoll::Yield(progress.require_nonzero())
    }

    fn retire(&mut self, cx: &mut RetireCx) {
        cx.retire_if_needed(self.selection);
        cx.retire_if_needed(self.values);
        self.mask_head.take();
        self.value_head.take();
        self.phase = NodePhase::Done;
    }
}
~~~

The all-false case explains why `coverage` cannot be inferred from `array.len()`. It also lets a
selective filter cancel projection uses before their physical request is issued.

## `CONJUNCT_PARALLEL`

`ConjunctParallelExec` races independent conjunct nodes. Each child produces positional masks in
the same row domain. Planning polls every child so their I/O is available for prefetch; execution
polls every child before declaring itself blocked.

The node has two outputs with different roles:

- mask batches are the exact in-band result consumed by `FILTER`; and
- each completed conjunct intersects an advisory demand cell immediately, allowing not-yet-issued
  sibling and projection I/O to shrink.

~~~rust
struct Conjunct {
    node: NodeId,
    head: Option<ChildHead>,
    plan_done: bool,
    done: bool,
}

struct ConjunctParallelExec {
    phase: NodePhase,
    rows: Range<u64>,
    cursor: u64,
    conjuncts: Vec<Conjunct>,
    remaining_demand: DemandRef,
    plan_mux: PlanMux,
    poll_cursor: usize,
    transition_budget: usize,
}

impl ExecNode for ConjunctParallelExec {
    fn next_plan(&mut self, cx: &mut PlanCx) -> PlanPoll {
        // Every child's IO carries remaining_demand. Early calls expose eager parallel IO;
        // delayed calls naturally behave like a cascade without changing operator code.
        self.plan_mux.next_with_demand(&mut self.conjuncts, &self.remaining_demand, cx)
    }

    fn execute(&mut self, cx: &mut ExecCx) -> ExecPoll {
        if self.cursor == self.rows.end {
            let children = self.conjuncts.iter().map(|c| c.node);
            return finish_children(children, &mut self.phase, cx);
        }

        if self.conjuncts.is_empty() {
            let coverage = self.cursor..self.rows.end;
            self.cursor = self.rows.end;
            return ExecPoll::Value(ValueBatch {
                value: Value::Mask(Mask::new_true(coverage.len())),
                coverage,
            });
        }

        let mut waits = WaitSet::new();
        let mut progress = Progress::none();

        // Poll all missing heads. In particular, do not return when the first conjunct says
        // Blocked(IO): another conjunct may already have runnable CPU work or a ready value.
        for conjunct in round_robin(&mut self.conjuncts, &mut self.poll_cursor) {
            if conjunct.done || conjunct.head.is_some() {
                continue;
            }

            match cx.execute(conjunct.node) {
                ExecPoll::Value(batch @ ValueBatch { value: Value::Mask(_), .. }) => {
                    debug_assert_eq!(batch.coverage.start, self.cursor);
                    self.remaining_demand.intersect(batch.coverage.clone(), batch.mask());
                    conjunct.head = Some(ChildHead::new(batch));
                    progress.record_transition();
                }
                ExecPoll::Blocked(child_waits) => waits.extend(child_waits),
                ExecPoll::Yield(child_progress) => progress += child_progress,
                ExecPoll::Done => return cx.error("conjunct ended before the morsel range"),
                other => return cx.type_error("conjunct must produce masks", other),
            }

            if progress.transitions() >= self.transition_budget {
                return ExecPoll::Yield(progress);
            }
        }

        // The final AND can advance only through the prefix represented by every child.
        if self.conjuncts.iter().all(|c| c.head.is_some()) {
            let end = self.conjuncts.iter()
                .map(|c| c.head.as_ref().unwrap().remaining_coverage().end)
                .min()
                .unwrap();
            let coverage = self.cursor..end;
            let mut result = Mask::new_true(coverage.len());
            for conjunct in &mut self.conjuncts {
                result &= conjunct.head.as_mut().unwrap().take_mask_through(end);
                drop_exhausted(&mut conjunct.head);
            }
            self.remaining_demand.intersect(coverage.clone(), &result);
            self.cursor = end;
            return ExecPoll::Value(ValueBatch {
                coverage,
                value: Value::Mask(result),
            });
        }

        if !waits.is_empty() {
            return ExecPoll::Blocked(waits);
        }
        ExecPoll::Yield(progress.require_nonzero())
    }

    fn retire(&mut self, cx: &mut RetireCx) {
        for conjunct in &mut self.conjuncts {
            cx.retire_if_needed(conjunct.node);
            conjunct.head.take();
        }
        self.phase = NodePhase::Done;
    }
}
~~~

An important fast path can be added after `skip(range)` exists: if any conjunct proves an entire
prefix false, AND is already final for that prefix. The node can emit an all-false mask immediately,
seal the demand empty, and tell the other conjuncts to skip that coverage. It must not merely drop
their values; their cursors must advance to the same end.

“Parallel” here does not require one task per conjunct. It means the scheduler can see all of their
I/O and each state machine refuses to hide runnable siblings behind its own blocked child. CPU work
below the task granularity floor can still run inline on the morsel's current worker.

## Planning refinements

Static operators (`FLAT`, `CHUNKED`, `STRUCT`) can normally reveal all reads by repeatedly pulling
their planning stream. `CHUNKED` may still return `Plan` before the non-trivial work of cutting a
morsel across many chunks. Data-dependent operators use the same internal boundary:

```text
call 1 -> IO([read dictionary codes])
call 2 -> Plan
          # stop; no dictionary refinement has run yet
call 3 -> if codes are not ready: Blocked(codes ticket)
          otherwise, internally find referenced values and return
          IO([read referenced dictionary value pages], demand = sealed gather set)
```

Likewise, list offsets or zoned metadata can unlock more planning. The CPU refinement is ordinary
code inside the later `next_plan(&mut self)` call. It is schedulable only in the sense that the
scheduler chooses when to grant that morsel another mutable planning turn; it is not packaged as a
separate task.

The following invariant keeps planning and execution consistent: **an exec node may block only on
a ticket already emitted by its planning stream, or return `Plan` before the later internal
refinement that will emit that ticket**. This prevents hidden I/O from appearing in `execute` while
keeping ownership of planning state entirely inside the morsel.

## Retirement and cancellation

`RETIRE` is semantically useful, not just a destructor:

- remove this morsel's uses from shared I/O cells;
- cancel unissued speculative uses whose remaining user set is empty;
- release decoded-array leases and buffered child tails;
- detach waiters so late completions do not wake a dead morsel; and
- allow a straddling request to become evictable after the last overlapping morsel retires.

A node retires only after it has returned `Done` and its parent has consumed or released every
value it emitted. Parent retirement recursively retires unfinished children during cancellation
or error unwinding.

## Separate filter and projection morsels later

One row target is unlikely to fit both sides forever. Filter columns are often narrow, while a
projection can contain many wide values; occasionally the reverse is true. The model should later
permit two morsel classes without changing operator contracts:

```text
FilterMorsel(rows = a..b)
    -> sealed SelectionBatch values

ProjectionMorsel(rows chosen by target projected bytes)
    -> consumes one or more sealed SelectionBatch slices
    -> runs projection subtree and FILTER
```

Filter morsels can be sized for predicate throughput. Projection morsels can be sized from
estimated bytes per surviving row and memory credits. A projection morsel may split or combine
filter selections because every batch carries explicit dense coverage.

This extension needs an ordered selection queue between the two classes and a rule for limit and
cancellation propagation. It does not need a second `ExecNode` API.

## Working decisions and open questions

The following are decisions for the first prototype:

1. One per-morsel exec node per physical plan node.
2. One initial scan-morsel class, about 128K rows with a byte cap.
3. Fixed morsels may cut physical arrays; stable keyed cache cells join straddling uses.
4. Planning is the resumable `IO([task]) | Plan` stream above.
5. `Plan` yields before internal refinement; the next `next_plan(&mut self)` call performs a
   bounded quantum inside the same morsel exec. It is never a detached task.
6. Execution returns `Done | Blocked(WaitSet) | Yield(Progress) | Value(ValueBatch)`.
7. Composite nodes poll every independent child before returning `Blocked`.
8. `RETIRE` releases leases and cancels dead speculative uses.

Questions to answer with the prototype:

- Should the shared cache retain decoded arrays or only read buffers? Start with decoded arrays
  for the straddling case and measure retained bytes and decode reuse.
- How much work may one internal planning quantum perform before returning another `Plan`? Use a
  transition/row budget and tune against I/O queue depth.
- How much out-of-order CPU look-ahead should `CHUNKED` permit? Start with zero or one head per
  child and measure memory versus latency hiding.
- Which nodes are safe to execute on open demand? Reads and slicing are safe; projection kernels
  that can trap on discarded rows require sealed demand.
- When should filter and projection split into separate morsel queues? Add this only after the
  single-morsel version reports per-side live bytes and time.

## Correctness properties for the prototype

At minimum, deterministic tests should vary I/O completion and child polling order and assert:

1. Every root row belongs to exactly one morsel of its class.
2. Every `ValueBatch` begins at its node's committed cursor and advances dense coverage.
3. `STRUCT` emits only aligned field coverage and retains every unconsumed tail exactly once.
4. `CHUNKED` emits in logical row order even if later children complete first.
5. `FILTER` output length equals the mask population, while output coverage equals dense input
   progress.
6. `CONJUNCT_PARALLEL` produces the same mask for every completion order.
7. `Blocked` always names a live event and is returned only after polling independent siblings.
8. `Yield` always records progress.
9. A physical request shared by two morsels is issued once and remains live until both retire.
10. Cancelling or retiring a morsel cannot wake it or evict storage still leased by another
    morsel.
