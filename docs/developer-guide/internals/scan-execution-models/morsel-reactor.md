# Morsel Reactor Architecture

## Status

This document consolidates the proposed behavior of self-paced scan execution after separating
morsel-local planning from externally scheduled I/O and CPU work. It refines the broader
[self-paced execution proposal](self-paced.md). The
[scheduler-visible work](scheduler-visible-work.md) note compares related DataFusion and DuckDB
designs, estimates dependency-graph cost, and gives a longer worked example. Unsettled policies are
kept in [morsel reactor ideas](morsel-reactor-ideas.md).

The central contract is:

> One thread at a time owns the mutable planning state for a morsel. It incrementally advances a
> reactor, returning every currently known I/O and CPU opportunity. The global scheduler decides
> which work to admit, and any worker may execute admitted CPU work. Results return as immutable
> facts to the morsel owner, which advances the reactor again.

The reactor coordinates work. It does not perform physical I/O or expensive CPU computation.

## System overview

```text
LayoutRef
  -> lower and optimize
PlanRef                              immutable physical operators
  -> compile one scan
ScanState                            domains, static reads, shared caches
  -> open fixed outer morsels
MorselReactor                        one mutable owner at a time
  -> advance facts and demand
PlanStep                             work offers, updates, gates, output
  -> global scheduler admits work
I/O executor and CPU workers         CPU work is stealable
  -> completion mailbox
MorselReactor::advance               expose the next frontier
  -> self-paced prefixes
RootRebatcher
  -> ArrayStream
```

Fixed morsels remain the units of outer ownership, ordering, cancellation, and coarse parallelism.
Inside a morsel, demand windows and natural storage boundaries determine the actual I/O, CPU, and
output units.

## Components and ownership

| Component | Lifetime | Owner | Responsibility |
| --- | --- | --- | --- |
| `PlanRef` | Session or scan | Immutable and shared | Physical operator structure and rewrites |
| `ScanState` | One scan | Shared scan coordinator | Domains, read catalog, stable identities, shared caches |
| `MorselReactor` | One morsel | Exactly one thread at a time | Demand, local graph state, cursors, gates, and output progress |
| `DemandLedger` | One morsel | Morsel reactor | Open candidate masks, pending refiners, sealing, summaries |
| `ReadCatalog` | One scan with morsel views | Scheduler-facing scan state | Logical read uses, physical keys, coverage, and deduplication |
| Fact and task slots | One morsel | Morsel reactor | Durable results and dependency routing |
| Global scheduler | Runtime | Shared | Admission, priorities, stealing, credits, and cancellation |
| I/O executor | Runtime | Shared | Execute admitted physical reads |
| CPU workers | Runtime | Shared | Execute admitted CPU tasks from any morsel |
| Root rebatcher | One output stream | Stream owner | Convert natural prefixes into consumer-sized batches |

A thread that owns a reactor may execute tasks from other morsels while its own morsel waits. Task
execution never grants mutable access to the reactor. A task owns its inputs and returns an owned
result or result handle through the completion mailbox.

Reactor ownership may move only while the reactor is not running. Moving the complete reactor is a
scheduling optimization; concurrent calls to `advance` on one reactor are forbidden.

## Static structure and dynamic expansion

Opening a scan compiles immutable plan structure into a dependency template. The template assigns:

- stable operator identities;
- row domains and one `DomainMap` per edge;
- static fact types and subscriptions;
- static read uses and coverage;
- conditional gate recipes; and
- rules for constructing task and result identities.

Opening a morsel instantiates mutable state from that template:

```rust
struct ExecGraph {
    nodes: SlotMap<ExecNodeId, NodeState>,
    facts: SlotMap<FactId, FactSlot>,
    tasks: SlotMap<TaskId, TaskSlot>,
    runnable: VecDeque<ExecNodeId>,
    queued: BitSet,
}
```

The template does not enumerate every future task. Some tasks depend on runtime values:

```text
decoded dictionary codes
  -> exact gather demand
  -> value-page addresses
  -> value-page reads
```

The dependency recipe exists when the template is compiled. Concrete value-page tasks appear only
after the codes fact exists. Dynamic expansion is therefore monotone realization of a compiled
reactor, not repeated reconstruction of the plan tree.

## The `advance` contract

`advance` ingests events and performs bounded, cheap transitions:

```rust
trait MorselReactor {
    fn advance(
        &mut self,
        events: impl Iterator<Item = MorselEvent>,
        transition_budget: usize,
    ) -> VortexResult<PlanStep>;
}

enum MorselEvent {
    TaskCompleted { task: TaskId, result: TaskResult },
    TaskFailed { task: TaskId, error: VortexError },
    CreditAvailable(CreditClass),
    OutputCapacityAvailable,
    Cancelled,
}

struct PlanStep {
    work: Vec<WorkUpdate>,
    output: Vec<ExecBatch>,
    gates: Vec<PendingGate>,
    locally_quiescent: bool,
    done: bool,
}
```

One call:

1. installs task results in durable fact slots;
2. enqueues subscribers of changed facts;
3. drains the dirty-node queue;
4. updates demand and cached frontiers;
5. expands newly resolved gates;
6. returns new or changed work offers;
7. returns output prefixes that can commit; and
8. stops at local quiescence or budget exhaustion.

`advance` may return work and output together. It does not encode progress as an exclusive
`MoreIo`, `RunCpu`, or `Blocked` state.

If `locally_quiescent` is true, no more local expansion is possible without an external event. If
it is false, the owner should call `advance` again promptly; no task completion is required first.
Already returned work may be scheduled while local expansion continues.

## Direct completion routing

Task completion does not wake the root and scan the execution tree. Each task names its output
fact, and each fact has a small subscriber list:

```rust
struct TaskSlot {
    owner: ExecNodeId,
    output: FactId,
    state: TaskState,
}

struct FactSlot {
    value: Option<FactValue>,
    generation: u64,
    subscribers: SmallVec<[ExecNodeId; 2]>,
}
```

```text
TaskId -> FactId -> exact subscribers -> dirty-node queue
```

Duplicate wakes are coalesced by the queued bit. Durable fact and task state, not event ordering,
determines behavior.

## Demand model

Demand is the shared connection among pruning, conjunctions, and projection. Every demand block is
an independently refinable row window, for example 1,024 rows.

```rust
struct DemandBlock {
    rows: Range<u64>,
    candidate: Arc<Mask>,
    generation: u64,
    remaining_predicates: PredicateSet,
    dynamic_inputs: DynamicInputSet,
    state: BlockState,
}

enum BlockState {
    Open,
    Sealed { exact: Arc<Mask> },
}
```

Demand only shrinks:

```text
Open(M0) -> Open(M1) -> Open(M2) -> Sealed(M3)
```

### Open demand

An open snapshot is an immutable upper bound on final demand:

```rust
struct OpenDemand {
    block: DemandBlockId,
    rows: Range<u64>,
    mask: Arc<Mask>,
    generation: u64,
    remaining_predicates: PredicateSet,
    expected_survivors: Option<usize>,
}
```

Open demand is visible to both filtering and projection planning. It may authorize:

- pruning and metadata work;
- predicate I/O;
- predicate CPU over an immutable input snapshot;
- candidate projection reads;
- infallible speculative CPU needed to discover conditional projection reads; and
- scheduler scoring and cancellation decisions.

Open demand does not authorize output commitment. It also does not authorize fallible or otherwise
demand-sensitive projection computation unless that operation has an explicit speculation-safety
classification.

### Sealed demand

A sealed snapshot is exact and immutable:

```rust
struct SealedDemand {
    block: DemandBlockId,
    domain: DomainId,
    rows: Range<u64>,
    mask: Arc<Mask>,
    mask_offset: usize,
}
```

Sealed demand may authorize exact projection computation and output commitment. Projection reads
already admitted under open demand retain their identities and are promoted from candidate to
required rather than reissued.

### Sealing

For a conjunction, a block seals when its candidate mask is final:

```text
candidate is empty
OR
all correctness-relevant predicates are complete or proven unnecessary
AND every dynamic demand input is frozen for this block
```

An empty candidate seals immediately because later intersections cannot add rows. Pruning evidence
may prove a predicate true, prove the whole block false, or leave row-level evaluation pending.
Optional evidence that has become irrelevant is retired before sealing. Late optional results cannot
modify a sealed block.

Dynamic filters require an explicit snapshot boundary. A block must either wait for the relevant
dynamic-filter version, freeze that version, or specify that later versions apply only to future
blocks.

Blocks may seal out of order. Ordered output commits only through the contiguous sealed frontier,
although candidate I/O and safe computation may run ahead.

## Work model

A work offer is descriptive, stable, and scheduler-visible:

```rust
struct WorkItem {
    id: WorkId,
    owner: ExecNodeId,
    kind: WorkKind,
    phase: WorkPhase,
    necessity: Necessity,
    authorization: WorkAuthorization,
    coverage: DomainCoverage,
    estimated: Cost,
    inputs: SmallVec<[FactId; 4]>,
    output: FactId,
}

enum WorkKind {
    Read(ReadSpec),
    Cpu(CpuSpec),
}

enum Necessity {
    Candidate,
    Required,
}

enum WorkAuthorization {
    CandidateRead(OpenDemandId),
    PredicateCompute(OpenDemandId),
    SpeculativeCompute(OpenDemandId),
    ExactCompute(SealedDemandId),
    InfallibleMetadata,
}
```

The reactor publishes lifecycle changes rather than inventing new identities after each demand
revision:

```rust
enum WorkUpdate {
    Offer(WorkItem),
    Rescore {
        id: WorkId,
        demand: DemandSnapshotId,
        estimated: Cost,
    },
    Promote {
        id: WorkId,
        authorization: WorkAuthorization,
    },
    Eliminate {
        id: WorkId,
    },
}
```

An unadmitted predicate task may be rescored against a newer, smaller open snapshot. A task already
running on an older snapshot may finish: the current candidate mask is a subset of its immutable
input, so its result remains usable. An in-flight read normally continues and may be reused even if
its immediate priority falls.

## Read catalog and gates

Static preparation exposes every read whose physical address is known. Each logical use contains:

- a stable `ReadUseId` and physical `ReadKey`;
- owner, domain, and coverage;
- estimated bytes and phase;
- candidate or required status; and
- any dependency gate.

Filter and projection uses may share one `ReadKey`. The scheduler performs one physical read and
publishes the result to every surviving logical use.

Data-dependent addresses use gates:

- dictionary values wait for decoded codes;
- list elements wait for decoded offsets;
- encoded pages may wait for an index or footer; and
- zoned data may wait for evidence.

A gate records why work is not concrete and which fact resolves it. When that fact arrives, only
the gate owner runs and expands stable read uses once.

## Pruning and evidence

Pruning nodes produce facts that refine or explain demand. They are not a second copy of the row
filter pipeline.

Evidence may:

- eliminate a block;
- prove one predicate true for a block;
- reduce the candidate mask;
- expose additional reads; or
- remain inconclusive.

Several predicates may share one metadata read and decode task. Their semantic results remain
separate even when their physical work is fused.

Optional evidence competes with row-level filtering. The scheduler may skip expensive evidence if
an exact predicate is already cheap or ready. The ledger seals once exact correctness no longer
depends on that evidence.

## Conjunctions

Each conjunction retains a semantic identity for correctness, selectivity reporting, adaptive
ordering, and fallible behavior. Physical tasks may be separate or fused.

For predicates `P0`, `P1`, and `P2`, one open block exposes all currently possible I/O and CPU
opportunities. The scheduler may choose:

```text
sequential:
  P0(M0) -> M1
  P1(M1) -> M2
  P2(M2) -> M3

concurrent:
  P0(M0), P1(M0), P2(M0)
  M3 = M0 & R0 & R1 & R2

hybrid:
  prefetch all inputs
  run P0 first
  run P1 on survivors
  run P2 early only if workers would otherwise idle
```

Every predicate CPU task receives an immutable demand snapshot. Applying its result requires:

```text
current candidate is a subset of the task's input snapshot
```

This makes concurrently produced masks safe to intersect in any completion order for deterministic,
infallible conjunctions.

Fallible predicates may require explicit ordering. Running a later predicate over a row that an
earlier predicate would remove can expose an error that sequential evaluation would not observe.
The plan must classify which predicates are commutative, speculation-safe, or ordered.

## Projection

One projection coordinator owns semantic output progress, but projection is not one monolithic
task. It may expose independent work for fields, expressions, dictionaries, lists, and other
operators.

Projection planning observes open demand. Before sealing it may:

- offer statically known reads as candidates;
- update read scores as candidate masks shrink;
- perform safe discovery reads and CPU work;
- expand gates for conditional reads; and
- eliminate work whose coverage is exactly empty.

After sealing it may:

- promote reads needed by the exact prefix;
- run exact or fallible decode and expression work;
- align field frontiers;
- construct compact values; and
- commit a dense output prefix.

The coordinator tracks committed, ready, and scheduled frontiers per child. Expensive decode and
evaluation are stealable CPU tasks. Pack alignment, mask slicing, and cursor changes are cheap local
transitions.

## Physical fusion versus semantic separation

Keep semantic facts separate when they affect:

- predicate completion and sealing;
- error ordering;
- demand derivation;
- cancellation;
- metrics or adaptive selectivity; or
- output commitment.

Fuse physical work when operations:

- use the same physical bytes or decoded input;
- share a demand snapshot and scheduling priority;
- are individually smaller than task-launch cost; and
- have compatible error and cancellation semantics.

For example, `total > 100 AND total < 10_000` may share one read and decode. A cheap selective
`status = 'OPEN'` and an expensive description regular expression should normally remain separate
so the first can shrink the second's demand.

## Scheduler interaction

The global scheduler receives facts, not operator-specific policy. Useful fields include:

- required versus candidate status;
- current candidate count and demand generation;
- estimated selectivity and confidence;
- I/O bytes, CPU cost, and retained-result cost;
- phase and distance from the commit frontier;
- shared read keys and cached inputs;
- dependency gates;
- cancellation group; and
- resource-credit class.

The scheduler may favor sequential filtering when queues are full and speculative concurrency when
workers or I/O capacity would otherwise idle. High-latency projection reads may run under open
demand when expected survival is high. Large projection reads may wait when remaining filters are
likely selective.

These choices affect resource use and latency, not correctness. The reactor supplies valid work
offers and authorization; the scheduler decides admission.

## Planning and work stealing

Each owner advances its morsel to a bounded planning horizon. Returned CPU tasks enter stealable
worker deques; returned reads enter the shared I/O scheduler. Completion messages target the owning
morsel mailbox.

```text
owner advances morsel A
  -> publishes A/W0, A/W1, A/W2
worker 3 steals A/W1
I/O executor runs A/W0
owner executes work from morsel B while A waits
results enter A mailbox
owner drains A mailbox and advances A again
```

Planning should replenish work before queues drain, but it should not expand unlimited speculative
work. The exact watermark and look-ahead policy are scheduler choices recorded in the ideas note.

## Pipelining within one morsel

Different demand blocks may occupy different phases simultaneously:

```text
block 0    exact projection and output
block 1    final predicate
block 2    first predicate
block 3    pruning evidence
blocks 4+  candidate read-ahead
```

This supplies stealable work without forcing every predicate for one block to run concurrently.
Compressed I/O may lead farther than decoded CPU work because its budget and retention cost are
tracked separately.

## Backpressure and release

The scheduler accounts separately for:

- in-flight and retained compressed bytes;
- decoded arrays and retained child tails;
- CPU task inputs and outputs;
- root output buffering; and
- oversized indivisible units.

The node capable of releasing a retained result owns and is charged for it. Results are released
when no uncommitted prefix or subscriber can use them. Candidate work cannot consume progress
credits reserved for required work that unblocks the oldest in-flight morsel.

## Errors and cancellation

Task failure is a durable fact routed to its owner. The reactor determines whether it is fatal,
irrelevant because demand became empty, or ordered behind an earlier result. Once a fatal error or
cancellation commits:

- no further output may commit;
- unadmitted work is eliminated;
- cancellable in-flight work is notified;
- retained results and read uses are released; and
- duplicate late completions are safely ignored or dropped.

Error ordering must be defined before fallible predicate or projection speculation is enabled.

## Output

An `ExecBatch` separates dense progress from compact values:

```rust
struct ExecBatch {
    rows: Range<u64>,
    values: ArrayRef,
    retained_bytes: usize,
}
```

The batch covers one non-empty dense prefix of sealed demand. `values.len()` equals the population
count of the exact mask over that prefix. An all-false mask produces zero values while still
advancing dense progress.

Parents align row-equivalent children by capping their requested end. A root rebatcher hides
natural page, segment, and operator boundaries from consumers.

## End-to-end example

For:

```sql
SELECT order_id, customer_name
FROM orders
WHERE status = 'OPEN' AND total > 100;
```

one block begins with open demand `M0`:

```text
filter offers:
  read status
  read total
  evaluate either predicate over M0 when its bytes are ready

projection offers:
  candidate read order_id
  candidate read customer-name codes

projection gate:
  customer-name value pages wait for decoded codes
```

The scheduler may issue all four reads when projection latency is high and expected selectivity is
low. It may issue only filter reads when projection bytes are large and expected selectivity is
high.

Suppose predicate results produce:

```text
M1 = M0 & StatusMask
M2 = M1 & TotalMask
remaining predicates = {}
```

The ledger seals `M2`. Existing projection reads are promoted without duplication. Decoded codes
resolve the value-page gate, causing the reactor to offer exact value-page reads. After their decode
and gather tasks complete, the projection coordinator packs the fields and commits the block's dense
prefix.

At every point the owner performs only cheap coordination. Any worker may execute the returned CPU
tasks.

## Correctness invariants

1. Exactly one thread mutates a morsel reactor at a time.
2. Tasks own their inputs and never retain mutable access to reactor state.
3. Demand shrinks monotonically within a block epoch.
4. Predicate results are applied only when current demand is a subset of their immutable input.
5. Only the ledger seals demand, and sealed demand never changes.
6. Open demand may authorize candidate I/O and explicitly safe CPU work, but not output commit.
7. Exact or fallible projection work requires sealed demand unless explicitly proven speculation-safe.
8. Work identities remain stable across rescore and candidate-to-required promotion.
9. Requiring one physical `ReadKey` through several logical uses performs at most one physical read.
10. Completion routes through durable facts to exact subscribers; event order is not semantic state.
11. A locally quiescent step has exposed every currently concrete work opportunity.
12. A non-quiescent step is re-advanced without waiting for an external event.
13. A sealed empty block may eliminate all remaining value work and still advances dense progress.
14. Output commits only a contiguous sealed prefix in the configured ordering mode.
15. Retained data is charged to the component that can release it.

## Relationship to the other notes

- [Self-paced execution](self-paced.md) contains the complete operator and row-domain proposal.
- [Scheduler-visible work](scheduler-visible-work.md) compares upstream systems, sizes the graph,
  and works through dynamic dictionary reads.
- [Morsel reactor ideas](morsel-reactor-ideas.md) holds scheduler policies and prototype choices
  that should not yet be treated as architectural requirements.
- [Plan execution experiment](self-paced-plan-exec-experiment.md) reduces this contract to a
  restricted executable study driven by an external scheduler.
- [Implementation plan](self-paced-implementation-plan.md) describes migration phases and gates.
