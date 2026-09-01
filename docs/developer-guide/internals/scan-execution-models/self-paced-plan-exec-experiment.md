# Self-Paced Plan Execution Experiment

This document proposes a small executable experiment for self-paced plan execution. It is not the
production implementation plan. Its purpose is to test the control-plane model of the
[morsel reactor](morsel-reactor.md) before integrating the complete Vortex expression, array,
layout, and scan stacks.

The experiment belongs in:

```text
vortex-layout/src/plan/exec/
```

It asks whether:

1. a morsel can expose I/O and CPU work without executing either;
2. independently completed conjuncts can monotonically reduce row demand;
3. later morsels can join results discovered by earlier morsels;
4. morsel retirement can prove when retained I/O and intermediate results are dead; and
5. a bounded `advance` can expose every currently concrete task without walking the whole graph.

The experiment should favor an inspectable event trace and strong invariants over generality or
peak performance.

The implemented experiment and its benchmark conclusions are recorded in the
[findings report](self-paced-plan-exec-findings.md).

## V1 comparison invariant

V1 must run with the layout's natural chunk boundaries. The self-paced morsel size is a candidate
scheduler parameter and must never be passed to V1 or used to subdivide V1 work. The comparison
harness therefore supplies `ScanBuilder::with_natural_splits` with boundaries derived directly
from `SourcePlan::chunks`; relying on the default `SplitBy::Layout` is incorrect for this
experiment because it may subdivide wide chunk spans at `IDEAL_SPLIT_SIZE`.

Both engines must scan the same input rows and may use the same worker-thread pool. Their work-unit
counts are intentionally different: V1 owns one split per natural layout chunk, while self-paced
owns the configured morsels (currently the primary comparison is 128K rows). Never report a run as
V1 versus self-paced if V1 was row-split using the self-paced morsel size or the layout
subdivision fallback.

## Scope

The only supported source shape is:

```text
Struct
├── Chunked(a)
│   ├── Flat(a0)
│   └── Flat(a1)
├── Chunked(b)
│   ├── Flat(b0)
│   └── Flat(b1)
└── Chunked(c)
    ├── Flat(c0)
    └── Flat(c1)
```

The supported query shape is a conjunction of field predicates followed by a field projection:

```text
filter:     a > 10 AND b < 5
projection: [a, c]
```

The first experiment deliberately has these restrictions:

- flat columns contain `i64` values;
- each conjunct reads one field and applies a simple comparison;
- projection selects fields rather than evaluating arbitrary expressions;
- field chunk boundaries are aligned across the Struct;
- a morsel may cross field-chunk boundaries and carries an ordered Flat slice list per field;
- an in-memory segment evaluator supplies the data plane;
- the only resolved work values are a segment `BufferHandle` and an `ArrayRef` with its
  inseparable summary; and
- the experiment does not replace `PlanVTable::execute`.

These restrictions preserve the scheduling problems under investigation while avoiding an early
dependency on general expression analysis, arbitrary encodings, and output assembly.

## Relationship to the architecture

The experiment is a reduced, executable form of the [morsel reactor](morsel-reactor.md) contract:

| Experiment | Architecture | Reduction |
| --- | --- | --- |
| `AdvanceResult` | `PlanStep` | No gates; only offer, promote, and revoke work updates |
| `Task` | `WorkItem` | Stable offers with only `Promote` and `Revoke` lifecycle updates |
| `ResolvedValue` | `FactValue` | Exactly two variants: a segment handle or an array plus inseparable metadata |
| Demand version | Demand generation | One demand block per morsel instead of 1,024-row blocks |
| Sealed demand | `BlockState::Sealed` | Sealing is whole-morsel, so out-of-order block sealing is not exercised |
| Resource node | Read-catalog entry plus fact retention | Whole-segment reads only; no gated or page reads |

The experiment deliberately drops:

- gates, because every read in the restricted shape is statically addressable;
- rescoring and general elimination, retaining only stable offers, candidate-to-required promotion,
  and revocation of work that has not been claimed;
- streamed morsel output, returning one root batch per morsel instead of the architecture's
  vector of sealed prefixes;
- multi-block demand ledgers, credits, and byte budgets; and
- worker threads, ownership migration, and stealing, because one external driver runs everything.

The minimal promotion and revocation updates are intentional findings: offer-only transport was
not sufficient once an external scheduler could retain a candidate after demand changed. If any
other dropped mechanism turns out to be load-bearing, that is a finding for the
[ideas note](morsel-reactor-ideas.md), not something to patch silently.

## Proposed module structure

```text
vortex-layout/src/plan/exec/
├── mod.rs
├── model.rs
├── slots.rs
├── graph.rs
├── reactor.rs
├── evaluate.rs
├── baseline.rs
└── tests.rs

vortex-layout/benches/
└── self_paced_plan_exec.rs

vortex-file/benches/
└── self_paced_vs_v1.rs
```

| Module | Responsibility |
| --- | --- |
| `model` | Source plans, query descriptions, identifiers, demand-array metadata, and batches |
| `slots` | Scan- and morsel-scoped typed slots, task ownership, and completion validation |
| `graph` | Scan-wide resource nodes, possible users, joined users, and retained results |
| `reactor` | Per-morsel state, bounded `advance`, completion handling, sealing, and retirement |
| `evaluate` | Reference external evaluator for I/O and CPU task payloads |
| `baseline` | Eager reference evaluation and the adapter for a fair V1 comparison |
| `tests` | Worked query, scheduler policies, invariants, traces, and measurements |

The public module should be marked experimental. Production plan execution remains unchanged.

## Implementation plan

Implement the experiment in stages. Each stage has an exit test and should remain reviewable on
its own.

### Phase 1: contracts and typed slots

Define identifiers, the scan- and morsel-scoped slot arenas, task states, `ResolvedValue`, array
summaries, completion validation, demand versions, and row-domain metadata. A worker returns a
resolved value to the execution owner; it does not mutate the reactor or slot store directly.

Exit when wrong-type, duplicate, failed, and stale completions plus claims of revoked offers are
rejected or handled without leaving a slot permanently running.

### Phase 2: restricted plan compilation

Compile the experimental `Struct<Chunked<Flat>>` model into canonical Flat resources, field-chunk
row offsets, cross-chunk morsels, possible-user sets, and per-morsel reverse resource lists.

Exit when global/local row mapping, resource interning, possible users, and graph-size accounting
are independently tested.

### Phase 3: task protocol and reference evaluator

Implement `Read`, `DecodeFlat`, `EvaluatePredicate`, `CombineDemand`, `SelectFlat`, and
`PackStruct`. Every offer declares only segment or array inputs and exactly one segment or array
output. Claiming an offer resolves and clones those inputs into an immutable runnable task while
acquiring their leases. Keep operation variants inside the fixed experimental evaluator rather
than the central completion protocol.

Exit when every operation can be claimed, run outside the reactor without slot-store access, and
return a validated completion.

### Phase 4: bounded per-morsel `advance`

Implement a dirty queue, transition budget, task emission, promotion and revocation updates,
demand-version adoption, sealing, and root output. `advance` performs control transitions only and
never evaluates or intersects arrays.

Exit when unrelated clean nodes are not visited, broad demand remains compact, and small and large
budgets eventually produce identical output.

### Phase 5: Flat, Struct, and Chunked input

Flat selects a morsel-specific array from a shared decode. Struct packs aligned field arrays.
Each field's Chunked layout maps the morsel range to an ordered list of Flat slices. A morsel may
therefore cross one or more aligned field-chunk boundaries; the evaluator concatenates selected
values from those slices before Struct packs the fields.

Exit when every node output and the root result match the eager reference evaluator.

### Phase 6: cross-morsel reuse and retirement

Track possible users, joined users, and outstanding task leases. Implement pinned, reusable, and
dead classifications plus retain-until-dead and evict-when-unpinned policies. Retirement walks a
morsel's reverse resource list rather than the whole graph.

Exit when later morsels reuse retained handles and arrays, eviction causes legal rereads, and no
dead or pinned resource is reclaimed incorrectly.

### Phase 7: scheduler and control-plane benchmarks

Add deterministic schedulers, virtual costs, metrics, traces, randomized completion schedules, and
the Divan control-plane benchmark. Produce the worked trace, policy table, scaling table, and
findings record described below.

Exit when all schedules match the eager result and the boundedness and lifetime questions have
measured answers.

### Phase 8: V1 optimized baseline

Adapt the experiment to consume the same real serialized layout, `SegmentSource`, `BufferHandle`
values, and Vortex array decoding as the [V1 `LayoutReader`](layout-reader-v1.md). Add the
apples-to-apples benchmark described below only after the control-plane experiment passes.

Exit when V1 and self-paced execution produce the same ordered logical output and their time, I/O,
reuse, first-batch, and memory measurements use identical fixtures and source-cache policies.

## Static plan and query

The experiment can use a small source-plan model:

```rust
enum SourcePlan {
    Flat(FlatPlan),
    Struct(StructPlan),
    Chunked(ChunkedPlan),
}

struct FlatPlan {
    segment: SegmentId,
    row_count: usize,
}
```

`ChunkedPlan` assigns each child a range in the root row domain:

```text
chunk 0: root rows [0, 8)
chunk 1: root rows [8, 16)
```

`StructPlan` declares that all field children share its row domain. A flat child identifies the
physical segment whose decoded values cover that domain.

The query is deliberately small:

```rust
struct ScanQuery {
    conjuncts: Vec<Conjunct>,
    projection: Vec<FieldId>,
}

struct Conjunct {
    field: FieldId,
    predicate: Predicate,
}

enum Predicate {
    Equal(i64),
    LessThan(i64),
    GreaterThan(i64),
}
```

Compilation validates the restricted source shape, creates one global resource node per canonical
flat segment, and determines every potential field use:

```text
a: predicate P0 and projection
b: predicate P1
c: projection
```

It records each resource's root-row coverage. That coverage maps morsels into segment-local rows
and identifies which unfinished morsels might reuse a result.

## Morsels and shrinking demand

Morsels partition the root row domain independently of aligned field-chunk boundaries. For two
eight-row chunks and a ten-row target:

```text
chunk 0: root rows [0, 8)
chunk 1: root rows [8, 16)

morsel 0: [0, 10)   # slices chunk 0 [0, 8) and chunk 1 [0, 2)
morsel 1: [10, 16)  # slices chunk 1 [2, 8)
```

Every morsel starts with an over-approximation containing all its rows:

```text
Demand0 = all rows in the morsel
Demand1 = Demand0 ∩ matches(P0)
Demand2 = Demand1 ∩ matches(P1)
```

The reactor checks:

```text
Demand0 ⊇ Demand1 ⊇ Demand2 ⊇ ... ⊇ SealedDemand
```

Resolved demand is a non-nullable boolean `ArrayRef`, held in a morsel-owned array slot:

```rust
struct DemandState {
    mask: ArraySlotId,
    version: DemandVersion,
    true_count: usize,
    sealed: bool,
}
```

`advance` does not evaluate or intersect these boolean arrays. Predicate evaluation produces a
boolean `ArrayRef`; `CombineDemand` is a CPU task that intersects the old demand with one or more
predicate results and produces another boolean `ArrayRef`. Every boolean-mask result carries a
mandatory `BooleanMaskSummary` beside its `ArrayRef`, so planning need not scan the array.

Demand seals when every correctness-relevant conjunct has completed or has been proven
unnecessary. Empty demand can seal immediately because no projection values are needed. Sealing
empty also revokes the morsel's unclaimed offers, marks running morsel-local results unwanted, and
removes the morsel from every resource's user sets. Task leases may keep a resource live until
claimed work finishes even when a speculative projection resource has no remaining morsel user.

Each predicate task receives an immutable demand snapshot. Concurrent predicates can run against
different supersets of the current demand. Their boolean-array results remain valid because a
`CombineDemand` task intersects them with the current, smaller demand; removed rows never re-enter
demand. Within one morsel the snapshots form a single chain ordered by `⊇`, so a result's
validity never depends on which snapshot its task happened to read.

## The global resource graph

The graph primarily records reusable-result lifetime across morsels. It is not an eagerly
materialized graph containing one task for every possible row or segment.

Each flat segment has a resource node resembling:

```rust
struct ResourceNode {
    segment: SegmentId,
    root_coverage: Range<u64>,
    segment_slot: SegmentSlotId,
    array_slot: ArraySlotId,
    unresolved_users: MorselSet,
    joined_users: MorselSet,
    state: ResourceState,
}
```

`unresolved_users` conservatively contains unfinished morsels that might use the resource.
`joined_users` contains morsels that have established an actual dependency.

```text
may_use = unresolved_users ∪ joined_users
```

This set never grows. A morsel either:

- moves from unresolved to joined when it discovers a use;
- is removed from unresolved when it proves no use is possible; or
- is removed from joined when it retires and releases its use.

Discovering a use therefore joins an explicit edge without increasing conservative global demand.

`joined_users` doubles as the wake-up list for shared completions: installing a value into one of
the node's slots marks every joined morsel dirty exactly once. Unresolved morsels are not woken;
they observe the resolved slot if and when they join.

## Resource state and lifetime

Result availability progresses independently of demand. The node's scan-owned slots remain the
only value store; `ResourceState` is a compact view over those slots and their task leases:

```rust
enum ResourceState {
    Absent,
    Reading(TaskId),
    SegmentReady,
    Decoding(TaskId),
    ArrayReady,
}
```

The lifetime classification is:

```text
Pinned
    At least one joined morsel or claimed task lease uses the result.

Reusable
    No joined morsel or claimed task lease uses it, but an unresolved morsel may use it later.

Dead
    No joined or unresolved morsel can use it and no claimed task lease holds it.
```

Pinned results cannot be discarded. Reusable results are cacheable but evictable under memory
pressure. Dead results should be discarded immediately. If a reusable result is evicted and a
later morsel joins the node, the reactor emits its work again.

Suppose morsels 0 and 1 share flat segment `a0`:

```text
initial:            unresolved={0,1}, joined={}
morsel 0 activates: unresolved={1},   joined={0}
morsel 0 retires:   unresolved={1},   joined={}     # reusable
morsel 1 activates: unresolved={},    joined={1}    # reuse result
morsel 1 retires:   unresolved={},    joined={}     # dead
```

The experiment may materialize `MorselSet` because its inputs are small. A production design would
use root-row ranges, chunk summaries, or hierarchical bitmaps so retirement need not scan every
resource node.

## Operator outputs and transformed arrays

Resource facts are not plan outputs. A decoded flat segment is reusable input; Flat selection and
Struct packing still need to produce morsel-local arrays. Chunked is compiled into the ordered Flat
slices consumed by those operations.

The experiment therefore contains two connected graphs:

```text
scan-wide resource graph
    BufferHandle -> decoded ArrayRef
    retained and reused across morsels

per-morsel operator graph
    Flat selections -> Struct output
    retired with the morsel
```

The complete resolved-value vocabulary is:

```rust
enum ResolvedValue {
    Segment(BufferHandle),
    Array(ResolvedArray),
}
```

The implementation should preserve type safety with two slot arenas rather than storing this enum
inside every slot:

```rust
struct SegmentSlot {
    state: SlotState<BufferHandle>,
}

struct ArraySlot {
    state: SlotState<ResolvedArray>,
}
```

Slots are the only value store. The scan owns the arenas holding shared resource slots; each
morsel owns the arenas holding its operator-output and demand slots. A slot identifier names its
scope, so retirement frees exactly the morsel-owned slots and can never reclaim a shared resource
by accident.

Every per-morsel plan node names an `ArraySlotId` for its output. Coverage and selection belong to
the node-edge metadata rather than forming another resolved value type:

```rust
struct NodeOutput {
    array: ArraySlotId,
    coverage: Range<u64>,
    selection: ArraySlotId,
}
```

The selection slot contains a boolean `ArrayRef`. `coverage` advances the plan even when selection
is sparse or empty. The selection lets a parent align independently produced fields, and the
output array length equals the selection's recorded `true_count`.

Only the root wraps these pieces for the scan caller:

```rust
struct ExecBatch {
    coverage: Range<u64>,
    selection: ArrayRef,
    array: ArrayRef,
}
```

`ExecBatch` is an output envelope, not a third resolved value used by future tasks.

### Flat output

Flat separates the reusable decoded segment from its morsel-specific output:

```text
Read(segment a0)
    -> shared BufferHandle

DecodeFlat(a0)
    -> shared decoded ArrayRef

SelectFlat(a0, local range, demand)
    -> per-morsel ArrayRef
```

Predicate tasks may read the shared decoded array directly with a local range and immutable demand
snapshot. Once demand seals, `SelectFlat` slices and gathers projection values into a compact field
batch. This avoids confusing a reusable whole-segment decode with the array returned by the Flat
plan for one morsel.

### Struct output

Struct sends the same row coordinates to its requested fields. When every projected field has a
batch with identical coverage and selection, it exposes a `PackStruct` CPU task. That task creates
the `StructArray` and fills the Struct array slot.

```text
Flat(a) batch ─┐
               ├─ PackStruct ─> Struct ArrayRef
Flat(c) batch ─┘
```

Struct does not read or decode data itself. It routes field demand, waits for aligned field
outputs, and describes the array assembly work.

### Chunked input

Chunked translates a root morsel range into ordered Flat slices. The fields of the Struct must have
aligned chunk boundaries, but a morsel need not align with them. Predicate evaluation and Flat
selection consume all overlapping slices and preserve root row order.

## Scheduler-visible work

The reactor exposes one ticket type covering both I/O and CPU work, plus the two lifecycle updates
that an external scheduler must observe:

```rust
struct Task {
    id: TaskId,
    class: WorkClass,
    necessity: Necessity,
    inputs: Vec<InputSlot>,
    output: OutputSlot,
    operation: Operation,
}

enum TaskUpdate {
    Offer(Task),
    Promote(TaskId),
    Revoke(TaskId),
}

enum WorkClass {
    Io,
    Cpu,
}

enum Necessity {
    Required,
    Candidate,
}

enum Operation {
    Read(ReadOp),
    DecodeFlat(DecodeOp),
    EvaluatePredicate(PredicateOp),
    CombineDemand(CombineOp),
    SelectFlat(SelectOp),
    PackStruct(PackOp),
    ConcatChunks(ConcatOp),
}
```

`Necessity` matches the architecture's candidate/required split. `Promote` changes a retained
offer from candidate to required without changing its identity. `Revoke` removes an unclaimed
offer that demand has made unnecessary. `ConcatChunks` remains reserved: cross-chunk morsels are
handled by the multi-slice predicate and Flat-selection operations, so the experiment does not
produce a per-chunk output that needs concatenation. The operation enum belongs to the
experiment's fixed reference evaluator, not the central completion protocol. Every task declares
only segment or array inputs and exactly one segment or array output:

```rust
enum InputSlot {
    Segment(SegmentSlotId),
    Array(ArraySlotId),
}

enum OutputSlot {
    Segment(SegmentSlotId),
    Array(ArraySlotId),
}
```

Operation payloads also contain immutable plan metadata such as row ranges, dtypes, or predicates.
An offered `Task` is descriptive and contains slot identifiers, not the slot values. Immediately
before execution, the scheduler claims it:

```rust
fn claim(&mut self, task: TaskId) -> VortexResult<ClaimResult>;

enum ClaimResult {
    Runnable(RunnableTask),
    Revoked,
}

struct RunnableTask {
    id: TaskId,
    inputs: Vec<ResolvedValue>,
    output: OutputSlot,
    operation: Operation,
}
```

Claiming is a cheap owner-thread transition. It atomically checks that the offer is still live,
clones each resolved `BufferHandle` or `ArrayRef`, changes the task from offered to running, and
acquires input and output leases. The runnable task owns those immutable clones and never accesses
the slot store or morsel reactor. Revoking an offered task releases its output reservation. A
running morsel-local task cannot be revoked; empty sealing or retirement marks its result unwanted,
and its eventual completion releases the leases without installing the result. Shared reads and
decodes continue normally because another morsel may still use their result.

For the worked query's first advance:

```text
required I/O:  read a for P0; read b for P1
candidate I/O: read c for the projection
```

Available segments enable `DecodeFlat`. Decoded predicate fields enable independent
`EvaluatePredicate` tasks, whose boolean arrays feed `CombineDemand`. The scheduler may run
predicates sequentially or concurrently. Projection field I/O and shared decode can happen under
open demand, but the first experiment waits for sealed demand before emitting `SelectFlat` and
`PackStruct` work.

## Incremental `advance`

```rust
fn advance(
    &mut self,
    morsel: MorselId,
    transition_budget: usize,
) -> VortexResult<AdvanceResult>;
```

`advance` performs bounded planning and cheap state transitions only. It does not perform segment
I/O, decode arrays, evaluate predicates, or gather projection values.

One invocation:

1. observes already-submitted completions;
2. refines current demand;
3. joins resources required by visible work;
4. exposes reads for missing resources;
5. exposes decode work for available `BufferHandle` values;
6. exposes predicates whose inputs are decoded;
7. exposes `CombineDemand` when predicate arrays are ready;
8. adopts completed demand-array slots and seals when all conjuncts resolve;
9. promotes or revokes retained candidate offers that sealing affected;
10. exposes Flat selection and Struct packing work for sealed demand;
11. propagates completed operator arrays toward the Chunked root;
12. returns the root batch when it is ready; and
13. stops at local quiescence or the transition budget.

```rust
struct AdvanceResult {
    work: Vec<TaskUpdate>,
    output: Option<ExecBatch>,
    state: MorselState,
}

enum MorselState {
    /// The transition budget expired; call `advance` again without waiting.
    Budgeted,
    /// No transition is possible until offered or running work makes external progress.
    Quiescent,
    /// The final output batch was returned; morsel slots are freed, and any leases still held
    /// by unwanted running work release as their completions drain.
    Retired,
}
```

Demand remains compact. A broad range controls a lazy task source; it does not build a task node
for every row or segment. The transition budget bounds planning effort and newly emitted updates
per call; it does not bound the number of offers the scheduler may retain across calls. `Budgeted`
and `Quiescent` mirror the architecture's `locally_quiescent` flag: work already returned remains
schedulable in both states, so exhausting the budget never hides an exposed task.

The external scheduler separately controls admission with a maximum number of claimed/running
tasks or equivalent I/O and CPU credits. The small-frontier policy uses that admission limit, not
the reactor's transition budget. Repeated bounded calls may therefore reveal all concrete work
while the scheduler still admits only as much execution as its queues can support.

### Streamed output batches

One root batch per morsel is a reduction, not the contract. A morsel is the scheduling and
ordering unit; its output should stream as ordered dense-prefix `ExecBatch` values while demand
fragments seal. The architecture's `PlanStep` already returns a vector of sealed prefixes, and
fragment sealing gives the experiment natural emission points. Streaming bounds retained output
memory, lets downstream consumers start before a morsel finishes, and makes time-to-first-batch
measurable. Under the streamed contract, `output` carries the prefixes sealed by this call and
`Retired` means the final prefix was emitted; the single-batch behavior remains the valid
degenerate stream. The pipeline executor implements the contract: `MorselPipeline` emits ordered
prefixes through a batch sink, one per chunk-boundary span shared by every projected field, and
releases each span's decoded chunks at emission. The reactor modes still emit one batch per
morsel.

## Completion and external evaluation

The scheduler reports task completion separately:

```rust
fn complete(&mut self, completion: Completion) -> VortexResult<()>;
```

Workers do not mutate the reactor or slot store. A completion carries exactly one of the two
resolved value types back to the execution owner:

```rust
struct Completion {
    task: TaskId,
    output: OutputSlot,
    result: VortexResult<ResolvedValue>,
}
```

The task table already records its inputs, output, owning node, and demand version, so the
completion protocol does not repeat semantic resource, conjunct, or node variants. The owner
validates the output kind, installs the value into its typed write-once slot, and marks dependants
dirty. Completion does not recursively drive the reactor.

An array result stores its value and inseparable metadata in the array slot:

```rust
struct ResolvedArray {
    array: ArrayRef,
    summary: ArraySummary,
}

enum ArraySummary {
    None,
    BooleanMask(BooleanMaskSummary),
}

struct BooleanMaskSummary {
    len: usize,
    true_count: usize,
}
```

`ResolvedValue::Array` contains `ResolvedArray`; it is still the array resolved-value kind, and no
task can name the summary separately. The task table declares the required summary shape. Every
predicate and demand-combination result must return a boolean-mask summary whose length agrees
with the array and whose count is in range. Like the array's values, the producer is responsible
for the summary's semantic correctness. The summary is correctness-bearing because empty sealing
uses it; it is not merely scheduling metadata.

Task identifiers and demand versions let the reactor reject duplicate or mismatched completions.
Because demand versions within one morsel only shrink, a predicate array produced from any earlier
version is a superset of current demand and remains usable. Staleness is therefore confined to
duplicates, wrong-slot completions, revoked offers, and unwanted results from running
morsel-local tasks.

Empty sealing and retirement revoke the morsel's offered tasks and emit `Revoke` updates. Running
morsel-local tasks are marked unwanted; their completions release their leases and install
nothing. One morsel never revokes shared resource work: a claimed read or decode completes into its
scan-owned slot, and the value is then classified pinned, reusable, or dead by its remaining users.
Input leases keep resources live even after their last morsel user disappears.

The experiment includes a reference evaluator, but the reactor never calls it:

```rust
fn evaluate(
    task: RunnableTask,
    segments: &InMemorySegments,
) -> VortexResult<Completion>;
```

A complete driver remains external:

```rust
loop {
    let progress = execution.advance(morsel, 16)?;

    scheduler.apply(progress.work);

    while let Some(task) = scheduler.next_admissible() {
        if let ClaimResult::Runnable(task) = execution.claim(task)? {
            execution.complete(evaluate(task, &segments)?)?;
        }
    }

    if let Some(batch) = progress.output {
        break batch;
    }
}
```

Alternative schedulers can choose I/O-first, predicate-first, projection-prefetch, or concurrent
policies without changing the reactor.

## Worked execution

For `a > 10 AND b < 5`, projecting `[a, c]`, one possible trace is:

```text
advance(morsel 0)
  -> Read(a0, required)
  -> Read(b0, required)
  -> Read(c0, candidate)

complete and decode a0
advance(morsel 0)
  -> Predicate(P0, demand=1111)

complete P0(array=0101)
advance(morsel 0)
  -> CombineDemand(1111, P0)

complete combined demand
advance(morsel 0)
  -> current demand=0101

complete and decode b0
advance(morsel 0)
  -> Predicate(P1, demand=0101)

complete P1(array=0001)
advance(morsel 0)
  -> CombineDemand(0101, P1)

complete combined demand
advance(morsel 0)
  -> sealed demand=0001
  -> Promote(Read(c0)), if the candidate read is still queued
  -> SelectFlat(a0, demand=0001)
  -> SelectFlat(c0, demand=0001), once c0 is decoded

complete Flat outputs
advance(morsel 0)
  -> PackStruct([a, c], demand=0001)

complete Struct output
advance(morsel 0)
  -> output one row
  -> retire morsel 0
```

If P1 ran earlier against `1111`, its result would remain valid and produce the same final
intersection. Morsel 1 can subsequently join resident `a0`, `b0`, and `c0` results without emitting
another read or decode.

## Measurements

The experiment exposes an event trace and metrics snapshot containing:

```text
advance calls and cheap transitions
nodes and slots inspected per advance
initial, current, and per-conjunct demand rows
I/O and CPU tasks offered, claimed, promoted, revoked, and completed, including demand combinations
shared byte and decode reuse hits
resident bytes and decoded rows
pinned, reusable, and dead resource counts
Flat and Struct output tasks and arrays
maximum offered and claimed/running task frontiers
total graph nodes, slots, and explicit edges
useful and unused speculative I/O bytes
deterministic virtual critical-path time
```

The trace records demand refinement, resource joins, task emission, claims, promotion,
revocation, completion, sealing, output, retirement, and eviction.

## Evaluation policies

Run the same input through four small external schedulers:

1. **Predicate-first:** prioritize the earliest unfinished conjunct.
2. **All-ready:** run every offered task before calling `advance` again.
3. **Projection-prefetch:** start projection I/O when the required queue is short.
4. **Small-frontier:** admit only a small number of claimed/running tasks while continuing bounded
   planning to quiescence.

All policies must produce identical rows and values. Their task counts, resident state, demand
reduction timing, and reuse rates may differ.

## V1 optimized baseline

After the control-plane experiment passes, compare it with the current optimized
[V1 `LayoutReader`](layout-reader-v1.md). This is a valid performance comparison only when both
paths consume the same serialized Vortex layout and segment source. The initial raw-`i64` toy
evaluator can establish semantics but cannot be timed fairly against V1.

### Shared real fixture

Construct one deterministic serialized fixture:

```text
16 chunks
65,536 rows per chunk
1,048,576 rows total
Struct<a: i64, b: i64, c: i64, d: i64>
one Flat segment per field per chunk
```

Generate `a` and `b` distributions that produce 1%, 50%, and 95% predicate selectivity. Both paths
receive the same:

- stored layout and segment buffers;
- filter and projection expressions;
- row ranges and ordered-output requirement;
- runtime concurrency;
- segment-source wrapper; and
- source-cache settings.

Executor-native decoded-array retention is recorded but is not forced to be identical. If V1 does
not provide such a cache, adding one in the harness would change the baseline rather than make it
fair.

The self-paced evaluator must use real `BufferHandle` values, serialized-array decoding, and
Vortex `ArrayRef` operations for this stage.

### V1 execution

Run V1 through `ScanBuilder` with natural layout-chunk boundaries computed once from
`SourcePlan::chunks`. Pass them explicitly so `SplitBy::Layout` cannot silently subdivide wide
chunks. Morsels are an implementation choice of self-paced execution and are never imposed on V1:

```rust
ScanBuilder::new(session, layout_reader)
    .with_filter(filter)
    .with_projection(projection)
    .with_natural_splits(Arc::clone(&fixture.natural_splits))
    .with_concurrency(concurrency)
    .into_array_stream()
```

Batch boundaries may still differ. Compare ordered logical output after concatenation rather than
requiring identical batches.

### Comparison matrix

Run each case through V1 and self-paced execution:

| Case | Filter | Projection | Information gained |
| --- | --- | --- | --- |
| Unfiltered | none | `[a, c, d]` | Minimum self-paced control-plane tax |
| Filter fields projected | `a > x AND b < y` | `[a, b]` | Reuse when filter and output share decoded fields |
| Separate projection | `a > x AND b < y` | `[c, d]` | Late-materialization opportunity |
| Highly selective | `P0=1%, P1=50%` | `[c, d]` | Avoidable projection I/O and CPU |
| Medium selective | `P0=50%, P1=50%` | `[c, d]` | Balanced scheduling case |
| Non-selective | `P0=95%, P1=95%` | `[c, d]` | Projection-prefetch and concurrency opportunity |

For every case, use:

```text
morsel rows: 4,096; 16,384; 65,536; 131,072
concurrency: 1; 4; 16
```

This produces 54 configurations per executor before cache variants.

### Cache variants

Run each important configuration in three comparable source states:

```text
cold
    fresh LayoutReader or self-paced graph, empty segment cache

warm structure
    reader or plan structure reused, resolved segment and array values evicted

warm source data
    source BufferHandle values retained; executor-native decoded values follow each executor's
    normal behavior
```

Also run both paths over a raw counting source and an equivalently shared/cached source. This
prevents either engine from winning only because it received a hidden cache unavailable to the
other.

Run self-paced decoded-`ArrayRef` retention as a separate capability experiment. Report its gain,
resident memory, and avoided decodes relative to self-paced execution without decoded retention;
do not present it as a cache-parity V1 ratio unless V1 later gains an equivalent public facility.

### Common instrumentation

Wrap the shared source to record:

```text
segment requests
unique segments
bytes returned
repeated requests
peak outstanding reads
```

For both engines record:

```text
total wall time
time to first output batch
rows per second
output rows and stable output hash
segment requests and bytes
output batch count
```

Additionally record self-paced transitions, predicate rows evaluated, speculative bytes used and
wasted, decode reuse, and graph/resource memory. Those internal metrics explain the common
headline measurements but are not themselves a direct V1 comparison.

### Baseline benchmark command

Add `vortex-file/benches/self_paced_vs_v1.rs` and run:

```bash
cargo bench -p vortex-file --bench self_paced_vs_v1
```

The benchmark must consume every output and compare its stable ordered hash before accepting its
timing. Report the result as ratios rather than isolated numbers:

| Scenario | Self-paced/V1 time | I/O ratio | First-batch ratio | Peak-memory ratio |
| --- | ---: | ---: | ---: | ---: |
| Unfiltered | | | | |
| 1% selective | | | | |
| 50% selective | | | | |
| 95% selective | | | | |
| Warm source reuse | | | | |

V1 is expected to win the first unfiltered, single-thread comparison. The experiment is promising
only if its overhead is bounded and any selective, high-latency, or reuse win is explained by
measured avoided work rather than unequal caching or inputs.

## Experiment summary

The experiment compiles one restricted `Struct<Chunked<Flat>>` source and a conjunctive query into
two kinds of runtime state:

```text
scan-wide resource graph
    owns possible and joined users
    retains segment handles and decoded arrays for possible reuse

per-morsel operator graph
    owns shrinking row demand
    runs conjuncts
    produces Flat, Struct, and Chunked ArrayRef values
```

The complete control and data flow is:

```text
root morsel demand
    -> Struct routes open demand to predicate and projection fields
    -> each field's Chunked layout maps root rows to one or more Flat slices
    -> Flat slices join canonical segment resources
    -> advance exposes Read and DecodeFlat tasks
    -> decoded arrays enable independent predicate tasks
    -> predicate arrays feed CombineDemand tasks
    -> resolved boolean arrays monotonically shrink demand
    -> all conjuncts resolved, so demand seals
    -> sealing promotes retained candidate offers to required
    -> Flat selects projected values for sealed demand
    -> Struct packs aligned field batches into a StructArray
    -> root wraps its ArrayRef and row metadata in ExecBatch
    -> morsel retires and releases resource joins
```

At no point does `advance` perform I/O or significant CPU work. It exposes immutable task tickets,
observes completed typed slots, performs bounded state transitions, and returns a batch only when
the root operator output slot is ready.

The scheduler controls task order. It can evaluate conjuncts sequentially, run them concurrently,
or prefetch projection data. Correctness depends only on shrinking demand, stable task inputs, and
validated completions, not on a particular scheduling policy.

## What the experiment should teach us

The experiment is intended to answer these design questions with traces and measurements rather
than intuition:

### Evidence matrix

| Information wanted | How the experiment obtains it | Evidence and decision |
| --- | --- | --- |
| Whether two resolved types are sufficient | Assert that every task input and output names only a `SegmentSlot` or `ArraySlot`; implement demand as boolean arrays | Any need for another reusable fact type is recorded as a failed assumption rather than added silently |
| Whether results are correct under scheduler freedom | Run FIFO, reversed, randomized, predicate-first, all-ready, and projection-prefetch schedules against the same inputs | Compare root arrays and selections with a simple eager reference evaluator |
| Whether stale conjunct work is safe | Start predicates from the same broad demand, complete them in every order, and combine their arrays after other demand refinements | Every schedule must produce the same sealed boolean array; rejected results identify missing version or superset rules |
| Whether `advance` is bounded | Sweep chunk, morsel, field, and conjunct counts while recording visited nodes, transitions, and emitted tickets per call | Work per call must be bounded by the transition budget plus a small dirty-frontier overhead, not total graph size |
| Whether the graph is compact | Increase row count without changing segments, then increase segments and morsels independently | Slot and node counts should follow plans, canonical resources, and active frontiers rather than logical row count |
| Whether cross-morsel retention is useful | Vary morsels per chunk, activation order, and cache budget | Measure byte/decode reuse, rereads, retained-byte-morsel time, and peak resident state |
| Whether retirement proves death cheaply | Retire morsels in ordered and randomized sequences | A resource becomes dead exactly after its last possible user disappears, without scanning the whole graph |
| Whether speculative projection is worthwhile | Sweep predicate selectivity, I/O latency, CPU cost, and projection overlap | Compare output-ready virtual time, unused speculative bytes, avoided reads, and peak memory |
| Whether operators compose cleanly | Validate Flat arrays, Struct packing, Chunked coordinate translation, and the root result independently | Each node should depend only on child slots and declared row-domain metadata; query policy leakage is a design failure |
| How the model compares with optimized V1 | Run both over the same serialized layout, source, expressions, row splits, concurrency, and source-cache policy | Compare output hashes, time, first batch, I/O, reuse, and memory as self-paced/V1 ratios |

Use deterministic virtual costs for the first scheduler comparisons. Each I/O and CPU task receives
a configured cost, so critical-path time can be compared without benchmark noise. Real timings are
not an objective until the model is connected to actual Vortex decoding and expressions.

The input matrix should vary:

- chunks and morsels per chunk;
- rows per morsel without changing the plan shape;
- conjunct count, selectivity, CPU cost, and completion order;
- projected fields that overlap or do not overlap predicate fields;
- segment I/O latency and decode cost;
- scheduler admission limit and reactor transition budget; and
- retained-resource memory budget.

Every run records the event trace, metrics snapshot, root output, and final slot/resource states.
This makes each conclusion reproducible from a small fixture rather than inferred from wall-clock
time alone.

The experiment should produce five review artifacts:

1. a complete worked-query trace showing slots, demand versions, tasks, joins, and retirement;
2. a policy comparison table containing correctness, virtual time, work, reuse, waste, and memory;
3. a scaling table showing graph size and `advance` work as rows, morsels, chunks, and conjuncts
   change;
4. the V1 ratio table with fixture and source-cache parity recorded; and
5. a short findings record that classifies every proposed invariant as supported, rejected, or
   still untested.

### Is `advance` actually cheap and bounded?

Measure transitions and work tickets per call. A broad demand must remain a compact description,
and one call must not walk the complete plan or enumerate every possible future task.

### Is shrinking demand enough to permit flexible scheduling?

Run conjuncts in different orders and concurrently. Results must remain identical when predicate
tasks complete from older demand supersets. The trace should show whether demand versions and
superset validation are sufficient or whether additional dependency state is needed.

### Is the resource/operator split correct?

Shared segment handles and decoded arrays should survive when later morsels may reuse them.
Morsel-specific Flat selections and Struct arrays should retire with their morsel. This reveals
whether the proposed keys and ownership boundaries retain too much or prevent useful reuse.

### Can possible future users drive useful lifetime decisions?

After one morsel retires, a resource should be pinned, reusable, or dead without guessing. The
experiment should quantify how long conservative possible-user sets retain data and whether late
joins achieve enough reuse to justify that retention.

### Are Chunked input and Struct output compositional?

Chunked should only translate root ranges into ordered Flat slices. Struct should only route field
demand and pack aligned outputs. Flat should own the physical resource boundary. If query-specific
state leaks deeply into these nodes, the plan/executor boundary needs revision.

### Does scheduler freedom improve the trade-off?

Compare predicate-first and projection-prefetch policies. The useful result is not merely that
both work, but a measured difference in eliminated rows, avoidable I/O, reuse, latency-hiding, and
peak resident state.

### What must change before production integration?

The experiment should leave a short list of proven interfaces and failed assumptions. In
particular, it should tell us whether to preserve:

- one owner and output slot per per-morsel execution node;
- one canonical resource node per reusable physical result;
- immutable task tickets and completion routing targets;
- open, shrinking, and sealed demand states;
- explicit coverage and selection on every returned batch; and
- bounded lazy task generation from `advance`.

Only those parts demonstrated by the experiment should be carried into real `PlanRef`,
`SegmentSource`, `ArrayRef`, and `BoundExpression` execution.

## What the experiment cannot establish

The first seven phases validate control-plane semantics and compare deterministic scheduling
trade-offs; they cannot establish production throughput or latency. The V1 baseline phase adds
real serialized arrays and decoding, so it can establish a fair in-memory relative cost against
V1. It still does not reproduce object-store or NVMe behavior, general expression workloads,
allocator pressure at production scale, or a complete asynchronous scan runtime. It also cannot
validate late row-domain mappings for Dict, List, or ListView plans, and it exercises none of the
gate, multi-block demand, rescoring, or ownership-migration mechanisms recorded in the
[morsel reactor ideas](morsel-reactor-ideas.md).

Those omissions are deliberate. A successful result means the interfaces and invariants are worth
testing in a complete scan; it does not mean the resulting implementation is already fast or
general.

## Required tests

The experiment should prove:

1. Every task input and output is a segment slot or array slot.
2. `advance` performs no I/O, array evaluation, or demand-array intersection.
3. `CombineDemand` produces boolean arrays whose selected rows never grow.
4. Predicate completion order does not affect sealed demand or output.
5. Projection reads can be visible while demand is open.
6. Flat selection and Struct packing wait for sealed demand.
7. Flat, Struct, and Chunked each fill an array slot with correct coverage and row identity.
8. Later morsels reuse segments and decoded arrays from earlier morsels.
9. Per-morsel transformed arrays are not accidentally retained as scan-wide resources.
10. Retirement changes a resource from pinned to reusable and eventually dead.
11. Dead resource state is released.
12. Chunk-global and segment-local row mappings are correct.
13. The transition budget bounds transitions and newly emitted updates per call, while scheduler
    admission independently bounds claimed/running work.
14. Graph size does not scale with logical row count when physical plan shape is unchanged.
15. Every scheduler policy produces the same output as the eager reference evaluator.
16. Failed, duplicate, and stale completions and claims of revoked offers cannot leave slots or
    resource leases live.
17. A completed shared read or decode wakes every joined morsel exactly once and wakes no
    unresolved morsel.
18. Sealing empty demand revokes unclaimed morsel offers and discards running morsel-local results
    without cancelling shared in-flight work.
19. V1 and self-paced execution produce the same ordered output hash for every shared fixture.
20. Cold, warm-structure, and warm-source-data comparisons use equivalent source state; native
    decoded-array retention is reported separately.
21. Claiming a task snapshots every resolved input and holds its leases until completion.
22. Promotion and revocation updates keep a retained external scheduler queue consistent with
    demand changes.
23. Every boolean predicate and demand result carries the mandatory length and true-count summary
    required for empty sealing.

## Non-goals

The first experiment does not provide:

- a production `PlanVTable` execution interface;
- arbitrary nested layouts or bound expressions;
- nullable or non-`i64` values;
- asynchronous runtime integration;
- object-store or NVMe performance conclusions;
- memory admission or a sophisticated eviction policy;
- cross-file sharing;
- dependency gates for data-dependent reads;
- work rescoring: promotion and revocation are the only lifecycle updates;
- fallible predicate semantics; or
- the final output-stream and ordering contract.

## Decision gate

The experiment should precede production integration. It succeeds if:

- the graph retains reusable state without eagerly constructing future tasks;
- segment and array slots are sufficient for all resolved work values;
- later morsels safely join existing resource nodes;
- demand and possible-user sets remain monotonic;
- retirement proves when results are dead;
- external scheduling policies preserve query results;
- `advance` cost is bounded and measurable, and scheduler admission bounds claimed/running
  work; and
- the V1 comparison has identical output and attributes any difference to measured work, I/O,
  reuse, or control-plane overhead.

Concretely, the evidence must show:

- zero output differences from the eager reference across all tested schedules;
- no task input or output outside the two typed slot arenas;
- no demand version whose selected row set is larger than its predecessor;
- no dropped resource with a joined or possible user, and no retained dead resource after cleanup;
- graph size unchanged when only logical row count grows without changing physical plan shape;
- no `advance` call emitting more work than its budget or visiting unrelated clean subtrees;
- no revoked offer that executes and no promotion that changes a task's identity or duplicates its
  I/O;
- an explicit measured frontier where projection prefetch helps and where it wastes enough I/O or
  memory to be rejected; and
- a V1 ratio table whose unfiltered case exposes the self-paced tax and whose selective or reuse
  differences agree with the recorded I/O, decode, and retention metrics.

If these properties require global rescans, demand growth, or scheduler-specific reactor behavior,
the model should be revised before replacing or extending production `PlanVTable` execution.
