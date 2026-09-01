# Scheduler-Visible Work Inside a Morsel

## Status and scope

This note sharpens one part of the [self-paced execution proposal](self-paced.md): how one morsel
exposes every currently actionable I/O and CPU operation to an external scheduler without running
expensive work in the coordination loop or rescanning the execution tree after every completion.
The [morsel reactor architecture](morsel-reactor.md) consolidates the resulting component and
interaction model. [Morsel reactor ideas](morsel-reactor-ideas.md) records policy alternatives that
remain exploratory. This note remains the upstream comparison, graph-cost analysis, and worked
example.

The required contract is:

> Given durable task results, facts, and demand, advance cheap local state to quiescence and return
> the complete set of currently actionable work. The scheduler chooses which work to admit. When
> admitted work completes, route its result directly to the affected nodes and expose the next
> frontier.

"Complete" means every operation whose address and inputs are currently known. A data-dependent
read cannot be returned before the metadata, offsets, or codes that identify it exist. Such future
work is represented by a named gate. Resolving the gate makes the concrete work visible on the next
advance.

The external scheduler owns:

- I/O and CPU admission;
- concurrency and worker-pool selection;
- priority, fairness, and cancellation;
- compressed, decoded, task, and output credits; and
- deduplication of physical reads across logical uses and morsels.

The morsel reactor owns:

- operator-local progress;
- dependency and fact propagation;
- demand authorization;
- translation across row domains;
- conditional gate expansion; and
- production and alignment of output prefixes.

Cheap coordination may execute inline. Physical I/O, decoding, expression evaluation, array
construction, and other expensive operations must be returned as work.

## Related upstream designs

The comparison below describes upstream source inspected on 20 August 2026. DataFusion's API is
experimental, and DuckDB's source is from the commit pinned by the Vortex
`origin/myrrc/duckdb-2.0` branch, so both may change.

### DataFusion morsel-driven I/O

DataFusion introduced this model as a sequence of changes:

- [use `ParquetPushDecoder` in `ParquetOpener`](https://github.com/apache/datafusion/pull/20839);
- [make the Parquet opener an explicit state machine](https://github.com/apache/datafusion/pull/21190);
- [split Bloom-filter I/O from CPU](https://github.com/apache/datafusion/pull/21285);
- [introduce `Morselizer`, `MorselPlanner`, and `MorselPlan`](https://github.com/apache/datafusion/pull/21327);
- [rewrite `FileStream` around morsels](https://github.com/apache/datafusion/pull/21342); and
- [dynamically schedule files from a shared queue](https://github.com/apache/datafusion/pull/21351).

The current
[`MorselPlanner`](https://github.com/apache/datafusion/blob/main/datafusion/datasource/src/morsel/mod.rs)
performs synchronous CPU planning and returns a `MorselPlan` containing:

```rust
struct MorselPlan {
    morsels: Vec<Box<dyn Morsel>>,                 // CPU-ready output work
    ready_planners: Vec<Box<dyn MorselPlanner>>,  // CPU-ready planning work
    pending_planner: Option<PendingMorselPlanner>,// one I/O future
}
```

A `Morsel` has all required input bytes and may decode them into a `RecordBatch` stream without
performing I/O. A planner is explicitly the unit of I/O, and there is at most one pending I/O
future per planner. DataFusion can have several planners and file streams in flight, and sibling
file streams can take files from a shared work queue.

The Parquet implementation has an explicit
[`ParquetOpenState`](https://github.com/apache/datafusion/blob/main/datafusion/datasource-parquet/src/opener/mod.rs)
whose load states contain I/O futures and whose other states perform CPU planning. After opening,
the
[`ParquetPushDecoder`](https://github.com/apache/datafusion/blob/main/datafusion/datasource-parquet/src/push_decoder.rs)
alternates among `NeedsData`, `Data`, and `Finished`: requested byte ranges are fetched and pushed
back into the decoder until it can produce a batch.

This gives DataFusion three valuable properties:

1. CPU planning does not accidentally hide I/O inside a large async closure.
2. I/O-complete morsels can move independently into CPU decoding.
3. Idle file-stream partitions can take unopened files from busy siblings.

Its present API is narrower than the proposed Vortex contract:

- one planner exposes at most one pending I/O future at a time;
- a `MorselPlan` does not describe an arbitrary mix of several I/O and CPU tasks with costs,
  coverage, and demand authorization;
- `FileStream` owns a single pending planner and an active morsel reader per stream;
- work stealing is currently at the unopened-file level; and
- dependencies and conditional future reads remain inside the planner state machine.

DataFusion is therefore strong evidence for separating CPU and I/O phases and returning work to a
caller. It is not yet a complete scheduler-visible dependency frontier within one morsel.

### DuckDB asynchronous source tasks

The Vortex `origin/myrrc/duckdb-2.0` branch pins DuckDB commit
[`b3062a5e`](https://github.com/duckdb/duckdb/tree/b3062a5e82f50d77ff6e1006a36f645a79bc4936).
The Vortex branch itself is an API-compatibility change: it updates vector integration and removes
optimizer hooks that are absent from that DuckDB revision. The relevant scheduling design is in
the pinned DuckDB source, not in the Vortex branch diff.

DuckDB's
[`AsyncResult`](https://github.com/duckdb/duckdb/blob/b3062a5e82f50d77ff6e1006a36f645a79bc4936/src/include/duckdb/parallel/async_result.hpp)
is closer to scheduler-visible work. A table function can return a blocked result containing a
vector of `AsyncTask` objects:

```cpp
class AsyncTask {
public:
    virtual void Execute() = 0;
    virtual idx_t GetIOSize() const { return 0; }
};

class AsyncResult {
    AsyncResultType result_type;
    vector<unique_ptr<AsyncTask>> async_tasks;
    TaskSchedulerType pool_type;
};
```

The
[`PhysicalTableScan`](https://github.com/duckdb/duckdb/blob/b3062a5e82f50d77ff6e1006a36f645a79bc4936/src/execution/operator/scan/physical_table_scan.cpp)
schedules those tasks through the executor and returns `SourceResultType::BLOCKED`. If the executor
cannot accept the asynchronous path, it can run the tasks synchronously. A taskless blocked result
means the function registered its own wake-up through the interrupt state.

This gives DuckDB properties worth retaining:

1. One source call can expose several independent tasks.
2. Tasks can select a worker pool and report known I/O size.
3. The pipeline parks instead of occupying a worker while the source is blocked.
4. Completion wakes the interrupted pipeline task.

It is still not the full proposed Vortex contract:

- `BLOCKED` and output are exclusive results;
- `AsyncTask` is executable but semantically opaque to the scheduler beyond pool and I/O size;
- the API does not expose row-domain coverage, candidate versus required status, or sealed demand;
- conditional dependencies remain private source state; and
- waking the pipeline does not itself identify the smallest affected operator inside a Vortex
  execution graph.

DuckDB demonstrates that returning a vector of work to an executor is practical. Vortex needs a
richer, descriptive work item and finer completion routing because compressed layout operators
compose inside one scan source.

### Comparison

| Property | DataFusion | DuckDB pinned revision | Proposed Vortex |
| --- | --- | --- | --- |
| CPU and I/O distinguished | Yes | Yes, by task/pool convention | Yes, explicit `WorkKind` |
| Several ready CPU items | Ready planners and morsels | Vector of `AsyncTask` | Arbitrary ready work set |
| Several ready I/O items from one activation | At most one per planner | Yes | Yes |
| Work returned as data | Partly | Executable task objects | Descriptive work items |
| Output and new work together | `MorselPlan` can hold morsels and planners | No, blocked or output | Yes |
| Conditional future work | Planner state | Source state | Named gates and fact subscribers |
| Demand attached to work | No exact row-mask capability | No | Candidate or sealed authorization |
| Coverage attached to work | File/morsel structure | Internal source state | Domain and dense row coverage |
| Completion routing | Poll pending planner/stream | Wake pipeline task | Direct task-to-fact-to-node routing |
| Current stealing scope | Unopened files across siblings | Pipeline tasks | Tasks across nodes, morsels, and scans |

## Proposed Vortex contract

The current self-paced proposal registers work through `DriveContext` and returns one of `Batch`,
`Blocked`, `Done`, or `Yield`. The stronger contract returns work, output, and waits together:

```rust
trait MorselReactor {
    fn resolve(
        &mut self,
        task: TaskId,
        result: TaskResult,
    ) -> VortexResult<()>;

    fn advance(
        &mut self,
        transition_budget: usize,
    ) -> VortexResult<PlanStep>;
}

struct PlanStep {
    ready: Vec<WorkItem>,
    output: Vec<ExecBatch>,
    gates: Vec<PendingGate>,
    locally_quiescent: bool,
    done: bool,
}
```

`advance` performs only bounded, cheap state transitions. It returns every work item discovered
before local quiescence or budget exhaustion. If the transition budget is exhausted,
`locally_quiescent` is false and the coordinator should be queued again immediately; already
discovered work remains available to the scheduler.

A work item is descriptive and has a stable identity:

```rust
struct WorkItem {
    id: WorkId,
    owner: ExecNodeId,
    kind: WorkKind,
    phase: WorkPhase,
    necessity: Necessity,
    coverage: DomainCoverage,
    authorization: DemandAuthorization,
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

enum DemandAuthorization {
    Open { generation: u64 },
    Sealed(OwnedSealedDemand),
    InfallibleMetadata,
}
```

Returning the same item again is harmless. `WorkId` is stable until the item completes, is
eliminated by demand, or the morsel is cancelled. Promoting a candidate read to required preserves
its identity and physical `ReadKey`, so speculative and blocking uses cannot issue duplicate I/O.

CPU work should be large enough to justify external scheduling. Cursor movement, fact publication,
mask slicing, gate expansion, and cached-frontier comparison remain inline. Decode, expression
evaluation, gather, and material array construction normally become `CpuSpec` values.

## Incremental dependency graph

`advance` must not start at the root and recursively inspect the complete operator tree after each
completion. Opening a morsel compiles the plan into an indexed reactor:

```rust
struct ExecGraph {
    nodes: SlotMap<ExecNodeId, NodeState>,
    facts: SlotMap<FactId, FactSlot>,
    tasks: SlotMap<TaskId, TaskSlot>,
    runnable: VecDeque<ExecNodeId>,
    queued: BitSet,
}

struct FactSlot {
    value: Option<FactValue>,
    generation: u64,
    subscribers: SmallVec<[ExecNodeId; 2]>,
}

struct TaskSlot {
    owner: ExecNodeId,
    output: FactId,
    state: TaskState,
}
```

Resolving a task performs:

1. index `TaskSlot` directly by `TaskId`;
2. store an owned result or result handle in its output `FactSlot`;
3. enqueue only that fact's subscribers, coalescing duplicate enqueues; and
4. drain the dirty-node queue until it is empty or the transition budget expires.

Node transitions may publish facts, which enqueue further subscribers. A parent subscribes to
cached child-frontier facts; it does not recursively drive the child to discover whether it
changed. A gated node subscribes to the fact that resolves its gate. Demand consumers subscribe to
the precise demand event they need.

```text
task completion
      |
      v
TaskId -> output FactId -> subscribers -> dirty-node queue
                                         |
                                         +-> new work
                                         +-> new facts
                                         +-> output prefix
                                         +-> resolved gates
```

Events are wake-up hints; fact and task slots are durable truth. Duplicate or coalesced wakes do
not change semantics.

### Demand subscriptions

Demand events should not wake every projection node:

| Demand change | Subscribers |
| --- | --- |
| Open candidate mask shrinks | Next predicate stage for that block |
| Predicate set for a block becomes empty | Demand ledger |
| Contiguous sealed frontier advances | Projection root and required-read promotion |
| Block becomes exactly empty | Read-catalog elimination queue |
| Summary generation changes | No eager catalog walk; entries rescore lazily on admission |

This preserves exact masks for correctness without turning every mask revision into a graph walk.

## How expensive is the dependency graph?

The graph must have one node per operator state and one slot per live task or fact, never one node
per row. Page and segment reads are catalog entries or task slots keyed by ranges, not permanent
execution nodes.

Let:

- `P` be the number of physical operator nodes;
- `E` be the number of operator edges and fact subscriptions;
- `T` be the number of live I/O and CPU tasks;
- `S` be the subscribers of one completed fact; and
- `D` be the local node transitions caused by that completion.

The expected costs are:

| Operation | Cost |
| --- | --- |
| Compile immutable plan metadata per scan | `O(P + E)` |
| Open mutable state for one morsel | `O(P + E)` initialization, with static subscriptions copied from a template |
| Insert or look up a stable task | Amortized `O(1)` using a slot map plus task-key index |
| Resolve one task | `O(1 + S + D)` |
| Drain to local quiescence | `O(number of actual state transitions)` |
| Intersect a 100,000-row exact mask | About 1,563 64-bit words |
| Rescore all reads | Avoided; use lazy demand generations |

`S` should normally be one or two: the owning node and perhaps a shared gate or parent. A struct
parent with twenty children scans twenty cached frontiers when one frontier changes; it does not
visit the twenty child subtrees. Twenty integer comparisons are simpler and likely cheaper than a
heap at that fan-out.

### Illustrative memory budget

The following is a sizing exercise, not a measured Rust layout. Assume one simple morsel has 12
operator states, 18 subscriptions, 16 live task slots, and 16 fact slots:

| Item | Illustrative size | Count | Total |
| --- | ---: | ---: | ---: |
| Operator state header | 128 bytes | 12 | 1.5 KiB |
| Subscription edge | 16 bytes | 18 | 288 bytes |
| Task slot | 64 bytes | 16 | 1 KiB |
| Fact slot excluding result payload | 48 bytes | 16 | 768 bytes |
| Dirty queue and bit sets | — | — | Less than 1 KiB |
| **Graph bookkeeping** | — | — | **Approximately 4–6 KiB** |

By comparison, one exact mask for 100,000 rows is 12,500 bytes. Three simultaneous exact masks are
about 36.6 KiB. Compressed buffers, decoded arrays, and task results are usually much larger. The
dependency graph should therefore be a secondary cost if it uses compact IDs and arenas.

The likely performance risks are not graph asymptotics but:

- allocating every work item separately;
- using hash maps where a generational slot index suffices;
- creating CPU tasks for operations cheaper than task launch;
- cache misses from boxed node state;
- waking the same node repeatedly before it runs;
- retaining completed result payloads after all subscribers consumed them; and
- creating task or graph nodes at row granularity.

The first implementation should use arenas or slot maps, `SmallVec` subscriber lists, a queued bit
for wake coalescing, stable `WorkId` values, and explicit result-release counts. It should measure:

- nanoseconds per `resolve` and per local transition;
- graph bytes per morsel;
- live task and fact high-water marks;
- dirty nodes per completion;
- drives or transitions per emitted row;
- duplicate wake coalescing; and
- scheduler task-launch time versus useful CPU time.

## Worked query

Consider a file containing orders. `customer_name` is dictionary encoded as an order-row codes
column plus a customer-domain values column:

```sql
SELECT order_id, customer_name
FROM orders
WHERE status = 'OPEN' AND total > 100;
```

For one illustrative morsel `[0..16)`, the physical plan is:

```text
MorselRoot [0..16)
  FilterCoordinator
    Eval(status == 'OPEN')
      SegmentScan(status)
    Eval(total > 100)
      SegmentScan(total)
  Pack
    SegmentScan(order_id)
    Take(customer_name)
      SegmentScan(customer_name.codes)       order-row domain
      SegmentScan(customer_name.values)      customer sub-root
  Rebatch
```

The important dependency graph is:

```text
read status -> test status -> StatusMask ----\
                                             +-> DemandLedger -> SealedDemand
read total  -> test total  -> TotalMask  ----/                     |
                                                                  +-> read order_id -> decode order_id --\
                                                                  |                                 +-> Pack -> output
                                                                  +-> read name codes -> decode codes             /
                                                                                         |                         /
                                                                                         v                        /
                                                                                   GatherDemand                 /
                                                                                      /    \                    /
                                                                            read page 0  read page 1           /
                                                                                |          |                   /
                                                                            decode 0   decode 1 -> Take ------/
```

This graph has 12 long-lived operator/coordinator nodes. The read, decode, predicate, and
gather operations are dynamic task slots created only while live.

### Initial advance

Demand is open and every row may survive. Static preparation already knows the status, total,
order-ID, and codes read addresses. The values-page addresses are unknown until the codes resolve.

```text
ready:
  W0 read status                   Candidate, predicate
  W1 read total                    Candidate, predicate
  W2 read order_id                 Candidate, projection
  W3 read customer_name.codes      Candidate, projection

gates:
  G0 customer_name.values pages wait for decoded codes
```

The scheduler sees all four reads. It may admit only `W0` and `W1` because they can eliminate most
rows. `W2` and `W3` remain stable candidates and will be returned again until admitted, eliminated,
or promoted.

### Predicate reads resolve

Completing `W0` writes `StatusBytes` and directly wakes only the status scan node. Its transition
returns:

```text
W4 decode status and evaluate status == 'OPEN'
```

Completing `W1` similarly returns `W5` for total. There is no root-to-leaf scan.

Suppose the CPU results are:

```text
StatusMask = {1, 3, 6, 11, 14}
TotalMask  = {1, 2, 6, 9, 11}
```

The second mask completion wakes the `DemandLedger`, which publishes:

```text
SealedDemand([0..16)) = {1, 6, 11}
```

Publishing that fact wakes the projection root and required-read promotion. Existing `W2` and `W3`
keep their IDs but change from candidate to required. If they were already in flight, no duplicate
read is issued.

### Projection reads resolve

The order-ID and codes reads each wake only their owner and return independent CPU tasks:

```text
W6 decode order_id for sealed rows {1, 6, 11}
W7 decode customer_name.codes for sealed rows {1, 6, 11}
```

Assume `W7` produces:

```text
row 1  -> customer 42
row 6  -> customer 17
row 11 -> customer 42
```

The codes fact wakes `Take`, which resolves `G0` into exact gather demand `{17, 42}`. Suppose those
IDs occupy two values pages. The next step returns all concrete I/O:

```text
W8 read customer_name.values page containing ID 17
W9 read customer_name.values page containing ID 42
```

No API could have returned these physical reads before `W7`; their addresses were genuinely
data-dependent. The initial step nevertheless exposed the dependency as `G0`, so the scheduler
knew why no name-value reads were available.

### Values and output resolve

The two page reads independently expose decode work:

```text
W10 decode values page for ID 17
W11 decode values page for ID 42
```

When both facts exist, `Take` returns a gather task while `Pack` may already hold the decoded
order IDs:

```text
W12 gather names [42, 17, 42]
```

Resolving `W12` advances the `Take` frontier and wakes `Pack`, which then returns:

```text
W13 pack order_id and customer_name
```

The final result records dense coverage separately from compact values:

```text
ExecBatch {
    rows: 0..16,
    values: 3 rows,
}
```

An all-false sealed demand would still commit dense progress through row 16 with zero compact
values and without scheduling `W2` through `W13`.

### Completion cost in the example

Resolving `W7` does not inspect all 11 operator nodes. It performs approximately:

```text
TaskSlot(W7)
  -> write CodesFact
  -> enqueue Take
  -> Take computes GatherDemand
  -> publish GatherDemand
  -> enqueue values sub-root
  -> values sub-root returns W8 and W9
```

That is one task lookup, two fact publications, two node transitions, and two new work items. The
cost follows the changed dependency path, not the full tree. Independent branches remain asleep.

## Recommended changes to the self-paced proposal

The implementation prototype should test this stronger returned-work interface before committing
to effectful `DriveContext::require_read` and `submit_cpu` calls:

1. Replace exclusive `DriveResult` control with a `PlanStep` that can contain work, output, gates,
   and completion state simultaneously.
2. Compile plan edges into fact subscriptions when a scan and morsel open.
3. Route `TaskId` directly to an output fact and subscribers.
4. Retain the ticket store as durable state; use completion events only to enqueue exact nodes.
5. Keep the per-scan `ReadCatalog`, with per-morsel logical uses and stable physical `ReadKey`
   deduplication.
6. Treat open demand as scheduling evidence and sealed demand as execution authorization.
7. Return every currently known work item before declaring local quiescence.
8. Represent unknown future reads as gates whose resolving facts have explicit subscribers.
9. Benchmark graph overhead independently from I/O, decode, and mask costs.

The DataFusion and DuckDB designs validate the CPU/I/O split from opposite directions. DataFusion
shows a planner returning CPU-ready morsels and explicit I/O continuations. DuckDB shows a source
returning several executor-owned tasks when it blocks. The Vortex design should combine those
ideas with row-domain coverage, immutable sealed demand, stable read identity, conditional gates,
and direct dependency routing inside a morsel.
