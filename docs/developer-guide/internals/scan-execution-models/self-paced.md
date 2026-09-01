# Proposed Self-Paced Plan Execution

This proposal keeps the useful outer unit from the current scanner—a configurable morsel of
roughly 100,000 rows—but replaces exact, recursive execution inside that morsel with a mutable
execution graph. Each operator may return its next natural contiguous prefix. Parents align, cap,
and combine child results, and the root rebatches internal fragments for the consumer.

The important refinement is that read discovery, demand refinement, and value execution are
different control planes:

- plans describe statically visible reads once per scan;
- a demand ledger refines exact filter masks and seals immutable row windows;
- a central scheduler admits reads and CPU tasks under independent budgets; and
- execution nodes are driven only to make value progress over sealed demand.

Underneath all three sits one shared abstraction: an explicit **row domain** and an explicit
transform between domains. Coordinate translation is not a per-operator concern; it is a declared
property of each parent-child edge, and the same declaration drives demand derivation, read
coverage, morsel-boundary discovery, and row identity.

An execution node does not repeatedly return a choice between I/O and CPU work. One drive call may
discover that different children need both kinds of work. It registers all independently runnable
work through an idempotent context, then returns a batch, a wait set, completion, or a fairness
yield.

The precise model is **sealed-demand, self-paced prefix execution**. Self-paced means that a node
chooses the end of the next contiguous prefix, within a bound its parent may set. It does not mean
that a node may return arbitrary rows or commit speculative work.

## Recommendation

Use the following layering:

~~~text
LayoutRef
  -> layout-specific lowering
PlanRef                              immutable physical operator tree
  -> generic rewrites
optimized PlanRef
  -> open one scan
     -> allocate row domains and edge maps
     -> build stable ReadCatalog     storage facts and dependency gates
     -> ScanState                    per-scan caches keyed by plan identity
  -> prepare one fixed morsel
     -> catalog view + ExecGraph     mutable operator state
DemandLedger                         exact, monotone mask refinement
  -> seal a contiguous window
SealedDemand
  -> ExecOp::drive
     -> register read and CPU tickets
     -> Batch | Blocked | Done | Yield
self-paced ExecBatch prefixes
  -> parent alignment
  -> root rebatching
ArrayStream                          consumer-sized arrays
~~~

This is an evolution of plan v2, not another layout-reader interface. Layouts describe storage,
plans describe physical operations, and execution nodes own per-morsel progress.

## Goals

- Preserve the clean, rewriteable plan-v2 operator tree.
- Retain fixed morsels as the unit of outer scan parallelism and ordering.
- Let segment, page, chunk, and encoding boundaries influence internal batch sizes.
- Expose useful I/O across the whole morsel without decoding the whole morsel eagerly.
- Allow independent children to register I/O and CPU work concurrently.
- Prevent open filter-mask revisions from repeatedly traversing projection state.
- Preserve fallible-expression semantics by executing projection only on sealed demand.
- Make coordinate translation explicit and reusable instead of reimplemented per operator.
- Bound compressed data, decoded data, retained results, CPU work, and output independently.
- Keep consumer batch sizes stable through a root rebatcher.

## Non-goals

- The first implementation does not require page-level reads from a format that only exposes a
  complete segment.
- Execution nodes do not choose global scheduling priority or directly submit physical I/O.
- A child does not return disconnected or out-of-order row intervals.
- Mutable cursors and per-scan caches do not live in PlanRef.
- The first implementation does not require an arena or lock-free execution graph.
- Speculative reads do not authorize speculative fallible computation.

## Two execution scales

The outer morsel and inner batch solve different problems:

| Scale | Typical size | Chosen by | Purpose |
| --- | ---: | --- | --- |
| Morsel | 100,000 rows | scan scheduler | Parallelism, ordering, cancellation, and bounded ownership |
| Internal prefix | about 8,000 rows, but variable | child and parent together | Natural decode and composition progress |
| Consumer batch | configurable | root rebatcher | Stable public stream shape |

A morsel has one logical owner. Its execution graph may be parked and resumed on different worker
threads, but two threads never call drive concurrently with mutable access to the same graph.
Scheduler-owned I/O and CPU tasks may run concurrently, and many morsels provide additional outer
parallelism.

The fixed morsel is therefore not the internal unit of computation. It is a container within which
the execution graph advances through independently sized prefixes.

## Row domains

A **domain** is a row universe. Two nodes share a domain when a row in one *is* a row in the other.
A **domain map** is the transform declared on a parent-child edge. Pure renumbering stays inside a
domain; only a genuine change of row universe crosses into a new one.

~~~rust
/// A row universe. Allocated when the scan is opened.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct DomainId(u32);

enum DomainMap {
    /// child row r is parent row r.
    Identity,
    /// child = parent - offset.
    Shift { offset: i64 },
    /// Shift, plus `extra` trailing rows.
    Fence { offset: i64, extra: u64 },
    /// child = parent / stride. Crosses into a new domain.
    Coarsen { stride: u64 },
    /// Monotone and contiguous, resolved by a gate. New domain.
    MonotoneGated { gate: GateId },
    /// Arbitrary gather, resolved by a gate. New domain.
    GatherGated { gate: GateId },
}
~~~

Every edge in the plan declares one:

| Edge | Map | Existing state it formalizes |
| --- | --- | --- |
| Pack to field or validity | Identity | equal row counts |
| Eval to input | Identity | — |
| RowIdxPartition to branch | Identity | equal row counts |
| Zoned to data | Identity | — |
| Take to codes | Identity | — |
| ListPack to validity | Identity | — |
| Concat to chunk *i* | Shift | `ConcatData::row_offsets` |
| RowIdx to child | Shift | `RowIdxData::row_offset` |
| ListPack to offsets | Fence, extra 1 | inline `row_range.end + 1` |
| Zoned to zone statistics | Coarsen | `zone_len` |
| ListPack to elements | MonotoneGated | `elements_range_from_offsets` |
| Take to values | GatherGated | full-domain execution today |

The set is closed even for third-party layouts, because they lower to these same operators.

### Operations

~~~rust
impl DomainMap {
    /// Child range covering a parent range. Read coverage and child requests.
    fn map_range(&self, parent: Range<u64>) -> VortexResult<Range<u64>>;
    /// Child demand for a parent demand.
    fn map_demand(&self, parent: SealedDemand<'_>) -> VortexResult<OwnedSealedDemand>;
    /// Largest parent prefix satisfied by a child committed to `child_end`.
    /// Defined only when `prefix_preserving()`.
    fn unmap_frontier(&self, child_end: u64) -> VortexResult<u64>;
    fn prefix_preserving(&self) -> bool;
    fn is_static(&self) -> bool;
}
~~~

`unmap_frontier` is what makes alignment work across a coordinate change. For ListPack it is a
search over decoded offsets for the largest `k` with `offsets[k] <= child_end`.

### The property that matters

| Map | Static | Prefix-preserving | Exact for fallible work |
| --- | :---: | :---: | :---: |
| Identity | ✔ | ✔ | ✔ |
| Shift | ✔ | ✔ | ✔ |
| Fence | ✔ | ✔ | ✔ |
| Coarsen | ✔ | ✔ | ✖ |
| MonotoneGated | ✖ | ✔ | ✔ |
| GatherGated | ✖ | ✖ | ✔ |

Only GatherGated breaks prefix progress. Everything else composes under one rule. In particular
ListPack's element child is monotone and contiguous, so an outer prefix maps to an element prefix
and ordinary prefix progress applies once its gate resolves. ListPack belongs with the easy
operators, not with Take.

A GatherGated child is not driven inside its parent's prefix cursor at all. It becomes a
**sub-root**: its own domain, its own prefix cursor over the child's full range, and a sparse demand
mask equal to the gather set. See TakeExec below.

### One map, four consumers

| Consumer | Use |
| --- | --- |
| Demand derivation | translate a sealed mask across an edge |
| Catalog coverage | compose maps from a read's owner up to the ledger domain |
| Morsel boundaries | walk static prefix-preserving edges, translating boundaries |
| Row identity | compose Shift maps to the file domain |

Without this, each consumer reimplements the same translation. Split discovery in
`vortex-scan-v2/src/splits.rs` is the clearest example: its per-operator switch is a hand-written
domain-map walk in which the `child.row_count() == plan.row_count()` test is an Identity check,
`row_offset + chunk_offset` is a Shift, and taking only Take's codes child is skipping a
GatherGated edge.

### Cost

For a flat, chunked, or struct scan every edge is Identity or Shift and the whole graph is one
domain. Coarsen, MonotoneGated, and GatherGated are the only maps that allocate a new DomainId, so
the common case pays a branch on a copy type and nothing else. With the mask offset described under
[Demand ledger](#demand-ledger), Shift derivation is a field update rather than an allocation.

## Three state tiers

| Tier | Lifetime | Sharing | Contents |
| --- | --- | --- | --- |
| PlanRef | cross-scan | immutable | operators, dtypes, row domains, expressions |
| ScanState | one scan | `Arc`, keyed by plan identity | catalog spine, domain maps, resolved metadata, dictionary value caches, zone maps, read-store handles |
| ExecGraph | one morsel | owned, never shared | cursors, retained state, tickets, scratch |

The rule is that anything a node would compute identically in every morsel belongs in ScanState.
Three costs depend on this:

- **Dictionary value domains.** Plan v2 executes Take's values child over its whole domain on every
  call. Under 100,000-row morsels a 1,000,000-row file rebuilds that domain ten times.
- **Catalog construction.** Segment identity and row coverage are morsel-independent facts. Building
  them per morsel costs `columns × segments-per-morsel` entries per morsel; building them per scan
  with per-morsel views costs that once.
- **Lazy plan lowering.** `PlanChildren` populates children on first access. That cache is correct
  on PlanRef today but needs a stated home so it does not grow into per-scan runtime state.

ScanState entries must be bounded and have an eviction policy. Dictionary value caches are the
obvious unbounded case.

## Four responsibilities

### Plan preparation

Scan-level preparation traverses the optimized plans for pruning, filtering, and projection. It
allocates domains, declares edge maps, and records all reads that are statically visible. Morsel
preparation opens mutable execution nodes and takes a view over the catalog for that row range.
Neither happens on every drive call or mask revision.

### Demand coordination

The root filter coordinator owns the exact candidate mask. It may shrink that mask as evidence and
predicates complete. Once every predicate that can affect a row window has completed, the
coordinator seals the window. A sealed mask is immutable.

### Scheduling

The scan scheduler owns read deduplication, admission, priorities, task slots, and resource
credits. Operators describe facts such as row coverage, byte size, phase, and dependencies. They
do not assign an absolute priority that ignores other morsels or queries.

### Value execution

Execution nodes consume sealed demand, inspect durable ticket state, register newly required work,
update cursors, and compose batches. Expensive decoding and expression evaluation run in
scheduler-owned tasks rather than inside the coordination loop.

Keeping these responsibilities separate prevents a mask update from becoming an instruction to
walk every projection node and reconsider every physical read.

## Preparation and read discovery

The conceptual preparation API has three contracts:

~~~rust
trait PlanExec {
    fn declare_domains(
        plan: &PlanRef,
        domains: &mut DomainBuilder,
    ) -> VortexResult<()>;

    fn describe_reads(
        plan: &PlanRef,
        scan: Range<u64>,
        catalog: &mut ReadCatalogBuilder,
    ) -> VortexResult<()>;

    fn open_exec(
        plan: &PlanRef,
        morsel: Range<u64>,
        context: &OpenContext,
    ) -> VortexResult<Box<dyn ExecOp>>;
}
~~~

The exact Rust placement is an implementation choice, and one traversal may perform the first two.
The semantic distinction is required regardless.

### Stable read catalog

A catalog entry describes a logical use of physical bytes:

~~~rust
struct ReadEntry {
    use_id: ReadUseId,
    key: ReadKey,
    owner: ExecNodeId,
    domain: DomainId,
    coverage: DomainCoverage,
    estimated_bytes: usize,
    phase: ReadPhase,
    gate: Option<GateId>,
}
~~~

ReadKey identifies the physical request and is stable across prefetch and required use. Several
ReadEntry values may share a key because a filter and projection, or two plan branches, can use the
same bytes. The scheduler merges those uses into one physical request.

Coverage is stated in the entry's own domain. Relating it to demand blocks is map composition from
the owning node up to the ledger domain:

- every map on the path is static and prefix-preserving: exact block coverage;
- a Coarsen on the path: exact but coarsened coverage;
- a gated map on the path: group coverage until the gate expands.

This is mechanical rather than a per-operator judgement, and it replaces the earlier rule that a
nested or lookup operator "may instead provide a conservative group".

Each logical read has two independent state axes:

~~~text
necessity:   Candidate | Required | Eliminated
data:        Unscheduled | Queued | InFlight | Ready | Consumed
~~~

Candidate means that current information permits the read to be useful. Required means that a
sealed execution prefix cannot progress without it. Eliminated means that monotone demand
refinement proved it unnecessary. Promotion from candidate to required uses the same ReadKey and
does not issue a duplicate physical request.

### Static reads and dynamic gates

Flat segments, chunk boundaries, and struct fields are visible when the plan is opened. The
catalog can enumerate their reads for the whole scan immediately, and a morsel takes a view. This
lets the scheduler run arbitrarily far ahead on compressed I/O, subject to its byte budget, without
repeatedly driving the projection tree.

Some reads genuinely cannot be identified yet, and these are exactly the edges whose map is not
static:

- Take needs decoded codes before it knows which value-domain pages matter (GatherGated).
- ListPack needs offsets before it knows the element range (MonotoneGated).
- Zoned execution may need evidence before data reads become worthwhile.
- An encoding may need a footer or index before page locations are known.

Preparation records a gate for these dependencies. When the gate result becomes available, the
owning node expands it once into stable catalog entries. This expansion is driven by new
information, not by polling or by every mask revision.

No API can schedule an address that is not derivable until CPU work completes. The design makes
that computational dependency explicit while allowing every already-known read to run ahead.

### Facts versus priority

Plans and execution nodes report:

- physical key and estimated bytes;
- domain and coverage;
- pruning, predicate, projection, or metadata phase;
- dependency gates;
- whether a use is candidate or currently required; and
- local reuse or cancellation relationships.

The scheduler combines those facts with global state:

- blocking versus speculative status;
- distance from the commit frontier;
- current demand summary;
- read sharing;
- per-morsel and global byte credits;
- fairness between morsels; and
- cancellation or limit state.

This avoids embedding a different, incompatible priority policy in every layout reader.

## Demand ledger

Passing a live, mutable, repeatedly shrinking mask through projection value nodes is the wrong
abstraction. Projection planning and the read catalog may observe immutable open snapshots and
summaries to offer candidate I/O, but exact or fallible value execution receives sealed demand.
This avoids waking the complete projection tree for every predicate completion and prevents a
fallible expression from running for rows that a later predicate removes.

Instead, each morsel owns a DemandLedger. It divides the morsel into modest fixed windows, for
example 1,024 rows:

~~~rust
struct DemandBlock {
    rows: Range<u64>,
    candidate: Mask,
    remaining_predicates: PredicateSet,
    revision: u32,
    state: BlockState,
}

enum BlockState {
    Open,
    Sealed,
}
~~~

Within one demand epoch:

- candidate masks may only shrink;
- remaining predicates may only complete;
- revision increases only while a block is open;
- sealing occurs once; and
- a sealed mask never changes.

Predicates may finish for blocks out of order. The ledger also tracks the contiguous sealed
frontier from the morsel's commit position. Projection can begin as soon as that frontier advances;
it does not need to wait for the entire 100,000-row morsel.

The execution API accepts an immutable capability:

~~~rust
struct SealedDemand<'a> {
    epoch: DemandEpoch,
    domain: DomainId,
    rows: Range<u64>,
    mask: &'a Mask,
    /// `rows.start` corresponds to `mask[mask_offset]`.
    mask_offset: usize,
}
~~~

The mask offset matters because the batch contract re-mints a sealed demand for the unconsumed
suffix after every prefix. Interpreting the mask relative to `rows.start` would force a sliced mask
per prefix, on the hot path, at roughly twelve prefixes per morsel per operator edge. With an
offset, both suffix re-minting and Shift derivation are field updates over one shared mask.

Constructing SealedDemand is restricted to DemandLedger. This makes it difficult to invoke a
demand-sensitive or fallible projection over provisional rows by accident.

An adaptive predicate also receives an immutable input mask for its own stage. After that
predicate finishes, the coordinator intersects its result into the open block and either schedules
the next predicate or seals the block. A task never observes an input mask changing underneath it.

### Derivation across a domain edge

Sealing is a claim about finality, not about row space, and translation preserves finality. The
operator that owns an edge's map may therefore derive across it:

~~~rust
impl SealedDemand<'_> {
    fn derive(&self, map: &DomainMap) -> VortexResult<OwnedSealedDemand>;
}
~~~

A naive rule that a derived demand must be implied by its parent's is wrong: Coarsen maps a row set
to the set of *covering* zones, which is a superset in row terms. Two rules apply instead:

1. **Completeness.** The derived demand covers every child row that any demanded parent row depends
   on. Violating this produces wrong values.
2. **Minimality for fallible work.** A map used to derive demand for fallible computation must
   contain no child row that no demanded parent row depends on. Identity, Shift, Fence,
   MonotoneGated, and GatherGated satisfy this exactly. Coarsen does not, so it may drive only
   infallible metadata work — which is what it feeds in practice.

Exact derivation is sometimes better than what a whole-range request would do. ListPack needs
`offsets[k]` and `offsets[k+1]` for each demanded outer row `k`, so the Fence derivation is
`d | (d << 1)`. Plan v2 requests all offsets unconditionally; for a sparse filter the derived form
reads materially fewer.

### Widening and epochs

Monotone shrinking is a correctness requirement, not merely an optimization. Once an executor has
discarded state, skipped rows, emitted output, or cancelled reads, it cannot safely accept a mask
that widens.

A selection or predicate change that can add rows would create a new epoch, and the implementation
would have to either restart the uncommitted suffix under that epoch or define snapshot semantics
that defer the change to a later scan.

**Whether this case is reachable is an open question** — see [Open questions](#open-questions).
No current API appears to widen demand: Selection is fixed when the scan is constructed, and
pruning, evidence, and predicates all intersect. The construct that most resembles widening is
incremental Take, where successive outer prefixes need more of the value domain; under the sub-root
model below that is not widening but a sequence of independent, immediately sealed value-domain
demands sharing a ScanState cache.

## Coarse demand summaries

For a 100,000-row morsel, an exact bit mask is only 12,500 bytes, or about 1,563 64-bit words.
Exact intersection and population count should be the default when a predicate result is
available. A probabilistic sketch is unnecessary for correctness and is unlikely to beat one
linear bitwise operation at this scale.

The scheduler still needs a cheap way to score many read entries without rescanning the exact mask
for every entry after every predicate. The ledger therefore maintains a block summary with two
independent facts and one cache:

~~~rust
struct BlockDemandSummary {
    generation: u64,
    /// Exact current candidate count per block. The authoritative fact.
    upper_counts: Vec<u16>,
    /// Block state, so a sealed empty block is distinguishable from an open one.
    sealed: BitSet,
    /// Cache of `upper_counts[i] > 0`, kept only so the scheduler can scan many
    /// blocks in one bitwise pass. Rebuilt from `upper_counts`, never written
    /// independently.
    maybe_nonempty: BitSet,
}
~~~

Earlier drafts also carried a separate tri-state Zero/All/Mixed summary and a `sealed_nonempty` bit
set. Both are derivable from the two facts above, and Phase 2's own rationale is that the
optimization must not become a second source of truth. Derive them.

If a read covers only blocks whose count is zero, it can be eliminated. A non-empty open block does
not prove that the read will eventually be needed.

If only the counts c1 and c2 of two masks over N rows are known, their intersection is bounded by:

~~~text
lower = max(0, c1 + c2 - N)
upper = min(c1, c2)
expected under independence = c1 * c2 / N
~~~

The expected value is useful for ordering speculative reads. It cannot prove emptiness. Only exact
position information or a zero count can do that.

An estimated per-block count derived from remaining predicate selectivities is a scheduling-only
heuristic and may be omitted from the first implementation. The available source,
`FilterExpr::report_selectivity`, records one rate per conjunct after that conjunct runs, globally
rather than per block, so applying it uniformly carries no per-block information. It can order
reads across different predicate sets but not across blocks under the same one.

Read entries store the generation at which they were last scored. The scheduler rescans an entry's
covered blocks lazily when that entry reaches the admission queue, rather than eagerly updating
every read after every mask change. Immediate scheduler signals are reserved for:

- promotion of a read that blocks a sealed prefix;
- proof that an entire read coverage is impossible;
- discovery of new gated reads; and
- expansion of the configured read horizon.

This coalesces demand churn without hiding correctness-critical changes.

## Execution API

The execution graph is pull-driven and runs to quiescence:

~~~rust
trait ExecOp: Send {
    fn drive(
        &mut self,
        request: &BatchRequest<'_>,
        context: &mut DriveContext<'_>,
    ) -> VortexResult<DriveResult>;
}

struct BatchRequest<'a> {
    demand: SealedDemand<'a>,
    /// Hard upper bound on the prefix end. Parents use this to align siblings.
    max_rows: u64,
    target_rows: usize,
    target_bytes: usize,
}

enum DriveResult {
    Batch(ExecBatch),
    Blocked(WaitSet),
    Done,
    Yield(Progress),
}
~~~

DriveResult is not a work queue:

- Batch commits a non-empty dense prefix of the request.
- Blocked says that no further local transition is possible until one of the named tickets or
  credits changes state.
- Done says that the operator has consumed its complete morsel domain.
- Yield says that useful local progress was made, but the transition budget was exhausted before
  reaching another outcome. It carries evidence of that progress.

I/O and CPU tasks are registered through DriveContext:

~~~rust
impl DriveContext<'_> {
    fn require_read(&mut self, use_id: ReadUseId) -> ReadTicket;
    fn submit_cpu(&mut self, key: CpuTaskKey, task: CpuTask) -> CpuTicket;
    fn read_result(&self, ticket: ReadTicket) -> Option<&ReadResult>;
    fn cpu_result(&self, ticket: CpuTicket) -> Option<&CpuResult>;
    fn wait_for_credit(&mut self, class: CreditClass, bytes: usize) -> CreditTicket;
}
~~~

Registration is idempotent. Requiring the same ReadUseId or submitting the same node-local task key
returns the existing ticket. CPU tasks own their inputs and return owned outputs; they never retain
mutable access to an execution node.

One Pack drive can therefore:

1. inspect completed tickets;
2. consume ready results;
3. drive every child whose head is missing;
4. register a read for one child and CPU work for another;
5. assemble a prefix if every child has a head; and
6. otherwise return one combined wait set.

The caller does not need to interpret an exclusive MoreIo or RunCpu state, and sibling work is not
serialized by the shape of an enum.

### Run to quiescence

Within a bounded transition budget, drive repeats cheap state transitions until one of the four
outcomes is reached. Cheap work includes ticket inspection, cursor changes, mask slicing, catalog
gate expansion, and array-head bookkeeping. Expensive decoding, expression evaluation, and array
construction become CPU tasks when they exceed a cost threshold.

Every multi-child operator must visit all missing children before returning Blocked. Waiting on the
first child without registering independent work for later children would create artificial
serialization and can deadlock a bounded scheduler.

### Progress obligations

Two non-terminal outcomes exist, and neither may spin:

- A Yield must carry evidence of progress: at minimum an increased transition count, ideally a
  frontier that moved. Two consecutive Yields with no frontier change and no ticket state change
  are a debug assertion, not a fairness event.
- A Batch is at least `min(request rows, MIN_PREFIX_ROWS)` unless bounded by an indivisible unit or
  a resource credit. Without this, a node returning one row forever satisfies every other invariant.

Drives per committed row is a metric with a debug-build ceiling.

Yield otherwise prevents a large ready graph from monopolizing its coordinator thread. The scheduler
may immediately queue the morsel again.

### Tickets, not event inboxes

Completion events only make a morsel runnable. Durable state lives in:

- the read catalog and read-result store;
- CPU task tickets and owned results;
- execution-node state variants;
- child BatchCursor values; and
- the demand ledger.

On wake-up, drive reads ticket state and derives the next transition. It does not replay an event
log or rely on receiving completion messages in a particular order. Duplicate and coalesced wakes
are therefore harmless.

### When drive is called

Projection drive is scheduled only when:

1. the contiguous sealed frontier grows beyond the projection commit frontier;
2. a read, CPU, or credit ticket in its current WaitSet changes state;
3. downstream output capacity becomes available;
4. cancellation, limit, or error state changes; or
5. the previous call returned Yield.

An intermediate predicate completion normally updates DemandLedger and its summary generation. It
does not wake projection unless it seals a new contiguous window. The read scheduler can
independently reconsider candidate catalog entries when it has admission capacity, so read-ahead
does not require projection polling.

This answers the drive-frequency problem: call drive at semantic progress boundaries, not at every
mask refinement and not merely because an unrelated event arrived.

## Three horizons and separate budgets

One row cursor cannot express all useful look-ahead. Each morsel has three logical horizons:

~~~text
commit/emit horizon       rows eligible to become the next output prefix
materialize horizon       rows for which decode or expression CPU may run
read horizon              rows whose compressed inputs may be prefetched
~~~

For example:

~~~text
rows       0          8k          24k                         100k
           |-----------|-----------|-----------------------------|
committed  ^ 8k
emit                   ^ next sealed prefix
materialize                         ^ bounded decoded look-ahead
read                                                             ^ whole morsel if credits allow
~~~

The read horizon can run far ahead because compressed buffers use a separate budget. The
materialize horizon remains closer to the commit frontier because decoded arrays and retained
results are often larger. Fallible or demand-sensitive computation may not cross the appropriate
sealed horizon. An operator may opt into provisional computation only when it proves that doing so
cannot change observable results or errors.

At minimum, account separately for:

- in-flight and retained compressed bytes;
- decoded arrays and retained results;
- CPU task inputs and outputs;
- root output buffering; and
- oversized indivisible units.

Throttle decoded lead by retained bytes, not only by rows. Fields can differ by orders of magnitude
in bytes per row.

### Progress guarantees

Two rules prevent a bounded budget from deadlocking:

- **Oversized units.** An indivisible segment, encoded block, or list value can exceed the normal
  budget. Progress then requires an explicit oversized-unit permit that runs one such unit in
  isolation.
- **Morsel age.** Credits are reserved per morsel at admission, and a morsel is admitted only if
  its worst case can be granted. The oldest in-flight morsel is never denied credit in any class:
  it can always drain and release, so global progress follows by induction on morsel age.

The second rule matters for classes other than compressed reads. Reserving progress credit for
blocking *reads* does not prevent hold-and-wait on *decoded* credit, where several morsels each
retain partial results and none can advance. Every morsel's Blocked can name a live condition while
no morsel is able to make progress; invariant 9 is a local check and cannot see that.

## Batch contract

An execution batch records dense coverage separately from compact values:

~~~rust
struct ExecBatch {
    rows: Range<u64>,
    values: ArrayRef,
    retained_bytes: usize,
    /// Debug builds only. The requester holds the sealed mask and can derive this;
    /// carrying it in release builds creates a second source of truth.
    #[cfg(debug_assertions)]
    demand: Mask,
}
~~~

For every Batch result:

1. rows.start equals the request start.
2. rows is a non-empty dense prefix within the sealed request, and does not exceed `max_rows`.
3. rows covers at least `min(request rows, MIN_PREFIX_ROWS)` unless bounded by an indivisible unit
   or a credit.
4. the demanded prefix is exactly the sealed mask sliced to rows.
5. values.len() equals that slice's true count.
6. values preserve demanded row order.
7. the operator commits the prefix exactly once.
8. subsequent requests begin at the previous rows.end.

Dense coverage proves progress. An all-false mask may return an empty values array while advancing
over a non-empty dense prefix.

Target rows and target bytes are soft limits; `max_rows` is hard. Natural boundaries may stop
earlier, and an acquired oversized permit may allow one indivisible unit to exceed the soft targets.
A node never exceeds the sealed request or its hard credits.

## Parent alignment

Row-equivalent children may prefer different boundaries. If a parent only ever took the minimum
head and retained the rest, it would emit the **union** of every child's boundary set: a 20-field
struct over a 100,000-row morsel with 8,000-row pacing could emit hundreds of short batches rather
than a dozen full ones, each with retained tails to match.

Because `max_rows` is a hard bound, the parent can cap instead:

1. **Round one** goes wide to every child so all their I/O is in flight, and min-of-heads sets the
   agreed length `L`.
2. **Later rounds** issue `max_rows = frontier + L` to every child.

Children that already hold decoded data past `L` return exactly `L`, because slicing a decoded array
is free. The union collapses to one boundary and the parent retains nothing. A child that genuinely
cannot stop at `L` returns shorter, and the parent re-learns `L` from that round. The parent
distinguishes the two without new API: `batch.rows.end == request max_rows` means capped, anything
shorter is a real constraint.

For a compact child batch, splitting at dense position cut uses rank:

~~~text
value_cut = demand[..cut].true_count()
left_values  = values[..value_cut]
right_values = values[value_cut..]
~~~

### Who retains a surplus

A child that decodes more than the parent asked for keeps the surplus in **node-local** state,
charged to that node's decoded credit and released by that node. Parent-owned retention exists only
where a child cannot re-slice its own output.

The child is the only party that knows whether re-slicing is free, and the only party that can
release. This rule also answers what memory is charged where after a batch is sliced, which
otherwise has to be settled case by case.

Capping does not manufacture granularity an encoding does not expose. If a child can only decode one
indivisible 64,000-row unit, it holds that unit regardless. Capping fixes every case where the child
*could* have stopped and was not asked to.

## Intra-morsel struct example

Consider struct(a, b) over a 100,000-row morsel. Field a is cheap and has 64,000-row segments.
Field b is wide and has 8,000-row segments:

~~~text
dense rows  0       8k      16k      24k               64k              100k
            |--------|--------|--------|-----------------|-----------------|
a segments  |---------------- A0 ------------------------|------ A1 --------|
b segments  |-- B0 --|-- B1 --|-- B2 --| ... |-- B7 --|-- B8 --| ... B12 |
Pack output |-- P0 --|-- P1 --|-- P2 --| ...                              |
~~~

At scan preparation the catalog exposes A0, A1, and B0 through B12. The I/O scheduler may prefetch
any of them within compressed-byte credits.

The first Pack drive visits both fields:

~~~text
Pack drive
  a: require A0 read
  b: require B0 read
  register both, then Blocked({A0, B0})

reads complete
  a: submit decode A0
  b: submit decode B0
  register both, then Blocked({decode A0, decode B0})

decodes complete
  a ready frontier = 64k
  b ready frontier = 8k
  Pack emits [0..8k), sets L = 8k
  a holds its own decoded [8k..64k)
~~~

After P0, the important frontiers are:

~~~text
                       a                  b                 Pack
committed              8k                 8k                8k
ready                   64k                8k                8k
CPU scheduled           64k                perhaps 16k       -
compressed read-ahead   perhaps 100k       perhaps 100k      -
retained decoded bytes  a[8k..64k), charged to a             -
~~~

Subsequent rounds request `max_rows = 16k`, `24k`, and so on from both fields. Field a serves them
from its own decoded segment and returns exactly the cap; field b decodes B1, B2, and so on. Pack
retains nothing and emits one batch per round.

Pack should not cause A1 to decode merely because a's row frontier is closer: A1 cannot advance Pack
until b reaches 64k, and its decoded bytes are costly. Compressed A1 may still be read ahead under
the separate I/O budget.

The parent commit frontier is the minimum row-equivalent ready frontier. Useful work forms a
wavefront around that minimum:

- read far ahead where compressed bytes are cheap and reusable;
- materialize the lagging child and a bounded amount beyond it;
- let a leading child hold its own decoded surplus only within its decoded-byte credits; and
- emit the largest common prefix currently available.

There is a fundamental constraint here. If A0 can only be decoded as one indivisible 64,000-row
unit, the system must either hold that decoded result, serialize until credit is available, or add
finer physical decode support. No state-machine API can manufacture parallelism or granularity that
the encoding does not expose. The API's job is to expose the constraint, schedule independent work,
and bound its memory cost.

## Operator behavior

| Operator | Child maps | Prefix-preserving throughout |
| --- | --- | --- |
| SegmentScan | none | ✔ |
| Concat | Shift per chunk | ✔ |
| Pack | Identity | ✔ |
| Eval | Identity | ✔ |
| RowIdx | Shift | ✔ |
| Zoned | Identity (data), Coarsen (zones) | ✔ |
| ListPack | Fence (offsets), Identity (validity), MonotoneGated (elements) | ✔ once gated |
| Take | Identity (codes), GatherGated (values) | ✖ for values |

### SegmentScanExec

Preparation registers the segment or independently addressable pages. Drive promotes the read use
needed by the sealed prefix, consumes its ticket when ready, and submits decode work. It may return
a page-sized prefix or slice a larger decoded segment into smaller batches, honouring `max_rows` so
its parent can align it against siblings.

The current plan-v2 SegmentScan requests and decodes the complete serialized segment before
slicing. The first executor adapter can preserve that behavior. Smaller physical reads require
format metadata and independently decodable pages; changing the execution API alone does not add
them.

### ConcatExec

Concat maps parent rows into one child at a time through a Shift map built from immutable row
offsets. It holds an output cursor and a current child cursor. Static preparation enumerates later
child segments for the whole scan, so Concat does not walk later children on every drive merely to
offer prefetch.

Concat normally returns at a child boundary or at the child's chosen prefix. It has no row-wise
sibling alignment because only one chunk owns each output row.

### PackExec

Pack propagates the same sealed row demand to every projected field and validity child across
Identity maps. It drives all missing heads, caps later rounds at the agreed length, and combines
the common prefix.

For each child it tracks:

- committed frontier: rows already consumed by Pack;
- ready frontier: contiguous materialized rows held by the child;
- scheduled frontier: decode or expression CPU already submitted; and
- retained bytes, which is the backpressure quantity that matters when field widths differ.

Physical read admission remains in the catalog scheduler and can be much farther ahead than these
CPU frontiers.

### EvalExec

Eval normally preserves its child's prefix. It schedules expression work only for demanded compact
values. Fallible or otherwise demand-sensitive expressions require SealedDemand. Small infallible
operations may execute inline or speculatively only when an explicit safety classification allows
it.

### ListPackExec

ListPack is prefix-preserving throughout and belongs with the row-equivalent core rather than with
Take.

Outer rows define output progress. The Fence map derives demand for offsets as `d | (d << 1)` over
`rows.start..rows.end + 1`. Decoded offsets expand the element gate and resolve the MonotoneGated
element map, whose `unmap_frontier` is a search for the largest `k` with `offsets[k] <= element_end`.
Element batches are buffered in the element domain until at least one complete outer-row prefix can
be assembled.

One list value is indivisible at the output boundary. An oversized list uses the oversized-unit
permit.

### TakeExec

Codes define outer-row progress across an Identity map. Values live behind the one GatherGated edge
in the system, so they are not driven inside Take's prefix cursor. The values child becomes a
**sub-root**: its own domain, its own prefix cursor over the full value range, and a sparse demand
mask equal to the gather set.

This is well-formed. The gather set derives from a sealed outer demand and decoded codes, both
final, so the value-domain demand is sealed the moment the gate expands. The value domain carries no
predicates, so nothing waits. Below that point ordinary prefix progress applies, including when the
values subtree is itself a Concat.

The three materialization strategies are then three widths of the same gather mask rather than three
architectures:

| Strategy | Gather mask | When |
| --- | --- | --- |
| Full | all-true | value domain below a byte threshold; the common fast path |
| Sparse | exact code set for one outer prefix | default |
| Incremental | one sealed demand per outer prefix, deduplicated by the ScanState value cache | large domains with repeated codes |

Default to sparse per-prefix gather backed by a bounded ScanState value cache, falling back to full
materialization below the byte threshold. Note that the incremental form needs no widening
machinery: successive prefixes mint independent sealed demands over a shared cache.

### ZonedExec

Zone metadata contributes evidence to DemandLedger across a Coarsen map. Because Coarsen is not
minimal, it may drive only infallible metadata work — reading a zone tells you about rows nobody
demanded, which is fine for statistics and not for a fallible expression.

Data reads remain candidates while affected blocks are open and can be eliminated when evidence
proves the coverage empty. Evidence and data may share the same scheduler, but evidence receives
phase facts from which the scheduler derives its priority.

### Row-index execution

Absolute row identity is composition of Shift maps up to the file domain, which is what
`RowIdxData::row_offset` already stores. Row indices therefore derive from dense coverage
coordinates rather than compact array positions, and prefix slicing and root rebatching preserve
them without a special coordinate rule.

## Morsel boundary discovery

Morsel ranges are derived, not switched on. Walk edges whose map is static and prefix-preserving,
translating boundaries through the map, and stop at gated maps. Combined with catalog coverage this
yields the boundaries that matter: the rows at which read coverage changes.

This replaces the per-operator switch in `vortex-scan-v2/src/splits.rs`, which computes exactly this
by hand — its `child.row_count() == plan.row_count()` test is an Identity check, its
`row_offset + chunk_offset` is a Shift, and its taking only Take's codes child is skipping a
GatherGated edge. Deriving boundaries generically also removes the last place where a third-party
layout would require editing a central module.

## Root filter and projection flow

The morsel coordinator owns scan phases:

~~~text
initial selection
  -> initialize exact candidate masks
metadata and index evidence
  -> shrink open DemandLedger blocks
open demand snapshots
  -> offer predicate work and candidate projection I/O
  -> run explicitly safe discovery CPU for conditional reads
predicate stages
  -> evaluate immutable stage masks
  -> intersect exact results
  -> seal completed blocks
contiguous sealed frontier advances
  -> promote exact projection work and drive values with SealedDemand
  -> receive self-paced prefixes
  -> root rebatch and commit
~~~

Projection reads may be admitted while blocks are still open because read discovery and scheduling
use conservative catalog coverage. This overlaps I/O with filtering and preserves what plan v2 gets
today from constructing projection futures before the filter mask resolves. Projection computation
waits for sealing when required by its semantics.

Adaptive filter ordering remains root policy. Individual value operators do not reorder top-level
conjuncts. A block-oriented coordinator may finish and seal early blocks while later blocks still
run predicates, allowing output and I/O to pipeline across one morsel.

## Scheduler and backpressure

The scheduler maintains:

- one deduplicated physical read store keyed by ReadKey;
- logical read uses and their domain coverage;
- separate required and speculative admission queues;
- lazy demand-generation scoring;
- CPU task tickets and result storage;
- compressed, decoded, task, and output credits, reserved per morsel;
- cancellation groups and release frontiers; and
- fairness across morsels, with the oldest never denied.

Required work may bypass speculative priority but not hard safety limits. If all normal credit is
held by work that cannot unblock the commit frontier, the scheduler must be able to stop further
speculation and reserve progress credit for blocking work.

Operators release retained results, decoded pages, task results, and read-store references once no
uncommitted prefix can use them. Shared physical buffers remain until every logical use releases
them.

## Root rebatching and multiple morsels

Natural internal fragments do not leak into ArrayStream. RebatchExec:

- concatenates small adjacent batches toward a consumer target;
- slices large batches without copying when possible;
- respects ordering, limit, cancellation, schema, and memory boundaries; and
- commits dense progress independently of compact output length.

RebatchExec depends on none of the execution-graph machinery and can be built against the current
executor, where it already decouples the public batch size from the 100,000-row split unit. Doing so
early delivers one of this design's benefits before any of its risk and gives batch-size measurements
a stable reference point.

The outer scheduler opens execution graphs for disjoint morsels. Ordered scans merge them by morsel
position; unordered scans may emit completed morsels sooner. A parked morsel does not retain a
worker thread, and scheduler-owned tasks from several morsels may occupy the worker pool.

## Correctness invariants

The implementation must enforce:

1. Only DemandLedger constructs SealedDemand, and only the operator owning an edge's DomainMap
   derives across that edge.
2. Derivation is complete: it covers every child row that a demanded parent row depends on.
3. Derivation that drives fallible work is also minimal. Coarsen is not minimal and may drive only
   infallible metadata work.
4. Prefix progress composes across every map except GatherGated; a GatherGated child is driven as a
   sub-root with its own cursor and its own sealed demand.
5. Demand shrinks monotonically within an epoch.
6. A Batch covers exactly one non-empty dense prefix of its request, within `max_rows`, and at least
   `MIN_PREFIX_ROWS` unless bounded by an indivisible unit or a credit.
7. Compact value cardinality equals the exact prefix-demand population count.
8. A parent commits only the intersection of row-equivalent child-ready prefixes.
9. Registering the same read use or CPU task key is idempotent.
10. Work completion changes ticket state; event order is not semantic state.
11. Every Blocked names a condition that can make progress possible, and every Yield carries
    evidence of progress.
12. A multi-child drive visits every missing child before blocking.
13. Speculative reads never commit rows or authorize unsafe computation.
14. CPU tasks own inputs and never mutate the execution graph concurrently.
15. Retained data is charged to the node that can release it, and a child that overshoots a cap
    charges itself.
16. Cancellation and errors prevent further commits and release unneeded work.
17. Root output preserves dense row order unless unordered morsel output was explicitly requested.

Debug builds should assert row coverage, mask length, rank, cardinality, frontier monotonicity,
derived-demand completeness, and credit ownership at operator boundaries.

## Settled design choices

The proposal recommends treating these as architectural constraints:

1. Fixed morsels remain the outer scheduling unit.
2. Inner results are child-chosen contiguous prefixes within a parent-set bound.
3. Row domains and their transforms are first-class. One DomainMap serves demand derivation,
   catalog coverage, morsel-boundary discovery, and row identity.
4. State has three tiers: immutable PlanRef, per-scan ScanState, per-morsel ExecGraph.
5. Static read discovery happens once per scan, with per-morsel views.
6. Data-dependent reads are exposed by explicit gates, which are exactly the non-static maps.
7. The central scheduler owns admission, deduplication, and final priority.
8. Open demand is owned by DemandLedger. Projection planning may consume immutable open snapshots
   and summaries for candidate I/O and explicitly safe discovery work; exact or fallible value
   execution receives sealed immutable demand.
9. Exact masks remain the correctness representation; block summaries are scheduler accelerators
   with one authoritative fact and derived caches.
10. Drive registers any mix of work and runs to quiescence, and every non-terminal outcome carries
    evidence of progress.
11. Parents align by capping the request; a child that overshoots retains the surplus itself.
12. Compressed read-ahead and decoded materialization have separate horizons and budgets. Credits
    are reserved per morsel and the oldest in-flight morsel is never denied.
13. Root rebatching isolates consumers from natural internal boundaries and can ship before the
    execution graph.

## Open questions

These are not implementation details and should be settled before the phases that depend on them.

| Question | Depends on it | Current evidence |
| --- | --- | --- |
| Can demand widen after a scan opens? | whether DemandEpoch exists at all, and what ExecGraph must be able to discard | Selection is fixed at construction; pruning, evidence, and predicates all intersect; the only dynamic predicate is applied as file pruning before the scan opens. Incremental Take resolves without widening. If no case exists, delete the epoch machinery and keep one debug assertion. |
| Do describe_reads and open_exec share one traversal? | the public plan hook | Prefer the smallest public API; decide after the vertical slice |
| Is 1,024 rows the right demand block? | mask cost versus coverage precision | Measure |
| Which computation classes may speculate, and who classifies them? | Eval and evidence behavior | Undecided |
| Which current ordered-error behavior is contractual? | root stream semantics | Must be recorded as an oracle before it can be preserved |

## Choices to validate in prototypes

| Choice | Initial default | Evidence needed to change it |
| --- | --- | --- |
| Demand block size | 1,024 rows | Mask cost, catalog coverage precision, and filter latency |
| Internal batch target | 8,192 rows | Decode throughput, first-batch latency, and retained memory |
| Minimum prefix | small fixed row count | Drives per committed row |
| Morsel size | 100,000 rows | Storage alignment, scheduler overhead, and parallelism |
| Execution storage | Boxed tree | Profiled dispatch, allocation, or recursion cost |
| Transition budget | Fixed small count per drive | Drives per batch and coordinator fairness |
| CPU task threshold | Inline cheap coordination; schedule decode/eval | Task launch cost and worker utilization |
| Speculative CPU | Disabled unless explicitly safe | Proven error semantics and retained-byte benefit |
| Segment granularity | Preserve whole-segment decode first | Page metadata and independent decode support |
| Scheduler scoring | Lazy generation-based rescoring | Queue churn and stale-priority measurements |
| Estimated block counts | Omitted | Demonstrated read-ordering benefit |
| Error ordering | Match current public behavior | Differential tests for ordered and unordered scans |

The [self-paced implementation plan](self-paced-implementation-plan.md) turns this design into
reviewable phases, with an adapter that keeps the current exact PlanVTable execution path available
until semantic and performance parity are demonstrated. The
[design review](self-paced-review.md) records the evidence behind the choices above.
