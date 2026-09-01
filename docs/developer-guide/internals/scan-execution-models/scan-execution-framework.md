# Scan Execution Framework

This document is the design for a production-shaped scan execution framework, synthesized from the
[self-paced experiment's](self-paced-plan-exec-experiment.md) measured evidence — the
[findings](self-paced-plan-exec-findings.md), the [reference](self-paced-executor-reference.md),
and the [handover](self-paced-plan-exec-handover.md) — and from the design discussion that
followed it. It defines the components, the traits each plan concept implements, and the
execution lifecycle. It is a design, not an implementation; every claim marked *measured* traces
to the findings report.

The model in three sentences: **planning symbolically transforms a row domain down the plan tree
(sliced for chunked, duplicated for struct, expanded for list) and that symbolic tree defines how
demand propagates; binding converts the plan into thread-free execution wiring — ledger slots,
writer tokens, gate placeholders — as pure layout; execution lazily realizes a graph of I/O and
CPU obligations at the demand frontier and streams output batches back as spans seal.**

## The three-phase lifecycle

```text
PLAN     symbolic map tree, gates named            shared, immutable, thread-free, per query
BIND     wire demand IDENTITY                      once per scan: layout, never compute
EXECUTE  flow demand VALUES                        parallel, lazy, demand-gated, per unit
```

The bind contract: `convert(plan, query, boundaries) -> { ledger layout, writer tokens, unit
descriptions, gate placeholders }`. Binding allocates identity — pre-sized slot arrays, one
single-writer token per slot, reader edges as indices — and never computes extents. The measured
line (*plan-time materialization of segment cutting regressed FineWeb ~0.34 -> 0.39*): O(units +
fragments) slot layout is binding's job; O(segments x units) concrete cuts belong to execution,
where sixteen threads do the arithmetic in parallel for ~free.

Units are bound to **units, not threads**: binding produces a thread-free wiring diagram, and the
pool late-binds units to whichever thread pulls them (*measured: static thread assignment lost to
cursor self-scheduling on every uneven workload*).

## The laws

1. Shared data is immutable-after-publish; mutable data has exactly one owner. APIs make
   violation unrepresentable: sealing consumes a writer token.
2. Demand is a monotone chain of `IS TRUE`-collapsed bounds. Empty is final. Sealed is immutable.
   Kleene (three-valued) state never escapes a single kernel's expression subtree.
3. Planning does layout, never compute.
4. No per-operation task reification: pool items obey granularity floors; graphs are phase-level
   (*measured: per-segment predicate tasks ran 2.3x slower than inline kernels*).
5. Wakes route directly (fact -> waiter -> pool); nothing rescans.
6. Work is deduplicated by its natural key: demand by (domain, fragment), physical facts by
   `SegmentId`, counts by range. Sharing falls out of keying, never out of special cases.
7. Everything is reported with its input demand; admission is decided centrally from per-unit
   frontier heads; central scheduler state is O(units), never O(tasks) (*measured: a central
   candidate queue scanned 23.8 entries per admission; unit-resident candidates scanned 1.67*).
8. Every batch's memory footprint is exactly its span; executor retention is one thread's working
   set.

## The traits

### `DomainMap`: one row-domain relationship, an open set

The transform between a parent and child row universe is a trait, because new relationships will
exist. Its symbolic queries are answerable at plan time and drive binding, boundary derivation,
and coverage; its transforms run at execution (after the gate fact, for gated maps).

```rust
trait DomainMap: Send + Sync {
    // symbolic — answerable at PLAN time
    fn is_static(&self) -> bool;              // false => a gate fact must resolve me first
    fn prefix_preserving(&self) -> bool;      // can prefixes stream through this edge?
    fn exact_for_fallible(&self) -> bool;     // Coarsen-like maps may drive metadata work only
    fn gate(&self) -> Option<GateKind>;       // which fact resolves me

    // transforms — callable at EXECUTE time
    fn map_range(&self, parent: Range<u64>) -> Range<u64>;
    fn map_demand(&self, parent: &Bound) -> Bound;      // down
    fn unmap_mask(&self, child: &Bound) -> Bound;       // up, pure renumbering only
}
```

`Identity`, `Shift`, `Fence`, `Coarsen`, `MonotoneGated`, and `GatherGated` are implementations,
not variants. The four symbolic queries are the admission test for a new relationship: a map that
cannot answer them cannot be scheduled soundly.

Maps are immutable. A gated map never mutates when its fact arrives: `realize(fact)` constructs a
fresh immutable concrete map, owned by the realizing unit or shared through a resource cell. A
map that "needs" interior mutability is holding a fact that belongs in the fact layer.

Ownership: `Edge { map: Box<dyn DomainMap> }` inside the plan tree; the tree root behind one
`Arc`; hot paths borrow `&dyn DomainMap`. Runtime-realized maps are `Arc<dyn DomainMap>` in
cells. Refcounts live at coarse grain only (*measured: per-task `Arc` sharing regressed ~3%*).

### `Edge`: children plus their relationship

An edge is `(child, map, role)` — the map belongs to the relationship, not to either node. One
node may have children under different maps (list: offsets under `Fence`, validity under
`Identity`, elements under `MonotoneGated`), and `Role` (field, validity, offsets, values) lets
binding and the drive loop treat them differently without node-specific traversal code.

```rust
struct Edge { child: PlanNodeRef, map: Box<dyn DomainMap>, role: Role }
```

### `PlanNode`: declare, then bind

```rust
trait PlanNode {
    fn edges(&self) -> &[Edge];                    // PLAN: symbolic shape
    fn bind(&self, b: &mut Binder) -> NodeId;      // BIND: allocate identity, recurse children
}
```

`bind` allocates ledger slots per (domain, fragment), mints writer tokens, places gate slots for
non-static edges, and records reader indices. It performs no data-shaped work.

### `ExecNode`: pure hooks, no control flow

The bound, executable counterpart of one plan node — one per node per scan, immutable, shared by
every unit. Edges own coordinates; the node owns **combining semantics**.

```rust
trait ExecNode {
    fn push(&self, span, demand: &Bound) -> Vec<Obligation>;   // demand down, cut + priced
    fn pull(&self, span, results: Children) -> NodeOut;        // results up: combine
    fn gate(&self, fact: &Fact) -> Vec<Obligation>;            // gated expansion
}

enum Obligation {
    Read   { segment: SegmentId, demanded: usize },   // a task iff demanded > 0
    Kernel { op, floor },                             // inline unless floor-exceeded and stolen
    Child  { node: NodeId, span, demand: Bound },     // recurse
    Needs  { gate: GateId },                          // park until the gate's fact seals
}
```

Hooks never await, never touch the pool or ledger, never hold a bound — they are pure functions
testable with plain values. Mutable execution state (bounds, cursors, parked obligations) lives
in the unit, which is what keeps `ExecNode` lock-free and shareable.

Most nodes need no hand-written `ExecNode`: a generic implementation covers any node whose
combine is "assemble children by coverage order" (chunked is the generic node over `Shift`
edges). Custom nodes exist where combining is semantic: struct's zip, list's Kleene any-per-run
reduce, dict's `take(values, codes)`. Extension is therefore two-tier — new coordinate
relationship: implement `DomainMap`; new combining semantics: implement `ExecNode`; new leaf
encoding: just a kernel.

### Domains and the demand ledger

One `Domain` per row universe — not per edge, not per node — allocated at bind:

```rust
struct Domain {
    id: DomainId,
    extent: Extent,                        // Static(rows) | Gated(gate_slot)
    fragments: Box<[FragmentSlot]>,        // this domain's slice of the ledger
    derives: Option<(DomainId, MapRef)>,
}

struct FragmentSlot {
    state:   AtomicU8,                     // Open | Sealed | SealedEmpty
    bound:   Bound,                        // current best mask + count, version-stamped
    version: AtomicU32,                    // bumps on open refinement
    waiter:  AtomicPtr<Item>,              // woken ONLY on seal / sealed-empty
}
```

Most plan "domains" never materialize:

| Kind | Example | Representation |
| --- | --- | --- |
| Root | scan rows | real `Domain`; the filter spine writes its bounds |
| Identity-shared | struct fields, list validity | the **same** `DomainId` — no object |
| Static-renumbered | chunk-local coordinates | none — `map_range` at the point of use |
| Gated-derived | list elements, dict values | real `Domain`, extent realized when the gate seals |

Update discipline — **pull refinements, push seals**: open-bound refinements update the slot in
place (single writer, version bump) and notify nobody; consumers read the current bound when they
price work, and any version they read is a valid superset forever. Derived-domain demand is
computed lazily at first need, memoized by (fragment, parent version); the default rule derives
gated domains only from sealed parent fragments, making derived demand final at birth. Only two
events push: `Sealed` (unlocks projection and sub-domains) and `SealedEmpty` (cancels dependent
obligations before they become tasks) — each one release-store plus a waiter drain.

Contention: writes are partitioned by construction (single writer per slot; fork-join kernels
write disjoint word-aligned sub-ranges), publication is one atomic per seal (~thousands per scan,
distributed), reads after seal are lock-free on frozen `Arc` buffers, and parking is one CAS. The
metrics layer counts CAS retries and parks from day one.

### Resource cells

Scan-wide once-cells keyed by physical identity: `SegmentId -> decoded array` and
`(SegmentId, conjunct) -> evaluated mask`, with the experiment's proven pinned / reusable / dead
refcount lifetime. The first toucher of a coarse filter segment evaluates it once at full width
(*the fast kernel regime*); every overlapping unit slices the mask. Same concurrency shape as a
ledger slot, keyed by segments instead of rows.

### `UnitDriver`: the drive loop behind a trait

```rust
enum Item { Unit(UnitId), Span(UnitId, SpanId), Kernel(..) }   // the pool's vocabulary

trait UnitDriver: Send + Sync {
    fn drive(&self, u: &mut UnitState, ctx: &mut ThreadCtx) -> Drive;
}
enum Drive { Parked, Retired }
```

A thread pulls `Item::Unit`, receives exclusive `&mut UnitState` (a pool guarantee — the item is
not re-enqueueable while running), and drives until nothing can proceed. Wakes re-enqueue the
item; any thread resumes it. The driver contract, enforced structurally where possible: one
thread at a time; never block a thread — park with registered waiters; publish bounds only
through writer tokens; issue reads only from priced obligations; emit spans in order; release the
working set on retirement.

Alteration levels, cheapest first: **parameters** (demand order, speculation, floors, credits);
**the routing table** — the single seam where every composition differs:

```rust
trait Route { fn route(&self, o: &Obligation, u: &UnitState) -> Placement; }
enum Placement { Inline, Pool, Park, Register }    // Register: visible dormant candidate
```

**stage hooks** (`on_bound`, `on_seal`, `on_gate`, `on_emit`); and finally **replace the driver**
(the experiment's pipeline and reactor are both honestly described as existing `UnitDriver`s).

### The drive loop

```text
        EXPAND   current stage's obligations
           |       Filtering(k): node.push(filter field k, bound_k)
           |       Projecting:   spans whose demand sealed
        PRUNE    demanded == 0 -> obligation dropped (a task never exists)
           |       bound empty  -> SEAL EMPTY, emit dense zero-value batch
        ISSUE    Read   -> cell hit? adopt now : start async I/O
           |       Kernel -> below floor? run inline : pool item
           |       Needs  -> CAS onto the gate/ledger waiter slot
   ready work? --yes--> EXECUTE inline --> ADOPT ----------------------.
        no                                                             |
        PARK the unit; the thread pulls the next item                  |
          wake: own I/O done, or a parked-on fact sealed --------------|
        ADOPT    install fact; own results publish via writer token    |
                 conjunct adopted -> intersect, next bound; last: SEAL |
                 gate fact -> node.gate(fact) -> new obligations ------'
        EMIT     completed spans in order: pull up, sink(batch), release chunks
        RETIRE   drop cell refcounts, clear cache, return to the pool
```

Fragments give prefix progress inside a unit: one fragment can seal and project while a sibling
is still on its first conjunct. Under the decoupled composition, SEAL publishes span items to the
pool instead of looping into projection locally — one `Placement` change, same machine.

## Demand semantics

Demand is definitionally two-valued (read-or-don't). SQL's three-valued logic is confined to the
expression layer by the collapse rule — `IS TRUE` distributes through AND and OR but not NOT:

```text
(a AND b) IS TRUE  <=>  (a IS TRUE) AND (b IS TRUE)     per-conjunct collapse is lawful
(a OR  b) IS TRUE  <=>  (a IS TRUE) OR  (b IS TRUE)     per-disjunct too
(NOT a)   IS TRUE  <=>  a IS FALSE                      stay Kleene beneath a NOT
```

The filter compiler normalizes NOT to the leaves (a negated comparison stays collapsible: nulls
still drop), coalesces same-field predicates, and emits an AND-spine of leaf conjuncts. Final
demand exists at the spine's end, but the working currency is the monotone chain of upper bounds
produced at every step — each one valid to price, skip, and speculate against. Empty bounds and
pruning (zone stats: min, max, null_count) finalize early. Disjunctions yield bounds only from
statistics or completion of all branches. Kernels over nullable columns compute
`cmp(values) AND validity`; validity otherwise rides inside arrays as payload — positions are
never null, values are.

Order freedom is safe by the superset rule: a result evaluated against any earlier bound of the
same fragment is a superset of every later bound and adopts by intersection. Reordering and
concurrency are pure execution choices; the output hash is invariant.

## Scheduling: report everything, admit centrally, keep the queue full

Every obligation is reported with its input demand and expected-value facts:

```rust
struct Reported {
    bound_version: u32, demanded: usize,
    bytes: usize, phase: Phase,
    remaining_selectivity: f32,        // expected further shrink, from observed stats
    necessity: Required | Candidate,
}
```

Reported items rest in their unit's frontier (law 7); the scheduler sees one **frontier-head
register per unit**. Admission for a candidate read weighs waiting against getting:

```text
EV(issue now) = latency_hidden(queue_depth, source) - P_skip x bytes
P_skip        = 1 - product of remaining conjunct selectivities
```

under a queue-fullness watermark: below the source's target depth, admit the best head even at
mildly negative EV (an idle queue slot hides latency for free); at depth, admit only positive EV.
Required work bypasses EV but draws from a reserved share, oldest unit first — speculation can
never starve the commit frontier. Issued candidates keep their identity on promotion; dormant
candidates evaporate at seal-empty having cost nothing. The two counters that define success:
entries-considered-per-admission (~1) and queue-idle time (~0).

The cascade and the prefetch-everything policies are the same machine at two points on this
curve: with in-memory latency the EV of waiting dominates (*measured: cascade optimal*); at
object-store latency the EV of issuing dominates (*measured: Q06's early bytes all became
required; Q01 wasted half — the score, not a global default, decides*).

## Compositions as configurations

| Composition | Unit boundaries | Ledger writers | `Route` |
| --- | --- | --- | --- |
| Unified (model 2) | union of splits | the unit, one seal | everything `Inline` |
| Decoupled (model 1) | filter splits | filter units, per prefix | spans -> `Pool` |
| Fork-join filter | filter splits | sub-range kernels + join seal | eval kernels -> `Pool` |
| Coarse-filter sharing | any | cell -> ledger publishes | unchanged |
| Prefetch-heavy | any | unchanged | candidates -> `Register` |

Filter parallelism comes from data (fork-join sub-ranges over whole segments); projection
parallelism comes from survivors (stealable sealed spans); the unit is only the ownership,
ordering, and sealing container — its size stops mattering for parallelism.

## Build order

Each step gated by the degenerate A/B: re-express current behavior in the new component at
measured-zero cost before any new composition ships (the gate the `FieldDomain` refactor passed).

1. **Demand ledger** — the enabling dependency; built with the thread-local fast path.
2. **Resource cells** — small, proven lifetime; fixes coarse-filter sharing and cross-unit
   re-reads.
3. **Pool with gated items** — sealed spans first (the Q6 makespan fix), then fork-join filter
   kernels (the coarse-filter starvation fix).
4. **EV admission and emitter credits** — once compositions exist that can run ahead.
5. **Filter compiler** (parallel track — pure); gated nodes (dict, list) and real I/O
   (ranged/multi-get source, per-unit read-ahead) when the layout restriction lifts.

The oracle stack is non-negotiable throughout: eager reference, ordered-hash gates,
per-iteration cold-scan I/O invariant, external engine oracle, and contention counters from day
one.
