# Scan Execution Graph Model

Working notes from a design discussion (2026-08-25) that continues the
[scan execution framework](scan-execution-framework.md). The framework document defines the
components and traits; this document re-derives them from a smaller foundation — a typed
dependency graph — records the decisions that discussion produced, and keeps the open questions
that still need answers. It is a thinking document, not a commitment.

The model in one sentence: **nodes are work (typed CPU, IO, or Plan), edges are cells (typed
demand or data), and every earlier machine — the ledger, the drive loop, the reactor's slots — is
a projection of that graph plus two materialization rules.**

## 1. What each state machine models

The framework and experiment documents define nine state machines. Sorted by underlying concern
rather than by owning component, there are only five concerns:

1. **Knowledge that grows toward finality.** The fragment slot (`Open -> Sealed | SealedEmpty`)
   is the epistemic state of demand: the bound may still shrink; at some point it is final. Gate
   facts and decoded arrays are the same thing with a degenerate lattice (absent, then final).
2. **Fulfillment of in-flight computation.** The resource machine's interior states (`Reading`,
   `Decoding`) describe work running against a fact, not the fact itself. The five-state resource
   machine conflates three knowledge states with two work-in-flight markers.
3. **Exclusive write ownership.** The reactor's result slot (`Empty -> Offered -> Running ->
   Ready`) mostly models who may produce a value, exactly once. Writer tokens minted at bind turn
   this from a state you check into a capability you hold; sealing consumes the token.
4. **Scheduling position of work.** Task states, `Required | Candidate`, and `Placement` model
   where work sits relative to the scheduler: described, dormant, admitted, running, done.
   `Placement` is a routing decision — an event, not a state.
5. **Progress of a sequential control locus.** Pipeline phases, drive-loop stages, and
   `Budgeted | Quiescent | Retired` are all a program counter; the last is the scheduler-visible
   summary of one (runnable, blocked, finished).

Retention (`pinned | reusable | dead`) is not a machine: it is derived from holders plus "does any
unretired unit's coverage overlap this."

### The unified primitives

Three primitives; everything else is derived.

**Primitive 1 — the fact cell.** One single-writer monotone cell, one park/wake mechanism, three
keyed tables:

```rust
Cell<V> {
    state:   Open(version) | Final,      // release-store on seal
    value:   V,                          // shrinks (bound) or fills (array, mask, map)
    writer:  Token,                      // minted at bind, consumed by seal
    waiters: AtomicPtr<Item>,            // CAS park; drained only on Final
}
// keyed by (domain, fragment)      V = Bound        — the demand ledger
// keyed by SegmentId / (seg, conj) V = Array | Mask — resource cells
// keyed by GateId                  V = realized map — gate facts
```

Demand is the general case (meaningful intermediate refinements: in-place update, version bump,
nobody woken). Physical facts and gates are the degenerate case whose only refinement is the
seal. `SealedEmpty` is not a third state: it is `Final && value.is_empty()`; cancellation is the
consumer's reaction to reading that. `Failed` also disappears: an error seals the cell with an
error value, so propagation rides the existing wake path.

**Primitive 2 — work, with a position.** An obligation plus a position in one lifecycle:
`Frontier -> (Registered) -> Admitted -> Running -> done`, where done means "my output cell
sealed" — work has no completion state of its own. The fast path (`Frontier -> run inline ->
seal`) touches no shared state; position is materialized only for the pool-visible minority.
`Required | Candidate` is an attribute of `Registered` work.

**Primitive 3 — the driver.** `UnitState` is a private program counter plus scratch (bounds,
parked obligations, span countdowns), with three scheduler-visible summaries: runnable, parked,
retired. Fact state lives with facts, work state with the scheduler's frontier registers, control
state privately in the unit — the reactor conflated the first two and the coordinator conflated
the last two, and both were the measured failure modes.

| Old machine | Was modeling | Becomes |
| --- | --- | --- |
| FragmentSlot `Open/Sealed/SealedEmpty` | demand knowledge | Cell; `SealedEmpty` derived |
| Resource `Absent..ArrayReady` | knowledge x in-flight work | two chained Cells + work positions |
| Result slot `Empty..Ready/Failed` | ownership x fulfillment x scheduling | token + Cell + position |
| Task offered/claimed/completed | scheduling position | work position (slow path only) |
| `Placement` | routing decision | a transition function, not a state |
| Morsel `Budgeted/Quiescent/Retired` | driver runnability | driver summary |
| Pipeline phases / drive stages | program counter | private PC |
| `pinned/reusable/dead` | retention | derived from refcounts + unit retirement |
| `Required/Candidate` | admission class | attribute on Registered work |

## 2. The graph

**Nodes are work, typed by resource class:**

- **IO(segment)** — produces a bytes fact. Latency-bound; admission is per-source queue depth.
- **CPU(kernel)** — decode, predicate eval, intersect, assemble. Compute-bound; admission is
  floors and the pool.
- **Plan(expander)** — produces *graph*: when run, it splices new IO/CPU/Plan nodes between its
  neighbors. Near-free, and constitutionally forbidden from compute: a Plan node consumes only
  metadata and already-produced facts (law 3 made structural).

**Edges are cells, typed by lattice:**

- **data edge** (flows up, producer to consumer): a fact cell — Empty then Final.
- **demand edge** (flows down, consumer to producer): a bound cell — Open, refining
  monotonically, then Final. Every node prices itself against its incoming demand.

Edges carry a `DomainMap` wherever they cross a row-universe boundary: the map is the edge label,
not a node.

### The laziness rules

The demand edge into a Plan node has three observable states, and they are the whole policy:

| Demand edge state | Meaning for the Plan node |
| --- | --- |
| Final, empty | Never runs. The subtree it would splice **never exists** — nothing to cancel. |
| Final, nonempty | Required work: expand now; produced IO/CPU nodes bypass EV. |
| Open | Expand **only under speculation**; everything produced is a Candidate, priced by EV. |

"Not enough IO in flight" is the trigger for the third row: below the per-source queue-depth
watermark, the scheduler runs frontier Plan nodes on their open (superset) bounds. The superset
rule keeps it sound: a read issued against an earlier bound adopts by intersection. Cascade,
eager, and adaptive stop being demand policies and become **planning-admission policies** — the
same graph at different points on the watermark curve.

Gated expansion also lands naturally: a Plan node that needs a decoded fact (list offsets, dict
codes) has a **data edge** into the fact's producing CPU node. `Needs(gate)` means "a Plan node
whose data input is not yet Final." The decode itself stays a CPU node: planning consumes facts
but never computes them.

### Virtual, not reified

This graph existed once, fully materialized, with a coordinator walking it — the reactor, 2.3x
slower. The graph is the **specification**; the machine traverses it mostly without materializing
it. Two rules recover the pipeline's speed:

- **Edge materialization rule:** an edge becomes a real cell iff it is shared (fan-out > 1) or
  crosses a park/pool boundary. Otherwise it is a value on the stack of an inline traversal.
  Fan-out is knowable at bind, so the fast path is branch-predictable.
- **Node fusion rule:** maximal chains of CPU nodes connected by private edges execute as one
  kernel invocation; the granularity floor governs the fused chain, never individual nodes.

Under these rules the compositions become traversal strategies over one graph: the pipeline is
depth-first inline with almost no materialized edges; decoupled materializes sealed-demand edges
and pools spans; fork-join materializes sub-range kernel edges; the reactor is the
"materialize everything" corner, retained as the observable, contract-checking configuration.

Two things stay outside the graph deliberately. **Ordering:** the graph is unordered; emission
order comes from units owning contiguous root-coverage spans. **Retention:** derived, as above.

## 3. Domain changes: edge, node, or neither

A domain change decomposes into three things, and dedup comes from keyed cells, never from nodes
(two consumers spawning two nodes *is* duplication; law 6 keys the work's output instead):

| Crossing | Work | Node? | Dedup point |
| --- | --- | --- | --- |
| Identity (struct fields) | none | no — same `DomainId` | n/a |
| Static arithmetic (Shift, Coarsen) | O(1)–O(log) | no — inline at use | none needed |
| Gated realize (offsets, gather set) | real CPU (+IO) | **yes** | concrete-map cell, keyed by gate |
| Derived demand in child domain | O(demanded) | first-needer computes | child ledger slot, (fragment, parent version) |

The realize node's output is a concrete-map fact cell keyed by gate identity; every edge crossing
that boundary data-depends on the same cell. Derived demand dedups through the child domain's
fragment slots — which is why Domain is per row universe, not per edge. Cheap static crossings
stay inline: deduping O(log) arithmetic through a shared cell costs more than recomputing it.

Decision recorded: gated child-domain ledgers are allocated when the gate seals (the extent is
unknown before then); allocation is cheap enough that laziness beyond that point buys nothing.

## 4. Responsibilities: the eager model first

Reasoning aid: set every demand edge to the constant top (all rows), Final at birth. The graph
becomes a pure data-dependency DAG and each responsibility is crisp. Laziness returns afterwards
as three refinements that change *when*, never *what*.

| Component | Creates | Cardinality | When |
| --- | --- | --- | --- |
| Lowering + optimizer | plan-tree nodes, edges with maps | O(layout shape) | once per query |
| Binder | domains + fragment slots, cells for shared edges, writer tokens, unit boundaries, one exec node per plan node | O(plan + units + fragments) | once per scan |
| Expanders (`expand`/`gate`) | the work nodes: IO, CPU, and their edges | O(segments x fragments) | eager: a pass after bind, staged only by gates |
| Realize nodes | concrete maps; child domains | O(gates) | when the gate's data input seals |

The driver runs nodes and seals cells; the scheduler admits runnable nodes; neither creates
structure. Exec nodes are **not** the numerous thing — one per plan node per scan, immutable,
shared by every unit. The numerous things are work nodes, created exclusively by expanders, and
under laziness most never exist.

Two observations the eager form surfaces:

- **Plan nodes are nodes even with zero laziness**: a gated expander cannot run before its fact
  exists — a data dependency, not a laziness artifact.
- **The eager graph is exactly `run_eager`**: every conjunct over every row, intersect, project
  all, select at emit. The eager path is the differential oracle, and should be the first thing
  the new code path can execute.

The three refinements, each a legal transformation because expanders are **pure functions of
(fragment, bound, facts)** — deferring a pure function changes when it runs, never what it
produces:

1. **Demand laziness** — real bounds replace top; Final-empty deletes subtrees pre-birth.
2. **Expansion laziness** — expanders run at first nonzero demand or under speculation, moving
   O(segments x fragments) cutting from a serial bind onto parallel execution threads (the
   measured plan-time-materialization regression, in reverse).
3. **Materialization laziness** — private edges become stack values; private chains fuse.

Any lazy configuration must hash-match the eager oracle on every workload: a property test over
the transformation, not over any node.

## 5. Worked example: `Struct(Chunked(Flat))`, two filters

Fields `a`, `b`, `c` with unaligned chunks; query `WHERE a > 5 AND b < 3, SELECT a, c`
(projecting `a` so filter/projection sharing appears). One root domain R; one unit, fragment
F = [0,100).

```text
chunks:  a = {[0,40), [40,100)}   b = {[0,60), [60,100)}   c = {[0,40), [40,70), [70,100)}
demand flows down the spine; data flows up; * marks a materialized (shared) cell

        bound0 = top  (F's slot in R's ledger, open)
           |
  Plan(a > 5)         cuts F against a's chunk table
  |  IO(a0) -> CPU(decode a0)* -> CPU(eval a>5 [0,40))   \
  |  IO(a1) -> CPU(decode a1)* -> CPU(eval a>5 [40,100)) -+-> CPU(unmap+combine) -> bound1
           |
  Plan(b < 3)         demanded by bound1 (cascade) or top (eager)
  |  IO(b0) -> CPU(decode b0) -> CPU(eval b<3 [0,60))    \    private chains -
  |  IO(b1) -> CPU(decode b1) -> CPU(eval b<3 [60,100))  -+-> CPU(unmap+combine) -> bound2 = SEAL
           |
  Plan(project {a, c})   span cuts = union of projected boundaries: [0,40) [40,70) [70,100)
  |  a: data edges into decode(a0)*, decode(a1)*  — no new IO
  |  IO(c0..c2) -> CPU(decode) -> per span: CPU(gather a) + CPU(gather c) -> CPU(pack) -> emit
```

What the graph says each concept *is*:

- **Struct is almost nothing**: identity edges create no nodes; struct contributes the span rule
  (boundary union of projected fields) and the pack combine.
- **Chunked disappears at runtime**: cutting arithmetic inside expanders plus Shift maps on
  edges. There is no Concat node executing anything — plan-tree nodes and graph nodes stop being
  one-to-one. (A conscious divergence from plan v2, where `Concat` executes.)
- **Flat is the only thing that touches the world**: the IO -> decode chains.
- **The filter is the demand spine**: eval kernels feeding combine nodes; `bound0 -> bound1 ->
  bound2` is one cell refined twice and sealed. The combine nodes are the only writers.
- **Projection is gathers under a sealed bound**.

Shared state for this whole scan-fragment: one fragment slot plus two decode cells. Everything
else is fused private chains — the graph-theoretic restatement of why the pipeline beat the
reactor.

Expansion is frontier-driven along three axes, and "expand all children" is the degenerate
setting of all three: **rows** (a cut is `partition_point` plus a walk of overlaps in the span —
non-overlapping chunks are arithmetic that never ran, not objects), **depth** (one level per
expand call; the driver recurses, with a leaf shortcut emitting Read+decode directly), **time**
(the three-state demand-edge rule). The one true early-expansion is speculation, and it is a
scheduler decision with a price — nodes cannot express eagerness.

## 6. The `ExecNode` trait

There are exactly three places in the graph where per-node-kind semantics appear; everything else
is generic driver work or a kernel. `Plan(..)` boxes are `expand` calls; realize nodes are `gate`
calls; the unmap/combine/pack CPU nodes are `combine` calls; eval/decode/gather are kernels
referenced by obligations; IO nodes are driver-issued.

```rust
trait ExecNode: Send + Sync {
    /// EXPAND — demand down. Cut `span` against my children, price each overlap, emit
    /// obligations. Pure: reads only immutable plan data and the given bound.
    fn expand(&self, span: SpanRef, demand: &Bound, out: &mut dyn ObligationSink);

    /// GATE — a non-static edge's fact sealed: realize the concrete map, then expand the
    /// gated edge (derived demand is sealed at birth). Default body: unreachable.
    fn gate(&self, gate: GateId, fact: &Fact, out: &mut dyn ObligationSink);

    /// COMBINE — results up. Children arrive pre-cut, pre-priced, and aligned; assemble one
    /// output. Returns a value; the DRIVER publishes it (a Bound through the slot's writer
    /// token, an Array through emit). Hooks never touch the ledger.
    fn combine(&self, span: SpanRef, children: ChildResults) -> VortexResult<NodeOut>;
}

enum NodeOut { Bound(Bound), Array(ArrayRef) }

enum Obligation {
    Read   { segment: SegmentId, demanded: u64, bytes: u64 },
    Kernel { kernel: KernelId, input: InputRef, span: SpanRef, floor: u32 },
    Child  { node: NodeId, span: SpanRef, demand: Bound },
    Needs  { gate: GateId },
}
```

Costs are structural: `expand` O(log chunks + overlaps), `combine` O(parts), `gate` O(fact);
per-row work exists only inside kernels (the no-dyn-in-row-loops rule made unbreakable). The
sink-shaped `expand` avoids a per-call allocation and lets the driver route obligations as
produced. Kernels are indices into a per-scan kernel table, keeping obligations flat and priceable.

| Exec node | `expand` | `gate` | `combine` |
| --- | --- | --- | --- |
| Generic coverage (Chunked, any static-map parent) | cut span against coverage table; price; emit Read/Kernel/Child per overlap | — | assemble by coverage order |
| Conjunct (Eval) | delegate cut to field edge, attach predicate kernel to leaf chains | — | unmap + AND of child masks -> `Bound` |
| Struct (Pack) | replicate demand handle to each field edge | — | zip into `StructArray` |
| Zoned (pruning) | metadata reads over the Coarsen edge (`exact_for_fallible`) | — | stats -> pruning `Bound` |
| Flat (leaf) | none standalone-trivial; chunked parent emits its Read+decode directly | — | pass-through |
| Dict | codes static; emit distinct-kernel + `Needs(values gate)` | gather set = sealed demand over values domain; expand values edge | `take(values, remapped codes)` |
| List | offsets static; `Needs(elements gate)` | realized offsets map; expand elements with run-expanded demand | run-collapse; Kleene any-per-run |

Most rows are not hand-written: generic-coverage is one implementation parameterized by a
coverage table and edge maps; Flat is zero implementations; the hand-written surface is exactly
where combining is semantic (intersect, zip, take, reduce). Filtering on a dict field needs no
new machinery: the planner evaluates the predicate over the (small) values domain and swaps the
row-side kernel for code-set membership — a different `KernelId` in an ordinary conjunct.

Dict nuance recorded: dict's values-domain **extent** is static (dictionary length is layout
metadata; the binder allocates the domain up front); only the *demand* over it is gated. List is
the stronger case where the extent itself waits on the fact.

Sketched impls from the discussion (design-shaped, not compiling code):

```rust
impl ExecNode for FlatExec {
    fn expand(&self, span: SpanRef, demand: &Bound, out: &mut dyn ObligationSink) {
        if demand.demanded == 0 { return; }
        out.emit(Obligation::Read { segment: self.segment, demanded: demand.demanded,
                                    bytes: self.estimated_bytes });
        out.emit(Obligation::Kernel { kernel: self.decode,
                                      input: InputRef::Segment(self.segment),
                                      span, floor: DECODE_FLOOR });
    }
    fn combine(&self, _span: SpanRef, children: ChildResults) -> VortexResult<NodeOut> {
        Ok(NodeOut::Array(children.sole_array()?))
    }
}

impl ExecNode for StructExec {
    fn expand(&self, span: SpanRef, demand: &Bound, out: &mut dyn ObligationSink) {
        for edge in self.fields.iter() {
            // Identity means SHARE: same domain, same bound handle, zero transform.
            out.emit(Obligation::Child { node: edge.child, span, demand: demand.share() });
        }
    }
    fn combine(&self, span: SpanRef, children: ChildResults) -> VortexResult<NodeOut> {
        let arrays = children.arrays_in_edge_order()?;
        Ok(NodeOut::Array(StructArray::try_new(self.names.clone(), arrays,
            span.selected(), Validity::NonNullable)?.into_array()))
    }
}

impl ExecNode for DictExec {
    fn expand(&self, span: SpanRef, demand: &Bound, out: &mut dyn ObligationSink) {
        if demand.demanded == 0 { return; }
        out.emit(Obligation::Child { node: self.codes.child, span, demand: demand.share() });
        // The driver wires this kernel's output to `self.gate`; its completion IS the seal.
        out.emit(Obligation::Kernel { kernel: self.distinct,
                                      input: InputRef::Node(self.codes.child), span, floor: 0 });
        out.emit(Obligation::Needs { gate: self.gate });
    }
    fn gate(&self, _gate: GateId, fact: &Fact, out: &mut dyn ObligationSink) {
        // The gather set is a demand over the values domain, sealed at birth.
        out.emit(Obligation::Child { node: self.values.child,
                                     span: SpanRef::whole(self.values_domain),
                                     demand: fact.as_gather_bound() });
    }
    fn combine(&self, _span: SpanRef, children: ChildResults) -> VortexResult<NodeOut> {
        let codes  = children.array(EDGE_CODES)?;
        let values = children.array(EDGE_VALUES)?;   // only the demanded pages, dense
        let gather = children.fact(self.gate)?;      // realized map, for renumbering
        Ok(NodeOut::Array(take_kernel(values, gather.remap_codes(codes)?)?))
    }
}
```

Value-page skipping in dict needs no code: the values subtree is a generic coverage node and the
gather bound prices its pages exactly like row demand prices chunks. The scan-wide dictionary
cache is the resource-cell layer keyed by `SegmentId`; the node stays ignorant of it.

## 7. Combine-once semantics

`combine` runs **exactly once per (node, span), with every input Final**. There is no update path
into combine, by construction: data cells are write-once; the only mutable thing (an open bound)
is an input to expansion and pricing, never to combine.

Mechanics: when `expand` emits a span's obligations, the unit's scratch records how many inputs
the span awaits. Each adoption decrements — O(1), a direct wake, nothing rescanned. The last
arrival triggers `combine` inline on the adopting thread.

Where the "updates" went:

- **Bound refinement is a slot meet, not a recombination.** Refinement adopted from this
  discussion: a conjunct's `combine` produces only the unmapped AND of its child masks; the meet
  with the current slot value happens at publish, in the driver. "Combine's inputs are immutable"
  becomes a theorem; stale (superset) evaluation is corrected by the same meet with no version
  bookkeeping in the hook.
- **Refinements notify nobody** (pull discipline); only `Sealed`/`SealedEmpty` push wakes.
- **Incrementality comes from granularity**: earlier output means smaller spans — many small
  complete combines, never repeated partial ones. Buffered inputs per span are one span's working
  set (law 8).

This is deliberately not a general incremental-dataflow engine: no memoization, no invalidation,
no delta-consuming combines, and therefore no glitch problem. The excluded shape is the
order-dependent sequential fold — see section 9.

## 8. Alignment and slicing

**The driver slices; `combine`'s contract is aligned children.** Alignment is a coordinate
concern, and edges own coordinates; nodes own combining semantics. The cut `expand` produced —
`(child_local, parent_local, demanded)` per overlap — is data in the span scratch; at
countdown-zero the driver builds `ChildResults` on the stack, slicing each adopted value to its
overlap (zero-copy; pass-through untouched when coverage equals the span). Counts flow from
pricing; nothing is recounted. Slicing is **per-edge, directed by the edge's map**: Identity and
Shift edges are sliced to span; a GatherGated edge's value arrives whole with its fact (dict
values are in dictionary coordinates — slicing them by span would be wrong).

Buffering has exactly two homes: **shared cells** (fan-out > 1; keyed, refcounted, released when
consumers drain) and **unit span scratch** (private countdowns). An ExecNode cannot buffer: it is
immutable and shared by every unit — node-level buffering is unrepresentable by design.

"Aligned" does not mean "single": chunked's combine still receives several parts (ordering and
wrapping them is its semantics), but each part is already in span coordinates with a known count.

## 9. Alternative considered: stateful push-and-zip nodes

Proposal examined: n children push arrays into a node that receives them statefully and zips
internally — a fold, strictly more general than `combine` (which is the fold that buffers
everything and runs once). Rejected for `ExecNode`, with the useful half recovered graph-natively:

- **State is per-span regardless of owner.** The node instance is shared by every unit, so
  "stateful node" means per-(node, span) accumulator state — which is what the driver's span
  scratch already is. The proposal only changes who defines the accumulator, at the cost of
  per-author coordinate and ordering bugs.
- **Out-of-order arrival collapses absorb into the buffer.** An absorb accepting part 3 before
  part 1 stores it (re-implementing the scratch); refusing it reintroduces prefix stalls
  (learning 10).
- **For assembly, absorbing early frees nothing.** List assembly is zero-copy wrapping; retained
  memory is identical either way. The one genuine win — compaction under selective demand — is
  expressible without statefulness: arrival-time processing is *more nodes*. A per-part
  compaction kernel (`decode -> gather demanded rows -> small part -> combine`) is the fold's
  absorb step reified as a pure node: priced, floor-governed, order-free.
- **List's other temptations dissolve the same way.** Kleene any-per-run early exit is a demand
  refinement (the run's bound seals on the first true, cancelling remaining element reads before
  they exist). Dynamic arity is fine: the countdown is set post-gate.

What is genuinely excluded, consciously: the order-dependent sequential fold. It serializes the
span, breaks eager-oracle equivalence (result depends on arrival schedule), and breaks the
legality of laziness (deferral would change *what*, not *when*). Scan-level pushdown aggregates,
if ever wanted, arrive as a separate explicitly-fold-shaped hook, never through `ExecNode`.

## 10. Decisions recorded from this discussion

1. Nodes are work (CPU | IO | Plan); edges are cells (demand | data); maps label edges.
2. One cell primitive, three keyed tables; `SealedEmpty` and `Failed` are derived, not states.
3. The graph is virtual: edges materialize iff shared or crossing a park/pool boundary; private
   CPU chains fuse. The reactor corner is the debug configuration.
4. Plan-node laziness is the three-state demand-edge rule; speculation is the scheduler running
   frontier Plan nodes on open bounds below the IO watermark; nodes cannot express eagerness.
5. Domain relationships are edge labels; gated realization is a node with a gate-keyed fact cell;
   derived demand dedups through the child domain's ledger; static crossings stay inline.
6. Gated child-domain ledgers allocate at gate seal; no laziness beyond that (allocation is
   cheap and sized).
7. Eager-first responsibilities: lowering -> binder -> expanders -> realize; exec nodes are
   O(plan) and shared; work nodes are O(segments x fragments) and created only by expanders. The
   eager configuration is `run_eager` and serves as the permanent differential oracle.
8. Expander purity is what makes every laziness a legal transformation.
9. `ExecNode` is three methods (`expand`, `gate`, `combine`) plus a kernel table; sink-shaped
   expand; kernels by table index; `combine` returns values and the driver publishes.
10. Chunked compiles away: plan-tree nodes and graph nodes are not one-to-one.
11. Combine runs once per (node, span) on Final inputs; countdown in span scratch; conjunct
    intersection happens as a slot meet at publish, not in the hook.
12. The driver slices to alignment from the cut-as-data, directed per-edge by the map;
    buffering lives only in shared cells and span scratch.
13. `ChildResults` passes all children at once, and carries facts as well as arrays (the dict
    remap wart, accepted for uniformity over per-edge-kind adoption in the driver).
14. Sequential folds are excluded from `ExecNode`; arrival-time work is per-part pure kernels.

## 11. Open questions

The [next-discussion document](scan-execution-graph-next-discussion.md) expands the largest of
these into problem statements with context, so a future session can start there the way this one
started from the framework document. Carried forward, roughly in the order they should be
settled:

1. **Pool scope.** Session-wide pool shared by all scans (small DataFusion-opener scans become
   single unit items; big scans share threads) versus per-scan pools for isolation?
2. **Build order.** Start at the doc's build order (ledger first), or at the pain point — unit
   formation plus the `select *` composition (splits.rs today only subdivides, never merges;
   fragment = natural split, unit = byte-budgeted coalescing) — or stand up the eager path
   end-to-end first as section 4 suggests?
3. **The closed `Obligation` enum.** Commit that new work classes are framework changes? (The
   admission test: can a node express its work as Read/Kernel/Child/Needs?)
4. **Observability vs virtuality.** How much of the graph does the debug/reactor configuration
   reify, and is that configuration always available (compile-time or runtime switch)?
5. **Are Plan nodes schedulable at all**, or always executed inline by whoever holds the
   frontier? (Lean: inline always — O(metadata), and pooling them reintroduces coordinator-shaped
   latency.)
6. **Speculation floor.** A minimum-density threshold for speculatively planning against an open
   bound (the planning-side analogue of the kernel regime switch)?
7. **`ChildResults` shape.** Finalize: slice of pre-adopted outputs in edge order with priced
   counts, plus facts; no lazy pulling inside `combine`.
8. **Kernel table representation.** Per-scan table built by the binder; what exactly is a
   `KernelId`'s payload (fn pointer + flat args?) so pricing data stays flat?
9. **Composition selection per scan.** Which plan properties choose the Route configuration
   (filter present, expected selectivity, projected byte width), and where does that decision
   live?
10. **Conformance harnesses.** Property suites for `DomainMap` (round-trip superset laws,
    prefix-preservation implies monotone `map_range`) and differential per-node harness for
    `ExecNode` against the eager driver — ship with the traits from day one?
11. **Grafting onto `PlanVTable`.** Add `edges()`/`bind()` alongside the current `execute` with
    unimplemented defaults so nodes migrate one at a time, keeping v2's future path as the oracle
    during the build?
12. **Emission and limits.** Where limit demand enters the graph (a first-k bound at the sink?)
    and how span-ordered emission interacts with cross-unit ordering restoration.
13. **Aggregate pushdown hook.** If scan-level count/min/max from stats is ever wanted, define
    the separate fold-shaped hook rather than widening `ExecNode`.
