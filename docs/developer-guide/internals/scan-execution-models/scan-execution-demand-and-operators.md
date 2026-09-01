# Scan Execution: Demand, Operators, and the Filter Law

Working notes from a design discussion (2026-08-25) that continues the
[scan execution graph model](scan-execution-graph-model.md). That document derived the execution
framework from a typed dependency graph with three primitives; this one records the next
conversation, which simplified it further. The demand system stops being a runtime propagation
network and becomes bind-time routing; the two-conformance question (positional versus compacted
values) collapses to a single value contract with one cardinality-changing node; and the
components regroup into a three-part architecture: stateful per-layout exec plans, a demand mask
system with planning-time pushdown, and a scheduler licensed to run work optimistically by one
commutation law. It is a thinking document, not a commitment.

The conclusion in one sentence: **operators are stateful and demand-ignorant, demand is a
bind-time-routed skipping mechanism whose only runtime content is masks, and speculation is legal
because selection commutes with total row-local kernels — with `filter` as the single node that
requires sealed demand.**

## 1. Refinements to the `ExecNode` trait

The graph model's trait (`expand` / `gate` / `combine`) survives, with five refinements from this
discussion:

1. **Declarative edges, derived expand.** `expand` bundles three jobs — coordinate cutting,
   pricing, obligation emission — and most nodes have an opinion about none of them. The base
   trait becomes `edges()` (child, `DomainMap`, coverage table) plus `combine`; a generic
   implementation derives `expand` from `edges()`. Hand-written `expand` remains as an override
   for the nodes that genuinely price or prune (Zoned, Dict, List). This makes ease-of-use
   measurable in tiers: tier 0 writes nothing (generic coverage), tier 1 writes `combine`,
   tier 2 overrides the cut, tier 3 adds a gate. A new encoding landing at tier 2 by default
   indicates a framework gap, not an author problem.
2. **`gate` is `expand` re-entered with a fact.** Give expand a `facts` input; it emits
   `Needs(gate)` and returns when a fact is absent, and the driver re-calls it at fact-seal.
   Obligations are keyed, so re-emission dedups through the cells. Dict becomes one linear
   function instead of two halves sharing a gate id.
3. **Uniform node output.** A node's output is a `Value` in one lattice (`Array | Bound | Map`),
   so dict's gather map is an ordinary edge rather than a fact side-channel in `ChildResults`,
   and realize nodes stop being a special kind (their value happens to be a map). This removes
   the "dict remap wart" accepted as decision 13.
4. **Combine must stay O(parts); per-row combines must be priced.** The graph model's sketches
   broke their own rule: `DictExec::combine` calls a take kernel and the conjunct's combine ANDs
   masks — per-row work running inline, unpriced, on whichever thread seals the last input
   (possibly an IO completion thread). Either combine only wraps (per-row work becomes emitted
   kernels — `InputRef::Node` already expresses this), or combine declares a cost and the driver
   inlines below a floor and pools above it. Lean: admit the declared-cost form up front so the
   granularity floor stays one uniform mechanism. The invariant: **a combine that touches rows
   must be priceable.**
5. **Combine-once stands; `absorb` is the one escape hatch.** Combine runs once per (node, span)
   on Final inputs. The question "can a node emit a partial value?" resolves as: combine-once
   forbids a partial value of a fixed span but permits an arbitrarily fine complete value of a
   smaller span, and the boundary is chosen in `expand` (statically) or by the driver's
   rebatcher (opportunistically — merging adjacent complete parts is generic, not node
   semantics). What is genuinely lost is arrival-time folding, and two of its wins are real:
   peak memory (combine-at-n holds n−1 parts hostage to a straggler; a fold retains one
   accumulator) and cache warmth (folding a part on the thread that sealed it touches bytes
   still in L2). The opt-in refinement:

   ```rust
   // Accumulator is a Value in the existing lattice — no per-node scratch type.
   fn absorb(&self, span: SpanRef, acc: Value, part: PartRef) -> Value;
   ```

   restricted to order-free (commutative, associative) folds. The default buffer-then-combine
   path is a free differential oracle: an order-dependent absorb diverges from it and fails the
   harness, so decision 14's exclusion becomes a test rather than a rule to trust. Absorb stays
   the refinement, not the base: with combine as the base, span scratch is homogeneous and the
   driver owns the countdown; absorb-only would make every author define scratch and re-implement
   arity (dynamic post-gate) by hand.

The struct zip clarification belongs here: combine is **not** a scheduled task with n dependency
edges — that is the reactor shape and the measured 2.3x. It is a continuation run inline at
countdown-zero on the adopting thread, which is exactly why refinement 4's cost discipline
matters.

## 2. Demand routing is bind-time, not runtime

In the graph model, demand "propagates": each Plan node reads its incoming bound, cuts, and
forwards derived demand — a hop per node. But each hop only applies the edge's `DomainMap` and a
coverage cut; the maps are statically known (except gated ones) and map composition is
associative. So the binder precomposes: for each demand **producer** (a conjunct, a limit, a
pruner) and each ultimate **consumer** (a scan leaf, an IO source), compute the composed map
`producer-domain -> consumer-domain` once and wire the producer's cell directly to the consumer.

- **Operators never see, forward, or handle demand.** They *subscribe*: a consumer that can
  exploit demand reads its wired cell through the precomposed map at batch boundaries; one that
  cannot ignores it and is merely eager — correct but unoptimized. Demand-correctness
  concentrates in producers (few, core-owned) and one binder pass (one algorithm, one property:
  composed map ≡ hop-by-hop composition).
- **Three things stay runtime, all localized:** gated maps snap their realized link into the
  routing table at fact-seal (the composition on both sides of the hole is still static); demand
  meets are the cell's meet, not propagation; data-dependent producers (pruning from decoded
  stats) run at runtime but still just write their cell.
- **Invariant:** demand routing is static wiring plus gate-snapped links; demand *content* is
  runtime; operators are subscribers, never forwarders.

Consequences to design in: the scheduler prices per-consumer counts from the routing table
(producer cell × composed map), and superset-adoption must hold across *composed* maps — a new
`DomainMap` suite property (composition preserves the superset law), provable once centrally.

## 3. The commutation law

Deferral needs a legality argument; this discussion found it as one law. For a kernel `f` and a
demand set `R` no larger than the open bound:

```text
f(sel_R(x)) = sel_M(R)(f(x))          M = the edge's DomainMap
```

Selection commutes with `f`, transported through the map. Read right-to-left it licenses
**speculation**: run `f` early on the open (superset) demand, correct afterwards with the closed
selection. Requirements, both testable:

1. **Row-local:** output row j depends only on input row j. Excludes aggregates, folds, limits —
   as intended.
2. **Total on the superset:** speculative execution touches rows the closed demand would have
   excluded, and those rows can be poison (division by zero, invalid bytes in a dead row). So
   kernels must not trap: errors are values, masked out if the row does not survive. A kernel
   that cannot be total is ineligible for speculation — a per-kernel flag, priced like
   everything else.

Corollaries: the conjunct slot-meet-at-publish (graph model §7) is this law with `f = eval` and
`sel = mask-AND`; the eager oracle is the law applied maximally (open = top everywhere, one
selection at the end), which is *why* it is a valid oracle; the three laziness refinements of
graph model §4 are all instances. The mask density switch is the law's economics: the correction
is free when the target is positional (nothing to do) and a real gather when it is compacted.

## 4. One value contract; gather is the only cardinality change

A long detour through "need versus definedness" (does a consumer want n→n with undefined rows,
or n→m with rows removed?) collapsed to a deletion:

> **Every value is positional over its domain (n→n; rows outside the need set are undefined and
> may hold anything). There is exactly one cardinality-changing primitive — gather-by-map — and
> it is an explicit node the planner places, never a mode an operator or kernel selects at
> runtime.**

"Compacted data" is not a second value shape; it is positional data *in a smaller domain*,
reached by crossing a domain-change edge whose map is the sealed mask — the same machinery as
dict values and list elements (graph model §3's "gated realize (gather set)" row already
contained it: the survivor domain is a gated child domain, realized when the mask seals, with
demand transported through it by unmap).

This kills the two-code-path objection: kernels have one contract, `gather` is one ordinary
priced node, and nothing branches on a mode at execution time. The audit across scan execution:

- Filter spine: eval kernels n→n in the root domain; masks meet in the cell; **zero** gathers.
- Projection/emit: one gather per column at the survivor-domain crossing; `select *` with no
  filter has an identity map and no gather node at all.
- Expensive predicate over sparse demand (cascade compaction): the one case that looks like a
  runtime mode choice. The actor is a **Plan node**: its expand reads the sealed prior bound's
  density — a fact — and splices either `eval(positional)` or `gather -> eval -> unmap`. A
  planning decision made late with runtime facts, through the mechanism that exists for exactly
  that; the kernel never knows.
- Indexes/zone maps produce bounds, not values; a probe that naturally returns survivors is a
  node whose output edge lives in the survivor domain, declared in the plan.
- Joins and aggregations sit above the scan and receive the emitted domain; a join probe's
  output is also gather-by-map into a new domain, weak evidence the primitive is right.

**At the leaf** the contract holds because the leaf has a second, cheaper cardinality mechanism
that is not gather: **cutting the domain**. Expand cuts demand against the segment/page table;
only overlapping extents get Read obligations; each read+decode produces a plain positional
array over that extent's (shifted) domain. A sparsely-demanded column is several small positional
pieces over domains that exist, while un-demanded extents never exist — no holey arrays.
Undefinedness at a leaf only ever means dead rows *inside* a block that was read (block-oriented
decode reads whole blocks; the rounding slack is precisely what totality tolerates). Row-granular
skipping inside a block, where priced, is the explicit `decode -> gather(demanded)` kernel pair —
the per-part compaction node of graph model §9.

So demand's entire runtime meaning is: **sections whose rows are all undefined are skipped —
no IO, no decode, no kernels — and where shape requires a value to exist (a zip needs all
fields), a canonical placeholder of the right length and dtype stands in, never read.** The
placeholder should be canonical (a designated constant/null array) so the oracle can hash
need-set rows cleanly and a wrongly-read placeholder fails loudly; a debug mode that poisons
placeholders catches that class. Two skip granularities, then: domain cutting (free, structural,
at expand) and gather (paid, a node); IO skipping is always the first kind, which is why demand
never reaches a read as a filter — by the time a read exists, its extent *is* the demand rounded
to block boundaries.

### Where eliding the gather wins big

Three cases where running positionally over dead rows beats compacting, with the first unbounded:

1. **Wide values.** `take` costs O(bytes moved); the saving is O(rows skipped). At 95%
   survivorship on a 200-byte string column, compaction copies ~190 bytes/row to avoid 5% of a
   downstream pass; across a k-conjunct chain, compact-per-stage copies the column k times while
   positional copies it zero times before emit (which compacts once anyway). String-heavy scans
   are bandwidth-limited, so this factor multiplies the whole scan.
2. **Structure destruction.** Positional values can stay encoded — runs, codes, packing intact —
   and run-aware or encoded-domain kernels execute at O(runs) over all rows including dead ones.
   Compaction at high survivorship shatters runs and forces decode-to-flat: 5% dead-row overhead
   avoided, 10–100x structure advantage lost.
3. **Sharing.** A decoded block with fan-out f is one buffer plus f selections positionally;
   compact-per-consumer forks f near-full copies and silently defeats the keyed-cell dedup.

All three invert under sparse demand, which is exactly when the planner splices the gather. The
default plan therefore has gathers **only at emit**, and the density-directed exception is a
Plan-node decision (§4 above), denominated in bytes moved and structure preserved, not row
counts.

## 5. Evidence: FlatReader v1 already runs this model in miniature

`vortex-layout/src/layouts/flat/reader.rs` contains the design ad hoc:

- `filter_evaluation` takes a positional mask (the need set, in domain coordinates) and has the
  exact density switch (`EXPR_EVAL_THRESHOLD = 0.2`): the dense arm evaluates over **all rows**
  then `bitand`s — the law's cheap half in production; the sparse arm does
  `filter -> eval -> intersect_by_rank` — gather, narrow eval, and rank-transport back.
  `intersect_by_rank` **is** demand unmap through the gather's domain change: rank is the gather
  map read backwards, bridging survivor coordinates to domain coordinates. (Every early
  compaction buys eval savings at the price of this transport — which belongs in the same
  inequality that decides whether to compact.)
- `projection_evaluation` is decode, one explicit `filter`, then the expression: the survivor
  crossing as an explicit operation at the leaf.
- The `TODO` on the threshold ("should probably be dynamic... perhaps expressions decide for
  themselves") is answered: the decision is planning's, priced from the kernel table, not the
  expression's and not a constant.

Three deltas separate v1 flat from the model, and only one is semantic:

1. Demand is a one-shot sealed `MaskFuture`, not a refinable cell — v1 flat only ever sees closed
   demand (the degenerate case; no speculation, no refinement after issue).
2. The projection filter is **mandatory, and the code says why**: *"we must filter first before
   applying the expression, as the expression may depend on the filtered rows being removed e.g.
   `CAST(a, u8) WHERE a < 256`"*. That is a direct counterexample to kernel totality: v1
   projection expressions may trap on dead rows, and correctness leans on compaction-first. The
   elective-gather optimization is unsound against today's expression semantics until compute
   has errors-as-values, or a `can_trap` classification gates elision to total expressions. This
   is the single concrete work item the whole definedness discussion reduces to.
3. Whole-segment decode always: consistent with the leaf story (extent cutting is the chunked
   layer's job), but the sparse arm's win today is eval-only, not IO or decode.

## 6. The three-part architecture

The discussion's target shape, stated as three parts:

1. **A stateful exec plan per layout.** Operator instances in the DuckDB/Velox style —
   thread-pinned, batch-pushing, holding buffers, accumulators, and scratch; split per unit or
   thread for parallelism. Statefulness is *legal because of part 2*: demand is not the
   operators' job, so instance state cannot corrupt skipping. An operator that exploits demand
   subscribes to its wired cell; one that ignores it is merely eager, never wrong — a kinder
   third-party failure mode than v2-style propagate-the-mask-or-break. (The graph model's fused
   private chains running on a pinned thread with `ThreadCtx` scratch were already this,
   anonymous; this names it as the author-visible thing.)
2. **A demand mask system.** Pushdown composed at planning into the producer-to-consumer routing
   table (§2); masks as the runtime content; a row-domain transform node exactly where a
   non-identity crossing occurs (chunk shifts static and inline; gather/list/dict gated and
   snapped at fact-seal). One demand lattice — need; no second demand type at execution.
3. **A scheduler that sees IO and CPU/state nodes and runs them optimistically.** The commutation
   law (§3) is its license: IO is always speculable (a superset read adopts by intersection);
   CPU is speculable when the kernel is total and row-local. Cascade, eager, adaptive, and
   prefetch are one policy dial — where on the open-to-closed spectrum each work item runs —
   priced by EV against the IO watermark and byte credits.

Two constraints make the composition sound:

- **`filter` (the gather node) is the single synchronization point: it requires sealed demand.**
  Everything else prefers sealed but may run speculatively on open demand. The seal is
  per-fragment/span, so this is a local wavefront, not a phase barrier.
- **Speculation stops at non-row-local state.** A stateful operator may accumulate speculative
  positional parts (accumulation of independent parts is order-free); any order-dependent fold
  consumes only post-filter, sealed-demand data. This is the absorb boundary of §1 relocated to
  the architecture level.

The eager oracle survives intact: demand = top, placeholders nowhere, filters run with all-true
selections — every part degenerates to plain eager execution, and every configuration of parts
must hash-match it on need-set rows.

### Demand subscribers are few and stratified

A follow-up observation pins down who actually consumes demand, shrinking open question 5:

- **Must consume (sealed):** gather/filter nodes — the sealed mask *is* their map; they cannot
  run without it.
- **Should consume (where skipping pays):** data-loading leaves — extent cutting is the only
  place demand converts to absent IO, and it is one cell read per expand.
- **May ignore (and it is not the operator's choice):** predicate and projection inputs. Running
  a conjunct's field IO+CPU in parallel with its siblings is the *scheduler declining to wait*
  for the prior bound, not an operator ignoring demand; loading a projection column before the
  mask seals is the scheduler running a leaf on the open bound (legal by superset adoption,
  priced against byte credits). Cascade versus eager-parallel and prefetch versus demand-wait
  are admission-timing policies on the same graph — the operator code is identical.

Consequence for counting: demanded-row counts are an **upper bound** that speculative admission
deliberately overshoots; the overshoot must be charged to the speculation budget, never counted
as free (lands in the admission machinery, next-discussion problem 4).

### The filter/project split-granularity mismatch

A problem the current implementations cannot express: today one split set serves the whole scan,
formed as the union of natural boundaries across *all* referenced columns
(`register_splits` -> `RowSplits`). A coarse-chunked filter column is therefore artificially cut
to the fine boundaries of the projected columns, and filter-phase work runs at projection
granularity — per-split fixed machinery multiplied by a count the filter never asked for. This
is distinct from the `select *` small-splits storm (next-discussion problem 1): that is "splits
too small absolutely"; this is "splits too small *for one phase* because another phase's
geometry leaked into the shared split set."

The graph model dissolves it in principle: span formation is per node — each expand cuts against
its own coverage, so the eval spine spans the filter column's chunks while the projection Plan
node spans the union of projected boundaries only (exactly as the worked example draws it), and
driver-side slicing absorbs the misalignment at combine. But this holds only if
**fragment/ledger granularity is not derived from the all-columns boundary union** — which makes
it a constraint on the unit-formation design (problem 1), not a free consequence.

## 6a. Two planes: in-band masks, out-of-band demand

A late refinement reconciles the engines-style streaming picture (DuckDB pipelines with
selection vectors in the chunk; Velox drivers with FilterProject and LazyVector-driven late
materialization) with the demand system:

- **Data plane (in-band, exact, authoritative).** Batches and conjunct masks stream through the
  operator chain; the AND and the gather happen where the masks arrive. The gather's requirement
  is that *its in-band mask input is final* — an ordinary dataflow dependency, no longer a
  demand-system event.
- **Control plane (out-of-band, advisory, never blocking).** The demand cells and bind-time
  routing survive as a side channel with weaker semantics: monotone shrinking supersets, read at
  admission points (leaf extent cuts, scheduler pricing), never waited on. The commutation law
  makes any admission-time snapshot sound; the in-band plane corrects everything at the gather.
  This is sideways information passing (DuckDB dynamic join filters, Velox dynamicFilters) made
  the primitive rather than a bolt-on — and because the plane only promises supersets, its
  content generalizes beyond exact masks to any conservative summary: range bounds, zone
  verdicts, bloom filters, IN-lists, a limit counter.

Optimistic conjunct IO and optimistic projection IO are then the same move: admit reads against
whatever the OOB cell currently holds (top if nothing landed). Cascade versus parallel is "how
stale was the snapshot at admission" — a continuum, not two modes. A lost or late OOB update can
only cost performance, never correctness; the differential harness should therefore run with the
OOB plane disabled and maximally delayed and require identical results.

## 6b. Morsel-driven build sketch

The concrete instantiation of the three parts, as currently intended:

- **Pipelines.** One pipeline per conjunct and a small number per projection, handed to the
  scheduler. Struct is *not* a pipeline breaker: fields run as sequential stages of one pipeline
  instance, with elective field-parallel fan-out only when a field's work exceeds the
  granularity floor. The only barrier-like point is the per-range mask meet, a countdown. IO
  tasks precede CPU pipelines; each fixed-size filter or projection range carries its demand
  mask input.
- **Morsel planning.** Each claimed morsel is planned **once**, against an OOB demand snapshot:
  empty snapshot skips the subtree; planning emits IO tasks, CPU pipeline activations, and
  further planning tasks. Demand is re-read at IO admission, so the shrink between plan time and
  issue is captured without re-planning (superset-sound; planned tasks whose demand sealed empty
  evaporate as candidates). The one exception to one-off planning: gated subtrees (dict values,
  list elements, zoned verdicts) re-plan **on facts** — that is the "more planning" arm, not
  re-planning on refinement.
- **Pruning is warm-up work, not a phase.** Zone-map metadata is file-scoped (a few small
  segments covering all zones), so the first morsels issue the pruning-metadata reads *and*
  their first conjunct's IO optimistically in parallel — the early wave simply does not benefit
  from pruning (the accepted speculation cost, overlapped with IO latency). When the stats fact
  seals, verdicts for every zone in the file are computed in one cheap bulk pass and written to
  the OOB cells; every subsequent morsel's planning snapshot already contains the prune, paying
  zero pruning IO and zero pruning compute, with some subtrees never planned at all. The more
  morsels a file has, the closer pruning is to free.
- **One-off planning, restated.** Planning runs once per morsel against the snapshot; a shrink
  after t0 is captured by one late look at the demand cell just before each read issues (drop if
  dead, shrink if smaller — safe because a read against stale larger demand is merely a superset).
  There is no re-planning on refinement. Dict/list are not an exception: t0 planning emits a
  deferred "plan this bit when fact X seals" note — one-off planning, part of which cannot start
  at t0.
- **Per-morsel stash.** Each morsel owns a scratch store keyed by (plan edge, range): decoded
  arrays shared between filter and projection, partially computed arrays, masks awaiting the
  meet. Lifetime is the morsel's; the whole stash drops at retire; the morsel's live bytes *are*
  the stash plus in-flight IO, making the deferred memory approximation exact. Cross-morsel
  sharing shrinks to a thin explicit list of scan-wide keyed cells (dictionary values, file
  stats, the pruning fact). Boundary case, recorded: once morsel boundaries stop being the
  all-columns split union, a projected column's chunk can straddle two morsels — default is to
  decode it twice (the bounded-duplicate principle); promote straddlers to scan-wide cells only
  if measurement says so.
- **Morsels are typed by row domain.** A domain change does not add bookkeeping inside the outer
  morsel; it spawns **child-domain morsels** at gate seal — own demand (the gather set, sealed
  at birth), own pipelines (the values/elements subplan), own stash — whose results land in the
  parent morsel's stash for the parent's combine. A heavy list-elements subtree becomes several
  child morsels: parallelism inside one outer row range. **Inner-domain morsels take priority
  over claiming new outer splits** — depth-first in work-stealing form (run own newest, steal
  oldest/outermost). This is the memory bound as much as a latency rule: finish-what's-started
  keeps work-in-progress near workers × domain-depth, and "all rows claimed" generalizes to
  "all rows claimed in every domain."
- **Parallelism.** Morsels are the parallel unit; per-morsel (not per-thread) operator state;
  work stealing when no new morsel can be claimed, with wakes preferring the owning worker's
  deque so continuations run warm. No new morsels when: (1) all rows claimed; (2) the memory
  limit binds (deferred — see below); (3) a limit has sealed the remaining tail.
- **IO coalescing.** Morsel planning emits the morsel's reads as one batch and the morsel's
  segments are file-adjacent, so batch-scope coalescing is expected to suffice; dedup by
  `SegmentId` comes separately from the keyed cells. Verified, not assumed: the stress matrix
  carries a cold-scan IO parity gate (same bytes, comparable request count versus V1), and only
  its failure justifies more machinery.
- **Emission and limits.** Pull-driven: the consumer stopping stops morsel claiming; in-flight
  morsels finish or park. Limit is a first-k demand producer at the sink writing into
  projection's OOB cells (per-morsel for ordered prefix consumption; a shared global survivor
  counter for unordered), transitively bounding filter work — the one legitimately cross-morsel
  cell.
- **Deferred, recorded:** the memory model — per-morsel live-byte approximation, attribution of
  shared cells (first-needer versus split versus shared pool), and the memory-times-ordering
  deadlock (complete-but-unemittable morsels holding bytes the oldest morsel needs; candidate
  fix: the oldest unemitted morsel is always admissible).

### Stress matrix versus V1

Correctness (differential, row-hash on the need set): every layout × {no filter, selective,
non-selective, all-false} × {aligned, unaligned chunks} × {nulls, trapping expressions} ×
range boundaries straddling chunk and page edges; OOB disabled and maximally delayed; absorb
versus its blanket impl. Scheduler invariants in a deterministic simulator: no deadlock under
memory × ordering × limit × cancellation; adversarial IO completion orders; steal-versus-wake
races. Performance gates: Q01/Q06 (the prefetch split), FineWeb `select *` (small-splits storm),
selective string predicates (cascade and wide-value elision), dict page skipping, cold-scan IO
parity, contention counters (entries-considered-per-admission ≈ 1, queue-idle ≈ 0). Layout
coverage audit: each V1 layout (flat, chunked, struct, dict, list, zoned, row_idx, partitioned,
table, compressed, buffered, repartition, foreign, file_stats) needs its one-line story or a V1
fallback adapter mid-tree during migration; plus the degenerate paths — `select *` reduced
machinery, tiny single-morsel scans without scheduler spin-up, repeated-scan fact reuse.

## 7. Decisions recorded from this discussion

1. Base trait is `edges()` + `combine` with expand derived generically; hand-written expand is an
   override tier; `gate` merges into re-entrant expand with facts.
2. Node outputs are uniformly `Value` (`Array | Bound | Map`); no fact side-channel in
   `ChildResults` (supersedes the graph model's decision 13 wart).
3. A combine that touches rows must be priceable: wrap-only combines run inline; per-row
   combines are declared-cost and floor-governed (or reified as emitted kernels).
4. `absorb(span, acc: Value, part) -> Value` is an opt-in refinement for order-free folds;
   buffer-then-combine is its blanket impl and free differential oracle; motivated by peak
   memory under stragglers and producer-thread cache warmth.
5. Demand routing is composed at bind into a producer-to-consumer table; operators subscribe and
   never forward; gated maps snap links at fact-seal; meets are cell meets.
6. The commutation law `f(sel_R(x)) = sel_M(R)(f(x))` for row-local, total-on-superset kernels,
   transported through edge maps, is the single legality argument for all speculation and
   laziness; totality (errors as values, never trap) is a kernel-eligibility flag.
7. One value contract: positional over the value's domain; gather-by-map is the only
   cardinality-changing primitive, always an explicit planned node; "compacted" is positional in
   a survivor (gated child) domain.
8. Two skip granularities: domain cutting in expand (free; all IO-level skipping) and gather (a
   priced node; default placement emit-only; early placement is a Plan-node expansion decision on
   density facts).
9. All-undef sections are skipped structurally; shape-required values are canonical placeholders
   (poisoned in debug), never read.
10. The elision economics are denominated in bytes moved and structure preserved (wide values,
    encoded-domain execution, sharing), not row counts.
11. `intersect_by_rank` is recognized as demand unmap through a gather's domain change; its cost
    belongs in the compaction-placement inequality.
12. Target architecture is the three parts of §6 with the two constraints: sealed-demand filter
    as the only sync point, and speculation stopping at non-row-local state.

## 8. Open questions

Carried forward or newly raised:

1. **Kernel totality audit.** How much of the current expression/compute layer is already total
   (`can_trap = false`)? The `CAST(a, u8) WHERE a < 256` class needs either errors-as-values or
   a conservative trap classification before elective gathers are sound.
2. **Placeholder representation.** Canonical constant/null array per dtype; how does the oracle
   hash need-set rows only, and what does the debug poison look like?
3. **Composed-map superset law.** Property suite addition: composition preserves
   superset-adoption; gated snap preserves it across the healed link.
4. **Pricing from the routing table.** Per-consumer demanded counts derived as
   producer-cell × composed-map — does this reproduce the per-edge counts EV admission assumed?
5. **Operator subscription API.** Narrowed by the subscriber stratification (§6): only leaves
   (extent cut) and gathers (map) need it; remaining question is what a leaf sees at expand
   (cell version, mapped bound, density) and what it may cache between batches.
5a. **Fragment formation must not use the all-columns boundary union** (the filter/project
   split-granularity mismatch, §6) — a constraint to carry into next-discussion problem 1's
   unit-formation algorithm, alongside charging speculative overshoot to the speculation
   budget in problem 4's counting.
6. **Where the density fact for late gather placement lives** — shared with problem 4's
   `remaining_selectivity` estimation machinery rather than new.
7. **Velox/DuckDB-style instance splitting** (part 1): instance per unit, per thread, or per
   pipeline — and how instance state interacts with unit coalescing (next-discussion problem 1).
8. The graph model's open questions 1–13 stand where not superseded (its 3 — the closed
   `Obligation` enum — is narrowed by decisions 1–2 here; its 7 — `ChildResults` shape — is
   settled by decision 2).
