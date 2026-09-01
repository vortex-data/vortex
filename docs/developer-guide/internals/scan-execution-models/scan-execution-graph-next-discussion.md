# Scan Execution Graph: Next Discussion

This document is the starting point for the next design conversation, the way the
[framework](scan-execution-framework.md) and experiment documents seeded the discussion recorded
in the [graph model](scan-execution-graph-model.md). Each section states one unresolved problem,
the context and evidence a fresh reader (or a fresh session) needs, what the last discussion
already concluded around it, and the concrete output the next conversation should produce.

A follow-up discussion recorded in
[demand, operators, and the filter law](scan-execution-demand-and-operators.md) has since
refined the computational framework — bind-time demand routing, the speculation commutation law,
the single positional value contract with gather as the only cardinality change, and the
three-part target architecture — which reshapes problems 3 and 4 below and partially settles the
graph model's open questions 3 and 7.

Ground rules carried over: decisions already recorded in
[graph model section 10](scan-execution-graph-model.md#10-decisions-recorded-from-this-discussion)
are settled unless new evidence reopens them; every performance claim must trace to the
[findings](self-paced-plan-exec-findings.md) or to a new measurement; the eager configuration is
the permanent oracle, so any proposal must describe its eager degenerate form first.

## Problem 1: unit formation and the `select *` small-splits storm

**The problem.** Layout-derived natural splits can be small (FineWeb: 1,823 natural splits for
14.9M rows). Today `SplitBy::Layout` in `vortex-scan-v2/src/splits.rs` only *subdivides* large
spans (`IDEAL_SPLIT_SIZE = 100_000`) and never merges small ones, and each split becomes an
independent task carrying full per-split machinery. Under `select *` the filter phase per split
is trivial, so fixed costs dominate.

**What was concluded.** In the graph model, unit size stops being the parallelism grain: filter
parallelism comes from data (fork-join sub-ranges), projection parallelism from survivors
(pooled sealed spans). Proposed shape: fragment = natural split (keeps prefix progress and cache
release layout-aligned); unit = byte/work-budgeted coalescing of consecutive fragments. For
`select *`: symbolic all-true demand (never materialized), no filter stage, spans seal at expand
and go straight to the pool — one unit could own a whole file and still saturate the cores.
Evidence: merge-16 morsels (learning 5–7, "a fixed split-count rollup is only a starting point");
select-all needs a reduced-machinery mode (learning 39).

**Next conversation should produce.** The unit-formation algorithm (byte target, work estimate
inputs, behavior when `estimated_bytes` is absent) and the per-scan composition-selection rule:
which plan properties (filter presence, expected selectivity, projected byte width) choose
between inline-cascade, span-pool decoupled, and fork-join, and where that decision lives.

## Problem 2: the drive loop's concrete shape

**The problem.** The graph model defines the driver abstractly (private PC, span countdowns,
three scheduler-visible summaries) but not its concrete data structures: the span scratch layout,
the countdown representation, how parked obligations are stored, and how a wake re-enters ADOPT
without rescanning.

**What was concluded.** Obligations are not tasks; only `Placement::Pool` mints pool items; wakes
re-enqueue the unit item directly (fact -> waiter -> pool). `combine` fires at countdown-zero on
the adopting thread. The unit's retained memory is one span working set (law 8).

**Next conversation should produce.** The `UnitState` struct sketch: frontier storage (per-unit
frontier-head register for the scheduler, law 7), span scratch entries (recorded cuts, adopted
values, countdown), the parked-obligation representation, and the wake path from a cell's waiter
drain to re-entering the drive loop mid-span. Also: what `ThreadCtx` holds (decode scratch,
per-thread caches) versus what moved into cells.

## Problem 3: binder mechanics and grafting onto `PlanVTable`

**The problem.** The binder creates domains, slots, shared-edge cells, tokens, unit boundaries,
and exec nodes — but the current plan layer (`vortex-layout/src/plan/vtable.rs`) exposes only
`execute` returning futures. How do the new hooks graft on so nodes migrate one at a time?

**What was concluded.** Add `edges()`/`bind()` alongside `execute` with unimplemented defaults;
keep the v2 future path alive as the oracle during the build (open question 11). Fan-out is
knowable at bind (which nodes touch which segments), so cell materialization is decided before
execution. Exec nodes are one per plan node per scan, immutable, shared.

**Next conversation should produce.** The `Binder` API sketch and the exec-node registry
(plan-node id to exec-node constructor), the bind output object (`ledger layout, writer tokens,
unit descriptions, gate placeholders` from the framework's bind contract, now concretized), and
the migration order for existing plans (SegmentScan, Concat, Eval, Pack first; Zoned; then Take
and ListPack as the gated pair).

## Problem 4: the scheduler's admission machinery

**The problem.** EV admission, the IO watermark, granularity floors, and Required-vs-Candidate
are defined as policy; the machinery (frontier-head registers, reserved Required share, promotion
on issue, the O(units) invariant) is not designed in detail.

**What was concluded.** Reported items rest in their unit's frontier; the scheduler sees one head
per unit; candidates evaporate at seal-empty having cost nothing; the two success counters are
entries-considered-per-admission (~1) and queue-idle time (~0). Speculation is the scheduler
running frontier Plan nodes on open bounds below the watermark — a possible speculation floor
(minimum bound density) is open question 6.

**Next conversation should produce.** The admission loop's data structures, how
`remaining_selectivity` is estimated and updated (the AdaptiveDemand survival-rate atomics are
the precedent), the reserved-share arithmetic for Required work, and whether the speculation
floor exists and at what threshold.

## Problem 5: emission, ordering, and limits

**The problem.** The graph is unordered; ordering lives in units owning contiguous coverage. But
cross-unit ordering restoration, the root rebatcher, limit pushdown, and cancellation were only
touched in passing (open question 12).

**What was concluded.** Spans emit in order within a unit; the experiment restored cross-morsel
order by index-sort at the end, which is fine for batch collection but not for a streaming
consumer with bounded memory. Limits are naturally a demand: a first-k bound at the sink that
refines as spans seal — but k flows *across* units, which is the one place demand is not
per-fragment-independent.

**Next conversation should produce.** The ordered-emission contract for streaming consumers
(credit-based? window of outstanding units?), how a global limit refines per-unit bounds without
a coordinator, and the cancellation story (consumer drops the stream: which cells seal-empty,
in what order, and what happens to in-flight IO).

## Problem 6: the memory story under real IO

**The problem.** The experiment ran in-memory; object-store latency changes the crossovers
(learning 54's caveat, the Q06/Q01 prefetch split). Speculative reads, per-unit read-ahead, and
shared cells retained across units all hold memory that law 8 does not yet bound globally.

**What was concluded.** Byte credits appeared in the layout27-derived designs and in the
framework's emitter credits (build-order step 4) but were not integrated into the graph model's
cell layer. Candidate reads have an EV byte charge; nothing yet caps total speculative bytes or
defines eviction for reusable cells under pressure.

**Next conversation should produce.** The budget model: per-scan or per-session byte accounting,
where credits are checked (admission only, or also cell insertion), and what "reusable" cells do
under pressure (drop and re-read is always safe — the bounded-duplicate principle from the
per-thread cache applies).

## Problem 7: conformance and the oracle stack as deliverables

**The problem.** The extension story rests on implementors running law suites instead of
internalizing eight laws (open question 10), but the suites do not exist as designs.

**What was concluded.** For `DomainMap`: property tests (map_demand/unmap_mask superset
round-trip; prefix-preserving implies monotone `map_range`; gated maps refuse transforms before
realization). For `ExecNode`: differential execution against the eager driver with the
ordered-output-hash gate; plus the transformation-level property that every lazy configuration
hash-matches the eager oracle per workload. The oracle stack (eager reference, ordered-hash
gates, cold-scan IO invariant, external engine oracle, contention counters) is non-negotiable
from day one.

**Next conversation should produce.** The concrete test-harness crate layout, the property list
per trait written as test names, and which invariants become debug assertions in the driver
(single-writer, superset-adoption, span-alignment) versus properties only the harness checks.

## Suggested order

Problems 2 and 3 unblock code (driver shape, binder graft) and should come first if the goal is
the eager end-to-end path; problem 1 is next since it delivers user-visible wins (`select *`)
with minimal machinery; 4–6 depend on measurements the eager path enables; 7 runs in parallel
with everything and gates all of it.
