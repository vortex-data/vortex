<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# One planning and morsel protocol for all scans

Design proposal, 2026-09-22. This is the stage and implementation plan for the
[ticket-based scan traits](TRAITS.md). It describes the intended system;
it does not claim that an existing executor implements it.

The goal is one scan implementation driven by a default scheduler or a caller's own
scheduler. Layout traversal, pruning, filtering, projection, and optional lookahead
all produce lazily discovered work. A scan never needs to build its complete plan
tree, enumerate every segment, or receive caller-specified row splits first.

## Pipeline and child counts

```text
optional external index / file discovery
    -> FooterOpen, one per candidate file
        -> FooterPrune
            -> IndexPrune, including zone maps
                -> PrepareEvalPlan(file plan, filter, projection)
                    -> WarmPlan                # optional IO announcement
                        -> FilterEval(filter part, projection continuation)
                            -> ProjectionSplit  # optional byte sizing
                                -> ProjectionEval(projection part, exact selection)
                                    -> Morsel -> array batches
```

Each arrow carries the source identity, relevant layout metadata, row domain, and
selection known at that point. The earlier selection is always preserved when a
later stage refines it. Metadata pruning is conservative; filter evaluation produces
the exact selection. Reading or warming a segment does not authorise output rows.

Counts below are logical children of **one instance**, emitted over successive
`compute()` calls, not a vector constructed up front. They describe the ordinary
path. An exposed speculative successor receives its matching `TicketComplete`
instead of a duplicate ordinary child.

| Component | Position and responsibility | Children produced |
| --- | --- | --- |
| FooterOpen | Open one candidate file and obtain its metadata | One FooterPrune on successful open; no per-column children |
| FooterPrune | Reject a whole file before reading lower-level indexes or data | Zero if rejected; otherwise one or more IndexPrune children for independent layout regions |
| IndexPrune | Refine a file region using zone maps or another available index | Zero to many surviving plan regions, lazily prepared into FilterEval children |
| PrepareEvalPlan / optional WarmPlan | Optimise a region's filter and projection plan; optionally announce all its segment dependencies | One FilterEval per prepared region, or ProjectionEval when there is no filter |
| FilterEval | Evaluate the predicate exactly, completing regions incrementally | Zero to many children from its continuation: direct projection or optional splitting; a simple case may return a morsel directly |
| ProjectionSplit | Size selected projection work by bytes when included in the composition | Zero if empty, one projection child if the view fits, otherwise several children over smaller selected views |
| ProjectionEval | Plan evaluation over an exact selection | Zero to many morsels, and intermediate planners when layout planning needs them |
| Morsel | Read, decode, evaluate, and return arrays | No planner children; zero to many array batches |

The builder omits unnecessary stages. Without an index, survivors go directly to
filter evaluation. Without a filter, the input selection is already exact and work
goes directly to projection. An already-open file supplies its existing footer;
an in-memory layout starts at the appropriate layout stage without file opening.
Cheap adjacent stages may be fused without changing these semantics.

Preparation and warming may be implemented as one adapter. The diagram separates
their responsibilities, not mandatory allocations. Warming walks the prepared plan;
it does not use `FilterEval::speculate()` to discover reads.

An external index is different from an index stored inside a file: it may run
**before FooterOpen** and produce zero to many candidate-file openers. A file-local
index normally needs the footer's layout and segment map, so runs after it.

## FooterOpen

**Construction:** source identity, available size or cached footer, incoming candidate
selection, and the scan's predicate/projection configuration. File identity must
distinguish incompatible versions for safe caching and deduplication.

**Work:** expose footer bootstrap IO, parse the result under `NeedsCompute`, and
repeat if the first read discovers another required range. Once known, the footer
provides the row count, dtype, segment map, and layout metadata. Retain metadata and
walk its children lazily; opening a file does not construct every scan node.

```text
FooterOpen(A)
    NeedsIO(tail bytes, or source length followed by tail bytes)
    NeedsCompute -> parse footer location
    NeedsIO(remaining footer bytes), if necessary
    NeedsCompute -> Planner(FooterPrune(A, footer, incoming selection))
    Done
```

This normally creates **one child**. Cached metadata skips the corresponding IO. An
empty file can be rejected by the next stage or short-circuited as a fused
optimisation. An open failure is an error, not evidence that the file has no rows.
Before the layout is known, do not invent speculative zone or filter children.

## FooterPrune

**Construction:** parsed footer, predicate, and incoming candidate selection.

**Work:** intersect the selection with the file domain and evaluate file-level
statistics. Unknown statistics retain candidates. If the file cannot match, finish
without reading zone maps or data. Otherwise lazily identify independent regions of
the index layout and create their pruning planners. Separate statistics segments,
if any, are explicit IO dependencies.

```text
FooterPrune(A, 4096 rows, predicate x > 50)
    footer says max(x) = 99: cannot reject A
    -> IndexPrune(A, layout region [0,2048))
    -> IndexPrune(A, layout region [2048,4096))

FooterPrune(B, 3072 rows, predicate x > 50)
    footer says max(x) = 40
    -> Done, no children
```

The surviving example creates **two children** because it has two independent index
layout regions. One map over the whole relevant file region needs only one pruning
planner. This count is not the number of projected columns, individual statistics
entries, or output batches. No index means construct the next filter or projection
work directly.

Once the footer is known, this stage can expose speculative index children before
finishing a substantial file-statistics computation. Each candidate has its own
ticket, later completed with the surviving domain or `None`.

## IndexPrune

**Construction:** one index layout/domain, predicate, and conservative input
selection. Zone-map pruning is an implementation of this stage.

**Work:** discover the required index segments, request them as a batch, evaluate
their pruning predicates, and construct work only for surviving regions. Index
layout boundaries determine natural result domains; a single whole-file map and
several separate map layouts need not produce the same planner structure.

```text
IndexPrune(A, [0,2048))
    NeedsIO(zone-map segments)
    NeedsCompute -> reject [1024,2048), retain [0,1024)
    -> FilterEval(A, [0,1024))

IndexPrune(A, [2048,4096))
    NeedsIO(zone-map segments)
    NeedsCompute -> retain [2048,4096)
    -> FilterEval(A, [2048,4096))
```

Each instance produces **zero to many filters**, as its surviving regions become
known. The constructor may group neighbouring survivors into one filter that itself
processes several regions. Missing or inconclusive index information retains rows;
it does not turn into an empty result.

For several indexed predicate columns, the default construction reads/evaluates
them together over a common row domain. A cascading construction can check a cheap
column before reading a more expensive one. Differing boundaries must be aligned
when combining selections. Two indexes covering the same rows must not independently
emit duplicate final filters; they refine one logical successor domain.

If a filter's layouts and candidate domain are already known, it may prepare while
the index is pending. The index then emits that filter's ticket completion instead
of constructing a second filter. Exact filter output still waits for the index
selection to become authoritative.

## FilterEval

**Construction:** the filter part of a prepared file-region plan, conservative
candidate selection, and a continuation retaining the projection part of that same
prepared plan. The builder supplies both parts after optimisation.

**Work:** request predicate inputs, decode and evaluate the predicate, and intersect
the result with the input selection. These are scheduled IO and CPU phases. Predicate
arrays may be produced by a private morsel, but are inputs to the filter, not scan
output. Conjunction can evaluate columns together or cascade through progressively
narrower selections; the constructor chooses the policy.

```text
FilterEval(A, [2048,4096))
    NeedsIO(predicate data for the next regions)
    compute region [2048,3072): 100 rows match
        -> ProjectionEval(A, exact selection of those 100 rows)
    compute region [3072,4096): no rows match
        -> no projection for that region
    Done
```

One filter produces **zero to many projection children**, one per nonempty finalised
output region in this construction. It does not wait for its whole input domain to
finish before releasing the first child. The output regions and their exact
selections retain file coordinates even when only a few rows survive.

If projection candidates were exposed early, the same example instead emits
`TicketComplete(T0, Some(exact_selection))` and later `TicketComplete(T1, None)`.
The parent does not care whether either candidate was prepared, left cold, or had its
preparation discarded. A ready ticket completion must not be held behind unrelated
predicate IO for a later region.

## ProjectionEval

**Construction:** the projection part of the prepared file-region plan and the exact
authorised selection, plus shared IO registrations or useful preparation. The filter
continuation retains this part while the filter computes its selection.

**Work:** walk the selected layout regions, resolve expression/decode requirements,
and construct array-producing morsels. Substantial planning for a flat array is an
explicit compute phase and may itself discover metadata IO.

```text
ProjectionEval(A, 100 selected rows, projection {p, q + 1})
    walk the next selected chunk/field region
    plan evaluation for its flat arrays
    -> Morsel(selected subset 0, prepared recipes)
    -> Morsel(selected subset 1, prepared recipes)
    Done
```

One projection can split into many morsels, driven concurrently on one worker or
in parallel by assigning their portable descriptions to workers before execution.

The live, non-`Send` objects in the current trait sketch cannot themselves be handed
to another worker; the description/start boundary must be made concrete for that
parallel path. Once started, each morsel stays with its owner, including during IO.

For `eval(chunked(flat))`, keep a private chunk cursor and plan each selected flat
when reached. For a struct, align fields that have different chunk boundaries before
assembling arrays. A morsel may still request data or discover a dictionary
dependency later. Construction does not mean all its bytes are already available.

## WarmPlan: prepare and warm the shared plan

**Input:** a file plan, the filter and projection expressions, a candidate row
domain, and construction policy. The adapter prepares and optimises this plan, then
walks all segment dependencies of the resulting plan. It announces reads across
both the filter and projection columns. The resulting plan is retained and its
relevant parts are passed to filter evaluation and the projection continuation.

Plan warming walks the optimised plan. It works with speculation disabled and
requires no projection tickets.
Optional speculative CPU preparation remains a separate use of the three traits.

```text
file plan + filter + projection, for one admitted region
    -> prepare and optimise
        -> all segment dependencies -> shared IO subsystem (MayNeed)
        -> filter part              -> FilterEval
        -> projection part          -> retained continuation
                                           |
                         exact filter selection arrives
                                           |
                                           v
                                      ProjectionEval
```

For `filter = x > 10` and `projection = {x, y + 1}`, suppose the prepared region
references x segments X0/X1 and y segment Y0. Warming announces `{X0, X1, Y0}`,
including validity, dictionary, or other dependencies present in the plan. The
filter gets its x plan; the projection gets its x/y plan. Shared dependencies use
the same source/segment identities, so warming and both consumers can reuse one
physical read. Sharing the plan and bytes does not automatically promise sharing
decoded arrays or expression results; that needs an explicit reusable computation.

"All columns and segments" means all dependencies in the optimised plan supplied
to this adapter. If field pruning removed an unused column, that column is no
longer part of this plan. If the supplied plan spans the whole file, a complete walk
can announce the whole file plan. To run only a little ahead, construction instead
admits bounded plan regions and warms all their dependencies. It must not pretend
that a full recursive file walk is bounded lookahead. Dependencies discovered only
after required metadata IO are added when that metadata becomes available.

The adapter can use the ordinary planner protocol. A proposed progression is:

```text
state()   -> NeedsCompute
compute() -> NeedsIO(all known prepared-plan requests, intent = MayNeed)
state()   -> NeedsCompute                    # handing off work needs no read result
compute() -> Planner(FilterEval(filter_part, projection_continuation))
```

Returning an IO batch from `compute()` publishes requests. The driver registers it
and rechecks `state()`; it must not unconditionally park on every published batch.
`State::NeedsIO` still represents work whose progress actually waits for inputs.
The concrete optional-request registration/ownership API is to be finalised, but
optional warming cannot become a barrier before handing the filter its work.

`MayNeed` and `Required` are proposed request metadata, not new planner traits.
The IO subsystem may prefetch, cache, coalesce, or decline optional reads. A filter
or projection needing bytes registers or promotes its own required interest and
progresses even if warming never started. Registration handles or cache leases
needed for reuse travel with the prepared work; retiring the warming adapter must
not cancel a child's required read. Rejected rows may leave warmed bytes unused.
Unused warming failures do not fail the scan.

The adapter emits **one filter per admitted prepared region**, or a projection when
there is no predicate. There are no additional speculative child objects just to
warm bytes. Its private region cursor and admission limits bound prepared plans,
pending requests, and retained buffers. Disabling warming skips read announcement
while using the same preparation and evaluation path.

All stages participate in the same source-wide IO pool, including index and morsel
reads. Register each whole discovered batch before its members become eligible;
requests discovered after dispatch join a later read. Source policy decides how
required and optional requests coalesce and must preserve required progress.

## Composing planners with Next

The composition API is a typed constructor for the next planner. This is a proposed
supporting type; the execution traits remain `Planner`, `Morsel`, and
`SpeculativePlanner`.

```rust
type Next<Input> = Arc<
    dyn Fn(Input) -> VortexResult<Box<dyn Planner>>
        + Send
        + Sync
>;
```

`Next<Input>` is reusable construction policy. It receives a stage's result and
constructs one planner to handle it. The returned live planner need not be `Send`
or `Sync`. The callback captures shared descriptions and configuration; mutable
execution state belongs to each returned planner. Constructors stay cheap: further
optimisation, splitting, IO discovery, and evaluation are scheduled through the
returned planner's `state()` and `compute()`.

For evaluation stages, the input can be a view over a shared plan:

```rust
struct PlanView {
    plan: PlanRef,
    rows: RowSelection, // Original row domain and current selection.
}
```

This is a conceptual shape; source context must remain available through the plan
or associated construction context. A filter view and projection view may share
nodes. A smaller view retains the relevant plan references and restricts its row
domain and selection, without rebuilding the whole plan.

### Filter directly into projection

```rust
let project: Next<PlanView> = Arc::new(|view| {
    Ok(Box::new(ProjectionEval::new(view)))
});

let filter = FilterEval::new(
    predicate_view,
    projection_view,
    project,
);
```

`FilterEval` evaluates the predicate view and retains the projection view. Whenever
it finalises a nonempty selection, it restricts the corresponding projection view
to those rows and calls `project(selected_view)`. That produces the next planner;
the callback does not execute projection. Empty selections produce no child.

The projection view is the remaining computation with the exact selection applied;
it must not cause the filter to be evaluated again.

### Insert byte-based projection splitting

This alternative changes the composition supplied to the same filter constructor:

```rust
let project: Next<PlanView> = Arc::new(|view| {
    Ok(Box::new(ProjectionEval::new(view)))
});

let split_then_project: Next<PlanView> = Arc::new(move |view| {
    Ok(Box::new(ProjectionSplit::new(
        view,
        target_bytes,
        project.clone(),
    )))
});

let filter = FilterEval::new(
    predicate_view,
    projection_view,
    split_then_project,
);
```

The constructor shapes are:

```text
FilterEval::new(predicate_view, projection_view, next: Next<PlanView>)
ProjectionSplit::new(selected_view, target_bytes, next: Next<PlanView>)
ProjectionEval::new(selected_view)
```

The filter does not know whether its successor evaluates directly or splits first.
The splitter retains its input view and performs sizing in its own `compute()`:

```text
view fits:
    next(original_view) -> one ProjectionEval child

view is too large:
    next(smaller_view_0) -> first ProjectionEval child
    on a later compute():
        next(smaller_view_1) -> second ProjectionEval child
    ...
```

Each piece preserves the exact selection. Pieces cover the original selected rows
without overlap or omission. The splitter constructs them lazily rather than
returning a precomputed vector. The initial sizing proposal is estimated decoded
or output bytes across the projected columns, so large string/binary values reduce
rows per morsel. A single row larger than the target makes that target soft.
Splitting execution does not necessarily split physical segment reads; several
children may share one read.

ProjectionSplit is a candidate for the default composition. Alongside the stages
already described, optional external-index/partition pruning is the other proposed
addition. Dictionary handling, struct assembly, and similar operations need not
become separate default planner stages merely because an implementation uses them.

### Implementing another composable planner

A stage stores its own input/cursor and a `Next<Output>`. When one output is ready,
its ordinary handoff is:

```rust
let child = (self.next)(output)?;
Ok(PlannerOutput::Planner(child))
```

The callback constructs one planner, and that planner may subsequently produce
zero, one, or many children. The driver queues each returned child and chooses when
to execute it. Constructor errors propagate through `VortexResult`.

`Output` is stage-specific. Footer opening passes an opened-file context; index
pruning passes surviving region information; evaluation can pass `PlanView`.
Metadata and selections do not need to masquerade as plans to use the same
composition mechanism. A stage with one result invokes its callback once; a stage
with several results can invoke it over several computations.

This is the ordinary construction path. If a logical successor was already
exposed through speculation, its parent emits the matching ticket completion
instead of also calling `Next` to create duplicate ordinary work. The separate
speculative construction and cold-reconstruction contracts still apply.

## Constructing one file's lazy work

Compose constructors directly, as with writer strategies. For example, opening a
file and passing its result to footer pruning has this shape:

```rust
// after_prune is the previously composed constructor for surviving file work.
let after_open: Next<OpenedFile> = Arc::new(move |opened| {
    Ok(Box::new(FooterPruning::new(
        opened,
        predicate.clone(),
        after_prune.clone(),
    )))
});

let footer = FooterPlanner::new(path, after_open.clone());
```

The callback receives the actual opened-file result, including its source identity,
footer, and layout information. It is invoked after the opening planner has done
its IO and parsing. Pruning, plan preparation, filtering, optional splitting, and
projection can be composed the same way, with the appropriate input at each edge.
These constructor names are sketches, not existing Rust APIs.

For an already-open file, composition starts from the available metadata or plan
view. Binding that needs file metadata, optimisation, and substantial construction
happen as scheduled CPU work. A physical data plan still needs its source context;
segment IDs alone do not identify a file. Preparation has this outline:

```text
prepare(region_plan, query, file_context):
    bind the query to the file schema and region domain
    optimise the filter/projection computation, retaining both result roles
    retain one shared prepared representation
    optionally announce every known segment dependency of that representation
    create a projection continuation referring to its projection part
    return FilterEval(filter part, incoming selection, projection continuation)
        # With no filter, return ProjectionEval directly.

when FilterEval finalises a nonempty selection:
    use the continuation to create ProjectionEval(projection part, exact selection)
```

The two parts need not be disjoint: a column can be used by both expressions. They
are shared references or views, not separately rebuilt copies of the entire file
plan. The preparation implementation must preserve the identity of the filter and
projection outputs through rewrites; blindly treating the first two children of
any optimised node as those roles is not sufficient. Projection evaluation still
uses the final selection, so warming alone cannot evaluate expressions on rejected
rows or publish output.

The lazy tree is constructed as work progresses: one input region leads to a
filter, a final selection leads to a projection, and planning selected layouts
leads to morsels. Constructors decide region boundaries. There is no up-front
vector of all filters, projections, or morsels.

## One query over files opened one at a time

Reuse the same composed continuation for each file. Here `after_open` is the
`Next<OpenedFile>` assembled above, ultimately capturing the filter/projection
configuration and downstream constructors:

```rust
let open_file: Next<FileSource> = Arc::new(move |source| {
    Ok(Box::new(FooterPlanner::new(source, after_open.clone())))
});

let root = FileSequencePlanner::new(file_cursor, open_file);
```

This represents one logical filter and projection over many files. Each file needs
its own physical plan binding because its source, row count, segments, and chunk
boundaries differ. The caller supplies the expressions once; it does not build a
separate scan manually for every file. A shared logical query does not require one
mutable filter execution object to be shared between files or threads.

The root `FileSequencePlanner` holds the file cursor and `Next<FileSource>`. It
constructs one FooterOpen at a time when admitted by the driver. Configure the
driver to admit at most one active file for strict sequential file processing.
FooterOpen passes its result to the same per-file continuation described above:

```text
FileSequencePlanner(files A/B/C, open_file), driver admits one active file
    admit A
        FooterOpen(A) -> FooterPrune(A) -> index / plan preparation
            -> warming -> filters -> projections -> morsels
    A's entire work scope finishes or is pruned
    admit B
        FooterOpen(B) -> ...
    B's entire work scope finishes or is pruned
    admit C
        FooterOpen(C) -> ...
```

The driver gates further root advancement while the active-file permit is occupied;
this does not require inventing an IO request or ticket for FileSequencePlanner to wait on.
FooterOpen reporting `Done` only means it has handed off work. The file permit is
released after all descendants finish or cancel, not when the opener or filter
finishes. Root cancellation or a satisfied limit stops admitting more files.

Root admission assigns a file-scope token to the opener and its descendants. The
driver accounts for outstanding work within that scope, including further child
production and exposed tickets; it installs a child's responsibility before
retiring its parent. This is explicit work-envelope bookkeeping, not knowledge of
private fields in each planner implementation.

This is the strict one-active-file version. Serialising only footer opens while
allowing earlier files' execution to overlap is a different admission policy. More
active files can later use the same construction with separate worker-owned live
state, while morsels within the currently active file can already run in parallel
through the portable-description dispatch boundary.

The cursor holds source descriptions, not open files or a prebuilt plan per file.
If file discovery itself needs IO, it can be scheduled discovery work. Expressions
can be bound once for compatible schemas; validate each file, or explicitly adapt
its schema before reusing a binding. If no schema is known initially, the first
footer establishes it. An empty source needs an externally known schema if callers
require an output dtype. Track file ordinal and file-local row coordinates through
each binding so different file lengths do not change selection or ordering semantics.

## Existing Plan support and remaining construction work

The repository already has shared [PlanRef values](../../vortex-layout/src/plan/typed.rs)
and [lazy child slots](../../vortex-layout/src/plan/children.rs).
The existing [lowering helper](../../vortex-layout/src/plan/lower.rs) constructs
physical-plan fixtures for tests; it is not a production planning API. The current
[optimiser](../../vortex-layout/src/plan/optimize.rs) recursively walks the children
of its input. Production construction must preserve lazy regions and use bounded
plans or incremental optimisation before promising bounded lookahead.

A data-only plan cannot recover pruning metadata that it does not represent.
Keep footer/index context alongside it or add an explicit plan representation for
that metadata.
A supplied data-only plan without that context starts at evaluation preparation;
the composition must not fabricate missing footer or index stages.

The construction types above, stage-role views, dependency walk, warming request
ownership, and file-scope completion accounting still need concrete APIs. They are
constructor/driver responsibilities and do not require more methods on every
Planner or Morsel merely to store scan policy.

## End-to-end child counts

For the two-file example above, suppose the first surviving filter finds 20 rows
and the second finds 100 rows in its first region and none in its second:

```text
2 FooterOpen
    -> 2 FooterPrune
        -> A survives: 2 IndexPrune; B pruned: 0
            -> 2 FilterEval (through shared plan preparation and optional warming)
                -> 2 real ProjectionEval
                    -> 3 morsels, if the second projection splits in two
                        -> 120 selected rows across the output batches
```

Warming can read projection segments whose rows are later rejected. It does not
create additional speculative children or change the 120 selected rows. Child
counts, batch counts, physical read counts, and output row counts are separate.

With two workers, file A can start on T0 and file B on T1. If B is rejected, T1 can
accept fresh portable descriptions from A before those tasks start, once that
dispatch boundary exists. It cannot take A's already-running filter, morsel, or
speculative object. Initial dispatch of new descriptions does not require stealing
from another worker's live queue; stealing remains a later scheduler option.

## Ordering belongs to the driver

Each morsel preserves selected input-row order within its batches and across its
successive batches. A sparse selection such as `[2, 9, 17]` produces three projected
rows in that order, without placeholders for the rejected rows. Projected fields
use the same selection so their rows remain aligned.

The scheduler and state-machine driver own ordering between morsels. They can run
work in any order and choose ordered streaming delivery, immediate unordered
delivery, or reordering after collecting the results. Morsels do not enforce a
global delivery policy, and `MorselOutput::Batch(ArrayRef)` need not change.

The driver retains logical scan positions in work/output envelopes: partition/file
order, the input domain assigned to the morsel, and its batch sequence. Positions
come from planning, not task creation time, IO completion, or ticket arrival order.
For ordered streaming it also needs planning progress: an earlier region may still
produce a child even when no morsel for it exists yet. Handing off a region transfers
that responsibility to its children; pruning closes it without output. The precise
metadata and progress API remain to be specified.

For example, B's batch over `[100,200)` can complete before A's batches over `[0,100)`.
An ordered driver emits A's batches as they arrive and holds B until A's region is
finished. An unordered driver emits B immediately. A collector can retain their
positions and arrange them after execution. Ordered streaming bounds later buffered
output and reserves capacity for the earliest unfinished work to make progress.

## Steps to realise the system

1. **Finish the construction and completion contract.** Keep the three small
   traits. Specify stable work/request/ticket identities, IO input delivery, and a
   cold construction recipe that an external scheduler can retain. Make fresh
   planner/morsel descriptions portable and distinguish them from started objects;
   the current `Box<dyn Planner/Morsel>` outputs alone only express local execution.
   Settle speculative `compute()` and a CPU-only checkpoint for ordinary planners
   if phases need to yield without producing a child. Define `MayNeed` promotion,
   errors, cancellation, and late-completion handling.

2. **Build the default driver against controllable IO.** Start with a local ready
   queue, pending input routes, and ticket entries; then exercise two owner-local
   workers. Use the same public protocol that a custom scheduler will use. Verify
   IO arriving during parking, a ticket arriving before preparation finishes,
   per-region completions, cold reconstruction, and cancellation of one consumer
   of a shared read. No work stealing is needed.

3. **Complete one real scan end to end.** Open a real file, reuse known footer
   metadata where possible, plan a flat projection, and emit arrays from morsels.
   Connect explicit requests to the existing source/cache/read infrastructure and
   keep all substantial parsing, planning, and decode work in scheduled compute.
   Compose both a file-plan entry and a lazy file-sequence entry using `Next<Input>`,
   sharing the query configuration and continuations across files. Exercise strict
   one-active-file admission, an already-open file, and an in-memory source.
   Compare results with the existing
   scan path before expanding coverage.

4. **Add the pruning and evaluation stages.** Implement FooterPrune, index/zone
   pruning, FilterEval, and deferred ProjectionEval. Start with flat, chunked, and
   struct layouts, including conjunction and differing field chunk boundaries.
   Prove that a rejected file does not read data, a rejected region does not produce
   projection work, and an early filter region releases output while later IO waits.
   Cover the remaining supported layouts and expressions before migrating callers
   that depend on them; do not label a partial layout implementation universal.
   Add ProjectionSplit as an optional composed stage and verify that direct and
   split projection produce the same selected rows without duplication or omission.

5. **Support scan semantics and parallel output.** Dispatch fresh descriptions to
   owners, interleave their IO, and preserve started-object ownership. Add bounded
   output buffering, row/partition selection, row-index coordinates, ordered output,
   and limits applied after filtering. Ordered output needs progress for regions
   that emit nothing, not just keys on returned arrays. These are requirements for
   replacing existing scans, even if an initial prototype is unordered.

6. **Add plan warming and optional speculation independently.** Prepare bounded
   filter/projection plans, announce all their known segment dependencies, and pass
   the prepared parts to the two evaluation stages. Verify warming works without
   any calls to `speculate()`. Separately implement optional per-successor CPU
   preparation, streamed tickets, cold fallback, and nested zone-to-filter-to-
   projection preparation. Support real work becoming ready before, during, or
   after warming. Show that warming on/off returns the same results, unused
   failures are ignored, and memory/over-read remain bounded. Measure bytes,
   request counts, wait time, and CPU before choosing defaults.

7. **Route every public scan entry point through the common engine.** Keep caller
   APIs as adapters while moving file, layout, multi-file, query-engine, and binding
   paths onto this protocol. Preserve lazy execution, cancellation, schema, order,
   selection, and limit semantics. Existing async streams or blocking iterators may
   remain at integration boundaries; the planner/morsel protocol has no futures.
   Switch each default after its parity checks pass, then remove redundant executor
   paths once the supported scan inventory is covered.

The first implementation deliverable should be steps 1-3 with explicit acceptance
cases, followed by small independently reviewable additions. Production cutover is
the completion of steps 4-7, not merely a successful flat-file prototype.

## Integration points checked in this repository

These are existing entry points and behaviours to integrate or preserve, not a
requirement to reproduce the current executor's architecture.

| Existing area | Relevance to the migration |
| --- | --- |
| [File opening](../../vortex-file/src/open.rs) | Footer discovery, provided metadata, initial segment reuse, and source construction |
| [File scan entry points](../../vortex-file/src/file.rs) | Both `scan()` and `data_source()` must ultimately use the new engine |
| [File statistics](../../vortex-file/src/v2/file_stats_reader.rs) | Existing conservative file-pruning semantics to preserve |
| [Zone layouts](../../vortex-layout/src/layouts/zoned/reader.rs) | Existing map boundaries, row offsets, and pruning semantics |
| [Segment source API](../../vortex-layout/src/segments/source.rs) and [file source](../../vortex-file/src/segments/source.rs) | The current source API requests individual segments; the proposed IO adapter must define whole-batch registration and optional `MayNeed` semantics |
| [Layout scan builder](../../vortex-layout/src/scan/scan_builder.rs) | Existing ordering, selection, filtering, projection, batching, and limit entry points |
| [Multi-file source](../../vortex-file/src/multi/mod.rs) | Lazy file opening and footer caching need to survive the migration |
| [Data-source protocol](../../vortex-scan/src/lib.rs) | Scan requests carry ordering, limits, row/partition selections; partition execution must stay lazy |
| [DuckDB scan integration](../../vortex-duckdb/src/file_reader.rs) | Route scan construction and execution through the common protocol while preserving engine scheduling contracts |
| [DataFusion](../../vortex-datafusion/src/v2/source.rs), [Python](../../vortex-python/src/file.rs), [C FFI](../../vortex-ffi/src/scan.rs), and [Java scan API](../../java/vortex-jni/src/main/java/dev/vortex/api/Scan.java) | Preserve consumer contracts and exercise their integration tests during cutover |

The protocol and generic scheduler belong below file-specific implementations;
file/footer planners belong with file metadata and layout planners with layout
knowledge. Choose concrete crate placement after checking the dependency graph.
Custom schedulers must not need default-scheduler private types to complete IO,
route tickets, start fresh work, or consume arrays.

## Acceptance cases before making this the default

- Cached/uncached footers, empty files, corrupt required metadata, and sources whose
  size is initially unknown.
- One shared filter/projection configuration over files of different lengths and
  chunking; one-active-file admission must wait for descendants, and a limit or
  cancellation must prevent opening unnecessary later files.
- Whole-file pruning, per-map pruning, absent indexes, mixed map boundaries, and
  conservative versus exact row selections.
- No filter, conjunctions, all/no/some matching rows, selected rows across chunk
  boundaries, and the existing supported expression/layout combinations.
- One filter emitting several projections, one projection emitting several morsels,
  and two workers completing them out of order without duplicates or omissions.
- Direct projection and insertion of ProjectionSplit produce identical selected
  rows, including sparse selections, a view below the byte target, and one row
  larger than the target.
- Ordinary versus speculative execution, cold/partial/complete preparation, nested
  tickets, dropped preparation, and errors in candidates that are later pruned.
- Shared reads across stages, atomic batch registration, declined warming,
  promotion to required reads, and cancellation while another consumer still needs
  the same bytes.
- Plan-driven warming includes filter-only, projection-only, and shared segments,
  works with speculation disabled, and hands off evaluation without waiting for
  optional reads. Optimisation does not silently traverse the entire lazy file.
- Ordered output across empty regions, post-filter limits, bounded memory under
  backpressure, and cancellation without stranded work or lost wakeups.
- Parity through every supported scan entry point before removing its previous path.
