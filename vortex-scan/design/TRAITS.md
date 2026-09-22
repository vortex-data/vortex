<!--
SPDX-License-Identifier: CC-BY-4.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Scan traits with scheduler-routed tickets

Design draft, 2026-09-22. This proposes lazy scan planning with explicit IO,
scheduler-routed tickets, `Blocked(ticket)`, and scheduler-owned disposal of
speculative preparation. It is a proposal, not an implementation.

See [the scan pipeline and implementation steps](PIPELINE.md) for stage
examples, child counts, plan-driven warming, single/multi-file construction, and
migration of scan callers.

## Common types

```rust
type IoBatch = Vec<(IoRequest, RequestScope)>;
type TicketData = Box<dyn Any + Send + Sync>;
type Speculation = (Box<dyn SpeculativePlanner>, TicketId);

enum State {
    Done,
    NeedsCompute,
    NeedsIO(IoBatch),
    Blocked(TicketId),
}

struct CompletedTicket {
    ticket: TicketId,
    input: Option<TicketData>,
}

enum PlannerOutput {
    Done,
    NeedsIO(IoBatch),
    Planner(Box<dyn Planner>),
    Morsel(Box<dyn Morsel>),
    TicketComplete(CompletedTicket),
}

enum MorselOutput {
    Done,
    NeedsIO(IoBatch),
    Batch(ArrayRef),
}
```

`TicketId` identifies one speculative successor. `TicketData` contains the parent
result needed to construct that successor: for example, surviving rows from zone
pruning or an exact filter selection. `None` means that successor was pruned. The
boxed, type-erased payload is a proposed spelling; its concrete representation is
still open. Row selections must include their coordinate domain.

`IoRequest` identifies a segment and its segment source (`SegmentGroup` or
`SegmentSourceId`, naming still open), or a bootstrap footer read for that source.
`RequestScope` identifies the consumer's input within that work item. Their concrete
types, including bootstrap reads and error delivery, remain to be specified. All
stages use the same source-wide IO service, including morsels and speculative work.

`state()` is a cheap, nonblocking inspection. It does not advance a layout cursor,
evaluate a predicate, or drain requests. `NeedsIO` describes the whole currently
known batch for the current dependency phase; there is no caller-provided budget or
row split. Repeated observations retain stable request identities. The scheduler
registers each logical consumer request once and parks it while the read is pending.
Another ready result may make the object runnable before every read finishes.

Publishing `PlannerOutput::NeedsIO` or `MorselOutput::NeedsIO` does not by itself
mean the producer must wait: register the batch and inspect `state()` again.
Plan-driven warming can publish optional reads and remain `NeedsCompute` to hand
off the prepared work immediately. Optional-read intent and registration ownership
are still construction/IO API choices. Warming walks the optimised plan shared by
filter and projection; it does not require `speculate()` or speculative tickets.

`Blocked(ticket)` specifically means speculative preparation cannot progress without
that parent's result. Pending reads use `NeedsIO`. Ordinary planners and morsels do
not return `Blocked` in this sketch. A speculative object remains registered under
its ticket while it reports `NeedsCompute` or `NeedsIO` too: completion does not have
to wait for it to reach `Blocked`.

## Planner

```rust
trait Planner {
    fn state(&self) -> State;

    fn compute(&mut self) -> VortexResult<PlannerOutput>;

    fn speculate(&mut self) -> VortexResult<Option<Speculation>> {
        Ok(None)
    }
}
```

A planner decides which work is needed. It privately owns layout cursors, current
row selections, pending inputs, and partially computed results. `compute()` advances
CPU work when the object reports `NeedsCompute`. It may discover another batch of
IO, produce one child planner or morsel, complete one ticket, or finish.

Returning a child or a ticket completion does not finish the parent. Successive
computations can produce more work, interleaved with further IO and CPU. Any planner
can return a morsel as soon as it has authorised array-producing work; a projection
planner is not a mandatory intermediate step.

Stages compose through a reusable constructor
`Next<Input> = Arc<dyn Fn(Input) -> VortexResult<Box<dyn Planner>> + Send + Sync>`.
The stage calls it when an ordinary successor's input is ready, then returns the
constructed planner to the driver. The callback constructs work; it does not run
the next stage. Its shared construction policy does not make the returned live
planner `Send` or `Sync`. See [composition examples](PIPELINE.md#composing-planners-with-next)
for direct filter-to-projection and insertion of a lazy byte-based ProjectionSplit.
Stages choose their own input types; a plan view is useful for evaluation but is
not required for metadata stages.

`speculate()` exposes at most one new candidate per call. It performs cheap
construction; substantial planning happens through the child's `compute()`. `None`
means no candidate is available now. Later progress can make more candidates known.
It can be called while the parent waits for IO. Each exposed successor gets a
distinct ticket, even when several successors cover the same rows.

For each logical successor the parent chooses exactly one delivery path:

- Unexposed: return `Planner` or `Morsel` when ready, or nothing if pruned.
- Exposed by `speculate()`: return `TicketComplete` exactly once when that successor's
  result is final. Do not also return the ordinary successor.

The parent remembers which successors were exposed, but does not track whether the
scheduler ran or discarded their preparation. On normal completion it emits all
owed ticket results before `Done`. Cancellation and failure terminate dependent
ticket entries through scheduler bookkeeping instead.

If region A is final while region B still waits for IO, the pending completion for A
makes the planner `NeedsCompute`. It must not hide that completion behind B's wait.

## Morsel

```rust
trait Morsel {
    fn state(&self) -> State;
    fn compute(&mut self) -> VortexResult<MorselOutput>;
}
```

A morsel produces arrays for an authorised piece of work. It can still need IO,
decoding, or expression evaluation. `Batch` transfers one array to the consumer and
leaves the morsel alive until it reports `Done`. A later computation can discover a
new dependency, such as a dictionary read, and return `NeedsIO`.

The morsel owns its selected rows, input handles, decode recipe, and output position.
Physical reads may cover more rows than its selection; emitted arrays contain only
selected rows. Backpressure is scheduler policy: do not call `compute()` to produce
another batch until the consumer can accept it.

A morsel preserves selected input-row order within each batch and across successive
batches. Ordering between morsels belongs to the scheduler and state-machine driver:
they may preserve scan order during delivery, emit batches as ready, or retain
position metadata and reorder collected results at the end. `Batch(ArrayRef)` stays
unchanged; work/output envelopes carry the logical scope and batch sequence needed
by the driver. Planner progress must also account for earlier regions that are
unexpanded or pruned without output. The concrete envelope and progress signatures
remain open; the traits do not gain an ordering-policy method. See
[ordering in the pipeline](PIPELINE.md#ordering-belongs-to-the-driver).

## Speculative planner

The following completes the preparation side of the trait as a proposal:
`compute()` updates private preparation and the scheduler then rechecks `state()`.
Its unit return avoids introducing a second kind of child-output event before the
parent's result is known. The preparation signature has not been separately agreed.

```rust
trait SpeculativePlanner {
    fn state(&self) -> State;

    fn compute(&mut self) -> VortexResult<()>;

    fn speculate(&mut self) -> VortexResult<Option<Speculation>> {
        Ok(None)
    }

    fn make_real(
        self: Box<Self>,
        ticket: CompletedTicket,
    ) -> VortexResult<Option<Box<dyn Planner>>>;
}
```

A speculative planner prepares one possible successor before its parent supplies
the constructor arguments that depend on its result. Preparation may discover IO,
walk known layouts, prepare decode recipes, or evaluate a predicate over a wider
candidate domain. Its `state()` reports `NeedsCompute` or `NeedsIO` while useful
preparation remains, then `Blocked(its_ticket)` when only the parent can unblock it.
It does not use `Done` to mean that preparation is finished.

`make_real` consumes the speculative object and its matching completion. This is
what earlier descriptions called activation: bind the parent's authoritative result
to the prepared child. `None` input prunes it. `Some(input)` constructs the real
planner, retaining useful buffers, recipes, pending reads, and child-ticket
associations. It may still return `None` if the authoritative scope is empty after
intersection with the candidate. A mismatched ticket is a protocol error.

Binding must work with no preparation, partial preparation, completed preparation,
or IO still in flight. It must be cheap; further CPU work belongs to the returned
planner. This sketch retains `Option<Planner>` as the return type. The resulting
planner can return a morsel; returning a morsel directly from `make_real` is an open
API choice, not required for correctness.

The optional nested `speculate()` supports a speculative filter exposing projection
candidates while waiting for a zone result. Those projections receive their own
tickets. Tentative filter results cannot complete these tickets until the filter's
own parent result has made the selection authoritative. When made real, the filter
retains the associations and completes each projection ticket as its region finishes.

Preparation failures must not reject rows that would have been pruned. The scheduler
retains a speculative failure with its candidate and reports it only if the
authorised work still needs the failed operation; otherwise it discards it or builds
the required work cold. The concrete retained-error and retry contract is still open.

## Dropping preparation is scheduler policy

The parent receives no notification when the scheduler drops speculative preparation.
It still emits exactly the same ticket completion. The scheduler must retain enough
cold construction information under that ticket to recover the required child.

```text
ticket entry:
    parent work id, child work id, ticket id
    cold child description / factory
    optional live speculative preparation
    associations with any exposed speculative descendants
```

On `TicketComplete(Some(input))`, reuse live preparation if present; otherwise create
a cold speculative instance from the retained description and bind `input`. On
`TicketComplete(None)`, discard the entry and its unneeded descendants. Discarding
preparation releases only that consumer's interest in shared reads.

An exposed candidate cannot be erased outright while its parent continues: the
completion would have nowhere to go and required output could be lost. The scheduler
may discard the expensive live object, while the cheap entry survives. Keeping a
cold object without ever running it is also valid.

The pair returned by `speculate()` does not by itself expose a reconstruction recipe.
Before implementation we must choose how construction provides that recipe to an
external scheduler: a description/factory alongside the pair, or a way to recover a
cold description from the live object. This document does not pretend the three
traits alone settle that ownership boundary. Rebuilding a parent that has already
exposed descendants must preserve their ticket identities, or cancel those entries
as a subtree and discard their stale completions without duplicating output.

## IO delivery, ownership, and coalescing

The scheduler maintains separate routes for IO consumers and parent results:

```text
read id   -> [(owner worker, work id, request scope), ...]
ticket id -> one child work entry, plus its producing parent
```

The parent and child agree on the ticket when `speculate()` returns the pair. The
scheduler records it then, not when the child first reports `Blocked`. IDs must be
unique among live routes and protected against stale completions if reused.

IO completion fills the consumer's input and requeues its owning work item. Ticket
completion binds or prunes a speculative child. Reading bytes never authorises rows.
The concrete input handle or delivery binding is supplied through construction and
still needs to be specified; these traits intentionally do not invent a `push_input`
method. Transition to a real planner must preserve pending IO routes, either with a
stable work identity and retained input handles or an explicit atomic handoff.

Register each newly discovered IO batch before dispatching its reads. The shared IO
service can deduplicate and coalesce across footer, zone, filter, projection, and
morsel consumers of the same source. A batch is not a completion barrier. An optional
coalescing planner can expose requests for a chosen row window, but coalescing is
not confined to the requests of one planner. Already dispatched reads cannot acquire
new byte ranges. Cancelling one consumer does not cancel another's required read.

Portable plan descriptions and ticket payloads are `Send + Sync`. Live planners,
morsels, and speculative planners need neither bound. A worker owns its live work
including while it waits for IO or tickets. A child is queued locally. The first
scheduler has no stealing; a later scheduler can move unstarted portable descriptions
without moving in-flight execution.

## Worker loop with speculation always admitted

This is a demo policy: admit every opportunity encountered, but take one step per
visit so speculation discovery does not prevent real work from progressing. A
production scheduler limits speculative IO, CPU, and memory.

```text
loop:
    drain incoming IO completions and ticket completions
        IO: deliver each input; enqueue its owner if it can progress
        ticket(None): retire the candidate and its speculative descendants
        ticket(Some): make live or cold preparation real; enqueue it locally

    choose one owned work item that is ready or has an unvisited speculation turn
        # This choice is where scheduling policy compares available work.

    if it is a planner or speculative planner:
        call speculate() once
        if Some(child, ticket):
            record ticket route and retain the cold construction recipe
            queue child locally; schedule another discovery turn for the parent

    match work.state():
        Done:
            retire ordinary work after its owed completions have been emitted
        NeedsIO(batch):
            register requests not previously registered; park on pending inputs
        Blocked(ticket):
            park the speculative object on its already registered ticket route
        NeedsCompute:
            compute once
            ordinary Planner/Morsel result: queue child, deliver array, register IO,
                route one ticket completion, or retire on Done
            speculative result: retain preparation; inspect state again
            requeue if further CPU or completion delivery is ready

    if no local work is runnable:
        wait for a completion, a new root description, or cancellation
```

Wake registration and completion delivery must be race-safe; a completion between
inspection and parking cannot be lost. New speculative opportunities must receive
a turn when their parent's state changes. The loop never runs `compute()` merely to
poll an object that is waiting. Parent failure or cancellation also retires its
dependent entries, including candidates whose normal ticket event will never arrive.

## Scan-stage sketches

| Stage | Ordinary progress | Speculative opportunity |
| --- | --- | --- |
| External index | Request index data; return footer planners for candidate files | Only when candidate source identities and constructor inputs are already known |
| Footer open | Read footer location/bytes; parse row count and layout metadata; return footer pruning | Cannot construct useful zone children before the layout is known |
| Footer pruning | Evaluate file statistics; return zone planners or complete their individual tickets | Zone work whose layouts are now known |
| Zone pruning | Read maps; compute conservative survivors at map boundaries; return filters or complete their tickets | Filter preparation over candidate regions |
| Filter | Read predicate inputs; compute an exact selection per region; return projections/morsels or complete their tickets | Projection preparation for each independently finalisable region |
| Projection | Walk selected layouts; perform deferred per-flat planning; return morsels | Optional additional candidates when their fixed inputs are known |
| Morsel | Read, decode, evaluate, and emit selected array batches | No speculative method in this sketch |

A single map over all rows gives one zone result domain; separate maps may give
separate results at their boundaries. Construction decides whether predicate columns
are evaluated together (the default) or cascaded to avoid expensive reads. Both zone
pruning and filter conjunction preserve row coordinates when combining selections.

For `eval(chunked(flat))`, a planner keeps a chunk cursor, chooses the next selected
flat, and performs its deferred planning through `compute()` and any required IO.
It constructs only the next useful child or morsel. A struct aligns fields with
different chunk boundaries before assembling batches. Row splits belong to this
construction and layout logic, not parameters to `state()` or `compute()`.

Example: one filter covers `[0,1024)` and exposes projection A with ticket A for
`[0,512)` and projection B with ticket B for `[512,1024)`. Both may prepare. The
scheduler drops A's prepared buffers but retains its cold recipe. The filter then
returns `TicketComplete(A, Some(selection_A))` while B still waits on predicate IO.
The scheduler builds A cold and runs its projection. Later the filter returns
`TicketComplete(B, None)` and B is discarded. No second projection A is emitted by
the filter, and it need not know A's preparation was dropped.

With two files, T0 owns file A's live chain and T1 owns file B's live chain. Shared
physical IO may serve both, but each completion returns to its owner's inputs.
Each file has its own row count and selections; equal chunking is never assumed.

## Remaining signature choices

- Confirm the proposed speculative `compute() -> VortexResult<()>` preparation API.
- Specify cold description/factory ownership so dropping preparation is supported by
  an external scheduler, including nested candidates.
- Specify IO completion bindings, stable identities, and error handling.
- Choose the concrete ticket payload representation and its type checking.
- Decide whether `make_real` should directly return a morsel as well as a planner.
- Decide whether ordinary computation needs a CPU-only checkpoint outcome. The
  current output enums do not yet provide one.
- Specify work/output position metadata and planner progress for driver-owned
  ordering, plus cancellation details, memory limits, and backpressure.

These are explicit design gaps to resolve before implementation, not hidden
requirements for a particular default scheduler.
