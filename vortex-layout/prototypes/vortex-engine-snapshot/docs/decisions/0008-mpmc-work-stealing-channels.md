# 0008: MPMC work-stealing channels and single-output operators

> **Status:** Accepted ADR.
> **Progress:** Accepted as the channel-exchange contract for the
> engine. Operators emit at most one output port; channels between
> pipelines are MPMC work-stealing deques sized by byte counters.
> **Open questions:** lane-affinity heuristic for steal preference
> when producer and consumer lane counts disagree; reorder-buffer
> admission policy.

## Status

Accepted.

## Context

Two assumptions in the early channel model turned out to be in
tension with the parallelism story:

1. Multiple output ports per operator. In practice every shipping
   operator declared zero or one output port. Multi-output added
   ABI surface and complicated requirement and capacity tracking
   without a real consumer.
2. A topology that named producer × consumer lane shape (SPSC,
   MPSC, SPMC, MPMC). Producer parallelism was silent in the
   topology label, and lane mismatches were absorbed at runtime
   by a `Mutex<Channel>` taken on every push and pop.

Crossbeam's work-stealing deque is the standard primitive for
this shape: `Worker<T>` is owner-only push/pop, and one or more
`Stealer<T>` handles read FIFO from peer threads.

## Decision

### Operators emit at most one output port

`OperatorSpec` carries `output: Option<OutputPortSpec>`. An
operator is a source (no output), a sink (no output), or a
transform (one output, one or more inputs).

A single output port may connect to multiple distinct consumer
input ports on different downstream operators. The channel
replicates each batch to every consumer port. Operator-fanout is
distinct from lane-fanout: when one consumer operator declares
`Parallelism::LaneSafe`, that single consumer port has multiple
lanes sharing the input. Lane-fanout is handled by work-stealing
inside the channel runtime.

Operators that previously needed multiple semantically distinct
output streams become multiple operators, each consuming what it
needs from the same upstream output port (or from a shared
`Resource`).

### Buffered channel exchanges are MPMC work-stealing

A buffered channel exchange uses an MPMC work-stealing queue
internally. The runtime materialises each producer lane as a
`Worker<Batch>` and each consumer lane as a set of
`Stealer<Batch>` handles, one per producer lane.

Every lane shape degenerates to the same code path:

| Producer lanes | Consumer lanes | Behaviour |
| --- | --- | --- |
| 1 | 1 | One worker, one stealer; FIFO pass-through. |
| 1 | M | One worker, M stealers; consumer lanes steal. |
| L | 1 | L workers, one stealer reading from all. |
| L | M | L workers, M stealers; consumers prefer matched-index then steal. |

Consumer lane `i` prefers stealing from producer lane `i mod L`
when present and falls through to peer producers. The preference
is a heuristic for cache locality, not a correctness invariant;
the steal fallback gives skew-tolerance for free. When `L > M`,
surplus producer lanes have no matched consumer; their batches
are picked up by whichever consumer lane next runs the
steal-fallback loop.

For operator-fanout (one output port → N distinct consumer input
ports), the runtime instantiates N independent work-stealing
queues, one per consumer port; producer pushes are replicated to
all N queues. Within each queue, lane-fanout works as above.

Lane mismatches between producer and consumer operators are
absorbed by the channel without graph-visible repartitioning.

Key-routed redistribution is **not** a channel concern. A query
that needs hash partitioning (e.g. shuffle before group-by)
splits into two tasks at that boundary; see
"Repartitioning is a task-graph boundary" below.

### Inline pipelines (within a pipeline) need no channel

Inside a single pipeline there are no channels. The source,
transforms, and sink talk through direct method calls under the
pipeline driver. Channel exchanges appear only at pipeline
boundaries.

This means that filter, projection, and other cheap chained
transforms inside one pipeline pay no inter-operator queue cost.
Buffered exchange channels appear only where two pipelines need
to be decoupled — at parallelism transitions, decoupling points
for back-pressure, or expensive operator handoffs that benefit
from queue amortisation.

### Capacity and back-pressure

Each `Worker<Batch>` is paired with an atomic byte counter. The
channel exposes `has_capacity` returning true while
`bytes_in_flight < current_grant`; a push increments the counter
and a pop or steal decrements it. A push that would exceed the
grant returns Pending; the producing sink consults
`has_capacity` before pushing or returns `Pending` from
`poll_send` until capacity is available.

Channel capacity bounds are byte-budgeted; a future memory
arbiter grants the current capacity within those bounds.

### Ordering is opt-in per consumer input port

Most parallel-to-serial reductions (`Sum`, `Count`, `Mean`,
`CountDistinct`, hash-table sinks) are commutative and do not care
about input order. Operators that *do* need ordered input
(`Limit`, top-N, ordered sinks, merge-preserving transforms)
declare it on the input port and the channel materialises a
span-keyed reorder buffer ahead of the consumer. The buffer's
bytes are charged to the channel's memory grant.

### Hash-partitioned redistribution stays in the resource layer

Channels never carry key-routed data. Build-side hash tables,
bloom filters, finalised group-by tables, and similar key-aware
redistribution remain `Resource`-mediated: the build sink
populates a `Resource`, and the probe transform reads from it.

### Repartitioning is a task-graph boundary

When a query needs partitioned redistribution — shuffle before
hash group-by, hash-partitioned join, sort-merge join after
sorting — the plan splits into multiple tasks at that boundary.
Repartition is not a graph operator and not a channel topology;
it is the boundary between two tasks.

The shape:

- The upstream task ends with a `RepartitionSink` operator that
  hashes each input batch's key by `partitions` and writes
  per-partition output to a shared exchange buffer (an
  `Arc`-shared, partition-indexed materializer).
- The downstream task begins with N `RepartitionSource`
  operators (one per partition), each reading from one slot of
  the exchange buffer.
- The driver schedules tasks in topological order across
  exchange-buffer dependencies.

The materializer trait abstracts where the exchange data lives:
in-memory today; future implementations can spill to disk or
stream over a network for distributed execution.

## Consequences

### What gets easier

- Lane-safe operators drop their channel-side coordination — no
  more `Mutex<Vec<Batch>>` mediating producers into a shared
  output. Each lane pushes to its own `Worker<Batch>`.
- The graph builder no longer reasons about lane-region
  boundaries.
- Skew tolerance is automatic: an idle consumer lane steals from
  any producer rather than waiting on its matched-index producer.

### What gets harder

- Operator authors who previously emitted on two output ports for
  two semantically distinct streams must split into two operators
  (each consuming the same upstream input or a shared resource).
- Reorder-buffer memory has to be reasoned about per ordered
  consumer.
- Cache locality across a lane-preserving chain is best-effort —
  the steal fallback can route a batch to a peer consumer lane on
  a different worker. The matched-index preference partially
  offsets this.

### What future work must respect

- Channel exchanges are the only mechanism for intra-task
  per-batch parallel handoff between pipelines. Operators must
  not implement private cross-lane queues.
- Operators that need ordered input declare it; operators that
  emit ordered output cannot rely on the channel preserving order
  without the consumer's ordering request.
- Key-aware redistribution between tasks goes through the
  exchange-buffer task-graph boundary, not channels.
  Operator-internal lookup tables (hash tables, bloom filters)
  stay on `Resource`s.

## Alternatives considered

**SPSC per matched lane pair, MPMC only for mismatches.**
Rejected. SPSC would be marginally faster on the matched path but
introduces two lowering shapes, doubles the runtime data
structures, and is skew-fragile: a slow consumer lane back-pressures
its matched producer even when peer consumer lanes are idle.
Work-stealing handles balanced and skewed inputs uniformly.

**`crossbeam-channel` MPMC.** Rejected. `crossbeam-channel`'s
MPMC variant is an internally-locked ring buffer; producer-side
pushes contend on a single lock. `crossbeam-deque`'s `Worker` is
owner-only and lock-free on push, and stealers contend only with
each other (not with the owner) on the opposite end of the deque.

**Explicit `Repartition` operator at every lane mismatch.**
Rejected. The operator would be pure plumbing: no semantic
translation, no work that the channel cannot do itself. Forcing
graph builders to insert it adds boilerplate without correctness
benefit.

**Multi-output via `Tee` channel.** Rejected. Single-output
operators with fan-out at the channel level cover every case the
prototype has hit. A `TeeOperator` would re-introduce multi-output
behaviour through a different name; better to keep the rule
simple.

**Order-preserving channel default.** Rejected. The common case is
commutative reduction; defaulting to ordered would force every
channel to materialise reorder state and would penalise the
operators that don't need it. Opt-in is the right default.

**MPMC at every operator boundary, no in-pipeline fusion.**
Rejected. A typical query chains many cheap row-preserving
operators (filter, projection, expression eval) whose total cost
per batch is on the order of a queue push plus a queue pop.
Forcing a queue between every adjacent operator pair would
inflate per-batch overhead by an order of magnitude on hot paths.
Pipelines are direct-call inside; channel exchanges apply only at
the boundaries where queue cost is amortised by parallel handoff,
decoupling, or the work the consuming operator performs per batch.
