# Documentation TODOs

> **Status:** Current open documentation work.
> **Progress:** Captures gaps that block an advanced developer
> from implementing or extending the engine from docs alone.
> **Open questions:** task ownership may change as work lands.

## P0 — Reference detail

- **Runtime traits.** [Runtime traits](../reference/runtime-traits.md)
  is the authoritative shape, but it doesn't yet show worked
  examples for the trickier patterns: a transform that emits
  several outputs per input, a sink that buffers and flushes on
  `poll_finish`, a source that fans out async I/O.
- **Spawn primitives.** [Spawn primitives](../reference/spawn-primitives.md)
  needs a worked pattern for "operator with multiple in-flight
  `WorkHandle`s" (e.g. read-ahead with a small queue).
- **Runtime filters reference.** Resolve the open questions in
  [Runtime filters reference](../reference/runtime-filters.md):
  - tightening semantics (write-once vs monotonic);
  - multi-key prefix `RangeFilter` for sort-merge cursor exchange;
  - source-side translator chains and gating on local-state
    ready.
- **Lowering API.** Add a "common errors" section to
  [Lowering API](../reference/lowering-api.md) describing the
  `BuildError` paths.

## P1 — Runtime architecture detail

- **Multi-thread executor and channel exchanges.** Expand
  [Runtime architecture](../architecture/runtime.md) to cover
  lane allocation, work-stealing affinity, and back-pressure paths
  once Milestone 3 lands.
- **Memory arbiter design.** Write the arbiter doc once
  Milestone 5 scope is locked.
- **Cooperative spill protocol.** Document the pressure signal
  and per-operator flush hooks.
- **I/O broker layer.** Document the broker on top of
  `SpawnRuntime` (priority heap, coalescing, hierarchical
  admission keys).

## P1 — Operator conformance

For each Wave-A and Wave-B operator (see
[Implementation plan](roadmap.md)), add a conformance spec
covering:

- config schema;
- declared domains, contracts, parallelism;
- which resources it publishes or subscribes to;
- input and output contracts;
- internal state machine (one diagram per operator);
- validation errors;
- conformance tests and golden examples.

Operators in scope:

- `Filter`
- `Project`
- `HashAggregate`
- `TopK`
- `Sort` (`SortRunSink` + `MergeKSource`)
- `HashJoin` (build + probe)
- `PreSortedMergeJoin`
- the existing `ParentChildMin`, `Limit`, `Gather`,
  `VortexScanSource`, `VortexAggregate`, `SumCountAggregate`,
  `IntSource`, `CollectSink`.

## P2 — Worked examples

- End-to-end ClickBench Q1, Q3, Q5, Q20 with skipped-work metrics
  (cross-reference Vortex layout zone pruning).
- End-to-end TPC-H Q1 or Q6.
- One worked join query that publishes a `BloomFilter` from build
  and consumes it on probe.
- One worked query that publishes a `RangeFilter` from `Limit`
  and consumes it at the source.

## How to update this page

When a TODO lands as docs, move the line into the changelog of
the relevant page rather than ticking it off here. When a TODO
turns out to be the wrong frame, delete it.
