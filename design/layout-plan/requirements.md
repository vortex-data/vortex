<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# Layout Plan (scan v2) — Requirements & Important Features

Living document. Captures the requirements, constraints, and decisions for the redesign of the
layout read path ("optimize once, then execute" replacement for the current `LayoutReader`
protocol). Companion document: [`plan-sketch.md`](./plan-sketch.md).

## Problem statement

The current read path (`vortex-layout/src/reader.rs`, driven by
`vortex-layout/src/scan/tasks.rs::split_exec`) calls each `LayoutReader` one
`(row_range, expr, mask)` at a time through a hardcoded prune → filter → project protocol.
Three structural problems follow:

1. **No lookahead.** A reader never knows which expressions or splits arrive later. The dict
   layout cannot choose between pushing a predicate into the code domain vs. evaluating it once
   over values and sharing the result, because whether the projection will decode values anyway
   is invisible. Cross-call state is ad-hoc per-reader caching, including a known-unsound
   per-expression values cache (`TODO(joe)` in `layouts/dict/reader.rs`: fallible expressions
   evaluated over values that may never be referenced after filtering).
2. **No sub-segment I/O.** `SegmentSource::request(SegmentId)` reads whole segments, and the
   prefetch/coalescing contract requires registering reads *before* filter masks resolve, so a
   flat leaf cannot read only the byte ranges needed for surviving rows.
3. **Fixed protocol.** Pruning/filter/projection phases are hardcoded in the scan driver.
   Aggregates have no place in it (`vortex-array`'s `aggregate_fn` machinery is unreachable from
   scans), and zone-map stats are only consumed as a pruning oracle, never as a data source for
   answering aggregates.

## Functional requirements

- **R1 — Prune + filter + project.** The plan must support pruning, filtering, and projection,
  where the projection may apply scalar functions and aggregations.
- **R2 — Aggregate pushdown, streaming, no shuffle.** Aggregates are computed as *partial,
  mergeable accumulator states* (reusing `vortex-array::aggregate_fn`), streamed in row order.
  Grouping contract: aggregations are folds over **row-index slices** supplied with the query
  (e.g. rows `0..7`, `7..14`, …); ungrouped aggregation is a single slice; engines derive slices
  for sorted-key grouping; value-based GROUP BY stays in the engine. Aggregates may be satisfied
  from statistics (zone maps, file stats, row counts) when provably exact, with data evaluation
  as fallback.
- **R3 — Filter pushdown.** Including dict code-domain rewrites, decided globally by the
  optimizer (which can see whether values are decoded by the projection anyway).
- **R4 — Dynamic filters.** Predicate values may be late-binding and updated mid-execution;
  pruning is re-run on update (today: version counters). Plan *structure* is unchanged by
  updates.
- **R5 — Pruning, extensible to index layouts.** Zone maps and file stats refine selection;
  future index layouts plug in as additional selection refiners through the same mechanism.
- **R6 — Sub-segment reads.** Plan byte ranges within a flat segment from demanded rows, using
  the inline array-tree metadata already stored in `FlatLayout` (`FLAT_LAYOUT_INLINE_ARRAY_NODE`).
  No file-format change required for v1.
- **R7 — Optimize once, execute many.** The plan is bound to a **layout tree** (one or more
  files) and is reusable across repeated executions over different row ranges (random access,
  DuckDB/DataFusion-style repeated scans).
- **R8 — Existing scan features preserved.** Row-index selection, limit, ordered/unordered
  streaming, row offsets (`row_idx`), struct validity, and metrics must carry over.

## Important properties & constraints

- **C1 — SIP is efficiency-only.** The shared selection/demand structures are *sideways
  information passing*: conservative, possibly imprecise, and never load-bearing for
  correctness. "Cannot be wrong, can be imprecise."
- **C2 — Exact results flow on edges.** Filter operators stream exact boolean/mask outputs on
  dataflow edges (as today's `MaskFuture`s do); consumers gate correctness on edge resolution,
  not on SIP state.
- **C3 — Preserve eager registration + coalescing.** Statically-known segment reads keep
  today's register-early-coalesce-later behavior. Computed (sub-segment) reads are an optimizer
  choice, applied where the demand information pays for the extra I/O round trip.
- **C4 — Monotone SIP.** Selection only shrinks (rows move to *cannot match*); demand only grows
  (rows move to *will be needed*). Limit / early termination is executor-level **cancellation**
  of plan regions, never retraction inside the SIP structures.
- **C5 — Two parallel implementations.** The new plan/execute stack is built alongside the
  existing `LayoutReader` path and switched over when ready. No incremental entanglement.
- **C6 — No file-format changes** required by the core design.
- **C7 — Lazy unroll.** Plan unrolling and optimization are incremental (per phase / per chunk
  / per row section, pipelined with execution). Plan size must not be O(all chunks) up front:
  unroll the first chunk of each row section, execute it, and unroll more concurrently.
- **C8 — Multiple row domains.** Root rows, dict-values rows, and zone indices are distinct
  coordinate domains. Mappings are static (zone→row-range, chunk offsets) or data-dependent
  (codes→values, an operator in the plan). Zone pruning and dict pushdown use the *same*
  refinement-through-mapping mechanism.
- **C9 — Row demand drives I/O.** Operators communicate row need/non-need through demand so
  that, e.g., a projection observing near-dense demand can start whole-segment downloads early
  instead of waiting for final masks; conjuncts skip rows other conjuncts already excluded
  (replacing adaptive conjunct *ordering* with shared selection).

## Non-goals (v1)

- Value-based GROUP BY inside the scan (engine concern; row-index slices only).
- Cross-file shuffle/exchange of any kind.
- Query-level template plans shared across layout trees (design must not preclude; not built).
- Format extensions for row-addressable segments beyond the existing array-tree metadata.
- Mid-execution re-optimization beyond dynamic-filter re-pruning.

## Decision log

| # | Decision | Notes |
|---|----------|-------|
| D1 | Plan is bound to a **layout tree** (one or more files). | Multi-file composition via existing tree-composing layouts. |
| D2 | **Two parallel implementations**, switch over later. | New module/crate beside `LayoutReader`. |
| D3 | Plan IR is an **operator DAG over segments**. | Optimizations are graph rewrites. |
| D4 | Aggregates are **streaming partial states, no shuffle**. | Mergeable accumulators; engine merges/finalizes. |
| D5 | **Plan/execute in phases; unroll + optimize lazily.** | Pipelined: unroll first chunk per row section, execute, unroll more. |
| D6 | **Row demand** is the adaptivity mechanism. | Ternary per row starting "maybe-needed"; operators publish/observe. |
| D7 | Sub-segment I/O v1 uses **existing array-tree metadata**; the missing piece is an I/O-planning operator over a flat segment. | No format change. |
| D8 | Aggregation grouping = **row-index slices**, streamed in order. | Ungrouped = one slice. |
| D9 | **Two separate SIP structures**: downward-only selection, upward-only demand. | Not one ternary lattice. |
| D10 | SIP representation: **cheap updates, cheap counts, exact-rows-on-request**; block-level approximation acceptable (e.g. 16-row granularity) on the demand side. | Exactness lives on edges (C2). |
| D11 | **Phase-driven executor**, demand-aware operators; operators never self-schedule. | |
| D12 | Masks/booleans **stream on edges as usual**; SIP is advisory only. | Removes any "finality" bookkeeping: correctness gates are dataflow dependencies. |
| D13 | Bounded runtime adaptivity lives **inside operators reading SIP** (e.g. density-threshold strategy choices, early whole-segment downloads). | DAG structure stays static; strategies are operator implementation details. |
| D14 | **Demand is a plan-time backward propagation over DAG edges**, refreshed at phase boundaries from current selection SIP; data-dependent demand (codes→values) is `Computed` from an edge. | Selection remains the runtime-refined structure filter results write into. |
| D15 | **Operators are stateful stream transformers**; edges are ordered streams of region-stamped patches. | Flat leaves emit patches as I/O resolves (sub-segment, incremental); struct aligns child streams; chunked buffers children for readahead. |
| D16 | *(provisional)* Expression nodes are **conjunct/field-opaque `Eval`s** in v1; shared-subexpression hoisting may be added as an optimizer pass. | Owner undecided; revisit once streaming operator semantics are settled. |
