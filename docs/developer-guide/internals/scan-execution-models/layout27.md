# `layout27` Scan Planning and Execution

This document describes the `layout27` work preserved on the `ji/layout27` branch, using commit
`9734b85de4` as the comparison point. It separates the broader architecture present on that branch
from the hybrid execution path used at that exact tip.

That distinction is essential: the branch contains a substantial prepared-read scheduler, but the
tip commit routes ordinary bound scans through V1 execution.

## Intended end-to-end model

The broader branch architecture is:

```text
LayoutRef
  -> layout vtable new_scan_plan hook
ScanPlanRef
  -> try_push_expr(projection and predicates)
pushed ScanPlanRef trees
  -> initialize per-scan state
  -> prepare reads, evidence, statistics, and splits
prepared handles
  -> create fixed-morsel ReadTask values
  -> expose required reads, prefetches, and a continuation
central ScanScheduler
  -> admit I/O by phase, priority, dedupe key, and byte budget
ArrayRef per morsel
```

This is the clearest of the existing models about the difference between an immutable physical
plan and a runtime instantiation.

## `ScanPlan`

`ScanPlan` is an immutable physical node with a dtype and row domain. Its responsibilities include:

- creating or reusing per-scan state;
- pushing an expression into the plan's row domain;
- preparing a value-read route;
- preparing split discovery;
- preparing exact or candidate predicate evidence;
- preparing aggregate partials; and
- exposing metadata statistics.

Layouts construct plans through a vtable hook rather than a central switch. That makes plan
lowering extensible to registered layout implementations.

Runtime state is keyed by plan identity in a scan state cache. Prepared handles bind a fixed route
through the plan to that state without making the `ScanPlanRef` mutable.

## Selection and demand

`layout27` introduces a useful distinction through `RowScope`:

```rust
pub struct RowScope<'a> {
    pub selection: &'a Mask,
    pub demand: &'a Mask,
}
```

Both masks use the same dense row coordinates, and `demand` must be a subset of `selection`.

- `selection` identifies rows still semantically live.
- `demand` identifies rows whose values the current operation actually needs.

This allows a layout to choose between compact reads and dense reads with sparse downstream
materialization. It also provides a better vocabulary for predicates, projections, and lookup
operators than one overloaded mask.

## Prepared reads and continuations

A `PreparedRead` represents a fixed, reusable read route. For one range and owned row scope it
creates a morsel-level `ReadTask`:

```rust
fn create_task(
    self: Arc<Self>,
    range: Range<u64>,
    rows: OwnedRowScope,
    phase: ScanIoPhase,
) -> VortexResult<Box<dyn ReadTask>>;
```

Converting a task into a step exposes:

- required reads that must complete before computation;
- prefetch reads that may run speculatively; and
- a continuation that returns either the final array or another `ReadTask`.

The `Continue` result lets a complex layout reveal dependencies incrementally. For example, one
step can read codes or offsets and a later step can formulate the value or element read that those
buffers imply.

## Scheduler-visible I/O

Every logical read has an opaque deduplication key, estimated bytes, phase, priority, and
cancellation group. The scheduler maintains a scan-wide resolved-read store and admits work under a
logical read-byte budget.

Tasks are assigned to lanes:

- scan-wide evidence;
- morsel evidence;
- residual predicate evaluation;
- projection; and
- aggregate work.

This makes policy explicit. Unlike V1 and current plan v2, the scheduler does not have to infer I/O
intent from the order in which futures happen to be constructed.

## Evidence and residual work

Prepared evidence can describe ranges that are proven true, proven false, or still candidates.
The scan combines evidence fragments and schedules exact predicate reads only for residual demand.
Evidence can be scan-scoped or morsel-scoped, and dynamic predicates can trigger rechecks before
projection.

This separates cheap metadata reasoning from exact value evaluation without pretending that a
metadata proof is itself a projected boolean array.

## Fixed morsels

Despite the more explicit scheduler, the data source still divides a scan into fixed row morsels.
Each prepared read task receives one exact range and returns the array for that request. Natural
split hints inform the morsel plan, and a completion frontier releases state behind finished rows.

The scheduler controls which morsel step runs next, but a child does not independently choose a
shorter output prefix. Parent alignment therefore remains implicit in the shared morsel boundary.

## Actual path at the `ji/layout27` tip

Commit `9734b85de4` is titled `Use ScanPlan planning with LayoutReader execution`. Its normal scan
path is intentionally hybrid:

```text
LayoutRef
  -> construct both LayoutReaderRef and ScanPlanRef
ScanBuilder
  -> use ScanPlan::try_push_expr for projection and predicates
ExpressionBoundLayoutReader
  -> retain the pushed plan as _plan
  -> delegate pruning/filter/projection to the V1 LayoutReader
bound_split_exec
  -> mirror V1 split execution
```

`ScanPlanLayoutReader` pairs the V1 reader with the plan. Split discovery still delegates to V1.
`ExpressionBoundLayoutReader` removes expressions from the execution method signatures, but its
methods call the underlying V1 reader with the stored expression. The pushed `_plan` proves that
planning succeeded but does not execute the prepared-read graph on this path.

Consequently, results from that branch tip demonstrate expression binding plus V1 execution. They
do not by themselves validate the full prepared-task scheduler as the ordinary file-scan path.

## Strengths

- Immutable planning and mutable runtime state have a clear boundary.
- Per-layout plan construction is extensible.
- Expressions cross the planning boundary once rather than at every split call.
- Selection and demand are modeled separately.
- Multi-step reads support data-dependent dependencies such as offsets and dictionary codes.
- Read cost, phase, priority, deduplication, cancellation, and prefetch are explicit.
- Evidence, residual predicates, projection, and aggregation share one scheduling vocabulary.
- A release frontier bounds retained state for ordered progress.

## Limitations

- The API surface and scheduler state machine are substantially more complex than V1 or plan v2.
- Fixed morsels still dictate returned array size.
- Layout plan implementations contain significant task-construction machinery.
- Correctness spans plan pushdown, prepared state, evidence combination, scheduler lanes, and task
  continuations, increasing the verification burden.
- The hybrid tip retains two trees and uses the V1 tree for actual value execution.
- Evaluating the branch without distinguishing the hybrid path can overstate how much of the new
  executor is exercised end to end.

## Lessons for the proposed model

The following ideas should be retained:

- layout-vtable lowering into a generic plan;
- a per-scan state cache separate from the plan;
- selection versus demand;
- prepared, scheduler-visible logical reads;
- required and speculative read sets;
- continuation-based data-dependent I/O;
- evidence as a side channel; and
- byte-budgeted admission and release frontiers.

The fixed-morsel result contract should be replaced with prefix-progress batches, and the prepared
runtime should be opened as an explicit execution-node tree. That lets a parent combine children
whose natural boundaries differ without making the root preselect every internal batch boundary.

## Branch implementation map

These paths are branch-qualified because they do not all exist on the current branch:

- Plan, row scope, prepared reads, and read tasks: `ji/layout27:vortex-scan/src/plan/mod.rs`
- Scheduler-visible read requests: `ji/layout27:vortex-scan/src/read.rs`
- Task lanes and dependencies: `ji/layout27:vortex-scan/src/task.rs`
- Scheduler and byte admission: `ji/layout27:vortex-scan/src/scheduler.rs`
- Scan data-source orchestration: `ji/layout27:vortex-scan/src/plan/data_source.rs`
- Layout-specific plan implementations: `ji/layout27:vortex-layout/src/scan/v2/layouts/`
- Hybrid reader wrappers: `ji/layout27:vortex-layout/src/reader.rs`
- Hybrid plan binding: `ji/layout27:vortex-layout/src/scan/scan_builder.rs`
- Bound V1 split execution: `ji/layout27:vortex-layout/src/scan/tasks.rs`
