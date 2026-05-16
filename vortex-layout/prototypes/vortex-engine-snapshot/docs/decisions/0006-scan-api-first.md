# 0006: Build a single-pass Scan API engine before shuffle

> **Status:** Accepted ADR.
> **Progress:** Scan-shaped, single-pass, single-node engine is the
> active scope. Shuffle and distributed execution are deferred until
> the local single-pass model is validated.
> **Open questions:** the exact exchange operator boundary that a
> future shuffle layer would attach to.

## Status

Accepted.

## Context

The engine serves data-loading workloads that are ordered, nested,
source-local, byte-sensitive, and often dominated by expensive
payload reads or effectful functions that should be skipped when
cheap metadata, predicates, or runtime filters prove they are
unnecessary.

It must also remain embeddable inside other data systems. A host
such as DataFusion, DuckDB, Velox, or an application-specific
runtime may still own broad SQL planning, repartitioning,
distributed execution, unordered joins, and unordered aggregation.

## Decision

Build the engine as an ordered, Vortex-native, single-pass
single-node engine.

The engine performs work in one pass over each ordered input,
while allowing staged reads of metadata, keys, offsets, codes,
stats, and payloads. Operators use domains, the lowering API,
pipelines, barriers, resources, and typed runtime filters.

The engine optimises for:

- deeply nested `struct`, `list`, and `list<struct>` schemas;
- zip joins over aligned fields;
- parent-child prefix relations and parent-field broadcast;
- pre-sorted merge joins where order is proven;
- ordered streaming reductions over existing domains;
- async and effectful data-loading functions represented as
  operators;
- functions with large input/output byte-size changes.

Shuffle, repartitioning, unordered hash aggregation, non-sorted
joins, global sorts, and distributed execution are deferred. When
required, they will be introduced as explicit exchange operators
that terminate one local fragment and open another.

## Consequences

The engine is not a general embedded analytical engine. Plans that
need arbitrary reorder, broad SQL coverage, or large unordered
state must use a host engine or wait for exchange-backed
operators.

The implementation plan must prove the Scan API bet with
measurements. Worked examples should track skipped payload rows,
payload bytes, effectful calls, runtime filter publication and
consumption, and resource updates.

The physical plan API can stay local-fragment-first while
preserving a future `fragments` wrapper. Adding exchange later
extends fragment boundaries without changing the core local task
model.

Vortex integration treats layout scanning as engine execution.
Layout metadata remains declarative source-local planning data;
the bound layout tree becomes executable engine source operators.

## Alternatives considered

Building broad relational coverage first was rejected because it
would force shuffle, hash joins, unordered aggregation, and global
sort into the critical path before proving the data-loading
workload.

Delegating all execution to an existing embedded engine was
rejected because the engine needs Vortex-native lazy arrays,
compressed intermediates, source-owned external work, and the
runtime-filter mechanism as first-class execution data.

Implementing shuffle immediately was rejected because it expands
scheduling, channel ownership, memory accounting, and distribution
scope before the local single-pass model is validated.
