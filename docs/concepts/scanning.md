# Scanning

Vortex scans are built around the layout tree stored in a file footer. A scan opens the file,
deserializes the root layout, expands that layout into a `ScanPlan` tree, and prepares executable
runtime handles for predicates, projections, statistics, and aggregates.

The query engine sees a standard scan request: a projection, an optional filter, ordering
requirements, limits, and split preferences. The layout and scan layers decide how to satisfy that
request with the least data movement.

```text
footer layout bytes
        |
        v
LayoutRef / Layout<V>
        |
        v
ScanPlan tree
        |
        +-- push expressions into layout-local row domains
        +-- prepare predicate evidence
        +-- prepare residual predicate reads
        +-- prepare projection reads
        +-- prepare statistics and aggregate answers
        |
        v
morsel execution -> array batches
```

## Layout Expansion

Each layout encoding has a layout vtable. The serialized form stores common fields such as dtype,
row count, child layouts, and segment IDs. Deserialization hoists those common fields into
`Layout<V>` and leaves only layout-specific metadata in `V::LayoutData`.

The layout vtable's scan hook expands a `Layout<V>` into a `ScanPlan`. This keeps serialized layout
concerns separate from runtime execution: layouts describe the physical organization of data,
whereas `ScanPlan`s expose what that organization can do during a scan.

Layout children are lazy. Accessing a child validates the dtype expected by the parent and
materializes that child from the same footer FlatBuffer only when a scan route actually needs it.
For example, a struct layout does not need to deserialize every column child when the query reads
only a few fields.

## Scan Plans

A `ScanPlan` is an immutable runtime view of a layout. It can:

- push an expression into the plan's row domain;
- prepare value reads for the plan's root value;
- prepare predicate evidence;
- provide natural split hints;
- answer statistics or partial aggregates from metadata; and
- release cached state behind a completed row frontier.

Pushing an expression returns another `ScanPlan` whose `root()` value is that expression. A struct
plan can route `field("a")` to the child for column `a`; a dictionary plan can apply some
expressions once over dictionary values and reuse the result with per-row codes; a generic apply
plan handles expressions that cannot be pushed into a specialized layout.

## Prepared Runtime Handles

Planning a scan creates prepared handles from the `ScanPlan` tree:

- `PreparedEvidence` produces evidence fragments for one predicate expression.
- `PreparedRead` reads one pushed projection or residual predicate expression.
- `PreparedStats` and `PreparedAggregate` answer metadata-backed statistics and aggregates.
- `PreparedSplit` reports row ranges that are natural units of scan work.

Prepared handles are scan-level runtime objects. They can hold child prepared handles and shared
state, but they do not choose the next row range themselves. The scan driver chooses explicit
morsel ranges and asks prepared handles to work on those ranges.

Each morsel carries a `RowScope`:

- `selection` says which rows in the requested range remain live.
- `demand` says which selected rows need meaningful values from this operation.

This lets a projection skip data that no longer affects output, while still preserving output
cardinality for selected rows.

## Predicate Evidence

Predicates are decomposed into independent expressions. Before reading row data for a predicate,
the scan asks available prepared evidence handles whether metadata can prove something about the
requested rows.

Evidence is a statement over the row domain. A zone map can prove that a range cannot match a
predicate; file or layout statistics can prove that a predicate is already satisfied; other
evidence sources can leave a range unknown. Unknown rows continue to residual predicate reads,
which materialize only the columns needed to compute the predicate exactly.

Prepared evidence handles are expected to be cheap relative to projection reads. They should use
layout metadata, statistics, indexes, or already-prepared shared state rather than speculatively
reading large data columns. Cheap evidence can also opt into a final `recheck_before_projection`
pass, which is useful when dynamic filters change while a morsel is in flight.

## Projection Pushdown

Projection pushdown is expression pushdown through the `ScanPlan` tree. The scan prepares reads only
for the requested output expressions, and each layout decides how much of its child tree those reads
need.

For a struct layout, field access routes to the named child and avoids unrelated columns. For a
chunked layout, the read is sliced by chunk and only overlapping chunks with demanded rows are
visited. For a dictionary layout, values can be shared across row ranges while codes are read for
the requested rows.

## State and Caches

The scan path uses several layers of state:

- The segment source owns physical I/O, coalescing, segment caching, and in-flight segment
  deduplication.
- The expanded `ScanPlan` tree is immutable and safe to share.
- `PrepareCtx` owns a prepared-state cache for scan/file-level state shared by prepared reads,
  evidence, aggregate, and stats handles.
- A layout plan can create child-local prepared-state caches so repeated pushes into the same child
  share decoded dictionaries, zone tables, or other expensive setup without leaking state across
  unrelated row domains.
- Morsel tasks own only the row range and masks needed for that operation.

When ordered scans advance, prepared reads and scan plans receive a release frontier. Layouts use
that frontier to drop caches that only cover rows that cannot be read again.

## Query Engine Integration

Query engines translate their native expressions into Vortex expressions and submit a scan request.
Vortex handles layout expansion, evidence, residual predicates, projection reads, and array
production. Integrations then export the produced Vortex arrays to the engine's preferred batch
format, such as Arrow `RecordBatch`es for DataFusion or DuckDB `DataChunk`s for DuckDB.
