# Scan API

:::{note}
The Scan API is on the roadmap and under active development. The core `Source` trait and scan pipeline
are functional, but the full API surface is not yet fully defined or implemented.
:::

The Vortex Scan API defines a standard interface between data storage and query engines. It solves the
N x M problem of having N different storage backends and M different query engines by providing a common
interface that both sides can implement against.

```
    Storage                                  Query Engines
    ───────                                  ─────────────

    Vortex Files   ──► ┌──────────────┐ ──►  DuckDB
    Parquet Files  ──► │   Scan API   │ ──►  DataFusion
    Iceberg Tables ──► └──────────────┘ ──►  Spark
```

Storage backends implement the `Source` trait for reads. Query engines issue a scan request
describing the filter and projection to push down, and the source returns a stream of
independently-executable splits that can be run concurrently to produce result arrays. An
equivalent `Sink` trait exists for the write path, accepting an array stream and writing it to
the underlying storage.

## Motivation

Traditional data integrations require each storage backend and query engine to agree on a common
interchange format, typically Apache Arrow. This means the storage backend must fully decompress its
data into Arrow arrays, even if the query engine could operate on the compressed representation
directly.

The Vortex Scan API avoids this by allowing data to flow between storage and query engines in its
native compressed encoding. For example, the DuckDB integration can receive FSST-encoded string
arrays directly from a Vortex file and pass them into DuckDB's own internal FSST format without
any decompression step.

## Source

A **Source** represents any scannable tabular data. It accepts a scan request (filter, projection,
limit) and returns a stream of independently-executable splits. An equivalent **Sink** interface
exists for the write path, allowing query engines to both read from and write to any storage
backend through a single pair of interfaces.

### Splits

A source produces splits, each representing an independent unit of work that can be executed in
parallel. A split typically corresponds to a range of rows in a layout, such as a chunk or a set
of row-group partitions.

Each split carries size and row count estimates that query engines use for scheduling decisions.
Splits can also be serialized for distributed execution across remote workers.

### Remote Sources

A source may front remote storage rather than local files. In this case, the split's execution
issues a remote call and receives the result over the network. The
[Vortex IPC format](../specs/ipc-format.md) can be used as the wire protocol for these calls, allowing
compressed arrays to be transferred without decompression. This gives remote sources the same
zero-decompression benefits as local scans -- the data stays in its compressed encoding end-to-end,
from remote storage through the network and into the query engine.

## Filter Pushdown

Filter expressions are decomposed into individual conjuncts (AND-separated terms) and evaluated
independently. The scan tracks the selectivity of each conjunct using a probabilistic sketch
and dynamically reorders them so that the most selective predicates are evaluated first. This
means that as a scan progresses, it learns the most efficient evaluation order for the filter.

Filters are evaluated in two stages. First, pruning evaluation uses statistics stored in
`ZonedLayout` (such as per-zone min/max values) to eliminate entire regions without reading any
data. Second, filter evaluation materializes only the filter-referenced columns and computes a
row mask of matching rows.

## Projection Pushdown

Projection expressions describe the output schema of the scan. The scan analyzes the projection
and filter expressions to compute two field masks: which columns are needed for filtering, and
which are needed for the final output. Only the union of these columns is read from storage.

Columns needed exclusively for filtering are discarded after the filter mask is computed, so they
never appear in the output stream. This separation ensures minimal data movement throughout the
pipeline.

## Query Engine Integration

Query engines integrate with the Scan API by translating their internal plan representations into
scan requests and consuming the resulting array stream in their preferred format. Integrations
exist for DuckDB, DataFusion, Spark, and Trino, with each engine converting its native filter
and projection representations into Vortex [expressions](expressions.md).

