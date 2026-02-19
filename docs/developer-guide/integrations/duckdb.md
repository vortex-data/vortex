# DuckDB

The `vortex-duckdb` crate integrates Vortex as a DuckDB extension, registering table functions
(`vortex_scan` and `read_vortex`) that allow DuckDB to scan Vortex files directly.

## Registration

Vortex registers itself as a DuckDB table function. The function accepts a file path or glob
pattern as its argument. During the bind phase, the first matching file is opened to extract the
schema and register result columns with DuckDB's planner. The remaining files are opened lazily
during execution.

## Multiple Files

Glob patterns are expanded at bind time to produce a list of file paths. Files are opened
concurrently during the global initialization phase, with a concurrency limit proportional to
the number of DuckDB worker threads to keep the I/O pipeline saturated without overwhelming
the system.

A `MultiScan` stream manages the set of active file scans. It prioritises completing in-progress
scans before opening new files, ensuring that DuckDB's execution threads always have data to
consume while background I/O proceeds. File footers are cached to avoid redundant parsing when
the same file appears in multiple queries.

## Threading Model

DuckDB uses a thread-per-core execution model where each worker thread pulls data from a shared
scan state. The Vortex integration bridges this with a single `CurrentThreadRuntime` instance
that all async work is dispatched through.

The global initialization phase creates an async stream of scan results and wraps it in a
`ThreadSafeIterator` backed by a multi-producer multi-consumer channel. Each DuckDB worker
thread calls into the iterator to pull the next batch, and the CRT's smol executor drives I/O
progress from whichever thread happens to be polling. This allows DuckDB to retain its
thread-per-core scheduling while Vortex handles async I/O transparently.

Because DuckDB threads block while waiting for data, the CRT may need background worker threads
to ensure I/O continues to make progress when all DuckDB threads are busy exporting data. See
the [runtime documentation](../internals/async-runtime.md) for more on this trade-off.

## Filter and Projection Pushdown

DuckDB's planner pushes filter predicates into the scan via the `pushdown_complex_filter`
callback. These are converted from DuckDB's bound expression representation into Vortex
expressions and stored alongside any table filter expressions. During scanning, the combined
filter is applied to the `ScanBuilder` for each file.

Files can be pruned entirely before opening if their statistics prove that no rows can match
the filter.

Projection pushdown maps DuckDB's requested column indices to Vortex field names and passes
them as a projection expression to the scan.

## Data Export

Scan results are exported to DuckDB's native `DataChunk` and `Vector` format. The exporter
walks the Vortex array tree and uses encoding-aware exporters where possible -- for example,
constant arrays, run-end arrays, and dictionary arrays can be exported without first
decompressing to a canonical form. Arrays that lack a specialized exporter fall back to
canonical (Arrow-compatible) conversion before export.

Results are exported in chunks matching DuckDB's standard vector size to align with its
vectorized execution model.

## Future Work

The current integration builds directly on the `ScanBuilder`, layout reader, and file APIs.
Future work will migrate it to use the [Scan API](/concepts/scanning) `Source` trait, unifying
file discovery, multi-file coordination, and pushdown behind a single interface shared across
all engine integrations.
