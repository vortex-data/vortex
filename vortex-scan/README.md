# vortex-scan

A high-performance scanning and (non-shuffling) query execution engine for the Vortex columnar format, featuring
work-stealing parallelism and exhaustively tested concurrent execution.

## Overview

The `vortex-scan` crate provides efficient scanning operations over Vortex arrays with support for:

- **Projection pushdown** - Only read the columns you need
- **Filter predicates** - Push filters down to the storage layer
- **Row selection** - Efficiently skip unwanted rows
- **Multi-threaded execution** - Work-stealing parallelism for CPU-bound operations
- **Async I/O** - Tokio-based async execution for I/O operations
- **Arrow integration** - Seamless conversion to Apache Arrow format

## Features

### Core Capabilities

- **ScanBuilder API**: Fluent interface for constructing scan operations
- **Flexible Execution**: Single-threaded, multi-threaded, and async execution modes
- **Row Filtering**: Support for complex boolean expressions and dynamic filters
- **Selection Modes**: Include/exclude by index or using Roaring bitmaps
- **Split Strategies**: Split scans by row count or file size for parallel processing

### Performance Features

- **Work Stealing**: Efficient work distribution across threads
- **Zero-Copy Operations**: Minimize memory allocations and copies
- **Pruning Evaluation**: Skip reading data that won't match filters
- **Concurrent Iteration**: Multiple threads can process results simultaneously

## Usage

### Basic Scan

```rust
use vortex_scan::ScanBuilder;
use vortex_array::expr::lit;

// Create a scan that reads specific columns with a filter
let scan = ScanBuilder::new(layout_reader)
.with_projection(select(["name", "age"]))
.with_filter(column("age").gt(lit(18)))
.build() ?;

// Execute the scan
for batch in scan.into_array_iter() ? {
let batch = batch ?;
// Process batch...
}
```

### Multi-threaded Execution

```rust
// Execute scan across multiple threads
let scan = ScanBuilder::new(layout_reader)
.with_projection(projection)
.with_filter(filter)
.into_array_iter_multithread() ?;

for batch in scan {
let batch = batch ?;
// Results are automatically collected from worker threads
}
```

### Arrow Integration

```rust
use arrow_array::RecordBatch;

// Convert scan results to Arrow RecordBatches
let reader = ScanBuilder::new(layout_reader)
.with_filter(filter)
.into_record_batch_reader(arrow_schema) ?;

for batch in reader {
let record_batch: RecordBatch = batch ?;
// Process Arrow RecordBatch...
}
```

### Row Selection

```rust
use vortex_scan::Selection;

// Select specific rows by index
let scan = ScanBuilder::new(layout_reader)
.with_selection(Selection::IncludeByIndex(indices.into()))
.build() ?;

// Or use row ranges
let scan = ScanBuilder::new(layout_reader)
.with_row_range(1000..2000)
.build() ?;
```

## Architecture

### Work-Stealing Queue

The crate implements a sophisticated work-stealing queue that allows multiple worker threads to efficiently share work:

- **Dynamic Task Addition**: Tasks can be added while processing is ongoing
- **Fair Work Distribution**: Threads steal work from each other to balance load
- **Lock-Free Operations**: Uses crossbeam's deque for efficient concurrent access

### Filter Optimization

Filters are automatically optimized using:

- **Conjunct Reordering**: Most selective filters are evaluated first
- **Dynamic Statistics**: Filter selectivity is tracked and used for optimization
- **Pruning Pushdown**: Filters are pushed to the storage layer when possible

### Memory Safety

All concurrent code has been verified using:

- **Loom Testing**: Exhaustive verification of all possible thread interleavings
- **Address Sanitizer**: Memory safety verification in CI
- **Debug Assertions**: Runtime checks for invariants in debug builds

## Testing

### Unit Tests

Run the standard test suite:

```bash
cargo test -p vortex-scan --all-features
```

### Loom Concurrency Tests

The crate includes comprehensive Loom tests that exhaustively verify concurrent behavior.
These tests run by default but can be disabled if need be:

```rust
# Skip Loom tests when using incompatible tools like address sanitizer
RUSTFLAGS="--cfg disable_loom" cargo test -p vortex-scan 
```

Loom tests verify:

- Memory ordering correctness in the work-stealing queue
- Absence of data races in filter expression evaluation
- Proper synchronization in concurrent task factories
- Thread termination conditions and cleanup

## Performance Considerations

### Concurrency Level

The default concurrency level is 2, meaning each worker thread can have 2 tasks in flight. This can be adjusted:

```rust
let scan = ScanBuilder::new(layout_reader)
.with_concurrency(4)  // Increase for more I/O parallelism
.build() ?;
```

### Buffer Sizes

The multi-threaded executor uses buffering based on the formula:

```rust
buffer_size = num_workers * concurrency
```

This controls how many splits are processed concurrently.

### Memory Usage

- **Streaming Processing**: Results are streamed rather than materialized
- **Bounded Buffers**: Memory usage is bounded by the concurrency level
- **Lazy Evaluation**: Computation is deferred until results are consumed

## Dependencies

Core dependencies:

- `vortex-array`: Core array types and operations (includes expression evaluation framework)
- `vortex-layout`: Layout reader abstraction
- `futures`: Async runtime abstractions
- `tokio` (optional): Multi-threaded async runtime
- `arrow-array` (optional): Arrow integration

## Feature Flags

- `default`: Standard features for most use cases
- `tokio`: Enable multi-threaded execution with Tokio runtime
- `roaring`: Support for Roaring bitmap selections
