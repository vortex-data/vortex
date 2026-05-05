# Vortex Roadmap

Synthesized from open issues and discussions on `vortex-data/vortex`, anchored on
the maintainers' own roadmap (discussion #6089) and the H1 2026 planning doc
(discussion #6456).

## GPU

Push CUDA-backed scan to production: faster data loading, more kernels, and JIT
pipelines so GPU paths beat CPU on the targeted queries.

- Epic #7712 — GPU Data Loading
- #6535 — Vortex CUDA support
- #6538 — CUDA backlog
- #6937 — Panic in stats computation on GPU arrays
- #6656 — CUDA-dyn dispatch: reduce generated assembly
- D#6240 — tracker: CUDA backed vortex-scan

## Variant

Add a Variant logical type with canonical array, zero-copy interop with Arrow
Parquet-Variant, and a `VariantGet` expression for projection.

- Epic #7717 — Variant Type and Array
- D — Support for `DType::Variant`

## Other Arrow types (Union, …)

Round out the logical type system with the remaining missing types (Union,
Interval, BF16, Map, FixedSizeBinary, DateTimeParts) under the extension-type
machinery.

- Epic #7683 — Extension Types (umbrella)
- #7705 — Add `Union` to `DType`
- #2969 — Interval DTypes
- #1734 — BF16 DType
- #6543 — JSON extension type
- #6540 — More Extension Types
- D — Add a `Map` DType
- D#6456 — DateTimeParts (H1 plan)

## Vector support

Build on the existing Vector extension type and similarity scan to add vector
indexes, ANN, and top-k search so Vortex is a viable backbone for vector
workloads.

- Epic #7704 — Vector Similarity Search
- #6865 — Tracking Issue: Tensor Extension Types
- #6854 — Tracking Issue: `Uuid` Extension Type

## DataFusion integration

Close the remaining DataFusion gaps: dynamic filter expressions, partition
reporting, casting compatibility, and richer scan metrics.

- Epic #2254 — Apache DataFusion Integration
- #4034 — Datafusion dynamic filter expressions
- #1505 — Report splits to DataFusion as partitions
- #5144 — `vortex_metadata` table UDF
- #5296 — Custom partitioning / simplifier rules
- #4322 — Substrait expression support
- #5912 — Vortex Substrait Rust
- D — DataFusion casting semantics incompatibilities
- D#5851 — More metrics to `DataSourceExec`

## DuckDB integration

Reach Parquet performance parity in DuckDB and ship the missing pushdown /
partitioning / object-store features.

- Epic #7716 — DuckDB performance parity with Parquet
- #7746 — Tracking Issue: hive partitioning in DuckDB
- #7734 — prefix/suffix/substring + `contains()` pushdown
- #3963, #3964, #3393 — table_function pushdown / type / partition stats
- #4106 — GCS & ADLSv2
- #4281 — q85 bad query plan
- #3897 — work-stealing exporter
- #4750, #5176, #6706 — perf regressions
- #5491 — build on more targets
- #6820 — tracing subscriber
- #4809, #6038 — exporter bugs

## Further integrations

Grow the language and engine ecosystem (Trino, Spark, Polars, ClickHouse, Arrow
Flight, Hugging Face, JS, HDFS/JFS) under one umbrella.

- Epic #7714 — Integrations
- #2599 — Trino Connector
- #7725 — Configure compression options from Spark
- #5135 — Arrow Flight gRPC
- #5379 — Hugging Face Datasets API
- D — Polars `pl.scan_vortex("dir/")`
- D#6425 — ClickHouse vortex-clickhouse crate
- D — HDFS / JFS support
- D — Javascript API
- D — Spark 4 support
- Bindings (separate workstream): D#7204 Java `writeBatchFfi`, D#7635 Python 3.10
  wheel, #7737 `cargo test -p vortex-python`, #5913 PyArrow Dataset full filter,
  D#7562 `filter+limit` ScanBuilder, D#7111 C++ examples

## Layout API design

Land the stream-based Reader v2 / Layout V2 with a default strategy that handles
small segments, constant columns, lists, and embedded indexes (bloom, text,
FST).

- Epic #7732 — Layout rework
- #6546 — Layout V2
- #6539 — New Scan API (ongoing)
- #3853 — Default strategy for <8k row segments
- #3874 — Default strategy to detect constant columns
- #3538, #3953 — Dict layout pushdown / short-circuit
- #3442 — `vortex-layout` filter_evaluation bug
- #6162 — Nullable struct layouts unsound
- D#6089 — ListLayout / VarBinLayout, bloom / text / inverted / FST indexes
- D — Zone pruning for LIKE filters

## Sub-segment reads

Allow readers to fetch arbitrary byte ranges of a flat layout instead of always
reading whole segments, cutting read amplification on `take`.

- D#6991 — Sub-Segment Range Read for FlatLayout
- D — Sub-segment reads for uncompressed flat layouts to reduce read
  amplification during take

## Finish lazy compute migration

Complete the move to the deferred-iterative execute model, deprecate the
canonicalize path, and add CSE + kernel selection so the new model also wins on
perf.

- Epic #7674 — Lazy & Iterative Execution
- Epic #6533 — Compute Overhaul
- #6258 — Move over to execute from to_canonical
- #5978 — Optimize `Mask::rank()`
- #5677 — Generalize SIMD `take` to `Copy`
- #4669 — Reduce round-trips in scan logic
- #4090 — Optimize round-trips for expression evaluation
- #1943 — Chunked compute shouldn't preserve chunks during scan

### Perf

- Same epics; perf-specific bugs: #5861 (small-batch writes), #4784 (FSST
  take), #5025 (scan memory), #4750/#5176 (DuckDB regressions).

## Forward compatibility

Stabilise the Vortex C ABI for arrays and expressions so external interpreters
(and a WASM-targeted file interp) can be built against it.

- Epic #7735 — Stabilise Vortex ABI

### Stable ABI

- #7483 — null-ptr / error propagation in C FFI
- #7248 — FFI: allow taking ownership of memory
- #3028 — FFI API audit
- #5109 — Mapping arrow arrays via FFI instead of arrow-rs
- #7324 — Memory leak while cloning session in FFI runtime

### WASM at points

- Epic #7735 calls out WASM-based file interp as the consumer of the stable ABI.

## Benchmark stability

Consolidate the benchmark tooling so SQL benchmarks are reliable, comparable,
and cheap to run in CI and locally.

- Epic #7718 — Benchmarks
- #4935 — Make benchmark targets consistent
- #6159 — Operators execution benchmarking
- #4130 — Add RealNest benchmark
- #3357 — Add LST Bench
- D#7066 — Bulk column decompress benchmarks
- D#6630 — Efficiency Heuristics
- D — Move benchmark orchestration from GH Actions to a bot
- D — Which Parquet version and settings in benchmarks?

## Extensible stats and indices

Replace the fixed Stat enum with pluggable AggregateFn-partial stats plus
pruning / falsify-verify expressions, then layer index structures (bloom, text,
inverted, FST) on top.

- Epic #7707 — Stats and AggregateFns
- #7235 — Non-string min/max truncation marker in zone map
- #6389 — File stats on nested fields
- #913 — Cardinality estimate stat
- #4581 — Too many stats
- #2683 — Comparisons over is_sorted should binary search
- #2684 — Pruning table to also return always-true
- #1440 — Shortcircuit compare operations using stats
- D#6089 — Embedded indexes (bloom, text, inverted, FST)
- D — Zone pruning for LIKE filters

## Improve list array compute and layout

Make list types first-class: stream-friendly list layout, consistent List /
ListView kernels, list-manipulation expressions, and higher-order functions
over lists.

- Epic #7679 — Improved List support
- #4842 — IsSortedKernel / MinMaxKernel for List, ListView, FixedSizeList
- #4889 — Better data layout for nested / repeated schemas
- #4914 — PCO stats mismatch causes ListArray offsets validation failure
- #4302 — VarBin take errors for >32-bit offsets
- #3859 — Push struct validity into children

### Higher-order functions

Lambdas / `var` expressions so users can express map / filter / reduce inside
Vortex expressions.

- D#7334 — Lambda / var expression to impl HoF

### List expr

A vocabulary of list-manipulation expressions (slice, contains, length, etc.)
on top of HoF.

- D — Add list manipulation expressions

### Fix up compute for List and ListView

Tidy and unify the List vs ListView kernels and conversions so picking one no
longer leaves capability gaps.

- #4978 — Clean up `list_contains` kernel for ListView
- #4987 — Switch preferred Arrow list encoding to ListView
- #5184 — ListView → List optimizations
- D#6470 — ListView Handling normalization
- D#6983 — FixedSizeList to use ListView

## Push-based writer

Move from the current pull/blocking writer to an explicit-flush, push-based API
that handles small batches, large values, and remote object stores efficiently.

- #4926 — File writer explicit flush interface
- #2921 — File write to ObjectStoreWriter explicit flush
- #4637 — Remove blocking write API
- #7210 — Default write strategy is unexpected
- #5861 — Write performance for small Arrow RecordBatch 20× slower than Parquet
- #3799 — Support values bigger than 4GB
- #4500 — Verify checksum after writing to object store
- D — `[python api] write_path` to support remote paths
- D — Write Vortex on HDFS / JFS

## Type system soundness

Tighten the formal semantics of `DType` (decimals, struct field uniqueness,
nullability, casts) so the type system is internally consistent.

- Epic #7706 — Vortex Type System
- #5820 — `DecimalArray` logical / physical mismatch
- #6900 — `StructArray` field names should be unique
- #5103 — `DType::Null.as_nonnullable()` returns nullable
- #2031 — Expand acceptable casts
- #4148 — Fix decimal casting
- #3633 — Add DType structs + trait impls
- D#6702 — Cast `boolean` to `primitive` and `utf8`
- D — Support decimal arithmetic

### Ext VTable

Land the extension-type VTable so extension types pick up first-class compute,
stats, and IO behaviour.

- D#6456 — H1 plan: Ext VTable (q1)
- Epic #7683 — Extension Types

### Type of ext

Settle the semantics for casting and storage of extension types so each ext
type has a well-defined relationship to its storage type.

- #7504 — Semantics of casting extension types
- D#6500 — Extension Data Types

## File format stability

Define and enforce on-disk forward / backward compatibility, including release
process, IPC stabilisation, schema evolution, and shared dictionaries.

- D — Vortex Stability
- D — Backwards Compatibility Testing
- D — Release process for breaking API changes
- D — Formalize the process for defining known features in file format
- D — Panic when handle schema evolution in vortex file
- D — Support file metadata like Parquet
- D#6089 — IPC stabilization, shared dictionaries across columns, checksum stat
  for reused dictionaries
- #3083 — Checksums and Signatures
- #1884 — Modular Encryption

## Encodings & compression

Round out the compressor (FastLanes RLE / Delta / Dict, lossy numerics,
validity compression, PCodec) and fix the known correctness / perf bugs in
existing codecs.

- Epic #2894 — Compressor Improvements
- Tracking #7697 — Compressor Optimizations
- #4784 — `take` on FSST explodes memory
- #1987 — FSST written into VarBinView vs VarBin
- #919 — ALP exponent sampling improvements
- #7245 — TurboQuant rotation bias
- #7268 — Sampling somehow compresses into 0 bytes
- #5225 — Compress validity arrays
- #3481 — SparseScheme for Floats
- #1655 — Avoid bit-packing at small ratios
- #6171 — Better compressor tests
- D#7095 — REE to support more types
- D — zstd array v2

### Delta & RLE

Add the missing FastLanes Delta and RLE encodings as first-class options in the
sampling compressor.

- D#6456 — H1 plan: FastLanes RLE / Delta / Dict
- Epic #2894

### PCodec outline

Upgrade PCodec, fix its validity-cast / list-offset issues, and document when
the compressor should pick it.

- D — PCodec upgrade
- D — PCodec-inspired ideas
- #5196 — Implement cast for PCO with Array validity to NonNullable
- #4914 — PCO stats mismatch on ListArray offsets

## I/O subsystem

Build a proper I/O layer with coalescing, resource awareness, and a non-`pread`
interface that fits NVMe / EBS / object stores and DuckDB- / Polars-style
runtimes.

- D#6456 — Coalescing, extension point, resource-aware runs (H1 plan)
- D#6089 — Responsive coalescing / sticky connections
- #4659 — Vortex I/O abstractions
- #4658 — Experiment with thread-per-core runtime
- #4661 — Migrate language bindings to CurrentThreadRuntime
- #4669 — Reduce round-trips in scan logic
- #4090 — Optimize round-trips for expression evaluation
- #4822 — IO Limits
- #1459 — Call `check_signals` in scan to support signal handlers
- D#6647 — Buffer allocators

### Non-pread

Move off the implicit `pread`-only assumption so we can use streaming / push
reads where the backend supports them.

- Surfaced in D#6456 I/O thread; ties into Reader v2 (#6546).

## Correctness / fuzzer

Treat soundness as a workstream: kill the known UB / SIGSEGV / unsound-layout
bugs, expand fuzz coverage, and convert thread panics into errors.

- #4220 — UB in several crates that don't run miri
- #6221 — Occasional SIGSEGV under high concurrency
- #6765 — Zip array mask handling unsound
- #6162 — Nullable struct layouts unsound
- #1732 — Improve error reporting and stack tracing
- #6039 — Fill validity on array using the fuzzer
- #4851 — Fuzz test take and zip size
- #1588 — Fuzzer support for arbitrary Extension arrays
- #2558 — Check for circular dependencies in CI
- D — Thread panics instead of errors
- D — Filter push down is incorrect for fallible operations
- D — Fuzzer hook for constructing arbitrary compressed array
- D — Auto-resolution of fuzz issues

## Documentation update

Refresh the user- and contributor-facing docs to match the new array and
compute models, with examples and `deny(missing_docs)` enabled.

- #1905 — Enable `deny(missing_docs)`
- #3996 — Intro guide for Vortex
- #4924 — Simple examples
- #4816 — Document compute functions
- #3340 — Add feature-gate info to docs
- #1904 — Document conventions for vortex-expr functions
- #3885 — Rename `nbytes` and improve docs
- D — Blog showing high compression vs Apache Parquet
- D — Platform / language / framework / OS support matrix

### New array model

Document the post-VTable-unification array model end-to-end so external
implementers can write encodings against a stable surface.

- Epic #7735 — Stabilise Vortex ABI
- #6544 — Unify VTables across all concepts

### New compute model

Document the deferred / iterative execute model, kernel selection, and how to
plug in custom kernels.

- Epic #7674 — Lazy & Iterative Execution
- Epic #6533 — Compute Overhaul

---

## Sources

- 188 open issues on `vortex-data/vortex` (as of 2026-05-05).
- ~83 open discussions across 4 pages.
- Maintainers' roadmap: discussion #6089.
- H1 2026 planning: discussion #6456.

`D#NNNN` refers to a GitHub Discussion; bare `D` refers to a discussion whose
number was not exposed on the listing page.
