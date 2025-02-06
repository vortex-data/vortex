# Benchmarks

There are a number of benchmarks in this repository that can be run using the `cargo bench` command. These behave more
or less how you'd expect.

There are also some binaries that are not run by default, but produce some reporting artifacts that can be useful for comparing vortex compression to parquet and debugging vortex compression performance. These are:

### `compress.rs`

This binary compresses a file using vortex compression and writes the compressed file to disk where it can be examined or used for other operations.

### `comparison.rs`

This binary compresses a dataset using vortex compression and parquet, taking some stats on the compression performance of each run, and writes out these stats to a csv.
    * This csv can then be loaded into duckdb and analyzed with the included comparison.sql script.

### `tpch_benchmark.rs`

This binary will run TPC-H query 1 using DataFusion, comparing the Vortex in-memory provider against Arrow and CSV.

For profiling, you can open in Instruments using the following invocation:

```
cargo instruments -p bench-vortex --bin tpch_benchmark --template Time --profile bench
```

## Memory allocators

If you don't want to use the default system allocator, there are `"jemalloc"` and `"mimalloc"` features available that configure a different allocators at compile time.

As of this writing, if both are enabled `mimalloc` will be used.

## Common Issues

If the benchmarks fail because of this error:

```
Failed to compress to parquet: No such file or directory (os error 2)
```

You likely do not have duckdb installed. On macOS, try this:

```
brew install duckdb
```
