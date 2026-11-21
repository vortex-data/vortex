# Benchmarks

There are a number of benchmarks in this repository that can be run using the `cargo bench` command. These behave more
or less how you'd expect.

There are also some binaries that are not run by default, but produce some reporting artifacts that can be useful for
comparing vortex compression to parquet and debugging vortex compression performance. These are:

### `compress.rs`

This binary compresses a file using vortex compression and writes the compressed file to disk where it can be examined
or used for other operations.


### `query_bench`

This is the unified benchmark runner that supports multiple benchmark suites including TPC-H, ClickBench, and TPC-DS.

To run the TPC-H benchmarks you can use:

```bash
cargo run --bin query_bench -- tpch
```

To run the ClickBench benchmarks:

```bash
cargo run --bin query_bench -- clickbench
```

For profiling, you can open in Instruments using the following invocation:

```
cargo instruments -p vortex-bench --bin query_bench --template Time --profile bench -- tpch
```

### Data directory

There is a data directory at `vortex/vortex-bench/data` where parquet and vortex files used for the benchmark runs
can be found.

## Parquet compression

All benchmark Parquet files use zstd level 3. Set the level explicitly, because the defaults of
the common writers are different:

- parquet-rs: `ZstdLevel::default()` is level 1, not level 3. Use
  `Compression::ZSTD(ZstdLevel::try_new(3)?)`, or `parquet_writer_properties()` from
  `src/utils/parquet.rs`.
- DuckDB: `COPY` writes Snappy by default. Use
  `(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3)`.
- `tpchgen-cli`: writes Snappy by default. Use `parquet --compression='ZSTD(3)'`.

Benchmarks that compare Parquet codecs on purpose, such as the compress-bench `--gpu-parquet-codec`
option and the Snappy Parquet whose size compress-bench and TPC-DS at SF1 record
(`parquet-snappy`), are exempt.

## Parquet layout

All Parquet the benchmarks generate uses the same writer settings, `parquet_writer_properties` in
`src/utils/parquet.rs`: zstd level 3, 64 KiB dictionaries, a page index, and pages of at most 1 MiB
and about 20,000 rows. Only the row groups differ. TPC-H, TPC-DS, nested TPC-H, JOB, Polar Signals,
and SpatialBench write Parquet in row groups of 122,880 rows, DuckDB's default. In our tests, these
smaller row groups were faster from local disk, because they give the query engines more units of
work to run in parallel. The benchmarks that read from S3 use row groups of 1,048,576 rows,
parquet-rs's default, which need fewer requests to object storage. Pass `--opt
row-group-rows=1048576` to `data-gen` to generate data for S3. The file `parquet.layout`, beside the
`parquet` directory, records the layout of its files: the row groups, and a version of the other
writer settings. `data-gen` removes and generates again Parquet whose layout is different or not
recorded, and the formats converted from it. SpatialBench keeps its `zone` files, which come from
outside, and writes the rest again.

## Memory allocators

If you don't want to use the default system allocator, there are `"jemalloc"` and `"mimalloc"` features available that
configure a different allocators at compile time.

As of this writing, if both are enabled `mimalloc` will be used.

## Common Issues

If the benchmarks fail because of this error:

```
Failed to compress to parquet: No such file or directory (os error 2)
```

You likely do not have the required packages installed. On macOS, try this:

```
brew install duckdb cmake ninja pkg-config vcpkg
```
