# vortex-sqllogictest

This crate tests correctness of our DuckDB and DataFusion integrations. It runs
[.slt](https://github.com/risinglightdb/sqllogictest-rs) tests with our queries
as well as 0.1 scale of TPC-H, TPC-DS, 1 shard of partitioned ClickBench,
the first 1,000 variants of StatPopGen, and a 1% proportional sample of the
Join Order Benchmark (JOB) IMDB tables. For these query sets, tests also verify
each query's `EXPLAIN` result.

## Running tests

```sh
# For running our own queries
cargo test -p vortex-sqllogictest --test sqllogictests

# For running TPC-H, TPC-DS, ClickBench, StatPopGen, or JOB you need to generate data first
./slt/generate_data.sh
# You can generate data for a specific dataset
./slt/generate_data.sh tpcds
# Then you can run the tests
cargo test -p vortex-sqllogictest --test sqllogictests

# You can run only DuckDB tests...
cargo nextest run -p vortex-sqllogictest -E 'test(/slt::duckdb::/)'
# ... or a single test on both DuckDB and DataFusion
cargo nextest run -p vortex-sqllogictest -E 'test(strings)'

# You can also list all tests
cargo nextest list -p vortex-sqllogictest
```

## Writing tests

To run a test only for DuckDB or DataFusion, place test in `slt/duckdb` or
`slt/datafusion`. To run a query inside test only for DuckDB or DataFusion,
place `onlyif duckdb` or `onlyif datafusion` before query.

To match output by regex in DuckDB, use `<REGEX>:pattern` or `<!REGEX>:pattern`
in query result.

To regenerate .slt file to match current output, use `--complete`.

```sh
cargo test -p vortex-sqllogictest --test sqllogictests -- --complete
# Regenerate output for a specific test
cargo test -p vortex-sqllogictest --test sqllogictests -- --complete strings
```

## Encoding checks

Each `.check` file under `slt/` (by convention `slt/<suite>/filecheck/<table>.check`) checks
the array encodings of generated Vortex files with `vortex-filecheck`, so a compressor change that
re-encodes benchmark data fails here instead of only showing up as a benchmark swing. A check
file starts with one or more `files: <regex>` lines naming the files it applies to; each regex
must match a whole path relative to `slt/`, such as `tpch/data/nation\.vortex`. The
directives match each file's encoding-tree dump partially, like LLVM FileCheck, and
`units: <glob>` applies the directives that follow it to every matching column chunk. See the
`vortex-filecheck` crate docs for the format.

```sh
# Print the dump a check file is matched against
cargo run -p vortex-tui --bin vx -- filecheck slt/tpch/data/nation.vortex
# Reproduce a failing check with full match explanations
cargo run -p vortex-tui --bin vx -- filecheck slt/tpch/data/nation.vortex \
  slt/tpch/filecheck/nation.check --explain
# Run only the encoding checks
cargo nextest run -p vortex-sqllogictest -E 'test(/slt::filecheck::/)'
```

Check files are written by hand; `--complete` does not rewrite them.
