# vortex-sqllogictest

This crate tests correctness of our DuckDB and DataFusion integrations. It runs
[.slt](https://github.com/risinglightdb/sqllogictest-rs) tests with our queries
as well as 0.1 scale of TPC-H, TPC-DS, and 1 shard of partitioned ClickBench.
For last three query sets, tests also verify each query's `EXPLAIN` result.

## Running tests

```sh
# For running our own queries
cargo test -p vortex-sqllogictest --test sqllogictests

# For running TPC-H, TPC-DS, or ClickBench you need to generate data first
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
