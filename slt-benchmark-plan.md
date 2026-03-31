# Plan: SLT-driven benchmark runner with integrated validation

## Context

The benchmark runners currently load queries from `.sql` files and validate against separate `.slt.no` result files
using a `RegenerateSlt` mode and a `validation.rs` module. This is fragile and adds complexity. Instead, use a *
*single `.slt` file per benchmark** as the source of truth for queries + expected results. The runner parses it with
`sqllogictest::parse_file()`, executes queries, benchmarks timing, and validates with `default_validator` — all in one
pass.

## Architecture

### SLT file format

Each benchmark has **one `.slt` file** containing all queries + expected results for all engines:

```
vortex-bench/tpch/slt/tpch.slt         # All 22 TPC-H queries
vortex-bench/clickbench/slt/clickbench.slt  # All 43 ClickBench queries
```

File contents:

```slt
# Optional setup (run once, not timed)
statement ok
CREATE VIEW lineitem_summary AS SELECT ...

# Queries with identical results across engines — written once
query TT rowsort bench_13
SELECT c_count, count(*) AS custdist FROM ...
----
0 50005
1 992

# Queries with engine-specific results — SQL duplicated with onlyif

onlyif datafusion
query TTTTTTTTTT rowsort bench_1
SELECT l_returnflag, l_linestatus, sum(l_quantity), ...
----
A F 37734107.00 56586554400.73 ...

onlyif duckdb
query TTTTTTTTTT rowsort bench_1
SELECT l_returnflag, l_linestatus, sum(l_quantity), ...
----
A F 37734107.00 56586554400.7280 ...
```

- `statement ok` records → setup (run once, no timing)
- `query ... bench_N` records → benchmark (run N times, collect timing, validate)
- Label format: `bench_<name>` — prefix `bench_` marks benchmarked queries, `<name>` used for filtering/reporting
- Queries with identical results across engines: written once (no conditions)
- Queries with different results: duplicated with `onlyif datafusion` / `onlyif duckdb`

### Runner flow

```
parse_file(slt_path)
  → for each Record::Statement: execute once (setup)
  → for each Record::Query:
      - check conditions (onlyif/skipif) against current engine label
      - skip if conditions don't match
      - extract query name from label (strip "bench_" prefix)
      - check filter (--queries 1,5)
      - run N iterations, collect timing
      - validate last result with default_validator against expected rows
      - record QueryMeasurement with name
```

### Filtering

`--queries 1,5,10` works unchanged — the runner strips the `bench_` prefix from labels, parses the rest as `usize`, and
matches against the `--queries` filter.

## Changes

### 1. Add `slt_path()` to Benchmark trait

**File:** `vortex-bench/src/benchmark.rs`

- Add method `fn slt_path(&self) -> Option<PathBuf>` returning path to the `.slt` file
- Default impl returns `None` (benchmarks without validation)
- TPC-H impl: `Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("tpch/slt/tpch.slt"))`
- ClickBench impl: `Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench/slt/clickbench.slt"))`

### 2. Add SLT-driven execution to `SqlBenchmarkRunner`

**File:** `vortex-bench/src/runner.rs`

- Add new method `run_slt()` (sync) and `run_slt_async()` (async) that:
    1. Call `parse_file::<DefaultColumnType>(slt_path)` to parse the `.slt` file
    2. Check each record's `conditions` against the engine label (e.g., `"datafusion"`, `"duckdb"`) — skip records whose
       conditions don't match
    3. Process matching records:
        - `Record::Statement { sql, .. }` → call setup callback once
        - `Record::Query { sql, expected: QueryExpect::Results { label, results, .. }, .. }` → check label starts with
          `bench_`, extract name, check filter, run N iterations via `run_query()`, validate with
          `default_validator(value_normalizer, &actual_rows, &results)` (using
          `datafusion_sqllogictest::value_normalizer`)
    4. Collect validation failures, bail at end if any
- Reuse existing `run_query()` and `record_query()` for timing/measurement
- Remove `BenchmarkMode::RegenerateSlt` variant
- Remove `write_slt_file()` method
- Remove `validate_query_result()` and `reference_path()` (replaced by inline validation in `run_slt`)
- Remove `expected_results_dir` field from `SqlBenchmarkRunner`

### 3. Delete `validation.rs`

**File:** `vortex-bench/src/validation.rs` — **delete**
**File:** `vortex-bench/src/lib.rs` — remove `pub mod validation;`

### 4. Keep dependencies

**File:** `vortex-bench/Cargo.toml`

- Keep `datafusion-sqllogictest = { workspace = true }` — use `value_normalizer` from it
- Keep `sqllogictest = "0.28"` — used for `parse_file`, `default_validator`, `Record`, `QueryExpect`, `Condition`

### 5. Create consolidated `.slt` files

- **`vortex-bench/tpch/slt/tpch.slt`** — all 22 TPC-H queries
    - 19 queries with identical results: written once (no conditions)
    - 3 queries with different results (q01, q08, q14): duplicated with `onlyif datafusion` / `onlyif duckdb`
    - SQL sourced from existing `tpch/queries/q{01..22}.sql` files
    - Expected results sourced from existing `.slt.no` result files
- **`vortex-bench/clickbench/slt/clickbench.slt`** — all 43 ClickBench queries
    - Same pattern: shared queries once, engine-specific with `onlyif`
- Delete old `vortex-bench/tpch/slt/results/` and `vortex-bench/clickbench/slt/results/` directories
- Delete old `vortex-bench/tpch/queries/` `.sql` files (SQL now lives in `.slt`)

### 6. Update CLI in benchmark binaries

**File:** `benchmarks/datafusion-bench/src/main.rs`

- Remove `--regenerate-slt` arg
- Keep `--validate` — when set, forces `iterations = 1` (fast validation mode)
- `--queries` unchanged — still takes `Vec<usize>`, matched against numeric part of `bench_N` labels
- Replace `run_all_async()` call with `run_slt_async()`:
    - Pass engine label `"datafusion"` for condition checking
    - Setup callback handles table registration (same as current `setup` closure)
    - Execute callback runs a single SQL query and returns result (same as current `execute` closure)
- Export logic stays the same

**File:** `benchmarks/duckdb-bench/src/main.rs` — same changes with engine label `"duckdb"`

- Remove `mod validation;` and delete `benchmarks/duckdb-bench/src/validation.rs` if it exists

### 7. Write Python regeneration script

**File:** `scripts/regenerate-slt.py` — **new file**

- Runs the benchmark binary with `--print-results -i 1 --formats parquet`
- Parses `=== Q{N} ===` delimited stdout output
- Generates consolidated `.slt` file with `bench_N` labels, rowsort, sorted results
- Handles `onlyif` for engine-specific results by comparing DataFusion vs DuckDB output
- Usage: `python scripts/regenerate-slt.py --benchmark tpch`

## Key functions to reuse

- `sqllogictest::parse_file::<DefaultColumnType>(path)` — parse `.slt` file, resolve includes → `Vec<Record>`
- `sqllogictest::default_validator(normalizer, actual, expected)` — compare rows
- `Condition::should_skip(labels)` — check if record should be skipped for current engine
- `Record::Query { conditions, sql, expected: QueryExpect::Results { label, results, sort_mode, .. }, .. }` — structured
  access
- `SqlBenchmarkRunner::run_query()` — existing timing/measurement infrastructure (lines 137–165)
- `SqlBenchmarkRunner::record_query()` — existing measurement recording (lines 174–215)

## Files touched

| File                                         | Action                                                                                                      |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `vortex-bench/src/benchmark.rs`              | Add `slt_path()` method                                                                                     |
| `vortex-bench/src/runner.rs`                 | Add `run_slt()`/`run_slt_async()`, remove RegenerateSlt/write_slt_file/validate_query_result/reference_path |
| `vortex-bench/src/validation.rs`             | Delete                                                                                                      |
| `vortex-bench/src/lib.rs`                    | Remove `pub mod validation`                                                                                 |
| `vortex-bench/Cargo.toml`                    | No changes (keep both deps)                                                                                 |
| `vortex-bench/tpch/slt/tpch.slt`             | New — consolidated TPC-H queries + results                                                                  |
| `vortex-bench/clickbench/slt/clickbench.slt` | New — consolidated ClickBench queries + results                                                             |
| `vortex-bench/tpch/slt/results/`             | Delete old directory                                                                                        |
| `vortex-bench/clickbench/slt/results/`       | Delete old directory                                                                                        |
| `benchmarks/datafusion-bench/src/main.rs`    | Remove `--regenerate-slt`, use `run_slt_async()`                                                            |
| `benchmarks/duckdb-bench/src/main.rs`        | Remove `--regenerate-slt`, use `run_slt()`                                                                  |
| `scripts/regenerate-slt.py`                  | New — regeneration script                                                                                   |

## Verification

```bash
cargo build -p vortex-bench -p datafusion-bench -p duckdb-bench
cargo clippy -p vortex-bench -p datafusion-bench -p duckdb-bench --all-targets --all-features
cargo +nightly fmt --all
```
