# Benchmarking

Vortex has a three-tier benchmarking system: microbenchmarks for individual operations,
end-to-end SQL benchmarks for query-level performance, and an orchestrator for running and
comparing benchmark suites.

## Microbenchmarks

Microbenchmarks use the Divan framework and live in `benches/` directories within individual
crates. They cover low-level operations such as encoding, decoding, compute kernels, buffer
operations, and scalar access.

Run microbenchmarks for a specific crate with:

```bash
cargo bench -p <crate-name>
```

## SQL Benchmarks

SQL benchmarks measure end-to-end query performance across different engines and file formats.
The `vortex-bench` crate provides a common `Benchmark` trait that each benchmark suite
implements, defining its queries, data generation, and expected results.

Available suites include TPC-H, TPC-DS, ClickBench, FineWeb, and others. Each suite can be
run against multiple engines (DataFusion, DuckDB) and formats (Parquet, Vortex, Vortex Compact,
Lance, DuckDB native).

### Data Generation

Before running SQL benchmarks, test data must be generated:

```bash
cargo run --release --bin data-gen -- <benchmark> --formats parquet,vortex
```

The data generator creates base Parquet data and converts it to each requested format. Scale
factors are configurable per suite (e.g. `--opt scale-factor=10.0` for TPC-H SF=10).

### Running SQL Benchmarks

SQL benchmarks can be run directly:

```bash
cargo run --release --bin datafusion-bench -- <benchmark>
cargo run --release --bin duckdb-bench -- <benchmark>
```

## Orchestrator

The `bench-orchestrator` is a Python CLI tool (`vx-bench`) that automates building, running,
storing, and comparing benchmark results. Install it with:

```bash
uv tool install "bench_orchestrator @ ./bench-orchestrator/"
```

### Running Benchmarks

```bash
# Run TPC-H on DataFusion and DuckDB, comparing Parquet and Vortex
vx-bench run tpch --engine datafusion,duckdb --format parquet,vortex

# Run a subset of queries with fewer iterations
vx-bench run tpch -q 1,6,12 -i 3

# Run with memory tracking
vx-bench run tpch --track-memory

# Run with CPU profiling
vx-bench run tpch --samply
```

### Comparing Results

```bash
# Compare formats/engines within the most recent run
vx-bench compare --run latest

# Compare across two labeled runs
vx-bench compare --runs baseline,feature
```

Comparison output is color-coded: green for improvements (>10%), yellow for neutral, red for
regressions.

### Result Storage

Results are stored as JSON Lines files under `target/vortex-bench/runs/`, with each run
containing metadata (git commit, timestamp, configuration) and per-query timing data. The
`vx-bench list` command shows recent runs.

## CI Benchmarks

Benchmarks run automatically in CI on two triggers:

- **Post-commit** -- compression and random access benchmarks run on every push to `develop`,
  with results uploaded for historical tracking.
- **PR benchmarks** -- triggered by the `action/benchmark` label or `[benchmark]` in the commit
  message. Results are compared against the latest `develop` run and posted as a PR comment.
- **SQL benchmarks** -- a parametric matrix of suites, engines, formats, and storage backends
  (NVMe, S3) run on labeled PRs and scheduled triggers.

All CI benchmarks run on dedicated instances with the `release_debug` profile and
`-C target-cpu=native` to produce representative numbers.
