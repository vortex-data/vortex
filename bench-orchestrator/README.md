# bench-orchestrator

A Python CLI tool for orchestrating Vortex benchmark runs, storing results, and comparing performance across different engines and formats.

## Installation

The best way to install the orchestrator seems to be:

```bash
uv tool install "bench_orchestrator @ ./bench-orchestrator/"
```

This installs the `vx-bench` command.

## Quick Start

```bash
# Run TPC-H benchmarks with DataFusion and DuckDB
vx-bench run tpch --engine datafusion,duckdb --format parquet,vortex

# List recent benchmark runs
vx-bench list

# Compare the two most recent runs
vx-bench compare --runs latest,<previous-run-id>
```

## Commands

### `run` - Execute Benchmarks

Run benchmark suites across multiple engines and formats.

```bash
vx-bench run <benchmark> [options]
```

**Arguments:**

- `benchmark`: Benchmark suite to run (`tpch`, `tpcds`, `clickbench`, `fineweb`, `gh-archive`, `public-bi`, `statpopgen`)

**Options:**

- `--engine, -e`: Engines to benchmark, comma-separated (default: `datafusion,duckdb`)
- `--format, -f`: Formats to benchmark, comma-separated (default: `parquet,vortex`)
- `--queries, -q`: Specific queries to run (e.g., `1,2,5`)
- `--exclude-queries`: Queries to skip
- `--iterations, -i`: Iterations per query (default: 5)
- `--label, -l`: Label for this run (useful for later reference)
- `--scale-factor, -s`: Scale factor for TPC benchmarks
- `--track-memory`: Enable memory usage tracking
- `--build/--no-build`: Build binaries before running (default: build)

### `compare` - Compare Results

Compare benchmark results between runs or specific configurations.

```bash
vx-bench compare [options]
```

**Options:**

- `--runs, -r`: Two run IDs to compare, comma-separated
- `--base, -b`: Base reference (`engine:format@run`)
- `--target, -t`: Target reference (`engine:format@run`)
- `--threshold`: Significance threshold (default: 0.10 = 10%)

### `list` - List Benchmark Runs

```bash
vx-bench list [options]
```

**Options:**

- `--benchmark, -b`: Filter by benchmark suite
- `--since`: Time filter (e.g., `7 days`, `2 weeks`)
- `--limit, -n`: Maximum runs to show (default: 20)

### `show` - Show Run Details

```bash
vx-bench show <run-ref>
```

**Arguments:**

- `run-ref`: Run ID, label, or `latest`

### `build` - Build Binaries

Build benchmark binaries without running benchmarks.

```bash
vx-bench build [options]
```

**Options:**

- `--engine, -e`: Engines to build (default: all)

### `clean` - Clean Old Results

```bash
vx-bench clean --older-than "30 days" [options]
```

**Options:**

- `--older-than`: Delete runs older than (required)
- `--keep-labeled`: Don't delete labeled runs (default: true)
- `--dry-run, -n`: Show what would be deleted

## Example Workflows

### 1. Basic Performance Comparison

Run benchmarks on your current branch and compare against a baseline:

```bash
# First, run benchmarks on your baseline (e.g., main branch)
git checkout main
vx-bench run tpch -e datafusion -f parquet,vortex -l baseline

# Switch to your feature branch and run again
git checkout feature/my-optimization
vx-bench run tpch -e datafusion -f parquet,vortex -l feature

# Compare the runs
vx-bench compare --runs baseline,feature
```

### 2. Quick Regression Check

Run a subset of queries to quickly check for regressions:

```bash
# Run only queries 1, 6, and 12 (fast queries)
vx-bench run tpch -q 1,6,12 -i 3 -l quick-check

# Compare against previous run
vx-bench compare --runs latest,<previous-run-id>
```

### 3. Cross-Engine Comparison

Compare performance across different query engines:

```bash
# Run all engines on the same data
vx-bench run tpch -e datafusion,duckdb -f parquet -l engine-comparison

# View results
vx-bench show latest
```

### 4. Format Performance Analysis

Analyze how different storage formats perform:

```bash
# Run comprehensive format comparison
vx-bench run tpch \
  -e datafusion \
  -f parquet,vortex,vortex-compact \
  -i 10 \
  -l format-analysis

# Compare specific format pairs
vx-bench compare \
  --base "datafusion:parquet@format-analysis" \
  --target "datafusion:vortex@format-analysis" \
```

### 5. Memory Usage Analysis

Track memory usage alongside performance:

```bash
vx-bench run tpch \
  -e datafusion \
  -f vortex \
  --track-memory \
  -l memory-profiling

vx-bench show memory-profiling
```

### 6. Scale Factor Testing

Test performance at different data scales:

```bash
# Run at SF1
vx-bench run tpch -s 1 -l sf1

# Run at SF10
vx-bench run tpch -s 10 -l sf10

# Compare scaling behavior
vx-bench compare --runs sf1,sf10
```

### 7. Excluding Problematic Queries

Skip queries that are known to fail or take too long:

```bash
# Exclude queries 15 and 21 (complex queries)
vx-bench run tpch --exclude-queries 15,21 -l partial-run
```

### 8. Historical Analysis

Find runs from the past week and compare trends:

```bash
# List recent runs
vx-bench list --since "7 days" --benchmark tpch

# Compare two specific historical runs
vx-bench compare --runs <run-id-1>,<run-id-2>
```

### 9. Cleanup Old Results

Keep your results directory manageable:

```bash
# Preview what would be deleted
vx-bench clean --older-than "30 days" --dry-run

# Delete old runs but keep labeled ones
vx-bench clean --older-than "30 days" --keep-labeled

# Delete all old runs including labeled
vx-bench clean --older-than "30 days" --no-keep-labeled
```

## Supported Engines and Formats

| Engine     | Supported Formats                          |
|------------|-------------------------------------------|
| datafusion | parquet, vortex, vortex-compact, lance    |
| duckdb     | parquet, vortex, vortex-compact, duckdb   |
| lance      | lance                                      |

## Target Reference Syntax

When using `--base` and `--target` options, use this format:

```
engine:format@run
```

- `engine`: Engine name (`datafusion`, `duckdb`, `lance`) or `*` for wildcard
- `format`: Format name (`parquet`, `vortex`, etc.) or `*` for wildcard
- `run`: Run ID, label, or `latest`

Examples:

- `duckdb:parquet@latest` - DuckDB with Parquet from the latest run
- `*:vortex@baseline` - All engines with Vortex from the "baseline" run
- `datafusion:*@2025-01-15` - All formats with DataFusion from a specific run

## Output Formats

### Terminal Output

Default output uses rich formatting with color-coded ratios:

- Green (with up arrow): Improvement (>10% faster)
- Red (with down arrow): Regression (>10% slower)
- Yellow: Neutral (within 10%)

## Data Storage

Results are stored in `<workspace>/target/vortex-bench/runs/`. Each run creates a directory containing:

- `metadata.json`: Run configuration and environment info
- `results.jsonl`: Raw benchmark results (JSON lines format)

## Build Configuration

Benchmarks are built with:

- Profile: `release_debug`
- RUSTFLAGS: `-C target-cpu=native -C force-frame-pointers=yes`

This enables native CPU optimizations while preserving debug symbols for profiling.
