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
# A comparison table is automatically displayed after the run
vx-bench run tpch --engine datafusion,duckdb --format parquet,vortex

# List recent benchmark runs
vx-bench list

# Compare engine:format combinations within a single run
vx-bench compare --run latest

# Compare multiple runs (2 or more)
vx-bench compare --runs run1,run2,run3
```

## Commands

### `run` - Execute Benchmarks

Run benchmark suites across multiple engines and formats. After completion, a comparison table is automatically displayed if there are multiple engine:format combinations.

```bash
vx-bench run <benchmark> [options]
```

**Arguments:**

- `benchmark`: Benchmark suite to run (`tpch`, `tpcds`, `clickbench`, `fineweb`, `gh-archive`, `polarsignals`, `public-bi`, `statpopgen`)

**Options:**

- `--engine, -e`: Engines to benchmark, comma-separated (default: `datafusion,duckdb`)
- `--format, -f`: Formats to benchmark, comma-separated (default: `parquet,vortex`)
- `--targets-json`: Exact benchmark targets as JSON, e.g. `'[{"engine":"datafusion","format":"parquet"}]'`
- `--queries, -q`: Specific queries to run (e.g., `1,2,5`)
- `--exclude-queries`: Queries to skip
- `--iterations, -i`: Iterations per query (default: 5)
- `--label, -l`: Label for this run (useful for later reference)
- `--track-memory`: Enable memory usage tracking
- `--build/--no-build`: Build binaries before running (default: build)
- `--output`: Optional compatibility output path for raw JSON lines

### `prepare-data` - Generate Benchmark Data

Generate only the data formats needed for a benchmark run.

```bash
vx-bench prepare-data <benchmark> [options]
```

**Options:**

- `--format, -f`: Formats to generate, comma-separated
- `--formats-json`: Exact data formats as JSON, e.g. `'["parquet","vortex"]'`
- `--opt`: Benchmark-specific options such as `scale-factor=10.0`

### `compare` - Compare Results

Compare benchmark results within a run or across multiple runs. Results are displayed in a pivot table format.

```bash
vx-bench compare [options]
```

**Options:**

- `--run`: Single run for within-run comparison (compares different engine:format combinations)
- `--runs, -r`: Multiple runs to compare, comma-separated (2 or more)
- `--baseline`: Baseline for comparison (engine:format for within-run, or run label for multi-run)
- `--engine`: Filter results to a specific engine
- `--format`: Filter results to a specific format
- `--threshold`: Significance threshold (default: 0.10 = 10%)

**Within-run comparison** (`--run`): Compares different engine:format combinations within a single run. Output shows one row per query, with columns for each engine:format combo.

**Multi-run comparison** (`--runs`): Compares the same benchmarks across multiple runs. Output shows one row per (query, engine, format) combination, with columns for each run.

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
# Comparison table is displayed automatically after the run
vx-bench run tpch -e datafusion,duckdb -f parquet -l engine-comparison

# Or compare within the run later
vx-bench compare --run engine-comparison
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

# Compare within the run (table shown automatically after run too)
vx-bench compare --run format-analysis

# Use a specific baseline
vx-bench compare --run format-analysis --baseline datafusion:parquet
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
vx-bench run tpch --opt scale-factor=1.0 -l sf1

# Run at SF10
vx-bench run tpch --opt scale-factor=10.0 -l sf10

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
| datafusion | arrow, parquet, vortex, vortex-compact, lance |
| duckdb     | parquet, vortex, vortex-compact, duckdb   |
| lance      | lance                                      |

`datafusion:lance` is executed via the `lance-bench` backend while preserving `datafusion:lance`
result labels.

## CI Usage

The SQL benchmark workflow uses explicit JSON target selection instead of parsing `engine:format`
strings in shell:

```bash
uv run --project bench-orchestrator vx-bench prepare-data tpch \
  --formats-json '["parquet","vortex","vortex-compact"]' \
  --opt scale-factor=1.0

uv run --project bench-orchestrator vx-bench run tpch \
  --targets-json '[{"engine":"datafusion","format":"parquet"},{"engine":"duckdb","format":"vortex"}]' \
  --opt scale-factor=1.0 \
  --output results.json \
  --no-build
```

## Output Format

Comparison results are displayed in a pivot table format:

**Within-run comparison** (`--run`):
```
┌───────┬──────────────────────┬────────────────────────┐
│ Query │ duckdb:parquet (base)│ duckdb:vortex          │
├───────┼──────────────────────┼────────────────────────┤
│     1 │ 100.5ms              │ 80.2ms (0.80x)         │
│     2 │ 200.1ms              │ 150.0ms (0.75x)        │
└───────┴──────────────────────┴────────────────────────┘
```

**Multi-run comparison** (`--runs`):
```
┌───────┬────────┬─────────┬──────────────┬──────────────────┐
│ Query │ Engine │ Format  │ run1 (base)  │ run2             │
├───────┼────────┼─────────┼──────────────┼──────────────────┤
│     1 │ duckdb │ parquet │ 100ms        │ 95ms (0.95x)     │
│     1 │ duckdb │ vortex  │ 80ms         │ 75ms (0.94x)     │
└───────┴────────┴─────────┴──────────────┴──────────────────┘
```

Ratios are color-coded:
- **Green**: Improvement (>10% faster, ratio < 0.9)
- **Red**: Regression (>10% slower, ratio > 1.1)
- **Yellow**: Neutral (within 10%)

## Data Storage

Results are stored in `<workspace>/target/vortex-bench/runs/`. Each run creates a directory containing:

- `metadata.json`: Run configuration and environment info
- `results.jsonl`: Raw benchmark results (JSON lines format)

## Build Configuration

Benchmarks are built with:

- Profile: `release_debug`
- RUSTFLAGS: `-C target-cpu=native -C force-frame-pointers=yes`
- Features: `unstable_encodings`

This enables native CPU optimizations while preserving debug symbols for profiling.
