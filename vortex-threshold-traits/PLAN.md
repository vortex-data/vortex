# ISA Threshold Finder - Planning Document

## Goal

Automatically detect crossover points where one algorithm implementation becomes faster than another across different CPU architectures, enabling Vortex to dynamically select optimal implementations at runtime.

## Components

The system is divided into three main components:

### 1. [Benchmark Runner & Grid Search](./plan/benchmark/plan.md)
Responsible for gathering performance data by running algorithms across parameter ranges.

- Parameter space exploration (linear, log, explicit scales)
- Grid search for crossover detection
- Binary search refinement for precise thresholds
- Statistical measurement (multiple iterations, warmup)
- See also: [Measurement Quality](./plan/benchmark/measurement.md)

### 2. [Data Storage & Production](./plan/storage/plan.md)
Responsible for persisting benchmark results and producing threshold data.

- SQLite backend for local/CI storage
- JSON export for CI artifacts
- Rust code generation for static dispatch tables
- Query interface for historical analysis

### 3. [CI Runners & Infrastructure](./plan/runners/plan.md)
Responsible for running benchmarks across multiple CPU architectures in CI.

- GitHub Actions workflow configuration
- Multi-architecture runner matrix
- Artifact collection and aggregation
- PR integration (comments, checks)

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Benchmark Runner | Partial | Grid search done, binary search not implemented |
| Data Storage | Partial | SQLite done, code generation done |
| CI Runners | Scaffold | Workflow template exists, not tested |

## Priority Order

1. **Benchmark Runner** - Core functionality, must work locally first
2. **Data Storage** - Need to persist and aggregate results
3. **CI Runners** - Scale to multiple architectures

## Open Questions

- [ ] What statistical significance level is required for crossover detection?
- [ ] How to handle noisy measurements on shared CI runners?
- [ ] Should thresholds be per-commit or aggregated over time windows?
- [ ] How to version/migrate threshold data when algorithms change?

## Sub-Documents

- [plan/benchmark/](./plan/benchmark/) - Benchmark runner & grid search
  - [plan.md](./plan/benchmark/plan.md) - Main benchmark planning
  - [measurement.md](./plan/benchmark/measurement.md) - Measurement quality & iteration mechanics
- [plan/storage/plan.md](./plan/storage/plan.md) - Data storage & production
- [plan/runners/plan.md](./plan/runners/plan.md) - CI runners & infrastructure
