# ISA Threshold Finder - State Document

## Overview

The ISA Threshold Finder is a system for automatically detecting crossover points where one algorithm implementation becomes faster than another across different CPU architectures. This enables Vortex to dynamically select the optimal implementation at runtime based on input size and CPU type.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Crate Structure                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  vortex-threshold-traits     Core trait definitions & builder API        │
│  ├── BenchmarkableAlgorithm  Trait for benchmarkable algorithms          │
│  ├── ThresholdBench          Criterion-like builder for easy setup       │
│  ├── ParameterScale          Linear/Log/Explicit parameter ranges        │
│  ├── Variant                 Algorithm variant with CPU feature reqs     │
│  ├── CpuClass                Runtime CPU detection (Intel/AMD/ARM)       │
│  └── BenchmarkStorage        Trait for result persistence                │
│                                                                          │
│  vortex-threshold-runner     CLI tool for running benchmarks             │
│  ├── GridSearch              Sweeps parameter space, finds crossovers    │
│  ├── examples/               Popcount (trait) and Sum (builder) demos    │
│  └── storage/sqlite          SQLite backend for result persistence       │
│                                                                          │
│  vortex-threshold-aggregator Merges results, generates Rust code         │
│  └── Generates LazyLock      Static dispatch tables per CpuClass         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## API Styles

### 1. Builder API (Recommended for most cases)

```rust
use vortex_threshold_traits::{ThresholdBench, ParameterScale, AlgorithmRegistry};

let mut registry = AlgorithmRegistry::new();

ThresholdBench::new("sum")
    .parameter("count", ParameterScale::log2(6, 20))
    .input(|size, seed| random_vec(size, seed))
    .baseline("naive", |data| sum_naive(data))
    .variant("unrolled", |data| sum_unrolled(data))
    .variant_if("avx2", is_x86_feature_detected!("avx2"), |data| sum_avx2(data))
    .variant_with_features("neon", &["neon"], |data| sum_neon(data))
    .register(&mut registry);
```

### 2. Trait API (For complex cases)

```rust
use vortex_threshold_traits::{BenchmarkableAlgorithm, ParameterScale, Variant};

struct PopcountBenchmark;

impl BenchmarkableAlgorithm for PopcountBenchmark {
    type Input = Vec<u64>;
    type Output = usize;

    fn name(&self) -> &'static str { "popcount" }
    fn parameter_name(&self) -> &'static str { "input_size" }
    fn parameter_scale(&self) -> ParameterScale { ParameterScale::log2(6, 20) }

    fn variants(&self) -> Vec<Variant> {
        vec![
            Variant::new("naive"),
            Variant::new("avx2").with_features(&["avx2"]),
        ]
    }

    fn generate_input(&self, param: usize, seed: u64) -> Self::Input { /* ... */ }
    fn ground_truth(&self, input: &Self::Input) -> Self::Output { /* ... */ }
    fn run_variant(&self, variant: &str, input: &Self::Input) -> Self::Output { /* ... */ }
}
```

## Storage Layer

### SQLite Backend (Optional Feature)

```rust
use vortex_threshold_runner::storage::SqliteStorage;

let storage = SqliteStorage::open("benchmarks.db")?;

// Store measurements
storage.store_measurements(&measurements)?;

// Query by algorithm, variant, CPU class, commit, time range
let query = MeasurementQuery::new()
    .algorithm("popcount")
    .cpu_class(CpuClass::IntelSapphire)
    .since(yesterday);
let results = storage.query_measurements(&query)?;

// Get threshold history for regression detection
let history = storage.get_threshold_history(
    "popcount", "naive", "simd", CpuClass::IntelSapphire, 10
)?;

// Compare thresholds between commits
let diffs = storage.compare_commits("abc123", "def456")?;
```

## CI Integration

GitHub Actions workflow (`.github/workflows/isa-thresholds.yml`) runs benchmarks on:

| Architecture | Runner | CPU Features |
|-------------|--------|--------------|
| Intel Sapphire Rapids | `runs-on: intel-sapphire` | AVX-512 |
| Intel Ice Lake | `runs-on: intel-icelake` | AVX-512 |
| AMD Genoa (Zen 4) | `runs-on: amd-genoa` | AVX-512 |
| AMD Milan (Zen 3) | `runs-on: amd-milan` | AVX2 |
| AWS Graviton 3 | `runs-on: graviton3` | NEON, SVE |
| AWS Graviton 2 | `runs-on: graviton2` | NEON |

## Current Status

### Completed
- [x] Core trait definitions (`BenchmarkableAlgorithm`)
- [x] Builder API (`ThresholdBench`) - criterion-like ergonomics
- [x] Parameter scales (Linear, Logarithmic, Explicit)
- [x] CPU feature detection and variant availability
- [x] CPU class detection (Intel/AMD/ARM families)
- [x] Grid search for crossover detection
- [x] JSON output for CI artifact collection
- [x] SQLite storage backend with query support
- [x] GitHub Actions workflow template
- [x] Example benchmarks (popcount, sum)
- [x] Result aggregation and Rust code generation

### Not Yet Implemented
- [ ] Binary search refinement for precise crossover points
- [ ] Statistical significance testing (confidence intervals)
- [ ] Automatic PR comments with threshold changes
- [ ] Integration with actual Vortex algorithms (rank, select, etc.)
- [ ] Dashboard/visualization for threshold trends

## File Locations

```
vortex/
├── vortex-threshold-traits/
│   ├── src/
│   │   ├── lib.rs           # Core traits, CpuClass, ParameterScale
│   │   ├── builder.rs       # ThresholdBench builder API
│   │   └── storage.rs       # BenchmarkStorage trait
│   └── Cargo.toml
│
├── vortex-threshold-runner/
│   ├── src/
│   │   ├── main.rs          # CLI, GridSearch implementation
│   │   ├── examples/
│   │   │   ├── mod.rs
│   │   │   ├── popcount.rs  # Trait-based example
│   │   │   └── sum.rs       # Builder-based example
│   │   └── storage/
│   │       ├── mod.rs
│   │       └── sqlite.rs    # SQLite implementation
│   └── Cargo.toml
│
├── vortex-threshold-aggregator/
│   ├── src/
│   │   └── main.rs          # Merges JSON, generates Rust code
│   └── Cargo.toml
│
└── .github/workflows/
    └── isa-thresholds.yml   # Multi-architecture CI workflow
```

## Usage

### Running Benchmarks Locally

```bash
# Build the runner
cargo build -p vortex-threshold-runner --release

# Run with default examples
./target/release/threshold-runner --output results.json

# Run specific algorithm
./target/release/threshold-runner --algorithm popcount --output results.json
```

### Aggregating Results

```bash
# Merge results from multiple architectures
./target/release/threshold-aggregator \
    --input intel-sapphire.json \
    --input amd-genoa.json \
    --input graviton3.json \
    --output src/thresholds.rs
```

### Generated Code Example

```rust
use std::sync::LazyLock;
use vortex_threshold_traits::CpuClass;

static POPCOUNT_THRESHOLDS: LazyLock<PopcountThresholds> = LazyLock::new(|| {
    match CpuClass::detect() {
        CpuClass::IntelSapphire => PopcountThresholds { naive_to_simd: 256 },
        CpuClass::AmdGenoa => PopcountThresholds { naive_to_simd: 512 },
        CpuClass::Graviton3 => PopcountThresholds { naive_to_simd: 128 },
        _ => PopcountThresholds { naive_to_simd: 256 }, // default
    }
});
```

## Design Decisions

1. **Two API styles**: Builder for simplicity, trait for control
2. **Static code generation**: Zero runtime cost, thresholds baked into binary
3. **SQLite for persistence**: Rich queries, no external dependencies
4. **CpuClass enum**: Coarse-grained grouping, not per-model thresholds
5. **Feature flags**: SQLite is optional (`--features sqlite`)
