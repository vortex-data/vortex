# Benchmark Runner & Grid Search

## Overview

The benchmark runner gathers performance data by executing algorithm variants across a parameter space and detecting crossover points where one variant becomes faster than another, based on input data statistics.

## Key Concepts

1. **Data**: Raw input to the algorithm (e.g., bitmap + position)
2. **Stats**: Computed from data, used for dispatch decisions (e.g., len, density)
3. **Variants**: Different algorithm implementations
4. **Params**: Tunable parameters per variant (e.g., chunk_size, unroll factor)
5. **Grid Search**: Measure all (stats × variant × params) combinations
6. **Crossovers**: Points where optimal (variant, params) changes

## Two Modes, One Definition

```rust
// Define benchmark once
let benchmark = ThresholdBench::new("rank")
    .stats(RankStats::compute)
    .generate(RankData::generate)
    .stats_grid(StatsGrid::new()
        .dimension("len", Scale::log2(6, 20))
        .dimension("density", Scale::steps(0.0, 1.0, 0.1)))
    .variant("naive", rank_naive)
    .variant_with_params("chunked", rank_chunked)
    .variant_with_params("avx2", rank_avx2).requires(&["avx2"])
    .build();

// Mode 1: Benchmark - run with default params, report table
benchmark.bench().run().print();

// Mode 2: Search - run full param grid, find crossovers
benchmark.search().run().print();
```

| Mode | Params Used | Output |
|------|-------------|--------|
| `bench()` | `P::default_params()` | Performance table |
| `search()` | `P::grid()` (all combinations) | Crossovers + winners |

---

## API Design

### User Workflow

```rust
// === 1. Define your data and stats types ===

struct RankData {
    bitmap: Vec<u64>,
    position: usize,
}

#[derive(Clone)]
struct RankStats {
    len: usize,
    density: f64,
}

impl RankStats {
    fn compute(data: &RankData) -> Self {
        let true_count: usize = data.bitmap.iter()
            .map(|w| w.count_ones() as usize).sum();
        let total_bits = data.bitmap.len() * 64;
        Self {
            len: data.position,
            density: true_count as f64 / total_bits as f64,
        }
    }

    fn generate(stats: &RankStats, seed: u64) -> RankData {
        RankData {
            bitmap: generate_bitmap(stats.len, stats.density, seed),
            position: stats.len,
        }
    }
}


// === 2. Define param structs with #[derive(ParamGrid)] ===

#[derive(Clone, ParamGrid)]
struct ChunkedParams {
    #[grid(1, 2, 4, 8, 16)]
    #[default = 4]
    chunk_size: usize,
}

#[derive(Clone, ParamGrid)]
struct Avx2Params {
    #[grid(1, 2, 4)]
    #[default = 2]
    unroll: usize,
}


// === 3. Write your algorithm implementations ===

fn rank_naive(data: &RankData) -> usize {
    rank_naive_impl(&data.bitmap, data.position)
}

fn rank_chunked(data: &RankData, params: &ChunkedParams) -> usize {
    rank_chunked_impl(&data.bitmap, data.position, params.chunk_size)
}

fn rank_avx2(data: &RankData, params: &Avx2Params) -> usize {
    unsafe { rank_avx2_impl(&data.bitmap, data.position, params.unroll) }
}


// === 4. Register benchmark ===

fn main() {
    let benchmark = ThresholdBench::new("rank")
        .stats(RankStats::compute)
        .generate(RankStats::generate)
        .stats_grid(StatsGrid::new()
            .dimension("len", Scale::log2(6, 20))
            .dimension("density", Scale::steps(0.0, 1.0, 0.1)))
        .variant("naive", rank_naive)
        .variant_with_params("chunked", rank_chunked)
        .variant_with_params("avx2", rank_avx2).requires(&["avx2"])
        .build();

    // Run based on CLI args
    match args.mode {
        Mode::Bench => benchmark.bench().run().print(),
        Mode::Search => benchmark.search().run().print(),
    }
}
```

---

## ParamGrid Derive Macro

### Basic Usage

```rust
#[derive(Clone, ParamGrid)]
struct ChunkedParams {
    #[grid(1, 2, 4, 8, 16)]   // values to search
    #[default = 4]            // value for bench mode
    chunk_size: usize,
}
```

### Grid Attribute Variants

| Syntax | Expands To | Example |
|--------|------------|---------|
| `#[grid(1, 2, 4, 8)]` | Explicit values | `[1, 2, 4, 8]` |
| `#[grid(range(1, 8))]` | Inclusive range | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| `#[grid(range(0, 100, step = 10))]` | Range with step | `[0, 10, 20, ..., 100]` |
| `#[grid(log2(6, 10))]` | Powers of 2 | `[64, 128, 256, 512, 1024]` |
| `#[grid(log(10, 1, 4))]` | Powers of base | `[10, 100, 1000, 10000]` |
| `#[grid(range(0.0, 1.0, step = 0.25))]` | Float range | `[0.0, 0.25, 0.5, 0.75, 1.0]` |

### Examples

```rust
// Explicit values
#[derive(Clone, ParamGrid)]
struct UnrollParams {
    #[grid(1, 2, 4, 8)]
    #[default = 4]
    unroll: usize,
}

// Range with step
#[derive(Clone, ParamGrid)]
struct TileParams {
    #[grid(range(16, 128, step = 16))]  // 16, 32, 48, 64, 80, 96, 112, 128
    #[default = 64]
    tile_size: usize,
}

// Log2 range (powers of 2)
#[derive(Clone, ParamGrid)]
struct BlockParams {
    #[grid(log2(6, 12))]  // 64, 128, 256, 512, 1024, 2048, 4096
    #[default = 256]
    block_size: usize,
}

// Float range
#[derive(Clone, ParamGrid)]
struct ThresholdParams {
    #[grid(range(0.0, 1.0, step = 0.1))]
    #[default = 0.5]
    threshold: f64,
}

// Boolean
#[derive(Clone, ParamGrid)]
struct FeatureParams {
    #[grid(true, false)]
    #[default = false]
    prefetch: bool,
}

// Multiple params (cartesian product)
#[derive(Clone, ParamGrid)]
struct SimdParams {
    #[grid(1, 2, 4)]              // 3 values
    #[default = 2]
    unroll: usize,

    #[grid(log2(8, 12))]          // 5 values: 256..4096
    #[default = 1024]
    block_size: usize,

    #[grid(true, false)]          // 2 values
    #[default = false]
    prefetch: bool,
}
// Total: 3 × 5 × 2 = 30 combinations for search
// Default: (2, 1024, false) for bench
```

### Generated Code

```rust
// For:
#[derive(Clone, ParamGrid)]
struct ChunkedParams {
    #[grid(log2(0, 4))]
    #[default = 4]
    chunk_size: usize,
}

// Macro generates:
impl ParamGrid for ChunkedParams {
    fn grid() -> Vec<Self> {
        [1, 2, 4, 8, 16].into_iter()
            .map(|chunk_size| ChunkedParams { chunk_size })
            .collect()
    }

    fn default_params() -> Self {
        ChunkedParams { chunk_size: 4 }
    }

    fn to_map(&self) -> BTreeMap<&'static str, ParamValue> {
        BTreeMap::from([
            ("chunk_size", ParamValue::Usize(self.chunk_size)),
        ])
    }

    fn field_names() -> &'static [&'static str] {
        &["chunk_size"]
    }
}

// For multiple params, generates cartesian product:
#[derive(Clone, ParamGrid)]
struct SimdParams {
    #[grid(1, 2, 4)]
    #[default = 2]
    unroll: usize,

    #[grid(true, false)]
    #[default = false]
    prefetch: bool,
}

// Generates 3 × 2 = 6 combinations:
impl ParamGrid for SimdParams {
    fn grid() -> Vec<Self> {
        let mut result = Vec::new();
        for unroll in [1, 2, 4] {
            for prefetch in [true, false] {
                result.push(SimdParams { unroll, prefetch });
            }
        }
        result
        // [(1,true), (1,false), (2,true), (2,false), (4,true), (4,false)]
    }

    fn default_params() -> Self {
        SimdParams { unroll: 2, prefetch: false }
    }

    // ...
}
```

---

## Core Traits

```rust
/// Trait for param structs that can enumerate their grid
pub trait ParamGrid: Clone + Send + Sync + 'static {
    /// All parameter combinations to search
    fn grid() -> Vec<Self>;

    /// Default params for bench mode
    fn default_params() -> Self;

    /// Convert to map for serialization/display
    fn to_map(&self) -> BTreeMap<&'static str, ParamValue>;

    /// Field names for reporting
    fn field_names() -> &'static [&'static str];
}

/// Implement for no-params case
impl ParamGrid for () {
    fn grid() -> Vec<Self> { vec![()] }
    fn default_params() -> Self { () }
    fn to_map(&self) -> BTreeMap<&'static str, ParamValue> { BTreeMap::new() }
    fn field_names() -> &'static [&'static str] { &[] }
}

/// Values that can appear in param grids
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Usize(usize),
    Isize(isize),
    Bool(bool),
    F64(f64),
}
```

---

## Benchmark Builder

```rust
pub struct ThresholdBench<Data, Stats, Output> {
    name: String,
    stats_fn: Arc<dyn Fn(&Data) -> Stats + Send + Sync>,
    generate_fn: Arc<dyn Fn(&Stats, u64) -> Data + Send + Sync>,
    stats_grid: StatsGrid,
    variants: Vec<DynVariant<Data, Output>>,
}

impl<Data, Stats, Output> ThresholdBench<Data, Stats, Output>
where
    Data: Send + Sync + 'static,
    Stats: Clone + Send + Sync + 'static,
    Output: PartialEq + Send + Sync + 'static,
{
    pub fn new(name: impl Into<String>) -> Self { ... }

    /// Set function to compute stats from data
    pub fn stats<F>(self, f: F) -> Self
    where F: Fn(&Data) -> Stats + Send + Sync + 'static { ... }

    /// Set function to generate data from stats
    pub fn generate<F>(self, f: F) -> Self
    where F: Fn(&Stats, u64) -> Data + Send + Sync + 'static { ... }

    /// Set the stats grid to search/benchmark over
    pub fn stats_grid(self, grid: StatsGrid) -> Self { ... }

    /// Register variant with no params
    pub fn variant<F>(self, name: &'static str, f: F) -> Self
    where F: Fn(&Data) -> Output + Send + Sync + 'static
    {
        self.variant_with_params::<(), _>(name, move |data, _| f(data))
    }

    /// Register variant with typed params
    pub fn variant_with_params<P, F>(self, name: &'static str, f: F) -> VariantBuilder<Self, P>
    where
        P: ParamGrid,
        F: Fn(&Data, &P) -> Output + Send + Sync + 'static,
    {
        VariantBuilder { bench: self, name, func: f, features: vec![] }
    }

    /// Build the benchmark
    pub fn build(self) -> Self { self }

    /// Run in benchmark mode (default params only)
    pub fn bench(&self) -> BenchRunner<'_, Data, Stats, Output> {
        BenchRunner::new(self)
    }

    /// Run in search mode (full param grid)
    pub fn search(&self) -> SearchRunner<'_, Data, Stats, Output> {
        SearchRunner::new(self)
    }
}

/// Builder for adding features to a variant
pub struct VariantBuilder<B, P> {
    bench: B,
    name: &'static str,
    func: ...,
    features: Vec<&'static str>,
}

impl<B, P> VariantBuilder<B, P> {
    /// Require CPU features for this variant
    pub fn requires(mut self, features: &[&'static str]) -> Self {
        self.features = features.to_vec();
        self
    }

    // Auto-finalizes when next builder method is called
}
```

---

## Bench Mode

Runs all variants at all stats points using **default params only**.

```rust
pub struct BenchRunner<'a, D, S, O> {
    benchmark: &'a ThresholdBench<D, S, O>,
    filter_stats: Option<Vec<S>>,
    baseline: Option<PathBuf>,
}

impl<'a, D, S, O> BenchRunner<'a, D, S, O> {
    /// Only benchmark at specific stats points
    pub fn at(mut self, stats: S) -> Self {
        self.filter_stats.get_or_insert_with(Vec::new).push(stats);
        self
    }

    /// Compare against saved baseline
    pub fn compare(mut self, path: impl AsRef<Path>) -> Self {
        self.baseline = Some(path.as_ref().to_path_buf());
        self
    }

    /// Run benchmarks
    pub fn run(self) -> BenchReport { ... }
}

pub struct BenchReport { ... }

impl BenchReport {
    /// Print Divan-style table
    pub fn print(&self) { ... }

    /// Save as baseline for future comparison
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> { ... }
}
```

### Bench Output

```
rank                                    fastest       │ median        │ samples │
├─ len=64, density=0.1                                │               │         │
│  ├─ naive                             12.3 ns       │ 14.5 ns       │ 100     │ ★
│  ├─ chunked(chunk_size=4)             15.2 ns       │ 17.3 ns       │ 100     │
│  └─ avx2(unroll=2)                    18.1 ns       │ 20.4 ns       │ 100     │
├─ len=1024, density=0.5                              │               │         │
│  ├─ naive                             823 ns        │ 891 ns        │ 100     │
│  ├─ chunked(chunk_size=4)             312 ns        │ 334 ns        │ 100     │ ★
│  └─ avx2(unroll=2)                    287 ns        │ 312 ns        │ 100     │
├─ len=65536, density=0.5                             │               │         │
│  ├─ naive                             52.3 µs       │ 54.8 µs       │ 100     │
│  ├─ chunked(chunk_size=4)             18.2 µs       │ 19.1 µs       │ 100     │
│  └─ avx2(unroll=2)                    8.23 µs       │ 8.89 µs       │ 100     │ ★
```

### Bench with Baseline Comparison

```rust
benchmark.bench().compare("baseline.json").run().print();
```

```
rank                                    median        │ vs baseline   │
├─ len=65536, density=0.5                             │               │
│  ├─ naive                             54.8 µs       │ +2.3%         │
│  ├─ chunked(chunk_size=4)             19.1 µs       │ -1.2%         │
│  └─ avx2(unroll=2)                    8.89 µs       │ -5.1%  ▼      │ faster!
```

---

## Search Mode

Runs all variants at all stats points using **full param grid**.

```rust
pub struct SearchRunner<'a, D, S, O> {
    benchmark: &'a ThresholdBench<D, S, O>,
    refine: bool,
}

impl<'a, D, S, O> SearchRunner<'a, D, S, O> {
    /// Enable binary search refinement of crossovers
    pub fn refine(mut self, enabled: bool) -> Self {
        self.refine = enabled;
        self
    }

    /// Run search
    pub fn run(self) -> SearchResults { ... }
}

pub struct SearchResults {
    pub measurements: Vec<Measurement>,
    pub winners: Vec<Winner>,
    pub crossovers: Vec<Crossover>,
}

impl SearchResults {
    /// Print summary
    pub fn print(&self) { ... }

    /// Export to JSON
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<()> { ... }
}
```

### Search Output

```
rank - Threshold Search Results
===============================

Configs tested: 12
  - naive (no params)
  - chunked: chunk_size ∈ [1, 2, 4, 8, 16]
  - avx2: unroll ∈ [1, 2, 4]

Stats grid: 195 points (15 lens × 13 densities)
Total measurements: 2,340

Winners by Region
─────────────────
len < 256           → naive
len ∈ [256, 2048)   → chunked(chunk_size=4)   [density < 0.3]
                    → chunked(chunk_size=8)   [density ≥ 0.3]
len ≥ 2048          → avx2(unroll=2)          [density < 0.5]
                    → avx2(unroll=4)          [density ≥ 0.5]

Crossovers
──────────
Along len (density=0.1):
  256:    naive → chunked(chunk_size=4)
  2048:   chunked(chunk_size=4) → avx2(unroll=2)

Along len (density=0.5):
  256:    naive → chunked(chunk_size=8)
  2048:   chunked(chunk_size=8) → avx2(unroll=4)

Along density (len=1024):
  0.3:    chunked(chunk_size=4) → chunked(chunk_size=8)

Along density (len=8192):
  0.5:    avx2(unroll=2) → avx2(unroll=4)
```

---

## Grid Search Algorithm

```rust
impl<D, S, O> SearchRunner<'_, D, S, O> {
    pub fn run(self) -> SearchResults {
        // 1. Enumerate all configs using P::grid() for each variant
        let configs = self.enumerate_configs();

        // 2. Generate stats points from stats_grid
        let stats_points = self.benchmark.stats_grid.points();

        // 3. Measure all (stats × config) combinations
        let measurements = self.measure_all(&stats_points, &configs);

        // 4. Find winner at each stats point
        let winners = self.find_winners(&measurements);

        // 5. Detect crossovers along each stats dimension
        let crossovers = self.find_crossovers(&winners);

        // 6. Optionally refine crossovers with binary search
        let crossovers = if self.refine {
            self.refine_crossovers(&crossovers)
        } else {
            crossovers
        };

        SearchResults { measurements, winners, crossovers }
    }

    fn enumerate_configs(&self) -> Vec<Config> {
        let mut configs = Vec::new();
        for variant in &self.benchmark.variants {
            if !variant.is_available() {
                continue;
            }
            // P::grid() returns all param combinations
            for params in variant.param_grid() {
                configs.push(Config {
                    variant: variant.name.to_string(),
                    params: params.to_map(),
                });
            }
        }
        configs
    }

    fn find_crossovers(&self, winners: &[Winner]) -> Vec<Crossover> {
        let mut crossovers = Vec::new();
        for dim in self.benchmark.stats_grid.dimensions() {
            crossovers.extend(self.find_crossovers_along_dim(winners, dim));
        }
        crossovers
    }

    fn refine_crossover(&self, crossover: &Crossover) -> Crossover {
        let mut low = crossover.threshold * 0.5;
        let mut high = crossover.threshold * 2.0;

        for _ in 0..10 {
            let mid = (low + high) / 2.0;
            let stats = crossover.stats_at(mid);
            let data = (self.benchmark.generate_fn)(&stats, self.seed);

            let time_below = self.measure_config(&crossover.below, &data);
            let time_above = self.measure_config(&crossover.above, &data);

            if time_below.median_ns < time_above.median_ns {
                low = mid;
            } else {
                high = mid;
            }
        }

        Crossover { threshold: (low + high) / 2.0, ..crossover.clone() }
    }
}
```

---

## CLI Integration

```bash
# Benchmark mode (default params)
$ cargo run -p threshold-runner -- rank --mode bench

# Benchmark at specific stats
$ cargo run -p threshold-runner -- rank --mode bench --len 65536 --density 0.5

# Benchmark with baseline comparison
$ cargo run -p threshold-runner -- rank --mode bench --compare baseline.json

# Save new baseline
$ cargo run -p threshold-runner -- rank --mode bench --save baseline.json

# Search mode (full grid)
$ cargo run -p threshold-runner -- rank --mode search

# Search with refinement
$ cargo run -p threshold-runner -- rank --mode search --refine

# Export results
$ cargo run -p threshold-runner -- rank --mode search --output results.json
```

---

## Measurement Quality

For accurate benchmarking, we follow best practices from Criterion and Divan. See **[measurement.md](measurement.md)** for full details.

**Key Principles:**
1. `black_box` on BOTH inputs AND outputs
2. Separate input generation from timing
3. Batch iterations to amortize timer overhead
4. Handle drops outside timing
5. Warmup phase before measurement

**Checklist:**
- [ ] `black_box` on inputs AND outputs
- [ ] Input generation outside timed section
- [ ] Batch iterations (~100µs per batch)
- [ ] Drops happen outside timed section
- [ ] Warmup phase (~100ms)
- [ ] Remove outliers (IQR method)
- [ ] Report 95% bootstrap confidence intervals

---

## Running & Output

### Defining a Benchmark Binary

```rust
// benches/rank_threshold.rs or src/bin/rank_bench.rs

use clap::{Parser, ValueEnum};
use vortex_threshold_traits::*;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "bench")]
    mode: Mode,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    compare: Option<PathBuf>,

    #[arg(long)]
    save: Option<PathBuf>,

    #[arg(long, default_value = "false")]
    refine: bool,

    // Filter to specific stats
    #[arg(long)]
    len: Option<usize>,

    #[arg(long)]
    density: Option<f64>,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Bench,
    Search,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Build the benchmark (same for both modes)
    let benchmark = ThresholdBench::new("rank")
        .stats(RankStats::compute)
        .generate(RankStats::generate)
        .stats_grid(StatsGrid::new()
            .dimension("len", Scale::log2(6, 20))
            .dimension("density", Scale::steps(0.0, 1.0, 0.1)))
        .variant("naive", rank_naive)
        .variant_with_params("chunked", rank_chunked)
        .variant_with_params("avx2", rank_avx2).requires(&["avx2"])
        .build();

    match args.mode {
        Mode::Bench => run_bench(&benchmark, &args),
        Mode::Search => run_search(&benchmark, &args),
    }
}

fn run_bench(benchmark: &ThresholdBench<...>, args: &Args) -> Result<()> {
    let mut runner = benchmark.bench();

    // Optional: filter to specific stats
    if let (Some(len), Some(density)) = (args.len, args.density) {
        runner = runner.at(RankStats { len, density });
    }

    // Optional: compare to baseline
    if let Some(ref path) = args.compare {
        runner = runner.compare(path);
    }

    // Run
    let report = runner.run();

    // Always print to terminal
    report.print();

    // Optional: export to JSON
    if let Some(ref path) = args.output {
        report.to_json(path)?;
        eprintln!("Results written to: {}", path.display());
    }

    // Optional: save as baseline
    if let Some(ref path) = args.save {
        report.save(path)?;
        eprintln!("Baseline saved to: {}", path.display());
    }

    Ok(())
}

fn run_search(benchmark: &ThresholdBench<...>, args: &Args) -> Result<()> {
    let results = benchmark.search()
        .refine(args.refine)
        .run();

    // Always print to terminal
    results.print();

    // Optional: export to JSON
    if let Some(ref path) = args.output {
        results.to_json(path)?;
        eprintln!("Results written to: {}", path.display());
    }

    Ok(())
}
```

### Terminal Output

#### Bench Mode

```
$ cargo run --release --bin rank_bench -- --mode bench

rank                                    fastest       │ median        │ mean          │ samples
────────────────────────────────────────────────────────────────────────────────────────────────
len=64, density=0.1
  naive                                 12.3 ns       │ 14.5 ns       │ 14.8 ns       │ 100  ★
  chunked(chunk_size=4)                 15.2 ns       │ 17.3 ns       │ 17.9 ns       │ 100
  avx2(unroll=2)                        18.1 ns       │ 20.4 ns       │ 21.2 ns       │ 100

len=1024, density=0.5
  naive                                 823 ns        │ 891 ns        │ 902 ns        │ 100
  chunked(chunk_size=4)                 287 ns        │ 312 ns        │ 318 ns        │ 100  ★
  avx2(unroll=2)                        312 ns        │ 334 ns        │ 341 ns        │ 100

len=65536, density=0.5
  naive                                 52.3 µs       │ 54.8 µs       │ 55.2 µs       │ 100
  chunked(chunk_size=4)                 18.2 µs       │ 19.1 µs       │ 19.4 µs       │ 100
  avx2(unroll=2)                        8.23 µs       │ 8.89 µs       │ 9.12 µs       │ 100  ★

★ = fastest at this stats point
```

#### Bench Mode with Baseline Comparison

```
$ cargo run --release --bin rank_bench -- --mode bench --compare baseline.json

rank                                    median        │ vs baseline
──────────────────────────────────────────────────────────────────────
len=65536, density=0.5
  naive                                 54.8 µs       │   +2.3%
  chunked(chunk_size=4)                 19.1 µs       │   -1.2%
  avx2(unroll=2)                        8.89 µs       │   -5.1%  ▼ faster

▲ = regression (slower), ▼ = improvement (faster)
```

#### Search Mode

```
$ cargo run --release --bin rank_bench -- --mode search

rank - Threshold Search Results
═══════════════════════════════════════════════════════════════════════

Configs tested: 12
  • naive (no params)
  • chunked: chunk_size ∈ {1, 2, 4, 8, 16}
  • avx2: unroll ∈ {1, 2, 4}

Stats grid: 195 points (15 lens × 13 densities)
Total measurements: 2,340
Time elapsed: 4m 23s

Winners by Region
─────────────────
  len < 256                → naive
  len ∈ [256, 2048)        → chunked(chunk_size=4)   when density < 0.3
                           → chunked(chunk_size=8)   when density ≥ 0.3
  len ≥ 2048               → avx2(unroll=2)          when density < 0.5
                           → avx2(unroll=4)          when density ≥ 0.5

Crossovers
──────────
  len=256      (density=0.1):  naive → chunked(chunk_size=4)
  len=2048     (density=0.1):  chunked(chunk_size=4) → avx2(unroll=2)
  len=256      (density=0.5):  naive → chunked(chunk_size=8)
  len=2048     (density=0.5):  chunked(chunk_size=8) → avx2(unroll=4)
  density=0.3  (len=1024):     chunked(chunk_size=4) → chunked(chunk_size=8)
  density=0.5  (len=8192):     avx2(unroll=2) → avx2(unroll=4)
```

### Machine-Readable Output (JSON)

#### BenchReport JSON

```json
{
  "benchmark": "rank",
  "timestamp": "2024-01-15T10:30:00Z",
  "machine": {
    "cpu_class": "IntelSapphire",
    "cpu_model": "Intel Xeon w9-3495X",
    "cpu_cores": 56,
    "memory_gb": 512
  },
  "results": [
    {
      "stats": {
        "len": 1024,
        "density": 0.5
      },
      "measurements": [
        {
          "variant": "naive",
          "params": {},
          "fastest_ns": 823.0,
          "median_ns": 891.0,
          "mean_ns": 902.3,
          "stddev_ns": 45.2,
          "ci_lower_ns": 875.0,
          "ci_upper_ns": 910.0,
          "samples": 100
        },
        {
          "variant": "chunked",
          "params": { "chunk_size": 4 },
          "fastest_ns": 287.0,
          "median_ns": 312.0,
          "mean_ns": 318.4,
          "stddev_ns": 21.3,
          "ci_lower_ns": 298.0,
          "ci_upper_ns": 328.0,
          "samples": 100
        },
        {
          "variant": "avx2",
          "params": { "unroll": 2 },
          "fastest_ns": 312.0,
          "median_ns": 334.0,
          "mean_ns": 341.2,
          "stddev_ns": 18.9,
          "ci_lower_ns": 320.0,
          "ci_upper_ns": 350.0,
          "samples": 100
        }
      ],
      "winner": {
        "variant": "chunked",
        "params": { "chunk_size": 4 }
      }
    }
  ]
}
```

#### SearchResults JSON

```json
{
  "benchmark": "rank",
  "timestamp": "2024-01-15T10:30:00Z",
  "machine": {
    "cpu_class": "IntelSapphire",
    "cpu_model": "Intel Xeon w9-3495X"
  },
  "config": {
    "configs_tested": 12,
    "stats_points": 195,
    "total_measurements": 2340,
    "elapsed_seconds": 263
  },
  "variants": [
    { "name": "naive", "params": [], "available": true },
    { "name": "chunked", "params": ["chunk_size"], "grid_size": 5, "available": true },
    { "name": "avx2", "params": ["unroll"], "grid_size": 3, "available": true, "features": ["avx2"] }
  ],
  "winners": [
    {
      "stats": { "len": 64, "density": 0.1 },
      "config": { "variant": "naive", "params": {} },
      "median_ns": 14.5
    },
    {
      "stats": { "len": 1024, "density": 0.5 },
      "config": { "variant": "chunked", "params": { "chunk_size": 4 } },
      "median_ns": 312.0
    }
  ],
  "crossovers": [
    {
      "dimension": "len",
      "threshold": 256,
      "below": { "variant": "naive", "params": {} },
      "above": { "variant": "chunked", "params": { "chunk_size": 4 } },
      "context": { "density": 0.1 },
      "confidence": 0.95
    },
    {
      "dimension": "density",
      "threshold": 0.3,
      "below": { "variant": "chunked", "params": { "chunk_size": 4 } },
      "above": { "variant": "chunked", "params": { "chunk_size": 8 } },
      "context": { "len": 1024 },
      "confidence": 0.92
    }
  ],
  "all_measurements": [
    {
      "stats": { "len": 64, "density": 0.1 },
      "config": { "variant": "naive", "params": {} },
      "median_ns": 14.5,
      "ci_lower_ns": 13.8,
      "ci_upper_ns": 15.2
    }
  ]
}
```

### Implementation

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct BenchReport {
    pub benchmark: String,
    pub timestamp: String,
    pub machine: MachineInfo,
    pub results: Vec<StatsResult>,
}

#[derive(Serialize, Deserialize)]
pub struct MachineInfo {
    pub cpu_class: CpuClass,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: usize,
}

#[derive(Serialize, Deserialize)]
pub struct StatsResult {
    pub stats: BTreeMap<String, f64>,
    pub measurements: Vec<VariantMeasurement>,
    pub winner: Option<Config>,
}

#[derive(Serialize, Deserialize)]
pub struct VariantMeasurement {
    pub variant: String,
    pub params: BTreeMap<String, ParamValue>,
    pub fastest_ns: f64,
    pub median_ns: f64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub ci_lower_ns: f64,
    pub ci_upper_ns: f64,
    pub samples: usize,
}

impl BenchReport {
    /// Print human-readable table to stdout
    pub fn print(&self) {
        println!("{}", self.benchmark);
        println!("{}", "─".repeat(80));

        for result in &self.results {
            // Print stats header
            let stats_str: Vec<_> = result.stats.iter()
                .map(|(k, v)| format!("{}={}", k, format_value(*v)))
                .collect();
            println!("\n{}", stats_str.join(", "));

            // Print measurements
            for m in &result.measurements {
                let params_str = format_params(&m.params);
                let name = if params_str.is_empty() {
                    m.variant.clone()
                } else {
                    format!("{}({})", m.variant, params_str)
                };

                let winner_mark = if result.winner.as_ref()
                    .map(|w| w.variant == m.variant && w.params == m.params)
                    .unwrap_or(false)
                {
                    " ★"
                } else {
                    ""
                };

                println!(
                    "  {:35} {:>12} │ {:>12} │ {:>12} │ {:>5}{}",
                    name,
                    format_time(m.fastest_ns),
                    format_time(m.median_ns),
                    format_time(m.mean_ns),
                    m.samples,
                    winner_mark
                );
            }
        }
    }

    /// Export to JSON file
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Save as baseline (alias for to_json with semantic meaning)
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.to_json(path)
    }

    /// Load baseline for comparison
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(serde_json::from_reader(file)?)
    }
}

impl SearchResults {
    /// Print human-readable summary to stdout
    pub fn print(&self) {
        println!("{} - Threshold Search Results", self.benchmark);
        println!("{}", "═".repeat(60));

        println!("\nConfigs tested: {}", self.variants.len());
        for v in &self.variants {
            if v.params.is_empty() {
                println!("  • {} (no params)", v.name);
            } else {
                println!("  • {}: {} ∈ {:?}", v.name, v.params.join(", "), v.grid_values);
            }
        }

        println!("\nStats grid: {} points", self.stats_points_count);
        println!("Total measurements: {}", self.total_measurements);

        println!("\nWinners by Region");
        println!("{}", "─".repeat(20));
        self.print_winners_by_region();

        println!("\nCrossovers");
        println!("{}", "─".repeat(10));
        for c in &self.crossovers {
            println!(
                "  {}={:<6} ({}): {} → {}",
                c.dimension,
                format_value(c.threshold),
                format_context(&c.context),
                format_config(&c.below),
                format_config(&c.above)
            );
        }
    }

    /// Export to JSON file
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

fn format_time(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:.1} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

fn format_params(params: &BTreeMap<String, ParamValue>) -> String {
    params.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(", ")
}
```

### CLI Reference

```bash
# === Bench Mode ===

# Run benchmark, display to terminal
cargo run --release --bin rank_bench -- --mode bench

# Run at specific stats point
cargo run --release --bin rank_bench -- --mode bench --len 65536 --density 0.5

# Export results to JSON
cargo run --release --bin rank_bench -- --mode bench --output results.json

# Save as baseline for future comparison
cargo run --release --bin rank_bench -- --mode bench --save baseline.json

# Compare against baseline
cargo run --release --bin rank_bench -- --mode bench --compare baseline.json

# Compare and export
cargo run --release --bin rank_bench -- --mode bench --compare baseline.json --output diff.json


# === Search Mode ===

# Run search, display to terminal
cargo run --release --bin rank_bench -- --mode search

# Run with binary search refinement
cargo run --release --bin rank_bench -- --mode search --refine

# Export results to JSON
cargo run --release --bin rank_bench -- --mode search --output thresholds.json

# Refine and export
cargo run --release --bin rank_bench -- --mode search --refine --output thresholds.json
```

---

## Files

```
vortex-threshold-traits/
├── src/
│   ├── lib.rs          # Core traits (ParamGrid, ParamValue)
│   ├── builder.rs      # ThresholdBench builder
│   ├── bench.rs        # BenchRunner, BenchReport
│   ├── search.rs       # SearchRunner, SearchResults
│   └── stats.rs        # StatsGrid, Scale
├── Cargo.toml

vortex-threshold-derive/
├── src/
│   └── lib.rs          # #[derive(ParamGrid)] proc macro
├── Cargo.toml

vortex-threshold-runner/
├── src/
│   ├── main.rs         # CLI entry point
│   ├── measure.rs      # Measurement helpers (black_box, stats)
│   └── output.rs       # Terminal formatting
├── Cargo.toml
```

---

## Next Steps

1. [ ] Implement `ParamGrid` trait in `vortex-threshold-traits`
2. [ ] Create `vortex-threshold-derive` crate with `#[derive(ParamGrid)]`
3. [ ] Implement `ThresholdBench` builder
4. [ ] Implement `BenchRunner` with Divan-style output
5. [ ] Implement `SearchRunner` with crossover detection
6. [ ] Add `black_box` and statistical analysis
7. [ ] Implement CLI with mode selection
8. [ ] Create example with `rank` benchmark
