# Measurement Quality & Iteration Mechanics

This document covers how to accurately measure benchmark iterations, avoiding noise from allocations, compiler optimizations, and timer overhead.

## Key Principles

Based on [Criterion](https://docs.rs/criterion/latest/criterion/struct.Bencher.html), [Divan](https://docs.rs/divan/latest/divan/struct.Bencher.html), and [Guillaume Endignoux's analysis](https://gendignoux.com/blog/2022/01/31/rust-benchmarks.html):

1. **`black_box` on BOTH inputs AND outputs** - Prevents compiler from pre-computing
2. **Separate input generation from timing** - Don't measure allocation
3. **Batch iterations** - Amortize timer overhead for fast functions
4. **Handle drops outside timing** - Don't measure deallocation
5. **Warmup phase** - Stabilize CPU frequency, fill caches

---

## The Problem: Naive Measurement

```rust
// ❌ WRONG - multiple issues:
let start = Instant::now();
let result = function(input);  // compiler may optimize away
let elapsed = start.elapsed(); // timer overhead dominates for fast functions
```

**Issues:**
- Compiler may pre-compute `function(input)` outside the loop
- Timer precision (~1µs) dominates for nanosecond functions
- Measures allocation and drop time

---

## Why `black_box` on BOTH Inputs and Outputs

From [Guillaume Endignoux's analysis](https://gendignoux.com/blog/2022/01/31/rust-benchmarks.html):

```rust
// ❌ WRONG - compiler sees same inputs every iteration
//           may pre-compute result outside the loop
for _ in 0..iterations {
    black_box(multiply(a, b));  // compiler knows a,b are constant!
}

// ✅ CORRECT - compiler can't assume anything about inputs
for _ in 0..iterations {
    black_box(multiply(black_box(a), black_box(b)));
}
```

**Without input `black_box`, the compiler can:**
1. Recognize inputs are loop-invariant
2. Compute `multiply(a, b)` once before the loop
3. Replace the loop body with just writing a constant

This produces "0 nanoseconds" benchmarks for what should be real computation.

---

## Batching: Why and How

### The Timer Precision Problem

System clocks have limited precision (~1µs). For a function that takes 10ns:
- Single measurement: "0ns, 0ns, 0ns, 0ns, 5µs, 0ns..."
- Batched (1000 iterations): "10µs" → 10ns per iteration

### Batch Size Tradeoffs

From [Criterion BatchSize](https://docs.rs/criterion/latest/criterion/enum.BatchSize.html):

| BatchSize | Overhead | Memory | Use Case |
|-----------|----------|--------|----------|
| Small | ~500 ps | High (millions of inputs) | Default, small data |
| Large | ~750 ps | Medium | Larger data structures |
| PerIteration | ~350 ns | Low (one at a time) | Huge data, file handles |

---

## Input Generation & Drops

### Criterion's Approach

```rust
// iter_batched: setup runs OUTSIDE timing, routine consumes input
b.iter_batched(
    || data.clone(),           // Setup: generate input (NOT timed)
    |data| sort(data),         // Routine: measured (consumes input)
    BatchSize::SmallInput
);

// iter_batched_ref: routine borrows input, drop happens AFTER timing
b.iter_batched_ref(
    || data.clone(),           // Setup: generate input
    |data| sort(data),         // Routine: measured (borrows &mut)
    BatchSize::SmallInput
);
// data dropped here, AFTER timing
```

### Divan's Approach

```rust
// with_inputs: generator runs OUTSIDE timing
bencher
    .with_inputs(|| generate_data())
    .bench_values(|data| process(data));  // consumes

bencher
    .with_inputs(|| generate_data())
    .bench_refs(|data| process(data));    // borrows &mut
```

**Divan optimizations:**
- If output is `!needs_drop` (e.g., `()`, `i32`), no storage allocated
- Inputs and outputs stored contiguously for cache-friendly access
- Uses CPU timestamp counter when available (lower overhead than `Instant`)

---

## Implementation

### Core Measurer

```rust
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Batch size controls memory vs overhead tradeoff
#[derive(Clone, Copy)]
pub enum BatchSize {
    /// ~500ps overhead, millions of copies in memory
    Small,
    /// ~750ps overhead, reduced memory pressure
    Large,
    /// ~350ns overhead, one input at a time
    PerIteration,
    /// Custom batch count
    NumBatches(usize),
}

pub struct Measurer {
    pub warmup_time: Duration,
    pub measurement_time: Duration,
    pub batch_size: BatchSize,
}

impl Default for Measurer {
    fn default() -> Self {
        Self {
            warmup_time: Duration::from_millis(100),
            measurement_time: Duration::from_millis(500),
            batch_size: BatchSize::Small,
        }
    }
}

impl Measurer {
    /// Measure a function that borrows input
    pub fn measure<I, O, S, R>(&self, setup: S, routine: R) -> MeasurementResult
    where
        S: Fn() -> I,      // Input generator (NOT timed)
        R: Fn(&I) -> O,    // Routine to measure (borrows input)
    {
        // === Phase 1: Warmup ===
        self.warmup(&setup, &routine);

        // === Phase 2: Estimate batch size ===
        let iters_per_batch = self.estimate_iterations(&setup, &routine);

        // === Phase 3: Batched measurement ===
        let samples = self.collect_samples(&setup, &routine, iters_per_batch);

        // === Phase 4: Statistics ===
        Self::compute_stats(&samples)
    }

    /// Measure a function that consumes input (like sorting)
    pub fn measure_consuming<I, O, S, R>(&self, setup: S, routine: R) -> MeasurementResult
    where
        S: Fn() -> I,      // Input generator
        R: Fn(I) -> O,     // Routine (takes ownership)
    {
        self.warmup_consuming(&setup, &routine);
        let iters_per_batch = self.estimate_iterations_consuming(&setup, &routine);
        let samples = self.collect_samples_consuming(&setup, &routine, iters_per_batch);
        Self::compute_stats(&samples)
    }

    fn warmup<I, O, S, R>(&self, setup: &S, routine: &R)
    where
        S: Fn() -> I,
        R: Fn(&I) -> O,
    {
        let start = Instant::now();
        while start.elapsed() < self.warmup_time {
            let input = setup();
            black_box(routine(black_box(&input)));
            // input dropped here, outside any timing
        }
    }

    fn warmup_consuming<I, O, S, R>(&self, setup: &S, routine: &R)
    where
        S: Fn() -> I,
        R: Fn(I) -> O,
    {
        let start = Instant::now();
        while start.elapsed() < self.warmup_time {
            let input = setup();
            black_box(routine(black_box(input)));
        }
    }

    fn estimate_iterations<I, O, S, R>(&self, setup: &S, routine: &R) -> usize
    where
        S: Fn() -> I,
        R: Fn(&I) -> O,
    {
        let input = setup();

        // Time 10 iterations to estimate
        let start = Instant::now();
        for _ in 0..10 {
            black_box(routine(black_box(&input)));
        }
        let elapsed = start.elapsed();
        let per_iter_ns = elapsed.as_nanos() / 10;

        // Target ~100µs per batch (balances overhead vs memory)
        let target_batch_ns = 100_000;
        let iters = (target_batch_ns / per_iter_ns.max(1)) as usize;

        // Apply batch size constraints
        match self.batch_size {
            BatchSize::Small => iters.clamp(100, 100_000),
            BatchSize::Large => iters.clamp(10, 10_000),
            BatchSize::PerIteration => 1,
            BatchSize::NumBatches(n) => n,
        }
    }

    fn estimate_iterations_consuming<I, O, S, R>(&self, setup: &S, routine: &R) -> usize
    where
        S: Fn() -> I,
        R: Fn(I) -> O,
    {
        // Time 10 iterations
        let start = Instant::now();
        for _ in 0..10 {
            let input = setup();
            black_box(routine(black_box(input)));
        }
        let elapsed = start.elapsed();
        let per_iter_ns = elapsed.as_nanos() / 10;

        let target_batch_ns = 100_000;
        let iters = (target_batch_ns / per_iter_ns.max(1)) as usize;

        match self.batch_size {
            BatchSize::Small => iters.clamp(100, 100_000),
            BatchSize::Large => iters.clamp(10, 10_000),
            BatchSize::PerIteration => 1,
            BatchSize::NumBatches(n) => n,
        }
    }

    fn collect_samples<I, O, S, R>(
        &self,
        setup: &S,
        routine: &R,
        iters_per_batch: usize,
    ) -> Vec<f64>
    where
        S: Fn() -> I,
        R: Fn(&I) -> O,
    {
        let mut samples = Vec::new();
        let measurement_start = Instant::now();

        while measurement_start.elapsed() < self.measurement_time {
            // Generate batch of inputs OUTSIDE timing
            let inputs: Vec<I> = (0..iters_per_batch).map(|_| setup()).collect();

            // Time the batch
            let batch_start = Instant::now();
            for input in &inputs {
                // black_box on BOTH input and output - CRITICAL!
                black_box(routine(black_box(input)));
            }
            let batch_elapsed = batch_start.elapsed();

            // Record per-iteration time
            let per_iter_ns = batch_elapsed.as_nanos() as f64 / iters_per_batch as f64;
            samples.push(per_iter_ns);

            // Inputs dropped here, OUTSIDE timing
        }

        samples
    }

    fn collect_samples_consuming<I, O, S, R>(
        &self,
        setup: &S,
        routine: &R,
        iters_per_batch: usize,
    ) -> Vec<f64>
    where
        S: Fn() -> I,
        R: Fn(I) -> O,
    {
        let mut samples = Vec::new();
        let measurement_start = Instant::now();

        while measurement_start.elapsed() < self.measurement_time {
            // Generate batch
            let inputs: Vec<I> = (0..iters_per_batch).map(|_| setup()).collect();

            // Time the batch - routine consumes inputs
            let batch_start = Instant::now();
            for input in inputs {
                black_box(routine(black_box(input)));
            }
            let batch_elapsed = batch_start.elapsed();

            let per_iter_ns = batch_elapsed.as_nanos() as f64 / iters_per_batch as f64;
            samples.push(per_iter_ns);
        }

        samples
    }

    fn compute_stats(samples: &[f64]) -> MeasurementResult {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Remove outliers (IQR method)
        let q1 = percentile(&sorted, 25.0);
        let q3 = percentile(&sorted, 75.0);
        let iqr = q3 - q1;
        let clean: Vec<f64> = sorted
            .iter()
            .filter(|&&x| x >= q1 - 1.5 * iqr && x <= q3 + 1.5 * iqr)
            .copied()
            .collect();

        if clean.is_empty() {
            return MeasurementResult::default();
        }

        // Compute statistics on clean data
        let median = percentile(&clean, 50.0);
        let mean = clean.iter().sum::<f64>() / clean.len() as f64;
        let variance =
            clean.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / clean.len() as f64;
        let stddev = variance.sqrt();
        let min = clean.first().copied().unwrap_or(0.0);
        let max = clean.last().copied().unwrap_or(0.0);

        // Bootstrap confidence interval
        let (ci_lower, ci_upper) = bootstrap_ci(&clean, 0.95, 1000);

        MeasurementResult {
            median_ns: median,
            mean_ns: mean,
            stddev_ns: stddev,
            min_ns: min,
            max_ns: max,
            ci_lower_ns: ci_lower,
            ci_upper_ns: ci_upper,
            samples: clean.len(),
            outliers_removed: sorted.len() - clean.len(),
        }
    }
}
```

### Statistical Helpers

```rust
#[derive(Debug, Clone, Default)]
pub struct MeasurementResult {
    pub median_ns: f64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub ci_lower_ns: f64,
    pub ci_upper_ns: f64,
    pub samples: usize,
    pub outliers_removed: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn bootstrap_ci(samples: &[f64], confidence: f64, iterations: usize) -> (f64, f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut bootstrap_medians = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Resample with replacement
        let resample: Vec<f64> = (0..samples.len())
            .map(|_| samples[rng.gen_range(0..samples.len())])
            .collect();
        let mut sorted = resample;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_medians.push(percentile(&sorted, 50.0));
    }

    bootstrap_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64) as usize;

    (
        bootstrap_medians.get(lower_idx).copied().unwrap_or(0.0),
        bootstrap_medians.get(upper_idx).copied().unwrap_or(0.0),
    )
}
```

---

## Integration with ThresholdBench

```rust
impl<D, S, O> SearchRunner<'_, D, S, O> {
    fn measure_config(&self, config: &Config, stats: &S) -> MeasurementResult {
        let measurer = Measurer::default();

        // Setup: generates fresh data with target stats
        // Runs OUTSIDE the timed section
        let setup = || {
            let seed = rand::random();
            (self.benchmark.generate_fn)(stats, seed)
        };

        // Routine: the actual computation we're measuring
        let routine = |data: &D| self.run_config(config, data);

        measurer.measure(setup, routine)
    }
}
```

---

## Checklist

When implementing measurement:

- [ ] `black_box` on inputs AND outputs
- [ ] Input generation runs outside timed section
- [ ] Batch iterations to amortize timer overhead (~100µs per batch)
- [ ] Drops happen outside timed section
- [ ] Warmup phase before measurement (~100ms)
- [ ] Remove outliers using IQR method
- [ ] Report confidence intervals (95% bootstrap CI)
- [ ] Record sample count and outliers removed

---

## References

- [Criterion Bencher docs](https://docs.rs/criterion/latest/criterion/struct.Bencher.html)
- [Criterion BatchSize docs](https://docs.rs/criterion/latest/criterion/enum.BatchSize.html)
- [Divan Bencher docs](https://docs.rs/divan/latest/divan/struct.Bencher.html)
- [Why my Rust benchmarks were wrong](https://gendignoux.com/blog/2022/01/31/rust-benchmarks.html) - Guillaume Endignoux
- [The Rust Performance Book - Benchmarking](https://nnethercote.github.io/perf-book/benchmarking.html)
