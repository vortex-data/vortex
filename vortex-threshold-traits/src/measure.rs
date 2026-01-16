//! Measurement utilities for accurate benchmarking.
//!
//! This module provides tools for measuring function execution time with
//! proper isolation from noise sources like allocations, compiler optimizations,
//! and timer overhead.
//!
//! # Key Principles
//!
//! Based on [Criterion](https://docs.rs/criterion) and [Divan](https://docs.rs/divan):
//!
//! 1. **`black_box` on BOTH inputs AND outputs** - Prevents compiler from pre-computing
//! 2. **Separate input generation from timing** - Don't measure allocation
//! 3. **Batch iterations** - Amortize timer overhead for fast functions
//! 4. **Handle drops outside timing** - Don't measure deallocation
//! 5. **Warmup phase** - Stabilize CPU frequency, fill caches
//!
//! # Example
//!
//! ```ignore
//! use vortex_threshold_traits::measure::{Measurer, BatchSize};
//!
//! let measurer = Measurer::default();
//!
//! let result = measurer.measure(
//!     || generate_input(),           // Setup: NOT timed
//!     |input| compute(input),        // Routine: timed
//! );
//!
//! println!("Median: {:.2}ns", result.median_ns);
//! ```

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

/// Batch size controls the memory vs overhead tradeoff.
///
/// Based on [Criterion's BatchSize](https://docs.rs/criterion/latest/criterion/enum.BatchSize.html).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchSize {
    /// ~500ps overhead per iteration, but stores many inputs in memory.
    /// Best for small inputs where memory isn't a concern.
    #[default]
    Small,

    /// ~750ps overhead per iteration, reduced memory pressure.
    /// Good for larger data structures.
    Large,

    /// ~350ns overhead per iteration, generates one input at a time.
    /// Use for huge data structures or resources like file handles.
    PerIteration,

    /// Custom number of iterations per batch.
    NumIterations(usize),
}

impl BatchSize {
    /// Returns the target number of iterations per batch.
    fn target_iterations(&self) -> usize {
        match self {
            BatchSize::Small => 10_000,
            BatchSize::Large => 1_000,
            BatchSize::PerIteration => 1,
            BatchSize::NumIterations(n) => *n,
        }
    }
}

/// Result of a measurement run.
#[derive(Debug, Clone, Default)]
pub struct MeasurementResult {
    /// Median time per iteration in nanoseconds.
    pub median_ns: f64,
    /// Mean time per iteration in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds.
    pub stddev_ns: f64,
    /// Minimum time in nanoseconds.
    pub min_ns: f64,
    /// Maximum time in nanoseconds.
    pub max_ns: f64,
    /// Lower bound of 95% confidence interval.
    pub ci_lower_ns: f64,
    /// Upper bound of 95% confidence interval.
    pub ci_upper_ns: f64,
    /// Number of samples collected (after outlier removal).
    pub samples: usize,
    /// Number of outliers removed.
    pub outliers_removed: usize,
}

/// Configurable measurer for benchmark timing.
#[derive(Debug, Clone)]
pub struct Measurer {
    /// Time to spend warming up before measurement.
    pub warmup_time: Duration,
    /// Time to spend collecting measurements.
    pub measurement_time: Duration,
    /// Batch size strategy.
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
    /// Creates a new measurer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the warmup time.
    #[must_use]
    pub fn warmup_time(mut self, duration: Duration) -> Self {
        self.warmup_time = duration;
        self
    }

    /// Sets the measurement time.
    #[must_use]
    pub fn measurement_time(mut self, duration: Duration) -> Self {
        self.measurement_time = duration;
        self
    }

    /// Sets the batch size strategy.
    #[must_use]
    pub fn batch_size(mut self, batch_size: BatchSize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Measures a function that borrows its input.
    ///
    /// The setup function generates input data and is NOT timed.
    /// The routine receives a reference to the input and IS timed.
    /// Input drops happen OUTSIDE the timed section.
    ///
    /// # Arguments
    ///
    /// * `setup` - Function that generates input data (not timed)
    /// * `routine` - Function to measure (receives `&I`, is timed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = measurer.measure(
    ///     || vec![1u64; 1000],      // Setup
    ///     |data| data.iter().sum(), // Routine
    /// );
    /// ```
    pub fn measure<I, O, S, R>(&self, setup: S, routine: R) -> MeasurementResult
    where
        S: Fn() -> I,
        R: Fn(&I) -> O,
    {
        // Phase 1: Warmup
        self.warmup(&setup, &routine);

        // Phase 2: Estimate iterations per batch
        let iters_per_batch = self.estimate_iterations(&setup, &routine);

        // Phase 3: Collect samples
        let samples = self.collect_samples(&setup, &routine, iters_per_batch);

        // Phase 4: Compute statistics
        compute_stats(&samples)
    }

    /// Measures a function that consumes its input.
    ///
    /// Use this for functions that take ownership, like sorting.
    ///
    /// # Arguments
    ///
    /// * `setup` - Function that generates input data (not timed)
    /// * `routine` - Function to measure (takes ownership of `I`, is timed)
    pub fn measure_consuming<I, O, S, R>(&self, setup: S, routine: R) -> MeasurementResult
    where
        S: Fn() -> I,
        R: Fn(I) -> O,
    {
        // Phase 1: Warmup
        self.warmup_consuming(&setup, &routine);

        // Phase 2: Estimate iterations per batch
        let iters_per_batch = self.estimate_iterations_consuming(&setup, &routine);

        // Phase 3: Collect samples
        let samples = self.collect_samples_consuming(&setup, &routine, iters_per_batch);

        // Phase 4: Compute statistics
        compute_stats(&samples)
    }

    fn warmup<I, O, S, R>(&self, setup: &S, routine: &R)
    where
        S: Fn() -> I,
        R: Fn(&I) -> O,
    {
        let start = Instant::now();
        while start.elapsed() < self.warmup_time {
            let input = setup();
            // black_box on BOTH input and output
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

        // Time 10 iterations to estimate per-iteration time
        let start = Instant::now();
        for _ in 0..10 {
            black_box(routine(black_box(&input)));
        }
        let elapsed = start.elapsed();
        let per_iter_ns = elapsed.as_nanos() / 10;

        // Target ~100µs per batch (balances overhead vs memory)
        let target_batch_ns: u128 = 100_000;
        let estimated = (target_batch_ns / per_iter_ns.max(1)) as usize;

        // Clamp to batch size constraints
        let target = self.batch_size.target_iterations();
        match self.batch_size {
            BatchSize::Small => estimated.clamp(100, target),
            BatchSize::Large => estimated.clamp(10, target),
            BatchSize::PerIteration => 1,
            BatchSize::NumIterations(n) => n,
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

        let target_batch_ns: u128 = 100_000;
        let estimated = (target_batch_ns / per_iter_ns.max(1)) as usize;

        let target = self.batch_size.target_iterations();
        match self.batch_size {
            BatchSize::Small => estimated.clamp(100, target),
            BatchSize::Large => estimated.clamp(10, target),
            BatchSize::PerIteration => 1,
            BatchSize::NumIterations(n) => n,
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
            // Generate batch of inputs OUTSIDE timing
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
}

/// Computes statistics from collected samples.
fn compute_stats(samples: &[f64]) -> MeasurementResult {
    if samples.is_empty() {
        return MeasurementResult::default();
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Remove outliers using IQR method
    let q1 = percentile(&sorted, 25.0);
    let q3 = percentile(&sorted, 75.0);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let clean: Vec<f64> = sorted
        .iter()
        .filter(|&&x| x >= lower_fence && x <= upper_fence)
        .copied()
        .collect();

    let outliers_removed = sorted.len() - clean.len();

    if clean.is_empty() {
        return MeasurementResult {
            outliers_removed,
            ..Default::default()
        };
    }

    // Compute statistics on clean data
    let median = percentile(&clean, 50.0);
    let mean = clean.iter().sum::<f64>() / clean.len() as f64;
    let variance = clean.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / clean.len() as f64;
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
        outliers_removed,
    }
}

/// Computes the percentile of a sorted slice.
#[allow(clippy::cast_possible_truncation)]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Computes bootstrap confidence interval for the median.
#[allow(clippy::cast_possible_truncation)]
fn bootstrap_ci(samples: &[f64], confidence: f64, iterations: usize) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    // Simple LCG random number generator (no external dependency)
    let mut seed: u64 = 0x5DEECE66D;
    let mut next_random = || {
        seed = seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        ((seed >> 16) as usize) % samples.len()
    };

    let mut bootstrap_medians = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Resample with replacement
        let mut resample: Vec<f64> = (0..samples.len()).map(|_| samples[next_random()]).collect();
        resample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bootstrap_medians.push(percentile(&resample, 50.0));
    }

    bootstrap_medians.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64) as usize;

    (
        bootstrap_medians.get(lower_idx).copied().unwrap_or(0.0),
        bootstrap_medians
            .get(upper_idx.min(bootstrap_medians.len() - 1))
            .copied()
            .unwrap_or(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_size_defaults() {
        assert_eq!(BatchSize::default(), BatchSize::Small);
        assert_eq!(BatchSize::Small.target_iterations(), 10_000);
        assert_eq!(BatchSize::Large.target_iterations(), 1_000);
        assert_eq!(BatchSize::PerIteration.target_iterations(), 1);
        assert_eq!(BatchSize::NumIterations(500).target_iterations(), 500);
    }

    #[test]
    fn test_measurer_builder() {
        let m = Measurer::new()
            .warmup_time(Duration::from_millis(50))
            .measurement_time(Duration::from_millis(100))
            .batch_size(BatchSize::Large);

        assert_eq!(m.warmup_time, Duration::from_millis(50));
        assert_eq!(m.measurement_time, Duration::from_millis(100));
        assert_eq!(m.batch_size, BatchSize::Large);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 0.001);
        assert!((percentile(&data, 0.0) - 1.0).abs() < 0.001);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_basic() {
        let samples: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let result = compute_stats(&samples);

        assert!(result.samples > 0);
        assert!(result.mean_ns > 0.0);
        assert!(result.median_ns > 0.0);
        assert!(result.ci_lower_ns <= result.median_ns);
        assert!(result.ci_upper_ns >= result.median_ns);
    }

    #[test]
    fn test_compute_stats_outliers() {
        // Data with clear outliers
        let mut samples: Vec<f64> = (0..100).map(|_| 100.0).collect();
        samples.push(10000.0); // outlier
        samples.push(1.0); // outlier

        let result = compute_stats(&samples);

        // Outliers should be removed
        assert!(result.outliers_removed > 0);
        // Median should be close to 100
        assert!((result.median_ns - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_measure_basic() {
        let measurer = Measurer::new()
            .warmup_time(Duration::from_millis(10))
            .measurement_time(Duration::from_millis(50));

        let result = measurer.measure(|| 42u64, |&x| x.wrapping_mul(x));

        assert!(result.samples > 0);
        assert!(result.median_ns > 0.0);
    }

    #[test]
    fn test_measure_consuming() {
        let measurer = Measurer::new()
            .warmup_time(Duration::from_millis(10))
            .measurement_time(Duration::from_millis(50));

        let result = measurer.measure_consuming(|| vec![1u64; 100], |v| v.into_iter().sum::<u64>());

        assert!(result.samples > 0);
        assert!(result.median_ns > 0.0);
    }
}
