//! Stats-aware benchmark builder for threshold finding.
//!
//! This module provides a builder API for defining benchmarks that search
//! over data statistics (like length and density) rather than just input size.
//!
//! # Example
//!
//! ```ignore
//! use vortex_threshold_traits::{StatsBench, StatsGrid, Scale};
//!
//! let benchmark = StatsBench::<RankData, RankStats, usize>::new("rank")
//!     .stats(RankStats::compute)
//!     .generate(|stats, seed| stats.generate(seed))
//!     .stats_grid(StatsGrid::new()
//!         .dimension("len", Scale::log2(6, 12))
//!         .dimension("density", Scale::steps(0.0, 1.0, 5)))
//!     .variant("naive", rank_naive)
//!     .variant("simd", rank_simd)
//!     .build();
//!
//! // Quick benchmark at specific stats
//! benchmark.bench().at(&stats).run();
//!
//! // Full grid search
//! benchmark.search().run();
//! ```

use std::fmt::Debug;
use std::sync::Arc;

use crate::MeasurementResult;
use crate::Measurer;
use crate::StatsGrid;
use crate::StatsPoint;

/// Function that computes stats from data.
type StatsFn<D, S> = Arc<dyn Fn(&D) -> S + Send + Sync>;

/// Function that generates data from stats.
type GenerateFn<D, S> = Arc<dyn Fn(&S, u64) -> D + Send + Sync>;

/// Function that runs a variant on data.
type VariantFn<D, O> = Arc<dyn Fn(&D) -> O + Send + Sync>;

/// A variant in the benchmark.
struct VariantEntry<D, O> {
    name: String,
    #[allow(dead_code)]
    features: Vec<String>,
    func: VariantFn<D, O>,
}

/// Builder for stats-aware threshold benchmarks.
///
/// Type parameters:
/// - `D`: Data type (raw input to the algorithm)
/// - `S`: Stats type (computed from data, used for dispatch)
/// - `O`: Output type (result of running the algorithm)
pub struct StatsBench<D, S, O> {
    name: String,
    stats_fn: Option<StatsFn<D, S>>,
    generate_fn: Option<GenerateFn<D, S>>,
    stats_grid: StatsGrid,
    baseline: Option<(String, VariantFn<D, O>)>,
    variants: Vec<VariantEntry<D, O>>,
}

impl<D, S, O> StatsBench<D, S, O>
where
    D: Clone + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static,
    O: Clone + PartialEq + Debug + Send + Sync + 'static,
{
    /// Creates a new stats-aware benchmark.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stats_fn: None,
            generate_fn: None,
            stats_grid: StatsGrid::new(),
            baseline: None,
            variants: Vec::new(),
        }
    }

    /// Sets the function that computes stats from data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// .stats(|data: &RankData| RankStats {
    ///     len: data.bitmap.len(),
    ///     density: compute_density(&data.bitmap),
    /// })
    /// ```
    #[must_use]
    pub fn stats<F>(mut self, f: F) -> Self
    where
        F: Fn(&D) -> S + Send + Sync + 'static,
    {
        self.stats_fn = Some(Arc::new(f));
        self
    }

    /// Sets the function that generates data from stats.
    ///
    /// # Example
    ///
    /// ```ignore
    /// .generate(|stats: &RankStats, seed: u64| {
    ///     RankData::generate_with_stats(stats, seed)
    /// })
    /// ```
    #[must_use]
    pub fn generate<F>(mut self, f: F) -> Self
    where
        F: Fn(&S, u64) -> D + Send + Sync + 'static,
    {
        self.generate_fn = Some(Arc::new(f));
        self
    }

    /// Sets the stats grid defining the search space.
    ///
    /// # Example
    ///
    /// ```ignore
    /// .stats_grid(StatsGrid::new()
    ///     .dimension("len", Scale::log2(6, 12))
    ///     .dimension("density", Scale::steps(0.0, 1.0, 5)))
    /// ```
    #[must_use]
    pub fn stats_grid(mut self, grid: StatsGrid) -> Self {
        self.stats_grid = grid;
        self
    }

    /// Sets the baseline variant (used as ground truth).
    #[must_use]
    pub fn baseline<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&D) -> O + Send + Sync + 'static,
    {
        self.baseline = Some((name.into(), Arc::new(f)));
        self
    }

    /// Adds a variant.
    #[must_use]
    pub fn variant<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&D) -> O + Send + Sync + 'static,
    {
        self.variants.push(VariantEntry {
            name: name.into(),
            features: Vec::new(),
            func: Arc::new(f),
        });
        self
    }

    /// Adds a variant with required CPU features.
    #[must_use]
    pub fn variant_with_features<F>(
        mut self,
        name: impl Into<String>,
        features: &[&str],
        f: F,
    ) -> Self
    where
        F: Fn(&D) -> O + Send + Sync + 'static,
    {
        self.variants.push(VariantEntry {
            name: name.into(),
            features: features.iter().map(|s| (*s).to_string()).collect(),
            func: Arc::new(f),
        });
        self
    }

    /// Builds the benchmark.
    ///
    /// # Panics
    ///
    /// Panics if stats, generate, or baseline are not set.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn build(self) -> BuiltStatsBench<D, S, O> {
        let stats_fn = self
            .stats_fn
            .expect("StatsBench requires .stats() to be set");
        let generate_fn = self
            .generate_fn
            .expect("StatsBench requires .generate() to be set");
        let (baseline_name, baseline_fn) = self
            .baseline
            .expect("StatsBench requires .baseline() to be set");

        BuiltStatsBench {
            name: self.name,
            stats_fn,
            generate_fn,
            stats_grid: self.stats_grid,
            baseline_name,
            baseline_fn,
            variants: self.variants,
        }
    }
}

/// A built stats-aware benchmark ready for running.
pub struct BuiltStatsBench<D, S, O> {
    name: String,
    stats_fn: StatsFn<D, S>,
    generate_fn: GenerateFn<D, S>,
    stats_grid: StatsGrid,
    baseline_name: String,
    baseline_fn: VariantFn<D, O>,
    variants: Vec<VariantEntry<D, O>>,
}

impl<D, S, O> BuiltStatsBench<D, S, O>
where
    D: Clone + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static,
    O: Clone + PartialEq + Debug + Send + Sync + 'static,
{
    /// Returns the benchmark name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the stats grid.
    #[must_use]
    pub fn stats_grid(&self) -> &StatsGrid {
        &self.stats_grid
    }

    /// Returns the variant names.
    #[must_use]
    pub fn variant_names(&self) -> Vec<&str> {
        let mut names = vec![self.baseline_name.as_str()];
        names.extend(self.variants.iter().map(|v| v.name.as_str()));
        names
    }

    /// Computes stats from data.
    pub fn compute_stats(&self, data: &D) -> S {
        (self.stats_fn)(data)
    }

    /// Generates data from stats.
    pub fn generate_data(&self, stats: &S, seed: u64) -> D {
        (self.generate_fn)(stats, seed)
    }

    /// Runs a variant on data.
    pub fn run_variant(&self, variant: &str, data: &D) -> O {
        if variant == self.baseline_name {
            return (self.baseline_fn)(data);
        }
        for v in &self.variants {
            if v.name == variant {
                return (v.func)(data);
            }
        }
        // Fallback to baseline
        (self.baseline_fn)(data)
    }

    /// Returns the baseline (ground truth) result.
    pub fn ground_truth(&self, data: &D) -> O {
        (self.baseline_fn)(data)
    }

    /// Creates a bench runner for quick comparison.
    #[must_use]
    pub fn bench(&self) -> BenchRunner<'_, D, S, O> {
        BenchRunner::new(self)
    }

    /// Creates a search runner for full grid search.
    #[must_use]
    pub fn search(&self) -> SearchRunner<'_, D, S, O> {
        SearchRunner::new(self)
    }
}

/// Runner for quick benchmark comparison at specific stats.
pub struct BenchRunner<'a, D, S, O> {
    bench: &'a BuiltStatsBench<D, S, O>,
    stats_point: Option<StatsPoint>,
    seed: u64,
}

impl<'a, D, S, O> BenchRunner<'a, D, S, O>
where
    D: Clone + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static,
    O: Clone + PartialEq + Debug + Send + Sync + 'static,
{
    fn new(bench: &'a BuiltStatsBench<D, S, O>) -> Self {
        Self {
            bench,
            stats_point: None,
            seed: 42,
        }
    }

    /// Sets the stats point to benchmark at.
    #[must_use]
    pub fn at(mut self, point: StatsPoint) -> Self {
        self.stats_point = Some(point);
        self
    }

    /// Sets the random seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Runs the benchmark and returns results.
    pub fn run<F>(self, stats_to_s: F) -> BenchResults
    where
        F: Fn(&StatsPoint) -> S,
    {
        let point = self.stats_point.unwrap_or_else(|| {
            // Use first point from grid, or empty point
            self.bench.stats_grid.iter().next().unwrap_or_default()
        });

        let stats = stats_to_s(&point);
        let measurer = Measurer::default();

        let mut results = Vec::new();

        for variant_name in self.bench.variant_names() {
            let seed = self.seed;
            let result = measurer.measure(
                || self.bench.generate_data(&stats, seed),
                |data| self.bench.run_variant(variant_name, data),
            );
            results.push(VariantResult {
                name: variant_name.to_string(),
                measurement: result,
            });
        }

        BenchResults {
            benchmark_name: self.bench.name.clone(),
            stats_point: point,
            variants: results,
        }
    }
}

/// Runner for full grid search.
pub struct SearchRunner<'a, D, S, O> {
    bench: &'a BuiltStatsBench<D, S, O>,
    seed: u64,
}

impl<'a, D, S, O> SearchRunner<'a, D, S, O>
where
    D: Clone + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static,
    O: Clone + PartialEq + Debug + Send + Sync + 'static,
{
    fn new(bench: &'a BuiltStatsBench<D, S, O>) -> Self {
        Self { bench, seed: 42 }
    }

    /// Sets the random seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Runs the full grid search and returns results.
    pub fn run<F>(self, stats_to_s: F) -> SearchResults
    where
        F: Fn(&StatsPoint) -> S,
    {
        let measurer = Measurer::new()
            .warmup_time(std::time::Duration::from_millis(50))
            .measurement_time(std::time::Duration::from_millis(200));

        let mut all_results = Vec::new();

        for point in self.bench.stats_grid.iter() {
            let stats = stats_to_s(&point);

            for variant_name in self.bench.variant_names() {
                let seed = self.seed;
                let result = measurer.measure(
                    || self.bench.generate_data(&stats, seed),
                    |data| self.bench.run_variant(variant_name, data),
                );

                all_results.push(GridMeasurement {
                    stats_point: point.clone(),
                    variant: variant_name.to_string(),
                    measurement: result,
                });
            }
        }

        // Find winners at each stats point
        let winners = self.find_winners(&all_results);

        SearchResults {
            benchmark_name: self.bench.name.clone(),
            total_points: self.bench.stats_grid.total_points(),
            total_variants: self.bench.variant_names().len(),
            measurements: all_results,
            winners,
        }
    }

    #[allow(clippy::use_debug, clippy::disallowed_types)]
    fn find_winners(&self, results: &[GridMeasurement]) -> Vec<Winner> {
        use std::collections::HashMap;

        // Group by stats point
        let mut by_point: HashMap<String, Vec<&GridMeasurement>> = HashMap::new();
        for r in results {
            let key = format!("{:?}", r.stats_point.iter().collect::<Vec<_>>());
            by_point.entry(key).or_default().push(r);
        }

        let mut winners = Vec::new();
        for (_, measurements) in by_point {
            if let Some(winner) = measurements.iter().min_by(|a, b| {
                a.measurement
                    .median_ns
                    .partial_cmp(&b.measurement.median_ns)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                winners.push(Winner {
                    stats_point: winner.stats_point.clone(),
                    variant: winner.variant.clone(),
                    median_ns: winner.measurement.median_ns,
                });
            }
        }

        winners
    }
}

/// Result for a single variant at a stats point.
#[derive(Debug, Clone)]
pub struct VariantResult {
    /// Variant name.
    pub name: String,
    /// Measurement result.
    pub measurement: MeasurementResult,
}

/// Results from a quick benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResults {
    /// Benchmark name.
    pub benchmark_name: String,
    /// Stats point that was benchmarked.
    pub stats_point: StatsPoint,
    /// Results for each variant.
    pub variants: Vec<VariantResult>,
}

impl BenchResults {
    /// Prints results in a readable format.
    #[allow(clippy::use_debug)]
    pub fn print(&self) {
        println!("Benchmark: {}", self.benchmark_name);
        println!("Stats: {:?}", self.stats_point.iter().collect::<Vec<_>>());
        println!();

        // Find baseline for comparison
        let baseline_ns = self.variants.first().map(|v| v.measurement.median_ns);

        for (i, v) in self.variants.iter().enumerate() {
            let comparison = if i == 0 {
                "(baseline)".to_string()
            } else if let Some(base) = baseline_ns {
                let diff = (v.measurement.median_ns - base) / base * 100.0;
                if diff < 0.0 {
                    format!("{:.1}% faster", -diff)
                } else {
                    format!("{:.1}% slower", diff)
                }
            } else {
                String::new()
            };

            println!(
                "  {:<20} {:>10.2} ns  {:>6.2} ns  {}",
                v.name, v.measurement.median_ns, v.measurement.stddev_ns, comparison
            );
        }
    }
}

/// A measurement at a specific grid point.
#[derive(Debug, Clone)]
pub struct GridMeasurement {
    /// Stats point.
    pub stats_point: StatsPoint,
    /// Variant name.
    pub variant: String,
    /// Measurement result.
    pub measurement: MeasurementResult,
}

/// Winner at a stats point.
#[derive(Debug, Clone)]
pub struct Winner {
    /// Stats point.
    pub stats_point: StatsPoint,
    /// Winning variant name.
    pub variant: String,
    /// Median time in nanoseconds.
    pub median_ns: f64,
}

/// Results from a full grid search.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Benchmark name.
    pub benchmark_name: String,
    /// Total number of stats points searched.
    pub total_points: usize,
    /// Total number of variants.
    pub total_variants: usize,
    /// All measurements.
    pub measurements: Vec<GridMeasurement>,
    /// Winners at each stats point.
    pub winners: Vec<Winner>,
}

impl SearchResults {
    /// Prints a summary of the search results.
    #[allow(clippy::use_debug)]
    pub fn print(&self) {
        println!("Search Results: {}", self.benchmark_name);
        println!(
            "Points: {} | Variants: {} | Total measurements: {}",
            self.total_points,
            self.total_variants,
            self.measurements.len()
        );
        println!();

        println!("Winners by stats point:");
        for w in &self.winners {
            println!(
                "  {:?} -> {} ({:.2} ns)",
                w.stats_point.iter().collect::<Vec<_>>(),
                w.variant,
                w.median_ns
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scale;

    // Simple test types
    #[derive(Clone)]
    struct TestData {
        values: Vec<u64>,
    }

    #[derive(Clone)]
    struct TestStats {
        len: usize,
    }

    fn compute_stats(data: &TestData) -> TestStats {
        TestStats {
            len: data.values.len(),
        }
    }

    fn generate_data(stats: &TestStats, seed: u64) -> TestData {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let hash = hasher.finish();

        TestData {
            values: (0..stats.len)
                .map(|i| hash.wrapping_add(i as u64))
                .collect(),
        }
    }

    fn sum_naive(data: &TestData) -> u64 {
        data.values.iter().fold(0u64, |a, &b| a.wrapping_add(b))
    }

    fn sum_chunked(data: &TestData) -> u64 {
        data.values
            .chunks(4)
            .map(|c| c.iter().fold(0u64, |a, &b| a.wrapping_add(b)))
            .fold(0u64, |a, b| a.wrapping_add(b))
    }

    #[test]
    fn test_stats_bench_builder() {
        let bench = StatsBench::<TestData, TestStats, u64>::new("test_sum")
            .stats(compute_stats)
            .generate(generate_data)
            .stats_grid(
                StatsGrid::new().dimension("len", Scale::explicit(vec![100.0, 200.0, 300.0])),
            )
            .baseline("naive", sum_naive)
            .variant("chunked", sum_chunked)
            .build();

        assert_eq!(bench.name(), "test_sum");
        assert_eq!(bench.variant_names(), vec!["naive", "chunked"]);
    }

    #[test]
    fn test_stats_bench_correctness() {
        let bench = StatsBench::<TestData, TestStats, u64>::new("test_sum")
            .stats(compute_stats)
            .generate(generate_data)
            .baseline("naive", sum_naive)
            .variant("chunked", sum_chunked)
            .build();

        let stats = TestStats { len: 1000 };
        let data = bench.generate_data(&stats, 42);

        let naive_result = bench.run_variant("naive", &data);
        let chunked_result = bench.run_variant("chunked", &data);

        assert_eq!(naive_result, chunked_result);
    }

    #[test]
    fn test_bench_runner() {
        let bench = StatsBench::<TestData, TestStats, u64>::new("test_sum")
            .stats(compute_stats)
            .generate(generate_data)
            .stats_grid(StatsGrid::new().dimension("len", Scale::explicit(vec![100.0])))
            .baseline("naive", sum_naive)
            .variant("chunked", sum_chunked)
            .build();

        let results = bench
            .bench()
            .at(StatsPoint::new().with("len", 100.0))
            .run(|point| TestStats {
                len: point.get_usize("len").unwrap_or(100),
            });

        assert_eq!(results.variants.len(), 2);
        assert!(results.variants[0].measurement.median_ns > 0.0);
    }

    #[test]
    fn test_search_runner() {
        let bench = StatsBench::<TestData, TestStats, u64>::new("test_sum")
            .stats(compute_stats)
            .generate(generate_data)
            .stats_grid(StatsGrid::new().dimension("len", Scale::explicit(vec![100.0, 200.0])))
            .baseline("naive", sum_naive)
            .variant("chunked", sum_chunked)
            .build();

        let results = bench.search().run(|point| TestStats {
            len: point.get_usize("len").unwrap_or(100),
        });

        assert_eq!(results.total_points, 2);
        assert_eq!(results.total_variants, 2);
        assert_eq!(results.measurements.len(), 4); // 2 points × 2 variants
        assert_eq!(results.winners.len(), 2); // 1 winner per point
    }
}
