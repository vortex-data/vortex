//! CLI tool for running ISA threshold benchmarks.
//!
//! This binary performs a grid search across algorithm variants and parameter
//! values, measuring performance to find crossover points where one implementation
//! becomes faster than another.

// Using std HashMap for serde compatibility in this CLI tool
#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;
use serde::Serialize;
use sysinfo::System;
use vortex_threshold_traits::AlgorithmRegistry;
use vortex_threshold_traits::BenchmarkResult;
use vortex_threshold_traits::CpuClass;
use vortex_threshold_traits::ParameterScale;
use vortex_threshold_traits::Variant;
use vortex_threshold_traits::VerifyResult;

pub mod examples;

/// CLI arguments for the threshold runner.
#[derive(Parser, Debug)]
#[command(name = "threshold-runner")]
#[command(about = "Run ISA threshold benchmarks to find algorithm crossover points")]
struct Args {
    /// Output file for benchmark results (JSON format).
    #[arg(short, long, default_value = "results.json")]
    output: PathBuf,

    /// Number of iterations per benchmark.
    #[arg(short, long, default_value = "100")]
    iterations: usize,

    /// Number of warmup iterations before benchmarking.
    #[arg(short, long, default_value = "10")]
    warmup: usize,

    /// Random seed for input generation.
    #[arg(short, long, default_value = "42")]
    seed: u64,

    /// Verify correctness of all variants before benchmarking.
    #[arg(long, default_value = "true")]
    verify: bool,

    /// Only run benchmarks for the specified algorithm.
    #[arg(long)]
    algorithm: Option<String>,

    /// Only run benchmarks for the specified variant.
    #[arg(long)]
    variant: Option<String>,

    /// Run built-in example benchmarks (popcount, etc.).
    #[arg(long, default_value = "false")]
    examples: bool,
}

/// Results from a complete benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResults {
    /// Metadata about the machine and run.
    pub metadata: RunMetadata,
    /// Results for each algorithm.
    pub algorithms: HashMap<String, AlgorithmResults>,
}

/// Metadata about the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Detected CPU class.
    pub cpu_class: CpuClass,
    /// CPU model name.
    pub cpu_model: String,
    /// Number of CPU cores.
    pub cpu_cores: usize,
    /// Total system memory in bytes.
    pub total_memory: u64,
    /// Target architecture (e.g., "x86_64", "aarch64").
    pub target_arch: String,
    /// Target OS (e.g., "linux", "macos").
    pub target_os: String,
    /// Timestamp of the run.
    pub timestamp: String,
}

impl RunMetadata {
    fn collect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_model = sys
            .cpus()
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        Self {
            cpu_class: CpuClass::detect(),
            cpu_model,
            cpu_cores: sys.cpus().len(),
            total_memory: sys.total_memory(),
            target_arch: std::env::consts::ARCH.to_string(),
            target_os: std::env::consts::OS.to_string(),
            timestamp: chrono_lite_timestamp(),
        }
    }
}

/// Simple timestamp without heavy chrono dependency.
fn chrono_lite_timestamp() -> String {
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Results for a single algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResults {
    /// Name of the algorithm.
    pub name: String,
    /// Name of the parameter being varied.
    pub parameter_name: String,
    /// Results for each variant.
    pub variants: HashMap<String, VariantResults>,
    /// Detected crossover points.
    pub crossovers: Vec<Crossover>,
}

/// Results for a single variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantResults {
    /// Name of the variant.
    pub name: String,
    /// Whether the variant is available on this CPU.
    pub available: bool,
    /// Required CPU features.
    pub required_features: Vec<String>,
    /// Benchmark results at each parameter value.
    pub measurements: HashMap<usize, BenchmarkResult>,
}

/// A detected crossover point between two variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crossover {
    /// The slower variant (before crossover).
    pub from_variant: String,
    /// The faster variant (after crossover).
    pub to_variant: String,
    /// Parameter value where crossover occurs.
    pub threshold: usize,
    /// Confidence interval lower bound.
    pub ci_low: usize,
    /// Confidence interval upper bound.
    pub ci_high: usize,
}

/// Grid search runner for finding ISA thresholds.
pub struct GridSearch {
    iterations: usize,
    seed: u64,
    verify: bool,
    algorithm_filter: Option<String>,
    variant_filter: Option<String>,
}

impl GridSearch {
    /// Creates a new grid search with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            iterations: 100,
            seed: 42,
            verify: true,
            algorithm_filter: None,
            variant_filter: None,
        }
    }

    /// Sets the number of benchmark iterations.
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the random seed for input generation.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Enables or disables correctness verification.
    #[must_use]
    pub fn verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }

    /// Filter to only run a specific algorithm.
    #[must_use]
    pub fn algorithm_filter(mut self, algorithm: Option<String>) -> Self {
        self.algorithm_filter = algorithm;
        self
    }

    /// Filter to only run a specific variant.
    #[must_use]
    pub fn variant_filter(mut self, variant: Option<String>) -> Self {
        self.variant_filter = variant;
        self
    }

    /// Runs the grid search on all registered algorithms.
    pub fn run(&self, registry: &AlgorithmRegistry) -> BenchmarkRunResults {
        let metadata = RunMetadata::collect();
        let mut algorithms = HashMap::new();

        for algo in registry.iter() {
            // Apply algorithm filter
            if let Some(ref filter) = self.algorithm_filter
                && algo.name() != filter
            {
                continue;
            }

            eprintln!("Benchmarking algorithm: {}", algo.name());
            let results = self.run_algorithm(algo);
            algorithms.insert(algo.name().to_string(), results);
        }

        BenchmarkRunResults {
            metadata,
            algorithms,
        }
    }

    fn run_algorithm(
        &self,
        algo: &dyn vortex_threshold_traits::DynBenchmarkableAlgorithm,
    ) -> AlgorithmResults {
        let variants = algo.variants();
        let scale = algo.parameter_scale();

        let mut variant_results = HashMap::new();

        for variant in &variants {
            // Apply variant filter
            if let Some(ref filter) = self.variant_filter
                && variant.name != *filter
            {
                continue;
            }

            eprintln!("  Variant: {}", variant.name);
            let results = self.run_variant(algo, variant, &scale);
            variant_results.insert(variant.name.clone(), results);
        }

        // Find crossover points
        let crossovers = self.find_crossovers(&variant_results, &scale);

        AlgorithmResults {
            name: algo.name().to_string(),
            parameter_name: algo.parameter_name().to_string(),
            variants: variant_results,
            crossovers,
        }
    }

    fn run_variant(
        &self,
        algo: &dyn vortex_threshold_traits::DynBenchmarkableAlgorithm,
        variant: &Variant,
        scale: &ParameterScale,
    ) -> VariantResults {
        let available = variant.is_available();
        let mut measurements = HashMap::new();

        if available {
            // Verify correctness first
            if self.verify {
                let verify_result = algo.verify_variant(&variant.name, 1024, self.seed);
                match verify_result {
                    VerifyResult::Correct => {
                        eprintln!("    Correctness verified");
                    }
                    VerifyResult::NotAvailable => {
                        eprintln!("    Variant not available");
                    }
                    VerifyResult::Incorrect { expected, actual } => {
                        eprintln!(
                            "    Incorrect output! Expected: {}, Got: {}",
                            expected, actual
                        );
                    }
                }
            }

            // Run benchmarks for each parameter value
            for param in scale.iter() {
                if let Some(result) =
                    algo.benchmark_variant(&variant.name, param, self.seed, self.iterations)
                {
                    eprintln!(
                        "    param={}: {:.2}ns +/- {:.2}ns",
                        param, result.mean_ns, result.stddev_ns
                    );
                    measurements.insert(param, result);
                }
            }
        } else {
            eprintln!("    Not available on this CPU");
        }

        VariantResults {
            name: variant.name.clone(),
            available,
            required_features: variant.required_features.clone(),
            measurements,
        }
    }

    fn find_crossovers(
        &self,
        variant_results: &HashMap<String, VariantResults>,
        scale: &ParameterScale,
    ) -> Vec<Crossover> {
        let mut crossovers = Vec::new();

        // Get available variants sorted by name for deterministic ordering
        let mut available_variants: Vec<_> = variant_results
            .values()
            .filter(|v| v.available && !v.measurements.is_empty())
            .collect();
        available_variants.sort_by(|a, b| a.name.cmp(&b.name));

        // Compare each pair of variants
        for i in 0..available_variants.len() {
            for j in (i + 1)..available_variants.len() {
                let v1 = &available_variants[i];
                let v2 = &available_variants[j];

                if let Some(crossover) = self.find_crossover_pair(v1, v2, scale) {
                    crossovers.push(crossover);
                }
            }
        }

        crossovers
    }

    fn find_crossover_pair(
        &self,
        v1: &VariantResults,
        v2: &VariantResults,
        scale: &ParameterScale,
    ) -> Option<Crossover> {
        let params: Vec<_> = scale.iter().collect();

        // Find where v2 becomes faster than v1
        let mut crossover_idx = None;
        let mut prev_v1_faster = None;

        for (idx, &param) in params.iter().enumerate() {
            let Some(r1) = v1.measurements.get(&param) else {
                continue;
            };
            let Some(r2) = v2.measurements.get(&param) else {
                continue;
            };

            let v1_faster = r1.mean_ns < r2.mean_ns;

            if let Some(prev) = prev_v1_faster {
                // v2 just became faster
                if prev && !v1_faster {
                    crossover_idx = Some(idx);
                    break;
                }
            }
            prev_v1_faster = Some(v1_faster);
        }

        let crossover_idx = crossover_idx?;
        let threshold = params[crossover_idx];

        // Calculate confidence interval
        let ci_low = if crossover_idx > 0 {
            params[crossover_idx - 1]
        } else {
            threshold
        };
        let ci_high = if crossover_idx + 1 < params.len() {
            params[crossover_idx + 1]
        } else {
            threshold
        };

        Some(Crossover {
            from_variant: v1.name.clone(),
            to_variant: v2.name.clone(),
            threshold,
            ci_low,
            ci_high,
        })
    }
}

impl Default for GridSearch {
    fn default() -> Self {
        Self::new()
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Create registry and optionally register built-in examples
    let mut registry = AlgorithmRegistry::new();

    if args.examples {
        eprintln!("Registering built-in example benchmarks...");
        examples::register_all(&mut registry);
    }

    if registry.is_empty() {
        eprintln!("No algorithms registered. To use this tool:");
        eprintln!("1. Use --examples to run built-in example benchmarks");
        eprintln!("2. Or build your library with the 'benchmark' feature");
        eprintln!("3. And register algorithms using your_lib::benchmarks::register_all()");
        eprintln!();
        eprintln!("Example usage in a custom binary:");
        eprintln!("  let mut registry = AlgorithmRegistry::new();");
        eprintln!("  my_lib::benchmarks::register_all(&mut registry);");
        eprintln!("  let results = GridSearch::new().run(&registry);");
        eprintln!();
        eprintln!("For now, generating an empty results file for demonstration.");
    }

    let results = GridSearch::new()
        .iterations(args.iterations)
        .seed(args.seed)
        .verify(args.verify)
        .algorithm_filter(args.algorithm)
        .variant_filter(args.variant)
        .run(&registry);

    // Write results to file
    let json = serde_json::to_string_pretty(&results)?;

    let mut file = File::create(&args.output)?;
    file.write_all(json.as_bytes())?;

    eprintln!("Results written to: {}", args.output.display());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        #[allow(clippy::exit)]
        std::process::exit(1);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_search_creation() {
        let search = GridSearch::new().iterations(50).seed(123).verify(false);

        assert_eq!(search.iterations, 50);
        assert_eq!(search.seed, 123);
        assert!(!search.verify);
    }

    #[test]
    fn test_empty_registry() {
        let registry = AlgorithmRegistry::new();
        let search = GridSearch::new();
        let results = search.run(&registry);

        assert!(results.algorithms.is_empty());
    }
}
