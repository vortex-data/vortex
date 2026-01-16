//! CLI tool for aggregating benchmark results and generating Rust threshold tables.
//!
//! This binary reads JSON results from multiple benchmark runs (one per CPU architecture),
//! aggregates them, and generates Rust code with lookup tables for runtime dispatch.

// Using std HashMap for serde compatibility in this CLI tool
#![allow(clippy::disallowed_types)]

use std::collections::HashMap;
use std::fs::File;
use std::fs::{self};
use std::io::BufWriter;
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use serde::Deserialize;
use serde::Serialize;
use vortex_threshold_traits::CpuClass;

/// CLI arguments for the threshold aggregator.
#[derive(Parser, Debug)]
#[command(name = "threshold-aggregator")]
#[command(about = "Aggregate benchmark results and generate Rust threshold tables")]
struct Args {
    /// Input directory containing JSON result files.
    #[arg(short, long)]
    input_dir: PathBuf,

    /// Output file for aggregated thresholds (JSON format).
    #[arg(short, long, default_value = "thresholds.json")]
    output: PathBuf,

    /// Path for generated Rust code.
    #[arg(short, long)]
    generate_rust: Option<PathBuf>,
}

/// Results from a complete benchmark run (mirrors threshold-runner output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRunResults {
    pub metadata: RunMetadata,
    pub algorithms: HashMap<String, AlgorithmResults>,
}

/// Metadata about the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub cpu_class: CpuClass,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub total_memory: u64,
    pub target_arch: String,
    pub target_os: String,
    pub timestamp: String,
}

/// Results for a single algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResults {
    pub name: String,
    pub parameter_name: String,
    pub variants: HashMap<String, VariantResults>,
    pub crossovers: Vec<Crossover>,
}

/// Results for a single variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantResults {
    pub name: String,
    pub available: bool,
    pub required_features: Vec<String>,
    pub measurements: HashMap<usize, BenchmarkResult>,
}

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub iterations: usize,
}

/// A detected crossover point between two variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crossover {
    pub from_variant: String,
    pub to_variant: String,
    pub threshold: usize,
    pub ci_low: usize,
    pub ci_high: usize,
}

/// Aggregated thresholds across all CPU architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedThresholds {
    /// Thresholds for each algorithm.
    pub algorithms: HashMap<String, AlgorithmThresholds>,
    /// Default thresholds for unknown CPUs.
    pub defaults: HashMap<String, HashMap<String, usize>>,
}

/// Thresholds for a single algorithm across CPU classes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmThresholds {
    /// Algorithm name.
    pub name: String,
    /// Thresholds per CPU class.
    pub cpu_thresholds: HashMap<CpuClass, HashMap<String, usize>>,
}

impl AggregatedThresholds {
    /// Creates a new empty aggregated thresholds structure.
    fn new() -> Self {
        Self {
            algorithms: HashMap::new(),
            defaults: HashMap::new(),
        }
    }

    /// Adds results from a single benchmark run.
    fn add_run(&mut self, results: &BenchmarkRunResults) {
        let cpu_class = results.metadata.cpu_class;

        for (algo_name, algo_results) in &results.algorithms {
            let algo_thresholds =
                self.algorithms
                    .entry(algo_name.clone())
                    .or_insert_with(|| AlgorithmThresholds {
                        name: algo_name.clone(),
                        cpu_thresholds: HashMap::new(),
                    });

            let mut thresholds = HashMap::new();
            for crossover in &algo_results.crossovers {
                let key = format!("{}_to_{}", crossover.from_variant, crossover.to_variant);
                thresholds.insert(key, crossover.threshold);
            }

            algo_thresholds
                .cpu_thresholds
                .insert(cpu_class, thresholds.clone());

            // Update defaults (use Unknown or last seen)
            if cpu_class == CpuClass::Unknown || !self.defaults.contains_key(algo_name) {
                self.defaults.insert(algo_name.clone(), thresholds);
            }
        }
    }

    /// Computes sensible defaults from all observations.
    fn compute_defaults(&mut self) {
        for (algo_name, algo_thresholds) in &self.algorithms {
            if !self.defaults.contains_key(algo_name) {
                // Use median of all thresholds as default
                let mut all_thresholds: HashMap<String, Vec<usize>> = HashMap::new();

                for thresholds in algo_thresholds.cpu_thresholds.values() {
                    for (key, value) in thresholds {
                        all_thresholds.entry(key.clone()).or_default().push(*value);
                    }
                }

                let defaults: HashMap<String, usize> = all_thresholds
                    .into_iter()
                    .map(|(key, mut values)| {
                        values.sort_unstable();
                        let median = values[values.len() / 2];
                        (key, median)
                    })
                    .collect();

                self.defaults.insert(algo_name.clone(), defaults);
            }
        }
    }
}

/// Rust code generator for threshold tables.
struct RustCodeGenerator {
    aggregated: AggregatedThresholds,
}

impl RustCodeGenerator {
    fn new(aggregated: AggregatedThresholds) -> Self {
        Self { aggregated }
    }

    fn generate(&self, output: &PathBuf) -> std::io::Result<()> {
        let file = File::create(output)?;
        let mut writer = BufWriter::new(file);

        self.write_header(&mut writer)?;
        self.write_cpu_class_enum(&mut writer)?;

        for (algo_name, algo_thresholds) in &self.aggregated.algorithms {
            self.write_algorithm_thresholds(&mut writer, algo_name, algo_thresholds)?;
        }

        Ok(())
    }

    fn write_header(&self, writer: &mut BufWriter<File>) -> std::io::Result<()> {
        writeln!(writer, "//! Auto-generated ISA threshold tables.")?;
        writeln!(writer, "//!")?;
        writeln!(
            writer,
            "//! This file is generated by `threshold-aggregator` from benchmark results."
        )?;
        writeln!(writer, "//! Do not edit manually.")?;
        writeln!(writer)?;
        writeln!(writer, "#![allow(dead_code)]")?;
        writeln!(writer)?;
        writeln!(writer, "use std::sync::LazyLock;")?;
        writeln!(writer)?;
        writeln!(writer, "use vortex_threshold_traits::CpuClass;")?;
        writeln!(writer)?;
        Ok(())
    }

    fn write_cpu_class_enum(&self, writer: &mut BufWriter<File>) -> std::io::Result<()> {
        writeln!(writer, "/// Detects the CPU class at runtime.")?;
        writeln!(writer, "#[inline]")?;
        writeln!(writer, "pub fn detect_cpu_class() -> CpuClass {{")?;
        writeln!(writer, "    CpuClass::detect()")?;
        writeln!(writer, "}}")?;
        writeln!(writer)?;
        Ok(())
    }

    fn write_algorithm_thresholds(
        &self,
        writer: &mut BufWriter<File>,
        algo_name: &str,
        algo_thresholds: &AlgorithmThresholds,
    ) -> std::io::Result<()> {
        let struct_name = format!("{}Thresholds", to_pascal_case(algo_name));
        let static_name = format!("{}_THRESHOLDS", algo_name.to_uppercase());

        // Collect all threshold keys
        let mut all_keys: Vec<String> = algo_thresholds
            .cpu_thresholds
            .values()
            .flat_map(|t| t.keys().cloned())
            .collect();
        all_keys.sort();
        all_keys.dedup();

        // Write struct definition
        writeln!(
            writer,
            "/// Threshold values for the {} algorithm.",
            algo_name
        )?;
        writeln!(writer, "#[derive(Debug, Clone, Copy)]")?;
        writeln!(writer, "pub struct {} {{", struct_name)?;
        for key in &all_keys {
            let field_name = to_snake_case(key);
            writeln!(
                writer,
                "    /// Threshold for {} transition.",
                key.replace('_', " ")
            )?;
            writeln!(writer, "    pub {}: usize,", field_name)?;
        }
        writeln!(writer, "}}")?;
        writeln!(writer)?;

        // Write static lazy initializer
        writeln!(
            writer,
            "/// Lazily initialized thresholds for {} based on CPU detection.",
            algo_name
        )?;
        writeln!(
            writer,
            "pub static {}: LazyLock<{}> = LazyLock::new(|| {{",
            static_name, struct_name
        )?;
        writeln!(writer, "    match detect_cpu_class() {{")?;

        // Write each CPU class
        for (cpu_class, thresholds) in &algo_thresholds.cpu_thresholds {
            let cpu_variant = cpu_class_to_string(*cpu_class);
            writeln!(
                writer,
                "        CpuClass::{} => {} {{",
                cpu_variant, struct_name
            )?;
            for key in &all_keys {
                let field_name = to_snake_case(key);
                let value = thresholds
                    .get(key)
                    .or_else(|| {
                        self.aggregated
                            .defaults
                            .get(algo_name)
                            .and_then(|d| d.get(key))
                    })
                    .copied()
                    .unwrap_or(512);
                writeln!(writer, "            {}: {},", field_name, value)?;
            }
            writeln!(writer, "        }},")?;
        }

        // Write default case
        writeln!(writer, "        _ => {} {{", struct_name)?;
        let defaults = self.aggregated.defaults.get(algo_name);
        for key in &all_keys {
            let field_name = to_snake_case(key);
            let value = defaults.and_then(|d| d.get(key)).copied().unwrap_or(512);
            writeln!(writer, "            {}: {},", field_name, value)?;
        }
        writeln!(writer, "        }},")?;

        writeln!(writer, "    }}")?;
        writeln!(writer, "}});")?;
        writeln!(writer)?;

        Ok(())
    }
}

/// Converts a CpuClass to its variant name string.
fn cpu_class_to_string(cpu_class: CpuClass) -> &'static str {
    match cpu_class {
        CpuClass::IntelSapphire => "IntelSapphire",
        CpuClass::IntelIceLake => "IntelIceLake",
        CpuClass::IntelSkylake => "IntelSkylake",
        CpuClass::AmdGenoa => "AmdGenoa",
        CpuClass::AmdMilan => "AmdMilan",
        CpuClass::AmdRome => "AmdRome",
        CpuClass::Graviton3 => "Graviton3",
        CpuClass::Graviton2 => "Graviton2",
        CpuClass::AppleSilicon => "AppleSilicon",
        CpuClass::Unknown => "Unknown",
    }
}

/// Converts a string to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split(|c: char| c == '_' || c == '-' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.map(|c| c.to_ascii_lowercase()))
                    .collect(),
                None => String::new(),
            }
        })
        .collect()
}

/// Converts a string to snake_case.
fn to_snake_case(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c == '-' {
                '_'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Read all JSON files from input directory
    let entries = fs::read_dir(&args.input_dir)?;

    let mut aggregated = AggregatedThresholds::new();
    let mut files_processed = 0;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            eprintln!("Processing: {}", path.display());

            let contents = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  Failed to read file: {}", e);
                    continue;
                }
            };

            let results: BenchmarkRunResults = match serde_json::from_str(&contents) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  Failed to parse JSON: {}", e);
                    continue;
                }
            };

            eprintln!(
                "  CPU: {} ({})",
                cpu_class_to_string(results.metadata.cpu_class),
                results.metadata.cpu_model
            );
            eprintln!("  Algorithms: {}", results.algorithms.len());

            aggregated.add_run(&results);
            files_processed += 1;
        }
    }

    if files_processed == 0 {
        return Err("No JSON files found in input directory".into());
    }

    aggregated.compute_defaults();

    // Write aggregated thresholds
    let json = serde_json::to_string_pretty(&aggregated)?;

    fs::write(&args.output, json)?;

    eprintln!(
        "Aggregated thresholds written to: {}",
        args.output.display()
    );

    // Generate Rust code if requested
    if let Some(rust_path) = &args.generate_rust {
        let generator = RustCodeGenerator::new(aggregated);
        generator.generate(rust_path)?;
        eprintln!("Generated Rust code written to: {}", rust_path.display());
    }

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
mod tests {
    use super::*;

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("foo-bar"), "FooBar");
        assert_eq!(to_pascal_case("simple"), "Simple");
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("HelloWorld"), "helloworld");
        assert_eq!(to_snake_case("foo-bar"), "foo_bar");
    }

    #[test]
    fn test_aggregated_thresholds_new() {
        let aggregated = AggregatedThresholds::new();
        assert!(aggregated.algorithms.is_empty());
        assert!(aggregated.defaults.is_empty());
    }
}
