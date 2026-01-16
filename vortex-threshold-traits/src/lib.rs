//! Trait definitions for ISA threshold detection.
//!
//! This crate provides the core [`BenchmarkableAlgorithm`] trait that algorithms
//! must implement to participate in automatic threshold detection. The threshold
//! finder system uses these implementations to determine crossover points where
//! one algorithm variant becomes faster than another across different CPU architectures.

use std::fmt::Debug;

use serde::Deserialize;
use serde::Serialize;

/// Defines the range of parameter values to sweep during benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterScale {
    /// Linear progression from start to end with a fixed step.
    Linear {
        /// Starting value (inclusive).
        start: usize,
        /// Ending value (inclusive).
        end: usize,
        /// Step size between values.
        step: usize,
    },
    /// Logarithmic progression using powers of a base.
    ///
    /// Generates values: base^start_exp, base^(start_exp+1), ..., base^end_exp
    Logarithmic {
        /// Starting exponent (inclusive).
        start_exp: u32,
        /// Ending exponent (inclusive).
        end_exp: u32,
        /// Base for exponentiation (typically 2 or 10).
        base: usize,
    },
    /// Explicit list of parameter values to test.
    Explicit(Vec<usize>),
}

impl ParameterScale {
    /// Creates a logarithmic scale with base 2.
    ///
    /// # Arguments
    ///
    /// * `start_exp` - Starting exponent (e.g., 6 for 64)
    /// * `end_exp` - Ending exponent (e.g., 20 for 1M)
    ///
    /// # Example
    ///
    /// ```
    /// use vortex_threshold_traits::ParameterScale;
    ///
    /// // Creates scale: 64, 128, 256, ..., 1048576
    /// let scale = ParameterScale::log2(6, 20);
    /// ```
    #[must_use]
    pub fn log2(start_exp: u32, end_exp: u32) -> Self {
        Self::Logarithmic {
            start_exp,
            end_exp,
            base: 2,
        }
    }

    /// Returns an iterator over all parameter values in this scale.
    pub fn iter(&self) -> Box<dyn Iterator<Item = usize>> {
        match self {
            Self::Linear { start, end, step } => {
                let start = *start;
                let end = *end;
                let step = *step;
                Box::new((start..=end).step_by(step))
            }
            Self::Logarithmic {
                start_exp,
                end_exp,
                base,
            } => {
                let base = *base;
                let start_exp = *start_exp;
                let end_exp = *end_exp;
                Box::new((start_exp..=end_exp).map(move |exp| base.pow(exp)))
            }
            Self::Explicit(values) => Box::new(values.clone().into_iter()),
        }
    }

    /// Returns the number of parameter values in this scale.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Linear { start, end, step } => ((end - start) / step) + 1,
            Self::Logarithmic {
                start_exp, end_exp, ..
            } => (end_exp - start_exp + 1) as usize,
            Self::Explicit(values) => values.len(),
        }
    }

    /// Returns true if this scale contains no parameter values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Describes a variant of an algorithm implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Name of the variant (e.g., "naive", "popcount", "avx2").
    pub name: String,
    /// CPU features required for this variant (e.g., ["avx2"] or ["neon"]).
    pub required_features: Vec<String>,
}

impl Variant {
    /// Creates a new variant with no required CPU features.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            required_features: Vec::new(),
        }
    }

    /// Adds required CPU features for this variant.
    #[must_use]
    pub fn with_features(mut self, features: &[&str]) -> Self {
        self.required_features = features.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Returns true if this variant is available on the current CPU.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.required_features.iter().all(|feature| {
            #[cfg(target_arch = "x86_64")]
            {
                match feature.as_str() {
                    "avx2" => std::arch::is_x86_feature_detected!("avx2"),
                    "avx" => std::arch::is_x86_feature_detected!("avx"),
                    "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
                    "sse4.2" => std::arch::is_x86_feature_detected!("sse4.2"),
                    "popcnt" => std::arch::is_x86_feature_detected!("popcnt"),
                    _ => false,
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                match feature.as_str() {
                    "neon" => true, // NEON is always available on AArch64
                    _ => false,
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let _ = feature;
                false
            }
        })
    }
}

/// Trait for algorithms that can be benchmarked for ISA threshold detection.
///
/// Implementations of this trait define how to generate inputs, run different
/// algorithm variants, and verify correctness. The threshold runner uses this
/// trait to perform grid searches and find crossover points.
///
/// # Type Parameters
///
/// - `Input`: The type of input data the algorithm operates on.
/// - `Output`: The type of output produced by the algorithm.
///
/// # Example
///
/// ```ignore
/// use vortex_threshold_traits::{BenchmarkableAlgorithm, ParameterScale, Variant};
///
/// struct RankBenchmark;
///
/// impl BenchmarkableAlgorithm for RankBenchmark {
///     type Input = (Vec<u64>, usize);
///     type Output = usize;
///
///     fn name(&self) -> &'static str { "rank" }
///     fn parameter_name(&self) -> &'static str { "input_size" }
///     fn parameter_scale(&self) -> ParameterScale { ParameterScale::log2(6, 20) }
///
///     fn variants(&self) -> Vec<Variant> {
///         vec![
///             Variant::new("naive"),
///             Variant::new("avx2").with_features(&["avx2"]),
///         ]
///     }
///
///     fn generate_input(&self, param: usize, seed: u64) -> Self::Input {
///         // Generate random input of size `param`
///         todo!()
///     }
///
///     fn ground_truth(&self, input: &Self::Input) -> Self::Output {
///         // Trusted reference implementation
///         todo!()
///     }
///
///     fn run_variant(&self, variant: &str, input: &Self::Input) -> Self::Output {
///         // Run the specified variant
///         todo!()
///     }
/// }
/// ```
pub trait BenchmarkableAlgorithm: Send + Sync {
    /// The type of input data for the algorithm.
    type Input: Send + Clone;
    /// The type of output produced by the algorithm.
    type Output: PartialEq + Debug + Send;

    /// Returns the name of the algorithm (e.g., "rank", "select").
    fn name(&self) -> &'static str;

    /// Returns the name of the parameter being varied (e.g., "input_size").
    fn parameter_name(&self) -> &'static str;

    /// Returns the scale of parameter values to benchmark.
    fn parameter_scale(&self) -> ParameterScale;

    /// Returns all available variants of this algorithm.
    fn variants(&self) -> Vec<Variant>;

    /// Generates input data for the given parameter value and random seed.
    fn generate_input(&self, param: usize, seed: u64) -> Self::Input;

    /// Computes the ground truth output for correctness verification.
    ///
    /// This should be a trusted reference implementation, typically the
    /// simplest (naive) variant.
    fn ground_truth(&self, input: &Self::Input) -> Self::Output;

    /// Runs the specified variant on the given input.
    fn run_variant(&self, variant: &str, input: &Self::Input) -> Self::Output;
}

/// Type-erased wrapper for benchmarkable algorithms.
///
/// This allows storing algorithms with different input/output types
/// in the same registry.
pub trait DynBenchmarkableAlgorithm: Send + Sync {
    /// Returns the name of the algorithm.
    fn name(&self) -> &'static str;

    /// Returns the name of the parameter being varied.
    fn parameter_name(&self) -> &'static str;

    /// Returns the scale of parameter values to benchmark.
    fn parameter_scale(&self) -> ParameterScale;

    /// Returns all available variants.
    fn variants(&self) -> Vec<Variant>;

    /// Generates input, runs a variant, and returns timing in nanoseconds.
    ///
    /// Returns `None` if the variant is not available on this CPU.
    fn benchmark_variant(
        &self,
        variant: &str,
        param: usize,
        seed: u64,
        iterations: usize,
    ) -> Option<BenchmarkResult>;

    /// Verifies that a variant produces correct output.
    fn verify_variant(&self, variant: &str, param: usize, seed: u64) -> VerifyResult;
}

/// Result of a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Mean time per iteration in nanoseconds.
    pub mean_ns: f64,
    /// Standard deviation in nanoseconds.
    pub stddev_ns: f64,
    /// Minimum time in nanoseconds.
    pub min_ns: u64,
    /// Maximum time in nanoseconds.
    pub max_ns: u64,
    /// Number of iterations run.
    pub iterations: usize,
}

/// Result of correctness verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifyResult {
    /// The variant is correct.
    Correct,
    /// The variant is not available on this CPU.
    NotAvailable,
    /// The variant produced incorrect output.
    Incorrect {
        /// Expected output (from ground truth).
        expected: String,
        /// Actual output from the variant.
        actual: String,
    },
}

impl<T> DynBenchmarkableAlgorithm for T
where
    T: BenchmarkableAlgorithm,
    T::Output: 'static,
{
    fn name(&self) -> &'static str {
        BenchmarkableAlgorithm::name(self)
    }

    fn parameter_name(&self) -> &'static str {
        BenchmarkableAlgorithm::parameter_name(self)
    }

    fn parameter_scale(&self) -> ParameterScale {
        BenchmarkableAlgorithm::parameter_scale(self)
    }

    fn variants(&self) -> Vec<Variant> {
        BenchmarkableAlgorithm::variants(self)
    }

    fn benchmark_variant(
        &self,
        variant: &str,
        param: usize,
        seed: u64,
        iterations: usize,
    ) -> Option<BenchmarkResult> {
        // Check if variant is available
        let variants = BenchmarkableAlgorithm::variants(self);
        let variant_info = variants.iter().find(|v| v.name == variant)?;
        if !variant_info.is_available() {
            return None;
        }

        let input = BenchmarkableAlgorithm::generate_input(self, param, seed);

        // Warmup
        for _ in 0..10 {
            std::hint::black_box(BenchmarkableAlgorithm::run_variant(self, variant, &input));
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            std::hint::black_box(BenchmarkableAlgorithm::run_variant(self, variant, &input));
            // Truncation is acceptable here: u64 nanoseconds spans ~584 years,
            // far exceeding any reasonable benchmark duration.
            #[allow(clippy::cast_possible_truncation)]
            let elapsed = start.elapsed().as_nanos() as u64;
            times.push(elapsed);
        }

        let mean_ns = times.iter().sum::<u64>() as f64 / iterations as f64;
        let variance = times
            .iter()
            .map(|&t| (t as f64 - mean_ns).powi(2))
            .sum::<f64>()
            / iterations as f64;
        let stddev_ns = variance.sqrt();
        let min_ns = times.iter().copied().min().unwrap_or(0);
        let max_ns = times.iter().copied().max().unwrap_or(0);

        Some(BenchmarkResult {
            mean_ns,
            stddev_ns,
            min_ns,
            max_ns,
            iterations,
        })
    }

    fn verify_variant(&self, variant: &str, param: usize, seed: u64) -> VerifyResult {
        let variants = BenchmarkableAlgorithm::variants(self);
        let Some(variant_info) = variants.iter().find(|v| v.name == variant) else {
            return VerifyResult::NotAvailable;
        };
        if !variant_info.is_available() {
            return VerifyResult::NotAvailable;
        }

        let input = BenchmarkableAlgorithm::generate_input(self, param, seed);
        let expected = BenchmarkableAlgorithm::ground_truth(self, &input);
        let actual = BenchmarkableAlgorithm::run_variant(self, variant, &input);

        if expected == actual {
            VerifyResult::Correct
        } else {
            VerifyResult::Incorrect {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", actual),
            }
        }
    }
}

/// Registry for benchmarkable algorithms.
#[derive(Default)]
pub struct AlgorithmRegistry {
    algorithms: Vec<Box<dyn DynBenchmarkableAlgorithm>>,
}

impl AlgorithmRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a benchmarkable algorithm.
    pub fn register<T>(&mut self, algorithm: T)
    where
        T: BenchmarkableAlgorithm + 'static,
        T::Output: 'static,
    {
        self.algorithms.push(Box::new(algorithm));
    }

    /// Returns an iterator over all registered algorithms.
    pub fn iter(&self) -> impl Iterator<Item = &dyn DynBenchmarkableAlgorithm> {
        self.algorithms.iter().map(|a| a.as_ref())
    }

    /// Returns the number of registered algorithms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.algorithms.len()
    }

    /// Returns true if no algorithms are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.algorithms.is_empty()
    }
}

/// CPU architecture classification for threshold lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuClass {
    /// Intel Sapphire Rapids or similar.
    IntelSapphire,
    /// Intel Ice Lake.
    IntelIceLake,
    /// Intel Skylake.
    IntelSkylake,
    /// AMD Genoa (Zen 4).
    AmdGenoa,
    /// AMD Milan (Zen 3).
    AmdMilan,
    /// AMD Rome (Zen 2).
    AmdRome,
    /// AWS Graviton 3.
    Graviton3,
    /// AWS Graviton 2.
    Graviton2,
    /// Apple Silicon (M1/M2/M3).
    AppleSilicon,
    /// Unknown CPU, use default thresholds.
    Unknown,
}

impl CpuClass {
    /// Detects the CPU class of the current system.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // Use CPUID to detect CPU vendor and model
            if let Some(vendor) = Self::get_x86_vendor() {
                if vendor.contains("GenuineIntel") {
                    return Self::detect_intel();
                } else if vendor.contains("AuthenticAMD") {
                    return Self::detect_amd();
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Check for AWS Graviton or Apple Silicon
            return Self::detect_arm();
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Unknown
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        Self::Unknown
    }

    #[cfg(target_arch = "x86_64")]
    fn get_x86_vendor() -> Option<String> {
        // Read vendor string from CPUID
        // SAFETY: CPUID is always safe to call on x86_64, it's a CPU instruction
        // that returns information about the processor.
        let cpuid = unsafe { std::arch::x86_64::__cpuid(0) };
        let vendor = [
            cpuid.ebx.to_le_bytes(),
            cpuid.edx.to_le_bytes(),
            cpuid.ecx.to_le_bytes(),
        ]
        .concat();
        String::from_utf8(vendor).ok()
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_intel() -> Self {
        // Check for AVX-512 as indicator of newer architectures
        if std::arch::is_x86_feature_detected!("avx512f") {
            Self::IntelSapphire
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Self::IntelSkylake
        } else {
            Self::Unknown
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_amd() -> Self {
        // Check for AVX-512 as indicator of Zen 4+
        if std::arch::is_x86_feature_detected!("avx512f") {
            Self::AmdGenoa
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Self::AmdMilan
        } else {
            Self::Unknown
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm() -> Self {
        // Try to read /proc/cpuinfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                if cpuinfo.contains("Neoverse-V1") || cpuinfo.contains("0xd40") {
                    return Self::Graviton3;
                }
                if cpuinfo.contains("Neoverse-N1") || cpuinfo.contains("0xd0c") {
                    return Self::Graviton2;
                }
            }
        }

        // macOS detection
        #[cfg(target_os = "macos")]
        {
            return Self::AppleSilicon;
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        Self::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_scale_log2() {
        let scale = ParameterScale::log2(6, 10);
        let values: Vec<_> = scale.iter().collect();
        assert_eq!(values, vec![64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_parameter_scale_linear() {
        let scale = ParameterScale::Linear {
            start: 100,
            end: 500,
            step: 100,
        };
        let values: Vec<_> = scale.iter().collect();
        assert_eq!(values, vec![100, 200, 300, 400, 500]);
    }

    #[test]
    fn test_parameter_scale_explicit() {
        let scale = ParameterScale::Explicit(vec![10, 50, 100, 1000]);
        let values: Vec<_> = scale.iter().collect();
        assert_eq!(values, vec![10, 50, 100, 1000]);
    }

    #[test]
    fn test_variant_creation() {
        let variant = Variant::new("avx2").with_features(&["avx2"]);
        assert_eq!(variant.name, "avx2");
        assert_eq!(variant.required_features, vec!["avx2"]);
    }

    #[test]
    fn test_cpu_class_detect() {
        // This just tests that detection doesn't panic
        let _class = CpuClass::detect();
    }
}
