//! Builder API for defining threshold benchmarks.
//!
//! This module provides a fluent, criterion-like API for defining benchmarks
//! without the boilerplate of implementing the full [`BenchmarkableAlgorithm`] trait.
//!
//! # Example
//!
//! ```ignore
//! use vortex_threshold_traits::ThresholdBench;
//! use vortex_threshold_traits::ParameterScale;
//!
//! ThresholdBench::new("popcount")
//!     .parameter("input_size", ParameterScale::log2(6, 20))
//!     .input(|size, seed| random_vec::<u64>(size, seed))
//!     .baseline("naive", popcount_naive)
//!     .variant("builtin", popcount_builtin)
//!     .variant_if("avx2", is_x86_feature_detected!("avx2"), popcount_avx2)
//!     .register(&mut registry);
//! ```

use std::fmt::Debug;
use std::sync::Arc;

use crate::AlgorithmRegistry;
use crate::BenchmarkableAlgorithm;
use crate::ParameterScale;
use crate::Variant;

/// A closure that generates input data.
type InputGenFn<I> = Arc<dyn Fn(usize, u64) -> I + Send + Sync>;

/// A closure that runs a variant on input data.
type VariantFn<I, O> = Arc<dyn Fn(&I) -> O + Send + Sync>;

/// Builder for threshold benchmarks with a fluent API.
///
/// # Example
///
/// ```ignore
/// ThresholdBench::new("popcount")
///     .parameter("input_size", ParameterScale::log2(6, 20))
///     .input(|size, seed| random_vec::<u64>(size, seed))
///     .baseline("naive", popcount_naive)
///     .variant("builtin", popcount_builtin)
///     .register(&mut registry);
/// ```
pub struct ThresholdBench<I, O> {
    name: &'static str,
    parameter_name: &'static str,
    parameter_scale: ParameterScale,
    input_gen: Option<InputGenFn<I>>,
    baseline: Option<(&'static str, VariantFn<I, O>)>,
    variants: Vec<VariantEntry<I, O>>,
}

struct VariantEntry<I, O> {
    name: &'static str,
    features: Vec<&'static str>,
    available: bool,
    func: VariantFn<I, O>,
}

impl<I, O> ThresholdBench<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: PartialEq + Debug + Send + Sync + 'static,
{
    /// Creates a new threshold benchmark with the given name.
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            parameter_name: "size",
            parameter_scale: ParameterScale::log2(6, 20),
            input_gen: None,
            baseline: None,
            variants: Vec::new(),
        }
    }

    /// Sets the parameter name and scale.
    #[must_use]
    pub fn parameter(mut self, name: &'static str, scale: ParameterScale) -> Self {
        self.parameter_name = name;
        self.parameter_scale = scale;
        self
    }

    /// Sets the input generator function.
    ///
    /// The function receives `(param, seed)` and returns input data.
    #[must_use]
    pub fn input<F>(mut self, generator: F) -> Self
    where
        F: Fn(usize, u64) -> I + Send + Sync + 'static,
    {
        self.input_gen = Some(Arc::new(generator));
        self
    }

    /// Sets the baseline variant (used as ground truth for verification).
    ///
    /// The baseline is always available and is used to verify other variants.
    #[must_use]
    pub fn baseline<F>(mut self, name: &'static str, func: F) -> Self
    where
        F: Fn(&I) -> O + Send + Sync + 'static,
    {
        self.baseline = Some((name, Arc::new(func)));
        self
    }

    /// Adds a variant that is always available.
    #[must_use]
    pub fn variant<F>(mut self, name: &'static str, func: F) -> Self
    where
        F: Fn(&I) -> O + Send + Sync + 'static,
    {
        self.variants.push(VariantEntry {
            name,
            features: Vec::new(),
            available: true,
            func: Arc::new(func),
        });
        self
    }

    /// Adds a variant that is conditionally available.
    ///
    /// Use this for ISA-specific variants:
    /// ```ignore
    /// .variant_if("avx2", is_x86_feature_detected!("avx2"), popcount_avx2)
    /// ```
    #[must_use]
    pub fn variant_if<F>(mut self, name: &'static str, available: bool, func: F) -> Self
    where
        F: Fn(&I) -> O + Send + Sync + 'static,
    {
        self.variants.push(VariantEntry {
            name,
            features: Vec::new(),
            available,
            func: Arc::new(func),
        });
        self
    }

    /// Adds a variant with required CPU features.
    ///
    /// Features are checked at runtime using [`Variant::is_available`].
    #[must_use]
    pub fn variant_with_features<F>(
        mut self,
        name: &'static str,
        features: &[&'static str],
        func: F,
    ) -> Self
    where
        F: Fn(&I) -> O + Send + Sync + 'static,
    {
        let variant = Variant::new(name).with_features(features);
        self.variants.push(VariantEntry {
            name,
            features: features.to_vec(),
            available: variant.is_available(),
            func: Arc::new(func),
        });
        self
    }

    /// Registers this benchmark with the given registry.
    ///
    /// # Panics
    ///
    /// Panics if no input generator or baseline has been set.
    #[allow(clippy::expect_used)]
    pub fn register(self, registry: &mut AlgorithmRegistry) {
        let input_gen = self
            .input_gen
            .expect("ThresholdBench requires an input generator (.input(...))");
        let (baseline_name, baseline_func) = self
            .baseline
            .expect("ThresholdBench requires a baseline (.baseline(...))");

        let bench = BuiltBenchmark {
            name: self.name,
            parameter_name: self.parameter_name,
            parameter_scale: self.parameter_scale,
            input_gen,
            baseline_name,
            baseline_func,
            variants: self.variants,
        };

        registry.register(bench);
    }

    /// Builds without registering (for testing).
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn build(self) -> impl BenchmarkableAlgorithm<Input = I, Output = O> {
        let input_gen = self
            .input_gen
            .expect("ThresholdBench requires an input generator (.input(...))");
        let (baseline_name, baseline_func) = self
            .baseline
            .expect("ThresholdBench requires a baseline (.baseline(...))");

        BuiltBenchmark {
            name: self.name,
            parameter_name: self.parameter_name,
            parameter_scale: self.parameter_scale,
            input_gen,
            baseline_name,
            baseline_func,
            variants: self.variants,
        }
    }
}

/// Internal struct that implements `BenchmarkableAlgorithm`.
struct BuiltBenchmark<I, O> {
    name: &'static str,
    parameter_name: &'static str,
    parameter_scale: ParameterScale,
    input_gen: InputGenFn<I>,
    baseline_name: &'static str,
    baseline_func: VariantFn<I, O>,
    variants: Vec<VariantEntry<I, O>>,
}

impl<I, O> BenchmarkableAlgorithm for BuiltBenchmark<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: PartialEq + Debug + Send + Sync + 'static,
{
    type Input = I;
    type Output = O;

    fn name(&self) -> &'static str {
        self.name
    }

    fn parameter_name(&self) -> &'static str {
        self.parameter_name
    }

    fn parameter_scale(&self) -> ParameterScale {
        self.parameter_scale.clone()
    }

    fn variants(&self) -> Vec<Variant> {
        let mut variants = vec![Variant::new(self.baseline_name)];
        for v in &self.variants {
            let mut variant = Variant::new(v.name);
            if !v.features.is_empty() {
                variant = variant.with_features(&v.features);
            }
            variants.push(variant);
        }
        variants
    }

    fn generate_input(&self, param: usize, seed: u64) -> Self::Input {
        (self.input_gen)(param, seed)
    }

    fn ground_truth(&self, input: &Self::Input) -> Self::Output {
        (self.baseline_func)(input)
    }

    fn run_variant(&self, variant: &str, input: &Self::Input) -> Self::Output {
        if variant == self.baseline_name {
            return (self.baseline_func)(input);
        }

        for v in &self.variants {
            if v.name == variant && v.available {
                return (v.func)(input);
            }
        }

        // Fallback to baseline if variant not found
        (self.baseline_func)(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParameterScale;

    fn sum_naive(data: &Vec<u64>) -> u64 {
        data.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))
    }

    fn sum_chunked(data: &Vec<u64>) -> u64 {
        data.chunks(4)
            .map(|c| c.iter().fold(0u64, |acc, &x| acc.wrapping_add(x)))
            .fold(0u64, |acc, x| acc.wrapping_add(x))
    }

    #[test]
    fn test_builder_basic() {
        let bench = ThresholdBench::new("sum")
            .parameter("count", ParameterScale::log2(4, 8))
            .input(|size, seed| {
                use rand::Rng;
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                (0..size).map(|_| rng.random()).collect()
            })
            .baseline("naive", sum_naive)
            .variant("chunked", sum_chunked)
            .build();

        assert_eq!(bench.name(), "sum");
        assert_eq!(bench.parameter_name(), "count");
        assert_eq!(bench.variants().len(), 2);

        let input = bench.generate_input(100, 42);
        let expected = bench.ground_truth(&input);
        let actual = bench.run_variant("chunked", &input);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_builder_variant_if() {
        let bench = ThresholdBench::new("test")
            .input(|size, _| vec![0u64; size])
            .baseline("base", |_: &Vec<u64>| 0u64)
            .variant_if("always", true, |_: &Vec<u64>| 0u64)
            .variant_if("never", false, |_: &Vec<u64>| 0u64)
            .build();

        let variants = bench.variants();
        assert_eq!(variants.len(), 3);
    }

    #[test]
    fn test_builder_register() {
        let mut registry = AlgorithmRegistry::new();

        ThresholdBench::new("test")
            .input(|size, _| vec![0u64; size])
            .baseline("base", |_: &Vec<u64>| 0u64)
            .register(&mut registry);

        assert_eq!(registry.len(), 1);
    }
}
