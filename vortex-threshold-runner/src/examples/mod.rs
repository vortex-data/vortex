//! Example benchmark implementations demonstrating the ISA threshold detection pattern.
//!
//! These examples show two approaches:
//! 1. **Trait-based** (`popcount`): Implement [`BenchmarkableAlgorithm`] for full control
//! 2. **Builder-based** (`sum`): Use [`ThresholdBench`] for a fluent, criterion-like API

mod popcount;
mod sum;

pub use popcount::PopcountBenchmark;
use vortex_threshold_traits::AlgorithmRegistry;

/// Registers all example benchmarks with the given registry.
pub fn register_all(registry: &mut AlgorithmRegistry) {
    // Trait-based example
    registry.register(PopcountBenchmark);

    // Builder-based example
    sum::register(registry);
}
