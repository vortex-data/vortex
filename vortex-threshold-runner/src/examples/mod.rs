//! Example benchmark implementations demonstrating the ISA threshold detection pattern.
//!
//! These examples show how to implement the [`BenchmarkableAlgorithm`] trait for
//! various algorithms with different ISA-specific implementations.

mod popcount;

pub use popcount::PopcountBenchmark;
use vortex_threshold_traits::AlgorithmRegistry;

/// Registers all example benchmarks with the given registry.
pub fn register_all(registry: &mut AlgorithmRegistry) {
    registry.register(PopcountBenchmark);
}
