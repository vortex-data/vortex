//! Example: Sum benchmark using the builder API.
//!
//! This demonstrates the fluent builder API for defining benchmarks,
//! which is simpler than implementing the full trait.

use rand::Rng;
use rand::SeedableRng;
use vortex_threshold_traits::AlgorithmRegistry;
use vortex_threshold_traits::ParameterScale;
use vortex_threshold_traits::ThresholdBench;

/// Registers sum benchmarks using the builder API.
pub fn register(registry: &mut AlgorithmRegistry) {
    ThresholdBench::new("sum")
        .parameter("count", ParameterScale::log2(6, 20))
        .input(|size, seed| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            (0..size).map(|_| rng.random::<u64>()).collect::<Vec<_>>()
        })
        .baseline("naive", |data| sum_naive(data))
        .variant("unrolled", |data| sum_unrolled(data))
        .variant("chunks", |data| sum_chunks(data))
        .register(registry);
}

/// Simple loop-based sum.
fn sum_naive(data: &[u64]) -> u64 {
    let mut total = 0u64;
    for &x in data {
        total = total.wrapping_add(x);
    }
    total
}

/// Manually unrolled 4x sum.
fn sum_unrolled(data: &[u64]) -> u64 {
    let mut a0 = 0u64;
    let mut a1 = 0u64;
    let mut a2 = 0u64;
    let mut a3 = 0u64;

    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        a0 = a0.wrapping_add(chunk[0]);
        a1 = a1.wrapping_add(chunk[1]);
        a2 = a2.wrapping_add(chunk[2]);
        a3 = a3.wrapping_add(chunk[3]);
    }

    let mut total = a0.wrapping_add(a1).wrapping_add(a2).wrapping_add(a3);
    for &x in remainder {
        total = total.wrapping_add(x);
    }
    total
}

/// Iterator-based chunked sum.
fn sum_chunks(data: &[u64]) -> u64 {
    data.chunks(64)
        .map(|chunk| chunk.iter().copied().fold(0u64, u64::wrapping_add))
        .fold(0u64, u64::wrapping_add)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_correctness() {
        let data: Vec<u64> = (0..1000).collect();
        let expected = sum_naive(&data);
        assert_eq!(sum_unrolled(&data), expected);
        assert_eq!(sum_chunks(&data), expected);
    }
}
