//! Example: Popcount (population count / bit counting) benchmark.
//!
//! This demonstrates threshold detection for a classic algorithm where SIMD
//! implementations become faster than scalar code at certain input sizes.

use rand::Rng;
use rand::SeedableRng;
use vortex_threshold_traits::BenchmarkableAlgorithm;
use vortex_threshold_traits::ParameterScale;
use vortex_threshold_traits::Variant;

/// Benchmark for population count (counting set bits) algorithms.
pub struct PopcountBenchmark;

impl BenchmarkableAlgorithm for PopcountBenchmark {
    type Input = Vec<u64>;
    type Output = usize;

    fn name(&self) -> &'static str {
        "popcount"
    }

    fn parameter_name(&self) -> &'static str {
        "input_size"
    }

    fn parameter_scale(&self) -> ParameterScale {
        // Test from 64 elements to 1M elements (powers of 2)
        ParameterScale::log2(6, 20)
    }

    fn variants(&self) -> Vec<Variant> {
        let mut variants = vec![
            Variant::new("naive"),
            Variant::new("lookup_table"),
            Variant::new("builtin"),
        ];

        // Add SIMD variants based on architecture
        #[cfg(target_arch = "x86_64")]
        {
            variants.push(Variant::new("popcnt").with_features(&["popcnt"]));
            variants.push(Variant::new("avx2").with_features(&["avx2", "popcnt"]));
        }

        #[cfg(target_arch = "aarch64")]
        {
            variants.push(Variant::new("neon").with_features(&["neon"]));
        }

        variants
    }

    fn generate_input(&self, param: usize, seed: u64) -> Self::Input {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..param).map(|_| rng.random()).collect()
    }

    fn ground_truth(&self, input: &Self::Input) -> Self::Output {
        popcount_naive(input)
    }

    fn run_variant(&self, variant: &str, input: &Self::Input) -> Self::Output {
        match variant {
            "naive" => popcount_naive(input),
            "lookup_table" => popcount_lookup(input),
            "builtin" => popcount_builtin(input),
            #[cfg(target_arch = "x86_64")]
            "popcnt" => popcount_popcnt(input),
            #[cfg(target_arch = "x86_64")]
            "avx2" => popcount_avx2(input),
            #[cfg(target_arch = "aarch64")]
            "neon" => popcount_neon(input),
            _ => popcount_naive(input),
        }
    }
}

/// Naive bit-by-bit counting implementation.
fn popcount_naive(data: &[u64]) -> usize {
    let mut count = 0usize;
    for &word in data {
        let mut w = word;
        while w != 0 {
            count += (w & 1) as usize;
            w >>= 1;
        }
    }
    count
}

/// Lookup table based implementation (256-byte table).
fn popcount_lookup(data: &[u64]) -> usize {
    // Precomputed popcount for each byte value
    static LOOKUP: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            // Count bits in byte value i
            #[allow(clippy::cast_possible_truncation)]
            let byte_val = i as u8;
            #[allow(clippy::cast_possible_truncation)]
            let bit_count = byte_val.count_ones() as u8;
            table[i] = bit_count;
            i += 1;
        }
        table
    };

    let mut count = 0usize;
    for &word in data {
        // Use little-endian for consistent behavior across platforms
        let bytes = word.to_le_bytes();
        for byte in bytes {
            count += LOOKUP[byte as usize] as usize;
        }
    }
    count
}

/// Uses Rust's built-in count_ones which compiles to POPCNT on supported CPUs.
fn popcount_builtin(data: &[u64]) -> usize {
    data.iter().map(|&w| w.count_ones() as usize).sum()
}

/// x86_64 POPCNT instruction implementation.
#[cfg(target_arch = "x86_64")]
fn popcount_popcnt(data: &[u64]) -> usize {
    // Uses count_ones which compiles to POPCNT when available
    data.iter().map(|&w| w.count_ones() as usize).sum()
}

/// x86_64 AVX2 implementation using VPSHUFB-based popcount.
#[cfg(target_arch = "x86_64")]
fn popcount_avx2(data: &[u64]) -> usize {
    // For simplicity, this falls back to the builtin which will use POPCNT.
    // A real AVX2 implementation would use VPSHUFB (parallel lookup) for
    // large arrays where the setup cost is amortized.

    if data.len() < 32 {
        // Below threshold, use scalar
        return popcount_builtin(data);
    }

    // For larger data, process in chunks (simplified - real impl would use VPSHUFB)
    // The key insight is that AVX2 VPSHUFB can process 32 bytes at once,
    // but has higher latency than scalar POPCNT for small inputs.
    data.iter().map(|&w| w.count_ones() as usize).sum()
}

/// AArch64 NEON implementation.
#[cfg(target_arch = "aarch64")]
fn popcount_neon(data: &[u64]) -> usize {
    // NEON has VCNT which counts bits in each byte.
    // For simplicity, use count_ones which compiles to efficient ARM instructions.
    data.iter().map(|&w| w.count_ones() as usize).sum()
}

#[cfg(test)]
mod tests {
    use vortex_threshold_traits::BenchmarkableAlgorithm;

    use super::*;

    #[test]
    fn test_popcount_correctness() {
        let benchmark = PopcountBenchmark;
        let input = benchmark.generate_input(1000, 42);
        let expected = benchmark.ground_truth(&input);

        // Test all available variants
        for variant in benchmark.variants() {
            if variant.is_available() {
                let result = benchmark.run_variant(&variant.name, &input);
                assert_eq!(
                    result, expected,
                    "Variant {} produced incorrect result",
                    variant.name
                );
            }
        }
    }

    #[test]
    fn test_popcount_known_values() {
        let data = vec![0u64, 1, 0xFF, 0xFFFF_FFFF_FFFF_FFFF];
        // 0 bits + 1 bit + 8 bits + 64 bits = 73 bits
        let expected = 73;
        assert_eq!(popcount_naive(&data), expected);
        assert_eq!(popcount_lookup(&data), expected);
        assert_eq!(popcount_builtin(&data), expected);
    }
}
