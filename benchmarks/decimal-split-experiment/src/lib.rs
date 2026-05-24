// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experiment: a split (hi, lo) struct-of-arrays layout for decimal i128/i256
//! storage, with AVX-512 multi-limb arithmetic kernels, compared against
//! Arrow's interleaved decimal kernels for both compression and arithmetic.
//!
//! See `src/bin/analyze.rs` for the report driver and `benches/decimal_arith.rs`
//! for the Divan benchmarks.

pub mod arrow_ref;
pub mod compress;
pub mod cpu;
pub mod data;
pub mod layout;
pub mod scalar;
pub mod simd;

#[cfg(test)]
mod tests {
    use arrow_buffer::i256;
    use rand::RngExt;
    use rand::SeedableRng;

    use crate::arrow_ref;
    use crate::data::Magnitude;
    use crate::data::gen_i128;
    use crate::data::gen_i256;
    use crate::layout::SplitI128;
    use crate::layout::SplitI256;
    use crate::scalar;
    use crate::simd;

    const N: usize = 1000; // not a multiple of 8, exercises the SIMD tail

    #[test]
    fn split_i128_roundtrips() {
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let values = gen_i128(N, mag, 1);
            assert_eq!(SplitI128::from_aos(&values).to_aos(), values);
        }
    }

    #[test]
    fn split_i256_roundtrips() {
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let values = gen_i256(N, mag, 2);
            assert_eq!(SplitI256::from_aos(&values).to_aos(), values);
        }
    }

    #[test]
    fn i128_add_matches_arrow_and_native() {
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let a = gen_i128(N, mag, 10);
            let b = gen_i128(N, mag, 20);

            let expected: Vec<i128> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_add(y)).collect();

            // Arrow interleaved kernel (precision 38 holds full i128).
            let arrow = arrow_ref::add_decimal128(
                &arrow_ref::decimal128(&a, 38, 0),
                &arrow_ref::decimal128(&b, 38, 0),
            );
            assert_eq!(arrow_ref::decimal128_values(&arrow), expected);

            // Split scalar and SIMD kernels.
            let sa = SplitI128::from_aos(&a);
            let sb = SplitI128::from_aos(&b);
            let mut out = sa.zeroed_like();

            scalar::add_i128_soa(&sa, &sb, &mut out);
            assert_eq!(out.to_aos(), expected, "scalar soa {mag:?}");

            let mut out_simd = sa.zeroed_like();
            simd::add_i128(&sa, &sb, &mut out_simd);
            assert_eq!(out_simd.to_aos(), expected, "simd soa {mag:?}");
        }
    }

    #[test]
    fn i128_add_full_range_wraps() {
        // Genuine full-width random values (top bits set), verifying carry into
        // and wraparound out of the very top bit. No Arrow here: Arrow's checked
        // kernel rejects results that exceed the declared precision.
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let a: Vec<i128> = (0..N).map(|_| rng.random::<i128>()).collect();
        let b: Vec<i128> = (0..N).map(|_| rng.random::<i128>()).collect();
        let expected: Vec<i128> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_add(y)).collect();

        let sa = SplitI128::from_aos(&a);
        let sb = SplitI128::from_aos(&b);
        let mut out = sa.zeroed_like();
        simd::add_i128(&sa, &sb, &mut out);
        assert_eq!(out.to_aos(), expected);
    }

    #[test]
    fn i128_sub_matches_native() {
        let a = gen_i128(N, Magnitude::Large, 30);
        let b = gen_i128(N, Magnitude::Large, 40);
        let expected: Vec<i128> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_sub(y)).collect();

        let sa = SplitI128::from_aos(&a);
        let sb = SplitI128::from_aos(&b);
        let mut out = sa.zeroed_like();
        simd::sub_i128(&sa, &sb, &mut out);
        assert_eq!(out.to_aos(), expected);
    }

    #[test]
    fn i256_add_matches_arrow_and_native() {
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let a = gen_i256(N, mag, 50);
            let b = gen_i256(N, mag, 60);

            let expected: Vec<i256> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_add(y)).collect();

            let arrow = arrow_ref::add_decimal256(
                &arrow_ref::decimal256(&a, 76, 0),
                &arrow_ref::decimal256(&b, 76, 0),
            );
            assert_eq!(arrow_ref::decimal256_values(&arrow), expected);

            let sa = SplitI256::from_aos(&a);
            let sb = SplitI256::from_aos(&b);
            let mut out = sa.zeroed_like();
            scalar::add_i256_soa(&sa, &sb, &mut out);
            assert_eq!(out.to_aos(), expected, "scalar soa {mag:?}");

            let mut out_simd = sa.zeroed_like();
            simd::add_i256(&sa, &sb, &mut out_simd);
            assert_eq!(out_simd.to_aos(), expected, "simd soa {mag:?}");
        }
    }
}
