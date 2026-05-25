// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experiment: a split (hi, lo) struct-of-arrays layout for decimal i128/i256
//! storage, with AVX-512 multi-limb arithmetic kernels, compared against
//! Arrow's interleaved decimal kernels for both compression and arithmetic.
//!
//! See `src/bin/analyze.rs` for the report driver and `benches/decimal_arith.rs`
//! for the Divan benchmarks.

pub mod aggregate;
pub mod arrow_ref;
pub mod compare;
pub mod compress;
pub mod cpu;
pub mod data;
pub mod layout;
pub mod muldiv;
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

            let mut out_u4 = sa.zeroed_like();
            simd::add_i128_u4(&sa, &sb, &mut out_u4);
            assert_eq!(out_u4.to_aos(), expected, "simd u4 {mag:?}");
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

    #[test]
    fn i128_compare_matches_arrow() {
        use crate::compare;
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let a = gen_i128(N, mag, 70);
            let b = gen_i128(N, mag, 71);
            // Force some equal elements to exercise the hi-equal tiebreak.
            let mut b = b;
            b[..50].copy_from_slice(&a[..50]);

            let arrow_lt = arrow_ref::lt_decimal128(
                &arrow_ref::decimal128(&a, 38, 0),
                &arrow_ref::decimal128(&b, 38, 0),
            );
            let arrow_eq = arrow_ref::eq_decimal128(
                &arrow_ref::decimal128(&a, 38, 0),
                &arrow_ref::decimal128(&b, 38, 0),
            );

            let sa = SplitI128::from_aos(&a);
            let sb = SplitI128::from_aos(&b);
            let mut lt = vec![0u8; compare::bitmap_len(N)];
            let mut eq = vec![0u8; compare::bitmap_len(N)];
            compare::lt_i128(&sa, &sb, &mut lt);
            compare::eq_i128(&sa, &sb, &mut eq);

            for i in 0..N {
                assert_eq!(
                    compare::get_bit(&lt, i),
                    arrow_ref::boolean_at(&arrow_lt, i),
                    "lt {mag:?} @ {i}"
                );
                assert_eq!(
                    compare::get_bit(&eq, i),
                    arrow_ref::boolean_at(&arrow_eq, i),
                    "eq {mag:?} @ {i}"
                );
            }
        }
    }

    #[test]
    fn i256_compare_matches_arrow() {
        use crate::compare;
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let a = gen_i256(N, mag, 80);
            let mut b = gen_i256(N, mag, 81);
            b[..50].copy_from_slice(&a[..50]);

            let arrow_lt = arrow_ref::lt_decimal256(
                &arrow_ref::decimal256(&a, 76, 0),
                &arrow_ref::decimal256(&b, 76, 0),
            );

            let sa = SplitI256::from_aos(&a);
            let sb = SplitI256::from_aos(&b);
            let mut lt = vec![0u8; compare::bitmap_len(N)];
            compare::lt_i256(&sa, &sb, &mut lt);

            for i in 0..N {
                assert_eq!(
                    compare::get_bit(&lt, i),
                    arrow_ref::boolean_at(&arrow_lt, i),
                    "lt {mag:?} @ {i}"
                );
            }
        }
    }

    #[test]
    fn i128_sum_widening_is_exact_and_overflow_safe() {
        use crate::aggregate;
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let v = gen_i128(N, mag, 90);
            let split = SplitI128::from_aos(&v);

            // Ground truth: fold each value into i256.
            let expected = v
                .iter()
                .fold(i256::ZERO, |acc, &x| acc.wrapping_add(i256::from_i128(x)));
            assert_eq!(
                aggregate::sum_i128_widening_scalar(&split),
                expected,
                "scalar {mag:?}"
            );
            assert_eq!(
                aggregate::sum_i128_widening(&split),
                expected,
                "simd {mag:?}"
            );

            if mag == Magnitude::Small {
                // Small decimals have hi == 0, so the lo-only fast path is exact.
                assert_eq!(aggregate::sum_i128_lo_only(&split), expected, "lo-only");
            }
        }

        // Overflow safety: a same-width i128 accumulator wraps; the widening one
        // stays exact.
        let overflow = vec![i128::MAX; 100];
        let split = SplitI128::from_aos(&overflow);
        let expected = i256::from_i128(i128::MAX).wrapping_mul(i256::from_i128(100));
        assert_eq!(aggregate::sum_i128_widening_scalar(&split), expected);
        assert_ne!(
            aggregate::sum_i128_naive_wrapping(&split),
            i128::MAX, // wrapped: definitely not the true sum
        );
    }

    #[test]
    fn i128_min_max_match_arrow() {
        use crate::aggregate;
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let v = gen_i128(N, mag, 91);
            let split = SplitI128::from_aos(&v);

            let arrow = arrow_ref::decimal128(&v, 38, 0);
            let amin = arrow_ref::min_decimal128(&arrow);
            let amax = arrow_ref::max_decimal128(&arrow);

            assert_eq!(
                aggregate::min_i128_scalar(&split),
                amin,
                "min scalar {mag:?}"
            );
            assert_eq!(
                aggregate::max_i128_scalar(&split),
                amax,
                "max scalar {mag:?}"
            );
            assert_eq!(aggregate::min_i128(&split), amin, "min simd {mag:?}");
            assert_eq!(aggregate::max_i128(&split), amax, "max simd {mag:?}");
        }
    }

    #[test]
    fn i128_mul_matches_arrow_and_native() {
        use crate::muldiv;
        for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
            let a = gen_i128(N, mag, 12);
            let b = gen_i128(N, mag, 13);
            let expected: Vec<i128> = a.iter().zip(&b).map(|(&x, &y)| x.wrapping_mul(y)).collect();

            // Arrow validates the product against precision 38, so only the small
            // case (product < 10^38) is Arrow-comparable; the kernels are checked
            // against native wrapping_mul for every magnitude below.
            if mag == Magnitude::Small {
                let arrow = arrow_ref::mul_decimal128(
                    &arrow_ref::decimal128(&a, 38, 0),
                    &arrow_ref::decimal128(&b, 38, 0),
                );
                assert_eq!(
                    arrow_ref::decimal128_values(&arrow),
                    expected,
                    "arrow {mag:?}"
                );
            }

            let sa = SplitI128::from_aos(&a);
            let sb = SplitI128::from_aos(&b);

            let mut aos = vec![0i128; N];
            muldiv::mul_i128_aos(&a, &b, &mut aos);
            assert_eq!(aos, expected, "aos {mag:?}");

            let mut sc = sa.zeroed_like();
            muldiv::mul_i128_soa_scalar(&sa, &sb, &mut sc);
            assert_eq!(sc.to_aos(), expected, "soa scalar {mag:?}");

            let mut si = sa.zeroed_like();
            muldiv::mul_i128(&sa, &sb, &mut si);
            assert_eq!(si.to_aos(), expected, "soa simd {mag:?}");
        }
    }

    #[test]
    fn i128_div_matches_native() {
        // Our kernel is truncating integer division. (Arrow's decimal div
        // rescales and rounds, so it is a different operation - benchmarked for
        // throughput but not asserted equal here.)
        use crate::muldiv;
        let a = gen_i128(N, Magnitude::Large, 14);
        let b: Vec<i128> = gen_i128(N, Magnitude::Small, 15)
            .into_iter()
            .map(|v| v + 1) // avoid zero divisors
            .collect();
        let expected: Vec<i128> = a.iter().zip(&b).map(|(&x, &y)| x / y).collect();

        let mut aos = vec![0i128; N];
        muldiv::div_i128_aos(&a, &b, &mut aos);
        assert_eq!(aos, expected, "aos div");

        let sa = SplitI128::from_aos(&a);
        let sb = SplitI128::from_aos(&b);
        let mut soa = sa.zeroed_like();
        muldiv::div_i128_soa(&sa, &sb, &mut soa);
        assert_eq!(soa.to_aos(), expected, "soa div");
    }

    /// Build a SplitI128 with an explicit constant high limb.
    fn split_with_const_hi(lo: Vec<u64>, hi_const: u64) -> SplitI128 {
        let hi = vec![hi_const; lo.len()];
        SplitI128 { lo, hi }
    }

    #[test]
    fn sum_const_hi_matches_widening() {
        use crate::aggregate;
        let lo: Vec<u64> = gen_i128(N, Magnitude::Small, 21)
            .iter()
            .map(|&v| v as u64)
            .collect();
        // Try zero, a positive, and a negative (sign-word) constant high limb.
        for hi_const in [0u64, 7, u64::MAX /* = -1 */] {
            let split = split_with_const_hi(lo.clone(), hi_const);
            let expected = aggregate::sum_i128_widening(&split);
            let got = aggregate::sum_i128_const_hi(&split.lo, hi_const);
            assert_eq!(got, expected, "hi_const={hi_const}");
        }
    }

    #[test]
    fn sum_blockwise_matches_widening() {
        use crate::aggregate;
        use crate::data::gen_i128_blocked;
        const BLK: usize = 256;
        for frac in [0.0, 0.5, 1.0] {
            let (values, meta) = gen_i128_blocked(5000, BLK, frac, 31);
            let split = SplitI128::from_aos(&values);
            let expected = aggregate::sum_i128_widening(&split);
            let got = aggregate::sum_i128_blockwise(&split.lo, &split.hi, &meta, BLK);
            assert_eq!(got, expected, "frac={frac}");
        }
    }

    #[test]
    fn lt_blockwise_matches_full() {
        use crate::compare;
        use crate::data::gen_i128_blocked;
        const BLK: usize = 256; // multiple of 8
        let (av, am) = gen_i128_blocked(5000, BLK, 0.5, 41);
        let (bv, bm) = gen_i128_blocked(5000, BLK, 0.5, 42);
        let sa = SplitI128::from_aos(&av);
        let sb = SplitI128::from_aos(&bv);

        let mut full = vec![0u8; compare::bitmap_len(av.len())];
        compare::lt_i128(&sa, &sb, &mut full);
        let mut blk = vec![0u8; compare::bitmap_len(av.len())];
        compare::lt_i128_blockwise(&sa, &am, &sb, &bm, BLK, &mut blk);

        for i in 0..av.len() {
            assert_eq!(
                compare::get_bit(&blk, i),
                compare::get_bit(&full, i),
                "@ {i}"
            );
        }
    }

    #[test]
    fn lt_const_hi_matches_full_compare() {
        use crate::compare;
        let la: Vec<u64> = gen_i128(N, Magnitude::Small, 22)
            .iter()
            .map(|&v| v as u64)
            .collect();
        let lb: Vec<u64> = gen_i128(N, Magnitude::Small, 23)
            .iter()
            .map(|&v| v as u64)
            .collect();

        // Equal high constants -> low-limb comparison; and differing highs ->
        // whole-column constant result. Both must match the full lexicographic
        // kernel.
        for (ca, cb) in [(0u64, 0u64), (5, 5), (0, 1), (3, 2), (u64::MAX, 0)] {
            let a = split_with_const_hi(la.clone(), ca);
            let b = split_with_const_hi(lb.clone(), cb);

            let mut full = vec![0u8; compare::bitmap_len(N)];
            compare::lt_i128(&a, &b, &mut full);

            let mut fast = vec![0u8; compare::bitmap_len(N)];
            compare::lt_i128_const_hi(&a.lo, ca, &b.lo, cb, &mut fast);

            for i in 0..N {
                assert_eq!(
                    compare::get_bit(&fast, i),
                    compare::get_bit(&full, i),
                    "ca={ca} cb={cb} @ {i}"
                );
            }
        }
    }
}
