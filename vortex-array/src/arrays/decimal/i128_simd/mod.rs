// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD-accelerated elementwise add and sum over `i128` (128-bit decimal) buffers.
//!
//! AVX-512 has no native 128-bit integer add — the widest is `vpaddq` (8 x 64-bit). The
//! kernels here implement 128-bit add-with-carry directly over the array-of-structs
//! `[lo, hi, lo, hi, ...]` layout of `Buffer<i128>`, without deinterleaving into a
//! struct-of-arrays form: a lanewise `vpaddq` produces the per-64-bit-lane sums, an
//! unsigned less-than comparison (`vpcmpuq`) detects the carry out of each low lane, and a
//! masked add (`vpaddq` under a write-mask) folds that carry into the adjacent high lane.
//!
//! A scalar fallback with identical wrapping semantics is used on non-`x86_64` hosts or
//! when `avx512f` is not detected at runtime.

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

#[cfg(target_arch = "x86_64")]
mod avx512;

/// Elementwise wrapping add of two equal-length `i128` slices into a new [`Buffer`].
///
/// On `x86_64` hosts with `avx512f` this uses a vectorized 128-bit add-with-carry,
/// otherwise it falls back to a scalar [`i128::wrapping_add`] loop. The result is
/// identical regardless of the path taken.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn add_i128(a: &[i128], b: &[i128]) -> Buffer<i128> {
    assert_eq!(a.len(), b.len(), "add_i128 requires equal-length inputs");

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: `avx512f` is available and the lengths are equal.
            return unsafe { avx512::add_i128_avx512(a, b) };
        }
    }

    add_i128_scalar(a, b)
}

/// Wrapping sum of all elements of an `i128` slice.
///
/// On `x86_64` hosts with `avx512f` this accumulates four partial 128-bit sums in a single
/// vector register before combining them, otherwise it falls back to a scalar loop.
pub fn sum_i128(values: &[i128]) -> i128 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: `avx512f` is available.
            return unsafe { avx512::sum_i128_avx512(values) };
        }
    }

    sum_i128_scalar(0, values)
}

/// Checked sum of `initial` plus all elements of an `i128` slice.
///
/// Returns [`None`] if any intermediate addition overflows `i128`, matching the semantics
/// of a scalar [`i128::checked_add`] reduction. On `x86_64` hosts with `avx512f` and
/// `avx512dq` this uses a vectorized add-with-carry that tracks signed overflow per lane.
pub fn sum_i128_checked(initial: i128, values: &[i128]) -> Option<i128> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq")
        {
            // SAFETY: `avx512f` and `avx512dq` are available.
            return unsafe { avx512::sum_i128_checked_avx512(initial, values) };
        }
    }

    sum_i128_checked_scalar(initial, values)
}

fn add_i128_scalar(a: &[i128], b: &[i128]) -> Buffer<i128> {
    let mut out = BufferMut::<i128>::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        out.push(x.wrapping_add(*y));
    }
    out.freeze()
}

fn sum_i128_scalar(initial: i128, values: &[i128]) -> i128 {
    values.iter().fold(initial, |acc, &v| acc.wrapping_add(v))
}

fn sum_i128_checked_scalar(initial: i128, values: &[i128]) -> Option<i128> {
    let mut acc = initial;
    for &v in values {
        acc = acc.checked_add(v)?;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    use super::*;

    fn reference_add(a: &[i128], b: &[i128]) -> Vec<i128> {
        a.iter().zip(b).map(|(x, y)| x.wrapping_add(*y)).collect()
    }

    #[rstest]
    #[case::empty(0)]
    #[case::sub_block(3)]
    #[case::one_block(4)]
    #[case::block_plus_remainder(7)]
    #[case::many_blocks(1000)]
    #[case::odd(1001)]
    fn add_matches_scalar(#[case] n: usize) {
        let mut rng = StdRng::seed_from_u64(0xA11CE ^ n as u64);
        let a: Vec<i128> = (0..n).map(|_| rng.random::<i128>()).collect();
        let b: Vec<i128> = (0..n).map(|_| rng.random::<i128>()).collect();

        let got = add_i128(&a, &b);
        assert_eq!(got.as_slice(), reference_add(&a, &b).as_slice());
        // The dispatched result must equal the pure scalar implementation.
        assert_eq!(got.as_slice(), add_i128_scalar(&a, &b).as_slice());
    }

    #[test]
    fn add_carry_edges() {
        // Exercises carry propagation out of the low 64 bits and wraparound of i128.
        let a = vec![
            u64::MAX as i128, // low half all ones -> carries on +1
            i128::MAX,
            -1i128,
            (1i128 << 64) - 1,
            i128::MIN,
        ];
        let b = vec![1i128, 1i128, 1i128, 1i128, -1i128];
        let got = add_i128(&a, &b);
        assert_eq!(got.as_slice(), reference_add(&a, &b).as_slice());
    }

    #[rstest]
    #[case(0)]
    #[case(4)]
    #[case(9)]
    #[case(1234)]
    fn sum_matches_scalar(#[case] n: usize) {
        let mut rng = StdRng::seed_from_u64(0x5C0FE ^ n as u64);
        // Keep magnitudes small so the wrapping and checked sums agree (no overflow).
        let v: Vec<i128> = (0..n)
            .map(|_| rng.random_range(-1_000_000i128..1_000_000))
            .collect();
        let expected: i128 = v.iter().copied().fold(0i128, i128::wrapping_add);

        assert_eq!(sum_i128(&v), expected);
        assert_eq!(sum_i128_checked(0, &v), Some(expected));
    }

    #[test]
    fn sum_checked_detects_overflow() {
        let big = i128::MAX / 2;
        // big + big + big overflows i128.
        let v = vec![big, big, big, big, big];
        assert_eq!(sum_i128_checked(0, &v), None);
        // Negative overflow as well.
        let small = i128::MIN / 2;
        let v = vec![small, small, small];
        assert_eq!(sum_i128_checked(0, &v), None);
    }

    #[test]
    fn sum_checked_initial_is_included() {
        let v = vec![10i128, 20, 30];
        assert_eq!(sum_i128_checked(5, &v), Some(65));
        assert_eq!(sum_i128_checked(i128::MAX - 1, &[2i128]), None);
    }
}
