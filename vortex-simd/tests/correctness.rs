//! Cross-tier correctness: every available kernel must agree with the scalar
//! oracle on the same inputs.

use rstest::rstest;
use vortex_simd::cpu::{Tier, tier};
use vortex_simd::kernels::scalar;
use vortex_simd::ops::IntOps;

fn make_inputs(count: i32) -> (Vec<i32>, Vec<i32>) {
    // Mix of equal, off-by-one, and large-delta pairs so eq mask is non-trivial.
    let lhs: Vec<i32> = (0..count).map(|idx| idx.wrapping_mul(7)).collect();
    let rhs: Vec<i32> = (0..count)
        .map(|idx| {
            if idx % 3 == 0 {
                idx.wrapping_mul(7)
            } else {
                idx
            }
        })
        .collect();
    (lhs, rhs)
}

fn bitmap_bytes(count: i32) -> usize {
    (count.unsigned_abs() as usize).div_ceil(8)
}

fn vec_len(count: i32) -> usize {
    count.unsigned_abs() as usize
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(7)]
#[case(8)]
#[case(15)]
#[case(16)]
#[case(17)]
#[case(31)]
#[case(32)]
#[case(33)]
#[case(1024)]
#[case(1031)] // not a multiple of any vector width
fn ops_add_matches_scalar(#[case] count: i32) {
    let (lhs, rhs) = make_inputs(count);
    let mut want = vec![0_i32; vec_len(count)];
    let mut got = vec![0_i32; vec_len(count)];
    scalar::add_i32(&lhs, &rhs, &mut want);
    (i32::ops().add)(&lhs, &rhs, &mut got);
    assert_eq!(want, got, "add mismatch at n={count}, tier={:?}", tier());
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(7)]
#[case(8)]
#[case(15)]
#[case(16)]
#[case(17)]
#[case(31)]
#[case(32)]
#[case(33)]
#[case(1024)]
#[case(1031)]
fn ops_eq_matches_scalar(#[case] count: i32) {
    let (lhs, rhs) = make_inputs(count);
    let bitmap_len = bitmap_bytes(count);
    let mut want = vec![0_u8; bitmap_len];
    let mut got = vec![0_u8; bitmap_len];
    scalar::eq_i32(&lhs, &rhs, &mut want);
    (i32::ops().eq)(&lhs, &rhs, &mut got);
    assert_eq!(want, got, "eq mismatch at n={count}, tier={:?}", tier());
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::*;
    use vortex_simd::arch::x86_64 as x;

    #[rstest]
    #[case(0)]
    #[case(7)]
    #[case(8)]
    #[case(15)]
    #[case(16)]
    #[case(17)]
    #[case(1024)]
    #[case(1031)]
    fn add_sse2_matches_scalar(#[case] count: i32) {
        if tier() < Tier::SSE42 {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let mut want = vec![0_i32; vec_len(count)];
        let mut got = vec![0_i32; vec_len(count)];
        scalar::add_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed SSE2.
        unsafe { x::add_i32_sse2(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(17)]
    #[case(1024)]
    #[case(1031)]
    fn add_avx2_matches_scalar(#[case] count: i32) {
        if tier() < Tier::AVX2 {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let mut want = vec![0_i32; vec_len(count)];
        let mut got = vec![0_i32; vec_len(count)];
        scalar::add_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed AVX2.
        unsafe { x::add_i32_avx2(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }

    #[rstest]
    #[case(16)]
    #[case(32)]
    #[case(33)]
    #[case(1024)]
    #[case(1031)]
    fn add_avx512_matches_scalar(#[case] count: i32) {
        if tier() < Tier::AVX512 {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let mut want = vec![0_i32; vec_len(count)];
        let mut got = vec![0_i32; vec_len(count)];
        scalar::add_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed AVX-512.
        unsafe { x::add_i32_avx512(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(17)]
    #[case(1024)]
    #[case(1031)]
    fn eq_avx2_matches_scalar(#[case] count: i32) {
        if tier() < Tier::AVX2 {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let bitmap_len = bitmap_bytes(count);
        let mut want = vec![0_u8; bitmap_len];
        let mut got = vec![0_u8; bitmap_len];
        scalar::eq_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed AVX2.
        unsafe { x::eq_i32_avx2(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }

    #[rstest]
    #[case(16)]
    #[case(32)]
    #[case(33)]
    #[case(1024)]
    #[case(1031)]
    fn eq_avx512_matches_scalar(#[case] count: i32) {
        if tier() < Tier::AVX512 {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let bitmap_len = bitmap_bytes(count);
        let mut want = vec![0_u8; bitmap_len];
        let mut got = vec![0_u8; bitmap_len];
        scalar::eq_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed AVX-512.
        unsafe { x::eq_i32_avx512(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::*;
    use vortex_simd::arch::aarch64 as neon;

    #[rstest]
    #[case(0)]
    #[case(7)]
    #[case(8)]
    #[case(1024)]
    #[case(1031)]
    fn add_neon_matches_scalar(#[case] count: i32) {
        if tier() < Tier::NEON {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let mut want = vec![0_i32; vec_len(count)];
        let mut got = vec![0_i32; vec_len(count)];
        scalar::add_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed NEON.
        unsafe { neon::add_i32_neon(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }

    #[rstest]
    #[case(8)]
    #[case(16)]
    #[case(17)]
    #[case(1024)]
    #[case(1031)]
    fn eq_neon_matches_scalar(#[case] count: i32) {
        if tier() < Tier::NEON {
            return;
        }
        let (lhs, rhs) = make_inputs(count);
        let bitmap_len = bitmap_bytes(count);
        let mut want = vec![0_u8; bitmap_len];
        let mut got = vec![0_u8; bitmap_len];
        scalar::eq_i32(&lhs, &rhs, &mut want);
        // SAFETY: tier check confirmed NEON.
        unsafe { neon::eq_i32_neon(&lhs, &rhs, &mut got) };
        assert_eq!(want, got);
    }
}
