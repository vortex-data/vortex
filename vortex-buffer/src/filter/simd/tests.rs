// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use rstest::rstest;

use super::*;
use crate::BitBuffer;
use crate::filter::scalar::compact_by_bitmap;

/// The scalar result of filtering `values` by `bits`.
fn expected<T: Copy>(values: &[T], bits: &BitBuffer) -> Vec<T> {
    let mut out = Vec::with_capacity(values.len());
    // SAFETY: `bits` covers `values`, and the output has room for every element.
    let written =
        unsafe { compact_by_bitmap(values, MaskBits::Bitmap(bits), out.spare_capacity_mut()) };
    // SAFETY: the walk initialized the first `written` elements.
    unsafe { out.set_len(written) };
    out
}

/// Filter `values` out of place with the SIMD kernel, if one applies.
fn compress_out_of_place<T: Copy>(values: &[T], bits: &BitBuffer) -> Option<Vec<T>> {
    let true_count = bits.true_count();
    let mut out = Vec::with_capacity(true_count + SLACK_BYTES / size_of::<T>());
    // SAFETY: `bits` covers `values`, and the output has a vector of slack.
    let written = unsafe {
        compress(
            values,
            MaskBits::Bitmap(bits),
            density(bits),
            out.spare_capacity_mut(),
        )
    }?;
    assert_eq!(written, true_count);
    // SAFETY: the kernel initialized the first `written` elements.
    unsafe { out.set_len(written) };
    Some(out)
}

fn density(bits: &BitBuffer) -> f64 {
    bits.true_count() as f64 / bits.len() as f64
}

fn make_bits(len: usize, offset: usize, pattern: impl Fn(usize) -> bool) -> BitBuffer {
    let backing =
        BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain((0..len).map(pattern)));
    BitBuffer::new_with_offset(backing.inner().clone(), len, offset)
}

type Pattern = fn(usize) -> bool;

fn patterns() -> [Pattern; 4] {
    [
        |i| i % 3 == 0,
        |i| i % 16 != 0,
        |i| (i / 64) % 2 == 0,
        |i| i < 3 || i % 61 == 60,
    ]
}

fn check<T: Copy + PartialEq + std::fmt::Debug>(values: &[T], bits: &BitBuffer) {
    let expected = expected(values, bits);

    if let Some(actual) = compress_out_of_place(values, bits) {
        assert_eq!(actual, expected);
    }

    let mut compacted = values.to_vec();
    // SAFETY: `bits` covers `compacted`.
    if let Some(new_len) =
        unsafe { compress_in_place(&mut compacted, MaskBits::Bitmap(bits), density(bits)) }
    {
        assert_eq!(&compacted[..new_len], expected.as_slice());
    }
}

#[rstest]
fn simd_matches_scalar(#[values(0, 5)] offset: usize, #[values(64, 151)] len: usize) {
    for pattern in patterns() {
        let bits = make_bits(len, offset, pattern);

        let u8_values: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let u16_values: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let u32_values: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let u64_values: Vec<u64> = (0..len).map(|i| i as u64).collect();

        check(&u8_values, &bits);
        check(&u16_values, &bits);
        check(&u32_values, &bits);
        check(&u64_values, &bits);
    }
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
fn check_engagement<T: Copy + PartialEq + std::fmt::Debug>(
    values: &[T],
    bits: &BitBuffer,
    engaged: bool,
) {
    let expected = expected(values, bits);

    let actual = compress_out_of_place(values, bits);
    assert_eq!(actual.is_some(), engaged);
    if let Some(actual) = actual {
        assert_eq!(actual, expected);
    }

    let mut compacted = values.to_vec();
    // SAFETY: `bits` covers `compacted`.
    let written =
        unsafe { compress_in_place(&mut compacted, MaskBits::Bitmap(bits), density(bits)) };
    assert_eq!(written.is_some(), engaged);
    if let Some(written) = written {
        assert_eq!(&compacted[..written], expected.as_slice());
    }
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
#[test]
fn engages_on_supported_cpus() {
    let bits = make_bits(256, 0, |i| i % 2 == 0);

    #[cfg(target_arch = "x86_64")]
    let expected = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt");
    #[cfg(target_arch = "aarch64")]
    let expected = true;

    check_engagement(
        &(0..256).map(|i| i as u8).collect::<Vec<_>>(),
        &bits,
        expected,
    );
    check_engagement(&(0u16..256).collect::<Vec<_>>(), &bits, expected);
    check_engagement(&(0u32..256).collect::<Vec<_>>(), &bits, expected);
    check_engagement(&(0u64..256).collect::<Vec<_>>(), &bits, expected);
}

#[test]
fn declines_sparse_and_short_masks() {
    let values: Vec<u32> = (0..1024).collect();

    let sparse = make_bits(1024, 0, |i| i % 128 == 0);
    assert!(compress_out_of_place(&values, &sparse).is_none());

    let short = make_bits(32, 0, |i| i % 2 == 0);
    assert!(compress_out_of_place(&values[..32], &short).is_none());
}

// AVX-512 machines need direct coverage of the otherwise-unselected AVX2-tier kernels.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[test]
fn avx2_kernels_match_scalar() {
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("popcnt")) {
        return;
    }

    fn check_kernel<T: Copy + PartialEq + std::fmt::Debug + Default>(
        kernel_out_of_place: Kernel,
        kernel_in_place: Kernel,
        values: &[T],
        bits: &BitBuffer,
    ) {
        let expected = expected(values, bits);

        let mut out = vec![T::default(); bits.true_count() + SLACK_BYTES / size_of::<T>()];
        // SAFETY: AVX2 and POPCNT were detected above and the output has a vector of slack.
        let written = unsafe {
            kernel_out_of_place(
                values.as_ptr().cast(),
                out.as_mut_ptr().cast(),
                MaskBits::Bitmap(bits),
            )
        };
        assert_eq!(written, bits.true_count());
        assert_eq!(&out[..written], expected.as_slice());

        let mut compacted = values.to_vec();
        let ptr = compacted.as_mut_ptr().cast::<u8>();
        // SAFETY: AVX2 and POPCNT were detected above; in-place compaction stays within the slice.
        let written = unsafe { kernel_in_place(ptr.cast_const(), ptr, MaskBits::Bitmap(bits)) };
        assert_eq!(written, bits.true_count());
        assert_eq!(&compacted[..written], expected.as_slice());
    }

    for pattern in patterns() {
        for len in [64, 151] {
            for offset in [0, 5] {
                let bits = make_bits(len, offset, pattern);
                let u8_values: Vec<u8> = (0..len).map(|i| i as u8).collect();
                let u16_values: Vec<u16> = (0..len).map(|i| i as u16).collect();
                let u32_values: Vec<u32> = (0..len as u32).collect();
                let u64_values: Vec<u64> = (0..len as u64).collect();
                check_kernel(
                    generic::compress_generic_8::<false>,
                    generic::compress_generic_8::<true>,
                    &u8_values,
                    &bits,
                );
                check_kernel(
                    generic::compress_generic_16::<false>,
                    generic::compress_generic_16::<true>,
                    &u16_values,
                    &bits,
                );
                check_kernel(
                    x86::compress_avx2_epi32::<false>,
                    x86::compress_avx2_epi32::<true>,
                    &u32_values,
                    &bits,
                );
                check_kernel(
                    x86::compress_avx2_epi64::<false>,
                    x86::compress_avx2_epi64::<true>,
                    &u64_values,
                    &bits,
                );
            }
        }
    }
}
