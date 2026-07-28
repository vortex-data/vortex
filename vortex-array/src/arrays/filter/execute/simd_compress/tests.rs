// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use rstest::rstest;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use super::super::slice;
use super::*;

/// `None` when the mask normalized to `AllTrue`/`AllFalse`, which `filter_buffer` never sees
/// (the filter fast paths intercept those before buffer-level dispatch).
fn mask_values(mask: &Mask) -> Option<&MaskValues> {
    match mask {
        Mask::Values(values) => Some(values.as_ref()),
        _ => None,
    }
}

fn make_mask(len: usize, offset: usize, pattern: impl Fn(usize) -> bool) -> Mask {
    let backing =
        BitBuffer::from_iter(std::iter::repeat_n(false, offset).chain((0..len).map(pattern)));
    Mask::from_buffer(BitBuffer::new_with_offset(
        backing.inner().clone(),
        len,
        offset,
    ))
}

type Pattern = fn(usize) -> bool;

fn patterns() -> Vec<(&'static str, Pattern)> {
    vec![
        ("all_but_first", |i| i != 0),
        ("only_first", |i| i == 0),
        ("sparse", |i| i % 97 == 0),
        ("mid", |i| i % 3 == 0),
        ("dense", |i| i % 16 != 0),
        ("alternating", |i| i % 2 == 0),
        ("word_blocks", |i| (i / 64) % 2 == 0),
        ("edges", |i| i < 3 || i % 61 == 60),
    ]
}

fn check<T: Copy + PartialEq + std::fmt::Debug>(values: &[T], mask: &Mask) {
    let Some(mask) = mask_values(mask) else {
        return;
    };
    let expected = slice::filter_slice_by_bitmap(values, mask);

    if let Some(actual) = filter_slice_by_bitmap(values, mask) {
        assert_eq!(actual.as_slice(), expected.as_slice());
    }

    let mut compacted = values.to_vec();
    if let Some(new_len) = filter_slice_mut_by_bitmap(&mut compacted, mask) {
        assert_eq!(&compacted[..new_len], expected.as_slice());
    }
}

#[rstest]
fn simd_matches_scalar(
    #[values(0, 3, 5)] offset: usize,
    #[values(64, 100, 151, 1000, 1024)] len: usize,
) {
    for (name, pattern) in patterns() {
        let mask = make_mask(len, offset, pattern);
        let mask_debug = format!("pattern={name} len={len} offset={offset}");

        let u8_values: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let u16_values: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let u32_values: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let u64_values: Vec<u64> = (0..len).map(|i| i as u64).collect();

        println!("checking {mask_debug}");
        check(&u8_values, &mask);
        check(&u16_values, &mask);
        check(&u32_values, &mask);
        check(&u64_values, &mask);
    }
}

#[cfg(all(any(target_arch = "x86_64", target_arch = "aarch64"), not(miri)))]
#[test]
fn engages_on_supported_cpus() {
    let values: Vec<u32> = (0..256).collect();
    let mask = make_mask(256, 0, |i| i % 2 == 0);
    let mask = mask_values(&mask).expect("alternating mask is mixed");

    #[cfg(target_arch = "x86_64")]
    let expected = is_x86_feature_detected!("avx2");
    // NEON is part of the aarch64 baseline.
    #[cfg(target_arch = "aarch64")]
    let expected = true;

    assert_eq!(filter_slice_by_bitmap(&values, mask).is_some(), expected);
}

/// Every architecture declines a mask too sparse or too short to amortize a per-chunk compress,
/// which is what routes those to the scalar strategies.
#[test]
fn declines_sparse_and_short_masks() {
    let values: Vec<u32> = (0..1024).collect();

    let sparse = make_mask(1024, 0, |i| i % 128 == 0);
    let sparse = mask_values(&sparse).expect("sparse mask is mixed");
    assert!(filter_slice_by_bitmap(&values, sparse).is_none());

    let short = make_mask(32, 0, |i| i % 2 == 0);
    let short = mask_values(&short).expect("alternating mask is mixed");
    assert!(filter_slice_by_bitmap(&values[..32], short).is_none());
}

/// The dispatcher never selects the AVX2 tier on AVX-512 machines, so exercise those kernels
/// directly. Covers both the `vpermd` kernels for 4/8-byte elements and the `pshufb` kernels
/// for 1/2-byte elements.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[test]
fn avx2_kernels_match_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    fn check_kernel<T: Copy + PartialEq + std::fmt::Debug + Default>(
        kernel_out_of_place: Kernel,
        kernel_in_place: Kernel,
        values: &[T],
        mask: &MaskValues,
    ) {
        let expected = slice::filter_slice_by_bitmap(values, mask);

        // One vector of slack, mirroring the allocation in `filter_slice_by_bitmap`.
        let mut out = vec![T::default(); mask.true_count() + SLACK_BYTES / size_of::<T>()];
        // SAFETY: AVX2 was detected above and the output has a vector of slack.
        let written =
            unsafe { kernel_out_of_place(values.as_ptr().cast(), out.as_mut_ptr().cast(), mask) };
        assert_eq!(written, mask.true_count());
        assert_eq!(&out[..written], expected.as_slice());

        let mut compacted = values.to_vec();
        let ptr = compacted.as_mut_ptr().cast::<u8>();
        // SAFETY: AVX2 was detected above; in-place compaction stays within the slice.
        let written = unsafe { kernel_in_place(ptr.cast_const(), ptr, mask) };
        assert_eq!(written, mask.true_count());
        assert_eq!(&compacted[..written], expected.as_slice());
    }

    for (_, pattern) in patterns() {
        for len in [64, 100, 151, 1000] {
            for offset in [0, 5] {
                let mask = make_mask(len, offset, pattern);
                let Some(mask) = mask_values(&mask) else {
                    continue;
                };
                let u8_values: Vec<u8> = (0..len).map(|i| i as u8).collect();
                let u16_values: Vec<u16> = (0..len).map(|i| i as u16).collect();
                let u32_values: Vec<u32> = (0..len as u32).collect();
                let u64_values: Vec<u64> = (0..len as u64).collect();
                check_kernel(
                    x86::compress_pshufb_epi8::<false>,
                    x86::compress_pshufb_epi8::<true>,
                    &u8_values,
                    mask,
                );
                check_kernel(
                    x86::compress_pshufb_epi16::<false>,
                    x86::compress_pshufb_epi16::<true>,
                    &u16_values,
                    mask,
                );
                check_kernel(
                    x86::compress_avx2_epi32::<false>,
                    x86::compress_avx2_epi32::<true>,
                    &u32_values,
                    mask,
                );
                check_kernel(
                    x86::compress_avx2_epi64::<false>,
                    x86::compress_avx2_epi64::<true>,
                    &u64_values,
                    mask,
                );
            }
        }
    }
}
