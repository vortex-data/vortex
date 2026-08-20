// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(all(target_arch = "x86_64", not(miri)))]

use std::panic::RefUnwindSafe;
use std::panic::catch_unwind;

use vortex_buffer::Buffer;

use super::super::FixedWidthTakeValue;
use super::take_avx2;
use crate::dtype::UnsignedPType;

fn take_avx2_if_supported<V: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[V],
    indices: &[I],
) -> Option<Buffer<V>> {
    if !is_x86_feature_detected!("avx2") {
        return None;
    }

    // SAFETY: AVX2 support was detected above, and `FixedWidthTakeValue` guarantees that every
    // byte in the values is initialized.
    Some(unsafe { take_avx2(values, indices) })
}

fn assert_avx2_take_panics<V, I>(values: &[V], indices: &[I], expected: &str)
where
    V: FixedWidthTakeValue + RefUnwindSafe,
    I: UnsignedPType + RefUnwindSafe,
{
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // SAFETY: AVX2 support was detected above, and `FixedWidthTakeValue` guarantees that every
    // byte in the values is initialized.
    let result = catch_unwind(|| unsafe { take_avx2(values, indices) });
    let Err(payload) = result else {
        panic!("take should panic for an invalid index");
    };
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some(expected));
}

macro_rules! test_cases {
    (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
        paste::paste! {
            $(
                #[test]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_simple_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                    let indices: Vec<$IDX> = (0..127).collect();

                    let Some(result) = take_avx2_if_supported(&values, &indices) else {
                        return;
                    };
                    assert_eq!(&values, result.as_slice());
                }

                #[test]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_empty_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = vec![];
                    let indices: Vec<$IDX> = (0..127).collect();

                    assert_avx2_take_panics(
                        &values,
                        &indices,
                        "cannot take a non-empty set of indices from an empty buffer",
                    );
                }

                #[test]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_invalid_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                    let indices: Vec<$IDX> = (127..=254).collect();

                    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
                }
            )+
        }
    };
}

test_cases!(
    index_type => u8,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u16,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u32,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u64,
    value_types => u32, i32, u64, i64, f32, f64
);

#[test]
fn last_valid_u8_index() {
    let values: Vec<i64> = (0..=255).collect();
    let indices: Vec<u8> = vec![255; 20];

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(&[255; 20], result.as_slice());
}

#[test]
fn last_valid_u16_index() {
    let values: Vec<i64> = (0..=65535).collect();
    let indices: Vec<u16> = vec![65535; 20];

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(&[65535; 20], result.as_slice());
}

#[test]
fn empty_values_and_indices() {
    let Some(result) = take_avx2_if_supported::<u32, u32>(&[], &[]) else {
        return;
    };

    assert!(result.is_empty());
}

#[test]
fn i32_gather_addressable_length_boundary() {
    assert!(super::i32_gather_can_address(i32::MAX as usize + 1));
    assert!(!super::i32_gather_can_address(i32::MAX as usize + 2));
}

#[test]
fn invalid_index_only_in_simd_block() {
    let values = vec![10u32, 20, 30];
    let indices = vec![3u32, 0, 1, 2, 0, 1, 2, 0, 1];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn simd_array_u8x4() {
    let values: Vec<[u8; 4]> = (1u32..=200).map(u32::to_le_bytes).collect();
    let indices: Vec<u32> = (0..200).collect();

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn scalar_fallback_u16() {
    let values: Vec<u16> = (1..=300).collect();
    let indices: Vec<u32> = (0..300).collect();

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn scalar_fallback_array_u8x16() {
    let values: Vec<[u8; 16]> = (0u128..200).map(u128::to_le_bytes).collect();
    let indices: Vec<u32> = (0..200).collect();

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn u32_max_index_in_u32_lane() {
    let values = vec![0u32; 8];
    // The first eight indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u32::MAX, 2, 3, 4, 5, 6, 7, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn u64_max_index_in_u32_lane() {
    let values = vec![0u32; 8];
    // The first four indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u64::MAX, 2, 3, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn u64_max_index_in_u64_lane() {
    let values = vec![0u64; 8];
    // The first four indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u64::MAX, 2, 3, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}
