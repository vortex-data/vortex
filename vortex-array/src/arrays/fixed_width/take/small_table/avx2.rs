// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 and AVX-512 byte-table take for `u8` codes and small one- or two-byte values.

use std::mem::size_of_val;
use std::sync::LazyLock;

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use std::arch::x86 as arch;
    } else {
        use std::arch::x86_64 as arch;
    }
}

use arch::__m256i;
use arch::__m512i;
use arch::_mm_loadu_si128;
use arch::_mm256_add_epi16;
use arch::_mm256_blendv_epi8;
use arch::_mm256_broadcastsi128_si256;
use arch::_mm256_cmpgt_epi8;
use arch::_mm256_cmpgt_epi16;
use arch::_mm256_cvtepu8_epi16;
use arch::_mm256_loadu_si256;
use arch::_mm256_max_epu8;
use arch::_mm256_max_epu16;
use arch::_mm256_mullo_epi16;
use arch::_mm256_set1_epi8;
use arch::_mm256_set1_epi16;
use arch::_mm256_setzero_si256;
use arch::_mm256_shuffle_epi8;
use arch::_mm256_storeu_si256;
use arch::_mm256_subs_epu8;
use arch::_mm256_subs_epu16;
use arch::_mm256_testz_si256;
use arch::_mm512_add_epi16;
use arch::_mm512_cmplt_epu8_mask;
use arch::_mm512_cvtepu8_epi16;
use arch::_mm512_loadu_si512;
use arch::_mm512_mask_blend_epi8;
use arch::_mm512_max_epu8;
use arch::_mm512_movepi8_mask;
use arch::_mm512_mullo_epi16;
use arch::_mm512_permutex2var_epi8;
use arch::_mm512_permutexvar_epi8;
use arch::_mm512_set1_epi8;
use arch::_mm512_set1_epi16;
use arch::_mm512_setzero_si512;
use arch::_mm512_storeu_si512;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use super::super::FixedWidthTakeValue;
use super::super::HAS_AVX2;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;

static HAS_AVX512_VBMI: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi")
});

pub(crate) fn try_take<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> Option<Buffer<T>> {
    let value_width = size_of::<T>();
    if I::PTYPE != PType::U8
        || values.is_empty()
        || !matches!(value_width, 1 | 2)
        || indices.len() < 64
    {
        return None;
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };
    if value_width == 1 && values.len() <= 32 && *HAS_AVX2 {
        // SAFETY: AVX2 was detected above. Values are one byte with no uninitialized bytes,
        // and the table contains at most 32 values.
        Some(unsafe { take_avx2(values, indices, allocator) })
    } else if value_width == 2 && values.len() <= 16 && *HAS_AVX2 {
        // SAFETY: AVX2 was detected above. Values have an initialized two-byte representation,
        // and the table contains at most 16 values.
        Some(unsafe { take_avx2_u16(values, indices, allocator) })
    } else if values.len().saturating_mul(value_width) <= 256 && *HAS_AVX512_VBMI {
        // SAFETY: AVX-512F, AVX-512BW, and AVX-512VBMI were detected above. Values have an
        // initialized one- or two-byte representation and the packed table is at most 256 bytes.
        Some(unsafe { take_avx512_vbmi(values, indices, allocator) })
    } else if value_width == 2 && values.len() <= 32 && *HAS_AVX2 {
        // SAFETY: AVX2 was detected above. Values have an initialized two-byte representation,
        // and the table contains at most 32 values.
        Some(unsafe { take_avx2_u16(values, indices, allocator) })
    } else if value_width == 1 && values.len() <= 64 && *HAS_AVX2 {
        // SAFETY: AVX2 was detected above. Values are one byte with no uninitialized bytes,
        // and the table contains at most 64 values.
        Some(unsafe { take_avx2(values, indices, allocator) })
    } else {
        None
    }
}

#[target_feature(enable = "avx2")]
unsafe fn take_avx2<T: FixedWidthTakeValue>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let low_len = values.len().min(16);
    let mut low_table = [values[0]; 16];
    low_table[..low_len].copy_from_slice(&values[..low_len]);
    // SAFETY: the table contains 16 initialized one-byte values.
    let low_table = unsafe { _mm_loadu_si128(low_table.as_ptr().cast()) };
    let low_table = _mm256_broadcastsi128_si256(low_table);

    let high_len = values.len().saturating_sub(16).min(16);
    let mut high_table = [values[0]; 16];
    if high_len != 0 {
        high_table[..high_len].copy_from_slice(&values[16..16 + high_len]);
    }
    // SAFETY: the table contains 16 initialized one-byte values.
    let high_table = unsafe { _mm_loadu_si128(high_table.as_ptr().cast()) };
    let high_table = _mm256_broadcastsi128_si256(high_table);
    let third_len = values.len().saturating_sub(32).min(16);
    let mut third_table = [values[0]; 16];
    if third_len != 0 {
        third_table[..third_len].copy_from_slice(&values[32..32 + third_len]);
    }
    // SAFETY: the table contains 16 initialized one-byte values.
    let third_table = unsafe { _mm_loadu_si128(third_table.as_ptr().cast()) };
    let third_table = _mm256_broadcastsi128_si256(third_table);
    let fourth_len = values.len().saturating_sub(48).min(16);
    let mut fourth_table = [values[0]; 16];
    if fourth_len != 0 {
        fourth_table[..fourth_len].copy_from_slice(&values[48..48 + fourth_len]);
    }
    // SAFETY: the table contains 16 initialized one-byte values.
    let fourth_table = unsafe { _mm_loadu_si128(fourth_table.as_ptr().cast()) };
    let fourth_table = _mm256_broadcastsi128_si256(fourth_table);
    let fifteen = _mm256_set1_epi8(15);
    let thirty_one = _mm256_set1_epi8(31);
    let forty_seven = _mm256_set1_epi8(47);
    let limit = _mm256_set1_epi8(
        i8::try_from(values.len() - 1).vortex_expect("table contains at most 64 values"),
    );
    let mut max_codes = _mm256_setzero_si256();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 32 <= indices.len() {
        // SAFETY: the loop condition guarantees a complete 32-byte vector is in bounds.
        let codes = unsafe { _mm256_loadu_si256(indices.as_ptr().add(offset).cast()) };
        max_codes = _mm256_max_epu8(max_codes, codes);
        let low = _mm256_shuffle_epi8(low_table, codes);
        let decoded = if high_len == 0 {
            low
        } else if third_len == 0 {
            let high = _mm256_shuffle_epi8(high_table, codes);
            _mm256_blendv_epi8(low, high, _mm256_cmpgt_epi8(codes, fifteen))
        } else {
            let second = _mm256_shuffle_epi8(high_table, codes);
            let low_half = _mm256_blendv_epi8(low, second, _mm256_cmpgt_epi8(codes, fifteen));
            let third = _mm256_shuffle_epi8(third_table, codes);
            let fourth = _mm256_shuffle_epi8(fourth_table, codes);
            let high_half =
                _mm256_blendv_epi8(third, fourth, _mm256_cmpgt_epi8(codes, forty_seven));
            _mm256_blendv_epi8(low_half, high_half, _mm256_cmpgt_epi8(codes, thirty_one))
        };
        // SAFETY: the output has capacity for every index and T is one byte wide.
        unsafe { _mm256_storeu_si256(output_ptr.add(offset).cast::<__m256i>(), decoded) };
        offset += 32;
    }
    let invalid = _mm256_subs_epu8(max_codes, limit);
    assert_eq!(
        _mm256_testz_si256(invalid, invalid),
        1,
        "take index out of bounds"
    );

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        spare[offset].write(values[code]);
    }
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}

#[target_feature(enable = "avx2")]
unsafe fn take_avx2_u16<T: FixedWidthTakeValue>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let value_bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) };
    let mut table_bytes = [0u8; 64];
    table_bytes[..value_bytes.len()].copy_from_slice(value_bytes);
    // SAFETY: each load reads a complete initialized 16-byte region of `table_bytes`.
    let table0 =
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(table_bytes.as_ptr().cast()) });
    let table1 = _mm256_broadcastsi128_si256(unsafe {
        _mm_loadu_si128(table_bytes.as_ptr().add(16).cast())
    });
    let table2 = _mm256_broadcastsi128_si256(unsafe {
        _mm_loadu_si128(table_bytes.as_ptr().add(32).cast())
    });
    let table3 = _mm256_broadcastsi128_si256(unsafe {
        _mm_loadu_si128(table_bytes.as_ptr().add(48).cast())
    });
    let seven = _mm256_set1_epi16(7);
    let fifteen = _mm256_set1_epi16(15);
    let twenty_three = _mm256_set1_epi16(23);
    let multiplier = _mm256_set1_epi16(0x0202);
    let lane_offsets = _mm256_set1_epi16(0x0100);
    let limit = _mm256_set1_epi16(
        i16::try_from(values.len() - 1).vortex_expect("table contains at most 32 values"),
    );
    let mut max_codes = _mm256_setzero_si256();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 16 <= indices.len() {
        // SAFETY: the loop condition guarantees a complete 16-byte input vector is in bounds.
        let codes = unsafe { _mm_loadu_si128(indices.as_ptr().add(offset).cast()) };
        let codes = _mm256_cvtepu8_epi16(codes);
        max_codes = _mm256_max_epu16(max_codes, codes);
        let byte_offsets = _mm256_add_epi16(_mm256_mullo_epi16(codes, multiplier), lane_offsets);
        let first = _mm256_shuffle_epi8(table0, byte_offsets);
        let decoded = if values.len() <= 8 {
            first
        } else if values.len() <= 16 {
            let second = _mm256_shuffle_epi8(table1, byte_offsets);
            _mm256_blendv_epi8(first, second, _mm256_cmpgt_epi16(codes, seven))
        } else {
            let second = _mm256_shuffle_epi8(table1, byte_offsets);
            let low_half = _mm256_blendv_epi8(first, second, _mm256_cmpgt_epi16(codes, seven));
            let third = _mm256_shuffle_epi8(table2, byte_offsets);
            let fourth = _mm256_shuffle_epi8(table3, byte_offsets);
            let high_half =
                _mm256_blendv_epi8(third, fourth, _mm256_cmpgt_epi16(codes, twenty_three));
            _mm256_blendv_epi8(low_half, high_half, _mm256_cmpgt_epi16(codes, fifteen))
        };
        // SAFETY: the output has capacity for every index and T is two bytes wide.
        unsafe {
            _mm256_storeu_si256(
                output_ptr.add(offset * size_of::<T>()).cast::<__m256i>(),
                decoded,
            )
        };
        offset += 16;
    }
    let invalid = _mm256_subs_epu16(max_codes, limit);
    assert_eq!(
        _mm256_testz_si256(invalid, invalid),
        1,
        "take index out of bounds"
    );

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        spare[offset].write(values[code]);
    }
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn take_avx512_vbmi<T: FixedWidthTakeValue>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let value_width = size_of::<T>();
    let table_byte_len = size_of_val(values);
    let value_bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), table_byte_len) };
    let mut tables = [[0u8; 64]; 4];
    for (table_index, chunk) in value_bytes.chunks(64).enumerate() {
        tables[table_index][..chunk.len()].copy_from_slice(chunk);
    }
    // SAFETY: every table contains 64 initialized one-byte values.
    let table0 = unsafe { _mm512_loadu_si512(tables[0].as_ptr().cast::<__m512i>()) };
    let table1 = unsafe { _mm512_loadu_si512(tables[1].as_ptr().cast::<__m512i>()) };
    let table2 = unsafe { _mm512_loadu_si512(tables[2].as_ptr().cast::<__m512i>()) };
    let table3 = unsafe { _mm512_loadu_si512(tables[3].as_ptr().cast::<__m512i>()) };
    let bound = (values.len() < 256).then(|| {
        _mm512_set1_epi8(
            u8::try_from(values.len())
                .vortex_expect("table contains fewer than 256 values")
                .cast_signed(),
        )
    });
    let mut max_codes = _mm512_setzero_si512();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    let codes_per_vector = 64 / value_width;
    while offset + codes_per_vector <= indices.len() {
        // SAFETY: the loop condition guarantees a complete input vector is in bounds.
        let (expanded_codes, byte_offsets) =
            unsafe { load_byte_offsets(indices.as_ptr().add(offset), value_width) };
        max_codes = _mm512_max_epu8(max_codes, expanded_codes);
        let decoded = if table_byte_len <= 64 {
            _mm512_permutexvar_epi8(byte_offsets, table0)
        } else if table_byte_len <= 128 {
            _mm512_permutex2var_epi8(table0, byte_offsets, table1)
        } else {
            let low = _mm512_permutex2var_epi8(table0, byte_offsets, table1);
            let high = _mm512_permutex2var_epi8(table2, byte_offsets, table3);
            _mm512_mask_blend_epi8(_mm512_movepi8_mask(byte_offsets), low, high)
        };
        // SAFETY: the output has capacity for every index and the store writes `codes_per_vector`
        // complete values.
        unsafe {
            _mm512_storeu_si512(
                output_ptr.add(offset * value_width).cast::<__m512i>(),
                decoded,
            )
        };
        offset += codes_per_vector;
    }
    if let Some(bound) = bound {
        assert_eq!(
            _mm512_cmplt_epu8_mask(max_codes, bound),
            u64::MAX,
            "take index out of bounds"
        );
    }

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        spare[offset].write(values[code]);
    }
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn load_byte_offsets(codes: *const u8, value_width: usize) -> (__m512i, __m512i) {
    match value_width {
        1 => {
            let codes = unsafe { _mm512_loadu_si512(codes.cast::<__m512i>()) };
            (codes, codes)
        }
        2 => {
            let codes = unsafe { _mm256_loadu_si256(codes.cast::<__m256i>()) };
            let codes = _mm512_cvtepu8_epi16(codes);
            let offsets = _mm512_add_epi16(
                _mm512_mullo_epi16(codes, _mm512_set1_epi16(0x0202)),
                _mm512_set1_epi16(0x0100),
            );
            (codes, offsets)
        }
        _ => unreachable!("caller restricts value width to one or two bytes"),
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexExpect;

    use super::HAS_AVX2;
    use super::HAS_AVX512_VBMI;
    use super::take_avx2;
    use super::take_avx2_u16;
    use super::take_avx512_vbmi;
    use crate::arrays::fixed_width::take::FixedWidthTakeValue;

    #[test]
    fn avx2_two_table_lookup() {
        if !*HAS_AVX2 {
            return;
        }

        for cardinality in [2usize, 4, 8, 16, 17, 31, 32, 33, 63, 64] {
            let values = (0..cardinality)
                .map(|value| {
                    u8::try_from(value)
                        .vortex_expect("cardinality is at most 32")
                        .wrapping_mul(37)
                        .wrapping_add(11)
                })
                .collect::<Vec<_>>();
            let indices = (0..97)
                .map(|index| {
                    u8::try_from((index * 37) % cardinality)
                        .vortex_expect("cardinality is at most 32")
                })
                .collect::<Vec<_>>();
            let expected = indices
                .iter()
                .map(|&index| values[usize::from(index)])
                .collect::<Vec<_>>();
            // SAFETY: AVX2 support was detected above.
            let taken = unsafe {
                take_avx2(
                    &values,
                    &indices,
                    &BufferAllocatorRef::statically_allocated(),
                )
            };
            assert_eq!(taken.as_slice(), expected);
        }
    }

    #[test]
    fn avx512_four_table_lookup() {
        if !*HAS_AVX512_VBMI {
            return;
        }

        for cardinality in [65usize, 127, 128, 129, 255, 256] {
            let values = (0..cardinality)
                .map(|value| {
                    u8::try_from(value)
                        .vortex_expect("cardinality is at most 256")
                        .wrapping_mul(37)
                        .wrapping_add(11)
                })
                .collect::<Vec<_>>();
            let indices = (0..321)
                .map(|index| {
                    u8::try_from((index * 37) % cardinality)
                        .vortex_expect("cardinality is at most 256")
                })
                .collect::<Vec<_>>();
            let expected = indices
                .iter()
                .map(|&index| values[usize::from(index)])
                .collect::<Vec<_>>();
            // SAFETY: AVX-512F, AVX-512BW, and AVX-512VBMI support was detected above.
            let taken = unsafe {
                take_avx512_vbmi(
                    &values,
                    &indices,
                    &BufferAllocatorRef::statically_allocated(),
                )
            };
            assert_eq!(taken.as_slice(), expected);
        }
    }

    #[test]
    fn avx512_wide_values() {
        if !*HAS_AVX512_VBMI {
            return;
        }

        check_avx512_wide::<u16>(128);
    }

    #[test]
    fn avx2_u16_table_lookup() {
        if !*HAS_AVX2 {
            return;
        }

        for cardinality in [2usize, 8, 9, 16, 17, 31, 32] {
            let values = (0..cardinality)
                .map(|value| {
                    u16::try_from(value)
                        .vortex_expect("cardinality is at most 32")
                        .wrapping_mul(379)
                        .wrapping_add(11)
                })
                .collect::<Vec<_>>();
            let indices = (0..321)
                .map(|index| {
                    u8::try_from((index * 37) % cardinality)
                        .vortex_expect("cardinality is at most 32")
                })
                .collect::<Vec<_>>();
            let expected = indices
                .iter()
                .map(|&index| values[usize::from(index)])
                .collect::<Vec<_>>();
            // SAFETY: AVX2 support was detected above.
            let taken = unsafe {
                take_avx2_u16(
                    &values,
                    &indices,
                    &BufferAllocatorRef::statically_allocated(),
                )
            };
            assert_eq!(taken.as_slice(), expected);
        }
    }

    fn check_avx512_wide<T>(cardinality: usize)
    where
        T: FixedWidthTakeValue + From<u8> + PartialEq + Debug,
    {
        let values = (0..cardinality)
            .map(|value| {
                T::from(
                    u8::try_from(value)
                        .vortex_expect("wide-value cardinality is at most 128")
                        .wrapping_mul(37)
                        .wrapping_add(11),
                )
            })
            .collect::<Vec<_>>();
        let indices = (0..321)
            .map(|index| {
                u8::try_from((index * 37) % cardinality)
                    .vortex_expect("wide-value cardinality is at most 128")
            })
            .collect::<Vec<_>>();
        let expected = indices
            .iter()
            .map(|&index| values[usize::from(index)])
            .collect::<Vec<_>>();
        // SAFETY: AVX-512F, AVX-512BW, and AVX-512VBMI support was detected above.
        let taken = unsafe {
            take_avx512_vbmi(
                &values,
                &indices,
                &BufferAllocatorRef::statically_allocated(),
            )
        };
        assert_eq!(taken.as_slice(), expected);
    }
}
