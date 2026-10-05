// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 and AVX-512 byte-table take for `u8` codes and small one-byte value tables.

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
use arch::_mm256_blendv_epi8;
use arch::_mm256_broadcastsi128_si256;
use arch::_mm256_cmpgt_epi8;
use arch::_mm256_loadu_si256;
use arch::_mm256_or_si256;
use arch::_mm256_set1_epi8;
use arch::_mm256_setzero_si256;
use arch::_mm256_shuffle_epi8;
use arch::_mm256_storeu_si256;
use arch::_mm256_subs_epu8;
use arch::_mm256_testz_si256;
use arch::_mm512_cmplt_epu8_mask;
use arch::_mm512_loadu_si512;
use arch::_mm512_permutexvar_epi8;
use arch::_mm512_set1_epi8;
use arch::_mm512_storeu_si512;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use super::super::FixedWidthTakeValue;
use super::super::HAS_AVX2;
use super::super::take_values_fallback;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;

static HAS_AVX512_VBMI: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi")
});

// SAFETY: u8 has no padding or uninitialized bytes.
unsafe impl FixedWidthTakeValue for u8 {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

// SAFETY: i8 has no padding or uninitialized bytes.
unsafe impl FixedWidthTakeValue for i8 {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

// SAFETY: Byte arrays have no padding and every byte is initialized.
unsafe impl<const N: usize> FixedWidthTakeValue for [u8; N] {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

fn take<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    if I::PTYPE != PType::U8 || values.is_empty() || size_of::<T>() != 1 || indices.len() < 64 {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };
    if values.len() <= 64 && *HAS_AVX512_VBMI {
        // SAFETY: AVX-512F, AVX-512BW, and AVX-512VBMI were detected above. Values are one byte
        // with no uninitialized bytes, and the table contains at most 64 values.
        unsafe { take_avx512_vbmi(values, indices, allocator) }
    } else if values.len() <= 32 && *HAS_AVX2 {
        // SAFETY: AVX2 was detected above. Values are one byte with no uninitialized bytes,
        // and the table contains at most 32 values.
        unsafe { take_avx2(values, indices, allocator) }
    } else {
        take_values_fallback(values, indices, allocator)
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

    let high_len = values.len().saturating_sub(16);
    let mut high_table = [values[0]; 16];
    high_table[..high_len].copy_from_slice(&values[16..]);
    // SAFETY: the table contains 16 initialized one-byte values.
    let high_table = unsafe { _mm_loadu_si128(high_table.as_ptr().cast()) };
    let high_table = _mm256_broadcastsi128_si256(high_table);
    let fifteen = _mm256_set1_epi8(15);
    let limit = _mm256_set1_epi8(
        i8::try_from(values.len() - 1).vortex_expect("table contains at most 32 values"),
    );
    let mut invalid = _mm256_setzero_si256();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 32 <= indices.len() {
        // SAFETY: the loop condition guarantees a complete 32-byte vector is in bounds.
        let codes = unsafe { _mm256_loadu_si256(indices.as_ptr().add(offset).cast()) };
        invalid = _mm256_or_si256(invalid, _mm256_subs_epu8(codes, limit));
        let low = _mm256_shuffle_epi8(low_table, codes);
        let decoded = if high_len == 0 {
            low
        } else {
            let high = _mm256_shuffle_epi8(high_table, codes);
            _mm256_blendv_epi8(low, high, _mm256_cmpgt_epi8(codes, fifteen))
        };
        // SAFETY: the output has capacity for every index and T is one byte wide.
        unsafe { _mm256_storeu_si256(output_ptr.add(offset).cast::<__m256i>(), decoded) };
        offset += 32;
    }
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
    let mut table = [values[0]; 64];
    table[..values.len()].copy_from_slice(values);
    // SAFETY: the table contains 64 initialized one-byte values.
    let table = unsafe { _mm512_loadu_si512(table.as_ptr().cast::<__m512i>()) };
    let bound = _mm512_set1_epi8(
        i8::try_from(values.len()).vortex_expect("table contains at most 64 values"),
    );
    let mut all_indices_valid = u64::MAX;
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 64 <= indices.len() {
        // SAFETY: the loop condition guarantees a complete 64-byte vector is in bounds.
        let codes = unsafe { _mm512_loadu_si512(indices.as_ptr().add(offset).cast::<__m512i>()) };
        all_indices_valid &= _mm512_cmplt_epu8_mask(codes, bound);
        // SAFETY: the output has capacity for every index and T is one byte wide.
        unsafe {
            _mm512_storeu_si512(
                output_ptr.add(offset).cast::<__m512i>(),
                _mm512_permutexvar_epi8(codes, table),
            )
        };
        offset += 64;
    }
    assert_eq!(all_indices_valid, u64::MAX, "take index out of bounds");

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

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferAllocatorRef;
    use vortex_error::VortexExpect;

    use super::HAS_AVX2;
    use super::take_avx2;

    #[test]
    fn avx2_two_table_lookup() {
        if !*HAS_AVX2 {
            return;
        }

        for cardinality in [17usize, 31, 32] {
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
}
