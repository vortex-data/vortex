// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 byte-table take for `u8` codes and at most 32 one-byte values.

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use std::arch::x86 as arch;
    } else {
        use std::arch::x86_64 as arch;
    }
}

use arch::__m256i;
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
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use super::super::FixedWidthTakeValue;
use super::super::HAS_AVX2;
use super::super::take_values_fallback;
use super::avx512::HAS_AVX512_VBMI;
use super::avx512::take_avx512;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;

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
    if I::PTYPE != PType::U8
        || values.is_empty()
        || values.len() > 32
        || size_of::<T>() != 1
        || indices.len() < 64
        || !*HAS_AVX2
    {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };
    // The existing single-table AVX2 path is already competitive for 1..=16 values.
    // Use VBMI where it also avoids the second table shuffle and blend.
    if values.len() > 16 && *HAS_AVX512_VBMI {
        // SAFETY: All required features were detected. T is one byte with initialized bytes,
        // and the table contains between 1 and 32 values.
        return unsafe { take_avx512(values, indices, allocator) };
    }
    // SAFETY: AVX2 was detected above. Values are one byte with no uninitialized bytes,
    // and the table contains between 1 and 32 values. Specializing the table count keeps the
    // existing single-table loop free of a per-vector branch or second shuffle.
    if values.len() <= 16 {
        unsafe { take_avx2::<T, false>(values, indices, allocator) }
    } else {
        unsafe { take_avx2::<T, true>(values, indices, allocator) }
    }
}

/// Takes one-byte values using one or two AVX2 lookup tables.
///
/// # Safety
///
/// Requires AVX2, one-byte T, and between 1 and 32 values. When `TWO_TABLES` is false,
/// the dictionary must contain at most 16 values.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn take_avx2<T: FixedWidthTakeValue, const TWO_TABLES: bool>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut table = [values[0]; 32];
    table[..values.len()].copy_from_slice(values);
    // SAFETY: Each half contains 16 initialized one-byte values.
    let low = _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(table.as_ptr().cast()) });
    let high =
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(table.as_ptr().add(16).cast()) });
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
        let taken = _mm256_shuffle_epi8(low, codes);
        let taken = if TWO_TABLES {
            // VPSHUFB uses the low four bits within each 128-bit lane. Bit four selects
            // which broadcast table supplies the result; the bounds check rejects other codes.
            _mm256_blendv_epi8(
                taken,
                _mm256_shuffle_epi8(high, codes),
                _mm256_cmpgt_epi8(codes, _mm256_set1_epi8(15)),
            )
        } else {
            taken
        };
        // SAFETY: the output has capacity for every index and T is one byte wide.
        unsafe { _mm256_storeu_si256(output_ptr.add(offset).cast::<__m256i>(), taken) };
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
